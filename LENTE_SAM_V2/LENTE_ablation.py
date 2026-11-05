#!/usr/bin/env python
"""
LENTE_ablation.py — studio di ablazione.
========================================
Esegue due varianti ablate del modello LENTE e ne confronta le prestazioni
con la pipeline standard (stesso split, stessa augmentation, stesse metriche):

    1) no_dfds            -> rimuove il blocco DFDS (DSDFEarlyGuidance).
                             Il terrain (slope+DEM) NON viene piu' usato per
                             modulare l'ottico; gli adapter restano attivi.

    2) no_dfds_no_adapter -> rimuove DFDS *e* gli Adapter.
                             Resta solo il fine-tuning base di SAM2
                             (patch embed espanso + mask_decoder scongelato).

Il modello completo (DFDS + Adapter) NON viene ri-addestrato qui: e' la
baseline gia' prodotta da LENTE_train.py. Questo script genera solo le varianti
ablate, in modo da poterle confrontare con quella baseline.

Per ogni variante vengono prodotti:
    - un checkpoint dedicato  (es. <base>_ablation_no_dfds.pth)
    - una history JSON in histories/  (HISTORY_<dataset>_ablation_<variante>.json),
      compatibile con LENTE_plot_history.py

Uso:
    python LENTE_ablation.py --dataset bijie
    python LENTE_ablation.py -d l4s --epochs 30 --batch-size 4
    python LENTE_ablation.py -d bijie --variants no_dfds_no_adapter
"""

import os
import json
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import Sam2Model

from LENTE_utils import (
    load_config, build_dataset, build_train_val_test_subsets, TransformSubset,
    criterion, get_best_mask, calculate_metrics, scale_boxes,
    save_checkpoint, Adapter, DSDFEarlyGuidance,
    PROMPT_SOURCES, resolve_prompt, _build_sam2_image_config,
)


# ------------------------------------------------------------------
# Varianti ablate richieste.
#   use_dfds    : se False, niente DSDFEarlyGuidance (terrain non usato).
#   use_adapter : se False, niente Adapter nei blocchi di attenzione.
# ------------------------------------------------------------------
VARIANTS = {
    "no_dfds":            {"use_dfds": False, "use_adapter": True},
    "no_dfds_no_adapter": {"use_dfds": False, "use_adapter": False},
}

# Metriche complete tracciate per epoca (coerenti con LENTE_plot_history.py).
METRIC_KEYS = ["loss", "iou", "iou_bg", "miou", "f1", "precision", "recall"]


# ============================================================
# MODELLO ABLATO
# ============================================================

class AblationLENTE(nn.Module):
    """
    Versione parametrica di LENTE per l'ablazione.

    Replica la logica di costruzione di LENTE (SAM2-Tiny + patch embed espanso
    + mask_decoder scongelato), ma rende opzionali il blocco DFDS e gli Adapter
    tramite i flag use_dfds / use_adapter.

    NB: il patch embed viene comunque espanso a n_optical canali — e' il
    meccanismo che permette di dare in input le bande ottiche, indipendente
    dalle ablazioni. Quando use_dfds=False i canali di terrain semplicemente
    non vengono utilizzati.
    """

    def __init__(self, sam_ckpt_path, n_optical=12, adapter_dim=64,
                 use_dfds=True, use_adapter=True, prompt_source="none",
                 image_size=1024):
        super().__init__()
        if prompt_source not in PROMPT_SOURCES:
            raise ValueError(f"prompt_source sconosciuto: {prompt_source!r} (attesi: {PROMPT_SOURCES})")
        self.n_optical = n_optical
        self.use_dfds = use_dfds
        self.use_adapter = use_adapter
        # Deve combaciare con quello della baseline, altrimenti l'ablazione
        # sta misurando il prompt invece del componente rimosso.
        self.prompt_source = prompt_source
        # Idem per la risoluzione: deve essere cfg["target_size"], la stessa a
        # cui il loop interpola le immagini (vedi _build_sam2_image_config).
        self.image_size = int(image_size)

        if self.image_size == 1024:
            self.model = Sam2Model.from_pretrained(sam_ckpt_path)
        else:
            self.model = Sam2Model.from_pretrained(
                sam_ckpt_path, config=_build_sam2_image_config(sam_ckpt_path, self.image_size))
        v_encoder = self.model.vision_encoder

        # --- DFDS (opzionale) ---
        self.early_guidance = DSDFEarlyGuidance() if use_dfds else None

        # --- Patch embed: espansione a n_optical canali (sempre presente) ---
        self.patch_proj = None
        for name, module in v_encoder.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                if n_optical == 3:
                    self.patch_proj = module
                else:
                    new_conv = nn.Conv2d(
                        n_optical, module.out_channels,
                        kernel_size=module.kernel_size, stride=module.stride,
                        padding=module.padding, bias=(module.bias is not None),
                    )
                    with torch.no_grad():
                        new_conv.weight[:, :3] = module.weight.clone()
                        nn.init.normal_(new_conv.weight[:, 3:], mean=0.0, std=0.001)
                        if module.bias is not None:
                            new_conv.bias.copy_(module.bias)
                    parts = name.split(".")
                    parent = v_encoder.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else v_encoder
                    setattr(parent, parts[-1], new_conv)
                    self.patch_proj = new_conv
                break
        if self.patch_proj is None:
            raise RuntimeError("Patch embed Conv2d(in=3) non trovato.")

        # --- Adapter (opzionali) ---
        self.adapters = nn.ModuleList()
        if use_adapter:
            for name, module in v_encoder.named_modules():
                if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear):
                    dim = module.proj.out_features if hasattr(module, "proj") else module.qkv.in_features
                    ad = Adapter(dim, adapter_dim); self.adapters.append(ad)
                    module.register_forward_hook(self._make_hook(ad))
                elif hasattr(module, "q_proj") and isinstance(module.q_proj, nn.Linear):
                    dim = module.out_proj.out_features if hasattr(module, "out_proj") else module.q_proj.in_features
                    ad = Adapter(dim, adapter_dim); self.adapters.append(ad)
                    module.register_forward_hook(self._make_hook(ad))

        self._freeze()

    @staticmethod
    def _make_hook(adapter):
        def hook(module, args, output):
            if isinstance(output, tuple):
                return (adapter(output[0]),) + output[1:]
            return adapter(output)
        return hook

    def _freeze(self):
        # Base SAM2 congelato; si scongela sempre patch_proj + mask_decoder.
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.mask_decoder.parameters():
            p.requires_grad = True
        for p in self.patch_proj.parameters():
            p.requires_grad = True
        # Componenti opzionali addestrabili solo se presenti.
        for ad in self.adapters:
            for p in ad.parameters():
                p.requires_grad = True
        if self.early_guidance is not None:
            for p in self.early_guidance.parameters():
                p.requires_grad = True

    def forward(self, x, boxes=None):
        n = self.n_optical
        optical = x[:, :n]            # (B, n_optical, H, W)
        if self.use_dfds:
            terrain = x[:, n:]        # (B, 2, H, W)
            g = self.early_guidance(terrain)
            optical = optical * g
        # Senza DFDS il terrain non viene usato: optical passa diretto.

        prompt = resolve_prompt(self.prompt_source, boxes, optical.shape[0],
                                optical.shape[-1], optical.device, optical.dtype)
        outputs = self.model(pixel_values=optical, input_boxes=prompt, multimask_output=False)
        scores = getattr(outputs, "iou_scores",
                         getattr(outputs, "iou_predictions",
                                 getattr(outputs, "pred_ious", None)))
        return outputs.pred_masks, scores


def build_ablation_model(cfg, device, use_dfds, use_adapter):
    return AblationLENTE(
        sam_ckpt_path=cfg["sam_ckpt_path"],
        n_optical=cfg["n_optical"],
        adapter_dim=cfg["adapter_dim"],
        use_dfds=use_dfds,
        use_adapter=use_adapter,
        prompt_source=cfg.get("prompt_source", "none"),
        image_size=cfg["target_size"],
    ).to(device)


# ============================================================
# LOOP DI VALUTAZIONE (val / test) — metriche complete
# ============================================================

def evaluate(model, loader, pos_weight, cfg, desc="[EVAL]"):
    """Ritorna un dict con loss + tutte le metriche medie sul loader."""
    DEVICE = cfg["device"]
    TARGET_SIZE = cfg["target_size"]
    model.eval()

    acc = {k: 0.0 for k in METRIC_KEYS}
    n_batches = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            imgs = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)
            boxes = batch["boxes"].to(DEVICE)
            orig_h, orig_w = imgs.shape[-2], imgs.shape[-1]
            imgs = F.interpolate(imgs, (TARGET_SIZE, TARGET_SIZE), mode="bilinear", align_corners=False)
            masks = F.interpolate(masks.unsqueeze(1).float(), (TARGET_SIZE, TARGET_SIZE),
                                  mode="nearest").squeeze(1).long()
            boxes = scale_boxes(boxes, orig_h, orig_w, TARGET_SIZE)

            with torch.amp.autocast(device_type=DEVICE):
                all_preds, scores = model(imgs, boxes)
                preds = get_best_mask(all_preds, scores)
                loss = criterion(preds, masks, pos_weight=pos_weight).item()

            iou_fg, iou_bg, miou, f1, prec, rec = calculate_metrics(preds, masks)
            acc["loss"] += loss
            acc["iou"] += iou_fg
            acc["iou_bg"] += iou_bg
            acc["miou"] += miou
            acc["f1"] += f1
            acc["precision"] += prec
            acc["recall"] += rec
            n_batches += 1

    n_batches = max(n_batches, 1)
    return {k: v / n_batches for k, v in acc.items()}


# ============================================================
# TRAINING DI UNA SINGOLA VARIANTE
# ============================================================

def train_variant(cfg, variant_name, use_dfds, use_adapter):
    DEVICE = cfg["device"]
    TARGET_SIZE = cfg["target_size"]

    # Path dedicati alla variante.
    base, ext = os.path.splitext(cfg["checkpoint_name"])
    ext = ext or ".pth"
    ckpt_path = f"{base}_ablation_{variant_name}{ext}"
    os.makedirs("histories", exist_ok=True)
    history_path = os.path.join("histories", f"HISTORY_{cfg['dataset_name']}_ablation_{variant_name}.json")

    print("\n" + "#" * 60)
    print(f"# ABLAZIONE: {variant_name}")
    print(f"#   use_dfds={use_dfds} | use_adapter={use_adapter}")
    print(f"#   checkpoint -> {ckpt_path}")
    print(f"#   history    -> {history_path}")
    print("#" * 60)

    # Dataset + split ricostruiti da zero per ogni variante: l'augmentation
    # offline muta full_ds.data, quindi un dataset fresco evita contaminazioni
    # tra varianti.
    full_ds = build_dataset(cfg)
    train_subset, val_subset, test_subset, pos_weight = build_train_val_test_subsets(full_ds, cfg)

    offline_aug = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)])
    train_trans = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5), ToTensorV2()])
    eval_trans = A.Compose([ToTensorV2()])

    train_pos_indices = [i for i in train_subset.indices if i in set(full_ds.positive_indices)]
    full_ds.add_augmented_samples(indices=train_pos_indices, transform=offline_aug,
                                  n_copies=cfg["aug_copies"], rng_seed=cfg["random_seed"])
    aug_start = len(full_ds.data) - len(train_pos_indices) * cfg["aug_copies"]
    aug_indices = list(range(aug_start, len(full_ds.data)))
    train_subset = Subset(full_ds, list(train_subset.indices) + aug_indices)

    train_ds = TransformSubset(train_subset, train_trans)
    val_ds = TransformSubset(val_subset, eval_trans)
    test_ds = TransformSubset(test_subset, eval_trans)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, pin_memory=True)

    # Modello ablato (training sempre da zero: niente resume).
    model = build_ablation_model(cfg, DEVICE, use_dfds=use_dfds, use_adapter=use_adapter)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri addestrabili: {n_trainable:,}")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=5)
    scaler = torch.amp.GradScaler(device=DEVICE)

    history = []
    best_iou = 0.0
    patience_counter = 0

    for epoch in range(cfg["epochs"]):
        # ---- TRAIN ----
        model.train()
        acc = {k: 0.0 for k in METRIC_KEYS}
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"[{variant_name}] Ep {epoch+1}/{cfg['epochs']} [TRAIN]")
        for batch in pbar:
            imgs = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)
            boxes = batch["boxes"].to(DEVICE)
            orig_h, orig_w = imgs.shape[-2], imgs.shape[-1]
            imgs = F.interpolate(imgs, (TARGET_SIZE, TARGET_SIZE), mode="bilinear", align_corners=False)
            masks = F.interpolate(masks.unsqueeze(1).float(), (TARGET_SIZE, TARGET_SIZE),
                                  mode="nearest").squeeze(1).long()
            boxes = scale_boxes(boxes, orig_h, orig_w, TARGET_SIZE)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=DEVICE):
                all_preds, scores = model(imgs, boxes)
                preds = get_best_mask(all_preds, scores)
                loss = criterion(preds, masks, pos_weight=pos_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            iou_fg, iou_bg, miou, f1, prec, rec = calculate_metrics(preds.detach(), masks)
            acc["loss"] += loss.item()
            acc["iou"] += iou_fg
            acc["iou_bg"] += iou_bg
            acc["miou"] += miou
            acc["f1"] += f1
            acc["precision"] += prec
            acc["recall"] += rec
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou_fg:.4f}", miou=f"{miou:.4f}")

        n_batches = max(n_batches, 1)
        train_metrics = {k: v / n_batches for k, v in acc.items()}

        # ---- VAL ----
        val_metrics = evaluate(model, val_loader, pos_weight, cfg, desc=f"[{variant_name}] [VAL]")
        scheduler.step(val_metrics["loss"])

        # ---- HISTORY (1 record per epoca) ----
        record = {"epoch": epoch + 1}
        for k in METRIC_KEYS:
            record[f"train_{k}"] = train_metrics[k]
            record[f"val_{k}"] = val_metrics[k]
        history.append(record)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        print(f"[{variant_name}] EPOCA {epoch+1} | "
              f"Val Loss: {val_metrics['loss']:.4f} | Val IoU: {val_metrics['iou']:.4f} | "
              f"Val mIoU: {val_metrics['miou']:.4f} | Val F1: {val_metrics['f1']:.4f}")

        # ---- CHECKPOINT / EARLY STOPPING (su val IoU, come la baseline) ----
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            save_checkpoint(model, optimizer, scheduler, epoch, best_iou, ckpt_path)
            print(f"  Record IoU: {best_iou:.4f} — checkpoint salvato in {ckpt_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"\n[{variant_name}] EARLY STOPPING — Miglior Val IoU: {best_iou:.4f}")
                break

    # ---- TEST (sul miglior checkpoint della variante) ----
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    test_metrics = evaluate(model, test_loader, pos_weight, cfg, desc=f"[{variant_name}] [TEST]")

    print("\n" + "=" * 50)
    print(f"RISULTATI TEST — ablazione '{variant_name}' ({cfg['dataset_name']})")
    print("=" * 50)
    for k in METRIC_KEYS:
        print(f"  Test {k:9s}: {test_metrics[k]:.4f}")

    # Aggiunge il blocco test_results alla history (formato di plot_history).
    history.append({"test_results": {f"test_{k}": test_metrics[k] for k in METRIC_KEYS}})
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return {"variant": variant_name, "best_val_iou": best_iou,
            "test": test_metrics, "checkpoint": ckpt_path, "history": history_path}


# ============================================================
# ENTRY POINT
# ============================================================

def run_ablation(cfg, variants):
    results = []
    for name in variants:
        flags = VARIANTS[name]
        res = train_variant(cfg, name, use_dfds=flags["use_dfds"],
                             use_adapter=flags["use_adapter"])
        results.append(res)

    # Riepilogo comparativo finale.
    print("\n" + "=" * 64)
    print(f"RIEPILOGO ABLAZIONE — {cfg['dataset_name']}")
    print("=" * 64)
    header = f"{'variante':22s} {'val_iou':>8s} {'test_iou':>9s} {'test_miou':>10s} {'test_f1':>8s}"
    print(header)
    print("-" * len(header))
    for r in results:
        t = r["test"]
        print(f"{r['variant']:22s} {r['best_val_iou']:8.4f} "
              f"{t['iou']:9.4f} {t['miou']:10.4f} {t['f1']:8.4f}")
    print("\nPer i grafici:  python LENTE_plot_history.py histories/HISTORY_*ablation*.json")
    return results


def apply_overrides(cfg, args):
    if args.epochs is not None:      cfg["epochs"] = args.epochs
    if args.batch_size is not None:  cfg["batch_size"] = args.batch_size
    if args.lr is not None:          cfg["learning_rate"] = args.lr
    if args.checkpoint is not None:  cfg["checkpoint_name"] = args.checkpoint
    return cfg


def main():
    ap = argparse.ArgumentParser(description="LENTE — studio di ablazione (no DFDS / no DFDS+adapter)")
    ap.add_argument("-d", "--dataset", default=None, help="nome dataset (vedi config.yaml)")
    ap.add_argument("-c", "--config", default="config.yaml", help="path config (default: config.yaml)")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS.keys()),
                    choices=list(VARIANTS.keys()),
                    help="quali varianti eseguire (default: tutte)")
    # override opzionali
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--checkpoint", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, dataset=args.dataset)
    cfg = apply_overrides(cfg, args)

    print("\n" + "=" * 50)
    print(f"  Dataset : {cfg['dataset_name']}  (n_optical={cfg['n_optical']}, type={cfg['dataset_type']})")
    print(f"  Device  : {cfg['device']} | epochs={cfg['epochs']} | batch={cfg['batch_size']} | lr={cfg['learning_rate']}")
    print(f"  Varianti: {', '.join(args.variants)}")
    print("=" * 50)

    run_ablation(cfg, args.variants)


if __name__ == "__main__":
    main()
