#!/usr/bin/env python
"""
LENTE_cv.py — K-fold (default 5) cross-validation.
=================================================
Esegue una cross-validation a K fold sul dataset scelto, riusando le stesse
utility di train/test (dataset, modello, loss, metriche, augmentation offline).

Per ogni fold:
  - il fold k-esimo e' il TEST set (tenuto da parte);
  - dai restanti campioni si ritaglia una porzione di VALIDATION (val_fraction)
    per l'early stopping; il resto e' TRAIN;
  - si allena un modello da zero con early stopping sull'IoU di validazione;
  - si valuta sul fold di test e si registrano le metriche.

Alla fine stampa media +/- deviazione standard di IoU / mIoU / F1 / Precision /
Recall sui K fold e salva un riepilogo JSON.

Uso:
    python LENTE_cv.py                                 # dataset = active_dataset del config
    python LENTE_cv.py --dataset bijie --folds 5
    python LENTE_cv.py -d l4s -k 5 --epochs 50 --batch-size 4 --lr 5e-5

Oppure tramite il launcher, se aggiunto a LENTE_run.py (mode "cv").
"""

import os
import csv
import json
import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from LENTE_utils import (
    load_config, build_dataset, TransformSubset,
    build_model, criterion, get_best_mask, calculate_metrics, scale_boxes,
)


# ============================================================
# SPLIT helpers
# ============================================================

def make_kfold_indices(n, k, seed):
    """Ritorna una lista di K array di indici (i fold di test), shuffle deterministico."""
    rng  = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng).tolist()
    return [perm[i::k] for i in range(k)]   # round-robin: fold bilanciati in dimensione


def compute_pos_weight(full_ds, train_indices, cfg):
    """Replica la logica di pos_weight di build_train_val_test_subsets."""
    pos_set = set(full_ds.positive_indices)
    mode = cfg.get("pos_weight_mode", "pixels")
    if mode == "heuristic":
        n_pos = sum(1 for i in train_indices if i in pos_set)
        n_neg = len(train_indices) - n_pos
        ls_frac  = 0.35
        pixel_ls = n_pos * ls_frac
        pixel_bg = n_neg + n_pos * (1.0 - ls_frac)
        return pixel_bg / max(pixel_ls, 1.0)
    px_fg = px_tot = 0
    for i in train_indices:
        m = np.asarray(full_ds.get_mask(i))
        px_fg  += int(m.sum())
        px_tot += m.size
    return (px_tot - px_fg) / max(px_fg, 1)


# ============================================================
# TRAIN / EVAL di un singolo fold
# ============================================================

def train_one_fold(cfg, full_ds, train_indices, val_indices, fold_id):
    """Allena un modello da zero su train_indices, early stopping su val_indices.
    Ritorna (model, pos_weight). Aggiunge campioni augmentati a full_ds: il chiamante
    deve ripristinare lo stato del dataset dopo il fold."""
    DEVICE      = cfg["device"]
    TARGET_SIZE = cfg["target_size"]

    pos_weight = compute_pos_weight(full_ds, train_indices, cfg)

    offline_aug = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)])
    train_trans = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5), ToTensorV2()])
    val_trans   = A.Compose([ToTensorV2()])

    # Augmentation offline solo sui positivi di TRAIN (come in LENTE_train.train)
    pos_set = set(full_ds.positive_indices)
    train_pos_indices = [i for i in train_indices if i in pos_set]
    len_before = len(full_ds.data)
    full_ds.add_augmented_samples(indices=train_pos_indices, transform=offline_aug,
                                  n_copies=cfg["aug_copies"], rng_seed=cfg["random_seed"] + fold_id)
    aug_indices = list(range(len_before, len(full_ds.data)))

    train_subset = Subset(full_ds, list(train_indices) + aug_indices)
    val_subset   = Subset(full_ds, list(val_indices))
    train_ds = TransformSubset(train_subset, train_trans)
    val_ds   = TransformSubset(val_subset,   val_trans)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, pin_memory=True)

    model = build_model(cfg, DEVICE)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=5)
    scaler    = torch.amp.GradScaler(device=DEVICE)

    ckpt_path = _fold_ckpt_name(cfg, fold_id)
    best_iou = 0.0
    patience_counter = 0
    print(f"\n--- Fold {fold_id+1} | train {len(train_subset)} (+aug) | val {len(val_subset)} | "
          f"pos_weight={pos_weight:.2f} ---")

    for epoch in range(cfg["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Fold {fold_id+1} Ep {epoch+1}/{cfg['epochs']} [TRAIN]")
        for batch in pbar:
            imgs, masks, boxes = batch["image"].to(DEVICE), batch["mask"].to(DEVICE), batch["boxes"].to(DEVICE)
            orig_h, orig_w = imgs.shape[-2], imgs.shape[-1]
            imgs  = F.interpolate(imgs, (TARGET_SIZE, TARGET_SIZE), mode="bilinear", align_corners=False)
            masks = F.interpolate(masks.unsqueeze(1).float(), (TARGET_SIZE, TARGET_SIZE), mode="nearest").squeeze(1).long()
            boxes = scale_boxes(boxes, orig_h, orig_w, TARGET_SIZE)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=DEVICE):
                all_preds, scores = model(imgs, boxes)
                preds = get_best_mask(all_preds, scores)
                loss = criterion(preds, masks, pos_weight=pos_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            iou_fg, _, _, _, _, _ = calculate_metrics(preds.detach(), masks)
            pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou_fg:.4f}")

        # validation
        model.eval()
        v_loss = v_iou = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Fold {fold_id+1} [VAL]"):
                imgs, masks, boxes = batch["image"].to(DEVICE), batch["mask"].to(DEVICE), batch["boxes"].to(DEVICE)
                orig_h, orig_w = imgs.shape[-2], imgs.shape[-1]
                imgs  = F.interpolate(imgs, (TARGET_SIZE, TARGET_SIZE), mode="bilinear", align_corners=False)
                masks = F.interpolate(masks.unsqueeze(1).float(), (TARGET_SIZE, TARGET_SIZE), mode="nearest").squeeze(1).long()
                boxes = scale_boxes(boxes, orig_h, orig_w, TARGET_SIZE)
                with torch.amp.autocast(device_type=DEVICE):
                    all_preds, scores = model(imgs, boxes)
                    preds = get_best_mask(all_preds, scores)
                    v_loss += criterion(preds, masks, pos_weight=pos_weight).item()
                iou_fg, _, _, _, _, _ = calculate_metrics(preds, masks)
                v_iou += iou_fg

        avg_val_loss = v_loss / max(len(val_loader), 1)
        avg_val_iou  = v_iou  / max(len(val_loader), 1)
        scheduler.step(avg_val_loss)
        print(f"  Fold {fold_id+1} EPOCA {epoch+1} | Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save({"model_state_dict": model.state_dict(), "best_iou": best_iou}, ckpt_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"  Fold {fold_id+1} EARLY STOPPING — best Val IoU: {best_iou:.4f}")
                break

    # ricarica i pesi migliori del fold
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    return model, pos_weight


@torch.no_grad()
def evaluate_fold(cfg, model, full_ds, test_indices, pos_weight):
    DEVICE      = cfg["device"]
    TARGET_SIZE = cfg["target_size"]

    test_subset = Subset(full_ds, list(test_indices))
    test_ds     = TransformSubset(test_subset, A.Compose([ToTensorV2()]))
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, pin_memory=True)

    model.eval()
    agg = {"loss": 0.0, "iou": 0.0, "miou": 0.0, "f1": 0.0, "prec": 0.0, "rec": 0.0}
    for batch in tqdm(test_loader, desc="[TEST fold]"):
        imgs, masks, boxes = batch["image"].to(DEVICE), batch["mask"].to(DEVICE), batch["boxes"].to(DEVICE)
        orig_h, orig_w = imgs.shape[-2], imgs.shape[-1]
        imgs  = F.interpolate(imgs, (TARGET_SIZE, TARGET_SIZE), mode="bilinear", align_corners=False)
        masks = F.interpolate(masks.unsqueeze(1).float(), (TARGET_SIZE, TARGET_SIZE), mode="nearest").squeeze(1).long()
        boxes = scale_boxes(boxes, orig_h, orig_w, TARGET_SIZE)
        with torch.amp.autocast(device_type=DEVICE):
            all_preds, scores = model(imgs, boxes)
            preds = get_best_mask(all_preds, scores)
            agg["loss"] += criterion(preds, masks, pos_weight=pos_weight).item()
        iou_fg, _, miou, f1, prec, rec = calculate_metrics(preds, masks)
        agg["iou"]  += iou_fg; agg["miou"] += miou; agg["f1"] += f1
        agg["prec"] += prec;   agg["rec"]  += rec

    n = max(len(test_loader), 1)
    return {k: v / n for k, v in agg.items()}


def _fold_ckpt_name(cfg, fold_id):
    base = os.path.splitext(cfg["checkpoint_name"])[0]
    return f"{base}_cv_fold{fold_id+1}.pth"


# ============================================================
# DRIVER
# ============================================================

def cross_validate(cfg, k=5):
    full_ds = build_dataset(cfg)
    n = len(full_ds)

    # stato originale del dataset (prima di qualsiasi augmentation) per ripristino tra i fold
    orig_n   = len(full_ds.data)
    orig_pos = list(full_ds.positive_indices)
    orig_neg = list(full_ds.negative_indices)

    folds = make_kfold_indices(n, k, cfg["random_seed"])
    val_frac = cfg["val_fraction"]

    results = []
    for fold_id in range(k):
        test_indices = folds[fold_id]
        rest = [i for j in range(k) if j != fold_id for i in folds[j]]

        # ritaglia la validation dal "rest" (proporzione val_fraction sul totale)
        n_val = max(1, int(round(len(rest) * val_frac)))
        val_indices   = rest[:n_val]
        train_indices = rest[n_val:]

        model, pos_weight = train_one_fold(cfg, full_ds, train_indices, val_indices, fold_id)
        metrics = evaluate_fold(cfg, model, full_ds, test_indices, pos_weight)
        metrics["fold"] = fold_id + 1
        metrics["n_test"] = len(test_indices)
        results.append(metrics)
        print(f"  Fold {fold_id+1} TEST | IoU {metrics['iou']:.4f} | mIoU {metrics['miou']:.4f} | "
              f"F1 {metrics['f1']:.4f} | P {metrics['prec']:.4f} | R {metrics['rec']:.4f}")

        # cleanup memoria + ripristino dataset al suo stato originale
        del model
        if cfg["device"] == "cuda":
            torch.cuda.empty_cache()
        del full_ds.data[orig_n:]
        full_ds.positive_indices = list(orig_pos)
        full_ds.negative_indices = list(orig_neg)

    _report(cfg, results, k)
    return results


def _report(cfg, results, k):
    keys = ["iou", "miou", "f1", "prec", "rec", "loss"]

    means = {key: float(np.mean([r[key] for r in results])) for key in keys}
    stds  = {key: float(np.std([r[key] for r in results])) for key in keys}
    summary = {key: {"mean": means[key], "std": stds[key]} for key in keys}

    # --- Costruisce la tabella come testo (riusata per stdout e file .txt) ---
    header = f"{'fold':<6}" + "".join(f"{key:>11}" for key in keys)
    lines = [
        "=" * 72,
        f"  {k}-FOLD CROSS-VALIDATION — {cfg['dataset_name']}",
        "=" * 72,
        header,
        "-" * len(header),
    ]
    for r in results:
        lines.append(f"{r['fold']:<6}" + "".join(f"{r[key]:>11.4f}" for key in keys))
    lines.append("-" * len(header))
    lines.append(f"{'mean':<6}" + "".join(f"{means[key]:>11.4f}" for key in keys))
    lines.append(f"{'std':<6}"  + "".join(f"{stds[key]:>11.4f}"  for key in keys))
    lines.append("=" * 72)

    table = "\n".join(lines)
    print("\n" + table)

    base = os.path.splitext(cfg["checkpoint_name"])[0]

    # --- JSON (per fold + summary) ---
    json_path = f"cv_results_{base}.json"
    out = {"dataset": cfg["dataset_name"], "k": k, "per_fold": results, "summary": summary}
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    # --- TXT (la stessa tabella leggibile) ---
    txt_path = f"cv_results_{base}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table + "\n")

    # --- CSV (per fold + righe mean/std) ---
    csv_path = f"cv_results_{base}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fold"] + keys)
        for r in results:
            w.writerow([r["fold"]] + [f"{r[key]:.6f}" for key in keys])
        w.writerow(["mean"] + [f"{means[key]:.6f}" for key in keys])
        w.writerow(["std"]  + [f"{stds[key]:.6f}"  for key in keys])

    print(f"Risultati salvati in:\n  {json_path}\n  {txt_path}\n  {csv_path}")


def main():
    ap = argparse.ArgumentParser(description="LENTE K-fold cross-validation")
    ap.add_argument("-d", "--dataset", default=None, help="dataset name (vedi config.yaml)")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("-k", "--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, dataset=args.dataset)
    if args.epochs is not None:     cfg["epochs"] = args.epochs
    if args.batch_size is not None: cfg["batch_size"] = args.batch_size
    if args.lr is not None:         cfg["learning_rate"] = args.lr

    print("=" * 56)
    print(f"  CV {args.folds}-fold | dataset={cfg['dataset_name']} | device={cfg['device']}")
    print(f"  epochs={cfg['epochs']} | batch={cfg['batch_size']} | lr={cfg['learning_rate']}")
    print("=" * 56)

    cross_validate(cfg, k=args.folds)


if __name__ == "__main__":
    main()
