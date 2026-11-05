"""
LENTE_multi_utils.py
====================
Modello LENTE *unificato* che accetta in un solo checkpoint sia ingressi L4S
(12 bande ottiche + slope + DEM = 14ch) sia ingressi Bijie (RGB + slope + DEM =
5ch), addestrato congiuntamente sui due dataset.

Differenza rispetto a `LENTE` (LENTE_utils.py): il patch-embed di SAM2 non e'
piu' fisso a un solo numero di canali. Viene sostituito con `DualPatchStem`, che
contiene DUE conv di ingresso (12->C per L4S, 3->C per Bijie) e instrada in base
a un flag `active`. Tutto il resto (encoder SAM2-Tiny frozen, adapter LoRA-like,
mask decoder, DSDF early-guidance a 2ch) e' condiviso fra i due domini.

I punti delicati del training congiunto (gestiti qui e in LENTE_multi_train.py):
  * pos_weight calcolato per-dataset e applicato al batch del dominio giusto;
  * batch omogenei alternati (un dominio per batch) con bilanciamento 50/50;
  * normalizzazione per-dataset dell'ottico (z-score) prima dello stem condiviso,
    cosi' l'encoder non vede scale incoerenti (riflettanze S2 vs RGB in [0,1]).

Riusa dataset, loss, metriche e IO da LENTE_utils.
"""

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from transformers import Sam2Model

from LENTE_utils import (
    load_config, build_dataset, build_train_val_test_subsets, TransformSubset,
    DSDFEarlyGuidance, Adapter,
    criterion, get_best_mask, calculate_metrics, scale_boxes,  # noqa: F401 (riesportati)
    save_checkpoint, load_checkpoint,                          # noqa: F401
)


# ============================================================
# CONFIG
# ============================================================

def load_multi_config(config_path="config.yaml"):
    """
    Ritorna un dict flat per il training congiunto: merge di `common` + `multi`,
    piu' `members_cfg` = { "l4s": <cfg flat>, "bijie": <cfg flat> } ottenuti
    riusando load_config() sui rispettivi blocchi `datasets:`.
    """
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    if "multi" not in raw:
        raise ValueError("Blocco 'multi' assente in config.yaml")

    multi = raw["multi"]
    members = multi.get("members", ["l4s", "bijie"])

    cfg = dict(raw["common"])
    cfg.update(multi)
    cfg["members_cfg"] = {name: load_config(config_path, dataset=name) for name in members}
    return cfg


# ============================================================
# DUAL INPUT STEM
# ============================================================

class DualPatchStem(nn.Module):
    """
    Sostituisce il Conv2d(3->C) interno del vision encoder di SAM2 con due
    conv di ingresso intercambiabili. `active` ("l4s"|"bijie") decide quale usare.

    - proj_bijie : Conv2d(3 -> C) inizializzato dai pesi RGB originali;
    - proj_l4s   : Conv2d(n_optical_l4s -> C), pesi RGB copiati su [:3] e i canali
                   extra inizializzati a N(0, 1e-3) (contributo iniziale minimo).
    Stessa logica di espansione gia' usata in LENTE_utils.LENTE.
    """
    def __init__(self, orig_conv: nn.Conv2d, n_optical_l4s: int = 12):
        super().__init__()
        C = orig_conv.out_channels
        has_bias = orig_conv.bias is not None
        k, s, p = orig_conv.kernel_size, orig_conv.stride, orig_conv.padding

        self.proj_bijie = nn.Conv2d(3, C, kernel_size=k, stride=s, padding=p, bias=has_bias)
        self.proj_l4s   = nn.Conv2d(n_optical_l4s, C, kernel_size=k, stride=s, padding=p, bias=has_bias)

        with torch.no_grad():
            # Bijie: copia esatta dei pesi RGB originali
            self.proj_bijie.weight.copy_(orig_conv.weight)
            # L4S: RGB sui primi 3 canali, rumore piccolo sui restanti
            self.proj_l4s.weight[:, :3].copy_(orig_conv.weight)
            nn.init.normal_(self.proj_l4s.weight[:, 3:], mean=0.0, std=0.001)
            if has_bias:
                self.proj_bijie.bias.copy_(orig_conv.bias)
                self.proj_l4s.bias.copy_(orig_conv.bias)

        self.active = "bijie"

    @property
    def _active_proj(self):
        return self.proj_l4s if self.active == "l4s" else self.proj_bijie

    # Il patch_embed di SAM2 accede a `projection.weight.dtype` prima del forward:
    # esponiamo weight/bias del conv attivo per compatibilita'.
    @property
    def weight(self):
        return self._active_proj.weight

    @property
    def bias(self):
        return self._active_proj.bias

    def forward(self, x):
        return self._active_proj(x)


# ============================================================
# MODELLO UNIFICATO
# ============================================================

class LENTEMulti(nn.Module):
    """
    SAM2-Tiny + Adapter + DSDF early-guidance con DUE stem di ingresso.
    forward(x, boxes, source): `source` in {"l4s","bijie"} instrada lo stem e la
    normalizzazione ottica del dominio corrispondente.
    """
    N_OPTICAL = {"l4s": 12, "bijie": 3}

    def __init__(self, sam_ckpt_path, n_optical_l4s=12, n_optical_bijie=3, adapter_dim=64):
        super().__init__()
        self.N_OPTICAL = {"l4s": n_optical_l4s, "bijie": n_optical_bijie}

        self.model = Sam2Model.from_pretrained(sam_ckpt_path)
        self.early_guidance = DSDFEarlyGuidance()
        v_encoder = self.model.vision_encoder

        # --- sostituisci il patch-embed Conv2d(in=3) con DualPatchStem ---
        self.patch_stem = None
        for name, module in v_encoder.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                stem = DualPatchStem(module, n_optical_l4s=n_optical_l4s)
                parts  = name.split(".")
                parent = v_encoder.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else v_encoder
                setattr(parent, parts[-1], stem)
                self.patch_stem = stem
                break
        if self.patch_stem is None:
            raise RuntimeError("Patch embed Conv2d(in=3) non trovato nel vision encoder.")

        # --- adapter LoRA-like sugli attention (riuso esatto di LENTE) ---
        self.adapters = nn.ModuleList()
        for name, module in v_encoder.named_modules():
            if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear):
                dim = module.proj.out_features if hasattr(module, "proj") else module.qkv.in_features
                ad = Adapter(dim, adapter_dim); self.adapters.append(ad)
                module.register_forward_hook(self._make_hook(ad))
            elif hasattr(module, "q_proj") and isinstance(module.q_proj, nn.Linear):
                dim = module.out_proj.out_features if hasattr(module, "out_proj") else module.q_proj.in_features
                ad = Adapter(dim, adapter_dim); self.adapters.append(ad)
                module.register_forward_hook(self._make_hook(ad))

        # --- normalizzazione per-dataset dell'ottico (buffer; default identita') ---
        # mean=0, std=1 -> nessun effetto finche' non si chiama set_optical_norm().
        self.register_buffer("mean_l4s",   torch.zeros(n_optical_l4s))
        self.register_buffer("std_l4s",    torch.ones(n_optical_l4s))
        self.register_buffer("mean_bijie", torch.zeros(n_optical_bijie))
        self.register_buffer("std_bijie",  torch.ones(n_optical_bijie))

        self._freeze()

    @staticmethod
    def _make_hook(adapter):
        def hook(module, args, output):
            if isinstance(output, tuple):
                return (adapter(output[0]),) + output[1:]
            return adapter(output)
        return hook

    def _freeze(self):
        for p in self.model.parameters(): p.requires_grad = False
        for p in self.model.mask_decoder.parameters(): p.requires_grad = True
        for p in self.patch_stem.parameters(): p.requires_grad = True      # entrambi gli stem
        for ad in self.adapters:
            for p in ad.parameters(): p.requires_grad = True
        for p in self.early_guidance.parameters(): p.requires_grad = True

    @torch.no_grad()
    def set_optical_norm(self, source, mean, std):
        """Imposta le statistiche z-score (per-canale) del dominio `source`."""
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std  = torch.as_tensor(std,  dtype=torch.float32).clamp_min(1e-6)
        getattr(self, f"mean_{source}").copy_(mean.to(getattr(self, f"mean_{source}").device))
        getattr(self, f"std_{source}").copy_(std.to(getattr(self, f"std_{source}").device))

    def forward(self, x, boxes, source):
        if source not in self.N_OPTICAL:
            raise ValueError(f"source sconosciuto: {source}")
        n = self.N_OPTICAL[source]
        optical = x[:, :n]       # (B, n, H, W)
        terrain = x[:, n:]       # (B, 2, H, W)

        # z-score per-dataset (identita' se stats = 0/1)
        mean = getattr(self, f"mean_{source}").view(1, -1, 1, 1)
        std  = getattr(self, f"std_{source}").view(1, -1, 1, 1)
        optical = (optical - mean) / std

        # DSDF early guidance (terrain 2ch -> guidance map che modula l'ottico)
        g = self.early_guidance(terrain)
        optical = optical * g

        # instrada lo stem del dominio corrente
        self.patch_stem.active = source

        if boxes.ndim == 2:
            boxes = boxes.unsqueeze(1)
        outputs = self.model(pixel_values=optical, input_boxes=boxes, multimask_output=False)
        scores = getattr(outputs, "iou_scores",
                         getattr(outputs, "iou_predictions",
                                 getattr(outputs, "pred_ious", None)))
        return outputs.pred_masks, scores


def build_multi_model(cfg, device):
    """Costruisce LENTEMulti dai canali ottici dei due membri nel config."""
    mc = cfg["members_cfg"]
    return LENTEMulti(
        sam_ckpt_path   = cfg["sam_ckpt_path"],
        n_optical_l4s   = mc["l4s"]["n_optical"],
        n_optical_bijie = mc["bijie"]["n_optical"],
        adapter_dim     = cfg["adapter_dim"],
    ).to(device)


# ============================================================
# DATI: preparazione per-dominio + schedule interleaved
# ============================================================

@torch.no_grad()
def compute_optical_stats(full_ds, indices, n_optical, max_samples=300):
    """Media/std per-canale dell'ottico (primi n_optical canali) sul train set."""
    idxs = list(indices)[:max_samples]
    s  = torch.zeros(n_optical, dtype=torch.float64)
    ss = torch.zeros(n_optical, dtype=torch.float64)
    count = 0
    for i in idxs:
        img = full_ds[i]["image"][:n_optical].double()        # (n,H,W)
        s  += img.sum(dim=(1, 2))
        ss += (img ** 2).sum(dim=(1, 2))
        count += img.shape[1] * img.shape[2]
    mean = (s / count)
    var  = (ss / count - mean ** 2).clamp_min(1e-12)
    return mean.float(), var.sqrt().float()


def prepare_member(member_cfg, cfg, offline_aug, train_trans, val_trans):
    """
    Costruisce dataset + split + augmentation offline dei positivi + loader
    train/val per UN dominio, e calcola pos_weight e statistiche z-score.

    Ritorna un dict con: source, train_loader, val_loader, pos_weight,
    n_optical, opt_mean, opt_std.
    """
    source  = member_cfg["dataset_name"]
    full_ds = build_dataset(member_cfg)
    train_subset, val_subset, _, pos_weight = build_train_val_test_subsets(full_ds, member_cfg)

    # stats z-score sul train ORIGINALE (prima dell'augmentation)
    n_opt = member_cfg["n_optical"]
    opt_mean, opt_std = compute_optical_stats(full_ds, train_subset.indices, n_opt)

    # augmentation offline dei positivi del train (come LENTE_train)
    pos_set = set(full_ds.positive_indices)
    train_pos = [i for i in train_subset.indices if i in pos_set]
    full_ds.add_augmented_samples(indices=train_pos, transform=offline_aug,
                                  n_copies=cfg["aug_copies"], rng_seed=cfg["random_seed"])
    aug_start   = len(full_ds.data) - len(train_pos) * cfg["aug_copies"]
    aug_indices = list(range(aug_start, len(full_ds.data)))
    train_subset = Subset(full_ds, list(train_subset.indices) + aug_indices)

    train_ds = TransformSubset(train_subset, train_trans)
    val_ds   = TransformSubset(val_subset,   val_trans)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, pin_memory=True)

    print(f"[{source}] n_optical={n_opt} | pos_weight={pos_weight:.2f} | "
          f"train_batches={len(train_loader)} val_batches={len(val_loader)} | "
          f"opt_mean~[{opt_mean.min():.3g},{opt_mean.max():.3g}] std~[{opt_std.min():.3g},{opt_std.max():.3g}]")

    return {
        "source": source,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "pos_weight": pos_weight,
        "n_optical": n_opt,
        "opt_mean": opt_mean,
        "opt_std": opt_std,
    }


def interleaved_batches(loaders):
    """
    Genera (source, batch) alternando un batch per dominio ad ogni "giro".
    Il loader piu' corto viene ri-iterato (oversampling) fino ad esaurire il
    piu' lungo, garantendo esposizione 50/50 dei due domini per epoca.
    """
    iters   = {s: iter(l) for s, l in loaders.items()}
    n_steps = max(len(l) for l in loaders.values())
    order   = list(loaders.keys())
    for _ in range(n_steps):
        for s in order:
            try:
                batch = next(iters[s])
            except StopIteration:
                iters[s] = iter(loaders[s])
                batch = next(iters[s])
            yield s, batch


def num_interleaved_steps(loaders):
    return max(len(l) for l in loaders.values()) * len(loaders)
