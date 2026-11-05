#!/usr/bin/env python
"""
LENTE_compare.py — inferenza comparativa LENTE vs D2FLS.
========================================================
Per il dataset scelto (l4s o bijie) carica ENTRAMBI i modelli gia' addestrati,
fa inferenza sulle STESSE immagini del test set e salva figure affiancate in
stile LENTE_visualize_results:

    [ RGB ] [ Ground Truth ] [ LENTE ] [ D2FLS ]

Le ultime due colonne sono "error profile" rispetto al GT:
    verde = TP (corretto)   rosso = FP (falso positivo)   giallo = FN (mancato)

Uso:
    python LENTE_compare.py --dataset bijie
    python LENTE_compare.py -d l4s --num-samples 10
    python LENTE_compare.py -d bijie --lente-ckpt best_lente_bijie.pth \
                            --d2fls-ckpt best_d2fls_bijie.pth

NB: entrambi i modelli leggono lo STESSO stack del dataset LENTE
(build_dataset), cosi' vedono esattamente le stesse immagini e lo stesso split.
Da quello stack ricaviamo l'input per D2FLS (ottico + DEM) secondo la mappa
D2FLS_INPUT qui sotto.
"""

import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Subset

from LENTE_utils import (
    load_config, build_dataset, build_train_val_test_subsets, TransformSubset,
    build_model, get_best_mask, scale_boxes,
)
from LENTE_visualize_results import _make_rgb
from D2FLS import D2FLS_Net, SWIN_VARIANT


# I parametri D2FLS per dataset (canali ottici, canale DEM, risoluzione, nome
# checkpoint) sono nel blocco `d2fls:` di ciascun dataset in config.yaml.


# ============================================================
# INFERENZA per-modello
# ============================================================

@torch.no_grad()
def infer_lente(model, imgs, boxes, target_size, orig_h, orig_w, device):
    """Ritorna (prob_map, pred_mask) alla risoluzione originale."""
    imgs_r  = F.interpolate(imgs, (target_size, target_size), mode="bilinear", align_corners=False)
    boxes_s = scale_boxes(boxes.clone(), orig_h, orig_w, target_size)
    with torch.amp.autocast(device_type=device):
        all_preds, scores = model(imgs_r, boxes_s)
        preds = get_best_mask(all_preds, scores)
        preds = F.interpolate(preds.float(), (orig_h, orig_w), mode="bilinear", align_corners=False)
    prob = torch.sigmoid(preds).squeeze().cpu().numpy()
    return prob, (prob > 0.5).astype(np.uint8)


@torch.no_grad()
def infer_d2fls(model, imgs, dmap, orig_h, orig_w, img_size, device):
    """imgs: (1, in_opt, H, W) ottico; dmap: (1,1,H,W) DEM. Ritorna (prob, mask)."""
    opt = F.interpolate(imgs, (img_size, img_size), mode="bilinear", align_corners=False)
    dem = F.interpolate(dmap, (img_size, img_size), mode="bilinear", align_corners=False)
    with torch.amp.autocast(device_type=device):
        logits = model(opt, dem)                       # (1, 2, img_size, img_size)
        logits = F.interpolate(logits.float(), (orig_h, orig_w), mode="bilinear", align_corners=False)
        prob = F.softmax(logits, dim=1)[:, 1]          # prob classe landslide
    prob = prob.squeeze().cpu().numpy()
    return prob, (prob > 0.5).astype(np.uint8)


def _error_profile(pred_mask, gt_np):
    em = np.zeros((*gt_np.shape, 3), dtype=np.uint8)
    em[(pred_mask == 1) & (gt_np == 1)] = [0, 255, 0]      # TP verde
    em[(pred_mask == 1) & (gt_np == 0)] = [255, 0, 0]      # FP rosso
    em[(pred_mask == 0) & (gt_np == 1)] = [255, 255, 0]    # FN giallo
    return em


# ============================================================
# DRIVER
# ============================================================

def compare(cfg, lente_ckpt, d2fls_ckpt, num_samples):
    DEVICE      = cfg["device"]
    TARGET_SIZE = cfg["target_size"]
    name        = cfg["dataset_name"]
    dcfg        = cfg["d2fls"]

    print("Loading dataset...")
    full_ds = build_dataset(cfg)
    _, _, test_subset, _ = build_train_val_test_subsets(full_ds, cfg)

    # stessa scelta di campioni di LENTE_visualize_results: positivi del test set
    pos_set = set(full_ds.positive_indices)
    if num_samples == 0:
        viz_indices = list(test_subset.indices)
    else:
        pos_test = [i for i in test_subset.indices if i in pos_set]
        viz_indices = (pos_test[:num_samples] if len(pos_test) >= num_samples
                       else list(test_subset.indices)[:num_samples])

    viz_ds     = TransformSubset(Subset(full_ds, viz_indices), A.Compose([ToTensorV2()]))
    viz_loader = DataLoader(viz_ds, batch_size=1, shuffle=False)

    # --- LENTE ---
    print(f"Loading LENTE from {lente_ckpt} ...")
    lente = build_model(cfg, DEVICE)
    lente.load_state_dict(torch.load(lente_ckpt, map_location=DEVICE)["model_state_dict"], strict=False)
    lente.eval()

    # --- D2FLS ---
    print(f"Loading D2FLS from {d2fls_ckpt} "
          f"(in_channels_opt={dcfg['in_channels_opt']}, img_size={dcfg['img_size']}) ...")
    d2fls = D2FLS_Net(swin_variant=SWIN_VARIANT,
                      in_channels_opt=dcfg["in_channels_opt"],
                      num_classes=2, pretrained=False).to(DEVICE)
    d2fls.load_state_dict(torch.load(d2fls_ckpt, map_location=DEVICE)["model_state_dict"], strict=False)
    d2fls.eval()

    out_dir = f"comparison_{name}"
    os.makedirs(out_dir, exist_ok=True)

    for i, batch in enumerate(viz_loader):
        imgs  = batch["image"].to(DEVICE)                  # (1, C, H, W)
        masks = batch["mask"].to(DEVICE)
        boxes = batch["boxes"].to(DEVICE)
        orig_h, orig_w = imgs.shape[-2], imgs.shape[-1]

        # LENTE: stack completo
        l_prob, l_mask = infer_lente(lente, imgs, boxes, TARGET_SIZE, orig_h, orig_w, DEVICE)

        # D2FLS: estrae ottico + DEM dallo stesso stack
        opt_in = imgs[:, dcfg["opt_channels"], :, :]
        dem_in = imgs[:, dcfg["dem_channel"]:dcfg["dem_channel"] + 1, :, :]
        d_prob, d_mask = infer_d2fls(d2fls, opt_in, dem_in, orig_h, orig_w, dcfg["img_size"], DEVICE)

        img_np = imgs.squeeze().cpu().numpy()
        gt_np  = masks.squeeze().cpu().numpy().astype(np.uint8)
        rgb    = _make_rgb(img_np, cfg)

        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        axes[0].imshow(rgb);                 axes[0].set_title("RGB Composite", fontsize=12)
        axes[1].imshow(gt_np, cmap="gray");  axes[1].set_title("Ground Truth", fontsize=12)
        axes[2].imshow(_error_profile(l_mask, gt_np)); axes[2].set_title("LENTE", fontsize=12)
        axes[3].imshow(_error_profile(d_mask, gt_np)); axes[3].set_title("D2FLS", fontsize=12)
        for ax in axes:
            ax.axis("off")
        fig.suptitle("Green: TP   |   Red: FP   |   Yellow: FN", fontsize=11, y=0.04)
        plt.tight_layout(pad=2.0)

        out_path = f"{out_dir}/compare_idx{viz_indices[i]}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\rSaved {i+1}/{len(viz_indices)}: {out_path}", end="")

    print(f"\nFatto. Confronti salvati in: {out_dir}/")


def main():
    ap = argparse.ArgumentParser(description="Inferenza comparativa LENTE vs D2FLS")
    ap.add_argument("-d", "--dataset", default=None, help="l4s | bijie (default: active_dataset del config)")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("-n", "--num-samples", type=int, default=8,
                    help="quante immagini di test confrontare (0 = tutte)")
    ap.add_argument("--lente-ckpt", default=None, help="override path checkpoint LENTE")
    ap.add_argument("--d2fls-ckpt", default=None, help="override path checkpoint D2FLS")
    args = ap.parse_args()

    cfg = load_config(args.config, dataset=args.dataset)
    name = cfg["dataset_name"]
    if "d2fls" not in cfg:
        raise ValueError(f"Dataset '{name}': manca il blocco 'd2fls:' in config.yaml.")

    lente_ckpt = args.lente_ckpt or cfg["checkpoint_name"]
    d2fls_ckpt = args.d2fls_ckpt or cfg["d2fls"]["checkpoint_name"]

    print("=" * 56)
    print(f"  CONFRONTO LENTE vs D2FLS — dataset={name} | device={cfg['device']}")
    print(f"  LENTE ckpt: {lente_ckpt}")
    print(f"  D2FLS ckpt: {d2fls_ckpt}")
    print("=" * 56)

    compare(cfg, lente_ckpt, d2fls_ckpt, args.num_samples)


if __name__ == "__main__":
    main()
