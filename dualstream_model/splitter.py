"""
split_dataset_h5.py
===================
Replica ESATTAMENTE lo split 60/20/20 usato in train_sam2tiny_adapter_h5.py
e lo materializza su disco (copia o symlink dei file .h5) + un manifest JSON.

Punto chiave (verificato sul codice di training):
  Lo split NON dipende dalle classi pos/neg. È una permutazione casuale
  degli indici con torch.randperm(seed=42), poi affettata:
      n_test  = int(N * 0.20)
      test    = shuffled[:n_test]
      n_val   = int(len(remaining) * (0.20 / 0.80))   # = 25% del residuo
      val     = remaining[:n_val]
      train   = remaining[n_val:]
  Gli indici mappano ai file via `sorted(glob.glob(img_dir/*.h5))`, identico
  a LandslideH5Dataset. Riproducendo questo, i set sono bit-identici.

  Il conteggio pos/neg serve solo per i log e per pos_weight: lo includiamo
  per verificare che lo split combaci con quello stampato dal training.

NOTA sull'augmentation:
  Nel training l'augmentation offline (HFlip/VFlip/Rotate90 x AUG_COPIES) viene
  applicata SOLO ai positivi del TRAIN, DOPO lo split, generando campioni
  sintetici. Non fa parte dello split del dataset originale, quindi qui NON
  viene materializzata. Se ti serve, posso aggiungerla come passo separato.

Dipendenze:
    pip install torch h5py numpy

Uso:
    python split_dataset_h5.py \
        --dataset-root /datadrive/landslide/SAM3/SAM2/LandSlide4Sense \
        --output-root  /datadrive/landslide/SAM3/SAM2/LandSlide4Sense_split \
        --mode copy        # copy | symlink | manifest-only
"""

import os
import glob
import json
import shutil
import argparse

import h5py
import numpy as np
import torch


# ============================================================
# CONFIG (default identici al training)
# ============================================================

DATASET_ROOT = r"/datadrive/landslide/dualstream_model/LandSlide4Sense/"
OUTPUT_ROOT  = r"/datadrive/landslide/dualstream_model/LandSlide4Sense_split"

IMG_DIR  = "TrainData/img"
MASK_DIR = "TrainData/mask"

VAL_FRACTION  = 0.20
TEST_FRACTION = 0.20
RANDOM_SEED   = 42

# heuristic di pos_weight, identica al training (solo per il manifest)
LS_FRAC = 0.35


# ============================================================
# 1. Enumerazione file (identica a LandslideH5Dataset)
# ============================================================

def enumerate_pairs(root_dir):
    """Ritorna lista di (img_path, mask_path) nell'ordine ESATTO del training."""
    img_dir  = os.path.join(root_dir, IMG_DIR)
    mask_dir = os.path.join(root_dir, MASK_DIR)

    all_images = sorted(glob.glob(os.path.join(img_dir, "*.h5")))
    if not all_images:
        raise FileNotFoundError(f"Nessun .h5 trovato in {img_dir}")

    pairs = []
    for img_path in all_images:
        mask_name = os.path.basename(img_path).replace("image_", "mask_")
        mask_path = os.path.join(mask_dir, mask_name)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask mancante per {img_path}: {mask_path}")
        pairs.append((img_path, mask_path))
    return pairs


def classify_pos_neg(pairs):
    """Indici positivi (mask.sum() > 0) / negativi — solo masks, veloce."""
    positive_indices, negative_indices = [], []
    for idx, (_img, mask_path) in enumerate(pairs):
        with h5py.File(mask_path, "r") as f:
            mask = f["mask"][:]
        if mask.sum() > 0:
            positive_indices.append(idx)
        else:
            negative_indices.append(idx)
    return positive_indices, negative_indices


# ============================================================
# 2. Split (logica copiata 1:1 da build_train_val_test_subsets)
# ============================================================

def build_split(n_total, positive_indices,
                val_fraction, test_fraction, random_seed):
    rng      = torch.Generator().manual_seed(random_seed)
    all_idx  = list(range(n_total))
    perm     = torch.randperm(n_total, generator=rng).tolist()
    shuffled = [all_idx[i] for i in perm]
    pos_set  = set(positive_indices)

    n_test        = int(len(shuffled) * test_fraction)
    test_indices  = shuffled[:n_test]
    remaining     = shuffled[n_test:]
    n_val         = int(len(remaining) * (val_fraction / (1.0 - test_fraction)))
    val_indices   = remaining[:n_val]
    train_indices = remaining[n_val:]

    def _count(idxs):
        n_p = sum(1 for i in idxs if i in pos_set)
        return n_p, len(idxs) - n_p

    n_pos_tr, n_neg_tr = _count(train_indices)
    n_pos_vl, n_neg_vl = _count(val_indices)
    n_pos_te, n_neg_te = _count(test_indices)

    n_pos_tot = len(positive_indices)
    n_neg_tot = n_total - n_pos_tot

    print(
        f"\nSplit 60/20/20:"
        f"\n  Totale : {n_total:5d}  ({n_pos_tot} pos + {n_neg_tot} neg)"
        f"\n  Train  : {len(train_indices):5d}  ({n_pos_tr} pos + {n_neg_tr} neg)"
        f"\n  Val    : {len(val_indices):5d}  ({n_pos_vl} pos + {n_neg_vl} neg)"
        f"\n  Test   : {len(test_indices):5d}  ({n_pos_te} pos + {n_neg_te} neg)"
    )

    pixel_ls   = n_pos_tr * LS_FRAC
    pixel_bg   = n_neg_tr + n_pos_tr * (1.0 - LS_FRAC)
    pos_weight = pixel_bg / max(pixel_ls, 1.0)
    print(f"  pos_weight: {pos_weight:.2f}\n")

    stats = {
        "train": {"n": len(train_indices), "pos": n_pos_tr, "neg": n_neg_tr},
        "val":   {"n": len(val_indices),   "pos": n_pos_vl, "neg": n_neg_vl},
        "test":  {"n": len(test_indices),  "pos": n_pos_te, "neg": n_neg_te},
        "pos_weight": pos_weight,
    }
    return train_indices, val_indices, test_indices, stats


# ============================================================
# 3. Materializzazione su disco
# ============================================================

def materialize(split_name, indices, pairs, output_root, mode):
    """Copia/symlink i file .h5 di uno split in <output_root>/<split>/{img,mask}."""
    img_out  = os.path.join(output_root, split_name, "img")
    mask_out = os.path.join(output_root, split_name, "mask")
    os.makedirs(img_out,  exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    file_records = []
    for idx in indices:
        img_src, mask_src = pairs[idx]
        img_dst  = os.path.join(img_out,  os.path.basename(img_src))
        mask_dst = os.path.join(mask_out, os.path.basename(mask_src))

        if mode == "copy":
            shutil.copy2(img_src,  img_dst)
            shutil.copy2(mask_src, mask_dst)
        elif mode == "symlink":
            for src, dst in ((img_src, img_dst), (mask_src, mask_dst)):
                if os.path.islink(dst) or os.path.exists(dst):
                    os.remove(dst)
                os.symlink(os.path.abspath(src), dst)
        # mode == "manifest-only": non si tocca il filesystem dei dati

        file_records.append({
            "orig_index": idx,
            "img":  os.path.basename(img_src),
            "mask": os.path.basename(mask_src),
        })

    if mode != "manifest-only":
        print(f"  {split_name:5s}: {len(indices)} coppie -> {os.path.join(output_root, split_name)} ({mode})")
    return file_records


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Split del dataset .h5 identico al training SAM2-Tiny.")
    ap.add_argument("--dataset-root", default=DATASET_ROOT)
    ap.add_argument("--output-root",  default=OUTPUT_ROOT)
    ap.add_argument("--val-fraction",  type=float, default=VAL_FRACTION)
    ap.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    ap.add_argument("--seed",          type=int,   default=RANDOM_SEED)
    ap.add_argument("--mode", choices=["copy", "symlink", "manifest-only"],
                    default="copy",
                    help="copy: copia i .h5; symlink: link simbolici; "
                         "manifest-only: salva solo il manifest JSON.")
    args = ap.parse_args()

    print(f"Enumerazione file in {args.dataset_root} ...")
    pairs = enumerate_pairs(args.dataset_root)
    print(f"  {len(pairs)} coppie img/mask trovate.")

    print("Classificazione pos/neg (lettura masks) ...")
    positive_indices, _negative_indices = classify_pos_neg(pairs)

    train_idx, val_idx, test_idx, stats = build_split(
        n_total=len(pairs),
        positive_indices=positive_indices,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        random_seed=args.seed,
    )

    os.makedirs(args.output_root, exist_ok=True)
    print(f"Materializzazione ({args.mode}) in {args.output_root} ...")
    manifest_files = {
        "train": materialize("train", train_idx, pairs, args.output_root, args.mode),
        "val":   materialize("val",   val_idx,   pairs, args.output_root, args.mode),
        "test":  materialize("test",  test_idx,  pairs, args.output_root, args.mode),
    }

    manifest = {
        "dataset_root":  os.path.abspath(args.dataset_root),
        "output_root":   os.path.abspath(args.output_root),
        "img_dir":       IMG_DIR,
        "mask_dir":      MASK_DIR,
        "val_fraction":  args.val_fraction,
        "test_fraction": args.test_fraction,
        "random_seed":   args.seed,
        "mode":          args.mode,
        "n_total":       len(pairs),
        "stats":         stats,
        "indices": {
            "train": train_idx,
            "val":   val_idx,
            "test":  test_idx,
        },
        "files": manifest_files,
    }

    manifest_path = os.path.join(args.output_root, "split_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest salvato in: {manifest_path}")
    print("Fatto.")


if __name__ == "__main__":
    main()