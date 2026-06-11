"""
train_sam2tiny_baseline_l4s.py
===============================
SAM2-Tiny baseline PURO su Landslide4Sense.

Architettura:
  - SAM2-Tiny invariato: riceve RGB [B,3,H,W] (bande B4,B3,B2)
  - Nessun modulo laterale, nessun adapter, nessun patch embed modificato
  - Trainabili: SOLO mask_decoder
  - Frozen   : tutto il vision encoder

Confronto diretto con train_sam2tiny_dualstream_l4s.py:
  - Stesso split 60/20/20  (train=solo pos, val=solo pos, test=stratificato)
  - Stessa augmentation offline 3x (ElasticTransform + flips)
  - Stesse metriche (iou_fg, iou_bg, miou, f1, prec, rec)
  - Stesso RANDOM_SEED

Uso:
    python train_sam2tiny_baseline_l4s.py
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import glob
import h5py
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from transformers import Sam2Model
import segmentation_models_pytorch as smp


# ============================================================
# CONFIGURAZIONE  (identica al dualstream per confronto pulito)
# ============================================================

HF_PATH           = "facebook/sam2-hiera-tiny"
BATCH_SIZE        = 16
TARGET_SIZE       = 1024
DATASET_ROOT      = r"/datadrive/landslide/SAM3/SAM2/LandSlide4Sense"
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS            = 100
PATIENCE          = 15
NEG_FRACTION      = 0.0
VAL_FRACTION      = 0.20
TEST_FRACTION     = 0.20
RANDOM_SEED       = 42
GRAD_ACCUM_TARGET = 8

torch.backends.cudnn.benchmark = True


# ============================================================
# 1. DATASET
# ============================================================

class LandslideSAMDataset(Dataset):
    IMG_DIR  = "TrainData/img"
    MASK_DIR = "TrainData/mask"

    def __init__(self, root_dir):
        img_dir  = os.path.join(root_dir, self.IMG_DIR)
        mask_dir = os.path.join(root_dir, self.MASK_DIR)

        all_images = sorted(glob.glob(os.path.join(img_dir, "*.h5")))
        if not all_images:
            raise FileNotFoundError(f"Nessun .h5 in {img_dir}")

        print(f"Caricamento TrainData in RAM ({len(all_images)} file)...")
        self.data             = []
        self.positive_indices = []
        self.negative_indices = []

        for img_path in tqdm(all_images, desc="Loading TrainData"):
            mask_name = os.path.basename(img_path).replace("image_", "mask_")
            mask_path = os.path.join(mask_dir, mask_name)
            with h5py.File(img_path, "r") as f:
                image = f["img"][:].astype(np.float32)
            with h5py.File(mask_path, "r") as f:
                mask = f["mask"][:].astype(np.float32)
            idx = len(self.data)
            self.data.append((image, mask))
            if mask.sum() > 0:
                self.positive_indices.append(idx)
            else:
                self.negative_indices.append(idx)

        print(f"Caricati {len(self.data)} campioni: "
              f"{len(self.positive_indices)} positivi, "
              f"{len(self.negative_indices)} negativi")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, mask = self.data[idx]
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask  = torch.from_numpy(mask).long()

        indices = torch.where(mask > 0)
        if len(indices[0]) > 0:
            box = torch.tensor([
                indices[1].min().float(), indices[0].min().float(),
                indices[1].max().float(), indices[0].max().float(),
            ])
        else:
            box = torch.tensor([0., 0., 10., 10.])
        return {"image": image, "mask": mask, "boxes": box}

    def add_augmented_samples(self, indices, transform, n_copies, rng_seed=0):
        import random
        random.seed(rng_seed); np.random.seed(rng_seed)
        n_added = 0
        for orig_idx in tqdm(indices, desc=f"Augmentation offline ({n_copies}x)"):
            image_np, mask_np = self.data[orig_idx]
            for _ in range(n_copies):
                aug = transform(image=image_np, mask=mask_np)
                new_idx = len(self.data)
                self.data.append((aug["image"], aug["mask"]))
                self.positive_indices.append(new_idx)
                n_added += 1
        print(f"  Aggiunti {n_added} campioni augmentati "
              f"(da {len(indices)} positivi x {n_copies} copie)")


# ============================================================
# 2. SPLIT 60/20/20
# ============================================================

def build_train_val_test_subsets(
    full_ds, neg_fraction, val_fraction, test_fraction, random_seed
):
    """
    Train : solo positivi  (coerente con il task)
    Val   : solo positivi  (guida scheduler in modo pulito)
    Test  : stratificato   (valutazione finale realistica)
    """
    rng      = torch.Generator().manual_seed(random_seed)
    all_idx  = list(range(len(full_ds)))
    perm     = torch.randperm(len(all_idx), generator=rng).tolist()
    shuffled = [all_idx[i] for i in perm]

    pos_set = set(full_ds.positive_indices)

    n_test       = int(len(shuffled) * test_fraction)
    test_indices = shuffled[:n_test]
    remaining_80 = shuffled[n_test:]

    n_val_pool  = int(len(remaining_80) * (val_fraction / (1.0 - test_fraction)))
    val_pool    = remaining_80[:n_val_pool]
    train_pool  = remaining_80[n_val_pool:]

    val_indices   = [i for i in val_pool   if i in pos_set]
    pos_train     = [i for i in train_pool if i in pos_set]
    neg_train     = [i for i in train_pool if i not in pos_set]

    n_neg         = min(int(len(pos_train) * neg_fraction), len(neg_train))
    neg_perm      = torch.randperm(len(neg_train), generator=rng).tolist()
    neg_selected  = [neg_train[i] for i in neg_perm[:n_neg]]

    train_indices = pos_train + neg_selected
    train_perm    = torch.randperm(len(train_indices), generator=rng).tolist()
    train_indices = [train_indices[i] for i in train_perm]

    def _count(idxs):
        n_p = sum(1 for i in idxs if i in pos_set)
        return n_p, len(idxs) - n_p

    total = len(full_ds)
    n_pos_tr, n_neg_tr = _count(train_indices)
    n_pos_vl, n_neg_vl = _count(val_indices)
    n_pos_te, n_neg_te = _count(test_indices)

    print(
        f"\nSplit 60/20/20:"
        f"\n  Totale : {total:5d}  ({len(full_ds.positive_indices)} pos + {len(full_ds.negative_indices)} neg)"
        f"\n  Train  : {len(train_indices):5d}  ({n_pos_tr} pos + {n_neg_tr} neg)  <- solo positivi"
        f"\n  Val    : {len(val_indices):5d}  ({n_pos_vl} pos + {n_neg_vl} neg)  <- solo positivi"
        f"\n  Test   : {len(test_indices):5d}  ({n_pos_te} pos + {n_neg_te} neg)  <- stratificato\n"
    )

    return (
        Subset(full_ds, train_indices),
        Subset(full_ds, val_indices),
        Subset(full_ds, test_indices),
    )


# ============================================================
# 3. MODELLO BASELINE — SAM2-Tiny puro, solo RGB, solo mask_decoder
# ============================================================

class SAM2TinyBaseline(nn.Module):
    """
    SAM2-Tiny invariato.
    Input : RGB [B, 3, H, W]  (bande B4, B3, B2 di Landslide4Sense)
    Trainabili: solo mask_decoder
    Frozen    : tutto il vision encoder
    """

    def __init__(self, hf_path: str = HF_PATH):
        super().__init__()
        print(f"Caricamento SAM2: {hf_path}")
        self.backbone = Sam2Model.from_pretrained(hf_path)
        self._freeze()
        self._print_trainable()

    def _freeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.mask_decoder.parameters():
            p.requires_grad = True

    def _print_trainable(self):
        tr  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in self.parameters())
        print(f"  Parametri trainabili: {tr:,} / {tot:,} ({100*tr/tot:.1f}%)")

    def forward(self, x: torch.Tensor, boxes: torch.Tensor):
        # Estrai solo le bande RGB (B4=idx3, B3=idx2, B2=idx1)
        rgb = x[:, [3, 2, 1], :, :]

        if boxes.ndim == 2:
            boxes = boxes.unsqueeze(1)

        out = self.backbone(
            pixel_values=rgb,
            input_boxes=boxes,
            multimask_output=False,
        )

        scores = None
        for attr in ("iou_scores", "iou_predictions", "pred_ious"):
            val = getattr(out, attr, None)
            if val is not None:
                scores = val; break

        return out.pred_masks, scores


# ============================================================
# 4. UTILS & LOSS
# ============================================================

dice_loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)


def get_best_mask(all_preds: torch.Tensor, scores) -> torch.Tensor:
    if all_preds.ndim == 5:
        all_preds = all_preds.squeeze(1)
    if scores is not None:
        if scores.ndim == 3:
            scores = scores.squeeze(1)
        best = torch.argmax(scores, dim=-1)
        bidx = torch.arange(all_preds.shape[0], device=all_preds.device)
        return all_preds[bidx, best].unsqueeze(1).contiguous()
    return all_preds[:, 0:1].contiguous()


def criterion(y_pred, y_true, lambda_dice=0.3):
    y_pred    = y_pred.contiguous()
    y_true_rs = F.interpolate(y_true.unsqueeze(1).float(),
                              size=y_pred.shape[2:], mode="nearest")
    return (F.binary_cross_entropy_with_logits(y_pred, y_true_rs)
            + lambda_dice * dice_loss_fn(y_pred, y_true_rs))


def calculate_metrics(preds, masks):
    """Returns: iou_fg, iou_bg, miou, f1, prec, rec"""
    if preds.shape[-1] != masks.shape[-1]:
        preds = F.interpolate(preds, size=masks.shape[-2:], mode="bilinear")
    pr  = (torch.sigmoid(preds) > 0.5).float()
    gt  = masks.unsqueeze(1).float()
    eps = 1e-7

    tp_fg = (pr       * gt      ).sum(dim=(2, 3))
    fp_fg = (pr       * (1 - gt)).sum(dim=(2, 3))
    fn_fg = ((1 - pr) * gt      ).sum(dim=(2, 3))
    iou_fg = ((tp_fg + eps) / (tp_fg + fp_fg + fn_fg + eps)).mean().item()
    prec   = ((tp_fg + eps) / (tp_fg + fp_fg          + eps)).mean().item()
    rec    = ((tp_fg + eps) / (tp_fg +           fn_fg + eps)).mean().item()
    f1     = (2 * prec * rec) / (prec + rec + eps)

    pr_bg = 1.0 - pr;  gt_bg = 1.0 - gt
    tp_bg = (pr_bg * gt_bg).sum(dim=(2, 3))
    fp_bg = (pr_bg * (1 - gt_bg)).sum(dim=(2, 3))
    fn_bg = ((1 - pr_bg) * gt_bg).sum(dim=(2, 3))
    iou_bg = ((tp_bg + eps) / (tp_bg + fp_bg + fn_bg + eps)).mean().item()

    return iou_fg, iou_bg, (iou_fg + iou_bg) / 2.0, f1, prec, rec


def scale_boxes(boxes, orig_h, orig_w, target, margin=3):
    sy, sx = target / orig_h, target / orig_w
    boxes[:, 0] = torch.clamp(boxes[:, 0] * sx - margin, min=0)
    boxes[:, 1] = torch.clamp(boxes[:, 1] * sy - margin, min=0)
    boxes[:, 2] = torch.clamp(boxes[:, 2] * sx + margin, max=target - 1)
    boxes[:, 3] = torch.clamp(boxes[:, 3] * sy + margin, max=target - 1)
    return boxes


def save_checkpoint(model, optimizer, scheduler, epoch, best_iou, path):
    torch.save({
        "epoch": epoch, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_iou": best_iou,
    }, path)


def load_checkpoint(model, optimizer, scheduler, path):
    if not os.path.exists(path):
        print(f"Nessun checkpoint in '{path}'. Avvio da zero.")
        return 0, 0.0
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    except Exception:
        pass
    print(f"Checkpoint caricato — Epoca {ckpt['epoch']}, Best IoU: {ckpt['best_iou']:.4f}")
    return ckpt["epoch"] + 1, ckpt["best_iou"]


# ============================================================
# 5. WRAPPER SUBSET CON TRASFORMAZIONI
# ============================================================

class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset; self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        sample = self.subset[idx]
        if self.transform is None:
            return sample
        img_np  = sample["image"].permute(1, 2, 0).numpy()
        mask_np = sample["mask"].numpy().astype(np.float32)
        aug     = self.transform(image=img_np, mask=mask_np)
        return {"image": aug["image"], "mask": aug["mask"].long(), "boxes": sample["boxes"]}


# ============================================================
# 6. TRAINING LOOP
# ============================================================

def train():
    ckpt_path    = "result/best_sam2tiny_baseline_l4s.pth"
    history_file = "result/history_sam2tiny_baseline_l4s.json"

    grad_accum_steps = max(1, GRAD_ACCUM_TARGET // BATCH_SIZE)
    print(f"\nBatch fisico: {BATCH_SIZE} | Grad accum: {grad_accum_steps} | "
          f"Batch effettivo: {BATCH_SIZE * grad_accum_steps}")

    # Nessuna augmentation — test baseline pulito
    train_trans = A.Compose([ToTensorV2()])
    val_trans   = A.Compose([ToTensorV2()])

    # Dataset e split
    full_ds = LandslideSAMDataset(DATASET_ROOT)

    train_subset, val_subset, test_subset = build_train_val_test_subsets(
        full_ds, neg_fraction=NEG_FRACTION, val_fraction=VAL_FRACTION,
        test_fraction=TEST_FRACTION, random_seed=RANDOM_SEED,
    )



    train_ds = TransformSubset(train_subset, train_trans)
    val_ds   = TransformSubset(val_subset,   val_trans)
    test_ds  = TransformSubset(test_subset,  val_trans)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = SAM2TinyBaseline(hf_path=HF_PATH).to(DEVICE)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=5)
    scaler = torch.amp.GradScaler("cuda")

    start_epoch, best_iou = load_checkpoint(model, optimizer, scheduler, ckpt_path)

    history = []
    if os.path.exists(history_file):
        with open(history_file) as f:
            history = json.load(f)

    print(f"\nTraining SAM2-Tiny Baseline | Epoche {start_epoch+1}/{EPOCHS} | "
          f"Batch {BATCH_SIZE} | Device {DEVICE}\n")
    patience_counter = 0

    for epoch in range(start_epoch, EPOCHS):

        # TRAIN
        model.train()
        t_loss = t_iou = t_iou_bg = t_miou = t_f1 = t_prec = t_rec = 0.0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS} [TRAIN]")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            imgs  = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)
            boxes = batch["boxes"].to(DEVICE)

            orig_h, orig_w = imgs.shape[-2], imgs.shape[-1]
            imgs  = F.interpolate(imgs, (TARGET_SIZE, TARGET_SIZE),
                                  mode="bilinear", align_corners=False)
            masks = (F.interpolate(masks.unsqueeze(1).float(),
                                   (TARGET_SIZE, TARGET_SIZE), mode="nearest")
                     .squeeze(1).long())
            boxes = scale_boxes(boxes, orig_h, orig_w, TARGET_SIZE)

            with torch.amp.autocast("cuda"):
                all_preds, scores = model(imgs, boxes)
                preds       = get_best_mask(all_preds, scores)
                loss        = criterion(preds, masks)
                loss_scaled = loss / grad_accum_steps

            scaler.scale(loss_scaled).backward()

            if (step + 1) % grad_accum_steps == 0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

            iou_fg, iou_bg, miou, f1, prec, rec = calculate_metrics(preds.detach(), masks)
            t_loss += loss.item(); t_iou += iou_fg; t_iou_bg += iou_bg
            t_miou += miou;        t_f1  += f1;     t_prec   += prec; t_rec += rec
            pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou_fg:.4f}",
                             miou=f"{miou:.4f}", f1=f"{f1:.4f}")

        if (step + 1) % grad_accum_steps != 0:
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

        # VAL
        model.eval()
        v_loss = v_iou = v_iou_bg = v_miou = v_f1 = v_prec = v_rec = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Ep {epoch+1}/{EPOCHS} [VAL]  "):
                imgs  = batch["image"].to(DEVICE)
                masks = batch["mask"].to(DEVICE)
                boxes = batch["boxes"].to(DEVICE)

                orig_h, orig_w = imgs.shape[-2], imgs.shape[-1]
                imgs  = F.interpolate(imgs, (TARGET_SIZE, TARGET_SIZE),
                                      mode="bilinear", align_corners=False)
                masks = (F.interpolate(masks.unsqueeze(1).float(),
                                       (TARGET_SIZE, TARGET_SIZE), mode="nearest")
                         .squeeze(1).long())
                boxes = scale_boxes(boxes, orig_h, orig_w, TARGET_SIZE)

                with torch.amp.autocast("cuda"):
                    all_preds, scores = model(imgs, boxes)
                    preds   = get_best_mask(all_preds, scores)
                    v_loss += criterion(preds, masks).item()

                iou_fg, iou_bg, miou, f1, prec, rec = calculate_metrics(preds, masks)
                v_iou += iou_fg; v_iou_bg += iou_bg; v_miou += miou
                v_f1  += f1;     v_prec   += prec;   v_rec  += rec

        ntr, nvl = len(train_loader), len(val_loader)
        avg = {
            "epoch":           epoch + 1,
            "train_loss":      t_loss   / ntr, "train_iou":       t_iou    / ntr,
            "train_iou_bg":    t_iou_bg / ntr, "train_miou":      t_miou   / ntr,
            "train_f1":        t_f1     / ntr, "train_precision":  t_prec   / ntr,
            "train_recall":    t_rec    / ntr,
            "val_loss":        v_loss   / nvl, "val_iou":         v_iou    / nvl,
            "val_iou_bg":      v_iou_bg / nvl, "val_miou":        v_miou   / nvl,
            "val_f1":          v_f1     / nvl, "val_precision":   v_prec   / nvl,
            "val_recall":      v_rec    / nvl,
        }

        scheduler.step(avg["val_loss"])

        print(
            f"EPOCA {epoch+1:3d}\n"
            f"  TRAIN | Loss:{avg['train_loss']:.4f}  IoU:{avg['train_iou']:.4f}  "
            f"IoU_bg:{avg['train_iou_bg']:.4f}  mIoU:{avg['train_miou']:.4f}  "
            f"F1:{avg['train_f1']:.4f}  Prec:{avg['train_precision']:.4f}  Rec:{avg['train_recall']:.4f}\n"
            f"  VAL   | Loss:{avg['val_loss']:.4f}  IoU:{avg['val_iou']:.4f}  "
            f"IoU_bg:{avg['val_iou_bg']:.4f}  mIoU:{avg['val_miou']:.4f}  "
            f"F1:{avg['val_f1']:.4f}  Prec:{avg['val_precision']:.4f}  Rec:{avg['val_recall']:.4f}"
        )

        history.append(avg)
        with open(history_file, "w") as f:
            json.dump(history, f, indent=4)

        if avg["val_iou"] > best_iou:
            best_iou = avg["val_iou"]
            save_checkpoint(model, optimizer, scheduler, epoch, best_iou, ckpt_path)
            print(f"  Record IoU: {best_iou:.4f}  mIoU:{avg['val_miou']:.4f}  "
                  f"F1:{avg['val_f1']:.4f} — checkpoint salvato")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"\nEARLY STOPPING — Miglior IoU val: {best_iou:.4f}")
                break

    print(f"\nTraining completato! Miglior IoU val: {best_iou:.4f}")

    # VALUTAZIONE FINALE SUL TEST SET
    print("\n" + "="*60)
    print("  VALUTAZIONE FINALE SUL TEST SET")
    print("="*60)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    te_loss = te_iou = te_iou_bg = te_miou = te_f1 = te_prec = te_rec = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="TEST"):
            imgs  = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)
            boxes = batch["boxes"].to(DEVICE)

            orig_h, orig_w = imgs.shape[-2], imgs.shape[-1]
            imgs  = F.interpolate(imgs, (TARGET_SIZE, TARGET_SIZE),
                                  mode="bilinear", align_corners=False)
            masks = (F.interpolate(masks.unsqueeze(1).float(),
                                   (TARGET_SIZE, TARGET_SIZE), mode="nearest")
                     .squeeze(1).long())
            boxes = scale_boxes(boxes, orig_h, orig_w, TARGET_SIZE)

            with torch.amp.autocast("cuda"):
                all_preds, scores = model(imgs, boxes)
                preds   = get_best_mask(all_preds, scores)
                te_loss += criterion(preds, masks).item()

            iou_fg, iou_bg, miou, f1, prec, rec = calculate_metrics(preds, masks)
            te_iou += iou_fg; te_iou_bg += iou_bg; te_miou += miou
            te_f1  += f1;     te_prec   += prec;   te_rec  += rec

    nte = len(test_loader)
    test_results = {
        "test_loss": te_loss/nte, "test_iou": te_iou/nte,
        "test_iou_bg": te_iou_bg/nte, "test_miou": te_miou/nte,
        "test_f1": te_f1/nte, "test_precision": te_prec/nte,
        "test_recall": te_rec/nte,
    }
    print(f"  Loss  : {test_results['test_loss']:.4f}\n"
          f"  IoU   : {test_results['test_iou']:.4f}\n"
          f"  IoU_bg: {test_results['test_iou_bg']:.4f}\n"
          f"  mIoU  : {test_results['test_miou']:.4f}\n"
          f"  F1    : {test_results['test_f1']:.4f}\n"
          f"  Prec  : {test_results['test_precision']:.4f}\n"
          f"  Rec   : {test_results['test_recall']:.4f}")

    history.append({"test_results": test_results})
    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Risultati salvati in: {history_file}")


# ============================================================
# 7. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train()