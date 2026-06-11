"""
train_sam2tiny_adapter_h5.py
=============================
SAM2-Tiny con Adapter + patch embed espanso a 12ch + DSDF Early Guidance (2ch terrain)
su Landslide4Sense con dataset .h5.

Differenze rispetto all'originale:
  - Dataset: file .h5 da TrainData/ (tutto in RAM) invece di .pt con split fissi
  - Split 60/20/20 stratificato (identico a SAM-LoRA, D2FLS, SAM2-Base-Plus)
  - Rimosso Hydra
  - pos_weight dinamico nella loss
  - calculate_metrics micro-average (evita gonfiamento mIoU su patch tutti-bg)
  - Test set finale

Input:
  optical (12ch): bande 0-11 Sentinel-2
  terrain (2ch) : banda 12 (Slope) + banda 13 (DEM)

Dipendenze:
    pip install torch transformers segmentation-models-pytorch h5py tqdm albumentations
"""

import os
import glob
import h5py
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import Sam2Model
import segmentation_models_pytorch as smp

# ============================================================
# CONFIGURAZIONE
# ============================================================

DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS          = 100
PATIENCE        = 15
BATCH_SIZE      = 8
TARGET_SIZE     = 1024

DATASET_ROOT    = r"/datadrive/landslide/SAM3/SAM2/LandSlide4Sense"
SAM_CKPT_PATH   = "facebook/sam2-hiera-tiny"
CHECKPOINT_NAME = "best_sam2tiny_adapter_h5.pth"
HISTORY_FILE    = "history_sam2tiny_adapter_h5.json"

VAL_FRACTION    = 0.20
TEST_FRACTION   = 0.20
RANDOM_SEED     = 42
AUG_COPIES      = 2          # copie augmentation offline per ogni positivo del train

# Bande L4S: 0-11=Sentinel-2, 12=Slope, 13=DEM
OPTICAL_BANDS   = list(range(12))   # bande 0-11
TERRAIN_BANDS   = [12, 13]          # Slope + DEM (input a DSDFEarlyGuidance)

torch.backends.cudnn.benchmark = True


# ============================================================
# 1. DATASET  (.h5 + split 60/20/20)
# ============================================================

class LandslideH5Dataset(Dataset):
    """
    Carica tutti i file .h5 di TrainData/ in RAM.
        <root>/TrainData/img/image_*.h5  -> key "img"  (128,128,14) HWC
        <root>/TrainData/mask/mask_*.h5  -> key "mask" (128,128)

    Ritorna dict:
        image : Tensor (14, H, W)  — tutti i 14 canali L4S
        mask  : Tensor (H, W)      — long 0/1
        boxes : Tensor (4,)        — bounding box GT landslide
    """
    IMG_DIR  = "TrainData/img"
    MASK_DIR = "TrainData/mask"

    def __init__(self, root_dir: str):
        img_dir  = os.path.join(root_dir, self.IMG_DIR)
        mask_dir = os.path.join(root_dir, self.MASK_DIR)

        all_images = sorted(glob.glob(os.path.join(img_dir, "*.h5")))
        if not all_images:
            raise FileNotFoundError(f"Nessun .h5 trovato in {img_dir}")

        print(f"Caricamento TrainData in RAM ({len(all_images)} file)...")
        self.data             = []
        self.positive_indices = []
        self.negative_indices = []

        for img_path in tqdm(all_images, desc="Loading TrainData"):
            mask_name = os.path.basename(img_path).replace("image_", "mask_")
            mask_path = os.path.join(mask_dir, mask_name)

            with h5py.File(img_path,  "r") as f:
                image = f["img"][:].astype(np.float32)   # (H,W,14)
            with h5py.File(mask_path, "r") as f:
                mask  = f["mask"][:].astype(np.int64)    # (H,W)

            idx = len(self.data)
            self.data.append((image, mask))           # teniamo tutto HWC (H,W,14)
            (self.positive_indices if mask.sum() > 0
             else self.negative_indices).append(idx)

        n_pos, n_neg = len(self.positive_indices), len(self.negative_indices)
        print(f"Caricati {len(self.data)} campioni: {n_pos} positivi, {n_neg} negativi")

    def add_augmented_samples(self, indices: list, transform, n_copies: int, rng_seed: int = 0):
        """
        Genera n_copies versioni augmentate offline per ogni indice in indices
        e le aggiunge a self.data / self.positive_indices.
        La transform NON deve includere ToTensorV2 (produce numpy HWC).
        """
        import random
        random.seed(rng_seed)
        np.random.seed(rng_seed)
        n_added = 0
        for orig_idx in tqdm(indices, desc=f"Augmentation offline ({n_copies}x)"):
            image_np, mask_np = self.data[orig_idx]   # (H,W,14), (H,W)
            for _ in range(n_copies):
                aug = transform(image=image_np, mask=mask_np.astype(np.float32))
                new_idx = len(self.data)
                self.data.append((aug["image"], aug["mask"]))
                self.positive_indices.append(new_idx)
                n_added += 1
        print(f"  Aggiunti {n_added} campioni augmentati "
              f"(da {len(indices)} positivi x {n_copies} copie)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_np, mask_np = self.data[idx]                        # (H,W,14), (H,W)
        img_t  = torch.from_numpy(image_np).permute(2, 0, 1)     # (14,H,W)
        mask_t = torch.from_numpy(mask_np).long()

        # Bounding box GT
        indices = torch.where(mask_t > 0)
        if len(indices[0]) > 0:
            box = torch.tensor([
                indices[1].min().float(), indices[0].min().float(),
                indices[1].max().float(), indices[0].max().float(),
            ])
        else:
            box = torch.tensor([0., 0., 10., 10.])

        return {"image": img_t, "mask": mask_t, "boxes": box}



# ============================================================
# 2b. WRAPPER AUGMENTATION
# ============================================================

class TransformSubset(Dataset):
    """Applica una transform albumentations a un Subset esistente."""
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        sample = self.subset[idx]
        if self.transform is None:
            return sample
        img_np  = sample["image"].permute(1, 2, 0).numpy()   # (H,W,14)
        mask_np = sample["mask"].numpy().astype(np.float32)
        aug     = self.transform(image=img_np, mask=mask_np)
        return {
            "image": aug["image"],
            "mask":  aug["mask"].long(),
            "boxes": sample["boxes"],
        }

# ============================================================
# 2. SPLIT 60/20/20
# ============================================================

def build_train_val_test_subsets(full_ds, val_fraction, test_fraction, random_seed):
    rng      = torch.Generator().manual_seed(random_seed)
    all_idx  = list(range(len(full_ds)))
    perm     = torch.randperm(len(all_idx), generator=rng).tolist()
    shuffled = [all_idx[i] for i in perm]
    pos_set  = set(full_ds.positive_indices)

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

    print(
        f"\nSplit 60/20/20:"
        f"\n  Totale : {len(full_ds):5d}  "
        f"({len(full_ds.positive_indices)} pos + {len(full_ds.negative_indices)} neg)"
        f"\n  Train  : {len(train_indices):5d}  ({n_pos_tr} pos + {n_neg_tr} neg)"
        f"\n  Val    : {len(val_indices):5d}  ({n_pos_vl} pos + {n_neg_vl} neg)"
        f"\n  Test   : {len(test_indices):5d}  ({n_pos_te} pos + {n_neg_te} neg)"
    )

    ls_frac    = 0.35
    pixel_ls   = n_pos_tr * ls_frac
    pixel_bg   = n_neg_tr + n_pos_tr * (1.0 - ls_frac)
    pos_weight = pixel_bg / max(pixel_ls, 1.0)
    print(f"  pos_weight: {pos_weight:.2f}\n")

    return (
        Subset(full_ds, train_indices),
        Subset(full_ds, val_indices),
        Subset(full_ds, test_indices),
        pos_weight,
    )


# ============================================================
# 3. MODELLO
# ============================================================

class Adapter(nn.Module):
    """Bottleneck adapter zero-delta: x + W2·ReLU(W1·x)."""
    def __init__(self, dim: int, adapter_dim: int = 64):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(dim, adapter_dim),
            nn.ReLU(inplace=True),
            nn.Linear(adapter_dim, dim),
        )
        nn.init.zeros_(self.adapter[2].weight)
        nn.init.zeros_(self.adapter[2].bias)

    def forward(self, x):
        return x + self.adapter(x)


class DSDFEarlyGuidance(nn.Module):
    """
    Guidance map da Slope+DEM (2ch) → G ∈ [0,1].
    Applicata come gain spaziale sull'ottico: optical_mod = optical ⊙ G
    """
    def __init__(self):
        super().__init__()
        self.guidance_net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, terrain: torch.Tensor) -> torch.Tensor:
        return self.guidance_net(terrain)   # (B,1,H,W)


class SAM2TinyAdapterModel(nn.Module):
    """
    SAM2-Tiny con:
      - patch embed espanso 3 → 12 canali (Sentinel-2 B1-B12)
        pesi RGB copiati, extra inizializzati con rumore piccolo
      - DSDF Early Guidance: terrain (Slope+DEM) modula l'ottico
      - Adapter bottleneck su tutti i blocchi attention
      - mask_decoder scongelato

    Trainabili: patch_proj + adapters + early_guidance + mask_decoder
    Frozen    : tutto il resto
    """
    def __init__(self, adapter_dim: int = 64):
        super().__init__()
        print(f"Caricamento SAM2-Tiny: {SAM_CKPT_PATH}")
        self.model          = Sam2Model.from_pretrained(SAM_CKPT_PATH)
        self.early_guidance = DSDFEarlyGuidance()
        v_encoder           = self.model.vision_encoder

        # ── 1. Patch embed 3 → 12 canali ─────────────────────────
        self.patch_proj = None
        for name, module in v_encoder.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                new_conv = nn.Conv2d(
                    12, module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    bias=(module.bias is not None),
                )
                with torch.no_grad():
                    new_conv.weight[:, :3] = module.weight.clone()
                    nn.init.normal_(new_conv.weight[:, 3:], mean=0.0, std=0.001)
                    if module.bias is not None:
                        new_conv.bias.copy_(module.bias)
                parts  = name.split(".")
                parent = (v_encoder.get_submodule(".".join(parts[:-1]))
                          if len(parts) > 1 else v_encoder)
                setattr(parent, parts[-1], new_conv)
                self.patch_proj = new_conv
                print(f"  Patch embed espanso: {name}  3 → 12 canali")
                break

        if self.patch_proj is None:
            raise RuntimeError("Patch embed Conv2d(in=3) non trovato.")

        # ── 2. Adapter su tutti i blocchi attention ───────────────
        self.adapters = nn.ModuleList()
        for name, module in v_encoder.named_modules():
            if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear):
                dim = (module.proj.out_features
                       if hasattr(module, "proj") else module.qkv.in_features)
                adapter = Adapter(dim, adapter_dim)
                self.adapters.append(adapter)
                module.register_forward_hook(self._make_hook(adapter))
            elif hasattr(module, "q_proj") and isinstance(module.q_proj, nn.Linear):
                dim = (module.out_proj.out_features
                       if hasattr(module, "out_proj") else module.q_proj.in_features)
                adapter = Adapter(dim, adapter_dim)
                self.adapters.append(adapter)
                module.register_forward_hook(self._make_hook(adapter))

        print(f"  Adapter iniettati in {len(self.adapters)} blocchi attention")
        self._freeze()
        self._print_trainable()

    @staticmethod
    def _make_hook(adapter: Adapter):
        def hook(module, args, output):
            if isinstance(output, tuple):
                return (adapter(output[0]),) + output[1:]
            return adapter(output)
        return hook

    def _freeze(self):
        for p in self.model.parameters():
            p.requires_grad = False
        # Scongela selettivamente
        for p in self.model.mask_decoder.parameters():
            p.requires_grad = True
        for p in self.patch_proj.parameters():
            p.requires_grad = True
        for adapter in self.adapters:
            for p in adapter.parameters():
                p.requires_grad = True
        for p in self.early_guidance.parameters():
            p.requires_grad = True

    def _print_trainable(self):
        tr  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in self.parameters())
        print(f"  Parametri trainabili: {tr:,} / {tot:,} ({100*tr/tot:.1f}%)")

    def forward(self, x: torch.Tensor, boxes: torch.Tensor):
        """
        x     : (B, 14, H, W)  — tutti i canali L4S
        boxes : (B, 4)
        """
        optical = x[:, :12]          # (B,12,H,W) Sentinel-2
        terrain = x[:, 12:]          # (B, 2,H,W) Slope + DEM

        # DSDF early guidance: modula l'ottico col terrain
        g       = self.early_guidance(terrain)   # (B,1,H,W)
        optical = optical * g                    # broadcasting su 12ch

        if boxes.ndim == 2:
            boxes = boxes.unsqueeze(1)

        outputs = self.model(
            pixel_values=optical,
            input_boxes=boxes,
            multimask_output=False,
        )

        scores = getattr(outputs, "iou_scores",      None)
        if scores is None:
            scores = getattr(outputs, "iou_predictions", None)
        if scores is None:
            scores = getattr(outputs, "pred_ious",       None)

        return outputs.pred_masks, scores


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


def criterion(y_pred: torch.Tensor, y_true: torch.Tensor,
              pos_weight: float = 1.0, lambda_dice: float = 0.3):
    y_pred    = y_pred.contiguous()
    y_true_rs = F.interpolate(y_true.unsqueeze(1).float(),
                              size=y_pred.shape[2:], mode="nearest")
    pw   = torch.tensor([pos_weight], device=y_pred.device)
    ce   = F.binary_cross_entropy_with_logits(y_pred, y_true_rs, pos_weight=pw)
    dice = dice_loss_fn(y_pred, y_true_rs)
    return ce + lambda_dice * dice


def calculate_metrics(preds: torch.Tensor, masks: torch.Tensor,
                      epsilon: float = 1e-7):
    """
    Macro-average per immagine — identica al DualStream.
    Returns: iou_fg, iou_bg, miou, f1, prec, rec
    """
    if preds.shape[-2:] != masks.shape[-2:]:
        preds = F.interpolate(preds, size=masks.shape[-2:],
                              mode="bilinear", align_corners=False)
    pr  = (torch.sigmoid(preds) > 0.5).float()
    gt  = masks.unsqueeze(1).float()

    tp_fg = (pr       * gt      ).sum(dim=(2, 3))
    fp_fg = (pr       * (1 - gt)).sum(dim=(2, 3))
    fn_fg = ((1 - pr) * gt      ).sum(dim=(2, 3))
    iou_fg = ((tp_fg + epsilon) / (tp_fg + fp_fg + fn_fg + epsilon)).mean().item()
    prec   = ((tp_fg + epsilon) / (tp_fg + fp_fg          + epsilon)).mean().item()
    rec    = ((tp_fg + epsilon) / (tp_fg +           fn_fg + epsilon)).mean().item()
    f1     = (2 * prec * rec) / (prec + rec + epsilon)

    pr_bg = 1.0 - pr; gt_bg = 1.0 - gt
    tp_bg = (pr_bg       * gt_bg      ).sum(dim=(2, 3))
    fp_bg = (pr_bg       * (1 - gt_bg)).sum(dim=(2, 3))
    fn_bg = ((1 - pr_bg) * gt_bg      ).sum(dim=(2, 3))
    iou_bg = ((tp_bg + epsilon) / (tp_bg + fp_bg + fn_bg + epsilon)).mean().item()

    return iou_fg, iou_bg, (iou_fg + iou_bg) / 2.0, f1, prec, rec


def scale_boxes(boxes: torch.Tensor,
                orig_h: int, orig_w: int,
                target: int, margin: int = 3) -> torch.Tensor:
    sx = target / orig_w
    sy = target / orig_h
    boxes[:, 0] = torch.clamp(boxes[:, 0] * sx - margin, min=0)
    boxes[:, 1] = torch.clamp(boxes[:, 1] * sy - margin, min=0)
    boxes[:, 2] = torch.clamp(boxes[:, 2] * sx + margin, max=target - 1)
    boxes[:, 3] = torch.clamp(boxes[:, 3] * sy + margin, max=target - 1)
    return boxes


def save_checkpoint(model, optimizer, scheduler, epoch, best_iou, path):
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_iou":             best_iou,
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
    print(f"Checkpoint caricato — Epoca {ckpt['epoch']}, "
          f"Best IoU: {ckpt['best_iou']:.4f}")
    return ckpt["epoch"] + 1, ckpt["best_iou"]


# ============================================================
# 5. TRAINING LOOP
# ============================================================

def train():
    full_ds = LandslideH5Dataset(DATASET_ROOT)
    train_subset, val_subset, test_subset, pos_weight = build_train_val_test_subsets(
        full_ds, VAL_FRACTION, TEST_FRACTION, RANDOM_SEED)

    # Augmentation offline (identica al DualStream: senza ToTensorV2, produce numpy HWC)
    offline_aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])

    # Augmentation on-the-fly (stesse trasformazioni + ToTensorV2)
    train_trans = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        ToTensorV2(),
    ])
    val_trans = A.Compose([ToTensorV2()])

    # Augmentation offline sui positivi del train
    train_pos_indices = [i for i in train_subset.indices
                         if i in set(full_ds.positive_indices)]
    full_ds.add_augmented_samples(
        indices=train_pos_indices, transform=offline_aug,
        n_copies=AUG_COPIES, rng_seed=RANDOM_SEED,
    )
    aug_start   = len(full_ds.data) - len(train_pos_indices) * AUG_COPIES
    aug_indices = list(range(aug_start, len(full_ds.data)))
    from torch.utils.data import Subset as _Subset
    train_subset = _Subset(full_ds, list(train_subset.indices) + aug_indices)
    print(f"  Train finale: {len(train_subset)} campioni "
          f"({len(train_pos_indices)} orig + {len(aug_indices)} aug)")

    train_ds = TransformSubset(train_subset, train_trans)
    val_ds   = TransformSubset(val_subset,   val_trans)
    test_ds  = TransformSubset(test_subset,  val_trans)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    model     = SAM2TinyAdapterModel(adapter_dim=64).to(DEVICE)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-5, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=5)
    scaler    = torch.amp.GradScaler("cuda")

    start_epoch, best_iou = load_checkpoint(
        model, optimizer, scheduler, CHECKPOINT_NAME)

    history          = []
    patience_counter = 0
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            history = json.load(f)

    print(f"\nTraining SAM2-Tiny Adapter | "
          f"Epoche {start_epoch+1}/{EPOCHS} | Batch {BATCH_SIZE} | {DEVICE}\n")

    for epoch in range(start_epoch, EPOCHS):

        # ── TRAIN ──────────────────────────────────────────────
        model.train()
        t_loss = t_iou = t_iou_bg = t_miou = t_f1 = t_prec = t_rec = 0.0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS} [TRAIN]")

        for batch in pbar:
            imgs  = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)
            boxes = batch["boxes"].to(DEVICE)

            orig_h, orig_w = imgs.shape[-2], imgs.shape[-1]
            imgs  = F.interpolate(imgs, (TARGET_SIZE, TARGET_SIZE),
                                  mode="bilinear", align_corners=False)
            masks = F.interpolate(masks.unsqueeze(1).float(),
                                  (TARGET_SIZE, TARGET_SIZE),
                                  mode="nearest").squeeze(1).long()
            boxes = scale_boxes(boxes, orig_h, orig_w, TARGET_SIZE)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                all_preds, scores = model(imgs, boxes)
                preds = get_best_mask(all_preds, scores)
                loss  = criterion(preds, masks, pos_weight=pos_weight)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            iou_fg, iou_bg, miou, f1, prec, rec = calculate_metrics(preds.detach(), masks)
            t_loss   += loss.item(); t_iou    += iou_fg; t_iou_bg += iou_bg
            t_miou   += miou;        t_f1     += f1
            t_prec   += prec;        t_rec    += rec
            pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou_fg:.4f}",
                             miou=f"{miou:.4f}", f1=f"{f1:.4f}")

        # ── VAL ────────────────────────────────────────────────
        model.eval()
        v_loss = v_iou = v_iou_bg = v_miou = v_f1 = v_prec = v_rec = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader,
                              desc=f"Ep {epoch+1}/{EPOCHS} [VAL]  "):
                imgs  = batch["image"].to(DEVICE)
                masks = batch["mask"].to(DEVICE)
                boxes = batch["boxes"].to(DEVICE)

                orig_h, orig_w = imgs.shape[-2], imgs.shape[-1]
                imgs  = F.interpolate(imgs, (TARGET_SIZE, TARGET_SIZE),
                                      mode="bilinear", align_corners=False)
                masks = F.interpolate(masks.unsqueeze(1).float(),
                                      (TARGET_SIZE, TARGET_SIZE),
                                      mode="nearest").squeeze(1).long()
                boxes = scale_boxes(boxes, orig_h, orig_w, TARGET_SIZE)

                with torch.amp.autocast("cuda"):
                    all_preds, scores = model(imgs, boxes)
                    preds  = get_best_mask(all_preds, scores)
                    v_loss += criterion(preds, masks,
                                        pos_weight=pos_weight).item()

                iou_fg, iou_bg, miou, f1, prec, rec = calculate_metrics(preds, masks)
                v_iou    += iou_fg; v_iou_bg += iou_bg; v_miou   += miou
                v_f1     += f1;     v_prec   += prec;   v_rec    += rec

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
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=4)

        if avg["val_iou"] > best_iou:
            best_iou = avg["val_iou"]
            save_checkpoint(model, optimizer, scheduler, epoch,
                            best_iou, CHECKPOINT_NAME)
            print(f"  Record IoU: {best_iou:.4f}  mIoU:{avg['val_miou']:.4f}  F1:{avg['val_f1']:.4f} — checkpoint salvato")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"\nEARLY STOPPING — Miglior IoU val: {best_iou:.4f}")
                break

    # ── TEST FINALE ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  VALUTAZIONE FINALE SUL TEST SET")
    print("=" * 60)

    ckpt = torch.load(CHECKPOINT_NAME, map_location=DEVICE)
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
            masks = F.interpolate(masks.unsqueeze(1).float(),
                                  (TARGET_SIZE, TARGET_SIZE),
                                  mode="nearest").squeeze(1).long()
            boxes = scale_boxes(boxes, orig_h, orig_w, TARGET_SIZE)

            with torch.amp.autocast("cuda"):
                all_preds, scores = model(imgs, boxes)
                preds   = get_best_mask(all_preds, scores)
                te_loss += criterion(preds, masks,
                                     pos_weight=pos_weight).item()

            iou_fg, iou_bg, miou, f1, prec, rec = calculate_metrics(preds, masks)
            te_iou    += iou_fg; te_iou_bg += iou_bg; te_miou   += miou
            te_f1     += f1;     te_prec   += prec;   te_rec    += rec

    nte = len(test_loader)
    test_results = {
        "test_loss":      te_loss   / nte, "test_iou":       te_iou    / nte,
        "test_iou_bg":    te_iou_bg / nte, "test_miou":      te_miou   / nte,
        "test_f1":        te_f1     / nte, "test_precision": te_prec   / nte,
        "test_recall":    te_rec    / nte,
    }
    print(
        f"  Loss  : {test_results['test_loss']:.4f}\n"
        f"  IoU   : {test_results['test_iou']:.4f}\n"
        f"  IoU_bg: {test_results['test_iou_bg']:.4f}\n"
        f"  mIoU  : {test_results['test_miou']:.4f}\n"
        f"  F1    : {test_results['test_f1']:.4f}\n"
        f"  Prec  : {test_results['test_precision']:.4f}\n"
        f"  Rec   : {test_results['test_recall']:.4f}"
    )
    history.append({"test_results": test_results})
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Risultati salvati in: {HISTORY_FILE}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train()