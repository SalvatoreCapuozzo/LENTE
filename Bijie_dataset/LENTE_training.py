"""
train_sam2tiny_adapter_bijie.py
================================
SAM2-Tiny con Adapter + DSDF Early Guidance (2ch terrain) sul Bijie Landslide Dataset.

Differenze rispetto alla versione Landslide4Sense (.h5):
  - Dataset: PNG (image RGB, mask, DEM) da Bijie-landslide-dataset
    https://gpcv.whu.edu.cn/data/Bijie_pages.html
    770 immagini landslide + 2003 non-landslide (TripleSat, dim. variabile)
  - Ottico a 3 canali RGB -> NESSUNA espansione del patch embed
    (si usa il Conv2d originale di SAM2, comunque scongelato per il dominio satellitare)
  - Terrain: il dataset fornisce solo il DEM; la Slope viene DERIVATA dal DEM
    con il gradiente (np.gradient), cosi' DSDFEarlyGuidance resta identica a 2ch
    [slope, dem] e il confronto con L4S e' architetturalmente pulito
  - Immagini di dimensione variabile -> resize a LOAD_SIZE in fase di caricamento
    (necessario per il batching); storage in uint8 per risparmiare RAM
  - I campioni non-landslide non hanno file mask -> maschera vuota
  - pos_weight calcolato dalle statistiche reali dei pixel del train set

Struttura attesa del dataset (dopo unzip di Bijie_landslide_dataset.zip):
    <root>/landslide/img/        df001.png ...   (RGB)
    <root>/landslide/mask/       df001_mask.png  (binaria)
    <root>/landslide/dem/        df001_dem.png
    <root>/non-landslide/img/    ...
    <root>/non-landslide/dem/    ...
  Il loader e' tollerante: cerca le sottocartelle tra piu' nomi comuni
  (img/image/images, mask/label, dem) e accoppia i file per "stem" comune,
  quindi adatta DATASET_ROOT e, se serve, i nomi in DIR_CANDIDATES.

Dipendenze:
    pip install torch transformers segmentation-models-pytorch tqdm albumentations pillow
"""

import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from PIL import Image
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
LOAD_SIZE       = 256        # dimensione comune a cui portare i ritagli variabili Bijie

DATASET_ROOT    = r"/datadrive/LENTE/Bijie_dataset/Bijie-landslide-dataset"
SAM_CKPT_PATH   = "facebook/sam2-hiera-tiny"
CHECKPOINT_NAME = "BEST_sam2tiny_adapter_bijie.pth"
HISTORY_FILE    = "HISTORY_sam2tiny_adapter_bijie.json"

VAL_FRACTION    = 0.20
TEST_FRACTION   = 0.20
RANDOM_SEED     = 42
AUG_COPIES      = 2          # copie augmentation offline per ogni positivo del train

torch.backends.cudnn.benchmark = True


# ============================================================
# 1. DATASET  (PNG Bijie + split 60/20/20)
# ============================================================

def _find_subdir(parent: str, candidates) -> str | None:
    for c in candidates:
        p = os.path.join(parent, c)
        if os.path.isdir(p):
            return p
    return None


def _stem(path: str) -> str:
    """Stem comune: rimuove estensione e suffissi _mask/_dem se presenti."""
    s = os.path.splitext(os.path.basename(path))[0]
    for suf in ("_mask", "_dem", "_label"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


class BijieDataset(Dataset):
    """
    Carica tutto il Bijie landslide dataset in RAM (uint8, resize a LOAD_SIZE).

    Ritorna dict:
        image : Tensor (5, H, W) float32 — [R, G, B, slope, dem] in [0,1]
                (slope derivata dal DEM via gradiente, normalizzata per-immagine)
        mask  : Tensor (H, W) long 0/1
        boxes : Tensor (4,) bounding box GT landslide
    """
    DIR_CANDIDATES = {
        "landslide":     ["landslide", "Landslide"],
        "non_landslide": ["non-landslide", "non_landslide", "Non-landslide", "nonlandslide"],
        "img":           ["img", "image", "images", "Image"],
        "mask":          ["mask", "masks", "label", "labels", "Mask"],
        "dem":           ["dem", "DEM", "dems"],
    }

    def __init__(self, root_dir: str):
        ls_dir  = _find_subdir(root_dir, self.DIR_CANDIDATES["landslide"])
        nls_dir = _find_subdir(root_dir, self.DIR_CANDIDATES["non_landslide"])
        if ls_dir is None:
            raise FileNotFoundError(
                f"Cartella 'landslide' non trovata in {root_dir}. "
                f"Contenuto: {os.listdir(root_dir)}")

        self.data             = []   # (img_u8 HW3, dem_u8 HW, mask_u8 HW)
        self.positive_indices = []
        self.negative_indices = []

        self._load_group(ls_dir,  has_mask=True)
        if nls_dir is not None:
            self._load_group(nls_dir, has_mask=False)
        else:
            print("ATTENZIONE: cartella non-landslide non trovata, uso solo i positivi.")

        n_pos, n_neg = len(self.positive_indices), len(self.negative_indices)
        print(f"Caricati {len(self.data)} campioni: {n_pos} positivi, {n_neg} negativi")

    def _load_group(self, group_dir: str, has_mask: bool):
        img_dir  = _find_subdir(group_dir, self.DIR_CANDIDATES["img"])
        dem_dir  = _find_subdir(group_dir, self.DIR_CANDIDATES["dem"])
        mask_dir = _find_subdir(group_dir, self.DIR_CANDIDATES["mask"]) if has_mask else None
        if img_dir is None or dem_dir is None:
            raise FileNotFoundError(
                f"Sottocartelle img/dem non trovate in {group_dir}. "
                f"Contenuto: {os.listdir(group_dir)}")
        if has_mask and mask_dir is None:
            raise FileNotFoundError(f"Cartella mask non trovata in {group_dir}")

        # indicizza dem (e mask) per stem
        dem_map  = {_stem(p): p for p in glob.glob(os.path.join(dem_dir, "*.png"))}
        mask_map = ({_stem(p): p for p in glob.glob(os.path.join(mask_dir, "*.png"))}
                    if has_mask else {})

        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        if not img_paths:
            raise FileNotFoundError(f"Nessun .png in {img_dir}")

        n_skip = 0
        desc = "Loading landslide" if has_mask else "Loading non-landslide"
        for img_path in tqdm(img_paths, desc=desc):
            st = _stem(img_path)
            dem_path = dem_map.get(st)
            if dem_path is None:
                n_skip += 1
                continue

            img = Image.open(img_path).convert("RGB").resize(
                (LOAD_SIZE, LOAD_SIZE), Image.BILINEAR)
            dem = Image.open(dem_path).convert("L").resize(
                (LOAD_SIZE, LOAD_SIZE), Image.BILINEAR)
            img_np = np.asarray(img, dtype=np.uint8)          # (H,W,3)
            dem_np = np.asarray(dem, dtype=np.uint8)          # (H,W)

            if has_mask:
                mask_path = mask_map.get(st)
                if mask_path is None:
                    n_skip += 1
                    continue
                m = Image.open(mask_path).convert("L").resize(
                    (LOAD_SIZE, LOAD_SIZE), Image.NEAREST)
                mask_np = (np.asarray(m) > 127).astype(np.uint8)  # (H,W) 0/1
            else:
                mask_np = np.zeros((LOAD_SIZE, LOAD_SIZE), dtype=np.uint8)

            idx = len(self.data)
            self.data.append((img_np, dem_np, mask_np))
            (self.positive_indices if mask_np.sum() > 0
             else self.negative_indices).append(idx)

        if n_skip:
            print(f"  {n_skip} file saltati (dem/mask mancante)")

    # ---------- conversione uint8 -> stack 5ch float ----------
    @staticmethod
    def _to_stack(img_u8: np.ndarray, dem_u8: np.ndarray) -> np.ndarray:
        """Ritorna (H,W,5) float32: RGB/255, slope norm, dem/255."""
        rgb = img_u8.astype(np.float32) / 255.0
        dem = dem_u8.astype(np.float32) / 255.0
        gy, gx = np.gradient(dem)
        slope  = np.sqrt(gx ** 2 + gy ** 2)
        smax   = slope.max()
        if smax > 0:
            slope = slope / smax
        return np.concatenate(
            [rgb, slope[..., None], dem[..., None]], axis=-1).astype(np.float32)

    def add_augmented_samples(self, indices: list, transform, n_copies: int, rng_seed: int = 0):
        """
        Genera n_copies versioni augmentate offline (flip/rotazioni: restano uint8)
        per ogni indice in indices e le aggiunge come nuovi positivi.
        """
        import random
        random.seed(rng_seed)
        np.random.seed(rng_seed)
        n_added = 0
        for orig_idx in tqdm(indices, desc=f"Augmentation offline ({n_copies}x)"):
            img_u8, dem_u8, mask_u8 = self.data[orig_idx]
            # impacchetta dem come 4o canale per trasformare tutto insieme
            stack = np.concatenate([img_u8, dem_u8[..., None]], axis=-1)  # (H,W,4) u8
            for _ in range(n_copies):
                aug = transform(image=stack, mask=mask_u8)
                a_img  = aug["image"][..., :3].copy()
                a_dem  = aug["image"][..., 3].copy()
                a_mask = aug["mask"].copy()
                new_idx = len(self.data)
                self.data.append((a_img, a_dem, a_mask))
                self.positive_indices.append(new_idx)
                n_added += 1
        print(f"  Aggiunti {n_added} campioni augmentati "
              f"(da {len(indices)} positivi x {n_copies} copie)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_u8, dem_u8, mask_u8 = self.data[idx]
        stack  = self._to_stack(img_u8, dem_u8)                  # (H,W,5)
        img_t  = torch.from_numpy(stack).permute(2, 0, 1)        # (5,H,W)
        mask_t = torch.from_numpy(mask_u8.astype(np.int64))

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
        img_np  = sample["image"].permute(1, 2, 0).numpy()   # (H,W,5)
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

    # pos_weight dalle statistiche REALI dei pixel del train
    # (i landslide Bijie coprono una frazione di pixel ben diversa da L4S)
    px_fg = px_tot = 0
    for i in train_indices:
        m = full_ds.data[i][2]
        px_fg  += int(m.sum())
        px_tot += m.size
    pos_weight = (px_tot - px_fg) / max(px_fg, 1)
    print(f"  Frazione pixel landslide (train): {px_fg/max(px_tot,1):.4f}"
          f"\n  pos_weight: {pos_weight:.2f}\n")

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
    """Bottleneck adapter zero-delta: x + W2*ReLU(W1*x)."""
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
    Guidance map da terrain 2ch (slope derivata + DEM) -> G in [0,1].
    Applicata come gain spaziale sull'ottico: optical_mod = optical (*) G
    Identica alla versione L4S.
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
    SAM2-Tiny per Bijie con:
      - ottico RGB nativo 3ch -> patch embed ORIGINALE (niente espansione),
        ma scongelato per adattarsi al dominio satellitare TripleSat
      - DSDF Early Guidance: terrain (slope derivata + DEM) modula l'ottico
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

        # ── 1. Patch embed: resta 3ch, lo individuiamo solo per scongelarlo ──
        self.patch_proj = None
        for name, module in v_encoder.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                self.patch_proj = module
                print(f"  Patch embed individuato (3ch, nessuna espansione): {name}")
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
        x     : (B, 5, H, W)  — [R,G,B, slope, dem]
        boxes : (B, 4)
        """
        optical = x[:, :3]           # (B,3,H,W) RGB TripleSat
        terrain = x[:, 3:]           # (B,2,H,W) slope derivata + DEM

        g       = self.early_guidance(terrain)   # (B,1,H,W)
        optical = optical * g                    # broadcasting su 3ch

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
    Macro-average per immagine — identica alla versione L4S.
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
# 5. TEST
# ============================================================

@torch.no_grad()
def evaluate_test(model, test_loader, pos_weight,
                  ckpt_path: str = CHECKPOINT_NAME,
                  history_file: str = HISTORY_FILE):
    """
    Valutazione finale sul test set: carica il best checkpoint,
    calcola le metriche e le appende allo storico. Ritorna il dict dei risultati.
    """
    print("\n" + "=" * 60)
    print("  VALUTAZIONE FINALE SUL TEST SET (Bijie)")
    print("=" * 60)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Best checkpoint caricato (epoca {ckpt['epoch']}, "
              f"val IoU {ckpt['best_iou']:.4f})")
    else:
        print(f"ATTENZIONE: '{ckpt_path}' non trovato, uso i pesi correnti.")
    model.eval()

    te_loss = te_iou = te_iou_bg = te_miou = te_f1 = te_prec = te_rec = 0.0
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

    history = []
    if os.path.exists(history_file):
        with open(history_file) as f:
            history = json.load(f)
    history.append({"test_results": test_results})
    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Risultati salvati in: {history_file}")
    return test_results


# ============================================================
# 6. TRAINING LOOP
# ============================================================

def train():
    full_ds = BijieDataset(DATASET_ROOT)
    train_subset, val_subset, test_subset, pos_weight = build_train_val_test_subsets(
        full_ds, VAL_FRACTION, TEST_FRACTION, RANDOM_SEED)

    # Augmentation offline (geometrica, lavora su uint8 HWC)
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
    train_subset = Subset(full_ds, list(train_subset.indices) + aug_indices)
    print(f"  Train finale: {len(train_subset)} campioni "
          f"({len(train_pos_indices)} pos orig + {len(aug_indices)} aug)")

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

    print(f"\nTraining SAM2-Tiny Adapter (Bijie) | "
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
            print(f"  Record IoU: {best_iou:.4f}  mIoU:{avg['val_miou']:.4f}  "
                  f"F1:{avg['val_f1']:.4f} — checkpoint salvato")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"\nEARLY STOPPING — Miglior IoU val: {best_iou:.4f}")
                break

    # ── TEST FINALE ────────────────────────────────────────────
    evaluate_test(model, test_loader, pos_weight)


# ============================================================
# ENTRY POINT
# ============================================================

def test_only():
    """Solo valutazione sul test set, senza training (stesso split, seed 42)."""
    full_ds = BijieDataset(DATASET_ROOT)
    _, _, test_subset, pos_weight = build_train_val_test_subsets(
        full_ds, VAL_FRACTION, TEST_FRACTION, RANDOM_SEED)
    test_ds     = TransformSubset(test_subset, A.Compose([ToTensorV2()]))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)
    model = SAM2TinyAdapterModel(adapter_dim=64).to(DEVICE)
    evaluate_test(model, test_loader, pos_weight)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_only()
    else:
        train()