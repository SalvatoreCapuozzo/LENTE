import os
import glob
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
import segmentation_models_pytorch as smp
from transformers import Sam2Model

# Optional deps (only needed for the respective datasets)
try:
    import h5py
except ImportError:
    h5py = None
try:
    from PIL import Image
except ImportError:
    Image = None


# ============================================================
# CONFIG LOADING
# ============================================================

def load_config(config_path="config.yaml", dataset=None):
    """
    Loads the unified YAML and returns a FLAT dict merging `common` with the
    chosen dataset block. `dataset` overrides `active_dataset` in the file.
    """
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    name = dataset or raw.get("active_dataset")
    if name not in raw.get("datasets", {}):
        raise ValueError(
            f"Dataset '{name}' non trovato nel config. "
            f"Disponibili: {list(raw.get('datasets', {}).keys())}")

    cfg = dict(raw["common"])
    cfg.update(raw["datasets"][name])
    cfg["dataset_name"] = name
    cfg["n_optical"]    = len(cfg["optical_bands"])
    return cfg


# ============================================================
# DATASETS
# ============================================================

class L4SDataset(Dataset):
    """Landslide4Sense (.h5). Stack a 14ch [12 Sentinel-2 + Slope + DEM]."""
    IMG_DIR  = "TrainData/img"
    MASK_DIR = "TrainData/mask"

    def __init__(self, root_dir, load_size=None):
        if h5py is None:
            raise ImportError("h5py richiesto per il dataset L4S (pip install h5py)")
        img_dir  = os.path.join(root_dir, self.IMG_DIR)
        mask_dir = os.path.join(root_dir, self.MASK_DIR)

        all_images = sorted(glob.glob(os.path.join(img_dir, "*.h5")))
        if not all_images:
            raise FileNotFoundError(f"Nessun .h5 trovato in {img_dir}")

        print(f"Caricamento L4S in RAM ({len(all_images)} file)...")
        self.data             = []
        self.positive_indices = []
        self.negative_indices = []

        for img_path in tqdm(all_images, desc="Loading TrainData"):
            mask_name = os.path.basename(img_path).replace("image_", "mask_")
            mask_path = os.path.join(mask_dir, mask_name)
            with h5py.File(img_path, "r") as f:
                image = f["img"][:].astype(np.float32)   # (H,W,14)
            with h5py.File(mask_path, "r") as f:
                mask = f["mask"][:].astype(np.int64)     # (H,W)
            idx = len(self.data)
            self.data.append((image, mask))
            (self.positive_indices if mask.sum() > 0 else self.negative_indices).append(idx)

        n_pos, n_neg = len(self.positive_indices), len(self.negative_indices)
        print(f"Caricati {len(self.data)} campioni: {n_pos} positivi, {n_neg} negativi")

    def get_mask(self, idx):
        return self.data[idx][1]

    def add_augmented_samples(self, indices, transform, n_copies, rng_seed=0):
        import random
        random.seed(rng_seed); np.random.seed(rng_seed)
        for orig_idx in tqdm(indices, desc=f"Augmentation offline ({n_copies}x)"):
            image_np, mask_np = self.data[orig_idx]
            for _ in range(n_copies):
                aug = transform(image=image_np, mask=mask_np.astype(np.float32))
                self.positive_indices.append(len(self.data))
                self.data.append((aug["image"], aug["mask"]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_np, mask_np = self.data[idx]
        img_t  = torch.from_numpy(image_np).permute(2, 0, 1)   # (14,H,W)
        mask_t = torch.from_numpy(np.asarray(mask_np)).long()
        return {"image": img_t, "mask": mask_t, "boxes": _bbox_from_mask(mask_t)}


def _find_subdir(parent, candidates):
    for c in candidates:
        p = os.path.join(parent, c)
        if os.path.isdir(p):
            return p
    return None


def _stem(path):
    s = os.path.splitext(os.path.basename(path))[0]
    for suf in ("_mask", "_dem", "_label"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


class BijieDataset(Dataset):
    """
    Bijie (PNG). Stack a 5ch [R,G,B, slope, dem], slope derivata dal DEM via
    gradiente. I negativi hanno maschera vuota.
    """
    DIR_CANDIDATES = {
        "landslide":     ["landslide", "Landslide"],
        "non_landslide": ["non-landslide", "non_landslide", "Non-landslide", "nonlandslide"],
        "img":           ["img", "image", "images", "Image"],
        "mask":          ["mask", "masks", "label", "labels", "Mask"],
        "dem":           ["dem", "DEM", "dems"],
    }

    def __init__(self, root_dir, load_size=256):
        if Image is None:
            raise ImportError("Pillow richiesto per il dataset Bijie (pip install pillow)")
        self.load_size = load_size or 256
        ls_dir  = _find_subdir(root_dir, self.DIR_CANDIDATES["landslide"])
        nls_dir = _find_subdir(root_dir, self.DIR_CANDIDATES["non_landslide"])
        if ls_dir is None:
            raise FileNotFoundError(
                f"Cartella 'landslide' non trovata in {root_dir}. Contenuto: {os.listdir(root_dir)}")

        print(f"Caricamento Bijie in RAM (load_size={self.load_size})...")
        self.data             = []   # (img_u8 HW3, dem_u8 HW, mask_u8 HW)
        self.positive_indices = []
        self.negative_indices = []

        self._load_group(ls_dir, has_mask=True)
        if nls_dir is not None:
            self._load_group(nls_dir, has_mask=False)
        else:
            print("ATTENZIONE: cartella non-landslide non trovata, uso solo i positivi.")

        n_pos, n_neg = len(self.positive_indices), len(self.negative_indices)
        print(f"Caricati {len(self.data)} campioni: {n_pos} positivi, {n_neg} negativi")

    def _load_group(self, group_dir, has_mask):
        img_dir  = _find_subdir(group_dir, self.DIR_CANDIDATES["img"])
        dem_dir  = _find_subdir(group_dir, self.DIR_CANDIDATES["dem"])
        mask_dir = _find_subdir(group_dir, self.DIR_CANDIDATES["mask"]) if has_mask else None
        if img_dir is None or dem_dir is None:
            raise FileNotFoundError(
                f"Sottocartelle img/dem non trovate in {group_dir}. Contenuto: {os.listdir(group_dir)}")
        if has_mask and mask_dir is None:
            raise FileNotFoundError(f"Cartella mask non trovata in {group_dir}")

        dem_map  = {_stem(p): p for p in glob.glob(os.path.join(dem_dir, "*.png"))}
        mask_map = ({_stem(p): p for p in glob.glob(os.path.join(mask_dir, "*.png"))} if has_mask else {})

        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        if not img_paths:
            raise FileNotFoundError(f"Nessun .png in {img_dir}")

        ls = self.load_size
        n_skip = 0
        desc = "Loading landslide" if has_mask else "Loading non-landslide"
        for img_path in tqdm(img_paths, desc=desc):
            st = _stem(img_path)
            dem_path = dem_map.get(st)
            if dem_path is None:
                n_skip += 1; continue

            img = Image.open(img_path).convert("RGB").resize((ls, ls), Image.BILINEAR)
            dem = Image.open(dem_path).convert("L").resize((ls, ls), Image.BILINEAR)
            img_np = np.asarray(img, dtype=np.uint8)
            dem_np = np.asarray(dem, dtype=np.uint8)

            if has_mask:
                mask_path = mask_map.get(st)
                if mask_path is None:
                    n_skip += 1; continue
                m = Image.open(mask_path).convert("L").resize((ls, ls), Image.NEAREST)
                mask_np = (np.asarray(m) > 127).astype(np.uint8)
            else:
                mask_np = np.zeros((ls, ls), dtype=np.uint8)

            idx = len(self.data)
            self.data.append((img_np, dem_np, mask_np))
            (self.positive_indices if mask_np.sum() > 0 else self.negative_indices).append(idx)

        if n_skip:
            print(f"  {n_skip} file saltati (dem/mask mancante)")

    def get_mask(self, idx):
        return self.data[idx][2]

    @staticmethod
    def _to_stack(img_u8, dem_u8):
        rgb = img_u8.astype(np.float32) / 255.0
        dem = dem_u8.astype(np.float32) / 255.0
        gy, gx = np.gradient(dem)
        slope  = np.sqrt(gx ** 2 + gy ** 2)
        smax   = slope.max()
        if smax > 0:
            slope = slope / smax
        return np.concatenate([rgb, slope[..., None], dem[..., None]], axis=-1).astype(np.float32)

    def add_augmented_samples(self, indices, transform, n_copies, rng_seed=0):
        import random
        random.seed(rng_seed); np.random.seed(rng_seed)
        for orig_idx in tqdm(indices, desc=f"Augmentation offline ({n_copies}x)"):
            img_u8, dem_u8, mask_u8 = self.data[orig_idx]
            stack = np.concatenate([img_u8, dem_u8[..., None]], axis=-1)  # (H,W,4) u8
            for _ in range(n_copies):
                aug = transform(image=stack, mask=mask_u8)
                self.positive_indices.append(len(self.data))
                self.data.append((aug["image"][..., :3].copy(),
                                  aug["image"][..., 3].copy(),
                                  aug["mask"].copy()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_u8, dem_u8, mask_u8 = self.data[idx]
        stack  = self._to_stack(img_u8, dem_u8)                # (H,W,5)
        img_t  = torch.from_numpy(stack).permute(2, 0, 1)      # (5,H,W)
        mask_t = torch.from_numpy(mask_u8.astype(np.int64))
        return {"image": img_t, "mask": mask_t, "boxes": _bbox_from_mask(mask_t)}


# Backward-compat alias (old scripts imported this name)
LandslideH5Dataset = L4SDataset


def build_dataset(cfg):
    """Factory: ritorna il dataset giusto in base a cfg['dataset_type']."""
    dt = cfg["dataset_type"]
    if dt == "h5":
        return L4SDataset(cfg["dataset_root"], load_size=cfg.get("load_size"))
    elif dt == "png":
        return BijieDataset(cfg["dataset_root"], load_size=cfg.get("load_size"))
    raise ValueError(f"dataset_type sconosciuto: {dt}")


def _bbox_from_mask(mask_t):
    indices = torch.where(mask_t > 0)
    if len(indices[0]) > 0:
        return torch.tensor([
            indices[1].min().float(), indices[0].min().float(),
            indices[1].max().float(), indices[0].max().float(),
        ])
    return torch.tensor([0., 0., 10., 10.])


class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

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


def build_train_val_test_subsets(full_ds, cfg):
    val_fraction  = cfg["val_fraction"]
    test_fraction = cfg["test_fraction"]
    rng      = torch.Generator().manual_seed(cfg["random_seed"])
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

    mode = cfg.get("pos_weight_mode", "pixels")
    if mode == "heuristic":
        ls_frac    = 0.35
        pixel_ls   = n_pos_tr * ls_frac
        pixel_bg   = n_neg_tr + n_pos_tr * (1.0 - ls_frac)
        pos_weight = pixel_bg / max(pixel_ls, 1.0)
    else:  # "pixels"
        px_fg = px_tot = 0
        for i in train_indices:
            m = np.asarray(full_ds.get_mask(i))
            px_fg  += int(m.sum())
            px_tot += m.size
        pos_weight = (px_tot - px_fg) / max(px_fg, 1)

    print(f"Split: train {len(train_indices)} | val {len(val_indices)} | test {len(test_indices)} "
          f"| pos_weight ({mode}): {pos_weight:.2f}")

    return (Subset(full_ds, train_indices), Subset(full_ds, val_indices),
            Subset(full_ds, test_indices), pos_weight)


# ============================================================
# MODEL: LENTE  (adattivo — espande il patch embed solo se n_optical != 3)
# ============================================================

class Adapter(nn.Module):
    def __init__(self, dim, adapter_dim=64):
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
    def __init__(self):
        super().__init__()
        self.guidance_net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, terrain):
        return self.guidance_net(terrain)


class LENTE(nn.Module):
    """
    SAM2-Tiny + Adapter + DSDF early guidance.
    n_optical = canali ottici (12 per L4S, 3 per Bijie). Se != 3 il patch embed
    viene espanso copiando i pesi RGB; se == 3 viene solo scongelato.
    Il terrain (2ch) modula l'ottico via guidance map.
    """
    def __init__(self, sam_ckpt_path, n_optical=12, adapter_dim=64):
        super().__init__()
        self.n_optical = n_optical
        self.model = Sam2Model.from_pretrained(sam_ckpt_path)
        self.early_guidance = DSDFEarlyGuidance()
        v_encoder = self.model.vision_encoder

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
                    parts  = name.split(".")
                    parent = v_encoder.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else v_encoder
                    setattr(parent, parts[-1], new_conv)
                    self.patch_proj = new_conv
                break
        if self.patch_proj is None:
            raise RuntimeError("Patch embed Conv2d(in=3) non trovato.")

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
        for p in self.patch_proj.parameters(): p.requires_grad = True
        for ad in self.adapters:
            for p in ad.parameters(): p.requires_grad = True
        for p in self.early_guidance.parameters(): p.requires_grad = True

    def forward(self, x, boxes):
        n = self.n_optical
        optical = x[:, :n]       # (B, n_optical, H, W)
        terrain = x[:, n:]       # (B, 2, H, W)
        g = self.early_guidance(terrain)
        optical = optical * g

        if boxes.ndim == 2: boxes = boxes.unsqueeze(1)
        outputs = self.model(pixel_values=optical, input_boxes=boxes, multimask_output=False)
        scores = getattr(outputs, "iou_scores",
                         getattr(outputs, "iou_predictions",
                                 getattr(outputs, "pred_ious", None)))
        return outputs.pred_masks, scores


# ============================================================
# LOSS / METRICS / IO
# ============================================================

dice_loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)


def get_best_mask(all_preds, scores):
    if all_preds.ndim == 5: all_preds = all_preds.squeeze(1)
    if scores is not None:
        if scores.ndim == 3: scores = scores.squeeze(1)
        best = torch.argmax(scores, dim=-1)
        bidx = torch.arange(all_preds.shape[0], device=all_preds.device)
        return all_preds[bidx, best].unsqueeze(1).contiguous()
    return all_preds[:, 0:1].contiguous()


def criterion(y_pred, y_true, pos_weight=1.0, lambda_dice=0.3):
    y_pred = y_pred.contiguous()
    y_true_rs = F.interpolate(y_true.unsqueeze(1).float(), size=y_pred.shape[2:], mode="nearest")
    pw = torch.tensor([pos_weight], device=y_pred.device)
    ce = F.binary_cross_entropy_with_logits(y_pred, y_true_rs, pos_weight=pw)
    dice = dice_loss_fn(y_pred, y_true_rs)
    return ce + lambda_dice * dice


def calculate_metrics(preds, masks, epsilon=1e-7):
    if preds.shape[-2:] != masks.shape[-2:]:
        preds = F.interpolate(preds, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    pr = (torch.sigmoid(preds) > 0.5).float()
    gt = masks.unsqueeze(1).float()

    tp_fg = (pr * gt).sum(dim=(2, 3))
    fp_fg = (pr * (1 - gt)).sum(dim=(2, 3))
    fn_fg = ((1 - pr) * gt).sum(dim=(2, 3))
    iou_fg = ((tp_fg + epsilon) / (tp_fg + fp_fg + fn_fg + epsilon)).mean().item()
    prec = ((tp_fg + epsilon) / (tp_fg + fp_fg + epsilon)).mean().item()
    rec = ((tp_fg + epsilon) / (tp_fg + fn_fg + epsilon)).mean().item()
    f1 = (2 * prec * rec) / (prec + rec + epsilon)

    pr_bg = 1.0 - pr; gt_bg = 1.0 - gt
    tp_bg = (pr_bg * gt_bg).sum(dim=(2, 3))
    fp_bg = (pr_bg * (1 - gt_bg)).sum(dim=(2, 3))
    fn_bg = ((1 - pr_bg) * gt_bg).sum(dim=(2, 3))
    iou_bg = ((tp_bg + epsilon) / (tp_bg + fp_bg + fn_bg + epsilon)).mean().item()

    return iou_fg, iou_bg, (iou_fg + iou_bg) / 2.0, f1, prec, rec


def scale_boxes(boxes, orig_h, orig_w, target, margin=3):
    sx = target / orig_w; sy = target / orig_h
    boxes[:, 0] = torch.clamp(boxes[:, 0] * sx - margin, min=0)
    boxes[:, 1] = torch.clamp(boxes[:, 1] * sy - margin, min=0)
    boxes[:, 2] = torch.clamp(boxes[:, 2] * sx + margin, max=target - 1)
    boxes[:, 3] = torch.clamp(boxes[:, 3] * sy + margin, max=target - 1)
    return boxes


def build_model(cfg, device):
    """Costruisce LENTE col numero di canali ottici giusto dal config."""
    return LENTE(sam_ckpt_path=cfg["sam_ckpt_path"],
                 n_optical=cfg["n_optical"],
                 adapter_dim=cfg["adapter_dim"]).to(device)


def save_checkpoint(model, optimizer, scheduler, epoch, best_iou, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_iou": best_iou,
    }, path)


def load_checkpoint(model, optimizer, scheduler, path, device):
    if not os.path.exists(path):
        return 0, 0.0
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    except Exception:
        pass
    return ckpt["epoch"] + 1, ckpt["best_iou"]
