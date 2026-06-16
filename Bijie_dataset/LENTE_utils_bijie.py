import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
from PIL import Image
import segmentation_models_pytorch as smp
from transformers import Sam2Model

# ============================================================
# DATASET & TRANSFORMS
# ============================================================
def _find_subdir(parent: str, candidates) -> str | None:
    for c in candidates:
        p = os.path.join(parent, c)
        if os.path.isdir(p):
            return p
    return None

def _stem(path: str) -> str:
    s = os.path.splitext(os.path.basename(path))[0]
    for suf in ("_mask", "_dem", "_label"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s

class BijieDataset(Dataset):
    DIR_CANDIDATES = {
        "landslide":     ["landslide", "Landslide"],
        "non_landslide": ["non-landslide", "non_landslide", "Non-landslide", "nonlandslide"],
        "img":           ["img", "image", "images", "Image"],
        "mask":          ["mask", "masks", "label", "labels", "Mask"],
        "dem":           ["dem", "DEM", "dems"],
    }

    def __init__(self, root_dir: str, load_size: int = 256):
        self.load_size = load_size
        ls_dir  = _find_subdir(root_dir, self.DIR_CANDIDATES["landslide"])
        nls_dir = _find_subdir(root_dir, self.DIR_CANDIDATES["non_landslide"])
        
        if ls_dir is None:
            raise FileNotFoundError(f"Cartella 'landslide' non trovata in {root_dir}.")

        self.data = []   
        self.positive_indices = []
        self.negative_indices = []

        self._load_group(ls_dir, has_mask=True)
        if nls_dir is not None:
            self._load_group(nls_dir, has_mask=False)
        else:
            print("ATTENZIONE: cartella non-landslide non trovata, uso solo i positivi.")

        print(f"Caricati {len(self.data)} campioni Bijie: {len(self.positive_indices)} positivi, {len(self.negative_indices)} negativi")

    def _load_group(self, group_dir: str, has_mask: bool):
        img_dir  = _find_subdir(group_dir, self.DIR_CANDIDATES["img"])
        dem_dir  = _find_subdir(group_dir, self.DIR_CANDIDATES["dem"])
        mask_dir = _find_subdir(group_dir, self.DIR_CANDIDATES["mask"]) if has_mask else None
        
        dem_map  = {_stem(p): p for p in glob.glob(os.path.join(dem_dir, "*.png"))}
        mask_map = {_stem(p): p for p in glob.glob(os.path.join(mask_dir, "*.png"))} if has_mask else {}
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))

        for img_path in tqdm(img_paths, desc="Loading Bijie Dataset"):
            st = _stem(img_path)
            dem_path = dem_map.get(st)
            if dem_path is None: continue

            img = Image.open(img_path).convert("RGB").resize((self.load_size, self.load_size), Image.BILINEAR)
            dem = Image.open(dem_path).convert("L").resize((self.load_size, self.load_size), Image.BILINEAR)
            img_np = np.asarray(img, dtype=np.uint8)          
            dem_np = np.asarray(dem, dtype=np.uint8)          

            if has_mask:
                mask_path = mask_map.get(st)
                if mask_path is None: continue
                m = Image.open(mask_path).convert("L").resize((self.load_size, self.load_size), Image.NEAREST)
                mask_np = (np.asarray(m) > 127).astype(np.uint8)  
            else:
                mask_np = np.zeros((self.load_size, self.load_size), dtype=np.uint8)

            idx = len(self.data)
            self.data.append((img_np, dem_np, mask_np))
            (self.positive_indices if mask_np.sum() > 0 else self.negative_indices).append(idx)

    @staticmethod
    def _to_stack(img_u8: np.ndarray, dem_u8: np.ndarray) -> np.ndarray:
        rgb = img_u8.astype(np.float32) / 255.0
        dem = dem_u8.astype(np.float32) / 255.0
        gy, gx = np.gradient(dem)
        slope  = np.sqrt(gx ** 2 + gy ** 2)
        if slope.max() > 0: slope = slope / slope.max()
        return np.concatenate([rgb, slope[..., None], dem[..., None]], axis=-1).astype(np.float32)

    def add_augmented_samples(self, indices: list, transform, n_copies: int, rng_seed: int = 0):
        import random
        random.seed(rng_seed)
        np.random.seed(rng_seed)
        for orig_idx in indices:
            img_u8, dem_u8, mask_u8 = self.data[orig_idx]
            stack = np.concatenate([img_u8, dem_u8[..., None]], axis=-1)  
            for _ in range(n_copies):
                aug = transform(image=stack, mask=mask_u8)
                self.positive_indices.append(len(self.data))
                self.data.append((aug["image"][..., :3].copy(), aug["image"][..., 3].copy(), aug["mask"].copy()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_u8, dem_u8, mask_u8 = self.data[idx]
        stack  = self._to_stack(img_u8, dem_u8)                  
        img_t  = torch.from_numpy(stack).permute(2, 0, 1)        
        mask_t = torch.from_numpy(mask_u8.astype(np.int64))

        indices = torch.where(mask_t > 0)
        box = torch.tensor([indices[1].min().float(), indices[0].min().float(), indices[1].max().float(), indices[0].max().float()]) if len(indices[0]) > 0 else torch.tensor([0., 0., 10., 10.])
        return {"image": img_t, "mask": mask_t, "boxes": box}

class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self): return len(self.subset)

    def __getitem__(self, idx):
        sample = self.subset[idx]
        if not self.transform: return sample
        aug = self.transform(image=sample["image"].permute(1, 2, 0).numpy(), mask=sample["mask"].numpy().astype(np.float32))
        return {"image": aug["image"], "mask": aug["mask"].long(), "boxes": sample["boxes"]}

def build_train_val_test_subsets(full_ds, val_fraction, test_fraction, random_seed):
    rng = torch.Generator().manual_seed(random_seed)
    shuffled = [list(range(len(full_ds)))[i] for i in torch.randperm(len(full_ds), generator=rng).tolist()]
    
    n_test = int(len(shuffled) * test_fraction)
    n_val = int(len(shuffled[n_test:]) * (val_fraction / (1.0 - test_fraction)))
    
    train_indices = shuffled[n_test:][n_val:]
    val_indices = shuffled[n_test:][:n_val]
    test_indices = shuffled[:n_test]

    px_fg = sum(full_ds.data[i][2].sum() for i in train_indices)
    px_tot = sum(full_ds.data[i][2].size for i in train_indices)
    pos_weight = (px_tot - px_fg) / max(px_fg, 1)

    return Subset(full_ds, train_indices), Subset(full_ds, val_indices), Subset(full_ds, test_indices), pos_weight

# ============================================================
# MODEL ARCHITECTURE
# ============================================================
class Adapter(nn.Module):
    def __init__(self, dim: int, adapter_dim: int = 64):
        super().__init__()
        self.adapter = nn.Sequential(nn.Linear(dim, adapter_dim), nn.ReLU(inplace=True), nn.Linear(adapter_dim, dim))
        nn.init.zeros_(self.adapter[2].weight)
        nn.init.zeros_(self.adapter[2].bias)

    def forward(self, x): return x + self.adapter(x)

class DSDFEarlyGuidance(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(2, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.Conv2d(16, 1, 1), nn.Sigmoid())
    def forward(self, terrain: torch.Tensor): return self.net(terrain)

class SAM2TinyAdapterModel(nn.Module):
    def __init__(self, sam_ckpt_path="facebook/sam2-hiera-tiny", adapter_dim: int = 64):
        super().__init__()
        self.model = Sam2Model.from_pretrained(sam_ckpt_path)
        self.early_guidance = DSDFEarlyGuidance()
        
        self.patch_proj = next((m for m in self.model.vision_encoder.modules() if isinstance(m, nn.Conv2d) and m.in_channels == 3), None)
        if not self.patch_proj: raise RuntimeError("Patch embed Conv2d(in=3) non trovato.")

        self.adapters = nn.ModuleList()
        for module in self.model.vision_encoder.modules():
            if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear):
                adapter = Adapter(module.proj.out_features if hasattr(module, "proj") else module.qkv.in_features, adapter_dim)
                self.adapters.append(adapter)
                module.register_forward_hook(lambda m, args, out, a=adapter: (a(out[0]),) + out[1:] if isinstance(out, tuple) else a(out))
            elif hasattr(module, "q_proj") and isinstance(module.q_proj, nn.Linear):
                adapter = Adapter(module.out_proj.out_features if hasattr(module, "out_proj") else module.q_proj.in_features, adapter_dim)
                self.adapters.append(adapter)
                module.register_forward_hook(lambda m, args, out, a=adapter: (a(out[0]),) + out[1:] if isinstance(out, tuple) else a(out))

        self._freeze()

    def _freeze(self):
        for p in self.model.parameters(): p.requires_grad = False
        for p in self.model.mask_decoder.parameters(): p.requires_grad = True
        for p in self.patch_proj.parameters(): p.requires_grad = True
        for adapter in self.adapters:
            for p in adapter.parameters(): p.requires_grad = True
        for p in self.early_guidance.parameters(): p.requires_grad = True

    def forward(self, x: torch.Tensor, boxes: torch.Tensor):
        optical, terrain = x[:, :3], x[:, 3:]
        optical = optical * self.early_guidance(terrain)
        if boxes.ndim == 2: boxes = boxes.unsqueeze(1)
        outputs = self.model(pixel_values=optical, input_boxes=boxes, multimask_output=False)
        return outputs.pred_masks, getattr(outputs, "iou_scores", getattr(outputs, "iou_predictions", getattr(outputs, "pred_ious", None)))

# ============================================================
# METRICS & UTILS
# ============================================================
dice_loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)

def get_best_mask(all_preds: torch.Tensor, scores) -> torch.Tensor:
    if all_preds.ndim == 5: all_preds = all_preds.squeeze(1)
    if scores is not None:
        if scores.ndim == 3: scores = scores.squeeze(1)
        return all_preds[torch.arange(all_preds.shape[0], device=all_preds.device), torch.argmax(scores, dim=-1)].unsqueeze(1).contiguous()
    return all_preds[:, 0:1].contiguous()

def criterion(y_pred, y_true, pos_weight: float = 1.0, lambda_dice: float = 0.3):
    y_true_rs = F.interpolate(y_true.unsqueeze(1).float(), size=y_pred.shape[2:], mode="nearest")
    return F.binary_cross_entropy_with_logits(y_pred.contiguous(), y_true_rs, pos_weight=torch.tensor([pos_weight], device=y_pred.device)) + lambda_dice * dice_loss_fn(y_pred, y_true_rs)

def calculate_metrics(preds, masks, epsilon=1e-7):
    if preds.shape[-2:] != masks.shape[-2:]: preds = F.interpolate(preds, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    pr, gt = (torch.sigmoid(preds) > 0.5).float(), masks.unsqueeze(1).float()
    tp_fg, fp_fg, fn_fg = (pr * gt).sum(dim=(2, 3)), (pr * (1 - gt)).sum(dim=(2, 3)), ((1 - pr) * gt).sum(dim=(2, 3))
    prec, rec = ((tp_fg + epsilon) / (tp_fg + fp_fg + epsilon)).mean().item(), ((tp_fg + epsilon) / (tp_fg + fn_fg + epsilon)).mean().item()
    return ((tp_fg + epsilon) / (tp_fg + fp_fg + fn_fg + epsilon)).mean().item(), 0.0, 0.0, (2 * prec * rec) / (prec + rec + epsilon), prec, rec

def scale_boxes(boxes, orig_h, orig_w, target, margin=3):
    sx, sy = target / orig_w, target / orig_h
    boxes[:, 0] = torch.clamp(boxes[:, 0] * sx - margin, min=0)
    boxes[:, 1] = torch.clamp(boxes[:, 1] * sy - margin, min=0)
    boxes[:, 2] = torch.clamp(boxes[:, 2] * sx + margin, max=target - 1)
    boxes[:, 3] = torch.clamp(boxes[:, 3] * sy + margin, max=target - 1)
    return boxes

def save_checkpoint(model, optimizer, scheduler, epoch, best_iou, path):
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "best_iou": best_iou}, path)

def load_checkpoint(model, optimizer, scheduler, path, device):
    if not os.path.exists(path): return 0, 0.0
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    try: optimizer.load_state_dict(ckpt["optimizer_state_dict"]); scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    except: pass
    return ckpt["epoch"] + 1, ckpt["best_iou"]