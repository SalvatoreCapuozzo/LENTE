import os
import glob
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
import segmentation_models_pytorch as smp
from transformers import Sam2Model

# ============================================================
# DATASET & SPLITS
# ============================================================

class LandslideH5Dataset(Dataset):
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
                image = f["img"][:].astype(np.float32)
            with h5py.File(mask_path, "r") as f:
                mask  = f["mask"][:].astype(np.int64)

            idx = len(self.data)
            self.data.append((image, mask))
            (self.positive_indices if mask.sum() > 0 else self.negative_indices).append(idx)

    def add_augmented_samples(self, indices: list, transform, n_copies: int, rng_seed: int = 0):
        import random
        random.seed(rng_seed)
        np.random.seed(rng_seed)
        for orig_idx in tqdm(indices, desc=f"Augmentation offline ({n_copies}x)"):
            image_np, mask_np = self.data[orig_idx]
            for _ in range(n_copies):
                aug = transform(image=image_np, mask=mask_np.astype(np.float32))
                new_idx = len(self.data)
                self.data.append((aug["image"], aug["mask"]))
                self.positive_indices.append(new_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_np, mask_np = self.data[idx]
        img_t  = torch.from_numpy(image_np).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask_np).long()

        indices = torch.where(mask_t > 0)
        if len(indices[0]) > 0:
            box = torch.tensor([
                indices[1].min().float(), indices[0].min().float(),
                indices[1].max().float(), indices[0].max().float(),
            ])
        else:
            box = torch.tensor([0., 0., 10., 10.])

        return {"image": img_t, "mask": mask_t, "boxes": box}

class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset    = subset
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
        return {"image": aug["image"], "mask":  aug["mask"].long(), "boxes": sample["boxes"]}

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
    ls_frac    = 0.35
    pixel_ls   = n_pos_tr * ls_frac
    pixel_bg   = n_neg_tr + n_pos_tr * (1.0 - ls_frac)
    pos_weight = pixel_bg / max(pixel_ls, 1.0)

    return Subset(full_ds, train_indices), Subset(full_ds, val_indices), Subset(full_ds, test_indices), pos_weight

# ============================================================
# MODEL: LENTE (Renamed from SAM2TinyAdapterModel)
# ============================================================

class Adapter(nn.Module):
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
        return self.guidance_net(terrain)

class LENTE(nn.Module):
    def __init__(self, sam_ckpt_path: str, adapter_dim: int = 64):
        super().__init__()
        self.model = Sam2Model.from_pretrained(sam_ckpt_path)
        self.early_guidance = DSDFEarlyGuidance()
        v_encoder = self.model.vision_encoder

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
                adapter = Adapter(dim, adapter_dim)
                self.adapters.append(adapter)
                module.register_forward_hook(self._make_hook(adapter))
            elif hasattr(module, "q_proj") and isinstance(module.q_proj, nn.Linear):
                dim = module.out_proj.out_features if hasattr(module, "out_proj") else module.q_proj.in_features
                adapter = Adapter(dim, adapter_dim)
                self.adapters.append(adapter)
                module.register_forward_hook(self._make_hook(adapter))

        self._freeze()

    @staticmethod
    def _make_hook(adapter: Adapter):
        def hook(module, args, output):
            if isinstance(output, tuple):
                return (adapter(output[0]),) + output[1:]
            return adapter(output)
        return hook

    def _freeze(self):
        for p in self.model.parameters(): p.requires_grad = False
        for p in self.model.mask_decoder.parameters(): p.requires_grad = True
        for p in self.patch_proj.parameters(): p.requires_grad = True
        for adapter in self.adapters:
            for p in adapter.parameters(): p.requires_grad = True
        for p in self.early_guidance.parameters(): p.requires_grad = True

    def forward(self, x: torch.Tensor, boxes: torch.Tensor):
        optical = x[:, :12]
        terrain = x[:, 12:]
        g = self.early_guidance(terrain)
        optical = optical * g

        if boxes.ndim == 2: boxes = boxes.unsqueeze(1)
        outputs = self.model(pixel_values=optical, input_boxes=boxes, multimask_output=False)

        scores = getattr(outputs, "iou_scores", getattr(outputs, "iou_predictions", getattr(outputs, "pred_ious", None)))
        return outputs.pred_masks, scores

# ============================================================
# UTILS & LOSS
# ============================================================

dice_loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)

def get_best_mask(all_preds: torch.Tensor, scores) -> torch.Tensor:
    if all_preds.ndim == 5: all_preds = all_preds.squeeze(1)
    if scores is not None:
        if scores.ndim == 3: scores = scores.squeeze(1)
        best = torch.argmax(scores, dim=-1)
        bidx = torch.arange(all_preds.shape[0], device=all_preds.device)
        return all_preds[bidx, best].unsqueeze(1).contiguous()
    return all_preds[:, 0:1].contiguous()

def criterion(y_pred: torch.Tensor, y_true: torch.Tensor, pos_weight: float = 1.0, lambda_dice: float = 0.3):
    y_pred = y_pred.contiguous()
    y_true_rs = F.interpolate(y_true.unsqueeze(1).float(), size=y_pred.shape[2:], mode="nearest")
    pw = torch.tensor([pos_weight], device=y_pred.device)
    ce = F.binary_cross_entropy_with_logits(y_pred, y_true_rs, pos_weight=pw)
    dice = dice_loss_fn(y_pred, y_true_rs)
    return ce + lambda_dice * dice

def calculate_metrics(preds: torch.Tensor, masks: torch.Tensor, epsilon: float = 1e-7):
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

def scale_boxes(boxes: torch.Tensor, orig_h: int, orig_w: int, target: int, margin: int = 3) -> torch.Tensor:
    sx = target / orig_w
    sy = target / orig_h
    boxes[:, 0] = torch.clamp(boxes[:, 0] * sx - margin, min=0)
    boxes[:, 1] = torch.clamp(boxes[:, 1] * sy - margin, min=0)
    boxes[:, 2] = torch.clamp(boxes[:, 2] * sx + margin, max=target - 1)
    boxes[:, 3] = torch.clamp(boxes[:, 3] * sy + margin, max=target - 1)
    return boxes

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
    except Exception: pass
    return ckpt["epoch"] + 1, ckpt["best_iou"]