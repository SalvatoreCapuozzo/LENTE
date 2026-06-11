"""
train_sam2tiny_lora_l4s.py
===========================
SAM2-Tiny con LoRA + patch embed espanso a 14 canali su Landslide4Sense.

Architettura:
  - Input    : 14 canali diretti [B, 14, H, W] (12 ottici + DEM + Slope)
  - patch_proj: Conv2d 3->14 (pesi RGB copiati, extra init a media RGB)
  - LoRA     : applicato a tutti i Linear dei blocchi attention del
               vision encoder (qkv/q_proj/k_proj/v_proj + out_proj/proj)
               rank=LORA_RANK, alpha=LORA_ALPHA
  - Trainabili: patch_proj + LoRA A/B matrices + mask_decoder
  - Frozen   : tutti gli altri pesi del vision encoder

Augmentation: solo flip/rotate offline (fisicamente corretta per DEM/Slope)

Uso:
    python train_sam2tiny_lora_l4s.py
    python train_sam2tiny_lora_l4s.py  # LORA_RANK modificabile in CONFIG
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import glob
import h5py
import json
import math
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
# CONFIGURAZIONE
# ============================================================

HF_PATH       = "facebook/sam2-hiera-tiny"
IN_CHANNELS   = 14        # 12 ottici + DEM + Slope
LORA_RANK     = 8         # rank LoRA — parametrico, modifica qui
LORA_ALPHA    = 16        # alpha LoRA (scaling = alpha/rank)
BATCH_SIZE    = 16
TARGET_SIZE   = 1024
AUG_COPIES    = 2         # copie offline per ogni positivo del train

DATASET_ROOT      = r"/datadrive/landslide/SAM3/SAM2/LandSlide4Sense"
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS            = 100
PATIENCE          = 15
NEG_FRACTION      = 1.0
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
                image = f["img"][:].astype(np.float32)   # (128,128,14) HWC
            with h5py.File(mask_path, "r") as f:
                mask = f["mask"][:].astype(np.float32)   # (128,128)
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

    val_indices = val_pool
    pos_train   = [i for i in train_pool if i     in pos_set]
    neg_train   = [i for i in train_pool if i not in pos_set]

    n_neg        = min(int(len(pos_train) * neg_fraction), len(neg_train))
    neg_perm     = torch.randperm(len(neg_train), generator=rng).tolist()
    neg_selected = [neg_train[i] for i in neg_perm[:n_neg]]

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
# 3. LORA
# ============================================================

class LoRALinear(nn.Module):
    """
    Sostituisce un nn.Linear con una versione LoRA:

        W_new = W_frozen + (B @ A) * (alpha / rank)

    - W_frozen : pesi originali, completamente frozen
    - A [rank, in_features]  : init gaussiana
    - B [out_features, rank] : init zero  → delta=0 all'inizio

    Lo scaling alpha/rank segue la convenzione del paper LoRA originale.
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        in_f  = linear.in_features
        out_f = linear.out_features

        # Copia il Linear originale e lo congela
        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        self.bias   = (
            nn.Parameter(linear.bias.data.clone(), requires_grad=False)
            if linear.bias is not None else None
        )

        # Matrici LoRA trainabili
        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        self.scale  = alpha / rank

        # Init A con gaussiana (come nel paper originale)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return base + lora


def inject_lora(vision_encoder: nn.Module, rank: int, alpha: float) -> int:
    """
    Sostituisce tutti i nn.Linear nei blocchi attention del vision encoder
    con LoRALinear. Targets: qkv, q_proj, k_proj, v_proj, out_proj, proj.

    Returns: numero di layer sostituiti.
    """
    TARGET_NAMES = {"qkv", "q_proj", "k_proj", "v_proj", "out_proj", "proj"}
    count = 0

    for module_name, module in vision_encoder.named_modules():
        # Cerca Linear che sono figli diretti di un blocco attention
        for child_name, child in module.named_children():
            if (
                isinstance(child, nn.Linear)
                and child_name in TARGET_NAMES
                and not isinstance(child, LoRALinear)
            ):
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(module, child_name, lora_layer)
                count += 1

    return count



# ============================================================
# 4. TERRAIN STREAM
# ============================================================

class TinyViTEncoder(nn.Module):
    """
    Encoder leggero per lo stream terrain (DEM+Slope, 2 canali).
    Input : [B, 2, H, W]
    Output: [B, embed_dim, H/patch_stride, W/patch_stride]
    """
    def __init__(self, in_channels=2, embed_dim=256, num_heads=4,
                 num_layers=2, patch_stride=16):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_stride, stride=patch_stride
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True, activation="gelu",
            norm_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.patch_embed(x)            # [B, E, Hd, Wd]
        B, E, Hd, Wd = feat.shape
        seq  = feat.flatten(2).transpose(1, 2)
        seq  = self.norm(self.transformer(seq))
        return seq.transpose(1, 2).reshape(B, E, Hd, Wd)


class TerrainCrossAttention(nn.Module):
    """
    Cross-attention sul neck del vision encoder.
    Query  : neck features [B, C, H, W]
    Key/Val: terrain features [B, E, Hd, Wd]
    Output : [B, C, H, W]

    wo inizializzato a zero -> pass-through identita' all'epoca 0.
    """
    def __init__(self, dim_neck: int, dim_terrain: int, num_heads: int = 8):
        super().__init__()
        self.wq   = nn.Linear(dim_neck,    dim_neck, bias=False)
        self.wk   = nn.Linear(dim_terrain, dim_neck, bias=False)
        self.wv   = nn.Linear(dim_terrain, dim_neck, bias=False)
        self.attn = nn.MultiheadAttention(dim_neck, num_heads, batch_first=True)
        self.wo   = nn.Linear(dim_neck, dim_neck, bias=False)
        self.norm = nn.LayerNorm(dim_neck)
        nn.init.zeros_(self.wo.weight)

    @staticmethod
    def safe_heads(dim: int) -> int:
        for h in [8, 4, 2, 1]:
            if dim % h == 0:
                return h
        return 1

    def forward(self, neck: torch.Tensor, terrain: torch.Tensor) -> torch.Tensor:
        if neck.ndim != 4:
            return neck
        B, C, H, W   = neck.shape
        _, E, Hd, Wd = terrain.shape

        # Porta neck a sequenza [B, H*W, C]
        n_seq = neck.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Interpola terrain alla risoluzione del neck
        if (Hd, Wd) != (H, W):
            terrain = F.interpolate(terrain, size=(H, W),
                                    mode="bilinear", align_corners=False)
        t_seq = terrain.permute(0, 2, 3, 1).reshape(B, H * W, E)

        q = self.wq(n_seq)
        k = self.wk(t_seq)
        v = self.wv(t_seq)
        out, _ = self.attn(q, k, v)
        fused  = n_seq + self.wo(self.norm(out))
        return fused.reshape(B, H, W, C).permute(0, 3, 1, 2)


# ============================================================
# 4. MODELLO
# ============================================================

class SAM2TinyLoRA(nn.Module):
    """
    SAM2-Tiny con:
      - patch embed espanso 3 -> IN_CHANNELS (14)
      - LoRA in tutti i Linear dei blocchi attention del vision encoder
      - TerrainStream: TinyViTEncoder(2ch) + CrossAttention sul neck
      - mask_decoder scongelato

    Trainabili: patch_proj + LoRA A/B + terrain_encoder + terrain_attn + mask_decoder
    Frozen    : tutti gli altri pesi
    """

    def __init__(
        self,
        hf_path:       str   = HF_PATH,
        in_channels:   int   = IN_CHANNELS,
        lora_rank:     int   = LORA_RANK,
        lora_alpha:    float = LORA_ALPHA,
        terrain_dim:   int   = 256,
        terrain_heads: int   = 4,
        terrain_layers:int   = 2,
        patch_stride:  int   = 16,
    ):
        super().__init__()
        print(f"Caricamento SAM2: {hf_path}")
        self.backbone = Sam2Model.from_pretrained(hf_path)

        # 1. Espandi patch embed 3 -> in_channels
        self.patch_proj = self._expand_patch_embed(in_channels)

        # 2. Inietta LoRA nel vision encoder
        n_lora = inject_lora(
            self.backbone.vision_encoder,
            rank=lora_rank, alpha=lora_alpha
        )
        print(f"  LoRA iniettato in {n_lora} layer (rank={lora_rank}, alpha={lora_alpha})")

        # Terrain stream (DEM + Slope, 2 canali)
        dim_neck = self._detect_neck_dim()
        self.terrain_encoder = TinyViTEncoder(
            in_channels=2, embed_dim=terrain_dim,
            num_heads=terrain_heads, num_layers=terrain_layers,
            patch_stride=patch_stride,
        )
        self.terrain_attn = TerrainCrossAttention(
            dim_neck=dim_neck, dim_terrain=terrain_dim,
            num_heads=TerrainCrossAttention.safe_heads(dim_neck),
        )
        self._z_terrain = None
        self._register_terrain_hook()
        print(f"  TerrainStream: TinyViTEncoder(2ch) + CrossAttention (neck_dim={dim_neck})")

        # 3. Gradient checkpointing (best-effort)
        n_ckpt = self._enable_gradient_checkpointing()
        if n_ckpt:
            print(f"  Gradient checkpointing: {n_ckpt} blocchi")

        # 4. Freeze tutto, poi scongela selettivamente
        self._freeze()
        self._print_trainable()

    # ── Neck dim detection ───────────────────────────────────────────────
    def _detect_neck_dim(self) -> int:
        v_enc    = self.backbone.vision_encoder
        detected = [None]
        neck_mod = getattr(v_enc, "neck", None)
        if neck_mod is None:
            for _, m in v_enc.named_modules():
                if isinstance(m, nn.Conv2d):
                    neck_mod = m
        if neck_mod is None:
            return 256
        def probe(module, args, output):
            raw = output[0] if isinstance(output, tuple) else output
            if raw.ndim == 4:
                detected[0] = raw.shape[1]
        handle = neck_mod.register_forward_hook(probe)
        try:
            dev = next(v_enc.parameters()).device
            dummy = torch.zeros(1, 3, 256, 256, device=dev)
            v_enc.eval()
            with torch.no_grad():
                try: v_enc(dummy)
                except Exception: pass
        finally:
            handle.remove()
        dim = detected[0]
        return int(dim) if (dim and dim >= 32) else 256

    # ── Terrain hook sul neck ─────────────────────────────────────────────
    def _register_terrain_hook(self):
        v_enc    = self.backbone.vision_encoder
        hook_mod = getattr(v_enc, "neck", None)
        if hook_mod is None:
            for _, m in v_enc.named_modules():
                if isinstance(m, nn.Conv2d):
                    hook_mod = m
        if hook_mod is None:
            raise RuntimeError("Neck non trovato.")

        def _apply(t):
            if not isinstance(t, torch.Tensor) or t.ndim != 4:
                return t
            if self._z_terrain is None:
                return t
            try:
                return self.terrain_attn(t, self._z_terrain)
            except Exception:
                import traceback; traceback.print_exc()
                return t

        def terrain_hook(module, args, output):
            if self._z_terrain is None:
                return output
            if isinstance(output, torch.Tensor):
                return _apply(output)
            if isinstance(output, tuple):
                features = output[0]
                if isinstance(features, torch.Tensor) and features.ndim == 4:
                    return (_apply(features),) + output[1:]
                if isinstance(features, (list, tuple)) and len(features) > 0:
                    new_f = list(features)
                    sizes = [
                        f.shape[-1] * f.shape[-2]
                        if isinstance(f, torch.Tensor) and f.ndim == 4
                        else float("inf")
                        for f in features
                    ]
                    idx = sizes.index(min(sizes))
                    new_f[idx] = _apply(features[idx])
                    return (type(features)(new_f),) + output[1:]
            return output

        hook_mod.register_forward_hook(terrain_hook)

    # ── Patch embed ──────────────────────────────────────────────────────
    def _expand_patch_embed(self, in_channels: int) -> nn.Conv2d:
        v_enc = self.backbone.vision_encoder
        for name, module in v_enc.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                new_conv = nn.Conv2d(
                    in_channels, module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    bias=(module.bias is not None),
                )
                with torch.no_grad():
                    # Copia pesi RGB originali (canali 0-2)
                    new_conv.weight[:, :3, :, :] = module.weight.clone()
                    # Canali extra (3-13): rumore piccolo vicino a zero
                    # Evita shock iniziale mantenendo il contributo dei nuovi
                    # canali trascurabile all'inizio del training
                    nn.init.normal_(new_conv.weight[:, 3:, :, :], mean=0.0, std=0.001)
                    if module.bias is not None:
                        new_conv.bias.copy_(module.bias)
                parts  = name.split(".")
                parent = (
                    v_enc.get_submodule(".".join(parts[:-1]))
                    if len(parts) > 1 else v_enc
                )
                setattr(parent, parts[-1], new_conv)
                print(f"  Patch embed espanso: {name}  3 -> {in_channels} canali")
                return new_conv
        raise RuntimeError("Patch embed Conv2d(in=3) non trovato.")

    # ── Gradient checkpointing ────────────────────────────────────────────
    def _enable_gradient_checkpointing(self) -> int:
        try:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            n = sum(1 for m in self.backbone.modules()
                    if getattr(m, "gradient_checkpointing", False))
            if n: return n
        except Exception:
            pass
        return 0

    # ── Freeze ────────────────────────────────────────────────────────────
    def _freeze(self):
        # 1. Congela tutto il backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # 2. Scongela mask_decoder
        for p in self.backbone.mask_decoder.parameters():
            p.requires_grad = True

        # 3. Scongela patch embed espanso
        for p in self.patch_proj.parameters():
            p.requires_grad = True

        # 4. Scongela LoRA A e B
        for module in self.backbone.vision_encoder.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.requires_grad = True
                module.lora_B.requires_grad = True

        # 5. Scongela terrain stream (non fa parte di self.backbone)
        for p in self.terrain_encoder.parameters():
            p.requires_grad = True
        for p in self.terrain_attn.parameters():
            p.requires_grad = True

    def _print_trainable(self):
        tr  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in self.parameters())
        print(f"  Parametri trainabili: {tr:,} / {tot:,} ({100*tr/tot:.1f}%)")

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor, boxes: torch.Tensor):
        """
        x     : [B, 14, H, W]  — tutte le 14 bande L4S
        boxes : [B, 4]         — (x_min, y_min, x_max, y_max)
        """
        # Calcola terrain encoding (DEM+Slope, indici 12-13)
        terrain = x[:, 12:14, :, :]
        self._z_terrain = self.terrain_encoder(terrain)

        if boxes.ndim == 2:
            boxes = boxes.unsqueeze(1)

        out = self.backbone(
            pixel_values=x,      # 14 canali, patch embed gestisce l'espansione
            input_boxes=boxes,
            multimask_output=False,
        )

        self._z_terrain = None

        scores = None
        for attr in ("iou_scores", "iou_predictions", "pred_ious"):
            val = getattr(out, attr, None)
            if val is not None:
                scores = val; break

        return out.pred_masks, scores


# ============================================================
# 5. UTILS & LOSS
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
    # peso per classe sbilanciata (~3.6x più negativi che positivi)
    pos_weight = torch.tensor([3.6], device=y_pred.device)
    return (F.binary_cross_entropy_with_logits(y_pred, y_true_rs, pos_weight=pos_weight)
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
# 6. WRAPPER SUBSET
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
# 7. TRAINING LOOP
# ============================================================

def train():
    ckpt_path    = f"best_sam2tiny_lora_r{LORA_RANK}_l4s_alldataset.pth"
    history_file = f"history_sam2tiny_lora_r{LORA_RANK}_l4s_alldataset.json"

    grad_accum_steps = max(1, GRAD_ACCUM_TARGET // BATCH_SIZE)
    print(f"\nBatch fisico: {BATCH_SIZE} | Grad accum: {grad_accum_steps} | "
          f"Batch effettivo: {BATCH_SIZE * grad_accum_steps}")

    # Augmentation offline — solo trasformazioni rigide (fisicamente corrette)
    offline_aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])

    # On-the-fly — solo ToTensorV2, nessuna alterazione radiometrica
    train_trans = A.Compose([ToTensorV2()])
    val_trans   = A.Compose([ToTensorV2()])

    # Dataset e split
    full_ds = LandslideSAMDataset(DATASET_ROOT)
    n_neg = len(full_ds.negative_indices)
    n_pos = len(full_ds.positive_indices)
    POS_WEIGHT = n_neg / n_pos

    train_subset, val_subset, test_subset = build_train_val_test_subsets(
        full_ds, neg_fraction=NEG_FRACTION, val_fraction=VAL_FRACTION,
        test_fraction=TEST_FRACTION, random_seed=RANDOM_SEED,
    )

    # Augmentation offline sui positivi del train
    train_pos_indices = [
        i for i in train_subset.indices if i in set(full_ds.positive_indices)
    ]
    full_ds.add_augmented_samples(
        indices=train_pos_indices, transform=offline_aug,
        n_copies=AUG_COPIES, rng_seed=RANDOM_SEED,
    )
    aug_start   = len(full_ds.data) - len(train_pos_indices) * AUG_COPIES
    aug_indices = list(range(aug_start, len(full_ds.data)))
    train_subset = Subset(full_ds, list(train_subset.indices) + aug_indices)
    print(f"  Train finale: {len(train_subset)} campioni "
          f"({len(train_pos_indices)} originali + {len(aug_indices)} augmentati)")

    train_ds = TransformSubset(train_subset, train_trans)
    val_ds   = TransformSubset(val_subset,   val_trans)
    test_ds  = TransformSubset(test_subset,  val_trans)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = SAM2TinyLoRA(
        hf_path=HF_PATH, in_channels=IN_CHANNELS,
        lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA,
    ).to(DEVICE)

    # Learning rate differenziato:
    #   - patch_proj e terrain stream: LR alto (imparano quasi da zero)
    #   - LoRA: LR medio (modifica pesi pre-addestrati con delta piccoli)
    #   - mask_decoder: LR basso (gia' specializzato da SAM2)
    lora_params    = [p for n, p in model.named_parameters()
                      if p.requires_grad and ("lora_A" in n or "lora_B" in n)]
    decoder_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and "mask_decoder" in n]
    new_params     = [p for n, p in model.named_parameters()
                      if p.requires_grad
                      and "lora_A" not in n and "lora_B" not in n
                      and "mask_decoder" not in n]

    optimizer = optim.AdamW([
        {"params": new_params,     "lr": 1e-3,  "name": "patch+terrain"},
        {"params": lora_params,    "lr": 1e-4,  "name": "lora"},
        {"params": decoder_params, "lr": 3e-5,  "name": "mask_decoder"},
    ], weight_decay=1e-4)

    n_new  = sum(p.numel() for p in new_params)
    n_lora = sum(p.numel() for p in lora_params)
    n_dec  = sum(p.numel() for p in decoder_params)
    print(f"  Param groups:")
    print(f"    patch+terrain : {n_new:,}  lr=1e-3")
    print(f"    lora          : {n_lora:,}  lr=1e-4")
    print(f"    mask_decoder  : {n_dec:,}  lr=3e-5")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=5)
    scaler = torch.amp.GradScaler("cuda")

    start_epoch, best_iou = load_checkpoint(model, optimizer, scheduler, ckpt_path)

    history = []
    if os.path.exists(history_file):
        with open(history_file) as f:
            history = json.load(f)

    print(f"\nTraining SAM2-Tiny LoRA r={LORA_RANK} | "
          f"Epoche {start_epoch+1}/{EPOCHS} | Batch {BATCH_SIZE} | Device {DEVICE}\n")
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
            t_miou += miou;        t_f1  += f1;     t_prec += prec; t_rec += rec
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
# 8. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train()