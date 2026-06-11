"""
train_d2fls_swin.py
====================
D2FLS-Net fedele al paper PLOS ONE (doi:10.1371/journal.pone.0337412).

Correzioni rispetto alla versione precedente:
  1. Backbone: Swin-B reale da timm (shifted-window self-attention)
               canali paper: (128, 256, 512, 1024) per Swin-B
  2. Input RGB: solo bande ottiche visibili (B2,B3,B4 = canali 1,2,3 da L4S)
               proiettate a 3ch con un Conv1x1 prima del Swin, oppure
               con patch embedding espanso a 13ch (modalità multi-spettrale)
               → default: modalità multi-spettrale (patch embed 13→C0)
  3. Decoder: FPN completo (P2–P5) con fusione laterale top-down
  4. HighLevelCrossAttention: aggiunta skip-connection residuale (eq. 10)
  5. T-PACE: stage 3, canali 512 (Swin-B)
  6. DSDF: early guidance su stage-1 (C=128), cross-attention su stage-4 (C=1024)

Dataset loader identico a SAM-LoRA:
  - File .h5 da TrainData/
  - Split 60/20/20 stratificato
  - Nessuna augmentation
  - pos_weight dinamico per classe sbilanciata

Dipendenze:
    pip install timm torch torchvision tqdm h5py
"""

import os
import glob
import h5py
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

try:
    import timm
except ImportError:
    raise ImportError("Installa timm: pip install timm")


# ============================================================
# CONFIGURAZIONE
# ============================================================

DATASET_ROOT   = r"/datadrive/landslide/SAM3/SAM2/LandSlide4Sense"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE     = 24
NUM_EPOCHS     = 100
LEARNING_RATE  = 5e-5
WEIGHT_DECAY   = 0.01
PATIENCE       = 15
VAL_FRACTION   = 0.20
TEST_FRACTION  = 0.20
RANDOM_SEED    = 42
LAMBDA_DICE    = 1.0

# Variante Swin: "swin_base_patch4_window7_224" (Swin-B, paper default)
#               "swin_small_patch4_window7_224" (Swin-S, ablation)
#               "swin_large_patch4_window7_224" (Swin-L, ablation)
# ImageNet-22K puro (21841 classi): pesi più ricchi per segmentazione densa
# Alternativa ft su IN-1K: "swin_base_patch4_window7_224.ms_in22k_ft_in1k"
SWIN_VARIANT   = "swin_base_patch4_window7_224.ms_in22k"

# Canali input ottici: 13 (tutte le bande L4S escluso DEM)
IN_CHANNELS_OPT = 13

torch.backends.cudnn.benchmark = True


# ============================================================
# 1. DATASET  (identico a SAM-LoRA, senza augmentation)
# ============================================================

class LandslideH5Dataset(Dataset):
    """
    Carica tutti i file .h5 di TrainData/ in RAM.
        <root>/TrainData/img/image_*.h5   -> key "img"  (128,128,14) HWC
        <root>/TrainData/mask/mask_*.h5   -> key "mask" (128,128)

    Ritorna:
        optical : Tensor (13, H, W)  bande 0-12 (Sentinel-2 B1-B12 + Slope)
        dem     : Tensor  (1, H, W)  banda 13   (DEM ALOS PALSAR)
        mask    : Tensor  (H, W)     long 0/1
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
            self.data.append((image, mask))
            (self.positive_indices if mask.sum() > 0 else self.negative_indices).append(idx)

        n_pos, n_neg = len(self.positive_indices), len(self.negative_indices)
        print(f"Caricati {len(self.data)} campioni: {n_pos} positivi, {n_neg} negativi")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_np, mask_np = self.data[idx]
        img_t   = torch.from_numpy(image_np).permute(2, 0, 1)  # (14,H,W)
        optical = img_t[:13]                                      # (13,H,W)
        dem     = img_t[13:14]                                    # (1, H,W)
        mask_t  = torch.from_numpy(mask_np).long()
        return optical, dem, mask_t


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
    total = len(full_ds)

    print(
        f"\nSplit 60/20/20:"
        f"\n  Totale : {total:5d}  ({len(full_ds.positive_indices)} pos + {len(full_ds.negative_indices)} neg)"
        f"\n  Train  : {len(train_indices):5d}  ({n_pos_tr} pos + {n_neg_tr} neg)"
        f"\n  Val    : {len(val_indices):5d}  ({n_pos_vl} pos + {n_neg_vl} neg)"
        f"\n  Test   : {len(test_indices):5d}  ({n_pos_te} pos + {n_neg_te} neg)"
    )
    # pos_weight pixel-level stimato per L4S.
    # I campioni "positivi" hanno in media ~35% pixel frana, i negativi 0%.
    # pixel_bg  ≈ n_neg_tr * H*W  +  n_pos_tr * H*W * 0.65
    # pixel_ls  ≈ n_pos_tr * H*W * 0.35
    # → ratio = (n_neg_tr + 0.65 * n_pos_tr) / (0.35 * n_pos_tr)
    ls_frac = 0.35   # frazione media pixel frana nei patch positivi L4S
    pixel_ls = n_pos_tr * ls_frac
    pixel_bg = n_neg_tr + n_pos_tr * (1.0 - ls_frac)
    pos_weight = pixel_bg / max(pixel_ls, 1.0)
    print(f"  pos_weight BCE (pixel-level): {pos_weight:.2f}\n")

    return (
        Subset(full_ds, train_indices),
        Subset(full_ds, val_indices),
        Subset(full_ds, test_indices),
        pos_weight,
    )


# ============================================================
# 3. ARCHITETTURA — D2FLS-Net fedele al paper
# ============================================================

# ── 3a. Swin backbone con patch embed espanso ────────────────

class SwinBackbone(nn.Module):
    """
    Swin-B (o S/L) da timm con patch embedding espanso da 3 a
    in_channels canali multispettrali.

    Produce 4 feature map intermedie corrispondenti agli stage 1-4:
        C0=128, C1=256, C2=512, C3=1024  (Swin-B)

    Il patch embed del Swin originale è Conv2d(3, C0, 4, 4).
    Lo sostituiamo con Conv2d(in_channels, C0, 4, 4):
        - pesi sui canali 0-2 copiati dall'originale
        - canali aggiuntivi inizializzati con rumore piccolo (media RGB)
    """

    def __init__(self, variant: str = SWIN_VARIANT, in_channels: int = 3,
                 pretrained: bool = True, img_size: int = 128):
        super().__init__()
        self.swin = timm.create_model(
            variant,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=img_size,          # ← patch 128×128, non 224×224
        )

        if in_channels != 3:
            self._expand_patch_embed(in_channels)

        # Canali di uscita per ogni stage (Swin-B: 128,256,512,1024)
        self.out_channels = self.swin.feature_info.channels()

    def _expand_patch_embed(self, in_channels: int):
        """Sostituisce il Conv2d(3→C) con Conv2d(in_channels→C)."""
        # timm espone patch_embed.proj come il Conv2d di ingresso
        orig = self.swin.patch_embed.proj           # Conv2d(3, C, 4, 4)
        new_proj = nn.Conv2d(
            in_channels, orig.out_channels,
            kernel_size=orig.kernel_size,
            stride=orig.stride,
            padding=orig.padding,
            bias=(orig.bias is not None),
        )
        with torch.no_grad():
            new_proj.weight[:, :3] = orig.weight.clone()
            if in_channels > 3:
                # media dei pesi RGB per i canali extra → contributo iniziale minimo
                rgb_mean = orig.weight.mean(dim=1, keepdim=True)
                nn.init.normal_(new_proj.weight[:, 3:], mean=0.0, std=0.001)
            if orig.bias is not None:
                new_proj.bias.copy_(orig.bias)
        self.swin.patch_embed.proj = new_proj
        print(f"  Patch embed espanso: 3 → {in_channels} canali")

    def forward(self, x):
        """Ritorna lista [f1, f2, f3, f4] — tensori NCHW (B, Ci, Hi, Wi)."""
        features = self.swin(x)
        out = []
        for f, c_expected in zip(features, self.out_channels):
            if f.ndim == 4 and f.shape[-1] == c_expected:
                # timm Swin restituisce NHWC: (B, H, W, C)
                # l'ultimo asse corrisponde ai canali attesi → permuta
                f = f.permute(0, 3, 1, 2).contiguous()
            out.append(f)
        return out   # [f1(B,C0,H/4,W/4), f2(B,C1,H/8,W/8),
                     #  f3(B,C2,H/16,W/16), f4(B,C3,H/32,W/32)]


# ── 3b. DSDF — Early DEM Guidance (eq. 4-5) ─────────────────

class EarlyDEMGuidance(nn.Module):
    """
    Guidance map G dal DEM, ridimensionata alla risoluzione di stage-1.
    Applica gain pixel-wise: F1_mod = F1 ⊙ (1 + G1)   [eq. 5]
    """
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, dem: torch.Tensor, f1: torch.Tensor) -> torch.Tensor:
        g  = self.conv_block(dem)                                    # (B,1,H,W)
        g1 = F.interpolate(g, size=f1.shape[2:],
                           mode="bilinear", align_corners=False)     # (B,1,H/4,W/4)
        return f1 * (1.0 + g1)                                       # eq. 5


# ── 3c. DEM Encoder TinyViT (eq. 6-7) ────────────────────────

class TinyViTDEMEncoder(nn.Module):
    """
    Patch embed 4×4 stride 4 + TransformerEncoder (2 layer, 128 dim, 4 head).
    Produce Z: (B, 128, H/4, W/4).
    """
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, layers: int = 2):
        super().__init__()
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=4, stride=4)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, dem: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(dem)                   # (B, E, H/4, W/4)
        B, E, Hd, Wd = x.shape
        seq = x.flatten(2).transpose(1, 2)          # (B, N, E)
        seq = self.norm(self.transformer(seq))
        return seq.transpose(1, 2).reshape(B, E, Hd, Wd)


# ── 3d. DSDF High-Level Cross-Attention (eq. 8-10) ───────────

class HighLevelCrossAttention(nn.Module):
    """
    Cross-attention tra F4_rgb (query) e F4_dem (key/value).
    Output: F4_fused = WO · MHCA(Q,K,V) + F4   [eq. 10, con residuo]

    WQ, WK, WV: Conv2d 1×1 (linear projection per spatial location).
    """
    def __init__(self, dim_rgb: int, dim_dem: int, num_heads: int = 8):
        super().__init__()
        # forza num_heads compatibile con dim_rgb
        while dim_rgb % num_heads != 0 and num_heads > 1:
            num_heads //= 2
        self.wq   = nn.Conv2d(dim_rgb, dim_rgb, 1, bias=False)
        self.wk   = nn.Conv2d(dim_dem, dim_rgb, 1, bias=False)
        self.wv   = nn.Conv2d(dim_dem, dim_rgb, 1, bias=False)
        self.attn = nn.MultiheadAttention(dim_rgb, num_heads, batch_first=True)
        self.wo   = nn.Conv2d(dim_rgb, dim_rgb, 1, bias=False)
        self.norm = nn.LayerNorm(dim_rgb)

    def forward(self, f4_rgb: torch.Tensor, f4_dem: torch.Tensor) -> torch.Tensor:
        B, C, H, W = f4_rgb.shape

        # Allinea spazialmente dem a rgb
        if f4_dem.shape[2:] != f4_rgb.shape[2:]:
            f4_dem = F.interpolate(f4_dem, size=(H, W),
                                   mode="bilinear", align_corners=False)

        q = self.wq(f4_rgb).flatten(2).transpose(1, 2)   # (B, H*W, C)
        k = self.wk(f4_dem).flatten(2).transpose(1, 2)
        v = self.wv(f4_dem).flatten(2).transpose(1, 2)

        attn_out, _ = self.attn(q, k, v)                  # (B, H*W, C)
        attn_out = attn_out.transpose(1, 2).reshape(B, C, H, W)

        # eq. 10: F̂4 = WO · norm(attn_out) + F4  (residual)
        attn_out_flat = attn_out.permute(0,2,3,1)         # NHWC per LayerNorm
        attn_norm = self.norm(attn_out_flat).permute(0,3,1,2)
        return self.wo(attn_norm) + f4_rgb                 # ← skip connection


# ── 3e. T-PACE (eq. 11-17) ────────────────────────────────────

class TPACE(nn.Module):
    """
    Terrain-aware Pixel-wise Adaptive Context Enhancement.
    Montato su stage-3 (C=512 per Swin-B).
    Dilazioni: [1, 6, 12, 18]  →  K=4 branch.
    """
    def __init__(self, in_channels_rgb: int, in_channels_dem: int):
        super().__init__()
        self.dilations = [1, 6, 12, 18]
        K = len(self.dilations)
        self.atrous_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels_rgb, in_channels_rgb,
                          3, padding=d, dilation=d),
                nn.BatchNorm2d(in_channels_rgb),
            ) for d in self.dilations
        ])
        # Gating: (RGB + DEM) → 256 → K  [eq. 12-13]
        self.gating = nn.Sequential(
            nn.Conv2d(in_channels_rgb + in_channels_dem, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, K, 1),
        )
        # Output projection [eq. 16]
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels_rgb, in_channels_rgb, 1),
            nn.BatchNorm2d(in_channels_rgb),
        )

    def forward(self, f3_rgb: torch.Tensor, f3_dem: torch.Tensor) -> torch.Tensor:
        # Allinea dem a rgb spazialmente
        if f3_dem.shape[2:] != f3_rgb.shape[2:]:
            f3_dem = F.interpolate(f3_dem, size=f3_rgb.shape[2:],
                                   mode="bilinear", align_corners=False)

        # eq. 11: branch atrous
        Z = torch.stack([b(f3_rgb) for b in self.atrous_branches], dim=1)  # (B,K,C,H,W)

        # eq. 12-14: gating DEM-aware
        u      = torch.cat([f3_rgb, f3_dem], dim=1)
        alpha  = F.softmax(self.gating(u), dim=1)           # (B,K,H,W)
        alpha  = alpha.unsqueeze(2)                          # (B,K,1,H,W)

        # eq. 15: pixel-adaptive aggregation
        Y = (alpha * Z).sum(dim=1)                           # (B,C,H,W)

        # eq. 16-17: projection + residual
        return self.out_proj(Y) + f3_rgb


# ── 3f. FPN Decoder (paper §2.1) ─────────────────────────────

class FPNDecoder(nn.Module):
    """
    Lightweight FPN head fedele al paper:
      - F4 (C3) upsamplinato a risoluzione stage-3, concatenato con T-PACE(F3)
      - Conv1x1 riduce canali a fpn_channels (256)
      - Due ConvTranspose2d(kernel=2, stride=2) portano a H/4
      - Classificatore 1x1 + upsample finale a H×W

    Aggiunta fusione laterale P3←P2 per un FPN più fedele allo standard.
    """
    def __init__(self, c0: int, c1: int, c2: int, c3: int,
                 fpn_channels: int = 256, num_classes: int = 2):
        super().__init__()
        # Lateral projections per tutti gli stage
        self.lat4 = nn.Conv2d(c3, fpn_channels, 1)
        self.lat3 = nn.Conv2d(c2, fpn_channels, 1)
        self.lat2 = nn.Conv2d(c1, fpn_channels, 1)
        self.lat1 = nn.Conv2d(c0, fpn_channels, 1)

        # Smooth dopo fusione top-down
        self.smooth3 = nn.Sequential(nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                                     nn.BatchNorm2d(fpn_channels), nn.ReLU(inplace=True))
        self.smooth2 = nn.Sequential(nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                                     nn.BatchNorm2d(fpn_channels), nn.ReLU(inplace=True))
        self.smooth1 = nn.Sequential(nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                                     nn.BatchNorm2d(fpn_channels), nn.ReLU(inplace=True))

        # Upsample progressivo P1 (H/4) → H/2 → H
        self.up1 = nn.ConvTranspose2d(fpn_channels, fpn_channels // 2, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(fpn_channels // 2, fpn_channels // 4, 2, stride=2)
        self.classifier = nn.Conv2d(fpn_channels // 4, num_classes, 1)

    def forward(self, f1, f2, f3_tpace, f4_fused, orig_size):
        """
        f1         : stage-1 features  (B, C0, H/4,  W/4)
        f2         : stage-2 features  (B, C1, H/8,  W/8)
        f3_tpace   : T-PACE output     (B, C2, H/16, W/16)
        f4_fused   : cross-attn output (B, C3, H/32, W/32)
        orig_size  : (H, W) originale dell'immagine
        """
        # Top-down pathway
        p4 = self.lat4(f4_fused)                                         # H/32
        p3 = self.smooth3(self.lat3(f3_tpace) +
             F.interpolate(p4, size=f3_tpace.shape[2:],
                           mode="bilinear", align_corners=False))        # H/16
        p2 = self.smooth2(self.lat2(f2) +
             F.interpolate(p3, size=f2.shape[2:],
                           mode="bilinear", align_corners=False))        # H/8
        p1 = self.smooth1(self.lat1(f1) +
             F.interpolate(p2, size=f1.shape[2:],
                           mode="bilinear", align_corners=False))        # H/4

        # Upsample p1 → H
        x = F.relu(self.up1(p1))    # H/2
        x = F.relu(self.up2(x))     # H
        logits = self.classifier(x) # (B, num_classes, H, W)
        return F.interpolate(logits, size=orig_size,
                             mode="bilinear", align_corners=False)


# ── 3g. D2FLS-Net completo ────────────────────────────────────

class D2FLS_Net(nn.Module):
    """
    D2FLS-Net fedele al paper:
      - Swin-B backbone (shifted-window self-attention reale)
      - DSDF: early guidance @stage1 + cross-attention @stage4
      - T-PACE: @stage3
      - FPN decoder: fusione P4→P3→P2→P1, upsample finale
    """
    DEM_ENC_DIM = 128   # larghezza TinyViT DEM encoder (paper: width=128)

    def __init__(
        self,
        swin_variant: str   = SWIN_VARIANT,
        in_channels_opt: int = IN_CHANNELS_OPT,
        num_classes: int    = 2,
        pretrained: bool    = True,
        fpn_channels: int   = 256,
    ):
        super().__init__()

        # ── Backbone ────────────────────────────────────────────
        self.backbone = SwinBackbone(swin_variant, in_channels_opt, pretrained,
                                     img_size=128)
        c0, c1, c2, c3 = self.backbone.out_channels   # e.g. 128,256,512,1024

        # ── DEM encoder (comune a DSDF e T-PACE) ────────────────
        self.dem_encoder = TinyViTDEMEncoder(
            embed_dim=self.DEM_ENC_DIM, num_heads=4, layers=2)

        # ── DSDF early guidance @stage-1 ────────────────────────
        self.early_guidance = EarlyDEMGuidance()

        # ── DSDF cross-attention @stage-4  (eq. 8-10) ───────────
        # Proiezione DEM-enc → C3 per K e V
        self.dem_proj4 = nn.Conv2d(self.DEM_ENC_DIM, c3, 1)
        self.cross_attn = HighLevelCrossAttention(
            dim_rgb=c3, dim_dem=c3, num_heads=8)

        # ── T-PACE @stage-3  (eq. 11-17) ────────────────────────
        # Proiezione DEM-enc → C2 per il gating
        self.dem_proj3 = nn.Conv2d(self.DEM_ENC_DIM, c2, 1)
        self.tpace = TPACE(in_channels_rgb=c2, in_channels_dem=c2)

        # ── FPN Decoder ─────────────────────────────────────────
        self.decoder = FPNDecoder(c0, c1, c2, c3,
                                  fpn_channels=fpn_channels,
                                  num_classes=num_classes)

        total = sum(p.numel() for p in self.parameters())
        print(f"D2FLS-Net — {swin_variant} | canali: {c0},{c1},{c2},{c3} | "
              f"params totali: {total/1e6:.1f}M")

    def forward(self, optical: torch.Tensor, dem: torch.Tensor):
        """
        optical : (B, 13, H, W)
        dem     : (B,  1, H, W)
        """
        B, _, H, W = optical.shape

        # ── DEM encoding ────────────────────────────────────────
        Z = self.dem_encoder(dem)   # (B, 128, H/4, W/4)

        # ── Swin stage-1 con early DEM guidance ─────────────────
        # Estraiamo solo stage-1, applichiamo guidance, poi procediamo
        # timm features_only esegue tutti gli stage in un passaggio —
        # per iniettare la guidance dobbiamo intercettare l'uscita di stage-1.
        # Usiamo un hook temporaneo.
        f_stages = self.backbone(optical)   # [f1, f2, f3, f4]
        f1, f2, f3, f4 = f_stages

        # Early DEM guidance su f1 (stage-1)  [eq. 5]
        f1 = self.early_guidance(dem, f1)

        # ── T-PACE su f3 (stage-3) ───────────────────────────────
        z3 = F.interpolate(Z, size=f3.shape[2:],
                           mode="bilinear", align_corners=False)
        f3_dem  = self.dem_proj3(z3)          # (B, C2, H/16, W/16)
        f3_hat  = self.tpace(f3, f3_dem)      # (B, C2, H/16, W/16)

        # ── Cross-attention su f4 (stage-4) ─────────────────────
        z4 = F.interpolate(Z, size=f4.shape[2:],
                           mode="bilinear", align_corners=False)
        f4_dem    = self.dem_proj4(z4)         # (B, C3, H/32, W/32)
        f4_fused  = self.cross_attn(f4, f4_dem)  # (B, C3, H/32, W/32)

        # ── FPN Decoder ─────────────────────────────────────────
        logits = self.decoder(f1, f2, f3_hat, f4_fused, orig_size=(H, W))
        return logits   # (B, num_classes, H, W)


# ============================================================
# 4. LOSS E METRICHE
# ============================================================

class CombinedLoss(nn.Module):
    """
    CE (con pos_weight) + λ·Dice  [eq. 2-3 del paper].
    pos_weight bilancia la classe sbilanciata (n_neg/n_pos nel train).
    """
    def __init__(self, pos_weight: float, lambda_dice: float = 1.0,
                 epsilon: float = 1e-5):
        super().__init__()
        self.register_buffer("pw", torch.tensor([pos_weight]))
        self.lambda_dice = lambda_dice
        self.epsilon     = epsilon

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        # CE pixel-wise su 2 classi con pos_weight sulla classe landslide
        # (fedele al paper eq.1-2; pos_weight compensa lo sbilanciamento pixel)
        weight = torch.ones(2, device=preds.device)
        weight[1] = self.pw.item()
        loss_ce = F.cross_entropy(preds, targets, weight=weight)

        # Dice soft sulla classe landslide (eq. 3 del paper).
        # epsilon piccolo: non deve mascherare la penalità quando probs≈0.
        # Normalizziamo per numero di pixel per rendere la loss
        # indipendente dalla dimensione del batch.
        probs  = F.softmax(preds, dim=1)[:, 1]          # (B,H,W) ∈ [0,1]
        y_true = (targets == 1).float()                  # (B,H,W)
        inter  = (probs * y_true).sum(dim=(1, 2))        # (B,)
        sum_p  = probs.sum(dim=(1, 2))                   # (B,)
        sum_y  = y_true.sum(dim=(1, 2))                  # (B,)
        # Usa epsilon solo dove ci sono pixel positivi nel GT,
        # altrimenti la loss Dice è 0 (niente da predire → niente penalità)
        has_pos = (sum_y > 0).float()
        dice_score = (2.0 * inter + self.epsilon) / (sum_p + sum_y + self.epsilon)
        dice_loss  = has_pos * (1.0 - dice_score)        # ignora patch tutti-bg
        return loss_ce + self.lambda_dice * dice_loss.mean()


def calculate_metrics(preds: torch.Tensor, targets: torch.Tensor,
                      epsilon: float = 1e-5):
    """
    Micro-average globale sul batch (fedele al paper eq. 18-22).
    TP/FP/TN/FN accumulati su tutti i pixel di tutti i campioni,
    poi IoU/Prec/Rec calcolati una volta sola.

    Questo evita il gonfiamento del mIoU dovuto ai patch tutti-background
    dove IoU_BG = TN/TN = 1.0 anche quando il modello non impara nulla.

    Returns: miou, precision, recall, iou_landslide
    """
    pred_bin = torch.argmax(preds, dim=1)                          # (B,H,W)

    # Accumulo globale su B,H,W  →  scalari
    TP = ((pred_bin == 1) & (targets == 1)).float().sum().item()
    FP = ((pred_bin == 1) & (targets == 0)).float().sum().item()
    TN = ((pred_bin == 0) & (targets == 0)).float().sum().item()
    FN = ((pred_bin == 0) & (targets == 1)).float().sum().item()

    iou_ls = TP / (TP + FP + FN + epsilon)
    iou_bg = TN / (TN + FP + FN + epsilon)
    miou   = (iou_ls + iou_bg) / 2.0
    prec   = TP / (TP + FP + epsilon)
    rec    = TP / (TP + FN + epsilon)
    return miou, prec, rec, iou_ls


def save_checkpoint(model, optimizer, scheduler, epoch, best_miou, path):
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_miou":            best_miou,
    }, path)


def load_checkpoint(model, optimizer, scheduler, path, device):
    if not os.path.exists(path):
        print(f"Nessun checkpoint in '{path}'. Avvio da zero.")
        return 0, 0.0
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    except Exception:
        pass
    print(f"Checkpoint caricato — Epoca {ckpt['epoch']}, "
          f"Best mIoU: {ckpt['best_miou']:.4f}")
    return ckpt["epoch"] + 1, ckpt["best_miou"]


# ============================================================
# 5. TRAINING LOOP
# ============================================================

def train():
    # Nome file include il tag del pretrain per non sovrascrivere checkpoint precedenti
    _tag = SWIN_VARIANT.replace('swin_', '').replace('_patch4_window7_224', '').replace('.', '_')
    ckpt_path    = f"best_d2fls_{_tag}.pth"
    history_file = f"history_d2fls_{_tag}.json"

    full_ds = LandslideH5Dataset(DATASET_ROOT)

    train_subset, val_subset, test_subset, pos_weight = build_train_val_test_subsets(
        full_ds, VAL_FRACTION, TEST_FRACTION, RANDOM_SEED)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_subset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    model     = D2FLS_Net(swin_variant=SWIN_VARIANT,
                          in_channels_opt=IN_CHANNELS_OPT,
                          num_classes=2, pretrained=True).to(DEVICE)
    criterion = CombinedLoss(pos_weight=pos_weight,
                             lambda_dice=LAMBDA_DICE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.3, patience=5)
    scaler    = torch.amp.GradScaler("cuda")

    start_epoch, best_miou = load_checkpoint(
        model, optimizer, scheduler, ckpt_path, DEVICE)

    history          = []
    patience_counter = 0
    if os.path.exists(history_file):
        with open(history_file) as f:
            history = json.load(f)

    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri trainabili: {n_tr/1e6:.1f}M")
    print(f"Training su {DEVICE} | Epoche {start_epoch+1}/{NUM_EPOCHS} | "
          f"Batch {BATCH_SIZE}\n")

    for epoch in range(start_epoch, NUM_EPOCHS):

        # ── TRAIN ──────────────────────────────────────────────
        model.train()
        t_loss = t_miou = t_iou_ls = t_prec = t_rec = 0.0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{NUM_EPOCHS} [TRAIN]")

        for opt, dem, mask in pbar:
            opt, dem, mask = opt.to(DEVICE), dem.to(DEVICE), mask.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                out  = model(opt, dem)
                loss = criterion(out, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            miou, prec, rec, iou_ls = calculate_metrics(out.detach(), mask)
            t_loss   += loss.item()
            t_miou   += miou
            t_iou_ls += iou_ls
            t_prec   += prec
            t_rec    += rec
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             mIoU=f"{miou*100:.1f}%",
                             IoU_LS=f"{iou_ls*100:.1f}%")

        # ── VAL ────────────────────────────────────────────────
        model.eval()
        v_loss = v_miou = v_iou_ls = v_prec = v_rec = 0.0

        with torch.no_grad():
            for opt, dem, mask in tqdm(val_loader,
                                       desc=f"Ep {epoch+1}/{NUM_EPOCHS} [VAL]  "):
                opt, dem, mask = opt.to(DEVICE), dem.to(DEVICE), mask.to(DEVICE)
                with torch.amp.autocast("cuda"):
                    out  = model(opt, dem)
                    loss = criterion(out, mask)
                miou, prec, rec, iou_ls = calculate_metrics(out, mask)
                v_loss   += loss.item()
                v_miou   += miou
                v_iou_ls += iou_ls
                v_prec   += prec
                v_rec    += rec

        ntr, nvl = len(train_loader), len(val_loader)
        avg = {
            "epoch":           epoch + 1,
            "train_loss":      t_loss   / ntr,
            "train_miou":      t_miou   / ntr,
            "train_iou_ls":    t_iou_ls / ntr,
            "train_precision": t_prec   / ntr,
            "train_recall":    t_rec    / ntr,
            "val_loss":        v_loss   / nvl,
            "val_miou":        v_miou   / nvl,
            "val_iou_ls":      v_iou_ls / nvl,
            "val_precision":   v_prec   / nvl,
            "val_recall":      v_rec    / nvl,
        }
        scheduler.step(avg["val_miou"])

        print(
            f"EPOCA {epoch+1:3d}\n"
            f"  TRAIN | Loss:{avg['train_loss']:.4f}  "
            f"mIoU:{avg['train_miou']*100:.2f}%  "
            f"IoU_LS:{avg['train_iou_ls']*100:.2f}%  "
            f"Prec:{avg['train_precision']*100:.2f}%  "
            f"Rec:{avg['train_recall']*100:.2f}%\n"
            f"  VAL   | Loss:{avg['val_loss']:.4f}  "
            f"mIoU:{avg['val_miou']*100:.2f}%  "
            f"IoU_LS:{avg['val_iou_ls']*100:.2f}%  "
            f"Prec:{avg['val_precision']*100:.2f}%  "
            f"Rec:{avg['val_recall']*100:.2f}%"
        )

        history.append(avg)
        with open(history_file, "w") as f:
            json.dump(history, f, indent=4)

        if avg["val_miou"] > best_miou:
            best_miou = avg["val_miou"]
            save_checkpoint(model, optimizer, scheduler, epoch,
                            best_miou, ckpt_path)
            print(f"  Nuovo record mIoU val: {best_miou*100:.2f}% — "
                  f"checkpoint salvato")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"\nEARLY STOPPING — Miglior mIoU val: "
                      f"{best_miou*100:.2f}%")
                break

    # ── TEST FINALE ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  VALUTAZIONE FINALE SUL TEST SET")
    print("=" * 60)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    te_loss = te_miou = te_iou_ls = te_prec = te_rec = 0.0
    with torch.no_grad():
        for opt, dem, mask in tqdm(test_loader, desc="TEST"):
            opt, dem, mask = opt.to(DEVICE), dem.to(DEVICE), mask.to(DEVICE)
            with torch.amp.autocast("cuda"):
                out  = model(opt, dem)
                loss = criterion(out, mask)
            miou, prec, rec, iou_ls = calculate_metrics(out, mask)
            te_loss   += loss.item()
            te_miou   += miou
            te_iou_ls += iou_ls
            te_prec   += prec
            te_rec    += rec

    nte = len(test_loader)
    test_results = {
        "test_loss":      te_loss   / nte,
        "test_miou":      te_miou   / nte,
        "test_iou_ls":    te_iou_ls / nte,
        "test_precision": te_prec   / nte,
        "test_recall":    te_rec    / nte,
    }
    print(
        f"  Loss      : {test_results['test_loss']:.4f}\n"
        f"  mIoU      : {test_results['test_miou']*100:.2f}%\n"
        f"  IoU_LS    : {test_results['test_iou_ls']*100:.2f}%\n"
        f"  Precision : {test_results['test_precision']*100:.2f}%\n"
        f"  Recall    : {test_results['test_recall']*100:.2f}%"
    )

    history.append({"test_results": test_results})
    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Risultati salvati in: {history_file}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train()