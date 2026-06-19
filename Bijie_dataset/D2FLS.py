"""
train_d2fls_bijie.py
====================
D2FLS-Net (Swin-B + DSDF + T-PACE + FPN) adattato al Bijie Landslide Dataset.

Differenze rispetto alla versione Landslide4Sense:
  - Dataset: PNG (image RGB, mask, DEM) da Bijie-landslide-dataset
    invece di .h5. Ottico = RGB (3 canali), DEM = 1 canale (input separato).
  - IN_CHANNELS_OPT = 3  ->  il patch embed Swin NON viene espanso (resta 3->C0),
    ma usa i pesi RGB pre-addestrati nativi.
  - NIENTE slope derivata: a differenza del DSDF di SAM2, qui il terreno entra
    attraverso il DEM encoder TinyViT dedicato (DSDF early guidance + cross-attn
    + gating T-PACE), quindi il DEM grezzo a 1 canale e' sufficiente.
  - Immagini di dimensione variabile -> resize a LOAD_SIZE in caricamento.
    img_size del Swin = LOAD_SIZE per coerenza con la position embedding.
  - I campioni non-landslide non hanno mask -> maschera vuota.
  - pos_weight dai pixel reali del train (non l'euristica 0.35 di L4S).

Struttura attesa del dataset:
    <root>/landslide/img|image/   df001.png ...   (RGB)
    <root>/landslide/mask/        df001_mask.png  (binaria)
    <root>/landslide/dem/         df001_dem.png
    <root>/non-landslide/img/     ...
    <root>/non-landslide/dem/     ...

Dipendenze:
    pip install timm torch torchvision tqdm pillow
"""

import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from PIL import Image

try:
    import timm
except ImportError:
    raise ImportError("Installa timm: pip install timm")


# ============================================================
# CONFIGURAZIONE
# ============================================================

DATASET_ROOT   = r"/datadrive/LENTE/Bijie_dataset/Bijie-landslide-dataset"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE     = 24
NUM_EPOCHS     = 100
LEARNING_RATE  = 5e-5
WEIGHT_DECAY   = 0.01
PATIENCE       = 15
VAL_FRACTION   = 0.20
TEST_FRACTION  = 0.20
RANDOM_SEED    = 42

# --- Loss (fedele al paper) -------------------------------------------------
# Il paper usa CROSS-ENTROPY pixel-wise come loss principale (eq. 1) e dichiara
# la Dice come OPZIONALE ("we optionally combine LCE with Dice", eq. 2-3).
# Default fedele: solo CE, senza pos_weight. Attiva la Dice / il pos_weight
# solo se il tuo split risulta molto sbilanciato.
USE_DICE       = False   # True -> CE + LAMBDA_DICE * Dice (eq. 2-3)
LAMBDA_DICE    = 1.0     # usato solo se USE_DICE = True
USE_POS_WEIGHT = False   # True -> CE pesata sulla classe frana (non nel paper)

LOAD_SIZE      = 256   # Bijie: 0.8 m GSD, ritagli di dim. variabile -> uniformati qui.
                       # img_size del Swin viene allineato a questo valore.
                       # (il paper non dichiara la dim.; 256 e' piu' adatto di 128 a
                       #  questa risoluzione. Abbassa a 128 se la GPU non regge.)

# Variante Swin. NOTA sui canali: il paper dichiara canali (96,192,384,768),
# che corrispondono a embed_dim=96 (Swin-T/S). "swin_base" in timm ha invece
# embed_dim=128 -> (128,256,512,1024). C'e' un'incoerenza nel paper (cita
# "Swin-B" come migliore ma elenca canali da embed_dim=96). Il codice legge i
# canali dinamicamente da feature_info, quindi funziona con qualunque variante.
# Per i canali ESATTI del paper usa una variante embed_dim=96:
#   "swin_small_patch4_window7_224.ms_in22k"   (96,192,384,768)
# Per il "Swin-B" nominale (miglior mIoU dichiarato su Bijie):
#   "swin_base_patch4_window7_224.ms_in22k"    (128,256,512,1024)
SWIN_VARIANT   = "swin_base_patch4_window7_224.ms_in22k"

# Bijie: ottico = RGB (3 canali). Il DEM e' input separato a 1 canale.
IN_CHANNELS_OPT = 3

torch.backends.cudnn.benchmark = True


# ============================================================
# 1. DATASET  (PNG Bijie -> optical 3ch + dem 1ch + mask)
# ============================================================

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


class LandslideH5Dataset(Dataset):
    """
    Nome mantenuto per compatibilita', ma carica il Bijie dataset (PNG).
    Tutto in RAM come uint8, resize a LOAD_SIZE.

    Ritorna (come la versione L4S, stesso ordine):
        optical : Tensor (3, H, W)  RGB in [0,1]
        dem     : Tensor (1, H, W)  DEM in [0,1]
        mask    : Tensor (H, W)     long 0/1
    """
    DIR_CANDIDATES = {
        "landslide":     ["landslide", "Landslide"],
        "non_landslide": ["non-landslide", "non_landslide", "Non-landslide", "nonlandslide"],
        "img":           ["img", "image", "images", "Image"],
        "mask":          ["mask", "masks", "label", "labels", "Mask"],
        "dem":           ["dem", "DEM", "dems"],
    }

    def __init__(self, root_dir: str, load_size: int = LOAD_SIZE):
        self.load_size = load_size
        ls_dir  = _find_subdir(root_dir, self.DIR_CANDIDATES["landslide"])
        nls_dir = _find_subdir(root_dir, self.DIR_CANDIDATES["non_landslide"])
        if ls_dir is None:
            raise FileNotFoundError(
                f"Cartella 'landslide' non trovata in {root_dir}. "
                f"Contenuto: {os.listdir(root_dir)}")

        print(f"Caricamento Bijie in RAM (load_size={load_size})...")
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
                f"Sottocartelle img/dem non trovate in {group_dir}. "
                f"Contenuto: {os.listdir(group_dir)}")
        if has_mask and mask_dir is None:
            raise FileNotFoundError(f"Cartella mask non trovata in {group_dir}")

        dem_map  = {_stem(p): p for p in glob.glob(os.path.join(dem_dir, "*.png"))}
        mask_map = ({_stem(p): p for p in glob.glob(os.path.join(mask_dir, "*.png"))}
                    if has_mask else {})

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
            img_np = np.asarray(img, dtype=np.uint8)   # (H,W,3)
            dem_np = np.asarray(dem, dtype=np.uint8)   # (H,W)

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_u8, dem_u8, mask_u8 = self.data[idx]
        optical = torch.from_numpy(img_u8.astype(np.float32) / 255.0).permute(2, 0, 1)  # (3,H,W)
        dem     = torch.from_numpy(dem_u8.astype(np.float32) / 255.0).unsqueeze(0)        # (1,H,W)
        mask_t  = torch.from_numpy(mask_u8.astype(np.int64))                              # (H,W)
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

    # pos_weight dai pixel REALI del train (Bijie != L4S)
    px_fg = px_tot = 0
    for i in train_indices:
        m = full_ds.data[i][2]
        px_fg  += int(m.sum())
        px_tot += m.size
    pos_weight = (px_tot - px_fg) / max(px_fg, 1)
    print(f"  Frazione pixel frana (train): {px_fg/max(px_tot,1):.4f}")
    print(f"  pos_weight calcolato: {pos_weight:.2f}  "
          f"(usato nella loss solo se USE_POS_WEIGHT=True)\n")

    return (
        Subset(full_ds, train_indices),
        Subset(full_ds, val_indices),
        Subset(full_ds, test_indices),
        pos_weight,
    )


# ============================================================
# 3. ARCHITETTURA — D2FLS-Net (invariata rispetto a L4S)
# ============================================================

class SwinBackbone(nn.Module):
    def __init__(self, variant=SWIN_VARIANT, in_channels=3, pretrained=True, img_size=LOAD_SIZE):
        super().__init__()
        self.swin = timm.create_model(
            variant, pretrained=pretrained, features_only=True,
            out_indices=(0, 1, 2, 3), img_size=img_size,
        )
        if in_channels != 3:
            self._expand_patch_embed(in_channels)
        self.out_channels = self.swin.feature_info.channels()

    def _expand_patch_embed(self, in_channels):
        orig = self.swin.patch_embed.proj
        new_proj = nn.Conv2d(
            in_channels, orig.out_channels,
            kernel_size=orig.kernel_size, stride=orig.stride,
            padding=orig.padding, bias=(orig.bias is not None),
        )
        with torch.no_grad():
            new_proj.weight[:, :3] = orig.weight.clone()
            if in_channels > 3:
                nn.init.normal_(new_proj.weight[:, 3:], mean=0.0, std=0.001)
            if orig.bias is not None:
                new_proj.bias.copy_(orig.bias)
        self.swin.patch_embed.proj = new_proj
        print(f"  Patch embed espanso: 3 -> {in_channels} canali")

    def forward(self, x):
        features = self.swin(x)
        out = []
        for f, c_expected in zip(features, self.out_channels):
            if f.ndim == 4 and f.shape[-1] == c_expected:
                f = f.permute(0, 3, 1, 2).contiguous()
            out.append(f)
        return out


class EarlyDEMGuidance(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, dem, f1):
        g  = self.conv_block(dem)
        g1 = F.interpolate(g, size=f1.shape[2:], mode="bilinear", align_corners=False)
        return f1 * (1.0 + g1)


class TinyViTDEMEncoder(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, layers=2):
        super().__init__()
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=4, stride=4)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, dem):
        x = self.patch_embed(dem)
        B, E, Hd, Wd = x.shape
        seq = x.flatten(2).transpose(1, 2)
        seq = self.norm(self.transformer(seq))
        return seq.transpose(1, 2).reshape(B, E, Hd, Wd)


class HighLevelCrossAttention(nn.Module):
    def __init__(self, dim_rgb, dim_dem, num_heads=8):
        super().__init__()
        while dim_rgb % num_heads != 0 and num_heads > 1:
            num_heads //= 2
        self.wq   = nn.Conv2d(dim_rgb, dim_rgb, 1, bias=False)
        self.wk   = nn.Conv2d(dim_dem, dim_rgb, 1, bias=False)
        self.wv   = nn.Conv2d(dim_dem, dim_rgb, 1, bias=False)
        self.attn = nn.MultiheadAttention(dim_rgb, num_heads, batch_first=True)
        self.wo   = nn.Conv2d(dim_rgb, dim_rgb, 1, bias=False)
        self.norm = nn.LayerNorm(dim_rgb)

    def forward(self, f4_rgb, f4_dem):
        B, C, H, W = f4_rgb.shape
        if f4_dem.shape[2:] != f4_rgb.shape[2:]:
            f4_dem = F.interpolate(f4_dem, size=(H, W), mode="bilinear", align_corners=False)
        q = self.wq(f4_rgb).flatten(2).transpose(1, 2)
        k = self.wk(f4_dem).flatten(2).transpose(1, 2)
        v = self.wv(f4_dem).flatten(2).transpose(1, 2)
        attn_out, _ = self.attn(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, C, H, W)
        attn_out_flat = attn_out.permute(0, 2, 3, 1)
        attn_norm = self.norm(attn_out_flat).permute(0, 3, 1, 2)
        return self.wo(attn_norm) + f4_rgb


class TPACE(nn.Module):
    def __init__(self, in_channels_rgb, in_channels_dem):
        super().__init__()
        self.dilations = [1, 6, 12, 18]
        K = len(self.dilations)
        self.atrous_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels_rgb, in_channels_rgb, 3, padding=d, dilation=d),
                nn.BatchNorm2d(in_channels_rgb),
            ) for d in self.dilations
        ])
        self.gating = nn.Sequential(
            nn.Conv2d(in_channels_rgb + in_channels_dem, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, K, 1),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels_rgb, in_channels_rgb, 1),
            nn.BatchNorm2d(in_channels_rgb),
        )

    def forward(self, f3_rgb, f3_dem):
        if f3_dem.shape[2:] != f3_rgb.shape[2:]:
            f3_dem = F.interpolate(f3_dem, size=f3_rgb.shape[2:], mode="bilinear", align_corners=False)
        Z = torch.stack([b(f3_rgb) for b in self.atrous_branches], dim=1)
        u      = torch.cat([f3_rgb, f3_dem], dim=1)
        alpha  = F.softmax(self.gating(u), dim=1).unsqueeze(2)
        Y = (alpha * Z).sum(dim=1)
        return self.out_proj(Y) + f3_rgb


class FPNDecoder(nn.Module):
    def __init__(self, c0, c1, c2, c3, fpn_channels=256, num_classes=2):
        super().__init__()
        self.lat4 = nn.Conv2d(c3, fpn_channels, 1)
        self.lat3 = nn.Conv2d(c2, fpn_channels, 1)
        self.lat2 = nn.Conv2d(c1, fpn_channels, 1)
        self.lat1 = nn.Conv2d(c0, fpn_channels, 1)
        self.smooth3 = nn.Sequential(nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                                     nn.BatchNorm2d(fpn_channels), nn.ReLU(inplace=True))
        self.smooth2 = nn.Sequential(nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                                     nn.BatchNorm2d(fpn_channels), nn.ReLU(inplace=True))
        self.smooth1 = nn.Sequential(nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                                     nn.BatchNorm2d(fpn_channels), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(fpn_channels, fpn_channels // 2, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(fpn_channels // 2, fpn_channels // 4, 2, stride=2)
        self.classifier = nn.Conv2d(fpn_channels // 4, num_classes, 1)

    def forward(self, f1, f2, f3_tpace, f4_fused, orig_size):
        p4 = self.lat4(f4_fused)
        p3 = self.smooth3(self.lat3(f3_tpace) +
             F.interpolate(p4, size=f3_tpace.shape[2:], mode="bilinear", align_corners=False))
        p2 = self.smooth2(self.lat2(f2) +
             F.interpolate(p3, size=f2.shape[2:], mode="bilinear", align_corners=False))
        p1 = self.smooth1(self.lat1(f1) +
             F.interpolate(p2, size=f1.shape[2:], mode="bilinear", align_corners=False))
        x = F.relu(self.up1(p1))
        x = F.relu(self.up2(x))
        logits = self.classifier(x)
        return F.interpolate(logits, size=orig_size, mode="bilinear", align_corners=False)


class D2FLS_Net(nn.Module):
    DEM_ENC_DIM = 128

    def __init__(self, swin_variant=SWIN_VARIANT, in_channels_opt=IN_CHANNELS_OPT,
                 num_classes=2, pretrained=True, fpn_channels=256, img_size=LOAD_SIZE):
        super().__init__()
        self.backbone = SwinBackbone(swin_variant, in_channels_opt, pretrained, img_size=img_size)
        c0, c1, c2, c3 = self.backbone.out_channels

        self.dem_encoder = TinyViTDEMEncoder(embed_dim=self.DEM_ENC_DIM, num_heads=4, layers=2)
        self.early_guidance = EarlyDEMGuidance()
        self.dem_proj4 = nn.Conv2d(self.DEM_ENC_DIM, c3, 1)
        self.cross_attn = HighLevelCrossAttention(dim_rgb=c3, dim_dem=c3, num_heads=8)
        self.dem_proj3 = nn.Conv2d(self.DEM_ENC_DIM, c2, 1)
        self.tpace = TPACE(in_channels_rgb=c2, in_channels_dem=c2)
        self.decoder = FPNDecoder(c0, c1, c2, c3, fpn_channels=fpn_channels, num_classes=num_classes)

        total = sum(p.numel() for p in self.parameters())
        print(f"D2FLS-Net — {swin_variant} | canali: {c0},{c1},{c2},{c3} | "
              f"params totali: {total/1e6:.1f}M")

    def forward(self, optical, dem):
        B, _, H, W = optical.shape
        Z = self.dem_encoder(dem)
        f1, f2, f3, f4 = self.backbone(optical)
        f1 = self.early_guidance(dem, f1)

        z3 = F.interpolate(Z, size=f3.shape[2:], mode="bilinear", align_corners=False)
        f3_dem  = self.dem_proj3(z3)
        f3_hat  = self.tpace(f3, f3_dem)

        z4 = F.interpolate(Z, size=f4.shape[2:], mode="bilinear", align_corners=False)
        f4_dem    = self.dem_proj4(z4)
        f4_fused  = self.cross_attn(f4, f4_dem)

        return self.decoder(f1, f2, f3_hat, f4_fused, orig_size=(H, W))


# ============================================================
# 4. LOSS E METRICHE  (invariate)
# ============================================================

class CombinedLoss(nn.Module):
    """
    Loss fedele al paper (eq. 1-3):
      - principale: cross-entropy pixel-wise su 2 classi (eq. 1)
      - opzionale : + lambda * Dice sulla classe frana (eq. 2-3), solo se use_dice
      - opzionale : pos_weight sulla classe frana (NON nel paper), solo se use_pos_weight

    Default (use_dice=False, use_pos_weight=False) = CE pura, come nel paper.
    """
    def __init__(self, pos_weight=1.0, lambda_dice=1.0, epsilon=1e-5,
                 use_dice=False, use_pos_weight=False):
        super().__init__()
        self.register_buffer("pw", torch.tensor([pos_weight]))
        self.lambda_dice    = lambda_dice
        self.epsilon        = epsilon
        self.use_dice       = use_dice
        self.use_pos_weight = use_pos_weight

    def forward(self, preds, targets):
        if self.use_pos_weight:
            weight = torch.ones(2, device=preds.device)
            weight[1] = self.pw.item()
            loss_ce = F.cross_entropy(preds, targets, weight=weight)
        else:
            loss_ce = F.cross_entropy(preds, targets)

        if not self.use_dice:
            return loss_ce

        probs  = F.softmax(preds, dim=1)[:, 1]
        y_true = (targets == 1).float()
        inter  = (probs * y_true).sum(dim=(1, 2))
        sum_p  = probs.sum(dim=(1, 2))
        sum_y  = y_true.sum(dim=(1, 2))
        has_pos = (sum_y > 0).float()
        dice_score = (2.0 * inter + self.epsilon) / (sum_p + sum_y + self.epsilon)
        dice_loss  = has_pos * (1.0 - dice_score)
        return loss_ce + self.lambda_dice * dice_loss.mean()


def calculate_metrics(preds, targets, epsilon=1e-5):
    pred_bin = torch.argmax(preds, dim=1)
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
        "epoch": epoch, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_miou": best_miou,
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
    print(f"Checkpoint caricato — Epoca {ckpt['epoch']}, Best mIoU: {ckpt['best_miou']:.4f}")
    return ckpt["epoch"] + 1, ckpt["best_miou"]


# ============================================================
# 5. TRAINING LOOP
# ============================================================

def train():
    _tag = SWIN_VARIANT.replace('swin_', '').replace('_patch4_window7_224', '').replace('.', '_')
    ckpt_path    = f"best_d2fls_bijie_{_tag}.pth"
    history_file = f"history_d2fls_bijie_{_tag}.json"

    full_ds = LandslideH5Dataset(DATASET_ROOT, load_size=LOAD_SIZE)
    train_subset, val_subset, test_subset, pos_weight = build_train_val_test_subsets(
        full_ds, VAL_FRACTION, TEST_FRACTION, RANDOM_SEED)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_subset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    model     = D2FLS_Net(swin_variant=SWIN_VARIANT, in_channels_opt=IN_CHANNELS_OPT,
                          num_classes=2, pretrained=True, img_size=LOAD_SIZE).to(DEVICE)
    criterion = CombinedLoss(pos_weight=pos_weight, lambda_dice=LAMBDA_DICE,
                             use_dice=USE_DICE, use_pos_weight=USE_POS_WEIGHT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=5)
    scaler    = torch.amp.GradScaler("cuda")

    start_epoch, best_miou = load_checkpoint(model, optimizer, scheduler, ckpt_path, DEVICE)
    history = []
    patience_counter = 0
    if os.path.exists(history_file):
        with open(history_file) as f:
            history = json.load(f)

    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri trainabili: {n_tr/1e6:.1f}M")
    print(f"Training su {DEVICE} | Epoche {start_epoch+1}/{NUM_EPOCHS} | Batch {BATCH_SIZE}\n")

    for epoch in range(start_epoch, NUM_EPOCHS):
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
            t_loss += loss.item(); t_miou += miou; t_iou_ls += iou_ls; t_prec += prec; t_rec += rec
            pbar.set_postfix(loss=f"{loss.item():.4f}", mIoU=f"{miou*100:.1f}%", IoU_LS=f"{iou_ls*100:.1f}%")

        model.eval()
        v_loss = v_miou = v_iou_ls = v_prec = v_rec = 0.0
        with torch.no_grad():
            for opt, dem, mask in tqdm(val_loader, desc=f"Ep {epoch+1}/{NUM_EPOCHS} [VAL]  "):
                opt, dem, mask = opt.to(DEVICE), dem.to(DEVICE), mask.to(DEVICE)
                with torch.amp.autocast("cuda"):
                    out  = model(opt, dem)
                    loss = criterion(out, mask)
                miou, prec, rec, iou_ls = calculate_metrics(out, mask)
                v_loss += loss.item(); v_miou += miou; v_iou_ls += iou_ls; v_prec += prec; v_rec += rec

        ntr, nvl = len(train_loader), len(val_loader)
        avg = {
            "epoch": epoch + 1,
            "train_loss": t_loss/ntr, "train_miou": t_miou/ntr, "train_iou_ls": t_iou_ls/ntr,
            "train_precision": t_prec/ntr, "train_recall": t_rec/ntr,
            "val_loss": v_loss/nvl, "val_miou": v_miou/nvl, "val_iou_ls": v_iou_ls/nvl,
            "val_precision": v_prec/nvl, "val_recall": v_rec/nvl,
        }
        scheduler.step(avg["val_miou"])

        print(
            f"EPOCA {epoch+1:3d}\n"
            f"  TRAIN | Loss:{avg['train_loss']:.4f}  mIoU:{avg['train_miou']*100:.2f}%  "
            f"IoU_LS:{avg['train_iou_ls']*100:.2f}%  Prec:{avg['train_precision']*100:.2f}%  "
            f"Rec:{avg['train_recall']*100:.2f}%\n"
            f"  VAL   | Loss:{avg['val_loss']:.4f}  mIoU:{avg['val_miou']*100:.2f}%  "
            f"IoU_LS:{avg['val_iou_ls']*100:.2f}%  Prec:{avg['val_precision']*100:.2f}%  "
            f"Rec:{avg['val_recall']*100:.2f}%"
        )

        history.append(avg)
        with open(history_file, "w") as f:
            json.dump(history, f, indent=4)

        if avg["val_miou"] > best_miou:
            best_miou = avg["val_miou"]
            save_checkpoint(model, optimizer, scheduler, epoch, best_miou, ckpt_path)
            print(f"  Nuovo record mIoU val: {best_miou*100:.2f}% — checkpoint salvato")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"\nEARLY STOPPING — Miglior mIoU val: {best_miou*100:.2f}%")
                break

    # ── TEST FINALE ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  VALUTAZIONE FINALE SUL TEST SET (Bijie)")
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
            te_loss += loss.item(); te_miou += miou; te_iou_ls += iou_ls; te_prec += prec; te_rec += rec

    nte = len(test_loader)
    test_results = {
        "test_loss": te_loss/nte, "test_miou": te_miou/nte, "test_iou_ls": te_iou_ls/nte,
        "test_precision": te_prec/nte, "test_recall": te_rec/nte,
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