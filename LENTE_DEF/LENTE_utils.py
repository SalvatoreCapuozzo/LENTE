import os
import contextlib
import glob
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
import segmentation_models_pytorch as smp
from transformers import Sam2Model
try:
    # Serve solo per girare a una risoluzione diversa dai 1024 del checkpoint.
    # Importato a parte cosi' una sua assenza non rompe tutti gli script.
    from transformers import Sam2Config
except ImportError:
    Sam2Config = None

# Optional deps (only needed for the respective datasets)
try:
    import h5py
except ImportError:
    h5py = None
try:
    from PIL import Image
except ImportError:
    Image = None
try:
    from scipy import ndimage
except ImportError:
    ndimage = None


# ============================================================
# CONFIG LOADING
# ============================================================

# Regimi di prompt ammessi (semantica in config.yaml e in resolve_prompt).
PROMPT_SOURCES = ("gt_box", "image_box", "none", "yolo_box", "clf_box")
# Suffisso applicato a checkpoint e history in base al regime di prompt.
PROMPT_TAGS = {"gt_box": "gtbox", "image_box": "imgbox", "none": "free",
               "yolo_box": "yolo", "clf_box": "clf"}

# Riquadro SENTINELLA delle patch vuote, in coordinate native.
# E' la costante che `_bbox_from_mask` restituisce quando la maschera e' vuota,
# e su cui il regime `gt_box` prendeva vuote = 1.0000 ESATTO su 306 patch: il
# modello imparava "questo pattern -> non disegnare niente".
SENTINEL_BOX = (0.0, 0.0, 10.0, 10.0)


def _tag_path(path, tag):
    base, ext = os.path.splitext(path)
    return f"{base}_{tag}{ext}"


def apply_prompt_tag(cfg):
    """
    Inserisce il regime di prompt nei nomi di checkpoint e history:
        best_lente_l4s.pth -> best_lente_l4s_free.pth

    Serve perche' i tre regimi producono modelli NON intercambiabili. Senza
    questo scriverebbero tutti sullo stesso file, e `load_checkpoint` (che in
    LENTE_train riprende in silenzio se il file esiste) mescolerebbe due
    esperimenti diversi senza dire nulla.

    Idempotente: se il tag c'e' gia' non lo raddoppia. Un --checkpoint esplicito
    da riga di comando viene applicato DOPO e resta intatto.
    """
    tag = PROMPT_TAGS[cfg["prompt_source"]]
    for key in ("checkpoint_name", "history_file"):
        if key in cfg and not os.path.splitext(cfg[key])[0].endswith(f"_{tag}"):
            cfg[key] = _tag_path(cfg[key], tag)
    return cfg


def load_config(config_path="config.yaml", dataset=None, prompt_source=None):
    """
    Loads the unified YAML and returns a FLAT dict merging `common` with the
    chosen dataset block. `dataset` overrides `active_dataset` in the file.
    `prompt_source` overrides the one in the file (usato dal flag --prompt).
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

    # Il percorso del dataset e' l'unica cosa che cambia fra una macchina e
    # l'altra, e config.yaml viaggia sincronizzato fra un server Linux e uno
    # Windows. Scriverlo nel file romperebbe l'altra macchina alla prossima
    # sincronizzazione; una variabile d'ambiente lo sposta solo dove serve, e
    # vale per TUTTI gli script senza doverla aggiungere a ciascuno.
    #   Linux:       export LENTE_DATASET_ROOT=/percorso
    #   PowerShell:  $env:LENTE_DATASET_ROOT="D:\percorso"
    esterno = os.environ.get(f"LENTE_{name.upper()}_ROOT") \
        or os.environ.get("LENTE_DATASET_ROOT")
    if esterno:
        cfg["dataset_root"] = esterno
        print(f"  dataset_root da variabile d'ambiente: {esterno}")

    cfg["prompt_source"] = prompt_source or cfg.get("prompt_source", "none")
    if cfg["prompt_source"] not in PROMPT_SOURCES:
        raise ValueError(
            f"prompt_source sconosciuto: {cfg['prompt_source']!r} (attesi: {PROMPT_SOURCES})")
    return apply_prompt_tag(cfg)


# ============================================================
# DATASETS
# ============================================================

class L4SDataset(Dataset):
    """Landslide4Sense (.h5). Stack a 14ch [12 Sentinel-2 + Slope + DEM]."""
    IMG_DIR  = "TrainData/img"
    MASK_DIR = "TrainData/mask"

    def __init__(self, root_dir, load_size=None, canali=None):
        # `canali` e' la lista ORDINATA degli indici da tenere, ottiche poi
        # terreno. Fino a oggi optical_bands serviva solo a derivare n_optical,
        # e il modello tagliava lo stack con x[:, :n_optical]: funzionava per
        # puro caso perche' optical_bands era [0..11], cioe' proprio i primi
        # dodici canali. Chiedere [3, 7] dava n_optical=2 e il modello leggeva
        # i canali 0 e 1 come ottici e i restanti DODICI come terreno.
        self.canali = None if canali is None else list(canali)
        if h5py is None:
            raise ImportError("h5py richiesto per il dataset L4S (pip install h5py)")
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(
                f"dataset_root non esiste: {root_dir}\n"
                f"Su una macchina diversa da quella del config, NON modificare "
                f"config.yaml: imposta la variabile d'ambiente nella stessa "
                f"finestra da cui lanci il comando.\n"
                f"  PowerShell:  $env:LENTE_DATASET_ROOT=\"<percorso>\"\n"
                f"  Linux:       export LENTE_DATASET_ROOT=<percorso>\n"
                f"Il percorso e' la cartella che CONTIENE TrainData.")
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
        if self.canali is not None:
            # ATTENZIONE: da qui in poi lo stack e' RIORDINATO. Chiunque indicizzi
            # per numero di banda (yolo_bands, rgb_channels, LENTE_bande) vede
            # questa lista, non le 14 originali.
            img_t = img_t[self.canali]
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
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(
                f"dataset_root non esiste: {root_dir}\n"
                f"Su una macchina diversa da quella del config, NON modificare "
                f"config.yaml: imposta nella stessa finestra la variabile "
                f"LENTE_BIJIE_ROOT (PowerShell: $env:LENTE_BIJIE_ROOT=\"<percorso>\").")
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
        # Selezione delle bande: si passa al dataset solo se e' davvero un
        # sottoinsieme, cosi' i run storici restano bit-identici.
        canali = list(cfg["optical_bands"]) + list(cfg.get("terrain_bands", []))
        if canali == list(range(14)):
            canali = None
        else:
            print(f"  SELEZIONE BANDE: {len(canali)} canali su 14 -> {canali} "
                  f"(ottiche {cfg['optical_bands']}, "
                  f"terreno {cfg.get('terrain_bands', [])})")
        return L4SDataset(cfg["dataset_root"], load_size=cfg.get("load_size"),
                          canali=canali)
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
        mask_t  = aug["mask"].long()
        # Il box va RICALCOLATO dalla maschera aumentata. Prima si restituiva
        # sample["boxes"], calcolato sulla maschera ORIGINALE: dopo un
        # ribaltamento o una rotazione il box indicava il punto sbagliato.
        # Con HorizontalFlip, VerticalFlip e RandomRotate90 a p=0.5 ciascuno,
        # l'87,5% dei campioni (1 - 0.5^3) aveva un box disallineato.
        # Irrilevante per il regime senza prompt e per YOLO, che calcola i
        # propri box sull'immagine gia' aumentata; decisivo per gt_box, dove il
        # modello imparava che il box e' quasi sempre sbagliato e a non
        # fidarsene.
        return {"image": aug["image"], "mask": mask_t, "boxes": _bbox_from_mask(mask_t)}


def build_train_val_test_subsets(full_ds, cfg):
    val_fraction  = cfg["val_fraction"]
    test_fraction = cfg["test_fraction"]
    rng      = torch.Generator().manual_seed(cfg["random_seed"])
    all_idx  = list(range(len(full_ds)))
    perm     = torch.randperm(len(all_idx), generator=rng).tolist()
    shuffled = [all_idx[i] for i in perm]
    pos_set  = set(full_ds.positive_indices)

    n_test        = int(len(shuffled) * test_fraction)

    # ── k-fold ──────────────────────────────────────────────────────────
    # Il fold f ruota la STESSA permutazione di f * n_test posizioni e poi
    # applica il taglio di sempre. Il fold 0 e' quindi identico allo split
    # usato finora: i checkpoint gia' addestrati sono il fold 0, senza
    # riaddestrarli. I test dei fold sono disgiunti e ogni fold ha lo stesso
    # numero di patch di test, val e train. Con test_fraction 0.2 si hanno 5
    # fold; i len % n_test campioni in coda alla permutazione (4 su L4S, 3 su
    # Bijie) non finiscono mai nel test.
    fold   = int(cfg.get("fold", 0))
    n_fold = int(round(1.0 / test_fraction))
    if not 0 <= fold < n_fold:
        raise ValueError(f"fold {fold} fuori da [0, {n_fold - 1}] con test_fraction {test_fraction}")
    if fold:
        shuffled = shuffled[fold * n_test:] + shuffled[:fold * n_test]

    test_indices  = shuffled[:n_test]
    remaining     = shuffled[n_test:]
    n_val         = int(len(remaining) * (val_fraction / (1.0 - test_fraction)))
    val_indices   = remaining[:n_val]
    train_indices = remaining[n_val:]

    def _count(idxs):
        n_p = sum(1 for i in idxs if i in pos_set)
        return n_p, len(idxs) - n_p
    n_pos_tr, n_neg_tr = _count(train_indices)

    # ── pos_weight della BCE ────────────────────────────────────────────
    # Misurato con LENTE_prompt_audit.py sul train reale:
    #                       L4S      Bijie
    #   "pixels"           42.14     34.56    (rapporto reale)
    #   "sqrt"              6.49      5.88
    #   "heuristic"         3.87      9.29
    #
    # "heuristic" assume ls_frac=0.35, cioe' una patch positiva coperta di frana
    # al 35%. La copertura reale e' 3.95% su L4S e 10.13% su Bijie: la costante
    # e' fuori scala e su L4S sotto-pesa la classe positiva di 10.9x.
    #
    # Perche' non "pixels" (il rapporto vero) adesso: con il GT box la predizione
    # era incanalata dentro un riquadro gia' denso di frana (21% su L4S, 60% su
    # Bijie), quindi un peso alto non faceva danni. Prompt-free il vincolo non
    # c'e' piu' e ribilanciare completamente spinge il modello ad accendere pixel
    # ovunque: la precision crolla. E' la stessa conclusione a cui si arriva in
    # Sen12Landslides/config.py (VIDEO_POS_WEIGHT_MODE = "sqrt").
    #
    # In piu' "sqrt" allinea i due domini (6.49 vs 5.88) invece di farli
    # differire di 9x, il che conta nel training congiunto di LENTE_multi_train,
    # dove batch di domini diversi si alternano dentro la stessa epoca.
    mode = cfg.get("pos_weight_mode", "sqrt")

    def _real_ratio():
        px_fg = px_tot = 0
        for i in train_indices:
            m = np.asarray(full_ds.get_mask(i))
            px_fg  += int(m.sum())
            px_tot += m.size
        return (px_tot - px_fg) / max(px_fg, 1)

    if mode == "heuristic":
        ls_frac    = 0.35
        pixel_ls   = n_pos_tr * ls_frac
        pixel_bg   = n_neg_tr + n_pos_tr * (1.0 - ls_frac)
        pos_weight = pixel_bg / max(pixel_ls, 1.0)
    elif mode == "sqrt":
        pos_weight = _real_ratio() ** 0.5
    elif mode == "fixed":
        pos_weight = float(cfg["pos_weight_fixed"])
    else:  # "pixels"
        pos_weight = _real_ratio()

    print(f"Split: fold {fold}/{n_fold} | train {len(train_indices)} | val {len(val_indices)} "
          f"| test {len(test_indices)} | pos_weight ({mode}): {pos_weight:.2f}")

    return (Subset(full_ds, train_indices), Subset(full_ds, val_indices),
            Subset(full_ds, test_indices), pos_weight)


def _registry_terratorch(max_stub=12):
    """
    Importa BACKBONE_REGISTRY tollerando i backbone di terratorch che non si
    caricano.

    PERCHE' SERVE. `import terratorch` tira dentro TUTTI i suoi backbone in
    blocco, e alcuni referenziano pesi di torchgeo che esistono solo nelle
    versioni recenti — per esempio ResNet50_Weights.SENTINEL2_ALL_SOFTCON. Con
    un torchgeo piu' vecchio l'import dell'intero pacchetto fallisce, anche se
    il backbone che ci serve, Prithvi, non c'entra niente.

    COSA FA. Al primo fallimento individua QUALE file di terratorch stava
    girando, mette un modulo VUOTO al suo posto in sys.modules e riprova:
    l'istruzione di import lo trova gia' presente e non lo esegue. Si ripete
    finche' l'import riesce o finche' i moduli saltati diventano troppi.

    COSA SI PERDE. Solo i backbone effettivamente saltati, che vengono
    STAMPATI. Se fra questi comparisse qualcosa di Prithvi, la costruzione
    fallirebbe subito dopo con un errore esplicito invece di proseguire.

    L'alternativa pulita e' `pip install -U torchgeo`, che pero' si porta
    dietro le proprie dipendenze e puo' toccare l'installazione di torch.
    """
    import sys
    import traceback
    import types

    def nome_modulo(percorso):
        p = str(percorso).replace("\\", "/")
        if "/terratorch/" not in p:
            return None
        resto = "terratorch/" + p.split("/terratorch/", 1)[1]
        if resto.endswith(".py"):
            resto = resto[:-3]
        if resto.endswith("/__init__"):
            resto = resto[:-9]
        return resto.replace("/", ".")

    saltati = []
    for _ in range(max_stub + 1):
        try:
            from terratorch.registry import BACKBONE_REGISTRY
            if saltati:
                print(f"  terratorch: saltati {len(saltati)} backbone "
                      f"incompatibili con questo torchgeo")
                for m in saltati:
                    print(f"    - {m.split('.')[-1]}")
            return BACKBONE_REGISTRY
        except ImportError as e:
            raise SystemExit(
                "Serve terratorch per l'encoder Prithvi:  pip install terratorch"
            ) from e
        except Exception as e:
            # Il file piu' PROFONDO di terratorch nello stack e' quello che sta
            # fallendo: e' lui da neutralizzare, non chi lo importa.
            colpevole = None
            for f in traceback.extract_tb(e.__traceback__):
                m = nome_modulo(f.filename)
                if m:
                    colpevole = m
            if colpevole is None or colpevole in saltati:
                raise SystemExit(
                    f"terratorch non si importa e non riesco a isolare il "
                    f"modulo colpevole.\n  Ultimo errore: {type(e).__name__}: {e}"
                    f"\n  Prova con:  pip install -U torchgeo") from e
            for k in [k for k in list(sys.modules) if k.startswith("terratorch")]:
                del sys.modules[k]
            saltati.append(colpevole)
            for m in saltati:
                sys.modules[m] = types.ModuleType(m)
    raise SystemExit(
        f"terratorch: {len(saltati)} moduli saltati e ancora non si importa.\n"
        f"  Prova con:  pip install -U torchgeo")


def _elenca_backbone(reg):
    """Nomi disponibili nel registro, provando le vie che terratorch espone."""
    nomi = set()
    for via in (lambda: list(reg),
                lambda: list(reg.keys()),
                lambda: [n for s in getattr(reg, "_sources", {}).values()
                         for n in list(s)]):
        try:
            nomi.update(str(x) for x in via())
        except Exception:
            pass
    return sorted(nomi)


def _costruisci_prithvi(reg, nome):
    """
    Costruisce il backbone provando piu' NOMI, perche' non c'e' accordo fra le
    fonti: la scheda HuggingFace documenta l'identificativo completo del
    repository, mentre le versioni di terratorch registrano nomi brevi propri
    (prithvi_eo_v2_300 e simili) e cambiano nel tempo.

    Se falliscono tutti, si ELENCANO i nomi che il registro contiene davvero
    filtrati su "prithvi": e' l'unica informazione che serve per sbloccarsi, e
    tenerla nell'eccezione evita di dover andare a cercarla altrove.
    """
    provati, errori = [], []
    candidati = [nome, nome.split("/")[-1],
                 "prithvi_eo_v2_300", "prithvi_eo_v2_300_tl",
                 "prithvi_eo_v2_600", "prithvi_eo_v2_600_tl",
                 "prithvi_eo_v1_100", "prithvi_vit_300", "prithvi_vit_100"]
    visti = set()
    for c in candidati:
        if c in visti:
            continue
        visti.add(c)
        provati.append(c)
        try:
            b = reg.build(c, pretrained=True)
            if c != nome:
                print(f"  nome del backbone: '{c}'  (l'identificativo "
                      f"'{nome}' non e' nel registro di questo terratorch)")
            return b, c
        except Exception as e:
            errori.append(f"{c}: {type(e).__name__}")

    disponibili = [n for n in _elenca_backbone(reg) if "prithvi" in n.lower()]
    raise SystemExit(
        "Nessuno dei nomi provati esiste nel registro di terratorch.\n"
        "  provati: " + ", ".join(provati) + "\n"
        "  nel registro ci sono questi backbone Prithvi:\n    "
        + ("\n    ".join(disponibili) if disponibili else
           "NESSUNO — terratorch non ha caricato i backbone Prithvi")
        + "\n  Scegline uno e passalo con  --prithvi-model <nome>")


class PrithviEncoder(nn.Module):
    """
    Encoder Prithvi-EO al posto di Hiera, con piramide SINTETICA (schema A).

    IL PROBLEMA CHE RISOLVE. Prithvi e' un ViT con patch 16: produce feature a
    UN SOLO stride, il 16. Il mask decoder di SAM2 ne pretende tre — 32 canali
    a stride 4, 64 a stride 8, 256 a stride 16. I due livelli superficiali qui
    non esistono e vengono COSTRUITI risalendo dal profondo.

    IL LIMITE, dichiarato perche' e' strutturale e non si aggira. Quei due
    livelli portano semantica ingrandita, non dettaglio: nel decoder di SAM2
    vengono sommati durante la risalita proprio per iniettare alta frequenza,
    e qui l'alta frequenza non c'e'. E' il difetto noto dello schema A, ed e'
    il motivo per cui gli schemi B e C tengono Hiera sui livelli superficiali.

    LE BANDE. Prithvi-EO vuole sei bande HLS. L4S ne ha dodici Sentinel-2, e
    NON contiene B8A: al suo posto va B8, che e' la vicina piu' prossima nel
    vicino infrarosso ma non la stessa banda. Restano fuori bordo rosso
    (B5, B6, B7), aerosol, vapore acqueo e cirri — fra cui B5, che la
    permutazione indica come SECONDA dipendenza del modello attuale.

    LA RISOLUZIONE. L'attenzione e' quadratica nel numero di token: a 1024
    pixel con patch 16 sono 4096 token contro i 196 su cui Prithvi e' stato
    addestrato. Su una 3090 non ci sta. Prithvi riceve quindi un ingresso di
    lato `lato` (default 512, cioe' 1024 token) e la sua mappa viene poi
    riportata alla griglia che il decoder si aspetta. Le due risoluzioni sono
    disaccoppiate di proposito, ed entrambe vanno dichiarate.
    """

    # Stesso contratto di UNetEncoder, imposto dal decoder di SAM2.
    USCITE = (32, 64, 256)

    # L'ordine e' quello che Prithvi si aspetta. A destra la banda Sentinel-2
    # equivalente e il suo indice nello stack L4S a 14 canali.
    BANDE = [("Blue", "B2", 1),
             ("Green", "B3", 2),
             ("Red", "B4", 3),
             ("Narrow NIR", "B8", 7),     # SOSTITUZIONE: L4S non ha B8A
             ("SWIR 1", "B11", 10),
             ("SWIR 2", "B12", 11)]

    def __init__(self, optical_bands, nome="ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
                 lato=512, congelato=True, adapter_dim=0, adapter_where="both",
                 lora_rank=0, lora_alpha=16.0, solo_profondo=False,
                 norm=None):
        super().__init__()
        BACKBONE_REGISTRY = _registry_terratorch()

        self.lato = int(lato)

        # --- quali canali dell'ingresso ottico servono a Prithvi -----------
        # `optical` arriva gia' ridotto e riordinato secondo optical_bands,
        # quindi l'indice L4S va tradotto nella POSIZIONE dentro quella lista.
        ob = list(optical_bands)
        sel, mancanti = [], []
        for nome_p, nome_s2, idx_l4s in self.BANDE:
            if idx_l4s in ob:
                sel.append(ob.index(idx_l4s))
            else:
                mancanti.append(f"{nome_p} ({nome_s2})")
        if mancanti:
            raise SystemExit(
                "Prithvi richiede sei bande e queste non sono in optical_bands: "
                + ", ".join(mancanti)
                + f"\n  optical_bands attuali: {ob}"
                + "\n  Servono gli indici L4S 1, 2, 3, 7, 10, 11.")
        self.register_buffer("sel", torch.tensor(sel, dtype=torch.long),
                             persistent=False)

        # --- adattamento della scala radiometrica --------------------------
        # Prithvi si aspetta riflettanza HLS normalizzata con le SUE statistiche;
        # il nostro dataset applica gia' la propria. Invece di indovinare la
        # conversione, la si impara: affine per canale, IDENTITA' all'inizio,
        # sei scale e sei traslazioni in tutto.
        sc, sp = _affine_da_stat([i for _, _, i in self.BANDE], norm, 6)
        self.scala = nn.Parameter(sc)
        self.sposta = nn.Parameter(sp)

        self.backbone, self.nome = _costruisci_prithvi(BACKBONE_REGISTRY, nome)
        if congelato:
            for q in self.backbone.parameters():
                q.requires_grad = False
            self.backbone.eval()
        self.congelato = congelato

        self.dim = d = self._dimensione()

        # Gli adapter si agganciano DOPO il congelamento, cosi' i loro
        # parametri restano gli unici con gradiente dentro l'encoder.
        self.adapters = nn.ModuleList()
        self.adattato = self._adatta(adapter_dim, adapter_where,
                                     lora_rank, lora_alpha)

        # --- collo piramidale ----------------------------------------------
        # I canali si RIDUCONO PRIMA di risalire di risoluzione. Tutti e tre i
        # livelli derivano dalla stessa mappa a stride 16: allargare i canali
        # prima di ingrandire non aggiunge informazione, moltiplica soltanto la
        # memoria — a 1024 il livello a stride 4 e' 256x256 per campione.
        c4, c8, c16 = self.USCITE
        self.testa16 = nn.Conv2d(d, c16, kernel_size=1)

        # Da stream AUSILIARIO (schema C) serve solo la mappa profonda: e' li'
        # che Prithvi vive nativamente, e i livelli superficiali restano quelli
        # di Hiera. Non costruirli evita anche la parte piu' cara del collo —
        # a 1024 il livello a stride 4 e' 256x256 per campione.
        self.solo_profondo = bool(solo_profondo)
        if self.solo_profondo:
            self.testa8 = self.testa4 = None
            print(f"    modo AUSILIARIO: solo il livello /16, "
                  f"i superficiali restano di Hiera")
            return

        self.testa8 = nn.Sequential(
            nn.Conv2d(d, c8, kernel_size=1),
            nn.ConvTranspose2d(c8, c8, kernel_size=2, stride=2),
            nn.GroupNorm(8, c8), nn.GELU(),
            nn.Conv2d(c8, c8, kernel_size=3, padding=1))
        self.testa4 = nn.Sequential(
            nn.Conv2d(d, c4, kernel_size=1),
            nn.ConvTranspose2d(c4, c4, kernel_size=2, stride=2),
            nn.GroupNorm(8, c4), nn.GELU(),
            nn.ConvTranspose2d(c4, c4, kernel_size=2, stride=2),
            nn.GroupNorm(8, c4), nn.GELU(),
            nn.Conv2d(c4, c4, kernel_size=3, padding=1))

        addestrabili = sum(q.numel() for q in self.parameters() if q.requires_grad)
        # self.nome, non nome: e' quello EFFETTIVAMENTE costruito, che puo'
        # differire da quello chiesto.
        print(f"  ENCODER SOSTITUITO: Prithvi {self.nome.split('/')[-1]}, "
              f"dim {d}, ingresso {self.lato}px "
              f"({(self.lato // 16) ** 2} token)")
        print(f"    bande: " + ", ".join(f"{p}<-{s}" for p, s, _ in self.BANDE)
              + "   (B8A assente da L4S, sostituita con B8)")
        print(f"    piramide sintetica + affine: {addestrabili:,} parametri "
              f"addestrabili | backbone "
              f"{'congelato' if congelato else 'ADDESTRABILE'}. "
              f"Hiera NON viene chiamato.")

    def _adatta(self, adapter_dim, adapter_where, lora_rank, lora_alpha):
        """
        Rende adattabile il backbone congelato, con Adapter e/o LoRA.

        PERCHE'. Congelato del tutto, Prithvi gareggia contro un Hiera che ha
        2,4 milioni di adapter dentro: non e' un confronto alla pari. E il
        paper di Prithvi su L4S usava LoRA proprio sull'encoder, quindi la
        configurazione congelata e' piu' debole del loro stesso riferimento.

        I moduli nuovi vivono su QUESTO oggetto, non dentro il backbone, cosi'
        restano addestrabili mentre i pesi originali restano fermi.

        COSTO. ViT-L ha larghezza 1024 su 24 blocchi: un Adapter per blocco su
        attenzione e MLP a dimensione 128 sarebbe 12,6 milioni di parametri,
        piu' di tutto il resto messo insieme. LoRA a rango 8 ne costa 0,8 —
        ed e' la via che il loro paper ha scelto.
        """
        if adapter_dim <= 0 and lora_rank <= 0:
            return False        # niente da agganciare: non serve nemmeno
                                # cercare i blocchi, e un backbone con una
                                # struttura diversa non deve far fallire il
                                # caso congelato, che e' il default
        blocchi = getattr(self.backbone, "blocks", None)
        if blocchi is None:
            for m in self.backbone.modules():
                b = getattr(m, "blocks", None)
                if b is not None and len(list(b)) > 1:
                    blocchi = b
                    break
        if blocchi is None:
            raise SystemExit(
                "Non trovo i blocchi transformer di Prithvi: senza, non so "
                "dove agganciare adapter o LoRA.")

        self.adapters = nn.ModuleList()
        n_lora = 0
        for blocco in blocchi:
            if adapter_dim > 0:
                for attr in (("attn",) if adapter_where in ("attn", "both") else ()) + \
                            (("mlp",) if adapter_where in ("mlp", "both") else ()):
                    mod = getattr(blocco, attr, None)
                    if mod is None:
                        continue
                    ad = Adapter(self.dim, adapter_dim)
                    self.adapters.append(ad)
                    mod.register_forward_hook(_hook_generico(ad))
            if lora_rank > 0:
                attn = getattr(blocco, "attn", None)
                if attn is not None:
                    for n in ("qkv", "proj"):
                        lin = getattr(attn, n, None)
                        if isinstance(lin, nn.Linear):
                            setattr(attn, n, LoRALinear(lin, lora_rank, lora_alpha))
                            n_lora += 1

        n_ad = sum(q.numel() for a in self.adapters for q in a.parameters())
        n_lo = sum(q.numel() for m in self.backbone.modules()
                   if isinstance(m, LoRALinear)
                   for q in m.parameters() if q.requires_grad)
        if n_ad or n_lo:
            print(f"    adattamento del backbone: adapter {n_ad:,} "
                  f"({len(self.adapters)} moduli, dim {adapter_dim}, "
                  f"{adapter_where}) | LoRA {n_lo:,} ({n_lora} layer, "
                  f"rango {lora_rank})")
        return bool(n_ad or n_lo)


    # ------------------------------------------------------------------
    def _dimensione(self):
        """Dimensione dell'embedding, chiesta al backbone o dedotta provando."""
        for attr in ("embed_dim", "out_channels", "num_features", "hidden_size"):
            v = getattr(self.backbone, attr, None)
            if isinstance(v, int):
                return v
            if isinstance(v, (list, tuple)) and v and isinstance(v[-1], int):
                return int(v[-1])
        with torch.no_grad():
            finto = torch.zeros(1, 1, 6, self.lato, self.lato)
            return self._mappa(self.backbone(finto), 1).shape[1]

    def _mappa(self, uscita, b):
        """
        Riduce cio' che il backbone restituisce a una mappa (B, D, h, w).

        terratorch non garantisce una forma sola: puo' tornare una lista di
        livelli, token (B, N, D) con o senza CLS, o gia' una mappa. Si
        normalizzano i casi noti e si FALLISCE A VOCE ALTA sugli altri, invece
        di indovinare una permutazione e misurare un modello senza senso.
        """
        if isinstance(uscita, (list, tuple)):
            uscita = uscita[-1]
        if uscita.ndim == 4:
            return uscita
        if uscita.ndim == 3:
            n, d = uscita.shape[1], uscita.shape[2]
            lato = int(round(n ** 0.5))
            if lato * lato != n:                 # c'e' un token di classe
                uscita = uscita[:, n - (lato := int(round((n - 1) ** 0.5))) ** 2:]
                n = lato * lato
            if lato * lato != uscita.shape[1]:
                raise RuntimeError(
                    f"token non quadrati: {uscita.shape}. Il collo piramidale "
                    f"non sa come disporli.")
            return uscita.transpose(1, 2).reshape(b, d, lato, lato)
        raise RuntimeError(f"uscita del backbone non riconosciuta: {uscita.shape}")

    def forward(self, optical):
        s = optical.shape[-1]
        x = optical.index_select(1, self.sel) * self.scala + self.sposta
        if x.shape[-1] != self.lato:
            x = F.interpolate(x, (self.lato, self.lato), mode="bilinear",
                              align_corners=False)

        # (B, C, T=1, H, W). Il patch embedding di Prithvi e' un Conv3d, e
        # Conv3d vuole (N, C, D, H, W): i CANALI PRIMA del tempo. La scheda
        # HuggingFace documenta (B, T, C, H, W), ma la forma del peso —
        # [1024, 6, 1, 16, 16] = (uscita, ingresso 6, kT 1, kH 16, kW 16) —
        # dice il contrario, e vince il peso.
        # no_grad SOLO se dentro il backbone non c'e' nulla da addestrare.
        # Con adapter o LoRA agganciati, quel no_grad taglierebbe il gradiente
        # ai moduli nuovi e li lascerebbe all'inizializzazione per sempre,
        # senza nessun errore: e' il tipo di guasto che si vede solo dal
        # risultato piatto.
        muto = self.congelato and not self.adattato
        ctx = torch.no_grad() if muto else contextlib.nullcontext()
        with ctx:
            f = self._mappa(self.backbone(x.unsqueeze(2)), x.shape[0])
        if muto:
            f = f.detach()

        # La griglia di Prithvi dipende da `lato`, quella che il decoder si
        # aspetta da target_size: si riallineano qui, in un punto solo.
        if f.shape[-1] != s // 16:
            f = F.interpolate(f.float(), (s // 16, s // 16), mode="bilinear",
                              align_corners=False).to(optical.dtype)
        if self.solo_profondo:
            # Stessa lunghezza, cosi' l'indicizzazione per livello resta una
            # sola in tutto il codice. Chi legge gli indici 0 e 1 in questo
            # modo sta sbagliando, e un None lo fa capire subito.
            return [None, None, self.testa16(f)]
        return [self.testa4(f), self.testa8(f), self.testa16(f)]


def _hook_generico(adapter):
    """
    Hook che applica un Adapter all'uscita di un modulo, tollerando le uscite
    a tupla che alcuni blocchi di attenzione restituiscono.
    """
    def hook(module, args, output):
        if isinstance(output, tuple):
            return (adapter(output[0]),) + output[1:]
        return adapter(output)
    return hook


def _affine_da_stat(indici, norm, n):
    """
    Scala e traslazione iniziali dell'affine d'ingresso.

    Senza statistiche: identita' (1, 0). Con statistiche: (x - media) / dev per
    ciascun canale, cioe' ingressi standardizzati — la distribuzione su cui gli
    encoder preaddestrati sono nati. `indici` sono gli indici nello stack L4S
    a 14 canali, cosi' ogni slot dell'encoder riceve le statistiche della sua
    banda vera, compresi i casi in cui una banda e' ripetuta (B8 al posto di
    B8A in TerraCodec).

    Resta un'INIZIALIZZAZIONE: i parametri continuano ad apprendere. Cambia
    solo il punto da cui partono.
    """
    if norm is None:
        return torch.ones(1, n, 1, 1), torch.zeros(1, n, 1, 1)
    m, d = norm
    sc = torch.tensor([1.0 / max(float(d[i]), 1e-6) for i in indici])
    sp = torch.tensor([-float(m[i]) / max(float(d[i]), 1e-6) for i in indici])
    return sc.view(1, n, 1, 1), sp.view(1, n, 1, 1)


def _costruisci_terracodec(nome, compression):
    """
    Carica un modello TerraCodec provando piu' nomi di funzione.

    Come per Prithvi non c'e' un solo nome canonico, e se falliscono tutti si
    ELENCA cio' che il pacchetto espone davvero: e' l'unica informazione utile
    per sbloccarsi.
    """
    try:
        import terracodec
    except ImportError as e:
        raise SystemExit(
            "Serve terracodec:  pip install terracodec\n"
            "  oppure: pip install git+https://github.com/IBM/TerraCodec") from e

    candidati = [nome,
                 f"terracodec_v1_{nome}_s2l2a",
                 f"terracodec_v1_{nome}",
                 "terracodec_v1_fp_s2l2a", "terracodec_v1_elic_s2l2a",
                 "terracodec_v1_tt_s2l2a", "terracodec_v1_flex_s2l2a"]
    provati, visti = [], set()
    for c in candidati:
        if c in visti or not isinstance(c, str):
            continue
        visti.add(c)
        fn = getattr(terracodec, c, None)
        if fn is None:
            continue
        provati.append(c)
        for kw in ({"pretrained": True, "compression": compression},
                   {"pretrained": True},
                   {}):
            try:
                m = fn(**kw)
                print(f"  TerraCodec: '{c}' con {kw or 'nessun argomento'}")
                return m, c
            except Exception:
                continue

    esposti = [n for n in dir(terracodec)
               if "terracodec" in n.lower() or "tec" in n.lower()]
    raise SystemExit(
        "Nessuna funzione di costruzione utilizzabile in terracodec.\n"
        "  provate: " + (", ".join(provati) if provati else "nessuna trovata")
        + "\n  il pacchetto espone:\n    " + "\n    ".join(esposti or ["niente"])
        + "\n  Passane una con  --terracodec-variante <nome>")


def _rete_di_analisi(modello):
    """
    Il ramo di ANALISI, cioe' l'encoder del codec.

    In CompressAI, da cui TerraCodec deriva, la trasformata di analisi si
    chiama `g_a` e quella di sintesi `g_s`. Si provano i nomi noti; in ultima
    istanza si cerca il sottomodulo che contiene piu' convoluzioni con stride,
    perche' e' quello che riduce la risoluzione.
    """
    for attr in ("g_a", "encoder", "analysis", "analysis_transform",
                 "g_a_image", "image_encoder"):
        m = getattr(modello, attr, None)
        if isinstance(m, nn.Module):
            return m, attr

    migliore, punteggio, nome = None, 0, None
    for n, m in modello.named_children():
        k = sum(1 for c in m.modules()
                if isinstance(c, nn.Conv2d) and max(c.stride) > 1)
        if k > punteggio:
            migliore, punteggio, nome = m, k, n
    if migliore is None:
        raise SystemExit(
            "Non trovo il ramo di analisi del codec.\n  sottomoduli: "
            + ", ".join(n for n, _ in modello.named_children()))
    return migliore, nome


class TerraCodecEncoder(nn.Module):
    """
    Encoder TerraCodec al posto di Hiera, con piramide VERA.

    LA DIFFERENZA CON LO SCHEMA A SU PRITHVI. Prithvi e' un ViT e produce
    feature a un solo stride: i livelli a /4 e /8 vanno inventati risalendo dal
    profondo, e portano semantica ingrandita invece che dettaglio. TerraCodec
    e' convoluzionale con QUATTRO stadi di stride — /2, /4, /8, /16 — quindi i
    due livelli superficiali esistono davvero e si prendono con degli hook.
    E' l'unico motivo per cui questo trapianto e' preferibile a quello.

    LE BANDE. TerraCodec S2L2A vuole le dodici bande del livello 2A. L4S ne ha
    undici delle dodici: manca B8A, al cui posto va B8, e avanza B10 che il
    livello 2A non ha e che qui si scarta. Undici bande giuste su dodici,
    contro le sei di Prithvi — e fra queste ci sono B5, B6, B7, cioe' il bordo
    rosso che la permutazione indica come seconda dipendenza del modello.

    L'OBIEZIONE CHE RESTA. E' addestrato a comprimere, non a rappresentare: la
    sua loss e' rate-distortion e nulla la spinge verso feature separabili per
    classe. Gli autori riportano meno di un punto di calo su segmentazione a
    380x di riduzione dati, che e' evidenza a favore ma sul loro compito.
    """

    USCITE = (32, 64, 256)

    # Ordine del livello 2A. A destra l'indice nello stack L4S a 14 canali.
    BANDE = [("B1", 0), ("B2", 1), ("B3", 2), ("B4", 3), ("B5", 4), ("B6", 5),
             ("B7", 6), ("B8", 7),
             ("B8A", 7),        # SOSTITUZIONE: L4S non ha B8A, si ripete B8
             ("B9", 8), ("B11", 10), ("B12", 11)]
    # B10 (indice 9) resta fuori: il livello 2A non la contiene, e la nostra
    # stessa misura le da' +0.0040 di permutazione, cioe' niente.

    def __init__(self, optical_bands, variante="fp", compression=10,
                 lato=0, congelato=True, norm=None):
        super().__init__()
        ob = list(optical_bands)
        sel, mancanti = [], []
        for nome_b, idx in self.BANDE:
            if idx in ob:
                sel.append(ob.index(idx))
            else:
                mancanti.append(nome_b)
        if mancanti:
            raise SystemExit(
                "TerraCodec richiede dodici bande e queste non sono in "
                "optical_bands: " + ", ".join(mancanti)
                + f"\n  optical_bands attuali: {ob}")
        self.register_buffer("sel", torch.tensor(sel, dtype=torch.long),
                             persistent=False)

        # Come per Prithvi: la conversione fra la nostra normalizzazione e
        # quella di SSL4EO-S12 non si indovina, si impara. Identita' all'inizio.
        sc, sp = _affine_da_stat([i for _, i in self.BANDE], norm, 12)
        self.scala = nn.Parameter(sc)
        self.sposta = nn.Parameter(sp)

        # `lato` 0 significa "la stessa risoluzione che riceve il decoder".
        # Qui si puo' permettere, a differenza di Prithvi: una rete
        # convoluzionale costa in modo LINEARE nei pixel, non quadratico nei
        # token, e cosi' le tre mappe escono gia' alle dimensioni giuste senza
        # nessun ridimensionamento.
        self.lato = int(lato)

        self.codec, self.nome = _costruisci_terracodec(variante, compression)
        self.analisi, nome_ramo = _rete_di_analisi(self.codec)
        if congelato:
            for q in self.codec.parameters():
                q.requires_grad = False
            self.codec.eval()
        self.congelato = congelato

        # Un hook su OGNI convoluzione con stride: i livelli si sceglieranno
        # poi in base alla dimensione spaziale misurata, non all'indice. Un
        # ordine dedotto dai nomi si romperebbe in silenzio al primo modello
        # con una struttura diversa.
        self._raccolte = []
        for m in self.analisi.modules():
            if isinstance(m, nn.Conv2d) and max(m.stride) > 1:
                m.register_forward_hook(self._raccogli)

        larghezze = self._sonda()
        c4, c8, c16 = self.USCITE
        self.testa4 = nn.Conv2d(larghezze[4], c4, kernel_size=1)
        self.testa8 = nn.Conv2d(larghezze[8], c8, kernel_size=1)
        self.testa16 = nn.Conv2d(larghezze[16], c16, kernel_size=1)

        addestrabili = sum(q.numel() for q in self.parameters()
                           if q.requires_grad)
        print(f"  ENCODER SOSTITUITO: TerraCodec '{self.nome}', ramo di "
              f"analisi '{nome_ramo}'")
        print(f"    livelli VERI presi con hook: "
              + ", ".join(f"/{k}:{v}ch" for k, v in sorted(larghezze.items())))
        print(f"    bande: undici delle dodici del livello 2A "
              f"(B8A assente da L4S, sostituita con B8; B10 scartata)")
        print(f"    teste + affine: {addestrabili:,} parametri addestrabili | "
              f"codec {'congelato' if congelato else 'ADDESTRABILE'}. "
              f"Hiera NON viene chiamato.")

    # ------------------------------------------------------------------
    def _raccogli(self, mod, ing, usc):
        self._raccolte.append(usc)

    @torch.no_grad()
    def _sonda(self, lato=256):
        """
        Una passata a vuoto per scoprire QUANTI canali ha ciascuno stride.

        Necessaria perche' le larghezze intermedie non sono documentate e
        cambiano fra le varianti. Misurarle e' l'unico modo di non cablarle.
        """
        self._raccolte = []
        self.analisi(torch.zeros(1, 12, lato, lato))
        larghezze = {}
        for t in self._raccolte:
            if t.ndim == 4:
                larghezze[lato // t.shape[-1]] = t.shape[1]
        self._raccolte = []
        mancanti = [k for k in (4, 8, 16) if k not in larghezze]
        if mancanti:
            raise SystemExit(
                f"Il ramo di analisi non produce gli stride {mancanti}.\n"
                f"  trovati: {sorted(larghezze.items())}\n"
                f"  Il contratto del decoder di SAM2 richiede 4, 8 e 16.")
        return larghezze

    def forward(self, optical):
        s = optical.shape[-1]
        x = optical.index_select(1, self.sel) * self.scala + self.sposta
        lato = self.lato or s
        if x.shape[-1] != lato:
            x = F.interpolate(x, (lato, lato), mode="bilinear",
                              align_corners=False)

        self._raccolte = []
        ctx = torch.no_grad() if self.congelato else contextlib.nullcontext()
        with ctx:
            self.analisi(x)
        prese = {}
        for t in self._raccolte:
            if t.ndim == 4:
                prese[lato // t.shape[-1]] = t.detach() if self.congelato else t
        self._raccolte = []

        fuori = []
        for k, testa in ((4, self.testa4), (8, self.testa8), (16, self.testa16)):
            f = testa(prese[k])
            if f.shape[-1] != s // k:
                f = F.interpolate(f.float(), (s // k, s // k), mode="bilinear",
                                  align_corners=False).to(optical.dtype)
            fuori.append(f)
        return fuori


class MGEM(nn.Module):
    """
    Multimodal Guided Enhancement Module, da TriGEFNet.

    Le feature dei due stream vengono concatenate in una GUIDA condivisa, e da
    quella guida ciascuno stream ricava la propria attenzione spaziale. Non e'
    una fusione: e' un arricchimento reciproco che lascia due stream distinti.
    Ogni modalita' assorbe contesto dall'altra continuando a esistere per se'.

    DIFFERENZA DALLA NOSTRA DSDF. La DSDF comprime pendenza e DEM in UN canale
    sigmoide che MOLTIPLICA l'ottico all'ingresso: e' a senso unico, a un solo
    canale, e puo' azzerare. Qui la guida e' multicanale, agisce a ogni livello
    dell'FPN, e la forma e' RESIDUA — puo' solo aggiungere.

    IDENTITA' ALL'INIZIO. La scala e' un parametro appreso partito da zero,
    quindi al primo passo l'uscita e' ESATTAMENTE l'ingresso. Serve perche' il
    mask decoder e' preaddestrato sulla distribuzione delle feature di Hiera:
    perturbarla al passo zero costa piu' di quanto il modulo possa rendere
    dopo. E' lo stesso motivo per cui il secondo strato degli adapter parte a
    zero.
    """

    def __init__(self, c_dom, c_aux, nascosto=32):
        super().__init__()
        self.guida = nn.Sequential(
            nn.Conv2d(c_dom + c_aux, nascosto, kernel_size=1, bias=False),
            nn.BatchNorm2d(nascosto), nn.ReLU(inplace=True),
            nn.Conv2d(nascosto, nascosto, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(nascosto), nn.ReLU(inplace=True),
        )
        self.att_dom = nn.Conv2d(nascosto, c_dom, kernel_size=1)
        self.att_aux = nn.Conv2d(nascosto, c_aux, kernel_size=1)
        self.scala_dom = nn.Parameter(torch.zeros(1))
        self.scala_aux = nn.Parameter(torch.zeros(1))

    def forward(self, f_dom, f_aux):
        g = self.guida(torch.cat([f_dom, f_aux], dim=1))
        d = f_dom + self.scala_dom * f_dom * torch.sigmoid(self.att_dom(g))
        a = f_aux + self.scala_aux * f_aux * torch.sigmoid(self.att_aux(g))
        return d, a


class DGFM(nn.Module):
    """
    Dominant-stream Guided Fusion Module, da TriGEFNet.

    Fusione ASIMMETRICA: uno stream comanda, l'altro passa solo dove il primo
    lo consente. Il gate nasce esclusivamente dal dominante, filtra
    l'ausiliario, poi i due si concatenano e una convoluzione 1x1 li riporta
    alla larghezza del dominante.

    PERCHE' L'ASIMMETRIA CALZA QUI PIU' CHE NEL PAPER. Da loro i tre stream
    sono tre ResNet34 equivalenti, e la scelta di quale comandi e' una
    convenzione. Da noi il dominante e' un encoder grande e PREADDESTRATO e
    l'ausiliario e' piccolo e nato da zero: la disparita' che DGFM presuppone
    e' reale, non dichiarata.

    IDENTITA' ALL'INIZIO, per costruzione. La convoluzione di fusione parte
    con l'IDENTITA' sui canali del dominante e ZERO su quelli dell'ausiliario,
    quindi al primo passo restituisce esattamente il dominante. Un'init
    casuale rimescolerebbe 256 canali di feature su cui il mask decoder e'
    preaddestrato, e il modello ripartirebbe da sotto zero.
    """

    def __init__(self, c_dom, c_aux, nascosto=32):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(c_dom, nascosto, kernel_size=1, bias=False),
            nn.BatchNorm2d(nascosto), nn.ReLU(inplace=True),
            nn.Conv2d(nascosto, c_aux, kernel_size=1),
        )
        self.fondi = nn.Conv2d(c_dom + c_aux, c_dom, kernel_size=1)
        with torch.no_grad():
            self.fondi.weight.zero_()
            self.fondi.bias.zero_()
            for i in range(c_dom):
                self.fondi.weight[i, i, 0, 0] = 1.0

    def forward(self, f_dom, f_aux):
        return self.fondi(torch.cat([f_dom,
                                     f_aux * torch.sigmoid(self.gate(f_dom))],
                                    dim=1))


class SkipGate(nn.Module):
    """
    Filtra una skip connection del decoder di SAM2 con cio' che il percorso
    PROFONDO ha capito. E' il GSRM di TriGEFNet, adattato.

    DOVE. Sam2MaskDecoder risale di risoluzione cosi' (modeling_sam2.py:1208):

        upscaled = upscale_conv1(image_embeddings) + feat_s1
        upscaled = act(upscale_conv2(upscaled) + feat_s0)

    `feat_s0` e `feat_s1` sono le mappe FPN a stride 4 e 8, sommate SENZA
    alcun filtro: la tessitura superficiale contribuisce sempre, anche dove il
    livello semantico non vede niente.

    PERCHE' PROPRIO QUESTO. E' il profilo del difetto misurato oggi sul val:
    la IoU sulle patch vuote e' rimasta IDENTICA (0.9069) per ogni soglia fra
    0.20 e 0.75, cioe' i falsi positivi hanno probabilita' sopra 0.75 — sono
    SICURI, e nessuna soglia puo' toglierli. Una skip non filtrata che porta
    dettaglio dentro il decoder anche dove il percorso profondo tace e' una
    spiegazione plausibile di quel comportamento, e questo gate ne e' il test.

    DIFFERENZA DAL PAPER, dichiarata. Loro generano il gate dalla feature del
    DECODER. Prenderla richiederebbe riscrivere il forward di transformers;
    qui il gate nasce dalla mappa FPN piu' profonda, che e' la stessa
    informazione semantica ed e' gia' nelle nostre mani prima di chiamare il
    decoder.

    INIZIALIZZAZIONE. Pesi a zero e bias a +4: sigmoid(4) = 0.982, cioe' il
    gate parte quasi COMPLETAMENTE APERTO e puo' solo imparare a chiudersi.
    Un bias a 0 darebbe sigmoid = 0.5 e dimezzerebbe le skip fin dal primo
    passo, cambiando la distribuzione su cui il decoder preaddestrato lavora.
    E' lo stesso motivo per cui il forget gate di una LSTM si inizializza
    positivo.
    """

    def __init__(self, c_shallow, c_deep, nascosto=32, bias_iniziale=4.0):
        super().__init__()
        self.g = nn.Sequential(
            nn.Conv2d(c_deep, nascosto, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nascosto, c_shallow, kernel_size=1),
        )
        nn.init.zeros_(self.g[-1].weight)
        nn.init.constant_(self.g[-1].bias, bias_iniziale)

    def forward(self, shallow, deep):
        d = F.interpolate(deep.float(), shallow.shape[-2:], mode="bilinear",
                          align_corners=False).to(shallow.dtype)
        return shallow * torch.sigmoid(self.g(d))


# ============================================================
# MODEL: LENTE  (adattivo — espande il patch embed solo se n_optical != 3)
# ============================================================

def _e_blocco_hiera(name):
    """`backbone.blocks.7` si', `backbone.blocks.7.attn` no."""
    parti = name.split(".")
    return len(parti) == 3 and parti[1] == "blocks" and parti[2].isdigit()


class _BloccoConv(nn.Module):
    """Due conv 3x3 con BatchNorm e GELU, con connessione residua."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out), nn.GELU(),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
        )
        self.salto = (nn.Identity() if c_in == c_out
                      else nn.Conv2d(c_in, c_out, 1, bias=False))
        self.att = nn.GELU()

    def forward(self, x):
        return self.att(self.conv(x) + self.salto(x))


class UNetEncoder(nn.Module):
    """
    Percorso contrattivo a U-Net che PRODUCE LO STESSO CONTRATTO dell'encoder
    di SAM2, cosi' il mask decoder resta identico e il confronto e' controllato:

        livello 0   (B,  32, S/4,  S/4)
        livello 1   (B,  64, S/8,  S/8)
        livello 2   (B, 256, S/16, S/16)   <- dove il decoder cross-attende

    PERCHE'. Su L4S `con frana` e' fermo attorno a 0.50 attraverso prompt,
    adapter, loss, risoluzione e scongelamento parziale. Tutti i modelli che
    vincono su questo dataset sono encoder-decoder ADDESTRATI PER INTERO vicino
    alla risoluzione nativa, mentre qui l'encoder e' Hiera preaddestrato su
    video RGB naturali e congelato. Questo modulo isola quella variabile: stesso
    decoder, stessa loss, stesso split, stesso tutto — cambia solo chi produce
    le feature.

    Il risultato e' informativo in entrambe le direzioni. Se un encoder piccolo
    addestrato da zero batte Hiera, il prior del foundation e' quello sbagliato
    per questo dominio. Se perde, il prior vale nonostante il disallineamento di
    dominio, ed e' una conclusione altrettanto riportabile.

    Volutamente PICCOLO: il punto non e' avere piu' capacita' — su questi
    dataset ne abbiamo misurato il danno tre volte — ma avere capacita'
    ADDESTRABILE nel posto giusto.
    """

    # canali dei tre livelli, imposti dal decoder di SAM2
    USCITE = (32, 64, 256)

    def __init__(self, in_ch, larghezza=48):
        super().__init__()
        w = larghezza
        # stem: S -> S/2. Niente pooling, la stride sta nella conv.
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, w, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w), nn.GELU(),
        )
        self.giu1 = nn.Sequential(nn.Conv2d(w, w * 2, 3, stride=2, padding=1,
                                            bias=False),
                                  nn.BatchNorm2d(w * 2), nn.GELU(),
                                  _BloccoConv(w * 2, w * 2))        # S/4
        self.giu2 = nn.Sequential(nn.Conv2d(w * 2, w * 4, 3, stride=2, padding=1,
                                            bias=False),
                                  nn.BatchNorm2d(w * 4), nn.GELU(),
                                  _BloccoConv(w * 4, w * 4))        # S/8
        self.giu3 = nn.Sequential(nn.Conv2d(w * 4, w * 8, 3, stride=2, padding=1,
                                            bias=False),
                                  nn.BatchNorm2d(w * 8), nn.GELU(),
                                  _BloccoConv(w * 8, w * 8))        # S/16
        # proiezioni ai canali che il decoder si aspetta
        self.p0 = nn.Conv2d(w * 2, self.USCITE[0], 1)
        self.p1 = nn.Conv2d(w * 4, self.USCITE[1], 1)
        self.p2 = nn.Conv2d(w * 8, self.USCITE[2], 1)

    def forward(self, x):
        x = self.stem(x)
        f0 = self.giu1(x)
        f1 = self.giu2(f0)
        f2 = self.giu3(f1)
        return [self.p0(f0), self.p1(f1), self.p2(f2)]


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


class LoRALinear(nn.Module):
    """
    Aggiornamento a rango basso di un Linear CONGELATO:  y = W x + (alpha/r) B A x

    DIFFERENZA DAGLI ADAPTER, che e' il motivo per cui vale la pena provarlo.
    L'Adapter e' un ramo residuo applicato all'USCITA del blocco di attenzione:
    prende feature gia' calcolate e le ritocca. LoRA invece corregge i PESI con
    cui l'attenzione viene calcolata — cambia cosa il blocco guarda, non come ne
    rifinisce il risultato. Su un encoder pre-addestrato su video RGB naturali e
    riusato su 12 bande Sentinel, il disallineamento sta proprio li'.

    B parte da ZERO, quindi all'inizio il modello e' identico a prima e puo'
    solo migliorare. Il gradiente arriva comunque ad A: d(out)/dB = A x != 0,
    quindi B si stacca da zero al primo passo e da li' in poi anche A si muove.
    E' lo stesso accorgimento degli Adapter, dove funziona.

    Costo: per un Linear d -> 3d (il qkv), LoRA aggiunge r*(d + 3d) = 4rd
    parametri. Con r=8 sono ~0.2M in tutto l'encoder, cioe' DIECI VOLTE MENO
    dei 2.42M di adapter che gia' abbiamo — ma applicati dove contano di piu'.
    """

    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.normal_(self.A.weight, std=1.0 / r)
        nn.init.zeros_(self.B.weight)
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + self.scaling * self.B(self.A(x))


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


def resolve_prompt(prompt_source, boxes, batch_size, size, device, dtype=torch.float32):
    """
    Traduce `prompt_source` nel tensore da passare come `input_boxes` a SAM2.

    Ritorna None per il regime prompt-free: in transformers 5.8.1 Sam2Model
    gestisce nativamente `input_boxes=None` + `input_points=None` inserendo un
    punto vuoto con label -1, che il prompt encoder mappa su `not_a_point_embed`.
    Non serve toccare il prompt encoder.

    `boxes` viene usato SOLO da "gt_box"; negli altri regimi e' ignorato, cosi'
    i loop di training possono continuare a passarlo senza sapere nulla del
    regime attivo.
    """
    if prompt_source in ("gt_box", "yolo_box", "clf_box"):
        # boxes=None e' LECITO: e' il regime "senza prompt", usato quando YOLO
        # non propone nulla. In quel caso il campione NON riceve un box finto
        # (che sarebbe un bias: il box a tutta immagine finirebbe sia sulle
        # patch vuote sia sui positivi mancati dal detector, insegnando al
        # modello ad arrendersi proprio dove dovrebbe provarci) ma nessun
        # prompt, cioe' il regime prompt-free vero.
        if boxes is None:
            return None
        return boxes.unsqueeze(1) if boxes.ndim == 2 else boxes
    if prompt_source == "image_box":
        # Stesso box per ogni campione: nessuna informazione sul dove, ma il
        # decoder riceve comunque un box come nel pretraining.
        b = torch.tensor([0.0, 0.0, float(size - 1), float(size - 1)],
                         device=device, dtype=dtype)
        return b.view(1, 1, 4).expand(batch_size, 1, 4).contiguous()
    if prompt_source == "none":
        return None
    raise ValueError(f"prompt_source sconosciuto: {prompt_source!r} (attesi: {PROMPT_SOURCES})")


def _build_sam2_image_config(sam_ckpt_path, image_size):
    """
    Config di Sam2Model alla risoluzione voluta.

    Gemella di `_build_sam2_config` in Sen12Landslides/lente_video.py:345, che
    fa la stessa cosa per Sam2VideoConfig. Qui manca `memory_attention_rope_
    feat_sizes`: e' la griglia del RoPE della memory attention, che nel modello
    IMAGE non esiste.

    SAM2 non e' vincolato ai 1024 del checkpoint: i position embedding di Hiera
    sono griglie minuscole (1,96,7,7) e (1,96,8,8) che vengono interpolate alla
    dimensione effettiva della feature map, quindi NESSUN peso cambia forma e il
    checkpoint si carica identico. Restano pero' due campi cablati sulla
    risoluzione che, se non aggiornati, fanno fallire il forward con errori di
    shape lontani dalla causa:

      * vision_config.backbone_feature_sizes -> come il modello rimodella le FPN
        in `_prepare_vision_features`; a 1024 vale [[256,256],[128,128],[64,64]],
        cioe' image_size/4, /8, /16. NON e' derivato da image_size.
      * image_size -> upsample delle maschere ad alta risoluzione e
        normalizzazione delle coordinate dei box nel prompt encoder.

    PERCHE' 512 su L4S (le patch native sono 128x128). Il mask decoder emette a
    image_size/4:
        1024 -> maschere 256x256  = upsample inutile, 4x tempo e memoria
         512 -> maschere 128x128  = ESATTAMENTE la risoluzione nativa
         256 -> maschere  64x64   = meta' del nativo, si supervisiona sotto il dato
    A 512 la loss e la IoU si calcolano sulla griglia vera invece che su pixel
    interpolati, e l'epoca costa un quarto.
    """
    if Sam2Config is None:
        raise RuntimeError(
            "transformers non espone Sam2Config: impossibile cambiare la "
            f"risoluzione di SAM2. Rimetti target_size: 1024 nel config.")
    cfg = Sam2Config.from_pretrained(sam_ckpt_path)
    s = int(image_size)
    cfg.image_size = s
    vc = getattr(cfg, "vision_config", None)
    if vc is None:
        raise RuntimeError(
            f"Sam2Config di {sam_ckpt_path} non ha vision_config: non si puo' "
            f"ridimensionare la griglia delle FPN. Rimetti target_size: 1024.")
    vc.image_size = s
    touched = ["image_size", "vision_config.image_size"]
    if hasattr(vc, "backbone_feature_sizes"):
        vc.backbone_feature_sizes = [[s // 4] * 2, [s // 8] * 2, [s // 16] * 2]
        touched.append("vision_config.backbone_feature_sizes")
    for sub in ("prompt_encoder_config", "mask_decoder_config"):
        sc = getattr(cfg, sub, None)
        if sc is not None and hasattr(sc, "image_size"):
            sc.image_size = s
            touched.append(f"{sub}.image_size")
    # Stampato apposta: se un campo cablato sulla risoluzione NON compare qui,
    # il forward fallira' con un errore di shape lontano dalla causa.
    print(f"  SAM2 ricostruito a {s}x{s} — campi aggiornati: {', '.join(touched)}")
    return cfg


class FPNCompleto(nn.Module):
    """
    Il ramo top-down che a SAM2 manca sotto S/16.

    PERCHE'. In sam2-hiera-tiny `fpn_top_down_levels` vale [2, 3]: la semantica
    profonda scende solo fino a S/16. Le mappe a S/8 e S/4 sono proiezioni
    laterali degli stadi 1 e 2 di Hiera (1 e 3 blocchi di profondita'), senza
    nessun contributo dall'alto: portano bordi e tessitura, non il concetto di
    frana. La sonda `cella` ha misurato la conseguenza: una frana che sta tutta
    in una cella S/16 viene presa nel 55% dei casi, una che ne tocca cinque o
    piu' nel 95%. Sulle maschere del test, per le frane sotto i 16 px la cella
    piu' piena e' frana solo al 28%: alla risoluzione dove SAM2 decide, il loro
    segnale e' minoranza.

    COSA FA. Come nell'FPN di Lin et al. (2017): la mappa piu' profonda viene
    ingrandita, proiettata ai canali del livello sotto e SOMMATA, un gradino
    alla volta, S/16 -> S/8 -> S/4. Cosi' le mappe che il decoder di SAM2 (e
    H-SAM, se c'e') usa per risalire di risoluzione portano anche semantica.

    DIFFERENZA CON GSRM, che ha dato un risultato nullo: GSRM MOLTIPLICAVA la
    skip per un gate, quindi poteva solo spegnere tessitura dove il livello
    profondo taceva. Qui si AGGIUNGE semantica dove la tessitura c'e'.

    PARTE DALL'IDENTITA'. L'ultima convoluzione di ogni proiezione e' a zero:
    al primo passo le mappe sono esattamente quelle di SAM2, e il decoder
    preaddestrato riceve la distribuzione che conosce.
    """

    def __init__(self, canali=(32, 64, 256)):
        super().__init__()
        c4, c8, c16 = canali

        def proiezione(ci, co):
            p = nn.Sequential(nn.Conv2d(ci, co, 1), nn.GELU(),
                              nn.Conv2d(co, co, 3, padding=1))
            nn.init.zeros_(p[-1].weight)
            nn.init.zeros_(p[-1].bias)
            return p
        self.a8 = proiezione(c16, c8)
        self.a4 = proiezione(c8, c4)

    @staticmethod
    def _su(x, rif):
        return F.interpolate(x.float(), rif.shape[-2:], mode="bilinear",
                             align_corners=False).to(rif.dtype)

    def forward(self, emb):
        f4, f8, f16 = emb
        f8 = f8 + self.a8(self._su(f16, f8))
        f4 = f4 + self.a4(self._su(f8, f4))
        return [f4, f8, f16]


class HSAMDecoder(nn.Module):
    """
    Secondo stadio alla H-SAM (Cheng et al., CVPR 2024), senza prompt.

    PERCHE'. SAM decide sulla griglia a stride 16: a 512 di ingresso da 128
    nativi la frana mediana di Sen12 occupa UNA cella di quella griglia. Le
    riletture (`refine_passes`) rimandano la maschera dentro lo STESSO decoder,
    quindi decidono ancora li'. Qui la maschera del primo stadio diventa una
    GUIDA per un decoder nuovo che risale fino a stride 4.

    I TRE PEZZI DEL PAPER, nella versione che usiamo:
      1. self-attention guidata dalla maschera e bilanciata per classe. Due
         termini additivi sui punteggi, ciascuno con un peso appreso che parte
         da ZERO: `guida` rafforza l'attenzione fra celle della stessa classe
         prevista; `bilancia` ripesa le chiavi perche' le due classi abbiano la
         stessa massa totale, cosi' le poche celle di frana non vengono
         sommerse dallo sfondo (97,5% dei pixel su L4S);
      2. cross-attention apprendibile verso due PROTOTIPI, frana e sfondo,
         ciascuno mediato sulla sua classe (bilanciato anche lui per
         costruzione) piu' un embedding di classe appreso;
      3. pixel decoder gerarchico: stride 16 -> 8 -> 4, a ogni gradino le
         feature FPN di Hiera alla stessa scala e la maschera a priori.

    PARTE DALL'IDENTITA'. L'uscita e' `priore + delta` con la convoluzione di
    `delta` inizializzata a zero: al primo passo il modello e' esattamente
    quello senza H-SAM, come ogni modulo nuovo di questo progetto.

    Oltre `lato_max` celle per lato l'attenzione lavora su una griglia mediata:
    a 1024 di ingresso sarebbero 4096 token e una matrice di attenzione da
    16,7 milioni di valori per testa.
    """

    def __init__(self, canali=(32, 64, 256), dim=128, teste=4, lato_max=32):
        super().__init__()
        c4, c8, c16 = canali
        self.teste, self.lato_max = teste, lato_max
        self.proj = nn.Conv2d(c16, dim, 1)
        self.n1, self.n2, self.n3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.att = nn.ModuleList([nn.Linear(dim, dim) for _ in range(4)])     # q k v o
        self.xatt = nn.ModuleList([nn.Linear(dim, dim) for _ in range(4)])
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.classi = nn.Parameter(torch.zeros(2, dim))
        self.guida = nn.Parameter(torch.zeros(1))
        self.bilancia = nn.Parameter(torch.zeros(1))

        def blocco(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU(),
                                 nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU())
        self.su8 = nn.ConvTranspose2d(dim, 64, 2, stride=2)
        self.f8 = blocco(64 + c8 + 1, 64)
        self.su4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.f4 = blocco(32 + c4 + 1, 32)
        self.uscita = nn.Conv2d(32, 1, 1)
        nn.init.zeros_(self.uscita.weight)
        nn.init.zeros_(self.uscita.bias)

    def _mha(self, lin, q_in, kv_in, bias=None):
        """Attenzione a mano: punteggi e softmax in float32, qualunque sia l'autocast."""
        B, N, C = q_in.shape
        M, d = kv_in.shape[1], C // self.teste
        q = lin[0](q_in).view(B, N, self.teste, d).transpose(1, 2)
        k = lin[1](kv_in).view(B, M, self.teste, d).transpose(1, 2)
        v = lin[2](kv_in).view(B, M, self.teste, d).transpose(1, 2)
        a = (q @ k.transpose(-1, -2)).float() / d ** 0.5
        if bias is not None:
            a = a + bias.unsqueeze(1)
        o = (a.softmax(-1).to(v.dtype) @ v).transpose(1, 2).reshape(B, N, C)
        return lin[3](o)

    @staticmethod
    def _a(x, rif):
        if x.shape[-2:] == rif.shape[-2:]:
            return x
        return F.interpolate(x.float(), rif.shape[-2:], mode="bilinear",
                             align_corners=False).to(x.dtype)

    def forward(self, emb, priore):
        f4, f8, f16 = emb
        x = self.proj(f16)
        B, C, H, W = x.shape
        passo = max(1, H // self.lato_max)
        xa = F.avg_pool2d(x, passo) if passo > 1 else x
        h, w = xa.shape[-2:]
        p = torch.sigmoid(F.interpolate(priore.float(), (h, w), mode="bilinear",
                                        align_corners=False)).flatten(1)      # (B, N)
        t = xa.flatten(2).transpose(1, 2)                                     # (B, N, C)

        # 1. self-attention guidata e bilanciata
        s = 2.0 * p - 1.0
        m_fr = p.mean(1, keepdim=True).clamp_min(1e-3)
        m_sf = (1.0 - p).mean(1, keepdim=True).clamp_min(1e-3)
        peso = 0.5 * p / m_fr + 0.5 * (1.0 - p) / m_sf                      # media 1
        bias = (self.guida * s.unsqueeze(2) * s.unsqueeze(1)
                + self.bilancia * torch.log(peso.clamp_min(1e-4)).unsqueeze(1))
        q = self.n1(t)
        t = t + self._mha(self.att, q, q, bias)

        # 2. cross-attention verso i prototipi di classe
        fr = (p.unsqueeze(2) * t).sum(1) / p.sum(1, keepdim=True).clamp_min(1e-3)
        sf = ((1.0 - p).unsqueeze(2) * t).sum(1) / (1.0 - p).sum(1, keepdim=True).clamp_min(1e-3)
        proto = torch.stack([fr, sf], 1) + self.classi
        t = t + self._mha(self.xatt, self.n2(t), proto.to(t.dtype))
        t = t + self.mlp(self.n3(t))

        y = t.transpose(1, 2).reshape(B, C, h, w)
        y = x + self._a((y - xa).to(x.dtype), x)       # torna alla griglia piena

        # 3. pixel decoder gerarchico
        z = self._a(self.su8(y), f8)
        z = self.f8(torch.cat([z, f8.to(z.dtype), self._a(priore, f8).to(z.dtype)], 1))
        z = self._a(self.su4(z), f4)
        z = self.f4(torch.cat([z, f4.to(z.dtype), self._a(priore, f4).to(z.dtype)], 1))
        return self._a(priore, f4) + self.uscita(z).to(priore.dtype)


class LENTE(nn.Module):
    """
    SAM2-Tiny + Adapter + DSDF early guidance.
    n_optical = canali ottici (12 per L4S, 3 per Bijie). Se != 3 il patch embed
    viene espanso copiando i pesi RGB; se == 3 viene solo scongelato.
    Il terrain (2ch) modula l'ottico via guidance map.

    prompt_source decide cosa riceve il mask decoder: vedi resolve_prompt() e il
    commento in config.yaml. Il default e' "none" (prompt-free), non "gt_box":
    quest'ultimo va chiesto esplicitamente, perche' e' il regime in cui il
    prompt contiene la risposta.
    """
    def __init__(self, sam_ckpt_path, n_optical=12, adapter_dim=64, prompt_source="none",
                 image_size=1024, unfreeze_prompt_encoder=False,
                 lora_rank=0, lora_alpha=16.0, adapter_where="attn",
                 unfreeze_blocks=0, encoder="hiera", unet_width=48, gsrm=False,
                 fusione="none", fusione_width=24, fusione_aux="terreno",
                 optical_bands=None,
                 prithvi_model="ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
                 prithvi_size=512, prithvi_frozen=True,
                 prithvi_adapter_dim=0, prithvi_lora=0,
                 tec_variante="fp", tec_compression=10, tec_size=0,
                 tec_frozen=True, norm_ingresso=None, hsam=False, fpn_completo=False):
        super().__init__()
        if prompt_source not in PROMPT_SOURCES:
            raise ValueError(f"prompt_source sconosciuto: {prompt_source!r} (attesi: {PROMPT_SOURCES})")
        self.n_optical = n_optical
        self._unfreeze_blocks = int(unfreeze_blocks)
        self.encoder_params = []
        self.prompt_source = prompt_source
        self.unfreeze_prompt_encoder = bool(unfreeze_prompt_encoder)
        # `image_size` DEVE combaciare con cfg["target_size"], la dimensione a cui
        # i loop interpolano le immagini: e' lo stesso numero, e va passato qui
        # perche' il prompt encoder e le FPN si costruiscono su quel valore.
        self.image_size = int(image_size)
        if self.image_size == 1024:
            # percorso storico: nessuna riscrittura di config, cosi' i checkpoint
            # gia' addestrati restano bit-identici come sono sempre stati caricati.
            self.model = Sam2Model.from_pretrained(sam_ckpt_path)
        else:
            self.model = Sam2Model.from_pretrained(
                sam_ckpt_path, config=_build_sam2_image_config(sam_ckpt_path, self.image_size))
        # Con multimask_output=False, in eval() SAM2 sostituisce la maschera 0
        # con la migliore delle maschere 1-3 quando la 0 e' "instabile". In
        # train() esce sempre la 0, quindi le teste 1-3 non ricevono mai
        # gradiente e restano quelle preaddestrate: la valutazione usciva da
        # teste mai addestrate su 167-296 patch su 464-760 (misurato il
        # 2026-11-04, val micro 0.5869 -> 0.5986 spegnendola). Spenta: in
        # valutazione esce la stessa maschera dell'addestramento.
        self.model.mask_decoder.dynamic_multimask_via_stability = False
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
                        if n_optical > 3:
                            # Piu' di tre bande: le prime tre ereditano i pesi
                            # RGB preaddestrati, le altre partono da rumore e
                            # devono imparare da zero sotto un encoder congelato.
                            new_conv.weight[:, :3] = module.weight.clone()
                            nn.init.normal_(new_conv.weight[:, 3:], mean=0.0, std=0.001)
                        else:
                            # MENO di tre bande. Copiare i primi n canali RGB
                            # ridurrebbe di 3/n la scala della risposta, perche'
                            # l'uscita e' la SOMMA sui canali d'ingresso: il
                            # primo strato uscirebbe smorzato e tutto cio' che
                            # sta a valle e' congelato e non puo' compensare.
                            # Si distribuisce invece la somma dei tre pesi sui
                            # canali disponibili, che conserva la scala quando
                            # le bande sono simili e resta una partenza sensata
                            # quando non lo sono.
                            somma = module.weight.sum(dim=1, keepdim=True)
                            new_conv.weight.copy_(
                                somma.repeat(1, n_optical, 1, 1) / n_optical)
                        if module.bias is not None:
                            new_conv.bias.copy_(module.bias)
                    parts  = name.split(".")
                    parent = v_encoder.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else v_encoder
                    setattr(parent, parts[-1], new_conv)
                    self.patch_proj = new_conv
                break
        if self.patch_proj is None:
            raise RuntimeError("Patch embed Conv2d(in=3) non trovato.")

        # ── canale DENSO del prompt (input_masks) ────────────────────────
        # Sparse e denso sono indipendenti nel prompt encoder di SAM2: il primo
        # sono DUE token d'angolo accodati alla sequenza, il secondo e' una
        # mappa che viene sommata pixel per pixel all'image embedding, cioe'
        # tocca tutte le ~4096 posizioni della griglia. Tre ordini di grandezza
        # di banda in piu', ed e' l'unico canale in cui piu' box (o una forma
        # non rettangolare) possono coesistere.
        #
        # `mask_embed` e' una conv, quindi l'AMPIEZZA dei valori conta: una
        # mappa tarata come logit non ha per forza la scala giusta per quel
        # canale. exp(0)=1 all'inizio, poi la rete se la aggiusta.
        # NB: va messa in un param group VELOCE (vedi LENTE_train), non a 1e-5
        # con i pesi pretrained — parte da zero come una testa nuova.
        self.dense_log_scale = nn.Parameter(torch.zeros(1))
        # Scala separata per il RAFFINAMENTO (la maschera che il modello ha
        # predetto al passaggio precedente, rimandata dentro). Separata e non
        # condivisa con dense_log_scale perche' le due mappe hanno origini
        # diverse — una viene da un detector, l'altra da SAM stesso — e il
        # modello deve poterle pesare in modo indipendente, fino ad azzerarne
        # una se non serve.
        self.refine_log_scale = nn.Parameter(torch.zeros(1))
        # Scala del SEGNALE DI PRESENZA: la probabilita' che un classificatore
        # binario assegna alla patch, data a SAM come mappa uniforme.
        #
        # E' la ricostruzione legittima di un meccanismo gia' misurato. Nel
        # regime `gt_box`, su una patch vuota `_bbox_from_mask` restituiva
        # sempre la costante [0,0,10,10]: il modello imparava "questo pattern ->
        # non disegnare niente" e prendeva vuote = 1.0000 ESATTO su 306 patch.
        # Il 76% del vantaggio di quel regime veniva da li', ed era un leak,
        # perche' il segnale nasceva dalla ground truth.
        # Qui la stessa informazione arriva da un modello addestrato, quindi e'
        # disponibile anche a inferenza.
        self.clf_log_scale = nn.Parameter(torch.zeros(1))
        self.mask_input_size = tuple(self.model.prompt_encoder.mask_input_size)

        # Secondo stadio H-SAM (vedi HSAMDecoder). Creato dopo _freeze(), come
        # gli altri moduli nuovi, quindi resta addestrabile. `ultimo_prior` e'
        # la maschera del PRIMO stadio dell'ultima forward: LENTE_train la
        # supervisiona a parte, come fa il paper.
        # FPN completo (vedi FPNCompleto): prima di H-SAM e del decoder.
        self.fpn = FPNCompleto() if fpn_completo else None
        self.fpn_params = list(self.fpn.parameters()) if fpn_completo else []
        if fpn_completo:
            print(f"  FPN completo: semantica fino a S/4, "
                  f"{sum(q.numel() for q in self.fpn_params):,} parametri, "
                  f"parte dall'identita'")
        self.hsam = HSAMDecoder() if hsam else None
        self.hsam_params = list(self.hsam.parameters()) if hsam else []
        self.ultimo_prior = None
        if hsam:
            print(f"  H-SAM: secondo stadio, {sum(q.numel() for q in self.hsam_params):,} "
                  f"parametri, parte dall'identita'")

        # Creato DOPO _freeze() del padre: i suoi parametri restano
        # requires_grad=True senza doverli riaccendere.
        # Lo slot vale per QUALUNQUE encoder sostitutivo, non solo la U-Net:
        # il nome e' rimasto quello del primo che ci e' passato. Il percorso a
        # valle — il ramo `image_embeddings` nel forward e il gruppo di
        # ottimizzazione dedicato — e' lo stesso per tutti.
        self.unet_encoder = None
        if encoder == "unet":
            self.unet_encoder = UNetEncoder(n_optical, larghezza=unet_width)
            print(f"  ENCODER SOSTITUITO: U-Net larghezza {unet_width}, "
                  f"{sum(q.numel() for q in self.unet_encoder.parameters()):,} "
                  f"parametri addestrabili. Hiera NON viene chiamato.")
        elif encoder == "terracodec":
            self.unet_encoder = TerraCodecEncoder(
                optical_bands, variante=tec_variante, compression=tec_compression,
                lato=tec_size, congelato=tec_frozen, norm=norm_ingresso)
        elif encoder == "prithvi":
            self.unet_encoder = PrithviEncoder(
                optical_bands, nome=prithvi_model, lato=prithvi_size,
                congelato=prithvi_frozen, adapter_dim=prithvi_adapter_dim,
                adapter_where=adapter_where, lora_rank=prithvi_lora,
                lora_alpha=lora_alpha, norm=norm_ingresso)

        # adapter_dim <= 0 disattiva gli Adapter: serve per l'ablazione
        # "solo LoRA", cosi' i due meccanismi si possono confrontare da soli.
        # DOVE agganciare gli Adapter. Ogni blocco transformer ha attenzione E
        # MLP, e l'MLP e' dove avviene la maggior parte della trasformazione.
        # Lo schema originale (Houlsby) ne mette DUE per blocco, attenzione +
        # MLP; la variante Pfeiffer uno solo dopo l'MLP. Questo codice ne
        # metteva uno solo dopo l'ATTENZIONE — cioe' nessuno dei due — e
        # lasciava i 12 MLP di Hiera completamente non adattati.
        #
        #   "attn"  solo attenzione   comportamento storico, DEFAULT: i run
        #                             gia' fatti restano bit-identici
        #   "mlp"   solo MLP          (Pfeiffer)
        #   "both"  entrambi          (Houlsby)
        #
        # L'uscita dell'MLP viene sommata al residuo FUORI dal modulo, quindi
        # l'hook qui modifica il contributo dell'MLP prima della somma: e'
        # esattamente la posizione di Houlsby.
        #
        # NOTA: LoRA resta agganciata alla SOLA attenzione in ogni caso — sono
        # i Linear delle proiezioni q/k/v, e negli MLP non esistono.
        if adapter_where not in ("attn", "mlp", "both"):
            raise ValueError(f"adapter_where sconosciuto: {adapter_where!r} "
                             f"(attesi: 'attn', 'mlp', 'both')")
        usa_attn = adapter_where in ("attn", "both")
        usa_mlp = adapter_where in ("mlp", "both")
        self.adapter_where = adapter_where
        self.n_adapter_attn = 0
        self.n_adapter_mlp = 0

        self.adapters = nn.ModuleList()
        self.lora = nn.ModuleList()
        for name, module in v_encoder.named_modules():
            if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear):
                dim = module.proj.out_features if hasattr(module, "proj") else module.qkv.in_features
                if adapter_dim > 0 and usa_attn:
                    ad = Adapter(dim, adapter_dim); self.adapters.append(ad)
                    module.register_forward_hook(self._make_hook(ad))
                    self.n_adapter_attn += 1
                if lora_rank > 0:
                    self._apply_lora(module, ("qkv", "proj"), lora_rank, lora_alpha)
            elif hasattr(module, "q_proj") and isinstance(module.q_proj, nn.Linear):
                dim = module.out_proj.out_features if hasattr(module, "out_proj") else module.q_proj.in_features
                if adapter_dim > 0 and usa_attn:
                    ad = Adapter(dim, adapter_dim); self.adapters.append(ad)
                    module.register_forward_hook(self._make_hook(ad))
                    self.n_adapter_attn += 1
                if lora_rank > 0:
                    self._apply_lora(module, ("q_proj", "k_proj", "v_proj", "out_proj"),
                                     lora_rank, lora_alpha)
            elif adapter_dim > 0 and usa_mlp and name.endswith(".mlp"):
                # larghezza del blocco = uscita dell'ULTIMO Linear dell'MLP
                lin = [m for m in module.modules() if isinstance(m, nn.Linear)]
                if not lin:
                    continue
                ad = Adapter(lin[-1].out_features, adapter_dim)
                self.adapters.append(ad)
                module.register_forward_hook(self._make_hook(ad))
                self.n_adapter_mlp += 1

        self._freeze()

        # GSRM: due gate sulle skip del decoder. Costruiti DOPO _freeze, cosi'
        # i loro parametri restano addestrabili senza eccezioni da aggiungere
        # li' dentro. I canali non sono cablati: upscale_conv1 va da
        # hidden_size a hidden_size//4 e upscale_conv2 a hidden_size//8,
        # quindi feat_s1 e feat_s0 hanno esattamente quelle due larghezze.
        self.gsrm_params = []
        self.gate_s0 = self.gate_s1 = None
        if gsrm:
            h = self.model.config.mask_decoder_config.hidden_size
            self.gate_s0 = SkipGate(h // 8, h)
            self.gate_s1 = SkipGate(h // 4, h)
            self.gsrm_params = (list(self.gate_s0.parameters())
                                + list(self.gate_s1.parameters()))
            print(f"  GSRM attivo: gate sulle skip a stride 4 ({h//8} canali) "
                  f"e 8 ({h//4} canali) | "
                  f"{sum(q.numel() for q in self.gsrm_params):,} parametri")

        # ---- fusione a due stream, da TriGEFNet ----------------------------
        # Lo stream ausiliario vede SOLO il terreno e produce le stesse tre
        # mappe FPN del dominante, cosi' MGEM e DGFM possono lavorare livello
        # per livello. La DSDF resta accesa: la permutazione ha misurato che il
        # modello si appoggia alla pendenza per +0.2055, ed e' la terza
        # dipendenza dopo B4 — spegnerla mentre si aggiunge uno stream nuovo
        # cambierebbe due cose insieme.
        self.fusione = fusione
        self.aux_encoder = None
        self.mgem = self.dgfm = None
        self.fusione_params = []
        # Su QUALI livelli agiscono i moduli, e cosa vede l'ausiliario.
        # Con la U-Net sul terreno tutti e tre; con Prithvi solo il profondo,
        # perche' un ViT a patch 16 non ha livelli superficiali e inventarli
        # e' proprio il difetto che lo schema C evita.
        self.fusione_aux = fusione_aux
        self.fusione_livelli = (2,) if fusione_aux == "prithvi" else (0, 1, 2)
        if fusione != "none":
            if fusione_aux == "prithvi":
                self.aux_encoder = PrithviEncoder(
                    optical_bands, nome=prithvi_model, lato=prithvi_size,
                    congelato=prithvi_frozen, adapter_dim=prithvi_adapter_dim,
                    adapter_where=adapter_where, lora_rank=prithvi_lora,
                    lora_alpha=lora_alpha, solo_profondo=True,
                    norm=norm_ingresso)
            else:
                self.aux_encoder = UNetEncoder(2, larghezza=fusione_width)
            canali = [UNetEncoder.USCITE[i] for i in self.fusione_livelli]
            if fusione in ("mgem", "entrambi"):
                self.mgem = nn.ModuleList([MGEM(c, c) for c in canali])
            if fusione in ("dgfm", "entrambi"):
                self.dgfm = nn.ModuleList([DGFM(c, c) for c in canali])
            self.fusione_params = [q for m in (self.aux_encoder, self.mgem,
                                               self.dgfm) if m is not None
                                   for q in m.parameters()]
            quale = ("Prithvi sulle sei bande HLS, solo livello /16"
                     if fusione_aux == "prithvi"
                     else f"U-Net sul terreno, larghezza {fusione_width}")
            print(f"  FUSIONE '{fusione}': stream ausiliario {quale} + "
                  f"{'MGEM ' if self.mgem is not None else ''}"
                  f"{'DGFM ' if self.dgfm is not None else ''}"
                  f"su {len(self.fusione_livelli)} livell"
                  f"{'o' if len(self.fusione_livelli) == 1 else 'i'} | "
                  f"{sum(q.numel() for q in self.fusione_params):,} parametri")

    def _apply_lora(self, module, nomi, r, alpha):
        """
        Sostituisce i Linear dell'attenzione con la loro versione LoRA. La
        sostituzione (invece di un hook) e' necessaria: LoRA ha bisogno
        dell'INGRESSO del Linear, mentre un forward_hook vede solo l'uscita.
        """
        for n in nomi:
            lin = getattr(module, n, None)
            if isinstance(lin, nn.Linear):
                w = LoRALinear(lin, r, alpha)
                setattr(module, n, w)
                self.lora.append(w)

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

        # SCONGELAMENTO DELL'ENCODER.
        # Finora l'unico punto in cui il modello poteva adattarsi al
        # multispettrale erano i ~70k parametri del patch embed, piu' gli
        # adapter. Tutto il resto di Hiera resta com'e' stato addestrato su
        # video RGB naturali.
        #
        # `unfreeze_blocks` = quanti blocchi FINALI di Hiera riaprire:
        #     0   nessuno (comportamento storico, default)
        #     n   gli ultimi n dei 12
        #    -1   tutti
        # Si parte dagli ULTIMI perche' i primi stadi codificano struttura
        # generica (bordi, tessiture) che vale anche fuori dominio, mentre la
        # semantica — cio' che qui e' sbagliato — sta in fondo. E i primi
        # stadi hanno le mappe piu' grandi, quindi costano di piu' e rendono
        # meno.
        #
        # NB: va SEMPRE accompagnato da un learning rate ridotto per questo
        # gruppo (vedi encoder_lr in LENTE_train), altrimenti al passo delle
        # teste nuove i pesi pretrained si distruggono in poche epoche.
        self.encoder_params = []
        n_unf = int(getattr(self, "_unfreeze_blocks", 0))
        if n_unf != 0:
            blocchi = [m for name, m in self.model.vision_encoder.named_modules()
                       if _e_blocco_hiera(name)]
            scelti = blocchi if n_unf < 0 else blocchi[-n_unf:]
            visti = set()
            for blk in scelti:
                for q in blk.parameters():
                    if id(q) in visti:
                        continue
                    visti.add(id(q))
                    q.requires_grad = True
                    self.encoder_params.append(q)
            print(f"  encoder SCONGELATO: {len(scelti)} blocchi Hiera, "
                  f"{sum(q.numel() for q in self.encoder_params):,} parametri")
        for ad in self.adapters:
            for p in ad.parameters(): p.requires_grad = True
        # LoRA: si addestrano solo A e B, il Linear originale resta congelato
        # (lo garantisce gia' LoRALinear.__init__, qui si riaccendono i rami).
        for w in self.lora:
            for p in w.A.parameters(): p.requires_grad = True
            for p in w.B.parameters(): p.requires_grad = True
        for p in self.early_guidance.parameters(): p.requires_grad = True
        # Il PROMPT ENCODER e' un modulo diverso dal mask_decoder e resta
        # congelato per default. Dentro c'e' `mask_embed` (MISURATO: 4.684
        # parametri, 1->4->16->256 canali), cioe' la pila di conv che traduce
        # `input_masks` nello spazio dell'image embedding — l'unico punto in cui
        # la mappa di YOLO entra nel modello.
        #
        # Perche' scongelarlo puo' avere senso: quei pesi sono stati addestrati
        # a leggere le maschere MORBIDE che SAM produceva nel raffinamento
        # iterativo, mentre noi gli diamo rettangoli rasterizzati. Finora
        # l'unico adattamento fra le due distribuzioni era un singolo scalare.
        # Perche' aspettarsi poco: 4.684 parametri con bottleneck a 4 canali
        # sono pochissima capacita'.
        # Rischio di sovradattamento: nullo, sono lo 0.07% degli addestrabili.
        if self.unfreeze_prompt_encoder:
            for p in self.model.prompt_encoder.parameters(): p.requires_grad = True

    def _embeddings(self, pixel_values):
        """
        Replica di Sam2Model.get_image_embeddings SENZA @torch.no_grad().

        Quel decoratore e' corretto per l'inferenza, ma qui i gradienti devono
        attraversare l'encoder: adapter, patch embed e blocchi scongelati sono
        tutti a monte. Usare il metodo originale li staccherebbe in silenzio.
        """
        b = pixel_values.shape[0]
        out = self.model.get_image_features(pixel_values, return_dict=True)
        feats = list(out.fpn_hidden_states)
        feats[-1] = feats[-1] + self.model.no_memory_embedding
        return [f.permute(1, 2, 0).view(b, -1, *size)
                for f, size in zip(feats, self.model.backbone_feature_sizes)]

    def _filtra_skip(self, emb):
        """Le due mappe superficiali passano filtrate da quella profonda."""
        emb = list(emb)
        if self.gate_s0 is not None:
            emb[0] = self.gate_s0(emb[0], emb[-1])
            emb[1] = self.gate_s1(emb[1], emb[-1])
        if getattr(self, "fpn", None) is not None:
            emb = self.fpn(emb)
        return emb

    def forward(self, x, boxes=None, dense=None, refine=None, presence=None):
        """
        `dense`: (B, 1, mask_input_size, mask_input_size) in [-1, 1], oppure None.
        `refine`: idem, ma e' la maschera che il modello stesso ha predetto al
            passaggio precedente, schiacciata con tanh nello stesso intervallo.

        RAFFINAMENTO ITERATIVO. `mask_embed` non e' un canale generico: SAM lo ha
        addestrato per rileggere le PROPRIE maschere e correggerle. Dandogli
        rettangoli rasterizzati lo usiamo fuori dalla distribuzione su cui e'
        nato; al secondo passaggio invece riceve proprio cio' che si aspetta.
        Le due mappe si SOMMANO, ciascuna con la sua scala appresa, cosi' il
        suggerimento del detector non viene buttato via al secondo giro e il
        modello puo' azzerare l'una o l'altra se non gli servono.

        E' il canale `input_masks`, indipendente da `input_boxes`. Serve a
        passare i box di YOLO come MAPPA invece che come rettangolo: cosi'
        possono essercene piu' d'uno, ciascuno pesato dalla propria confidenza,
        e sparisce la soglia secca — il modello impara da solo quanto fidarsi di
        una proposta debole. Un campione senza box riceve una mappa costante,
        che e' neutra (una costante additiva uniforme sull'image embedding).

        La scala e' appresa perche' `mask_embed` e' una convoluzione e
        l'ampiezza dei valori conta.
        """
        n = self.n_optical
        optical = x[:, :n]       # (B, n_optical, H, W)
        terrain = x[:, n:]       # (B, 2, H, W)
        g = self.early_guidance(terrain)
        optical = optical * g
        # `terrain` NON viene consumato qui: con la fusione attiva ha anche un
        # encoder tutto suo, e le due strade hanno ruoli diversi — la DSDF
        # modula l'ingresso ottico, lo stream ausiliario porta feature di
        # terreno fino al livello dell'FPN.

        prompt = resolve_prompt(self.prompt_source, boxes, optical.shape[0],
                                optical.shape[-1], optical.device, optical.dtype)
        m = None
        if dense is not None:
            m = dense.float() * torch.exp(self.dense_log_scale)
        if refine is not None:
            r = refine.float() * torch.exp(self.refine_log_scale)
            m = r if m is None else m + r
        if presence is not None:
            # Mappa UNIFORME: nessuna localizzazione, solo "quanto e' probabile
            # che qui ci sia qualcosa". E' la forma giusta per un classificatore,
            # che non ha informazione spaziale — e non ha nessun problema di
            # risoluzione, essendo costante.
            # Sommata alle altre invece che sostituirle: il modello puo' usare la
            # presenza per calibrare quanto essere aggressivo E i box per sapere
            # dove guardare. Ogni termine ha la sua scala appresa, quindi puo'
            # anche azzerarne una se non gli serve.
            h, w = self.mask_input_size
            p = (2.0 * presence.float().view(-1, 1, 1, 1) - 1.0).expand(-1, 1, h, w)
            p = p * torch.exp(self.clf_log_scale)
            m = p if m is None else m + p
        input_masks = None if m is None else m.to(optical.dtype)
        if self.unet_encoder is not None:
            # Encoder sostituito: Hiera non viene chiamato affatto. Le tre mappe
            # entrano dal parametro `image_embeddings`, che Sam2Model accetta al
            # posto di `pixel_values` — il mask decoder e il prompt encoder
            # restano quelli di SAM2, quindi l'unica variabile cambiata e' chi
            # produce le feature.
            emb = self.unet_encoder(optical)
            # `no_memory_embedding` viene normalmente sommato all'ultima mappa
            # da get_image_embeddings: lo replichiamo, cosi' il decoder riceve
            # la stessa distribuzione su cui e' stato addestrato.
            nm = getattr(self.model, "no_memory_embedding", None)
            if nm is not None:
                emb[-1] = emb[-1] + nm.view(1, -1, 1, 1)
            emb = self._filtra_skip(emb)
            outputs = self.model(image_embeddings=emb, input_boxes=prompt,
                                 input_masks=input_masks, multimask_output=False)
        elif self.fusione != "none":
            # Due encoder in parallelo sulle stesse tre scale. L'ottico passa
            # da Hiera, il terreno dallo stream ausiliario; MGEM li arricchisce
            # a vicenda tenendoli distinti, DGFM li fonde con l'ottico che
            # comanda. In uscita la larghezza e' quella del dominante, quindi
            # il mask decoder non si accorge di niente.
            emb = self._embeddings(optical)
            # Prithvi seleziona da se' le sue sei bande dall'ottico; la U-Net
            # ausiliaria vede invece il solo terreno.
            aux = self.aux_encoder(
                optical if self.fusione_aux == "prithvi" else terrain)
            if self.mgem is not None:
                for k, i in enumerate(self.fusione_livelli):
                    emb[i], aux[i] = self.mgem[k](emb[i], aux[i])
            if self.dgfm is not None:
                for k, i in enumerate(self.fusione_livelli):
                    emb[i] = self.dgfm[k](emb[i], aux[i])
            elif self.mgem is not None:
                # MGEM da solo non fonde: senza DGFM l'ausiliario resterebbe
                # inutilizzato e il modulo sarebbe una decorazione. Si somma,
                # che e' la fusione piu' neutra possibile.
                for i in self.fusione_livelli:
                    emb[i] = emb[i] + aux[i]
            emb = self._filtra_skip(emb)
            outputs = self.model(image_embeddings=emb, input_boxes=prompt,
                                 input_masks=input_masks, multimask_output=False)
        elif self.gate_s0 is not None or self.hsam is not None or self.fpn is not None:
            # Il percorso con image_embeddings esiste solo per poter filtrare
            # le skip: senza GSRM si resta su pixel_values, cosi' ogni run
            # storico ripercorre esattamente il codice di prima.
            emb = self._filtra_skip(self._embeddings(optical))
            outputs = self.model(image_embeddings=emb, input_boxes=prompt,
                                 input_masks=input_masks, multimask_output=False)
        else:
            outputs = self.model(pixel_values=optical, input_boxes=prompt,
                                 input_masks=input_masks, multimask_output=False)
        scores = getattr(outputs, "iou_scores",
                         getattr(outputs, "iou_predictions",
                                 getattr(outputs, "pred_ious", None)))
        if self.hsam is not None:
            # `emb` esiste in tutti i rami tranne pixel_values, che con H-SAM
            # attivo non viene mai preso (vedi il ramo di GSRM qui sopra).
            priore = get_best_mask(outputs.pred_masks, scores)
            self.ultimo_prior = priore
            return self.hsam(emb, priore).unsqueeze(1), None
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


# Configurazione del termine di regione. Va impostata UNA volta con
# configure_criterion(cfg); i default riproducono esattamente il comportamento
# storico (Dice di smp sul batch), cosi' i run vecchi restano confrontabili.
_LOSS = {"region": "dice", "per_image": False, "beta": 0.7, "lambda_dice": 0.3,
         "focal_gamma": 0.0,
         "small_gamma": 0.0, "small_ref": 500.0, "small_max": 10.0,
         "blob": 0.0}
_lovasz_fn = None

# Una frana e' una componente connessa a 8 vicini: con 4 vicini una frana
# unita solo per un angolo verrebbe contata come due.
_OTTO = np.ones((3, 3), dtype=int)


def _small_weight_map(y_true, gamma, ref, wmax):
    """
    Peso per pixel inversamente proporzionale all'AREA DELLA FRANA a cui il
    pixel appartiene:

        w = clamp( (ref / area_componente) ** gamma , 1, wmax )      sui pixel di frana
        w = 1                                                        sullo sfondo

    PERCHE' PER COMPONENTE e non per immagine. `loss_per_image` fa gia' pesare
    ogni PATCH uguale, ma dentro una patch con due frane — una da 5000 pixel e
    una da 50 — la grande domina lo stesso con un rapporto di 100:1. Qui il
    conteggio scende all'oggetto: sbagliare la piccola costa quanto sbagliare
    la grande, che e' cio' che serve visto che le frane piccole sono quelle che
    il modello perde (MISURATO sull'export: il 49% dei bersagli sta sotto i 64
    pixel e pesa il 4.6% dell'area totale).

    `ref` e' l'area di riferimento IN PIXEL alla risoluzione della loss (128x128
    con target_size 512, cioe' quella nativa). Sotto `ref` il peso sale, sopra
    scende ma non va mai sotto 1: le frane grandi non vengono penalizzate, le
    piccole vengono amplificate. `wmax` evita che una frana da 3 pixel prenda un
    peso da 400.

    Le componenti si ricalcolano a ogni batch perche' le maschere arrivano
    AUMENTATE. Costa ~1 ms per batch: `ndimage.label` su 128x128 e' immediato, e
    l'assegnazione dei pesi e' vettorizzata via indicizzazione dell'etichetta.
    """
    if ndimage is None:
        raise RuntimeError("loss_small_gamma richiede scipy (from scipy import ndimage)")
    m = (y_true.detach().squeeze(1) > 0.5).cpu().numpy().astype(np.uint8)
    w = np.ones(m.shape, dtype=np.float32)
    for i in range(m.shape[0]):
        if m[i].sum() == 0:
            continue
        lab, n = ndimage.label(m[i])
        if n == 0:
            continue
        aree = np.asarray(ndimage.sum(m[i], lab, index=range(1, n + 1)), dtype=np.float32)
        pesi = np.clip((ref / np.maximum(aree, 1.0)) ** gamma, 1.0, wmax)
        # indice 0 = sfondo -> peso 1
        w[i] = np.concatenate([[1.0], pesi])[lab]
    return torch.from_numpy(w).unsqueeze(1).to(y_true.device)


def configure_criterion(cfg):
    """
    Sceglie il termine di regione della loss. Da chiamare all'inizio di
    training e test — se i due divergono si valuta con una loss diversa da
    quella con cui si e' addestrato.
    """
    global _lovasz_fn
    _LOSS.update(region=cfg.get("loss_region", "dice"),
                 per_image=bool(cfg.get("loss_per_image", False)),
                 beta=float(cfg.get("loss_tversky_beta", 0.7)),
                 lambda_dice=float(cfg.get("lambda_dice", 0.3)),
                 focal_gamma=float(cfg.get("loss_focal_gamma", 0.0)),
                 small_gamma=float(cfg.get("loss_small_gamma", 0.0)),
                 small_ref=float(cfg.get("loss_small_ref", 500.0)),
                 small_max=float(cfg.get("loss_small_max", 10.0)),
                 blob=float(cfg.get("loss_blob", 0.0)))
    if _LOSS["region"] == "lovasz":
        _lovasz_fn = smp.losses.LovaszLoss(mode="binary", per_image=_LOSS["per_image"])
    if _LOSS["blob"] > 0:
        if _LOSS["region"] == "lovasz":
            raise SystemExit("la blob loss e' definita sui conteggi tp/fp/fn: "
                             "con loss_region lovasz non ha una forma per frana")
        if ndimage is None:
            raise SystemExit("la blob loss richiede scipy (from scipy import ndimage)")
    print(f"  loss: BCE(pos_weight) + {_LOSS['lambda_dice']} x {_LOSS['region']}"
          + (f" (beta={_LOSS['beta']})" if _LOSS["region"] == "tversky" else "")
          + ("  PER IMMAGINE" if _LOSS["per_image"] else "  sul batch"))
    if _LOSS["small_gamma"] > 0:
        g, r, x = _LOSS["small_gamma"], _LOSS["small_ref"], _LOSS["small_max"]
        print(f"  ENFASI FRANE PICCOLE: peso = ({r:.0f}/area)^{g} sui pixel di frana, "
              f"limitato a {x:.0f}x")
        for a in (25, 50, 100, 500, 2000):
            print(f"      frana da {a:5d} px  ->  peso {min((r/a)**g, x):5.2f}")
    if _LOSS["blob"] > 0:
        print(f"  BLOB LOSS: {_LOSS['region']} frana per frana (componenti a 8 "
              f"vicini), peso {_LOSS['lambda_dice']} x {_LOSS['blob']:g} = "
              f"{_LOSS['lambda_dice'] * _LOSS['blob']:.3f}  "
              f"(blob/globale = {_LOSS['blob']:g}; il paper usa alpha 2, beta 1)")


def _region_loss(y_pred, y_true):
    """
    Termine di regione, in tre varianti che condividono la stessa formula.

    PER IMMAGINE vs SUL BATCH — e' la leva sulle frane PICCOLE. Accumulando su
    tutto il batch (il default di smp, dims=(0,2)) il punteggio e' dominato
    dall'area: una frana da 5000 pixel conta 50 volte una da 100. Calcolandolo
    per immagine e poi mediando, OGNI PATCH pesa uguale a prescindere da quanto
    e' grande la frana che contiene. E' il singolo cambiamento piu' mirato sul
    problema degli oggetti piccoli.

    TVERSKY generalizza il Dice sbilanciando il costo dei due errori:
    beta > 0.5 punisce i FALSI NEGATIVI piu' dei falsi positivi. Gli oggetti
    piccoli si perdono quasi sempre per omissione, quindi beta alto li recupera.
    Dice e' esattamente Tversky con alpha = beta = 0.5.

    LOVASZ e' un surrogato convesso diretto della IoU — la metrica che poi
    misuriamo, mentre Dice e BCE ne sono approssimazioni.

    ATTENZIONE: tutte e tre spingono il modello ad accendere PIU' pixel, quindi
    peggiorano le patch vuote — che valgono il 40% del punteggio. Vanno usate
    insieme a un gate sulle patch vuote, non al posto suo.
    """
    if _LOSS["region"] == "lovasz":
        return _lovasz_fn(y_pred, y_true)

    p = torch.sigmoid(y_pred)
    if _LOSS["per_image"]:
        p = p.flatten(1); t = y_true.flatten(1)
        tp = (p * t).sum(1); fp = (p * (1 - t)).sum(1); fn = ((1 - p) * t).sum(1)
    else:
        p = p.reshape(-1); t = y_true.reshape(-1)
        tp = (p * t).sum(); fp = (p * (1 - t)).sum(); fn = ((1 - p) * t).sum()

    a, b = (1.0 - _LOSS["beta"], _LOSS["beta"]) if _LOSS["region"] == "tversky" else (0.5, 0.5)
    score = (tp + 1.0) / (tp + a * fp + b * fn + 1.0)
    return (1.0 - score).mean()


def _blob_loss(y_pred, y_true):
    """
    blob loss (Kofler et al., IPMI 2023): lo stesso termine di regione, ma
    calcolato FRANA PER FRANA e poi mediato. Ogni frana pesa uguale, qualunque
    sia la sua area — che e' come e' fatta la colonna `con frana`, e non come
    e' fatto il Dice globale, dominato dalle frane grandi.

    Per la frana n il dominio e' la patch MENO le altre frane (eq. 1 del
    paper). Lo sfondo resta: un falso positivo ovunque nella patch pesa su
    ogni frana. Le altre frane escono: trovarle o no non conta per la n.
    Sui conteggi diventa

        tp_n = somma di p sulla frana n
        fp_n = somma di p sullo sfondo          (uguale per ogni n)
        fn_n = area_n - tp_n

    con la stessa formula e lo stesso smoothing di _region_loss. Media sulle
    frane della patch, poi sulle patch che ne hanno almeno una. Le patch vuote
    non hanno frane: restano affidate al termine globale, come nel paper.

    DIVERSO da loss_small_gamma, che peggiorava tutto (0.5654 -> 0.5172):
    quello gonfiava il peso dei pixel delle frane piccole nella BCE, cioe'
    spingeva solo ad accenderli; qui ogni frana ha il suo Dice, in cui anche i
    falsi positivi della patch entrano al denominatore.
    """
    p = torch.sigmoid(y_pred.float())[:, 0]
    m = (y_true.detach()[:, 0] > 0.5).cpu().numpy()
    a, b = (1.0 - _LOSS["beta"], _LOSS["beta"]) if _LOSS["region"] == "tversky" else (0.5, 0.5)
    perdite = []
    for i in range(m.shape[0]):
        if not m[i].any():
            continue
        lab, n = ndimage.label(m[i], structure=_OTTO)
        lab = torch.from_numpy(lab).to(p.device).long().flatten()
        somme = torch.zeros(n + 1, device=p.device).index_add(0, lab, p[i].flatten())
        aree = torch.bincount(lab, minlength=n + 1).float()
        tp, fp = somme[1:], somme[0]
        fn = aree[1:] - tp
        score = (tp + 1.0) / (tp + a * fp + b * fn + 1.0)
        perdite.append((1.0 - score).mean())
    if not perdite:
        return y_pred.sum() * 0.0
    return torch.stack(perdite).mean()


def criterion(y_pred, y_true, pos_weight=1.0, lambda_dice=None):
    y_pred = y_pred.contiguous()
    y_true_rs = F.interpolate(y_true.unsqueeze(1).float(), size=y_pred.shape[2:], mode="nearest")
    pw = torch.tensor([pos_weight], device=y_pred.device)

    # Enfasi sulle frane piccole: entra come peso PER PIXEL nella BCE, non nel
    # termine di regione. Il Dice e' gia' normalizzato per area, quindi pesarlo
    # per area sarebbe ridondante; la BCE invece conta un termine per pixel ed
    # e' li' che le frane grandi dominano il gradiente.
    w = None
    if _LOSS["small_gamma"] > 0:
        w = _small_weight_map(y_true_rs, _LOSS["small_gamma"],
                              _LOSS["small_ref"], _LOSS["small_max"]).to(y_pred.dtype)
    # FOCAL come generalizzazione della BCE, non come termine aggiuntivo:
    # sommarle conterebbe due volte lo stesso segnale. Con focal_gamma = 0 il
    # fattore (1-pt)^gamma vale 1 ovunque e questo ramo e' BIT-IDENTICO alla
    # BCE precedente, quindi i run storici restano riproducibili.
    #
    # Perche' provarla: il modello e' fermo sulle frane piccole (con frana
    # 0.4928) mentre va bene su quelle grandi. La focal toglie peso agli esempi
    # gia' risolti e lo sposta su quelli difficili, che sono esattamente quelli.
    # NB: un tentativo nella stessa direzione — loss_small_gamma, che pesava
    # per AREA della componente invece che per difficolta' — ha peggiorato tutto
    # (micro 0.5654 -> 0.5172). Meccanismo diverso, ma il precedente c'e'.
    ce_pix = F.binary_cross_entropy_with_logits(
        y_pred, y_true_rs, weight=w, pos_weight=pw, reduction="none")
    g = _LOSS["focal_gamma"]
    if g > 0:
        with torch.no_grad():
            prob = torch.sigmoid(y_pred)
            pt = prob * y_true_rs + (1.0 - prob) * (1.0 - y_true_rs)
            mod = (1.0 - pt).pow(g)
        ce_pix = ce_pix * mod
    ce = ce_pix.mean()

    lam = _LOSS["lambda_dice"] if lambda_dice is None else lambda_dice
    if _LOSS["region"] == "dice" and not _LOSS["per_image"]:
        reg = dice_loss_fn(y_pred, y_true_rs)      # percorso storico, bit-identico
    else:
        reg = _region_loss(y_pred, y_true_rs)
    if _LOSS["blob"] > 0:
        # Pesata RELATIVAMENTE al termine di regione: nel paper globale e blob
        # sono entrambi Dice, con alpha 2 e beta 1. Qui il Dice globale pesa
        # `lam`, quindi la blob pesa lam x beta/alpha. Metterla accanto alla
        # loss intera la farebbe pesare piu' del Dice globale, nella zona
        # dove l'ablazione del paper peggiora.
        return ce + lam * (reg + _LOSS["blob"] * _blob_loss(y_pred, y_true_rs))
    return ce + lam * reg


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


def as_refine_map(preds, out_size):
    """
    La maschera predetta -> mappa da rimandare dentro come `refine`.

    Due scelte, entrambe deliberate:

    * DETACH. Il gradiente non attraversa il passaggio precedente. E' quello che
      fa SAM nel suo addestramento iterativo: senza, la rete diventa ricorrente
      a due passi, la memoria raddoppia e l'ottimizzazione si destabilizza,
      perche' il secondo passaggio puo' migliorare la loss peggiorando il primo.
      Con il detach ogni passaggio e' un problema di predizione a se'.

    * TANH invece dei logit grezzi. I logit di una maschera possono valere +-10,
      la mappa di YOLO sta in [-1, 1]: sommandole, la prima seppellirebbe la
      seconda e la scala appresa partirebbe da un punto assurdo. tanh e'
      monotona e li porta nello stesso intervallo senza perdere l'ordinamento,
      cosi' exp(0)=1 e' un punto di partenza sensato per entrambe.
    """
    r = torch.tanh(preds.detach().float())
    if r.shape[-2:] != (out_size, out_size):
        r = F.interpolate(r, (out_size, out_size), mode="bilinear", align_corners=False)
    return r


def clf_boxes(p, thr, orig_h, orig_w):
    """
    Il prompt del regime "clf_box", nella stessa forma che aveva il `gt_box`.

    Per ogni campione, in coordinate NATIVE (verranno poi scalate da
    scale_boxes come qualunque altro box):

        p <  thr   ->  SENTINEL_BOX [0,0,10,10]   "qui non c'e' niente"
        p >= thr   ->  box a tutta immagine       "qui c'e' qualcosa"

    PERCHE' QUESTA FORMA. Nel regime `gt_box` la sentinella era la costante che
    `_bbox_from_mask` restituiva sulle maschere vuote, e il modello imparava
    "questo pattern -> non disegnare niente": MISURATO, vuote = 1.0000 esatto su
    306 patch, ed era il 76% del vantaggio di quel regime. Ma il segnale nasceva
    dalla ground truth, quindi era un leak.

    Qui la stessa distinzione viene da un classificatore addestrato: stesso
    meccanismo, stesso canale (`input_boxes`), informazione disponibile anche a
    inferenza. Il box a tutta immagine non localizza nulla — non e' il suo
    compito — ma tiene il decoder nel regime "promptato" del pretraining e
    rende il contrasto con la sentinella netto e sempre uguale.
    """
    x0, y0, x1, y1 = SENTINEL_BOX
    full = torch.tensor([0.0, 0.0, float(orig_w - 1), float(orig_h - 1)],
                        device=p.device)
    sent = torch.tensor([x0, y0, x1, y1], device=p.device)
    pos = (p >= thr).view(-1, 1)
    return torch.where(pos, full.expand(p.numel(), 4), sent.expand(p.numel(), 4)).clone()


def iou_per_sample(preds, masks, epsilon=1e-7):
    """
    IoU di ciascun campione, invece della media del batch.

    Serve quando i campioni NON sono intercambiabili: su una patch vuota il
    punteggio e' 1.0 o 0.0 secco, su una con frana e' graduale, e mediarli
    insieme nasconde quale dei due si sta muovendo. Stessa formula ed epsilon
    di calculate_metrics, cosi' i numeri restano confrontabili.
    """
    if preds.shape[-2:] != masks.shape[-2:]:
        preds = F.interpolate(preds, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    pr = (torch.sigmoid(preds) > 0.5).float()
    gt = masks.unsqueeze(1).float()
    tp = (pr * gt).sum((2, 3)); fp = (pr * (1 - gt)).sum((2, 3)); fn = ((1 - pr) * gt).sum((2, 3))
    return ((tp + epsilon) / (tp + fp + fn + epsilon)).squeeze(1)


def jitter_boxes(boxes, size, rel_std=0.1, max_px=20.0):
    """
    Rumore gaussiano sui box GT usati come prompt in TRAINING.

    Serve in due situazioni, per lo stesso motivo di fondo — il modello non deve
    abituarsi a prompt di qualita' oracolo che a inferenza non avra':

      * scheduled sampling del boxhead: senza, il decoder vede box perfetti
        finche' p_gt > 0 e poi di colpo box imprecisi, una distribuzione mai
        incontrata proprio quando deve reggersi da solo;
      * cascata YOLO -> SAM2: LENTE va addestrato con box degradati QUANTO
        QUELLI CHE YOLO produrra' davvero. `rel_std` va calibrato sull'errore
        misurato del detector (vedi LENTE_yolo_cascade.py, comando `calibrate`),
        non lasciato al default.

    SAM stesso e' stato addestrato con box rumorosi, non con rettangoli minimi
    perfetti. La deviazione standard e' proporzionale al lato del box (rel_std)
    e limitata a max_px, cosi' su una frana piccola il rumore non la sposta
    fuori dal riquadro.

    NON va applicato in validazione o test: li' il prompt deve essere quello
    che il sistema produrra' davvero.
    """
    if rel_std <= 0:
        return boxes
    w = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0)
    h = (boxes[:, 3] - boxes[:, 1]).clamp(min=1.0)
    std = torch.stack([w, h, w, h], dim=1) * rel_std
    noisy = boxes + torch.randn_like(boxes) * std.clamp(max=max_px)
    # riordina gli angoli e tiene il box dentro l'immagine
    x0 = noisy[:, 0].minimum(noisy[:, 2]).clamp(0, size - 1)
    x1 = noisy[:, 0].maximum(noisy[:, 2]).clamp(0, size - 1)
    y0 = noisy[:, 1].minimum(noisy[:, 3]).clamp(0, size - 1)
    y1 = noisy[:, 1].maximum(noisy[:, 3]).clamp(0, size - 1)
    return torch.stack([x0, y0, x1, y1], dim=1)


def scale_boxes(boxes, orig_h, orig_w, target, margin=3):
    sx = target / orig_w; sy = target / orig_h
    boxes[:, 0] = torch.clamp(boxes[:, 0] * sx - margin, min=0)
    boxes[:, 1] = torch.clamp(boxes[:, 1] * sy - margin, min=0)
    boxes[:, 2] = torch.clamp(boxes[:, 2] * sx + margin, max=target - 1)
    boxes[:, 3] = torch.clamp(boxes[:, 3] * sy + margin, max=target - 1)
    return boxes


def _norm_da_cfg(cfg):
    """Le statistiche per canale, solo se chieste E presenti nel config."""
    if not cfg.get("norm_ingresso", False):
        return None
    m, d = cfg.get("band_mean"), cfg.get("band_std")
    if m is None or d is None:
        raise SystemExit(
            "--norm-ingresso richiede band_mean e band_std nel blocco del "
            "dataset in config.yaml: calcolale con statistiche_ingresso.py")
    print("  affine d'ingresso inizializzata con le statistiche del train "
          "(ingressi standardizzati)")
    return (m, d)


def build_model(cfg, device):
    """
    Costruisce LENTE col numero di canali ottici giusto dal config.

    `image_size` viene da `target_size`, la stessa chiave a cui i loop
    interpolano le immagini: se i due divergessero, il prompt encoder
    normalizzerebbe i box su una dimensione diversa da quella reale.
    """
    m = LENTE(sam_ckpt_path=cfg["sam_ckpt_path"],
              n_optical=cfg["n_optical"],
              adapter_dim=cfg["adapter_dim"],
              prompt_source=cfg.get("prompt_source", "none"),
              image_size=cfg["target_size"],
              unfreeze_prompt_encoder=cfg.get("unfreeze_prompt_encoder", False),
              lora_rank=int(cfg.get("lora_rank", 0)),
              lora_alpha=float(cfg.get("lora_alpha", 16.0)),
              adapter_where=cfg.get("adapter_where", "attn"),
              unfreeze_blocks=int(cfg.get("unfreeze_blocks", 0)),
              encoder=cfg.get("encoder", "hiera"),
              unet_width=int(cfg.get("unet_width", 48)),
              gsrm=bool(cfg.get("gsrm", False)),
              fusione=cfg.get("fusione", "none"),
              fusione_width=int(cfg.get("fusione_width", 24)),
              fusione_aux=cfg.get("fusione_aux", "terreno"),
              optical_bands=cfg.get("optical_bands"),
              prithvi_model=cfg.get("prithvi_model",
                                    "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"),
              prithvi_size=int(cfg.get("prithvi_size", 512)),
              prithvi_frozen=bool(cfg.get("prithvi_frozen", True)),
              prithvi_adapter_dim=int(cfg.get("prithvi_adapter_dim", 0)),
              prithvi_lora=int(cfg.get("prithvi_lora", 0)),
              tec_variante=cfg.get("tec_variante", "fp"),
              tec_compression=int(cfg.get("tec_compression", 10)),
              tec_size=int(cfg.get("tec_size", 0)),
              tec_frozen=bool(cfg.get("tec_frozen", True)),
              norm_ingresso=_norm_da_cfg(cfg),
              hsam=bool(cfg.get("hsam", False)),
              fpn_completo=bool(cfg.get("fpn_completo", False))).to(device)
    n_ad = sum(p.numel() for a in m.adapters for p in a.parameters())
    n_lo = sum(p.numel() for w in m.lora for p in list(w.A.parameters()) + list(w.B.parameters()))
    n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  adapter {n_ad:,} (attn x{m.n_adapter_attn}, mlp x{m.n_adapter_mlp}) | "
          f"LoRA {n_lo:,} ({len(m.lora)} layer) | addestrabili {n_tr:,}")
    return m


def save_checkpoint(model, optimizer, scheduler, epoch, best_iou, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_iou": best_iou,
    }, path)


def run_record_path(checkpoint_name, prefix):
    """
    Nome del file JSON derivato dal CHECKPOINT, non da `history_file` del config.

    Motivo pratico: ogni run viene lanciato con `--checkpoint <nome>`, quindi
    derivando da li' il record segue sempre il modello a cui appartiene e due
    esperimenti non possono sovrascriversi a vicenda. Con la chiave del config
    tutti i run dello stesso dataset finirebbero nello stesso file.

        best_lente_l4s_dense_b.pth  ->  history_best_lente_l4s_dense_b.json
    """
    base = os.path.splitext(os.path.basename(checkpoint_name))[0]
    return f"{prefix}_{base}.json"


def _jsonable(cfg):
    """Sottoinsieme del config che identifica il regime di un run."""
    keys = ("dataset_name", "target_size", "adapter_dim", "prompt_source",
            "yolo_prompt", "yolo_weights", "yolo_conf", "yolo_dense_conf",
            "yolo_dense_max_det", "yolo_bands", "yolo_imgsz", "refine_passes",
            "refine_lambda", "loss_region", "loss_per_image", "loss_tversky_beta",
            "lambda_dice", "unfreeze_prompt_encoder", "learning_rate",
            "batch_size", "pos_weight_mode", "random_seed", "aug_copies",
            # Aggiunte dopo: mancavano tutte, e la prima e' quella che conta.
            # `select_on` assente faceva ricadere stato_run.py sul default
            # "totale", quindi riportava l'epoca sbagliata come contenuto del
            # .pth su OGNI run addestrato con --select-on micro. Il checkpoint
            # era giusto: era il rapporto a mentire.
            "select_on", "patience", "epochs", "accum_steps", "weight_decay",
            "adapter_where", "unfreeze_blocks", "encoder_lr_mult",
            "encoder", "unet_width", "unet_lr", "gsrm", "gsrm_lr",
            "optical_bands", "terrain_bands", "n_optical", "loss_focal_gamma",
            "fusione", "fusione_width", "fusione_lr", "fusione_aux",
            "prithvi_model", "prithvi_size", "prithvi_frozen",
            "gt_senza_sentinella",
            "prithvi_adapter_dim", "prithvi_lora",
            "tec_variante", "tec_compression", "tec_size", "tec_frozen",
            "norm_ingresso", "loss_blob",
            # mancavano: senza, un run sulle sole positive o con box degradati
            # era indistinguibile dagli altri guardando il suo history
            "train_positives_only", "prompt_jitter", "prompt_dropout")
    return {k: cfg[k] for k in keys if k in cfg}


def write_json(path, payload):
    """
    Scrittura atomica: prima su un file temporaneo, poi rename. Cosi' un
    interrupt a meta' epoca non lascia un JSON troncato e illeggibile — che e'
    esattamente quando serve di piu'.
    """
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


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
