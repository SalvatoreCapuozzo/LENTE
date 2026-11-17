"""Dataset adapters: native format -> RawSample with labels mapped to atlas entries.

coco   COCO-style instance annotations (TACO, ZeroWaste-f, BePLi v2, UAVVaste, ...)
voc    PASCAL-VOC style folders, optional instance PNGs and depth (MJU-Waste)
masks  image folder + binary mask folder (StreetView-Waste overflow, PlastOPol, ...)

Every adapter takes `class_map` (native name -> atlas entry or null) and reports
unmapped classes loudly instead of guessing.
"""
from __future__ import annotations

import pickle
import random
from pathlib import Path

import cv2
import numpy as np

from .canonical import IGNORE, RawSample


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _components(waste: np.ndarray, min_area=16) -> list[np.ndarray]:
    n, lab = cv2.connectedComponents((waste == 1).astype(np.uint8))
    return [(lab == i).astype(np.uint8) for i in range(1, n) if (lab == i).sum() >= min_area]


def _ann_to_mask(seg, h: int, w: int) -> np.ndarray:
    """pycocotools COCO.annToMask without the COCO object (same branches)."""
    from pycocotools import mask as mask_utils
    if isinstance(seg, list):
        rle = mask_utils.merge(mask_utils.frPyObjects(seg, h, w))
    elif isinstance(seg["counts"], list):
        rle = mask_utils.frPyObjects(seg, h, w)
    else:
        rle = seg
    return mask_utils.decode(rle)


def split_ids(ids: list, fractions: dict, seed: int) -> dict:
    ids = sorted(ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    a = round(n * fractions["train"])
    b = a + round(n * fractions["val"])
    return {"train": ids[:a], "val": ids[a:b], "test": ids[b:]}


class BaseAdapter:
    name: str
    entries_per_item: list[set]

    def __len__(self):
        return len(self.items)

    def drop_items_with(self, entries: set[str]) -> int:
        """Remove every image that contains a held-out entry. Removing only the
        boxes would teach the model to call those objects background."""
        if not entries:
            return 0
        keep = [i for i, es in enumerate(self.entries_per_item) if not (es & entries)]
        dropped = len(self.items) - len(keep)
        self.items = [self.items[i] for i in keep]
        self.entries_per_item = [self.entries_per_item[i] for i in keep]
        self._select(keep)
        return dropped

    def _select(self, keep: list[int]):
        """Keep only these item indices in any per-item store (subclasses)."""

    def entries_present(self) -> set[str]:
        return set().union(*self.entries_per_item) if self.entries_per_item else set()


class CocoAdapter(BaseAdapter):
    def __init__(self, name, images_dir, ann_file, class_map=None, surroundings=None, scene_map=None,
                 image_ids=None, has_instances=True, depth_dir=None, max_items=None, negatives_dir=None):
        from pycocotools.coco import COCO
        self.name = name
        self.images_dir = Path(images_dir)
        self.depth_dir = Path(depth_dir) if depth_dir else None
        # negatives_dir: <stem>.png, > 0 = pixel known NOT to be waste. When given, only
        # annotated objects and these pixels are supervised / scored; every other
        # pixel is ignored (ZeroWaste-f: paper is waste but unannotated, the belt is not)
        self.negatives_dir = Path(negatives_dir) if negatives_dir else None
        self.coco = COCO(str(ann_file))
        self.surroundings, self.has_instances = surroundings, has_instances
        cats = {c["id"]: c["name"] for c in self.coco.loadCats(self.coco.getCatIds())}
        class_map = class_map or {}
        # default: the native name IS the atlas entry name (checked by the registry);
        # the key "*" maps every unlisted class, e.g. {"*": "waste"} for binary datasets
        self.cat2entry = {cid: class_map.get(n, class_map.get("*", n)) for cid, n in cats.items()}
        self.scene = {}
        ds = self.coco.dataset
        if scene_map and "scene_annotations" in ds:
            sc = {c["id"]: c["name"] for c in ds.get("scene_categories", [])}
            for a in ds["scene_annotations"]:
                vals = [scene_map.get(sc[b]) for b in a.get("background_ids", []) if sc.get(b) in scene_map]
                vals = [v for v in vals if v]
                if vals:
                    self.scene[a["image_id"]] = vals[0]
        ids = list(self.coco.getImgIds()) if image_ids is None else list(image_ids)
        if max_items:
            ids = ids[:max_items]
        self.items = ids
        self.entries_per_item = []
        # imgToAnns, not loadAnns(getAnnIds(...)): TACO repeats annotation ids, and the
        # id lookup then returns another image's object (wrong mask and wrong class)
        for iid in ids:
            anns = self.coco.imgToAnns[iid]
            self.entries_per_item.append({self.cat2entry[a["category_id"]] for a in anns} - {None})
        # Compact per-image store, then the COCO object is dropped. DataLoader workers
        # are forks: reading Python dicts bumps refcounts, which copies the touched pages
        # into each worker, so every worker slowly grew a private copy of all
        # annotations (70k+ on BePLi). Bytes inside one numpy buffer are never copied.
        blobs, meta, off = [], [], 0
        for iid in ids:
            info = self.coco.imgs[iid]
            anns = [(a["category_id"], int(a.get("iscrowd", 0)), a["segmentation"]) for a in self.coco.imgToAnns[iid]]
            b = pickle.dumps(anns, protocol=pickle.HIGHEST_PROTOCOL)
            blobs.append(b)
            meta.append((off, len(b), info["height"], info["width"]))
            off += len(b)
        self._buf = np.frombuffer(b"".join(blobs), np.uint8)
        self._meta = np.array(meta, np.int64).reshape(-1, 4)
        self._files = np.array([self.coco.imgs[iid]["file_name"] for iid in ids], dtype=np.str_)
        del self.coco

    def _select(self, keep):
        self._meta, self._files = self._meta[keep], self._files[keep]

    def load(self, i) -> RawSample:
        iid = self.items[i]
        off, n, ah, aw = (int(v) for v in self._meta[i])
        anns = pickle.loads(self._buf[off:off + n].tobytes())
        fname = str(self._files[i])
        img = _read_rgb(self.images_dir / fname)
        h, w = img.shape[:2]
        insts, ents = [], []
        waste = np.zeros((h, w), np.uint8)
        for cat, crowd, seg in anns:
            m = _ann_to_mask(seg, ah, aw)
            if ah != h or aw != w:
                # EXIF-rotated files: annotation frame differs from decoded frame
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            if crowd:
                waste[m > 0] = 1
                continue
            insts.append(m.astype(np.uint8))
            ents.append(self.cat2entry.get(cat))
            waste[m > 0] = 1
        valid = None
        if self.negatives_dir is not None:
            neg = cv2.imread(str(self.negatives_dir / (Path(fname).stem + ".png")), cv2.IMREAD_GRAYSCALE)
            if neg is None:
                raise FileNotFoundError(self.negatives_dir / (Path(fname).stem + ".png"))
            if neg.shape != (h, w):
                neg = cv2.resize(neg, (w, h), interpolation=cv2.INTER_NEAREST)
            valid = ((neg > 0) | (waste > 0)).astype(np.uint8)
        depth = None
        if self.depth_dir is not None:
            p = self.depth_dir / (Path(fname).with_suffix(".npy"))
            if p.exists():
                depth = np.load(p).astype(np.float32)
        return RawSample(image=img, instances=insts, entries=ents, dataset=self.name, image_id=str(iid),
                         waste=waste, valid=valid, surroundings=self.scene.get(iid, self.surroundings),
                         has_instances=self.has_instances, depth=depth)


class VocAdapter(BaseAdapter):
    """VOC layout: JPEGImages/, SegmentationClass/, optional SegmentationObject/,
    optional depth folder, ImageSets/Segmentation/<split>.txt."""

    def __init__(self, name, root, split, waste_values=(1,), ignore_value=255, entry=None, surroundings=None,
                 instance_dir="SegmentationObject", depth_dir=None, image_ext=".jpg", max_items=None,
                 waste_colors=None):
        self.name, self.root = name, Path(root)
        self.waste_values, self.ignore_value = set(waste_values), ignore_value
        # waste_colors: decode the class PNG by palette COLOUR (VOC semantics) instead of
        # raw index. MJU-Waste stores waste as index 0 drawn (128,0,0), background as
        # index 1 drawn black, and empty images with an all-black palette.
        self.waste_colors = np.array(waste_colors, np.uint8) if waste_colors else None
        self.entry, self.surroundings = entry, surroundings
        self.inst_dir = self.root / instance_dir if instance_dir and (self.root / instance_dir).exists() else None
        self.depth_dir = self.root / depth_dir if depth_dir else None
        self.image_ext = image_ext
        ids = (self.root / "ImageSets" / "Segmentation" / f"{split}.txt").read_text().split()
        self.items = ids[:max_items] if max_items else ids
        self.entries_per_item = [{entry} - {None} for _ in self.items]

    def load(self, i) -> RawSample:
        sid = self.items[i]
        img = _read_rgb(self.root / "JPEGImages" / f"{sid}{self.image_ext}")
        pim = _read_palette(self.root / "SegmentationClass" / f"{sid}.png")
        cls = np.array(pim)
        if self.waste_colors is not None:
            rgb = np.array(pim.convert("RGB"))
            waste = (rgb[:, :, None, :] == self.waste_colors[None, None]).all(-1).any(-1).astype(np.uint8)
            if self.ignore_value is not None:
                waste[(rgb == (224, 224, 192)).all(-1)] = IGNORE       # VOC "void" colour
        else:
            waste = np.where(np.isin(cls, list(self.waste_values)), 1, 0).astype(np.uint8)
            waste[cls == self.ignore_value] = IGNORE
        if self.inst_dir is not None:
            obj = np.array(_read_palette(self.inst_dir / f"{sid}.png"))
            ids = [v for v in np.unique(obj) if v not in (0, self.ignore_value)]
            insts = [((obj == v) & (waste == 1)).astype(np.uint8) for v in ids]
            insts = [m for m in insts if m.any()]
            has_inst = True
        else:
            insts, has_inst = _components(waste), False
        depth = None
        if self.depth_dir is not None:
            for ext in (".png", ".npy", ".tiff"):
                p = self.depth_dir / f"{sid}{ext}"
                if p.exists():
                    depth = np.load(p) if ext == ".npy" else cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                    depth = depth.astype(np.float32)
                    break
        return RawSample(image=img, instances=insts, entries=[self.entry] * len(insts), dataset=self.name,
                         image_id=sid, waste=waste, surroundings=self.surroundings, has_instances=has_inst,
                         depth=depth)


def _read_palette(path: Path):
    from PIL import Image
    return Image.open(path)


class MaskFolderAdapter(BaseAdapter):
    """images/<id>.<ext> + masks/<id>.png, mask > threshold = waste. Instances are
    connected components (flagged has_instances=False: no contour supervision)."""

    def __init__(self, name, images_dir, masks_dir, ids=None, mask_suffix=".png", threshold=0, entry=None,
                 surroundings=None, ignore_value=None, max_items=None):
        self.name = name
        self.images_dir, self.masks_dir = Path(images_dir), Path(masks_dir)
        self.mask_suffix, self.threshold, self.ignore_value = mask_suffix, threshold, ignore_value
        self.entry, self.surroundings = entry, surroundings
        files = sorted(p for p in self.images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if ids is not None:
            keep = set(ids)
            files = [p for p in files if p.stem in keep]
        self.items = files[:max_items] if max_items else files
        self.entries_per_item = [{entry} - {None} for _ in self.items]

    def load(self, i) -> RawSample:
        p = self.items[i]
        img = _read_rgb(p)
        m = cv2.imread(str(self.masks_dir / f"{p.stem}{self.mask_suffix}"), cv2.IMREAD_GRAYSCALE)
        if m.shape != img.shape[:2]:
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        waste = (m > self.threshold).astype(np.uint8)
        if self.ignore_value is not None:
            waste[m == self.ignore_value] = IGNORE
        insts = _components(waste)
        return RawSample(image=img, instances=insts, entries=[self.entry] * len(insts), dataset=self.name,
                         image_id=p.stem, waste=waste, surroundings=self.surroundings, has_instances=False)
