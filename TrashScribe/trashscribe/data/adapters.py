"""Dataset adapters: native format -> RawSample with labels mapped to atlas entries.

coco   COCO-style instance annotations (TACO, ZeroWaste-f, BePLi v2, UAVVaste, ...)
voc    PASCAL-VOC style folders, optional instance PNGs and depth (MJU-Waste)
masks  image folder + binary mask folder (StreetView-Waste overflow, PlastOPol, ...)

Every adapter takes `class_map` (native name -> atlas entry or null) and reports
unmapped classes loudly instead of guessing.
"""
from __future__ import annotations

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
        return dropped

    def entries_present(self) -> set[str]:
        return set().union(*self.entries_per_item) if self.entries_per_item else set()


class CocoAdapter(BaseAdapter):
    def __init__(self, name, images_dir, ann_file, class_map=None, surroundings=None, scene_map=None,
                 image_ids=None, has_instances=True, depth_dir=None, max_items=None):
        from pycocotools.coco import COCO
        self.name = name
        self.images_dir = Path(images_dir)
        self.depth_dir = Path(depth_dir) if depth_dir else None
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
        for iid in ids:
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=iid, iscrowd=None))
            self.entries_per_item.append({self.cat2entry[a["category_id"]] for a in anns} - {None})

    def load(self, i) -> RawSample:
        iid = self.items[i]
        info = self.coco.loadImgs(iid)[0]
        img = _read_rgb(self.images_dir / info["file_name"])
        h, w = img.shape[:2]
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=iid, iscrowd=None))
        insts, ents = [], []
        waste = np.zeros((h, w), np.uint8)
        for a in anns:
            if info["height"] != h or info["width"] != w:
                # EXIF-rotated files: annotation frame differs from decoded frame
                m = cv2.resize(self.coco.annToMask(a), (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                m = self.coco.annToMask(a)
            if a.get("iscrowd", 0):
                waste[m > 0] = 1
                continue
            insts.append(m.astype(np.uint8))
            ents.append(self.cat2entry.get(a["category_id"]))
            waste[m > 0] = 1
        depth = None
        if self.depth_dir is not None:
            p = self.depth_dir / (Path(info["file_name"]).with_suffix(".npy"))
            if p.exists():
                depth = np.load(p).astype(np.float32)
        return RawSample(image=img, instances=insts, entries=ents, dataset=self.name, image_id=str(iid),
                         waste=waste, surroundings=self.scene.get(iid, self.surroundings),
                         has_instances=self.has_instances, depth=depth)


class VocAdapter(BaseAdapter):
    """VOC layout: JPEGImages/, SegmentationClass/, optional SegmentationObject/,
    optional depth folder, ImageSets/Segmentation/<split>.txt."""

    def __init__(self, name, root, split, waste_values=(1,), ignore_value=255, entry=None, surroundings=None,
                 instance_dir="SegmentationObject", depth_dir=None, image_ext=".jpg", max_items=None):
        self.name, self.root = name, Path(root)
        self.waste_values, self.ignore_value = set(waste_values), ignore_value
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
        cls = np.array(_read_palette(self.root / "SegmentationClass" / f"{sid}.png"))
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
