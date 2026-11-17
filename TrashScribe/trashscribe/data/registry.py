"""Build dataset adapters from configuration.

A dataset block looks like (see configs/datasets.example.yaml):

    - name: taco
      type: coco
      images_dir: /data/TACO/data
      ann_file: /data/TACO/data/annotations.json
      split: {seed: 42, fractions: {train: 0.7, val: 0.1, test: 0.2}}
      atlas: atlas/taco.yaml
      scene_map: {"Pavement": "pavement or road", ...}

or, with official splits, `splits: {train: {...}, val: {...}, test: {...}}`
whose inner keys override the outer ones (images_dir, ann_file, split for VOC).
"""
from __future__ import annotations

import copy
from pathlib import Path

from ..schema import Atlas
from .adapters import CocoAdapter, MaskFolderAdapter, VocAdapter, split_ids
from .synthetic import SyntheticAdapter


def _resolve(root: Path, p):
    if p is None:
        return None
    p = Path(p).expanduser()
    return p if p.is_absolute() else root / p


def build_adapter(cfg: dict, split: str, root: Path, max_items=None):
    a = _build_adapter(cfg, split, root, max_items)
    if a is not None:
        # eval_only_entries: classes of the novel atlas that a dataset contains. Their
        # objects train the waste mask only; they are never described, never count as
        # seen, and at test time they are named / retrieved from their text entry.
        a.eval_only = frozenset(cfg.get("eval_only_entries") or [])
    return a


def subsample_adapter(a, n: int, seed: int = 0) -> list[int]:
    """Keep a seeded random subset of n items (E10: n labelled images of a new site). Returns the
    kept indices of the full split, for the record."""
    import random
    idx = list(range(len(a.items)))
    random.Random(seed).shuffle(idx)
    keep = sorted(idx[:n])
    a.items = [a.items[i] for i in keep]
    a.entries_per_item = [a.entries_per_item[i] for i in keep]
    a._select(keep)
    return keep


def trained_entries(adapters) -> set[str]:
    """Atlas entries a model is supervised on: present in the data, not eval-only."""
    out: set[str] = set()
    for a in adapters:
        out |= a.entries_present() - getattr(a, "eval_only", frozenset())
    return out


def _build_adapter(cfg: dict, split: str, root: Path, max_items=None):
    c = copy.deepcopy(cfg)
    if "splits" in c:
        if split not in c["splits"]:
            return None
        c.update(c.pop("splits")[split])
    t = c["type"]
    if t == "coco":
        ids = None
        if "split" in c:
            from pycocotools.coco import COCO  # noqa: F401  (ensures dependency early)
            full = CocoAdapter(c["name"], _resolve(root, c["images_dir"]), _resolve(root, c["ann_file"]))
            ids = split_ids(full.items, c["split"]["fractions"], c["split"]["seed"])[split]
        return CocoAdapter(c["name"], _resolve(root, c["images_dir"]), _resolve(root, c["ann_file"]),
                           class_map=c.get("class_map"), surroundings=c.get("surroundings"),
                           scene_map=c.get("scene_map"), image_ids=ids,
                           has_instances=c.get("has_instances", True),
                           depth_dir=_resolve(root, c.get("depth_dir")), max_items=max_items,
                           negatives_dir=_resolve(root, c.get("negatives_dir")))
    if t == "voc":
        return VocAdapter(c["name"], _resolve(root, c["root"]), c.get("split_file", split),
                          waste_values=c.get("waste_values", (1,)), entry=c.get("entry"),
                          surroundings=c.get("surroundings"), instance_dir=c.get("instance_dir", "SegmentationObject"),
                          depth_dir=c.get("depth_dir"), image_ext=c.get("image_ext", ".jpg"), max_items=max_items,
                          waste_colors=c.get("waste_colors"))
    if t == "masks":
        ids = None
        if "split" in c:
            probe = MaskFolderAdapter(c["name"], _resolve(root, c["images_dir"]), _resolve(root, c["masks_dir"]))
            ids = split_ids([p.stem for p in probe.items], c["split"]["fractions"], c["split"]["seed"])[split]
        return MaskFolderAdapter(c["name"], _resolve(root, c["images_dir"]), _resolve(root, c["masks_dir"]), ids=ids,
                                 mask_suffix=c.get("mask_suffix", ".png"), threshold=c.get("threshold", 0),
                                 entry=c.get("entry"), surroundings=c.get("surroundings"),
                                 ignore_value=c.get("ignore_value"), max_items=max_items)
    if t == "synthetic":
        n = c.get("n", {}).get(split, 200) if isinstance(c.get("n"), dict) else c.get("n", 200)
        return SyntheticAdapter(c["name"], n=n, size=c.get("size", 256), seed=c.get("seed", 0), split=split,
                                env_bias=c.get("env_bias", 0.8))
    raise ValueError(f"unknown dataset type {t!r}")


def build_split(ds_cfgs: list[dict], names: list[str] | None, split: str, atlas: Atlas, root: Path,
                holdout: set[str] = frozenset(), max_items=None, strict=True) -> list:
    """Adapters for one split. Held-out entries remove whole images from train/val."""
    out = []
    for cfg in ds_cfgs:
        if names is not None and cfg["name"] not in names:
            continue
        a = build_adapter(cfg, split, root, max_items=max_items)
        if a is None:
            continue
        unknown = sorted(e for e in a.entries_present() if e not in atlas and e not in a.eval_only)
        if unknown:
            msg = f"[{cfg['name']}] entries not in the atlas: {unknown}"
            if strict:
                raise KeyError(msg + " (add them to the atlas or map them with class_map)")
            print("WARNING", msg, "-> trained as plain waste, no descriptors")
        if split in ("train", "val") and holdout:
            n = a.drop_items_with(set(holdout))
            leaked = a.entries_present() & set(holdout)
            assert not leaked, f"held-out entries leaked into {split}: {leaked}"
            print(f"[{cfg['name']}/{split}] dropped {n} images containing held-out classes")
        print(f"[{cfg['name']}/{split}] {len(a)} images")
        out.append(a)
    return out
