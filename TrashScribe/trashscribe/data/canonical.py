"""Canonical waste representation shared by every dataset.

Adapters emit a RawSample in native resolution and native labels already mapped
to atlas entry names. `letterbox` brings it to the model resolution, the object
transplant may then mix it with a background from another dataset, and
`finalize` applies augmentation and builds the training tensors.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np
import torch

from ..schema import Atlas, Schema, context_target, entry_target

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IGNORE = 255


@dataclass
class RawSample:
    image: np.ndarray                         # (H, W, 3) uint8 RGB
    instances: list[np.ndarray]               # each (H, W) uint8 in {0, 1}
    entries: list[str | None]                 # atlas entry per instance (None: describe nothing)
    dataset: str
    image_id: str
    waste: np.ndarray | None = None           # (H, W) uint8 {0, 1, IGNORE}; None -> union of instances
    valid: np.ndarray | None = None           # (H, W) uint8 {0, 1}; padding = 0
    surroundings: str | None = None           # value of the contextual field, if known
    has_instances: bool = True                # instance masks are real (not connected components)
    depth: np.ndarray | None = None           # (H, W) float32, any scale
    meta: dict = field(default_factory=dict)

    def waste_mask(self) -> np.ndarray:
        if self.waste is not None:
            return self.waste
        h, w = self.image.shape[:2]
        m = np.zeros((h, w), np.uint8)
        for inst in self.instances:
            m |= inst.astype(np.uint8)
        return m


def letterbox(s: RawSample, size: int) -> RawSample:
    """Aspect-preserving resize of the longest side to `size`, padded top-left."""
    h, w = s.image.shape[:2]
    k = size / max(h, w)
    nh, nw = max(1, round(h * k)), max(1, round(w * k))

    def put(arr, interp, fill=0, dtype=None):
        r = cv2.resize(arr, (nw, nh), interpolation=interp)
        out = np.full((size, size) + r.shape[2:], fill, dtype=dtype or r.dtype)
        out[:nh, :nw] = r
        return out

    img = put(s.image, cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    waste = put(s.waste_mask(), cv2.INTER_NEAREST)
    valid0 = s.valid if s.valid is not None else np.ones((h, w), np.uint8)
    valid = put(valid0, cv2.INTER_NEAREST)
    insts, ents = [], []
    for m, e in zip(s.instances, s.entries):
        r = put(m, cv2.INTER_NEAREST)
        if r.any():
            insts.append(r)
            ents.append(e)
    depth = put(s.depth.astype(np.float32), cv2.INTER_LINEAR) if s.depth is not None else None
    return replace(s, image=img, waste=waste, valid=valid, instances=insts, entries=ents, depth=depth,
                   meta={**s.meta, "scale": k, "orig_hw": (h, w)})


def build_augment(cfg: dict | None) -> A.Compose:
    """Geometric + photometric augmentation at model resolution. The photometric
    part is deliberately strong: camera and illumination are what differ most
    between waste datasets and must not be what the model keys on."""
    import albumentations as A   # training only: the edge runtime must not need it
    cfg = cfg or {}
    s = float(cfg.get("strength", 1.0))
    t = [
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(1 - 0.4 * s, 1 + 0.4 * s), translate_percent=(-0.1 * s, 0.1 * s),
                 rotate=(-15 * s, 15 * s), border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.7),
        A.ColorJitter(brightness=0.4 * s, contrast=0.4 * s, saturation=0.4 * s, hue=0.05 * s, p=0.8),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.MotionBlur(blur_limit=(3, 9))], p=0.3 * s),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.25 * s),
        A.ImageCompression(quality_range=(35, 95), p=0.3 * s),
        A.ToGray(p=0.05),
        A.RandomShadow(p=0.15 * s),
    ]
    if cfg.get("vflip", False):          # aerial imagery has no canonical "up"
        t.insert(1, A.VerticalFlip(p=0.5))
        t.insert(2, A.RandomRotate90(p=0.5))
    return A.Compose(t)


def pack_instances(insts: list[np.ndarray]) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Instance masks -> a few uint16 label layers (objects that do not overlap
    share a layer) and, per instance, its (layer, label). Geometric augmentation
    then warps K layers instead of N masks; with nearest-neighbour resampling this
    is exact, since "pixel == label" commutes with it. BePLi has up to 363 objects
    per image: N full-resolution masks per sample exhausted host RAM."""
    layers: list[np.ndarray] = []
    n_lab: list[int] = []
    where: list[tuple[int, int]] = []
    for m in insts:
        mb = m.astype(bool)
        for k, L in enumerate(layers):
            if not L[mb].any():
                break
        else:
            layers.append(np.zeros(m.shape, np.uint16))
            n_lab.append(0)
            k = len(layers) - 1
        n_lab[k] += 1
        layers[k][mb] = n_lab[k]
        where.append((k, n_lab[k]))
    return layers, where


class Canonicalizer:
    """RawSample at model resolution -> dict of tensors."""

    def __init__(self, schema: Schema, atlas: Atlas, size: int, train: bool, aug_cfg: dict | None = None,
                 min_area: int = 16):
        self.schema, self.atlas, self.size, self.train = schema, atlas, size, train
        self.aug = build_augment(aug_cfg) if train else None
        self.min_area = min_area
        sur = schema.field("surroundings").values if "surroundings" in schema.slices else []
        self.env_values = list(sur)

    def __call__(self, s: RawSample) -> dict:
        if s.image.shape[0] != self.size or s.image.shape[1] != self.size:
            s = letterbox(s, self.size)
        img, waste, valid, insts, ents, depth = s.image, s.waste_mask(), s.valid, s.instances, s.entries, s.depth
        if valid is None:
            valid = np.ones(img.shape[:2], np.uint8)
        layers, where = pack_instances(insts)
        del insts
        if self.aug is not None:
            masks = [waste, valid] + layers + ([depth] if depth is not None else [])
            out = self.aug(image=img, masks=masks)
            img, masks = out["image"], out["masks"]
            waste, valid = masks[0], masks[1]
            layers = list(masks[2:2 + len(layers)])
            depth = masks[-1].astype(np.float32) if depth is not None else None
        n_lab = [0] * len(layers)
        for k, lab in where:
            n_lab[k] = max(n_lab[k], lab)
        # minlength from the labels assigned, not from L.max(): augmentation can push
        # an object out of the frame
        counts = [np.bincount(L.ravel(), minlength=n + 1) for L, n in zip(layers, n_lab)]
        keep = [i for i, (k, lab) in enumerate(where) if int(counts[k][lab]) >= self.min_area]
        ents = [ents[i] for i in keep]

        S, S4 = self.size, self.size // 4
        x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        valid_t = (valid > 0) & (waste != IGNORE)
        waste_t = (waste == 1)

        if keep:      # one object at a time: a float32 stack of all masks at S x S took >1 GB on BePLi
            inst4 = np.stack([cv2.resize((layers[where[i][0]] == where[i][1]).astype(np.float32), (S4, S4),
                                         interpolation=cv2.INTER_AREA) for i in keep])
        else:
            inst4 = np.zeros((0, S4, S4), np.float32)
        contour = np.zeros((S4, S4), np.float32)
        k = np.ones((3, 3), np.uint8)
        for m in inst4:
            b = (m > 0.5).astype(np.uint8)
            contour = np.maximum(contour, (b - cv2.erode(b, k)).astype(np.float32))

        ctx_t, ctx_m = context_target(s.surroundings, self.schema)
        tgts, msks = [], []
        for e in ents:
            t, m = entry_target(self.atlas[e] if (e is not None and e in self.atlas) else None, self.schema)
            tgts.append(t + ctx_t)
            msks.append(m | ctx_m)
        V = self.schema.n_values
        env_id = self.env_values.index(s.surroundings) if s.surroundings in self.env_values else -1
        return {
            "image": torch.from_numpy(x.transpose(2, 0, 1).copy()),
            "waste": torch.from_numpy(waste_t[None].astype(np.float32)),
            "valid": torch.from_numpy(valid_t[None]),
            "depth": torch.from_numpy(depth[None].astype(np.float32)) if depth is not None else None,
            "inst": torch.from_numpy(inst4).to(torch.float16),
            "contour": torch.from_numpy(contour[None]),
            "contour_valid": bool(s.has_instances),
            "desc_t": torch.from_numpy(np.stack(tgts)) if tgts else torch.zeros(0, V),
            "desc_m": torch.from_numpy(np.stack(msks)) if msks else torch.zeros(0, V, dtype=torch.bool),
            "entries": ents,
            "env_id": env_id,
            "dataset": s.dataset,
            "image_id": s.image_id,
        }


def collate(batch: list[dict]) -> dict:
    out = {k: torch.stack([b[k] for b in batch]) for k in ("image", "waste", "valid", "contour")}
    out["contour_valid"] = torch.tensor([b["contour_valid"] for b in batch])
    out["env_id"] = torch.tensor([b["env_id"] for b in batch], dtype=torch.long)
    depths = [b["depth"] for b in batch]
    out["depth"] = torch.stack(depths) if all(d is not None for d in depths) else None
    for k in ("inst", "desc_t", "desc_m", "entries", "dataset", "image_id"):
        out[k] = [b[k] for b in batch]
    return out


def to_device(batch: dict, device) -> dict:
    # non_blocking only on CUDA (pinned memory); on MPS it can corrupt small copies
    nb = torch.device(device).type == "cuda"
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=nb)
        elif isinstance(v, list) and v and torch.is_tensor(v[0]):
            out[k] = [t.to(device, non_blocking=nb) for t in v]
        else:
            out[k] = v
    return out
