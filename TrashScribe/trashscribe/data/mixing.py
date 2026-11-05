"""Multi-dataset training: object transplant across environments and
dataset-balanced sampling.

Why transplant. In every waste dataset the object and its surroundings are
correlated (beach -> fragments and nets, conveyor -> cardboard and film, street
-> cans and cigarettes). A model can therefore "recognise" materials from the
background. Pasting annotated objects from one dataset onto images of another
breaks that correlation at the pixel level, so the only stable signal left for
the descriptor head is the object itself; it also lets the environment
adversary remove the background from object embeddings without fighting the
descriptor loss.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .canonical import Canonicalizer, RawSample, letterbox


def transplant(src: RawSample, bg: RawSample, rng: np.random.Generator, max_objects=6,
               scale=(0.6, 1.3), harmonize=0.5, feather=1.5, min_area=64) -> RawSample:
    """Paste up to `max_objects` instances of `src` onto `bg` (both letterboxed)."""
    S = bg.image.shape[0]
    cand = [i for i, m in enumerate(src.instances) if int(m.sum()) >= min_area]
    if not cand:
        return bg
    picks = rng.choice(cand, size=int(rng.integers(1, min(max_objects, len(cand)) + 1)), replace=False)
    img = bg.image.astype(np.float32).copy()
    valid_bg = bg.valid if bg.valid is not None else np.ones((S, S), np.uint8)
    ys, xs = np.nonzero(valid_bg)
    if len(ys) == 0:
        return bg
    vy0, vy1, vx0, vx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    region_px = img[vy0:vy1, vx0:vx1].reshape(-1, 3)
    bg_mu, bg_sd = region_px.mean(0), region_px.std(0) + 1e-3
    waste = bg.waste_mask().copy()
    valid = valid_bg.copy()
    insts = [m.copy() for m in bg.instances]
    ents = list(bg.entries)
    for i in picks:
        m = src.instances[i]
        yy, xx = np.nonzero(m)
        y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
        crop = src.image[y0:y1, x0:x1].astype(np.float32)
        cm = m[y0:y1, x0:x1].astype(np.float32)
        k = float(rng.uniform(*scale))
        h, w = cm.shape
        k = min(k, 0.6 * (vy1 - vy0) / h, 0.6 * (vx1 - vx0) / w)
        nh, nw = max(2, int(h * k)), max(2, int(w * k))
        crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
        cm = cv2.resize(cm, (nw, nh), interpolation=cv2.INTER_NEAREST)
        if cm.sum() < 4:
            continue
        a = float(rng.uniform(0, harmonize))
        if a > 0:
            px = crop[cm > 0]
            mu, sd = px.mean(0), px.std(0) + 1e-3
            crop = crop * (1 - a) + a * ((crop - mu) / sd * bg_sd + bg_mu)
        ty = int(rng.integers(vy0, max(vy0 + 1, vy1 - nh)))
        tx = int(rng.integers(vx0, max(vx0 + 1, vx1 - nw)))
        nh, nw = min(nh, S - ty), min(nw, S - tx)
        cm, crop = cm[:nh, :nw], crop[:nh, :nw]
        alpha = np.clip(cv2.GaussianBlur(cm, (0, 0), feather) * 1.5, 0, 1) if feather > 0 else cm
        alpha = np.maximum(alpha, cm * 0.9)[..., None]
        reg = img[ty:ty + nh, tx:tx + nw]
        img[ty:ty + nh, tx:tx + nw] = reg * (1 - alpha) + crop * alpha
        full = np.zeros((S, S), np.uint8)
        full[ty:ty + nh, tx:tx + nw] = (cm > 0).astype(np.uint8)
        insts = [(q & (1 - full)).astype(np.uint8) for q in insts]
        insts.append(full)
        ents.append(src.entries[i])
        waste[full == 1] = 1
        valid[full == 1] = 1
    keep = [j for j, q in enumerate(insts) if int(q.sum()) >= 16]
    return RawSample(image=np.clip(img, 0, 255).astype(np.uint8), instances=[insts[j] for j in keep],
                     entries=[ents[j] for j in keep], dataset=bg.dataset,
                     image_id=f"{src.image_id}->{bg.image_id}", waste=waste, valid=valid,
                     surroundings=bg.surroundings, has_instances=src.has_instances and bg.has_instances,
                     depth=bg.depth, meta={"transplant_from": src.dataset})


class MultiDataset(Dataset):
    def __init__(self, adapters: list, canon: Canonicalizer, transplant_p: float = 0.0,
                 transplant_cfg: dict | None = None, cross_dataset_only: bool = True):
        self.adapters = [a for a in adapters if len(a) > 0]
        self.canon = canon
        self.size = canon.size
        self.transplant_p = transplant_p if canon.train else 0.0
        self.tcfg = transplant_cfg or {}
        self.cross_only = cross_dataset_only and len(self.adapters) > 1
        self.index = [(ai, j) for ai, a in enumerate(self.adapters) for j in range(len(a))]
        self._rng, self._pid = None, None

    @property
    def rng(self) -> np.random.Generator:
        if self._rng is None or self._pid != os.getpid():
            self._pid = os.getpid()
            self._rng = np.random.default_rng((torch.initial_seed() + self._pid) % 2 ** 32)
        return self._rng

    def __len__(self):
        return len(self.index)

    def raw(self, k: int) -> RawSample:
        ai, j = self.index[k]
        return letterbox(self.adapters[ai].load(j), self.size)

    def __getitem__(self, k):
        s = self.raw(k)
        if self.transplant_p > 0 and self.rng.random() < self.transplant_p:
            ai = self.index[k][0]
            pool = [a for a in range(len(self.adapters)) if a != ai] if self.cross_only else list(range(len(self.adapters)))
            a2 = int(self.rng.choice(pool))
            j2 = int(self.rng.integers(len(self.adapters[a2])))
            bg = letterbox(self.adapters[a2].load(j2), self.size)
            s = transplant(s, bg, self.rng, **self.tcfg)
        return self.canon(s)

    def sample_weights(self, alpha: float = 0.5) -> torch.Tensor:
        """Temperature sampling over datasets: P(d) ∝ N_d^alpha (alpha=0 uniform over
        datasets, 1 proportional to size); uniform within each dataset."""
        sizes = np.array([len(a) for a in self.adapters], dtype=np.float64)
        pd = sizes ** alpha / (sizes ** alpha).sum()
        return torch.tensor([pd[ai] / sizes[ai] for ai, _ in self.index], dtype=torch.double)

    def dataset_of(self, k: int) -> str:
        return self.adapters[self.index[k][0]].name
