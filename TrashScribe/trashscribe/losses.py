"""Training objectives.

L = w_mask (BCE + Dice) on the waste mask           every dataset
  + w_contour BCE on object boundaries               datasets with real instances
  + w_desc  descriptor loss on mask-pooled objects   objects whose class is described
  + w_dense descriptor loss on sampled pixels        idem (makes free-text heatmaps usable)
  + w_adv   environment CE through gradient reversal objects whose surroundings are known
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .schema import Schema


def mask_loss(logits, target, valid, pos_weight=2.0, dice_w=1.0):
    lg = F.interpolate(logits.float(), size=target.shape[-2:], mode="bilinear", align_corners=False)
    v = valid.float()
    pw = torch.tensor(pos_weight, device=lg.device)
    bce = F.binary_cross_entropy_with_logits(lg, target, pos_weight=pw, reduction="none")
    bce = (bce * v).sum() / v.sum().clamp_min(1.0)
    p, t = torch.sigmoid(lg) * v, target * v
    inter = (p * t).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + 1)
    return bce + dice_w * dice.mean()


def contour_loss(logits, target, flags, pos_weight=4.0):
    if not bool(flags.any()):
        return logits.sum() * 0.0
    lg, t = logits[flags].float(), target[flags]
    pw = torch.tensor(pos_weight, device=lg.device)
    return F.binary_cross_entropy_with_logits(lg, t, pos_weight=pw)


def field_layout(schema: Schema):
    return [(schema.slices[f.name], f.kind, f.scope) for f in schema.fields]


def descriptor_loss(logits, tgt, msk, layout, scopes=("intrinsic", "contextual")):
    """Soft-target CE for single-valued fields, BCE for multi-valued ones. A field
    contributes only for objects where it is described (mask), so a blank field
    is never read as a negative."""
    total = logits.new_zeros(())
    count = 0
    for sl, kind, scope in layout:
        if scope not in scopes:
            continue
        m = msk[:, sl].all(1)
        if not bool(m.any()):
            continue
        lg, t = logits[m, sl].float(), tgt[m, sl]
        if kind == "single":
            l = -(t * F.log_softmax(lg, -1)).sum(-1)
        else:
            l = F.binary_cross_entropy_with_logits(lg, t, reduction="none").mean(-1)
        total = total + l.sum()
        count += int(m.sum())
    return total / max(count, 1)


def sample_object_pixels(desc_map, inst, n_per_obj=16, min_cov=0.5):
    """Pick up to n cells (stride 8) inside each object. Returns (M, D) features and
    the object index (into the flattened object list) of each sampled cell."""
    B, D, h, w = desc_map.shape
    feats, owner, base = [], [], 0
    for b, m in enumerate(inst):
        if m is None or m.shape[0] == 0:
            continue
        cov = F.adaptive_avg_pool2d(m[:, None].float(), (h, w))[:, 0]
        f = desc_map[b].flatten(1)
        for j in range(m.shape[0]):
            idx = torch.nonzero(cov[j].flatten() >= min_cov).flatten()
            if idx.numel() == 0:
                continue
            if idx.numel() > n_per_obj:
                idx = idx[torch.randperm(idx.numel(), device=idx.device)[:n_per_obj]]
            feats.append(f[:, idx].T)
            owner.append(torch.full((idx.numel(),), base + j, dtype=torch.long, device=desc_map.device))
        base += m.shape[0]
    if not feats:
        return desc_map.new_zeros(0, D), torch.zeros(0, dtype=torch.long, device=desc_map.device)
    return torch.cat(feats), torch.cat(owner)


def dann_lambda(progress: float, gamma: float = 10.0) -> float:
    """Gradient-reversal schedule of Ganin & Lempitsky (2015): 0 -> 1 over training."""
    return 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0
