"""From maps to monitoring indicators and a written report.

instances: waste probability minus contours -> seeds -> every waste pixel is
assigned to its nearest seed. No per-object decoder call, so cost does not grow
with the number of objects (important on a Jetson in a cluttered scene).
"""
from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
import torch
from scipy import ndimage


def extract_instances(waste_prob: np.ndarray, contour_prob: np.ndarray, thr=0.5, cthr=0.5, min_cells=6):
    """(h, w) maps at stride 4 -> list of boolean (h, w) instance masks."""
    fg = waste_prob > thr
    seeds = fg & (contour_prob < cthr)
    n, lab = cv2.connectedComponents(seeds.astype(np.uint8))
    if n <= 1:
        n, lab = cv2.connectedComponents(fg.astype(np.uint8))
        if n <= 1:
            return []
    idx = ndimage.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
    full = lab[idx[0], idx[1]]
    full[~fg] = 0
    out = [(full == k) for k in range(1, int(full.max()) + 1)]
    return [m for m in out if m.sum() >= min_cells]


def pool_numpy(desc: torch.Tensor, inst: list[np.ndarray], ring_cells=3):
    """desc (D, h8, w8); inst at stride 4 -> z_int, z_ctx (N, D) via the training pooling."""
    from .models.trashscribe import pool_objects
    if not inst:
        z = desc.new_zeros(0, desc.shape[0])
        return z, z
    m = torch.from_numpy(np.stack(inst).astype(np.float32)).to(desc.device)
    zi, zc, _ = pool_objects(desc[None], [m], ring_cells)
    return zi, zc


def frame_report(waste_prob, valid_frac, inst, named: dict | None, namer=None, posterior=None) -> dict:
    """Indicators for one frame. `named` is the output of Namer.name on the instances."""
    total = waste_prob.size * max(valid_frac, 1e-6)
    objs, by_stream, by_material = [], defaultdict(float), defaultdict(float)
    for k, m in enumerate(inst):
        area = float(m.sum() / total)
        ys, xs = np.nonzero(m)
        o = {"id": k, "area_frac": area, "bbox_s4": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]}
        if named is not None:
            o["name"] = None if bool(named["abstain"][k]) else named["name"][k]
            o["confidence"] = float(named["prob"][k])
            if namer is not None and o["name"] is not None:
                st = namer.c.entry_streams[namer.c.entry_names.index(o["name"])]
                o["stream"] = st
                by_stream[st or "unspecified"] += area
            elif o["name"] is None:
                by_stream["set aside"] += area
        if namer is not None and posterior is not None:
            d = namer.describe(posterior[k])
            o["description"] = d
            for mat, p in d.get("material", []):
                by_material[mat] += area * p
        objs.append(o)
    return {"coverage": float((waste_prob > 0.5).sum() / total), "count": len(inst), "objects": objs,
            "composition_by_stream": dict(by_stream), "composition_by_material": dict(by_material)}


def sentence(obj: dict) -> str:
    """Report line composed only from scored descriptor values (every clause traces
    to a visual property, nothing is generated)."""
    d = obj.get("description", {})
    parts = []
    for fld in ("material", "rigidity", "optics", "shape", "state"):
        vals = [v for v, _ in d.get(fld, [])]
        if vals:
            parts.append(" and ".join(vals))
    desc = "; ".join(parts) if parts else "no confident description"
    name = obj.get("name") or "set aside (no atlas entry explains it)"
    st = f", stream: {obj['stream']}" if obj.get("stream") else ""
    return f"object {obj['id']} ({100 * obj['area_frac']:.1f}% of frame): {desc} -> {name}{st}"
