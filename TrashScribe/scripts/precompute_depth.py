"""Relative depth maps for the geometry-guidance ablation (E5 'geometry'), from a
monocular model, saved as <depth_dir>/<image path>.npy mirroring the images.

    python scripts/precompute_depth.py --images ~/data/waste/TACO/data --out ~/data/waste/TACO/depth
then add `depth_dir: TACO/depth` to that dataset block. RGB-D datasets (MJU-Waste)
use their sensor depth instead.
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from trashscribe.device import pick_device

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
ap.add_argument("--max-side", type=int, default=1024)
a = ap.parse_args()
from transformers import AutoImageProcessor, AutoModelForDepthEstimation  # noqa: E402
dev = pick_device()
proc = AutoImageProcessor.from_pretrained(a.model)
net = AutoModelForDepthEstimation.from_pretrained(a.model).to(dev).eval()
root, out = Path(a.images), Path(a.out)
files = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
for i, p in enumerate(files):
    dst = out / p.relative_to(root).with_suffix(".npy")
    if dst.exists():
        continue
    im = Image.open(p).convert("RGB")
    im.thumbnail((a.max_side, a.max_side))
    with torch.no_grad():
        d = net(**proc(images=im, return_tensors="pt").to(dev)).predicted_depth
        d = torch.nn.functional.interpolate(d[:, None], size=im.size[::-1], mode="bicubic")[0, 0]
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, d.float().cpu().numpy().astype(np.float16))
    if i % 100 == 0:
        print(f"{i}/{len(files)}")
