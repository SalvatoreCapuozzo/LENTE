"""Frozen CLIP image teacher for the distilled CLIP field (E8).

For every ground-truth object of the (augmented) training batch it crops a square
around the object's box, 1.35x its longest side (the context rule of the raw-CLIP
reference in baseline_zeroshot.py / posthoc.py), directly on the GPU with roi_align
(adaptive sampling = area averaging, no aliasing for large objects), and returns
CLIP ViT-B/16 image embeddings (projected, L2-normalised). No gradient, no extra
data pass; the targets see exactly the pixels the student sees, transplants and
photometric augmentation included.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .data.canonical import IMAGENET_MEAN, IMAGENET_STD

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def object_boxes(inst: list[torch.Tensor], stride: int = 4):
    """Pixel boxes (x0, y0, x1, y1) of each instance mask (stride-4 masks) and the
    image index of each; objects with no mask cell get an empty box (size 0)."""
    boxes, bidx = [], []
    for b, m in enumerate(inst):
        if m is None or m.shape[0] == 0:
            continue
        on = m > 0.5
        rows, cols = on.any(2), on.any(1)                                  # (n, H4), (n, W4)
        H4, W4 = m.shape[-2:]
        ar_h = torch.arange(H4, device=m.device)[None]
        ar_w = torch.arange(W4, device=m.device)[None]
        big = 10 ** 6
        y0 = torch.where(rows, ar_h, big).amin(1)
        y1 = torch.where(rows, ar_h, -1).amax(1) + 1
        x0 = torch.where(cols, ar_w, big).amin(1)
        x1 = torch.where(cols, ar_w, -1).amax(1) + 1
        empty = ~rows.any(1)
        bx = torch.stack([x0, y0, x1, y1], 1).float() * stride
        bx[empty] = 0
        boxes.append(bx)
        bidx.append(torch.full((m.shape[0],), b, dtype=torch.long, device=m.device))
    if not boxes:
        dev = inst[0].device if inst else "cpu"
        return torch.zeros(0, 4, device=dev), torch.zeros(0, dtype=torch.long, device=dev)
    return torch.cat(boxes), torch.cat(bidx)


class ClipTeacher:
    def __init__(self, device, name="openai/clip-vit-base-patch16", context=1.35, min_side=48, max_objects=24):
        from transformers import CLIPVisionModelWithProjection
        self.dev = device
        dt = torch.float16 if device.type == "cuda" else torch.float32
        self.model = CLIPVisionModelWithProjection.from_pretrained(name, torch_dtype=dt).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dtype = dt
        self.size = int(self.model.config.image_size)
        self.context, self.min_side, self.max_objects = context, min_side, max_objects
        im = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        isd = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        cm = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        cs = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        # x_clip = (x_in * std_in + mean_in - mean_clip) / std_clip  = a * x_in + c
        self.a = (isd / cs).to(device)
        self.c = ((im - cm) / cs).to(device)

    @torch.no_grad()
    def __call__(self, image: torch.Tensor, inst: list[torch.Tensor], generator=None):
        """image (B, 3, H, W) ImageNet-normalised; inst per image (n_b, H/4, W/4).
        Returns (sel, emb): indices into the concatenated objects (the order of
        pool_objects) and their (len(sel), 512) normalised CLIP embeddings."""
        from torchvision.ops import roi_align
        boxes, bidx = object_boxes(inst, stride=image.shape[-1] // inst[0].shape[-1] if inst else 4)
        if boxes.shape[0] == 0:
            return torch.zeros(0, dtype=torch.long, device=image.device), None
        wh = boxes[:, 2:] - boxes[:, :2]
        ok = wh.max(1).values >= self.min_side          # longest side: thin objects (straws) stay
        sel = torch.nonzero(ok).flatten()
        if sel.numel() == 0:
            return sel, None
        if sel.numel() > self.max_objects:
            perm = torch.randperm(sel.numel(), device=sel.device, generator=generator)[: self.max_objects]
            sel = sel[perm.sort().values]
        b = boxes[sel]
        ctr = (b[:, :2] + b[:, 2:]) / 2
        half = (b[:, 2:] - b[:, :2]).max(1).values[:, None] * self.context / 2
        rois = torch.cat([bidx[sel, None].float(), ctr - half, ctr + half], 1)
        x = image.float() * self.a + self.c
        crops = roi_align(x, rois, output_size=self.size, spatial_scale=1.0, sampling_ratio=-1, aligned=True)
        emb = self.model(pixel_values=crops.to(self.dtype)).image_embeds
        return sel, F.normalize(emb.float(), dim=-1)


def distill_loss(z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - cosine between pooled student vectors and teacher embeddings."""
    return (1 - (F.normalize(z, dim=-1) * target).sum(-1)).mean()
