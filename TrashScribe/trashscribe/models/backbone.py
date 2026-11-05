"""Shared encoder + class-agnostic waste decoder.

Sam2Backbone is the LENTE recipe applied to RGB waste imagery: SAM2-Hiera-Tiny
frozen, zero-initialised adapters (and optionally LoRA) inside the encoder, the
mask decoder trained prompt-free so that it answers one question for every
dataset: "which pixels are waste?".

TinyBackbone exposes the same interface with a few convolutions; it exists only
so that tests and the synthetic sanity benchmark run without downloading SAM2.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .peft import Adapter, LoRALinear, MixStyle, output_hook

# channels of the three feature levels handed to the heads: S/4, S/8, S/16
LEVEL_CHANNELS = (32, 64, 256)


def _sam2_config(ckpt: str, image_size: int):
    """Sam2Config at a resolution other than 1024 (same logic as LENTE).

    Hiera's position embeddings are interpolated, so no weight changes shape; but
    `backbone_feature_sizes` and the prompt/decoder image sizes are hard-wired
    to the resolution and must be rewritten.
    """
    from transformers import Sam2Config
    cfg = Sam2Config.from_pretrained(ckpt)
    s = int(image_size)
    if s % 32:
        raise ValueError(f"image_size must be a multiple of 32, got {s}")
    cfg.image_size = s
    vc = cfg.vision_config
    vc.image_size = s
    if hasattr(vc, "backbone_feature_sizes"):
        vc.backbone_feature_sizes = [[s // 4] * 2, [s // 8] * 2, [s // 16] * 2]
    for sub in ("prompt_encoder_config", "mask_decoder_config"):
        sc = getattr(cfg, sub, None)
        if sc is not None and hasattr(sc, "image_size"):
            sc.image_size = s
    return cfg


class Sam2Backbone(nn.Module):
    def __init__(self, ckpt="facebook/sam2.1-hiera-tiny", image_size=1024, adapter_dim=64,
                 adapter_where="attn", lora_rank=0, lora_alpha=16.0, train_decoder=True,
                 train_patch_embed=False, mixstyle_p=0.0):
        super().__init__()
        from transformers import Sam2Model
        self.image_size = int(image_size)
        cfg = None if self.image_size == 1024 else _sam2_config(ckpt, self.image_size)
        self.model = Sam2Model.from_pretrained(ckpt, config=cfg) if cfg else Sam2Model.from_pretrained(ckpt)
        # Same fix as LENTE: in eval SAM2 would swap mask 0 for an untrained one.
        self.model.mask_decoder.dynamic_multimask_via_stability = False
        for p in self.model.parameters():
            p.requires_grad = False

        enc = self.model.vision_encoder
        self.adapters = nn.ModuleList()
        self.lora = nn.ModuleList()
        if adapter_where not in ("attn", "mlp", "both"):
            raise ValueError("adapter_where must be attn|mlp|both")
        for name, mod in list(enc.named_modules()):
            if hasattr(mod, "qkv") and isinstance(mod.qkv, (nn.Linear, LoRALinear)):
                if adapter_dim > 0 and adapter_where in ("attn", "both"):
                    ad = Adapter(mod.proj.out_features, adapter_dim)
                    self.adapters.append(ad)
                    mod.register_forward_hook(output_hook(ad))
                if lora_rank > 0:
                    for attr in ("qkv", "proj"):
                        w = LoRALinear(getattr(mod, attr), lora_rank, lora_alpha)
                        setattr(mod, attr, w)
                        self.lora.append(w)
            elif adapter_dim > 0 and adapter_where in ("mlp", "both") and name.endswith(".mlp"):
                ad = Adapter(mod.proj_out.out_features, adapter_dim)
                self.adapters.append(ad)
                mod.register_forward_hook(output_hook(ad))

        self.mixstyle = MixStyle(p=mixstyle_p, channel_last=True) if mixstyle_p > 0 else None
        if self.mixstyle is not None:
            enc.backbone.patch_embed.register_forward_hook(output_hook(self.mixstyle))

        if train_decoder:
            for p in self.model.mask_decoder.parameters():
                p.requires_grad = True
        if train_patch_embed:
            for p in enc.backbone.patch_embed.parameters():
                p.requires_grad = True

    def features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """[S/4 (32ch), S/8 (64ch), S/16 (256ch)] as (B, C, h, w). Gradients flow
        (Sam2Model.get_image_embeddings is decorated with no_grad, so it is not used)."""
        b = x.shape[0]
        out = self.model.get_image_features(x, return_dict=True)
        feats = list(out.fpn_hidden_states)
        feats[-1] = feats[-1] + self.model.no_memory_embedding
        return [f.permute(1, 2, 0).reshape(b, -1, *size)
                for f, size in zip(feats, self.model.backbone_feature_sizes)]

    def decode(self, feats: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Prompt-free waste logits (B, 1, S/4, S/4) and predicted IoU (B, 1)."""
        out = self.model(image_embeddings=feats, multimask_output=False)
        masks = out.pred_masks.flatten(1, 2)[:, :1]
        return masks, out.iou_scores.flatten(1)[:, :1]

    @torch.no_grad()
    def merge_lora_(self):
        """Fold LoRA into the frozen Linears (export). Adapters stay as modules."""
        for mod in self.model.vision_encoder.modules():
            for attr in ("qkv", "proj"):
                w = getattr(mod, attr, None)
                if isinstance(w, LoRALinear):
                    setattr(mod, attr, w.merged())
        self.lora = nn.ModuleList()


class TinyBackbone(nn.Module):
    """Same interface as Sam2Backbone, a few convolutions. Tests / sanity only."""

    def __init__(self, image_size=256, width=32, **_):
        super().__init__()
        self.image_size = int(image_size)
        c0, c1, c2 = LEVEL_CHANNELS

        def blk(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.stem = nn.Sequential(blk(3, width, 2), blk(width, c0, 2))      # S/4
        self.s1 = blk(c0, c1, 2)                                              # S/8
        self.s2 = nn.Sequential(blk(c1, 128, 2), blk(128, c2, 1))             # S/16
        self.head = nn.Sequential(nn.Conv2d(c0 + c1 + c2, 64, 3, 1, 1), nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 1, 1))
        self.iou = nn.Linear(c2, 1)
        self.adapters, self.lora, self.mixstyle = nn.ModuleList(), nn.ModuleList(), None

    def features(self, x):
        f0 = self.stem(x)
        f1 = self.s1(f0)
        return [f0, f1, self.s2(f1)]

    def decode(self, feats):
        f0, f1, f2 = feats
        h = f0.shape[-2:]
        z = torch.cat([f0, F.interpolate(f1, h, mode="bilinear", align_corners=False),
                       F.interpolate(f2, h, mode="bilinear", align_corners=False)], 1)
        return self.head(z), self.iou(f2.mean((2, 3)))

    def merge_lora_(self):
        pass
