"""TrashScribe: segment waste, describe it in words, name it by reading an atlas.

    image ──(optional geometric gate)──► SAM2-Hiera-Tiny + adapters (frozen core)
                                              │
                ┌─────────────────────────────┼──────────────────────────┐
                ▼                             ▼                          ▼
        waste decoder (where)        contour head (split)      descriptor field (what)
        prompt-free, 1 class         touching objects          per-pixel vector in the
        shared by every dataset                                frozen text space
                                                                        │
                                       mask pooling: inside (intrinsic) / ring (context)
                                                                        │
                                   phrases of the schema, re-projected by W_t, scored
                                   with per-field tau/b only ──► descriptor posterior
                                                                        │
                                          atlas (text file) ──► name or abstain
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Sam2Backbone, TinyBackbone
from .heads import (ContourHead, DescriptorField, EnvironmentAdversary, FieldScorer,
                    GeometryGuidance, TextReprojection)


@dataclass
class ModelConfig:
    backbone: str = "sam2"                  # sam2 | tiny (tests only)
    sam_ckpt: str = "facebook/sam2.1-hiera-tiny"
    image_size: int = 1024
    adapter_dim: int = 64
    adapter_where: str = "attn"
    lora_rank: int = 0
    lora_alpha: float = 16.0
    train_decoder: bool = True
    train_patch_embed: bool = False
    mixstyle_p: float = 0.0
    geometry: bool = False
    desc_hidden: int = 256
    text_rank: int = 128
    ring_cells: int = 3
    adversary: bool = True
    closed_vocab_probe: bool = False        # reference only: one weight vector per descriptor


class TrashScribe(nn.Module):
    def __init__(self, cfg: ModelConfig, text_dim: int, n_fields: int, n_env: int):
        super().__init__()
        self.cfg = cfg
        kw = dict(image_size=cfg.image_size, adapter_dim=cfg.adapter_dim, adapter_where=cfg.adapter_where,
                  lora_rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha, train_decoder=cfg.train_decoder,
                  train_patch_embed=cfg.train_patch_embed, mixstyle_p=cfg.mixstyle_p)
        if cfg.backbone == "sam2":
            self.backbone = Sam2Backbone(ckpt=cfg.sam_ckpt, **kw)
        elif cfg.backbone == "tiny":
            self.backbone = TinyBackbone(**kw)
        else:
            raise ValueError(f"unknown backbone {cfg.backbone!r}")
        self.geometry = GeometryGuidance() if cfg.geometry else None
        self.desc = DescriptorField(text_dim, cfg.desc_hidden)
        self.contour = ContourHead(cfg.desc_hidden)
        self.text_proj = TextReprojection(text_dim, cfg.text_rank)
        self.scorer = FieldScorer(n_fields)
        self.adversary = EnvironmentAdversary(text_dim, n_env) if (cfg.adversary and n_env > 1) else None
        # Closed-vocabulary reference (the "Probe" of the MIRF paper): a linear head
        # with a weight per training descriptor. It cannot score a phrase it was
        # not trained on, which is exactly what the comparison is meant to show.
        self.probe = None
        self._n_probe = 0
        # vocabulary: raw frozen text embeddings of every schema value. Swappable
        # at inference (extended schema) because nothing is indexed by it.
        self.register_buffer("vocab", torch.zeros(0, text_dim), persistent=False)
        self.register_buffer("field_idx", torch.zeros(0, dtype=torch.long), persistent=False)
        self.register_buffer("contextual", torch.zeros(0, dtype=torch.bool), persistent=False)

    # ------------------------------------------------------------ vocabulary
    def set_vocabulary(self, emb: torch.Tensor, field_idx: torch.Tensor, contextual_fields: torch.Tensor):
        """emb (V, D) raw text embeddings; field_idx (V,); contextual_fields (n_fields,) bool."""
        dev = self.text_proj.A.weight.device
        self.vocab = emb.to(dev, torch.float32)
        self.field_idx = field_idx.to(dev)
        self.contextual = contextual_fields.to(dev)[self.field_idx]
        if self.cfg.closed_vocab_probe:
            if self.probe is None:
                self.probe = nn.Linear(emb.shape[1], emb.shape[0]).to(dev)
                self._n_probe = emb.shape[0]
            elif emb.shape[0] != self._n_probe:
                raise ValueError("closed-vocabulary probe cannot score a vocabulary it was not trained on")

    def phrase_matrix(self) -> torch.Tensor:
        return self.text_proj(self.vocab)

    # --------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, depth: torch.Tensor | None = None) -> dict:
        if self.geometry is not None and depth is not None:
            x = self.geometry(x, depth)
        feats = self.backbone.features(x)
        waste, iou = self.backbone.decode(feats)
        dmap, h8 = self.desc(feats)
        contour = self.contour(feats[0], h8)
        return {"waste": waste, "contour": contour, "desc": dmap, "iou": iou}

    def score(self, z_int: torch.Tensor, z_ctx: torch.Tensor | None = None) -> torch.Tensor:
        """(N, V) logits; contextual fields read from the ring embedding."""
        if self.probe is not None:
            lg = self.probe(z_int)
            if z_ctx is not None and bool(self.contextual.any()):
                lg = torch.where(self.contextual[None], self.probe(z_ctx), lg)
            return lg
        P = self.phrase_matrix()
        logits = self.scorer(z_int, P, self.field_idx)
        if z_ctx is not None and bool(self.contextual.any()):
            logits = torch.where(self.contextual[None], self.scorer(z_ctx, P, self.field_idx), logits)
        return logits

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_groups(self, lr_heads: float, lr_peft: float, lr_decoder: float, wd: float):
        peft = {id(p) for m in (self.backbone.adapters, self.backbone.lora) for p in m.parameters()}
        dec = set()
        if isinstance(self.backbone, Sam2Backbone):
            dec = {id(p) for p in self.backbone.model.mask_decoder.parameters()}
        g_peft, g_dec, g_head, g_nowd = [], [], [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in peft:
                g_peft.append(p)
            elif id(p) in dec:
                g_dec.append(p)
            elif n.startswith("scorer.") or p.ndim == 1:
                g_nowd.append(p)
            else:
                g_head.append(p)
        groups = [{"params": g_head, "lr": lr_heads, "weight_decay": wd, "name": "heads"},
                  {"params": g_nowd, "lr": lr_heads, "weight_decay": 0.0, "name": "heads_nowd"},
                  {"params": g_peft, "lr": lr_peft, "weight_decay": wd, "name": "peft"},
                  {"params": g_dec, "lr": lr_decoder, "weight_decay": wd, "name": "decoder"}]
        return [g for g in groups if g["params"]]


def pool_objects(desc_map: torch.Tensor, inst: list[torch.Tensor], ring_cells: int = 3):
    """Mask pooling of the descriptor field.

    desc_map: (B, D, h, w) at stride 8. inst: per image (n_b, H4, W4) soft masks at
    stride 4 (coverage in [0, 1]). Returns z_int (N, D) pooled inside each object,
    z_ctx (N, D) pooled over a ring around it (object itself excluded), and the
    image index (N,) of each object.
    """
    B, D, h, w = desc_map.shape
    zi, zc, bi = [], [], []
    for b, m in enumerate(inst):
        if m is None or m.shape[0] == 0:
            continue
        m8 = F.adaptive_avg_pool2d(m[:, None].to(desc_map.dtype), (h, w))[:, 0]          # (n, h, w)
        occ = (m8 > 0).to(desc_map.dtype)
        dil = F.max_pool2d(occ[:, None], 2 * ring_cells + 1, 1, ring_cells)[:, 0]
        ring = (dil - occ).clamp_min(0)
        f = desc_map[b].flatten(1)                                                        # (D, hw)
        zi.append((m8.flatten(1) @ f.T) / m8.flatten(1).sum(1, keepdim=True).clamp_min(1e-6))
        zc.append((ring.flatten(1) @ f.T) / ring.flatten(1).sum(1, keepdim=True).clamp_min(1e-6))
        bi.append(torch.full((m.shape[0],), b, dtype=torch.long, device=desc_map.device))
    if not zi:
        z = desc_map.new_zeros(0, D)
        return z, z, torch.zeros(0, dtype=torch.long, device=desc_map.device)
    return torch.cat(zi), torch.cat(zc), torch.cat(bi)


class TrashScribeEdge(nn.Module):
    """Export graph for TensorRT / ONNX Runtime: one image in, three maps out.

    waste_prob (B,1,S/4,S/4), contour_prob (B,1,S/4,S/4), desc (B,D,S/8,S/8).
    Phrase scoring, pooling and naming run after the engine on small tensors
    (see trashscribe/edge_runtime.py), against a phrase matrix precomputed offline.
    """

    def __init__(self, model: TrashScribe):
        super().__init__()
        self.m = model

    def forward(self, x):
        o = self.m(x)
        return torch.sigmoid(o["waste"]), torch.sigmoid(o["contour"]), o["desc"]
