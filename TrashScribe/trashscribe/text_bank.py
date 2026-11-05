"""Frozen text encoder and on-disk phrase cache.

The text encoder is only needed when the vocabulary changes. Embeddings are
cached by (model id, phrase), so the edge device ships the cache and never runs
the encoder unless a user types a free-text query.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


class TextBank:
    def __init__(self, model_id: str = "openai/clip-vit-base-patch16", device="cpu",
                 cache_path: str | Path | None = None):
        self.model_id = model_id
        self.device = torch.device(device)
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, torch.Tensor] = {}
        self._model = self._tok = None
        if self.cache_path and self.cache_path.exists():
            blob = torch.load(self.cache_path, map_location="cpu", weights_only=True)
            if blob.get("model_id") == model_id:
                self._cache = dict(zip(blob["phrases"], blob["emb"]))

    # ---------------------------------------------------------------- encoder
    def _load(self):
        if self._model is not None:
            return
        if self.model_id.startswith("hash:"):
            return
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer
        self._tok = CLIPTokenizer.from_pretrained(self.model_id)
        self._model = CLIPTextModelWithProjection.from_pretrained(self.model_id).eval().to(self.device)
        for p in self._model.parameters():
            p.requires_grad = False

    @property
    def dim(self) -> int:
        if self.model_id.startswith("hash:"):
            return int(self.model_id.split(":")[1])
        if self._cache:
            return next(iter(self._cache.values())).numel()
        self._load()
        return self._model.config.projection_dim

    @torch.no_grad()
    def _encode_raw(self, phrases: list[str]) -> torch.Tensor:
        if self.model_id.startswith("hash:"):
            # TEST-ONLY encoder: deterministic pseudo-random unit vectors. No
            # semantics; used so unit tests run without downloading weights.
            d = self.dim
            out = []
            for p in phrases:
                seed = int(hashlib.sha1(p.encode()).hexdigest()[:8], 16)
                g = torch.Generator().manual_seed(seed)
                out.append(torch.randn(d, generator=g))
            return F.normalize(torch.stack(out), dim=-1)
        self._load()
        embs = []
        for i in range(0, len(phrases), 256):
            tok = self._tok(phrases[i:i + 256], padding=True, truncation=True, return_tensors="pt").to(self.device)
            e = self._model(**tok).text_embeds
            embs.append(F.normalize(e.float(), dim=-1).cpu())
        return torch.cat(embs)

    def encode(self, phrases: list[str]) -> torch.Tensor:
        """(N, D) unit-norm embeddings, cached."""
        missing = [p for p in dict.fromkeys(phrases) if p not in self._cache]
        if missing:
            for p, e in zip(missing, self._encode_raw(missing)):
                self._cache[p] = e
        return torch.stack([self._cache[p] for p in phrases])

    def encode_ensembles(self, ensembles: list[list[str]]) -> torch.Tensor:
        """Prompt ensembling: mean of unit embeddings, renormalised. (N, D)."""
        flat = [p for ens in ensembles for p in ens]
        e = self.encode(flat)
        out, k = [], 0
        for ens in ensembles:
            out.append(F.normalize(e[k:k + len(ens)].mean(0), dim=-1))
            k += len(ens)
        return torch.stack(out)

    def save(self):
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        phrases = list(self._cache)
        torch.save({"model_id": self.model_id, "phrases": phrases,
                    "emb": torch.stack([self._cache[p] for p in phrases]) if phrases else torch.zeros(0)},
                   self.cache_path)

    def describe(self) -> str:
        return json.dumps({"model_id": self.model_id, "cached_phrases": len(self._cache)})
