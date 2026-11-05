"""From a description to a name, by reading the atlas.

Three evidence streams, each failing differently (as in the MIRF pipeline):
  dap    descriptor agreement: the descriptor posterior against each entry's
         profile, per intrinsic field, IDF-weighted (Lampert et al., 2014)
  proto  text-space prototype: the object embedding against the weighted mean of
         the re-projected phrases that the entry names
  prose  the object embedding against the entry's free-text description
Contextual fields have weight 0 here. Streams are standardised per object across
entries and combined linearly; an object is named only if the top entry clears
a confidence AND a margin threshold, otherwise it is set aside.

Everything the namer needs is a handful of tensors (`AtlasCache`), computed once
from the trained model and saved, so the edge device needs neither the text
encoder nor the training code to name objects.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .schema import Atlas, Schema, entry_target


@dataclass
class AtlasCache:
    phrases: torch.Tensor          # (V, D) re-projected, unit norm
    field_idx: torch.Tensor        # (V,)
    field_kind: list               # per field: "single" | "multi"
    field_scope: list              # per field: "intrinsic" | "contextual"
    field_weight: torch.Tensor     # (F,)
    field_names: list
    value_names: list              # (V,) "field: value"
    log_tau: torch.Tensor          # (F,)
    bias: torch.Tensor             # (F,)
    profile: torch.Tensor          # (E, V) entry profiles (soft)
    profile_mask: torch.Tensor     # (E, V) bool: field described by the entry
    idf: torch.Tensor              # (V,)
    proto: torch.Tensor            # (E, D)
    prose: torch.Tensor            # (E, D)
    entry_names: list
    entry_streams: list

    def save(self, path):
        torch.save(asdict(self), path)

    @staticmethod
    def load(path) -> "AtlasCache":
        return AtlasCache(**torch.load(path, map_location="cpu", weights_only=False))


@torch.no_grad()
def build_cache(model, schema: Schema, atlas: Atlas, text_bank) -> AtlasCache:
    """`schema` may be an EXTENDED schema (new values from the atlas): new phrases
    get the temperature and bias of their field, no new parameter."""
    E = text_bank.encode_ensembles(schema.phrases())
    fidx = schema.field_index()
    P = F.normalize(model.text_proj(E.to(model.text_proj.A.weight.device)).float().cpu(), dim=-1)
    prof, pm = zip(*[entry_target(e, schema) for e in atlas.entries])
    prof = torch.from_numpy(np.stack(prof))
    pm = torch.from_numpy(np.stack(pm))
    n_e = len(atlas.entries)
    df = (prof > 0).float().sum(0)
    idf = torch.log((1 + n_e) / (1 + df)) + 1.0
    fw = torch.tensor([f.weight if f.scope == "intrinsic" else 0.0 for f in schema.fields])
    w_v = fw[fidx] * idf
    proto = F.normalize((prof * w_v) @ P + 1e-8, dim=-1)
    prose_txt = [e.prose or e.name for e in atlas.entries]
    prose_raw = text_bank.encode(prose_txt).to(model.vocab.device)
    prose = F.normalize(model.text_proj(prose_raw).float().cpu(), dim=-1)
    return AtlasCache(phrases=P, field_idx=fidx, field_kind=[f.kind for f in schema.fields],
                      field_scope=[f.scope for f in schema.fields], field_weight=fw,
                      field_names=[f.name for f in schema.fields],
                      value_names=[f"{a}: {b}" for a, b in schema.flat],
                      log_tau=model.scorer.log_tau.detach().float().cpu(),
                      bias=model.scorer.bias.detach().float().cpu(), profile=prof, profile_mask=pm, idf=idf,
                      proto=proto, prose=prose, entry_names=[e.name for e in atlas.entries],
                      entry_streams=[e.stream for e in atlas.entries])


class Namer:
    def __init__(self, cache: AtlasCache, weights=None, conf_thr=0.0, margin_thr=0.0, temperature=1.0):
        self.c = cache
        self.w = {"dap": 1.0, "proto": 1.0, "prose": 0.5, **(weights or {})}
        self.conf_thr, self.margin_thr, self.T = conf_thr, margin_thr, temperature
        self.contextual = torch.tensor([cache.field_scope[i] == "contextual" for i in cache.field_idx.tolist()])
        # calibrated stacking (Chao et al., 2016): gamma subtracted from the fused
        # score of every class seen in training; set by calibrate_seen_penalty()
        self.seen_penalty = 0.0
        self.seen_mask = torch.zeros(len(cache.entry_names), dtype=torch.bool)

    def set_seen(self, seen: set, penalty: float):
        self.seen_mask = torch.tensor([n in seen for n in self.c.entry_names])
        self.seen_penalty = float(penalty)

    # ---------------------------------------------------------------- scoring
    def logits(self, z_int: torch.Tensor, z_ctx: torch.Tensor | None = None) -> torch.Tensor:
        c = self.c
        tau, b = c.log_tau.exp()[c.field_idx], c.bias[c.field_idx]
        lg = F.normalize(z_int.float(), dim=-1) @ c.phrases.T * tau + b
        if z_ctx is not None and bool(self.contextual.any()):
            lc = F.normalize(z_ctx.float(), dim=-1) @ c.phrases.T * tau + b
            lg = torch.where(self.contextual[None], lc, lg)
        return lg

    def posterior(self, logits: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(logits)
        for fi, kind in enumerate(self.c.field_kind):
            sel = self.c.field_idx == fi
            out[:, sel] = logits[:, sel].softmax(-1) if kind == "single" else logits[:, sel].sigmoid()
        return out

    def evidence(self, z_int, z_ctx=None) -> dict:
        c = self.c
        post = self.posterior(self.logits(z_int, z_ctx))
        num = torch.zeros(post.shape[0], len(c.entry_names))
        den = torch.zeros(len(c.entry_names))
        for fi, scope in enumerate(c.field_scope):
            if scope != "intrinsic" or c.field_weight[fi] <= 0:
                continue
            sel = c.field_idx == fi
            a = F.normalize(post[:, sel] * c.idf[sel], dim=-1)
            p = F.normalize(c.profile[:, sel] * c.idf[sel], dim=-1)
            described = c.profile_mask[:, sel].all(1).float()
            num += c.field_weight[fi] * (a @ p.T) * described
            den += c.field_weight[fi] * described
        dap = num / den.clamp_min(1e-6)
        zn = F.normalize(z_int.float(), dim=-1)
        return {"dap": dap, "proto": zn @ c.proto.T, "prose": zn @ c.prose.T, "posterior": post}

    @staticmethod
    def _std(x):
        return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)

    def fuse(self, ev: dict) -> torch.Tensor:
        s = sum(self.w[k] * self._std(ev[k]) for k in ("dap", "proto", "prose") if self.w.get(k, 0) > 0)
        s = s / max(sum(v for k, v in self.w.items() if k in ("dap", "proto", "prose")), 1e-6)
        return s - self.seen_penalty * self.seen_mask.float()[None]

    def name(self, z_int, z_ctx=None, restrict: list[str] | None = None) -> dict:
        ev = self.evidence(z_int, z_ctx)
        fused = self.fuse(ev)
        if restrict is not None:
            keep = torch.tensor([n in restrict for n in self.c.entry_names])
            fused = fused.masked_fill(~keep[None], float("-inf"))
        prob = (fused / self.T).softmax(-1)
        top2 = prob.topk(min(2, prob.shape[1]), dim=-1)
        p1 = top2.values[:, 0]
        p2 = top2.values[:, 1] if prob.shape[1] > 1 else torch.zeros_like(p1)
        abstain = (p1 < self.conf_thr) | ((p1 - p2) < self.margin_thr)
        names = [self.c.entry_names[i] for i in top2.indices[:, 0].tolist()]
        return {"name": names, "prob": p1, "abstain": abstain, "fused": fused, **ev}

    def describe(self, post_row: torch.Tensor, thr=0.5) -> dict:
        """Readable description: argmax of single fields, values above thr of multi ones."""
        c, out = self.c, {}
        for fi, (fname, kind) in enumerate(zip(c.field_names, c.field_kind)):
            idx = torch.nonzero(c.field_idx == fi).flatten()
            p = post_row[idx]
            if kind == "single":
                j = int(p.argmax())
                # a single-valued field is stated only when one value dominates;
                # otherwise the report says nothing rather than guess
                out[fname] = [(c.value_names[idx[j]].split(": ", 1)[1], float(p[j]))] if p[j] >= thr else []
            else:
                out[fname] = [(c.value_names[idx[j]].split(": ", 1)[1], float(p[j]))
                              for j in torch.nonzero(p > thr).flatten().tolist()]
        return out


def calibrate_seen_penalty(namer: Namer, z_int, z_ctx, true: list, seen: set, keep=0.9, grid=None) -> dict:
    """Largest gamma keeping >= `keep` of the seen-class accuracy obtained at gamma=0,
    measured on VALIDATION objects (seen classes only, held-out classes never
    involved). Larger gamma leaves more room for classes the model never saw."""
    namer.set_seen(seen, 0.0)
    idx = [i for i, t in enumerate(true) if t in seen]
    if not idx:
        return {"gamma": 0.0}
    zi, zc = z_int[idx], z_ctx[idx]
    tt = [true[i] for i in idx]

    def acc():
        pred = namer.name(zi, zc)["name"]
        per = [np.mean([p == c for p, t in zip(pred, tt) if t == c]) for c in sorted(set(tt))]
        return float(np.mean(per))
    base = acc()
    best = 0.0
    for g in (grid if grid is not None else np.linspace(0, 3, 31)):
        namer.set_seen(seen, float(g))
        if acc() >= keep * base:
            best = float(g)
    namer.set_seen(seen, best)
    return {"gamma": best, "val_seen_acc_at_0": base, "keep": keep}
