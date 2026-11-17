"""Training and evaluation loop shared by every experiment.

A run lives in runs/<run_name>/ with: config.yaml (resolved), history.json (one
row per epoch), best.pt / last.pt (trainable weights only: the frozen SAM2 and
text encoder are reloaded from their hub ids), val/test metrics JSON, and an
atlas cache for the edge runtime.
"""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

from . import device as D
from .data.canonical import Canonicalizer, collate, to_device
from .data.mixing import MultiDataset
from .data.registry import build_split, trained_entries
from .data.synthetic import synthetic_atlas
from .losses import contour_loss, dann_lambda, descriptor_loss, field_layout, mask_loss, sample_object_pixels
from .metrics import SegMeter, ausuc, descriptor_map, environment_leakage, gzsl, retrieval
from .models.trashscribe import ModelConfig, TrashScribe, pool_objects
from .progress import Progress
from .naming import Namer, build_cache
from .schema import Atlas, Schema
from .text_bank import TextBank

PKG_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ config
def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _depth_view(d: dict, geometry: bool) -> dict:
    """geometry_depth_dir -> depth_dir for geometry runs; dropped otherwise (no depth I/O)."""
    d = copy.deepcopy(d)
    for blk in [d] + list((d.get("splits") or {}).values()):
        g = blk.pop("geometry_depth_dir", None)
        if geometry and g:
            blk["depth_dir"] = g
    return d


def load_config(path, overrides: list[str] | None = None, _top: bool = True) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    # a config that names its own datasets_file replaces the registry it inherits
    own_registry = bool(cfg.get("datasets_file")) and "datasets" not in cfg
    base = cfg.pop("_base_", None)
    if base:
        cfg = _merge(load_config(path.parent / base, _top=False), cfg)
    if cfg.get("datasets_file") and ("datasets" not in cfg or own_registry):
        cfg["datasets"] = yaml.safe_load((PKG_ROOT / cfg["datasets_file"]).read_text())["datasets"]
    for ov in overrides or []:
        k, v = ov.split("=", 1)
        node = cfg
        keys = k.split(".")
        for kk in keys[:-1]:
            node = node.setdefault(kk, {})
        node[keys[-1]] = yaml.safe_load(v)
    if _top and cfg.get("extra_datasets"):
        # datasets added by ONE config on top of the registry (15 Nov: waste-free negatives for
        # E3b); never in the registry itself, where train_datasets: null would pick them up
        have = {d["name"] for d in cfg.get("datasets") or []}
        cfg["datasets"] = list(cfg.get("datasets") or []) + [d for d in cfg["extra_datasets"] if d["name"] not in have]
    if _top and cfg.get("datasets"):
        geo = bool((cfg.get("model") or {}).get("geometry"))
        cfg["datasets"] = [_depth_view(d, geo) for d in cfg["datasets"]]
    return cfg


def _p(p):
    p = Path(p)
    return p if p.is_absolute() else PKG_ROOT / p


def model_config(cfg: dict) -> ModelConfig:
    names = {f.name for f in fields(ModelConfig)}
    return ModelConfig(**{k: v for k, v in (cfg.get("model") or {}).items() if k in names})


def load_vocab(cfg: dict):
    schema = Schema.load(_p(cfg["schema"]))
    atlases = [Atlas.load(_p(a)) for a in cfg.get("atlases", [])]
    if any(d.get("type") == "synthetic" for d in cfg.get("datasets", [])):
        atlases.append(synthetic_atlas())
    atlas = Atlas.merge(atlases) if atlases else Atlas("empty", [])
    return schema, atlas


def build_model(cfg: dict, schema: Schema, text_bank: TextBank, device) -> TrashScribe:
    E = text_bank.encode_ensembles(schema.phrases())
    env_vals = schema.field("surroundings").values if "surroundings" in schema.slices else []
    model = TrashScribe(model_config(cfg), text_dim=E.shape[1], n_fields=len(schema.fields), n_env=len(env_vals))
    model.set_vocabulary(E, schema.field_index(), torch.tensor([f.scope == "contextual" for f in schema.fields]))
    return model.to(device)


def trainable_state(model: TrashScribe) -> dict:
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    sd = model.state_dict()
    out = {k: v for k, v in sd.items() if k in keep}
    for n, b in model.named_buffers():
        if not n.startswith("backbone.model.") and n in sd:
            out[n] = b
    return out


def load_run(run_dir, device="auto", which="best.pt"):
    """(model, cfg, schema, atlas, text_bank) from a run directory."""
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    dev = D.pick_device(device)
    schema, atlas = load_vocab(cfg)
    tb = TextBank(cfg["text_model"], dev, cache_path=_p(cfg.get("text_cache", "runs/_cache/text.pt")))
    model = build_model(cfg, schema, tb, dev)
    sd = torch.load(run_dir / which, map_location=dev, weights_only=False)["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in sd if k in unexpected]
    if bad:
        raise RuntimeError(f"checkpoint keys not in model: {bad[:5]}...")
    model.eval()
    return model, cfg, schema, atlas, tb


def posterior(logits: torch.Tensor, schema: Schema) -> torch.Tensor:
    out = torch.empty_like(logits)
    for f in schema.fields:
        sl = schema.slices[f.name]
        out[:, sl] = logits[:, sl].softmax(-1) if f.kind == "single" else logits[:, sl].sigmoid()
    return out


# ------------------------------------------------------------------ trainer
class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.tc = cfg["train"]
        self.device = D.pick_device(cfg.get("device", "auto"))
        D.setup(self.device, cfg.get("seed", 42))
        self.run_dir = _p(cfg.get("output_dir", "runs")) / cfg["run_name"]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        self.progress = Progress(self.run_dir)
        self.progress(phase="setup", epochs=int(cfg["train"]["epochs"]), patience=int(cfg["train"].get("patience", 10)))
        self.schema, self.atlas = load_vocab(cfg)
        self.layout = field_layout(self.schema)
        self.text_bank = TextBank(cfg["text_model"], self.device, cache_path=_p(cfg.get("text_cache", "runs/_cache/text.pt")))
        self.model = build_model(cfg, self.schema, self.text_bank, self.device)
        self.text_bank.save()
        self.size = self.model.cfg.image_size
        root = Path(cfg.get("data_root", ".")).expanduser()
        holdout = set(cfg.get("holdout") or [])
        mx = cfg.get("max_items")
        ds = cfg.get("datasets", [])
        self.train_adapters = build_split(ds, cfg.get("train_datasets"), "train", self.atlas, root, holdout, mx,
                                          strict=cfg.get("strict_atlas", True))
        self.val_adapters = build_split(ds, cfg.get("val_datasets") or cfg.get("train_datasets"), "val", self.atlas,
                                        root, holdout, mx, strict=cfg.get("strict_atlas", True))
        # E10: only n labelled images of a dataset (a new site) reach training (seeded subset)
        for a in self.train_adapters:
            sub = (self.tc.get("subsample") or {}).get(a.name)
            if sub:
                from .data.registry import subsample_adapter
                keep = subsample_adapter(a, int(sub["n"]), int(sub.get("seed", 0)))
                print(f"[{a.name}/train] subsampled to {len(keep)} images (seed {sub.get('seed', 0)})", flush=True)
                (self.run_dir / f"subsample_{a.name}.json").write_text(json.dumps(
                    {"n": len(keep), "seed": int(sub.get("seed", 0)), "indices": keep,
                     "items": [str(x) for x in a.items]}, indent=1))
        self.seen = trained_entries(self.train_adapters)
        self.holdout = holdout
        # E7: descriptor values never supervised ("field: value"), scored at test from text only
        names = [f"{f.name}: {v}" for f in self.schema.fields for v in f.values]
        unsup = list(self.tc.get("loss", {}).get("unsupervised_values") or [])
        bad = [v for v in unsup if v not in names]
        if bad:
            raise KeyError(f"unsupervised_values not in the schema: {bad}")
        self.drop = torch.tensor([n in unsup for n in names], dtype=torch.bool, device=self.device) if unsup else None
        # E8: frozen CLIP image teacher for the distilled CLIP field
        self.teacher = None
        if self.model.clip_field is not None and float(self.tc["loss"].get("clip", 0)) > 0:
            from .clip_teacher import ClipTeacher
            self.teacher = ClipTeacher(self.device, **(self.tc.get("clip_teacher") or {}))
        self.info = {"device": D.describe(self.device),
                     "params_total": sum(p.numel() for p in self.model.parameters()),
                     "params_trainable": sum(p.numel() for p in self.model.parameters() if p.requires_grad)}
        print(json.dumps(self.info, indent=1))

    def loaders(self):
        tc = self.tc
        canon_tr = Canonicalizer(self.schema, self.atlas, self.size, True, tc.get("aug"))
        canon_va = Canonicalizer(self.schema, self.atlas, self.size, False)
        tr = MultiDataset(self.train_adapters, canon_tr, tc.get("transplant_p", 0.0), tc.get("transplant"),
                          cross_dataset_only=tc.get("transplant_cross_only", True))
        va = MultiDataset(self.val_adapters, canon_va)
        sampler = WeightedRandomSampler(tr.sample_weights(tc.get("dataset_alpha", 0.5), tc.get("dataset_share")),
                                        num_samples=int(tc.get("epoch_size") or len(tr)), replacement=True)
        nw = int(tc.get("num_workers", 4))
        # pinned batches go through CUDA's caching host allocator, which keeps every block:
        # ~1.4 GB of host RAM on E4 (batch sizes vary with the object count). Off on small hosts.
        pin = self.device.type == "cuda" and bool(tc.get("pin_memory", True))
        kw = dict(collate_fn=collate, num_workers=nw, pin_memory=pin, persistent_workers=nw > 0)
        # validation workers are not kept alive between epochs: idle persistent
        # workers of both loaders exhaust host RAM on small machines
        nwv = int(tc.get("val_num_workers", nw))
        kwv = dict(collate_fn=collate, num_workers=nwv, pin_memory=pin)
        if nwv > 0:   # batches waiting in shared memory per worker (RAM)
            kwv["prefetch_factor"] = int(tc.get("val_prefetch", 2))
        return (DataLoader(tr, batch_size=tc["batch_size"], sampler=sampler, drop_last=True, **kw),
                DataLoader(va, batch_size=tc.get("val_batch_size", tc["batch_size"]), shuffle=False, **kwv))

    def step_loss(self, batch, progress: float) -> tuple[torch.Tensor, dict]:
        lc, m = self.tc["loss"], self.model
        with D.autocast(self.device, self.tc.get("amp", True)):
            out = m(batch["image"], batch["depth"])
        parts = {"mask": mask_loss(out["waste"], batch["waste"], batch["valid"], lc.get("pos_weight", 2.0)),
                 "contour": contour_loss(out["contour"], batch["contour"], batch["contour_valid"])}
        desc = out["desc"].float()
        inst = [t.float() for t in batch["inst"]]
        zi, zc, bidx = pool_objects(desc, inst, m.cfg.ring_cells)
        dt = torch.cat(batch["desc_t"]) if zi.shape[0] else desc.new_zeros(0, self.schema.n_values)
        dm = torch.cat(batch["desc_m"]) if zi.shape[0] else torch.zeros(0, self.schema.n_values, dtype=torch.bool,
                                                                         device=desc.device)
        has = dm.any(1)
        parts["desc"] = (descriptor_loss(m.score(zi[has], zc[has]), dt[has], dm[has], self.layout, drop=self.drop)
                         if bool(has.any()) else desc.sum() * 0)
        if lc.get("dense", 0) > 0 and bool(has.any()):
            f, owner = sample_object_pixels(desc, inst, lc.get("dense_pixels", 16))
            keep = has[owner]
            parts["dense"] = (descriptor_loss(m.score(f[keep]), dt[owner[keep]], dm[owner[keep]], self.layout,
                                              scopes=("intrinsic",), drop=self.drop)
                              if bool(keep.any()) else desc.sum() * 0)
        if m.adversary is not None and lc.get("adv", 0) > 0 and zi.shape[0]:
            env = batch["env_id"][bidx]
            k = env >= 0
            if bool(k.any()):
                lam = dann_lambda(progress) * float(lc.get("adv_grl", 1.0))
                parts["adv"] = F.cross_entropy(m.adversary(zi[k], lam), env[k])
        if self.teacher is not None and zi.shape[0]:
            from .clip_teacher import distill_loss
            sel, tgt = self.teacher(batch["image"], inst)
            if tgt is not None:
                zclip = pool_objects(out["clip"].float(), inst, m.cfg.ring_cells)[0]
                parts["clip"] = distill_loss(zclip[sel], tgt)
        w = {"mask": lc.get("mask", 1.0), "contour": lc.get("contour", 0.5), "desc": lc.get("desc", 1.0),
             "dense": lc.get("dense", 0.0), "adv": lc.get("adv", 0.0), "clip": lc.get("clip", 0.0)}
        total = sum(w[k] * v for k, v in parts.items())
        return total, {k: float(v.detach()) for k, v in parts.items()}

    def train(self, resume: bool = False):
        tc = self.tc
        dl_tr, dl_va = self.loaders()
        opt = torch.optim.AdamW(self.model.param_groups(float(tc["lr_heads"]), float(tc["lr_peft"]),
                                                        float(tc["lr_decoder"]), float(tc["weight_decay"])))
        # accum_steps > 1: batch_size is the micro-batch, the optimizer sees
        # batch_size * accum_steps (lets batch 8 run on an 8 GB GPU as 4 x 2)
        accum = max(1, int(tc.get("accum_steps", 1)))
        total_steps = tc["epochs"] * (len(dl_tr) // accum)
        warm = int(tc.get("warmup_steps", 200))

        def lr_lambda(s):
            if s < warm:
                return (s + 1) / warm
            q = (s - warm) / max(1, total_steps - warm)
            return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * q))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        scaler = D.grad_scaler(self.device, tc.get("amp", True))
        hist, best, bad, step, start = [], -1e9, 0, 0, 1
        init = tc.get("init_from")
        if init and not (resume and (self.run_dir / "last.pt").exists()):
            # E9 (15 Nov): adapt an existing model to a new site (fine-tuning, fresh optimiser and
            # schedule). Ignored when resuming: last.pt already holds the fine-tuned weights.
            ck = torch.load(_p(init), map_location=self.device, weights_only=False)
            missing, unexpected = self.model.load_state_dict(ck["model"], strict=False)
            bad_keys = [k for k in ck["model"] if k in unexpected]
            if bad_keys:
                raise RuntimeError(f"init_from: keys not in model: {bad_keys[:5]}")
            print(f"initialised from {init} (epoch {ck.get('epoch')})", flush=True)
        if resume and (self.run_dir / "last.pt").exists():
            ck = torch.load(self.run_dir / "last.pt", map_location=self.device, weights_only=False)
            if "opt" in ck:
                self.model.load_state_dict(ck["model"], strict=False)
                opt.load_state_dict(ck["opt"])
                sched.load_state_dict(ck["sched"])
                scaler.load_state_dict(ck["scaler"])
                hist, best, bad, step, start = ck["hist"], ck["best"], ck["bad"], ck["step"], ck["epoch"] + 1
                print(f"resumed after epoch {ck['epoch']} (best {best:.4f}, bad {bad})")
        for ep in range(start, tc["epochs"] + 1):
            if bad >= tc.get("patience", 10):
                break
            self.model.train()
            t0, agg, n = time.time(), {}, 0
            self.progress(phase="train", epoch=ep, step=0, steps=len(dl_tr) // accum, epoch_started=t0)
            opt.zero_grad(set_to_none=True)
            for mi, batch in enumerate(dl_tr):
                batch = to_device(batch, self.device)
                loss, parts = self.step_loss(batch, step / max(1, total_steps))
                scaler.scale(loss / accum).backward()
                if (mi + 1) % accum:
                    for k, v in parts.items():
                        agg[k] = agg.get(k, 0.0) + v / accum
                    agg["total"] = agg.get("total", 0.0) + float(loss.detach()) / accum
                    continue
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(), tc.get("grad_clip", 1.0))
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                n += 1
                self.progress(step=n)
                if accum > 1:
                    parts = {k: v / accum for k, v in parts.items()}
                    loss = loss / accum
                for k, v in parts.items():
                    agg[k] = agg.get(k, 0.0) + v
                agg["total"] = agg.get("total", 0.0) + float(loss.detach())
                if tc.get("max_steps_per_epoch") and n >= tc["max_steps_per_epoch"]:
                    break
            row = {"epoch": ep, "time_s": round(time.time() - t0, 1), **{f"train_{k}": v / max(n, 1)
                                                                        for k, v in agg.items()}}
            if ep % tc.get("val_every", 1) == 0:
                tv = time.time()
                self.progress(phase="val", batch=0, batches=len(dl_va))
                vm = self.evaluate(dl_va, naming=False)
                row["val_s"] = round(time.time() - tv, 1)
                row.update(val_iou=vm["seg"]["_all"]["iou"], val_bf1=vm["seg"]["_all"]["bf1"],
                           val_desc_map=vm["desc"].get("_macro_intrinsic", {}).get("map", float("nan")),
                           val_desc_floor=vm["desc"].get("_macro_intrinsic", {}).get("floor", float("nan")))
                score = row["val_iou"] + (0 if math.isnan(row["val_desc_map"]) else row["val_desc_map"])
                if tc.get("select") == "last":
                    # E10: fixed short schedule, the last epoch is tested (validation holds no image of
                    # the new site, so it cannot choose an epoch for it); never early-stops
                    score = float(ep)
                row["val_score"] = score
                if score > best:
                    best, bad = score, 0
                    self.save("best.pt", ep, row)
                    (self.run_dir / "val_metrics.json").write_text(json.dumps(vm, indent=1, default=float))
                else:
                    bad += 1
            hist.append(row)
            self.progress(force=True, last_epoch=ep, best=best, bad=bad, epoch_s=round(time.time() - t0, 1))
            (self.run_dir / "history.json").write_text(json.dumps(hist, indent=1))
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
            self.save("last.pt", ep, row, opt=opt.state_dict(), sched=sched.state_dict(),
                      scaler=scaler.state_dict(), hist=hist, best=best, bad=bad, step=step)
            if bad >= tc.get("patience", 10):
                print(f"early stop at epoch {ep}")
                break
        self.progress(phase="train_done", epochs_run=len(hist))
        summary = {"run": self.cfg["run_name"], "best_val_score": best, "epochs_run": len(hist), **self.info}
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        return hist

    def save(self, name, epoch, row, **resume_state):
        """resume_state (optimizer, scheduler, history...) only goes into last.pt."""
        tmp = self.run_dir / (name + ".tmp")
        torch.save({"model": trainable_state(self.model), "epoch": epoch, "row": row,
                    "config": self.cfg, **resume_state}, tmp)
        tmp.replace(self.run_dir / name)          # atomic: a crash never leaves a truncated checkpoint

    # ------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self, loader, naming=True, namer: Namer | None = None, retrieval_entries=None,
                 n_perm=2000) -> dict:
        return evaluate(self.model, loader, self.schema, self.device, namer=namer if naming else None,
                        seen=self.seen, unseen=self.holdout, retrieval_entries=retrieval_entries, n_perm=n_perm,
                        amp=self.tc.get("amp", True), progress=self.progress,
                        skip_values=None if self.drop is None else self.drop.cpu().numpy())


@torch.no_grad()
def evaluate(model, loader, schema, device, namer=None, seen=frozenset(), unseen=frozenset(),
             retrieval_entries=None, n_perm=2000, amp=True, return_raw=False, progress=None,
             skip_values=None) -> dict:
    model.eval()
    seg = SegMeter()
    P, T, M, Z, ZC, ENV, ENT = [], [], [], [], [], [], []
    if progress is not None:
        progress(batch=0, batches=len(loader))
    for bi, batch in enumerate(loader):
        if progress is not None:
            progress(batch=bi + 1)
        batch = to_device(batch, device)
        with D.autocast(device, amp):
            out = model(batch["image"], batch["depth"])
        prob = torch.sigmoid(F.interpolate(out["waste"].float(), size=batch["waste"].shape[-2:], mode="bilinear",
                                           align_corners=False))
        seg.update(prob, batch["waste"], batch["valid"], batch["dataset"])
        inst = [t.float() for t in batch["inst"]]
        zi, zc, bidx = pool_objects(out["desc"].float(), inst, model.cfg.ring_cells)
        if zi.shape[0] == 0:
            continue
        P.append(posterior(model.score(zi, zc), schema).cpu())
        T.append(torch.cat(batch["desc_t"]).cpu())
        M.append(torch.cat(batch["desc_m"]).cpu())
        Z.append(zi.cpu())
        ZC.append(zc.cpu())
        ENV.append(batch["env_id"][bidx].cpu())
        ENT += [e for ents in batch["entries"] for e in ents]
    res = {"seg": seg.result()}
    if not P:
        res["desc"] = {}
        return res
    P, T, M = torch.cat(P).numpy(), torch.cat(T).numpy(), torch.cat(M).numpy()
    Z, ZC, ENV = torch.cat(Z), torch.cat(ZC), torch.cat(ENV).numpy()
    if return_raw:
        res["_raw"] = {"z": Z, "zc": ZC, "entries": ENT, "env": ENV}
    res["desc"] = descriptor_map(P, T, M, schema, skip_values)
    res["n_objects"] = int(len(ENT))
    if namer is not None:
        named = namer.name(Z, ZC)
        idx = [i for i, e in enumerate(ENT) if e is not None and e in namer.c.entry_names
               and namer.c.profile_mask[namer.c.entry_names.index(e)].any()]
        pred = [named["name"][i] for i in idx]
        true = [ENT[i] for i in idx]
        res["naming"] = gzsl(pred, true, set(seen) & set(true), set(unseen) & set(true),
                             abstained=named["abstain"][idx].numpy() if idx else None)
        res["naming_ausuc"] = ausuc(named["fused"].numpy()[idx], true, namer.c.entry_names,
                                    set(seen) & set(true), set(unseen) & set(true), penalized=set(seen))
        for stream in ("dap", "proto", "prose"):            # each evidence stream alone
            only = named[stream].numpy()
            pred_s = [namer.c.entry_names[j] for j in only[idx].argmax(1)] if idx else []
            res[f"naming_{stream}"] = gzsl(pred_s, true, set(seen) & set(true), set(unseen) & set(true))
        if retrieval_entries:
            if progress is not None:
                progress(phase=progress.state.get("phase", "eval") + ": retrieval", entries=len(retrieval_entries))
            fused = named["fused"].numpy()
            ent = np.array([e if e is not None else "" for e in ENT])
            res["retrieval"] = {}
            for e in retrieval_entries:
                if e in namer.c.entry_names and (ent == e).any():
                    col = fused[:, namer.c.entry_names.index(e)]
                    res["retrieval"][e] = retrieval(col, (ent == e).astype(int), n_perm=n_perm, n_boot=n_perm)
    res["leakage_intrinsic"] = environment_leakage(Z.numpy(), ENV)
    res["leakage_context"] = environment_leakage(ZC.numpy(), ENV)
    return res
