"""Post-hoc analyses on trained checkpoints (no retraining).

    # 1. per-object embeddings of a run on one split (GPU with a hard memory cap, or CPU)
    python scripts/posthoc.py dump runs/e4_heldout_classes --split test --clip
    python scripts/posthoc.py dump runs/e4_heldout_classes --split val --datasets taco zerowaste bepli
    # 2. analyses on the dumps
    python scripts/posthoc.py naming  runs/e4_heldout_classes     # seen-class bias: calibration variants
    python scripts/posthoc.py clipret runs/e4_heldout_classes     # retrieval of unseen classes vs raw CLIP
    python scripts/posthoc.py leakage runs/e4_heldout_classes runs/e5_ablate_no_adversary ...

Dumps go to runs/<run>/posthoc/<split>.pt, results to runs/<run>/posthoc/<analysis>.json.
Objects are the ground-truth instances, pooled exactly as in scripts/evaluate.py.
"""
import _bootstrap  # noqa: F401
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trashscribe import device as D
from trashscribe.data.canonical import IMAGENET_MEAN, IMAGENET_STD, Canonicalizer, collate
from trashscribe.data.mixing import MultiDataset
from trashscribe.data.registry import build_split, trained_entries
from trashscribe.engine import _p, load_run
from trashscribe.metrics import ausuc, gzsl, retrieval
from trashscribe.models.trashscribe import pool_objects
from trashscribe.naming import AtlasCache, Namer


def _vec(o):
    return o if torch.is_tensor(o) else o.pooler_output


# ---------------------------------------------------------------------------- dump
@torch.no_grad()
def dump(a):
    run = Path(a.run)
    dev = torch.device(a.device)
    if dev.type == "cuda" and a.mem_frac:
        # this process fails before it can take memory the training next to it needs
        torch.cuda.set_per_process_memory_fraction(a.mem_frac, 0)
    model, cfg, schema, atlas, tb = load_run(run, device=dev)
    root = Path(cfg.get("data_root", ".")).expanduser()
    holdout = set(cfg.get("holdout") or [])
    if a.split == "test":
        names = cfg.get("test_datasets")
        hold = set()
    else:
        names = cfg.get("val_datasets") or cfg.get("train_datasets")
        hold = holdout
    if a.datasets:
        names = [n for n in (names or [d["name"] for d in cfg["datasets"]]) if n in a.datasets]
    ad = build_split(cfg["datasets"], names, a.split, atlas, root, hold, None, strict=cfg.get("strict_atlas", True))
    canon = Canonicalizer(schema, atlas, model.cfg.image_size, False)
    dl = DataLoader(MultiDataset(ad, canon), batch_size=1, collate_fn=collate, num_workers=a.workers)
    if a.clip:
        from transformers import CLIPModel, CLIPProcessor
        cp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        cm = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(dev).eval()
        if dev.type == "cuda":
            cm = cm.half()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    Z, ZC, ENT, ENV, DS, IMG, CL, N_EMPTY, ZD = [], [], [], [], [], [], [], [], []
    t0 = time.time()
    for i, b in enumerate(dl):
        x = b["image"].to(dev)
        with D.autocast(dev, dev.type == "cuda"):
            out = model(x, None)
        inst = [t.float().to(dev) for t in b["inst"]]
        zi, zc, bidx = pool_objects(out["desc"].float(), inst, model.cfg.ring_cells)
        if zi.shape[0] == 0:
            continue
        Z.append(zi.half().cpu())
        ZC.append(zc.half().cpu())
        if "clip" in out:   # E8: distilled CLIP field, pooled inside each object
            ZD.append(F.normalize(pool_objects(out["clip"].float(), inst, model.cfg.ring_cells)[0], dim=-1).half().cpu())
        ENT += [e for ents in b["entries"] for e in ents]
        ENV.append(b["env_id"][bidx.cpu()])
        DS += [b["dataset"][int(j)] for j in bidx.cpu()]
        IMG += [i] * zi.shape[0]
        if a.clip:  # raw CLIP on the 1.35x box crop of each object (same rule as baseline_zeroshot.py)
            img = ((b["image"][0] * std + mean).clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
            crops, empty = [], []
            for j, m in enumerate(b["inst"][0]):
                ys, xs = torch.nonzero(m > 0.5, as_tuple=True)
                if len(ys) == 0:          # object lost at stride 4: zero CLIP vector (counted)
                    ys, xs = torch.nonzero(m >= m.max(), as_tuple=True)
                    empty.append(j)
                y0, y1, x0, x1 = 4 * ys.min(), 4 * (ys.max() + 1), 4 * xs.min(), 4 * (xs.max() + 1)
                cy, cx, h, w = (y0 + y1) / 2, (x0 + x1) / 2, (y1 - y0) * 1.35, (x1 - x0) * 1.35
                crops.append(img[max(0, int(cy - h / 2)):int(cy + h / 2) + 1,
                                 max(0, int(cx - w / 2)):int(cx + w / 2) + 1])
            V = []
            for k in range(0, len(crops), 64):
                pix = cp(images=crops[k:k + 64], return_tensors="pt")["pixel_values"].to(dev, cm.dtype)
                V.append(F.normalize(_vec(cm.get_image_features(pixel_values=pix)).float(), dim=-1).half().cpu())
            V = torch.cat(V)
            V[empty] = 0
            N_EMPTY.append(len(empty))
            CL.append(V)
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"{i + 1}/{len(dl)} images, {sum(len(z) for z in Z)} objects, {el / 60:.1f} min, "
                  f"eta {(len(dl) - i - 1) * el / (i + 1) / 60:.1f} min", flush=True)
    env_names = list(schema.field("surroundings").values) if "surroundings" in schema.slices else []
    res = {"z": torch.cat(Z), "zc": torch.cat(ZC), "entries": ENT, "env": torch.cat(ENV), "env_names": env_names,
           "dataset": DS, "image": torch.tensor(IMG), "split": a.split, "run": str(run)}
    if ZD:
        res["zclip"] = torch.cat(ZD)
    if a.clip:
        res["clip"] = torch.cat(CL)
        res["clip_empty"] = int(sum(N_EMPTY))
        print("objects without a mask cell at stride 4 (zero CLIP vector):", res["clip_empty"])
    assert len(ENT) == res["z"].shape[0] == len(DS)
    outd = run / "posthoc"
    outd.mkdir(exist_ok=True)
    torch.save(res, outd / f"{a.split}.pt")
    print("wrote", outd / f"{a.split}.pt", res["z"].shape, f"{(time.time() - t0) / 60:.1f} min")


# ---------------------------------------------------------------------------- helpers
def _load(run, split):
    return torch.load(Path(run) / "posthoc" / f"{split}.pt", weights_only=False)


def _seen_unseen(run):
    """Trained entries (seen) and the scored unseen entries, as in scripts/evaluate.py."""
    import yaml
    from trashscribe.engine import load_vocab
    cfg = yaml.safe_load((Path(run) / "config.yaml").read_text())
    schema, atlas = load_vocab(cfg)
    root = Path(cfg.get("data_root", ".")).expanduser()
    holdout = set(cfg.get("holdout") or [])
    tr = build_split(cfg["datasets"], cfg.get("train_datasets"), "train", atlas, root, holdout, None,
                     strict=cfg.get("strict_atlas", True))
    seen = trained_entries(tr)
    summ = json.loads((Path(run) / "test_metrics.json").read_text())["summary"]
    return cfg, seen, set(summ.get("unseen_classes") or [])


def _namer(run, cfg, weights=None):
    nc = cfg.get("eval", {}).get("naming", {})
    return Namer(AtlasCache.load(Path(run) / "atlas_cache.pt"), weights or nc.get("weights"), 0.0, 0.0)


def _scored(namer, ents):
    c = namer.c
    return [i for i, e in enumerate(ents) if e is not None and e in c.entry_names
            and bool(c.profile_mask[c.entry_names.index(e)].any())]


def _std_rows(x):
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)


def _fused(namer, ev, w):
    s = sum(w[k] * _std_rows(ev[k]) for k in w if w[k] > 0)
    return s / sum(v for v in w.values() if v > 0)


def _gz(fused, names, true, seen, unseen):
    pred = [names[j] for j in fused.argmax(1)]
    return gzsl(pred, true, seen & set(true), unseen & set(true))


def _evidence(namer, d):
    ev = namer.evidence(d["z"].float(), d["zc"].float())
    return {k: ev[k].numpy() for k in ("dap", "proto", "prose")}


# ---------------------------------------------------------------------------- naming
class _Acc:
    """Vectorised per-class top-1 (same definition as metrics.gzsl / metrics.ausuc)."""

    def __init__(self, names, true, seen, unseen):
        self.t = np.array([names.index(x) for x in true])
        present = set(true)
        self.seen = np.array(sorted(names.index(c) for c in seen & present), int)
        self.unseen = np.array(sorted(names.index(c) for c in unseen & present), int)
        self.n = np.bincount(self.t, minlength=len(names)).astype(float)
        self.is_seen = np.array([n in seen for n in names])

    def __call__(self, F_):
        ok = (F_.argmax(1) == self.t).astype(float)
        per = np.bincount(self.t, weights=ok, minlength=len(self.n)) / np.maximum(self.n, 1)
        s = float(per[self.seen].mean()) if len(self.seen) else float("nan")
        u = float(per[self.unseen].mean()) if len(self.unseen) else float("nan")
        h = 2 * s * u / (s + u) if (s + u) > 0 and not np.isnan(s + u) else float("nan")
        return {"seen": s, "unseen": u, "H": h}

    def ausuc(self, F_, n=200):
        span = float(F_.max() - F_.min())
        pts = sorted({tuple(self(F_ - g * self.is_seen[None]).values())[:2] for g in np.linspace(-span, span, n)})
        s_, u_ = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        o = np.argsort(s_)
        h = [2 * a * b / (a + b) if a + b > 0 else 0.0 for a, b in pts]
        k = int(np.argmax(h))
        return {"ausuc": float(np.trapz(u_[o], s_[o])) if len(pts) > 1 else 0.0, "best_H_oracle": float(h[k])}


def naming(a):
    run = Path(a.run)
    cfg, seen, unseen = _seen_unseen(run)
    namer = _namer(run, cfg)
    names = namer.c.entry_names
    te, va = _load(run, "test"), _load(run, "val")
    it, iv = _scored(namer, te["entries"]), _scored(namer, va["entries"])
    acc_t = _Acc(names, [te["entries"][i] for i in it], seen, unseen)
    acc_v = _Acc(names, [va["entries"][i] for i in iv], seen, set())
    evt = {k: v[it] for k, v in _evidence(namer, te).items()}
    evv = {k: v[iv] for k, v in _evidence(namer, va).items()}
    is_seen = acc_t.is_seen
    W0 = {"dap": 1.0, "proto": 1.0, "prose": 0.5}
    rows = {}
    print(f"test: {len(it)} named objects, {len(acc_t.seen)} seen / {len(acc_t.unseen)} unseen classes; "
          f"val: {len(iv)} objects", flush=True)

    def score(label, ft, fv=None, note=""):
        r = acc_t(ft)
        r.update(acc_t.ausuc(ft))
        if fv is not None:  # seen accuracy on validation: the only signal an honest rule may use
            r["val_seen"] = acc_v(fv)["seen"]
        r["note"] = note
        rows[label] = r
        print(f"{label:60s} seen {r['seen']:.3f} unseen {r['unseen']:.3f} H {r['H']:.3f} "
              f"AUSUC {r['ausuc']:.3f} oracleH {r['best_H_oracle']:.3f}", flush=True)

    def gamma_rule(fv, keep):
        """largest gamma keeping `keep` of the gamma=0 seen accuracy on validation"""
        base = acc_v(fv)["seen"]
        best = 0.0
        for g in np.linspace(0, 3, 61):
            if acc_v(fv - g * is_seen[None])["seen"] >= keep * base:
                best = g
        return best

    for wl, w in [("dap+proto+prose (paper)", W0), ("descriptor agreement only", {"dap": 1.0}),
                  ("prose only", {"prose": 1.0}), ("prototype only", {"proto": 1.0}),
                  ("dap+prose", {"dap": 1.0, "prose": 1.0})]:
        ft, fv = _fused(namer, evt, w), _fused(namer, evv, w)
        score(f"{wl}, gamma=0", ft, fv)
        for keep in (0.9, 0.8, 0.7, 0.5):
            g = gamma_rule(fv, keep)
            score(f"{wl}, gamma={g:.2f} (val keep {keep})", ft - g * is_seen[None], fv - g * is_seen[None])
        # entry-wise standardisation: each entry's score is centred and scaled by its own
        # distribution over validation objects (no unseen label used). Trained classes score
        # high on everything; this removes that per-entry offset.
        mu, sd = fv.mean(0, keepdims=True), fv.std(0, keepdims=True) + 1e-6
        score(f"{wl}, entry-standardised on val", (ft - mu) / sd, (fv - mu) / sd,
              "inductive: statistics from validation objects")
        mu_t, sd_t = ft.mean(0, keepdims=True), ft.std(0, keepdims=True) + 1e-6
        score(f"{wl}, entry-standardised on test (transductive)", (ft - mu_t) / sd_t, None,
              "transductive: unlabelled test pool")
    (run / "posthoc" / "naming.json").write_text(json.dumps(rows, indent=1, default=float))


# ---------------------------------------------------------------------------- CLIP retrieval
def clipret(a):
    run = Path(a.run)
    cfg, seen, unseen = _seen_unseen(run)
    namer = _namer(run, cfg)
    names = namer.c.entry_names
    te = _load(run, "test")
    vk = a.visual
    assert vk in te, f"no '{vk}' in the dump (raw CLIP: dump with --clip; zclip: a run with model.clip_field)"
    vlab = "raw CLIP" if vk == "clip" else "distilled CLIP field"
    from transformers import CLIPModel, CLIPProcessor
    from trashscribe.schema import Atlas
    cp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    cm = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").eval()
    from trashscribe.engine import load_vocab
    _, atlas = load_vocab(cfg)
    full = Atlas.merge([atlas] + [Atlas.load(_p(p)) for p in cfg.get("novel_atlases", [])])
    with torch.no_grad():
        prose = [full[n].prose or n if n in full else n for n in names]
        tp = cp(text=prose, return_tensors="pt", padding=True, truncation=True)
        PR = F.normalize(_vec(cm.get_text_features(**tp)), dim=-1)
        tn = cp(text=[f"a photo of {n.lower().replace('_', ' ')}, a piece of litter" for n in names],
                return_tensors="pt", padding=True)
        NM = F.normalize(_vec(cm.get_text_features(**tn)), dim=-1)
    clip = te[vk].float()
    sims = {f"{vlab}: vs atlas sentence": (clip @ PR.T).numpy(),
            f"{vlab}: vs class name": (clip @ NM.T).numpy()}
    ev = _evidence(namer, te)
    W0 = {"dap": 1.0, "proto": 1.0, "prose": 0.5}
    ours = {"TrashScribe fused (paper)": _fused(namer, ev, W0),
            "TrashScribe descriptor agreement": ev["dap"],
            "TrashScribe prose": ev["prose"]}
    combo = {f"TrashScribe fused + {vlab} sentence": _fused(None, {"a": _std_rows(ours["TrashScribe fused (paper)"]),
                                                                    "b": _std_rows(sims[f"{vlab}: vs atlas sentence"])},
                                                           {"a": 1.0, "b": 1.0}),
             f"TrashScribe fused + {vlab} name": _fused(None, {"a": _std_rows(ours["TrashScribe fused (paper)"]),
                                                                "b": _std_rows(sims[f"{vlab}: vs class name"])},
                                                       {"a": 1.0, "b": 1.0})}
    ent = np.array([e if e is not None else "" for e in te["entries"]])
    rows = {}
    for label, M in {**sims, **ours, **combo}.items():
        r = {}
        for e in sorted(unseen):
            if e in names and (ent == e).any():
                rr = retrieval(M[:, names.index(e)], (ent == e).astype(int), n_boot=a.n_perm, n_perm=a.n_perm)
                r[e] = {k: rr[k] for k in ("ap", "lift", "p", "n_pos")}
        rows[label] = r
        sig = sum(v["p"] < 0.01 for v in r.values())
        print(f"{label:42s} mean AP {np.mean([v['ap'] for v in r.values()]):.4f}  "
              f"median lift {np.median([v['lift'] for v in r.values()]):7.1f}  p<0.01: {sig}/{len(r)}  " +
              " ".join(f"{k[:10]}={v['ap']:.3f}" for k, v in r.items()), flush=True)
    # naming top-1 by the same scores, for the record
    it = _scored(namer, te["entries"])
    true_t = [te["entries"][i] for i in it]
    for label, M in {**sims, **ours, **combo}.items():
        g = _gz(M[it], names, true_t, seen, unseen)
        rows[label]["_naming"] = g
        print(f"  naming {label:40s} seen {g['seen']:.3f} unseen {g['unseen']:.3f} H {g['H']:.3f}")
    (run / "posthoc" / ("clipret.json" if vk == "clip" else f"clipret_{vk}.json")).write_text(
        json.dumps(rows, indent=1, default=float))


# ---------------------------------------------------------------------------- leakage
def leakage(a):
    """Environment probe inside each class: can the surroundings be decoded from the
    intrinsic embedding of objects OF THE SAME CLASS? The pooled probe cannot tell this
    apart from 'the class predicts the place' (buoys only on beaches)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    out = {}
    for run in a.runs:
        d = _load(run, "test")
        env = d["env"].numpy().copy()
        ents = np.array([e if e is not None else "" for e in d["entries"]])
        if a.datasets:   # same object pool for every run
            env[~np.isin(np.array(d["dataset"]), a.datasets)] = -1
        res = {}
        for key in ("z", "zc"):
            X = F.normalize(d[key].float(), dim=-1).numpy()
            tot_n = tot_ok = tot_maj = 0
            per = {}
            for c in sorted(set(ents) - {""}):
                m = (ents == c) & (env >= 0)
                envs, cnt = np.unique(env[m], return_counts=True)
                keep_env = envs[cnt >= a.min_per_env]
                if len(keep_env) < 2:
                    continue
                mm = m & np.isin(env, keep_env)
                y = env[mm]
                folds = min(5, int(np.bincount(y).max()), int(min(np.unique(y, return_counts=True)[1])))
                if folds < 2:
                    continue
                clf = LogisticRegression(max_iter=2000, C=1.0)
                pred = cross_val_predict(clf, X[mm], y, cv=StratifiedKFold(folds, shuffle=True, random_state=0))
                ok = int((pred == y).sum())
                maj = int(np.unique(y, return_counts=True)[1].max())
                per[c] = {"n": int(mm.sum()), "n_env": int(len(keep_env)), "acc": ok / mm.sum(), "majority": maj / mm.sum()}
                tot_n += int(mm.sum())
                tot_ok += ok
                tot_maj += maj
            res[key] = {"acc": tot_ok / max(tot_n, 1), "majority": tot_maj / max(tot_n, 1), "n": tot_n,
                        "classes": len(per), "per_class": per}
        out[str(run)] = res
        print(f"{Path(run).name:30s} within-class probe  intrinsic {res['z']['acc']:.3f}  context {res['zc']['acc']:.3f}"
              f"  majority {res['z']['majority']:.3f}  ({res['z']['n']} objects, {res['z']['classes']} classes)", flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))


# ---------------------------------------------------------------------------- descriptor values
def values(a):
    """AP of single descriptor values over test objects, from the entries' atlas profiles:
    a run that supervised the value, a run that never did (E7), and raw CLIP crops."""
    from trashscribe.engine import load_vocab
    import yaml
    rows = {}
    ref, zrefs = None, []
    for run in a.runs:
        d = _load(run, "test")
        cache = AtlasCache.load(Path(run) / "atlas_cache.pt")
        namer = Namer(cache)
        post = namer.posterior(namer.logits(d["z"].float(), d["zc"].float())).numpy()
        rows[Path(run).name] = _value_aps(cache, d["entries"], post, a.values)
        if "zclip" in d:
            zrefs.append((run, d, cache))
        if "clip" in d and ref is None:
            ref = (run, d, cache)
    todo = ([(*ref, "clip")] if ref is not None else []) + [(*z, "zclip") for z in zrefs]
    for run, d, cache, key in todo:   # raw crop / distilled field vs the value's prompt ensemble
        cfg = yaml.safe_load((Path(run) / "config.yaml").read_text())
        schema, _ = load_vocab(cfg)
        from transformers import CLIPModel, CLIPProcessor
        cp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        cm = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").eval()
        ens = dict(zip([f"{f.name}: {v}" for f in schema.fields for v in f.values], schema.phrases()))
        sims = np.zeros((d[key].shape[0], len(cache.value_names)), np.float32)
        with torch.no_grad():
            for j, vn in enumerate(cache.value_names):
                if vn in a.values and vn in ens:
                    t = cp(text=ens[vn], return_tensors="pt", padding=True)
                    e = F.normalize(F.normalize(_vec(cm.get_text_features(**t)), dim=-1).mean(0), dim=-1)
                    sims[:, j] = (d[key].float() @ e).numpy()
        lab = "raw CLIP crops" if key == "clip" else f"{Path(run).name}: distilled CLIP field"
        rows[lab] = _value_aps(cache, d["entries"], sims, a.values)
    for lab, r in rows.items():
        print(f"{lab:28s} " + "  ".join(f"{k.split(': ')[1][:22]}: AP {v['ap']:.3f} (floor {v['floor']:.3f}, n+ {v['n_pos']})"
                                         for k, v in r.items()), flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=1, default=float))


def _value_aps(cache, entries, scores, wanted):
    from sklearn.metrics import average_precision_score
    names = cache.entry_names
    idx = np.array([names.index(e) if e in names else -1 for e in entries])
    out = {}
    for vn in wanted:
        if vn not in cache.value_names:
            continue
        j = cache.value_names.index(vn)
        f = int(cache.field_idx[j])
        cols = (cache.field_idx == f).numpy()
        ok = idx >= 0
        described = np.zeros(len(idx), bool)
        described[ok] = cache.profile_mask[idx[ok]][:, cols].all(1).numpy()
        y = np.zeros(len(idx), int)
        y[ok] = (cache.profile[idx[ok], j] > 0).numpy().astype(int)
        y = y[described]
        if 0 < y.sum() < len(y):
            out[vn] = {"ap": float(average_precision_score(y, scores[described, j])), "floor": float(y.mean()),
                       "n_pos": int(y.sum()), "n": int(len(y))}
    return out


# ---------------------------------------------------------------------------- zoom-in tiling
@torch.no_grad()
def tiles(a):
    """Test-time zoom-in for small objects (aerial litter, beach fragments): besides the
    full frame, the canonical image is cut into k x k overlapping tiles, each enlarged to
    the model size, predicted, and pasted back (mean over overlaps). Scored exactly like
    evaluate.py (SegMeter, threshold 0.5, targets of the canonical frame). Reports full,
    tiled, and their mean, per test dataset. No retraining."""
    from trashscribe.metrics import SegMeter
    run = Path(a.run)
    dev = torch.device(a.device)
    if dev.type == "cuda" and a.mem_frac:
        torch.cuda.set_per_process_memory_fraction(a.mem_frac, 0)
    model, cfg, schema, atlas, tb = load_run(run, device=dev)
    root = Path(cfg.get("data_root", ".")).expanduser()
    names = [n for n in (cfg.get("test_datasets") or [d["name"] for d in cfg["datasets"]])
             if not a.datasets or n in a.datasets]
    ad = build_split(cfg["datasets"], names, "test", atlas, root, set(), None, strict=cfg.get("strict_atlas", True))
    canon = Canonicalizer(schema, atlas, model.cfg.image_size, False)
    dl = DataLoader(MultiDataset(ad, canon), batch_size=1, collate_fn=collate, num_workers=a.workers)
    k, ov = a.k, a.overlap
    meters = {m: SegMeter() for m in ("full", f"tiles{k}x{k}", "mean(full, tiles)")}

    def prob(inp):
        with D.autocast(dev, dev.type == "cuda"):
            lg = model(inp, None)["waste"]
        return torch.sigmoid(F.interpolate(lg.float(), size=inp.shape[-2:], mode="bilinear", align_corners=False))

    t0 = time.time()
    for i, b in enumerate(dl):
        x = b["image"].to(dev)
        S = x.shape[-1]
        tgt, val = b["waste"].to(dev), b["valid"].to(dev)
        vfull = F.interpolate(val.float(), size=(S, S), mode="nearest")[0, 0] > 0
        pf = prob(x)
        acc, cnt = torch.zeros_like(pf), torch.zeros_like(pf)
        t = S // k
        for r in range(k):
            for c in range(k):
                y0, y1 = max(0, r * t - ov), min(S, (r + 1) * t + ov)
                x0, x1 = max(0, c * t - ov), min(S, (c + 1) * t + ov)
                if not bool(vfull[y0:y1, x0:x1].any()):     # letterbox padding only
                    continue
                up = F.interpolate(x[..., y0:y1, x0:x1], size=(S, S), mode="bilinear", align_corners=False)
                pt = F.interpolate(prob(up), size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=False)
                acc[..., y0:y1, x0:x1] += pt
                cnt[..., y0:y1, x0:x1] += 1
        ptl = torch.where(cnt > 0, acc / cnt.clamp_min(1), pf)
        down = lambda p: F.interpolate(p, size=tgt.shape[-2:], mode="bilinear", align_corners=False)  # noqa: E731
        for m, p in zip(meters, (pf, ptl, (pf + ptl) / 2)):
            meters[m].update(down(p), tgt, val, b["dataset"])
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"{i + 1}/{len(dl)} images, {el / 60:.1f} min, eta {(len(dl) - i - 1) * el / (i + 1) / 60:.1f} min",
                  flush=True)
    res = {m: v.result() for m, v in meters.items()}
    for ds in sorted({d for r in res.values() for d in r}):
        print(f"{ds:10s} " + "  ".join(f"{m}: IoU {res[m][ds]['iou']:.3f} P {res[m][ds]['precision']:.3f} "
                                        f"R {res[m][ds]['recall']:.3f}" for m in meters), flush=True)
    outd = run / "posthoc"
    outd.mkdir(exist_ok=True)
    (outd / f"tiles{k}.json").write_text(json.dumps({"k": k, "overlap": ov, "datasets": names, "res": res},
                                                    indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dump")
    p.add_argument("run")
    p.add_argument("--split", default="test")
    p.add_argument("--datasets", nargs="*")
    p.add_argument("--clip", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--mem-frac", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=2)
    p = sub.add_parser("naming")
    p.add_argument("run")
    p = sub.add_parser("clipret")
    p.add_argument("run")
    p.add_argument("--visual", default="clip", choices=["clip", "zclip"])
    p.add_argument("--n-perm", type=int, default=2000)
    p = sub.add_parser("leakage")
    p.add_argument("runs", nargs="+")
    p.add_argument("--min-per-env", type=int, default=5)
    p.add_argument("--datasets", nargs="*", default=["taco", "zerowaste", "bepli"])
    p.add_argument("--out", default=str(_p("runs/_posthoc_leakage.json")))
    p = sub.add_parser("tiles")
    p.add_argument("run")
    p.add_argument("--datasets", nargs="*")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--overlap", type=int, default=96)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--mem-frac", type=float, default=0.17)
    p.add_argument("--workers", type=int, default=2)
    p = sub.add_parser("values")
    p.add_argument("runs", nargs="+")
    p.add_argument("--values", nargs="+", required=True)
    p.add_argument("--out", default=str(_p("runs/_posthoc_values.json")))
    a = ap.parse_args()
    {"dump": dump, "naming": naming, "clipret": clipret, "leakage": leakage, "values": values,
     "tiles": tiles}[a.cmd](a)


if __name__ == "__main__":
    main()
