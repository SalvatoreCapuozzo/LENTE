"""Qualitative figure of the paper: one test image per dataset with its ground truth,
the instances and names of the multi-domain model (E4), the waste mask of the model
that never saw that dataset (E3 LODO), and the written description of a few objects.

    python scripts/make_figures.py      # -> paper/figures/qualitative.pdf (+ .png preview, .json)

Images are drawn by a fixed rule (seeded shuffle of the test split, first image that
meets a layout criterion), never chosen by how well the model does on them. CPU only,
so it can run next to the GPU queue.
"""
import _bootstrap  # noqa: F401
import argparse
import gc
import json
import random
import textwrap
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from trashscribe.data.canonical import IGNORE  # noqa: E402
from trashscribe.data.registry import build_adapter  # noqa: E402
from trashscribe.edge_runtime import preprocess  # noqa: E402
from trashscribe.engine import _p, load_run  # noqa: E402
from trashscribe.monitoring import extract_instances, pool_numpy  # noqa: E402
from trashscribe.naming import AtlasCache, Namer  # noqa: E402

# dataset, LODO run, number of annotated objects, must contain, minimum area of that object
# (or of the largest one) as a fraction of the image, so that it is visible at print size
ROWS = [
    ("taco", "e3_lodo_taco", (1, 8), "holdout", 0.01),
    ("zerowaste", "e3_lodo_zerowaste", (2, 10), None, 0.01),
    ("uavvaste", "e3_lodo_uavvaste", (2, 12), None, 0.001),
    ("mju", "e3_lodo_mju", (1, 2), None, 0.005),
    ("bepli", "e3_lodo_bepli", (4, 40), "Fishing net", 0.005),
]
TITLES = {"taco": "TACO", "zerowaste": "ZeroWaste-f", "uavvaste": "UAVVaste", "mju": "MJU-Waste",
          "bepli": "BePLi v2"}
PALETTE = np.array([[230, 159, 0], [86, 180, 233], [0, 158, 115], [240, 228, 66], [0, 114, 178],
                    [213, 94, 0], [204, 121, 167], [153, 153, 153]], np.float32)  # Okabe-Ito
MARK = ["1", "2", "3"]


def pick(adapter, rule, need, holdout, seed, min_frac):
    idx = list(range(len(adapter)))
    random.Random(seed).shuffle(idx)
    for i in idx:
        s = adapter.load(i)
        h, w = s.image.shape[:2]
        n = len(s.instances)
        if w < h or not (rule[0] <= n <= rule[1]):
            continue
        want = holdout if need == "holdout" else {need} if need else None
        areas = [float((m > 0).sum()) / (h * w) for m, e in zip(s.instances, s.entries)
                 if want is None or e in want]
        if not areas or max(areas) < min_frac:
            continue
        return i, s
    raise RuntimeError(f"no image in {adapter.name} meets the rule")


@torch.no_grad()
def predict(model, img, size):
    """Waste probability at the image resolution, instances (stride 4, letterbox frame),
    descriptor field, and the letterbox geometry."""
    x, (nh, nw), _ = preprocess(img, size)
    o = model(torch.from_numpy(x))
    wp = torch.sigmoid(o["waste"])[0, 0].float().numpy()
    cp = torch.sigmoid(o["contour"])[0, 0].float().numpy()
    s4 = wp.shape[0]
    assert s4 == size // 4, wp.shape
    valid = np.zeros_like(wp, bool)
    valid[:max(1, nh // 4), :max(1, nw // 4)] = True
    inst = extract_instances(wp * valid, cp)
    h0, w0 = img.shape[:2]
    crop = wp[:max(1, nh // 4), :max(1, nw // 4)]
    full = cv2.resize(crop, (w0, h0), interpolation=cv2.INTER_LINEAR)
    return full, inst, o["desc"][0].float(), (nh, nw)


def to_image(m, nh, nw, shape):
    h0, w0 = shape
    return cv2.resize(m[:max(1, nh // 4), :max(1, nw // 4)].astype(np.uint8), (w0, h0),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def iou(pred, gt):
    ok = gt != IGNORE
    p, g = pred & ok, (gt == 1) & ok
    u = (p | g).sum()
    return float((p & g).sum() / u) if u else float("nan")


def outline(ax, mask, color, lw=0.9):
    if mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=lw)


def latex(t):
    t = str(t).replace("_", " ")
    for ch in "&%#$":
        t = t.replace(ch, "\\" + ch)
    return t


def short_desc(d):
    keep = []
    for f in ("material", "rigidity", "optics", "shape"):
        vals = [v for v, _ in d.get(f, [])]
        if vals:
            keep.append(" & ".join(vals))
    return "; ".join(keep) if keep else "no confident description"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multi", default="e4_heldout_classes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--listed", type=int, default=2, help="objects written out per image")
    ap.add_argument("--out", default="paper/figures/qualitative")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    runs = Path(_p("runs"))

    model, cfg, schema, atlas, tb = load_run(runs / a.multi, device="cpu")
    size = model.cfg.image_size
    holdout = set(cfg.get("holdout") or [])
    root = Path(cfg.get("data_root", ".")).expanduser()
    dcfg = {d["name"]: d for d in cfg["datasets"]}
    nc = cfg.get("eval", {}).get("naming", {})
    tm = json.loads((runs / a.multi / "test_metrics.json").read_text())
    gamma = tm["summary"].get("calibration", {}).get("gamma", 0.0)
    assert gamma == 0.0, "seen-class penalty != 0: pass the seen set to Namer.set_seen"
    namer = Namer(AtlasCache.load(runs / a.multi / "atlas_cache.pt"), nc.get("weights"), 0.0, 0.0)

    rows = []
    for ds, lodo, rule, need, min_frac in ROWS:
        ad = build_adapter(dcfg[ds], "test", root)
        i, s = pick(ad, rule, need, holdout, a.seed, min_frac)
        gt = s.waste_mask().copy()
        if s.valid is not None:
            gt[s.valid == 0] = IGNORE
        prob, inst, desc, (nh, nw) = predict(model, s.image, size)
        objs = []
        if inst:
            zi, zc = pool_numpy(desc, inst, model.cfg.ring_cells)
            named = namer.name(zi, zc)
            for k, m in enumerate(inst):
                mi = to_image(m, nh, nw, s.image.shape[:2])
                # ground-truth entry of the annotated object it overlaps most
                best, ent = 0.0, None
                for gm, ge in zip(s.instances, s.entries):
                    inter = float((mi & (gm > 0)).sum())
                    if inter > best:
                        best, ent = inter, ge
                objs.append({"mask": mi, "area": int(mi.sum()), "name": named["name"][k],
                             "prob": float(named["prob"][k]), "gt": ent,
                             "desc": short_desc(namer.describe(named["posterior"][k]))})
        rows.append({"ds": ds, "lodo": lodo, "index": i, "image_id": s.image_id, "sample": s, "gt": gt,
                     "prob": prob, "objs": objs, "iou_multi": iou(prob > 0.5, gt)})
        print(ds, s.image_id, "objects gt", len(s.instances), "pred", len(objs),
              f"IoU {rows[-1]['iou_multi']:.3f}", flush=True)
    del model
    gc.collect()

    for r in rows:
        m, *_ = load_run(runs / r["lodo"], device="cpu")
        r["prob_lodo"], *_ = predict(m, r["sample"].image, m.cfg.image_size)
        r["iou_lodo"] = iou(r["prob_lodo"] > 0.5, r["gt"])
        print(r["ds"], "LODO", f"IoU {r['iou_lodo']:.3f}", flush=True)
        del m
        gc.collect()

    # ------------------------------------------------------------------ figure
    # drawn at the paper's text width (13.1 cm) so that lettering is printed at its true size
    W_IN = 5.16
    plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.titlesize": 7.5})
    ratios = [r["sample"].image.shape[0] / r["sample"].image.shape[1] for r in rows]
    col_w = W_IN / 3.0
    fig, axes = plt.subplots(len(rows), 3, figsize=(W_IN, col_w * sum(ratios) + 0.38),
                             gridspec_kw={"height_ratios": ratios, "wspace": 0.03, "hspace": 0.05})
    Ht = col_w * sum(ratios) + 0.38
    fig.subplots_adjust(left=0.04, right=0.995, bottom=0.2 / Ht, top=1 - 0.17 / Ht)
    heads = ["Input and annotation", "Multi-domain (E4)", "Dataset unseen (E3)"]
    record = []
    for ri, r in enumerate(rows):
        s = r["sample"]
        # display copies at <= 900 px wide (the PDF does not need the full resolution)
        k_ = min(1.0, 900 / s.image.shape[1])
        dsz = (round(s.image.shape[1] * k_), round(s.image.shape[0] * k_))
        img = cv2.resize(s.image, dsz, interpolation=cv2.INTER_AREA)
        gt = cv2.resize(r["gt"], dsz, interpolation=cv2.INTER_NEAREST)
        for o in r["objs"]:
            o["mask_d"] = cv2.resize(o["mask"].astype(np.uint8), dsz, interpolation=cv2.INTER_NEAREST).astype(bool)
        H, W = gt.shape
        ign = gt == IGNORE
        base = img.astype(np.float32)
        base[ign] = 0.35 * base[ign] + 0.65 * 255  # ignored pixels washed out
        # 1. input + ground truth outline
        ax = axes[ri, 0]
        ax.imshow(base.astype(np.uint8))
        outline(ax, gt == 1, "#00e000")
        ax.set_ylabel(TITLES[r["ds"]], fontsize=7.5, fontweight="bold", labelpad=2)
        # 2. E4 instances; the objects described in the table are numbered
        objs = sorted(r["objs"], key=lambda o: -o["area"])
        special = [o for o in objs if o["gt"] in holdout or o["gt"] == "Fishing net"]
        listed = (special[:1] + [o for o in objs if not any(o is t for t in special[:1])])[:a.listed]
        over = img.astype(np.float32).copy()
        for k, o in enumerate(objs):
            over[o["mask_d"]] = 0.45 * over[o["mask_d"]] + 0.55 * PALETTE[k % len(PALETTE)]
        ax = axes[ri, 1]
        ax.imshow(over.astype(np.uint8))
        for o in objs:
            outline(ax, o["mask_d"], "white", 0.5)
        for k, o in enumerate(listed):
            m = o["mask_d"]
            ys, xs = np.nonzero(m)
            if xs.max() - xs.min() > 0.12 * W:  # large object: number at its most interior point
                dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
                y, x = np.unravel_index(int(dt.argmax()), dt.shape)
            else:                               # small object: number beside it, not on it
                x = xs.max() + 0.045 * W
                if x > 0.95 * W:
                    x = xs.min() - 0.045 * W
                y = float(np.clip(ys.mean(), 0.08 * H, 0.92 * H))
            ax.text(x, y, MARK[k], color="black", fontsize=6.5, fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec="black", lw=0.5))
        badge = dict(fc="black", alpha=0.65, lw=0, pad=1.0)
        ax.text(0.02, 0.96, f"IoU {r['iou_multi']:.2f}", transform=ax.transAxes, va="top", fontsize=6.5,
                color="white", bbox=badge)
        # 3. LODO waste mask vs ground truth
        pl = cv2.resize((r["prob_lodo"] > 0.5).astype(np.uint8), dsz, interpolation=cv2.INTER_NEAREST) > 0
        lo = base.copy()
        lo[pl] = 0.45 * lo[pl] + 0.55 * np.array([230, 30, 30])
        ax = axes[ri, 2]
        ax.imshow(lo.astype(np.uint8))
        outline(ax, gt == 1, "#00e000")
        ax.text(0.02, 0.96, f"IoU {r['iou_lodo']:.2f}", transform=ax.transAxes, va="top", fontsize=6.5,
                color="white", bbox=badge)
        for c in range(3):
            axes[ri, c].set_xticks([])
            axes[ri, c].set_yticks([])
            for sp in axes[ri, c].spines.values():
                sp.set_linewidth(0.4)
        record.append({"dataset": r["ds"], "image_id": str(r["image_id"]), "test_index": r["index"],
                       "iou_multi": r["iou_multi"], "iou_lodo": r["iou_lodo"], "n_gt": len(s.instances),
                       "n_pred": len(objs),
                       "listed": [{q: o[q] for q in ("name", "prob", "gt", "desc", "area")} for o in listed]})
    for c, h in enumerate(heads):
        axes[0, c].set_title(h, fontsize=7.5, pad=3)
    fig.legend(handles=[Patch(fc="none", ec="#00e000", label="annotated waste"),
                        Patch(fc=(0.9, 0.12, 0.12, 0.55), label="predicted waste (E3)"),
                        Patch(fc=(0.92, 0.92, 0.92), ec="0.6", label="not annotated, not scored")],
               loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.012),
               handlelength=1.4, columnspacing=1.2)
    out = Path(_p(a.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01, dpi=300)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.01, dpi=150)
    meta = {"multi_run": a.multi, "seed": a.seed, "rule": "seeded shuffle of the test split, first landscape image "
            "with the stated number of annotated objects whose largest (TACO: held-out; BePLi: fishing-net) "
            "object covers min_frac of the image", "rows_rule": [list(r) for r in ROWS], "rows": record}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    # table of the numbered objects, \input by the paper under the figure
    tex = []
    for rec in record:
        for k, o in enumerate(rec["listed"]):
            gtn = o["gt"] or ("waste (single-class dataset)" if rec["dataset"] in ("uavvaste", "mju") else "background")
            if o["gt"] in holdout:
                gtn += " (held out)"
            elif o["gt"] == "Fishing net":
                gtn += " (text only)"
            ds = TITLES[rec["dataset"]] if k == 0 else ""
            tex.append(f"{ds} & {MARK[k]} & {latex(gtn)} & {latex(o['name'])} & {latex(o['desc'])}\\\\")
    out.with_name(out.name + "_objects.tex").write_text(
        "% generated by scripts/make_figures.py -- do not edit\n" + "\n".join(tex) + "\n")
    print("wrote", out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
