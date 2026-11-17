"""Qualitative comparison with the families of existing work, on the SAME five test
images as the qualitative figure (scripts/make_figures.py: seeded rule, never chosen
by how well any method does on them).

Columns (each a family of prior work on waste, plus a general-purpose state of the art):
  input + annotation
  image-level CLIP zero-shot   (zero-shot waste classification with CLIP / VLMs: one label
                                per image over TrashNet-style categories)
  Grounding DINO -> SAM 2      (open-vocabulary waste detection by names, boxes turned into
                                masks; the "Grounded-SAM" recipe)
  SAM 3                        (promptable concept segmentation, text prompt "litter" etc.)
  SegFormer-B2, LODO           (fully supervised semantic segmentation; only if its run exists)
  TrashScribe                  (E4 instances + waste IoU, numbered objects described in the table)

Below the figure, a table compares the DESCRIPTION of the numbered objects: the attributes
raw CLIP gives on the object crop (zero-shot attribute prompting) vs TrashScribe vs the
annotation (atlas profile of the annotated class).

Every method runs on CPU in its own process (memory bounded, the GPU stays with the queue):
    python scripts/make_comparison.py run trashscribe | clip | gdino | sam3 | segformer
    python scripts/make_comparison.py draw
Predictions are cached in paper/figures/comparison_cache/.
"""
import _bootstrap  # noqa: F401
import argparse
import gc
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from make_figures import (IGNORE, MARK, PALETTE, ROWS, TITLES, iou, latex, outline, pick,  # noqa: E402
                          predict, short_desc, to_image)
from trashscribe.data.registry import build_adapter  # noqa: E402
from trashscribe.engine import _p, load_config, load_run, load_vocab  # noqa: E402
from trashscribe.monitoring import pool_numpy  # noqa: E402
from trashscribe.naming import AtlasCache, Namer  # noqa: E402

CACHE = Path(_p("paper/figures/comparison_cache"))
QUERIES = ["a piece of litter", "trash", "garbage", "plastic waste", "a discarded bottle", "a discarded can"]
SAM3_PROMPTS = ["litter", "trash", "garbage", "plastic waste"]
TRASHNET = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
FIELDS = ("material", "rigidity", "optics", "shape")


def samples(seed=0):
    cfg = load_config(_p("configs/exp/e4_heldout_classes.yaml"))
    root = Path(cfg.get("data_root", ".")).expanduser()
    dcfg = {d["name"]: d for d in cfg["datasets"]}
    holdout = set(cfg.get("holdout") or [])
    out = []
    for ds, lodo, rule, need, min_frac in ROWS:
        ad = build_adapter(dcfg[ds], "test", root)
        i, s = pick(ad, rule, need, holdout, seed, min_frac)
        gt = s.waste_mask().copy()
        if s.valid is not None:
            gt[s.valid == 0] = IGNORE
        out.append({"ds": ds, "lodo": lodo, "index": i, "image_id": str(s.image_id), "sample": s, "gt": gt})
    return out, cfg


def _save(name, obj):
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(obj, CACHE / f"{name}.pt")


def _vec(o):
    return o if torch.is_tensor(o) else o.pooler_output


# ---------------------------------------------------------------- methods
@torch.no_grad()
def run_trashscribe(rows, cfg):
    model, cfg, schema, atlas, tb = load_run(_p("runs/e4_heldout_classes"), device="cpu")
    nc = cfg.get("eval", {}).get("naming", {})
    namer = Namer(AtlasCache.load(_p("runs/e4_heldout_classes/atlas_cache.pt")), nc.get("weights"), 0.0, 0.0)
    res = []
    for r in rows:
        s = r["sample"]
        prob, inst, desc, (nh, nw) = predict(model, s.image, model.cfg.image_size)
        objs = []
        if inst:
            zi, zc = pool_numpy(desc, inst, model.cfg.ring_cells)
            named = namer.name(zi, zc)
            for k, m in enumerate(inst):
                mi = to_image(m, nh, nw, s.image.shape[:2])
                objs.append({"mask": np.packbits(mi), "shape": mi.shape, "area": int(mi.sum()),
                             "name": named["name"][k], "prob": float(named["prob"][k]),
                             "desc": namer.describe(named["posterior"][k])})
        res.append({"prob": prob.astype(np.float16), "objs": objs})
        print("trashscribe", r["ds"], len(objs), "objects", flush=True)
    _save("trashscribe", res)


@torch.no_grad()
def run_clip(rows, cfg):
    """Image-level zero-shot label (prior work) + zero-shot attributes on object crops."""
    from transformers import CLIPModel, CLIPProcessor
    cp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    cm = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").eval()
    schema, _ = load_vocab(cfg)
    t = cp(text=[f"a photo of {c} waste" for c in TRASHNET], return_tensors="pt", padding=True)
    TN = F.normalize(_vec(cm.get_text_features(**t)), dim=-1)
    T = []
    for ens in schema.phrases():
        tt = cp(text=ens, return_tensors="pt", padding=True)
        T.append(F.normalize(F.normalize(_vec(cm.get_text_features(**tt)), dim=-1).mean(0), dim=-1))
    T = torch.stack(T)
    ts = torch.load(CACHE / "trashscribe.pt", weights_only=False)
    res = []
    for r, tr in zip(rows, ts):
        img = r["sample"].image
        z = F.normalize(_vec(cm.get_image_features(**cp(images=img, return_tensors="pt"))), dim=-1)
        p = (100 * z @ TN.T).softmax(-1)[0]
        crops = []
        for o in tr["objs"]:
            m = np.unpackbits(o["mask"])[: o["shape"][0] * o["shape"][1]].reshape(o["shape"]).astype(bool)
            ys, xs = np.nonzero(m)
            cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
            h, w = (ys.max() - ys.min() + 1) * 1.35, (xs.max() - xs.min() + 1) * 1.35
            crops.append(img[max(0, int(cy - h / 2)):int(cy + h / 2) + 1, max(0, int(cx - w / 2)):int(cx + w / 2) + 1])
        attrs = []
        if crops:
            zc = F.normalize(_vec(cm.get_image_features(**cp(images=crops, return_tensors="pt"))), dim=-1)
            lg = 100 * zc @ T.T
            for k in range(len(crops)):
                d = {}
                for f in schema.fields:
                    if f.name not in FIELDS:
                        continue
                    sl = schema.slices[f.name]
                    q = lg[k, sl].softmax(-1)
                    j = int(q.argmax())
                    d[f.name] = [(f.values[j], float(q[j]))]
                attrs.append(d)
        res.append({"image_label": TRASHNET[int(p.argmax())], "image_p": float(p.max()),
                    "image_top": [(TRASHNET[j], float(p[j])) for j in p.argsort(descending=True)[:3].tolist()],
                    "crop_attrs": attrs})
        print("clip", r["ds"], res[-1]["image_top"], flush=True)
    _save("clip", res)


@torch.no_grad()
def _sam2_from_boxes(img, boxes):
    from transformers import Sam2Model, Sam2Processor
    global _SAM2
    if "_SAM2" not in globals():
        _SAM2 = (Sam2Processor.from_pretrained("facebook/sam2.1-hiera-tiny"),
                 Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny").eval())
    sp, sm = _SAM2
    if not len(boxes):
        return np.zeros((0,) + img.shape[:2], bool)
    si = sp(images=img, input_boxes=[[list(map(float, b)) for b in boxes]], return_tensors="pt")
    o = sm(**si, multimask_output=False)
    m = sp.post_process_masks(o.pred_masks, si["original_sizes"])[0]
    return (m[:, 0] > 0).numpy()


@torch.no_grad()
def run_gdino(rows, cfg, box_thr=0.3, text_thr=0.25, max_area=0.5):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    gp = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    gm = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").eval()
    text = ". ".join(QUERIES) + "."
    res = []
    for r in rows:
        img = r["sample"].image
        inp = gp(images=img, text=text, return_tensors="pt")
        out = gm(**inp)
        det = gp.post_process_grounded_object_detection(out, inp.input_ids, threshold=box_thr, text_threshold=text_thr,
                                                        target_sizes=[img.shape[:2]])[0]
        boxes = det["boxes"].numpy()
        labels = [str(x) for x in det.get("text_labels", det.get("labels", []))]
        scores = det["scores"].numpy()
        # usual Grounded-SAM post-filter: "trash" / "garbage" also fire on the whole scene
        h, w = img.shape[:2]
        keep = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= max_area * h * w) if len(boxes) else []
        boxes, scores = boxes[keep], scores[keep]
        labels = [l for l, k in zip(labels, keep) if k]
        masks = _sam2_from_boxes(img, boxes)
        res.append({"boxes": boxes, "labels": labels, "scores": scores,
                    "masks": [np.packbits(m) for m in masks], "shape": img.shape[:2]})
        print("gdino", r["ds"], len(boxes), "boxes", flush=True)
    _save("gdino", res)


@torch.no_grad()
def run_sam3(rows, cfg, thr=0.5):
    from transformers import Sam3Model, Sam3Processor
    sp = Sam3Processor.from_pretrained("facebook/sam3")
    # fp32: this CPU has no native bf16 (AVX2 only) and the bf16 path ran on ~1.5 cores
    dt = torch.float32
    sm = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=dt).eval()
    res = []
    for r in rows:
        img = r["sample"].image
        allm, alls, alll = [], [], []
        inp = sp(images=img, return_tensors="pt")
        vis = sm.get_vision_features(pixel_values=inp.pixel_values.to(dt))
        for q in SAM3_PROMPTS:
            ti = sp(text=q, return_tensors="pt")
            out = sm(vision_embeds=vis, input_ids=ti.input_ids, attention_mask=ti.attention_mask)
            det = sp.post_process_instance_segmentation(out, threshold=thr, mask_threshold=0.5,
                                                        target_sizes=inp.get("original_sizes").tolist())[0]
            for m, sc in zip(det["masks"], det["scores"]):
                allm.append(np.packbits(m.numpy().astype(bool)))
                alls.append(float(sc))
                alll.append(q)
        res.append({"masks": allm, "scores": alls, "labels": alll, "shape": img.shape[:2]})
        print("sam3", r["ds"], len(allm), "masks", flush=True)
        gc.collect()
    _save("sam3", res)


@torch.no_grad()
def run_lodo(rows, cfg):
    """TrashScribe trained WITHOUT the image's dataset (E3 LODO): the fair counterpart of the
    zero-shot methods, which never saw any of the five datasets."""
    res = []
    for r in rows:
        m, *_ = load_run(_p(f"runs/{r['lodo']}"), device="cpu")
        prob, *_ = predict(m, r["sample"].image, m.cfg.image_size)
        res.append({"prob": prob.astype(np.float16)})
        print("lodo", r["ds"], flush=True)
        del m
        gc.collect()
    _save("lodo", res)


@torch.no_grad()
def run_segformer(rows, cfg):
    from transformers import SegformerForSemanticSegmentation
    from trashscribe.edge_runtime import preprocess
    res = []
    for r in rows:
        ck = _p(f"runs/{r['lodo']}__segformer_mit-b2/best.pt")
        if not ck.exists():
            res.append(None)
            continue
        net = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b2", num_labels=1,
                                                               ignore_mismatched_sizes=True).eval()
        net.load_state_dict(torch.load(ck, map_location="cpu"))
        img = r["sample"].image
        x, (nh, nw), _ = preprocess(img, 1024)
        lg = net(pixel_values=torch.from_numpy(x)).logits
        p = torch.sigmoid(F.interpolate(lg, size=(1024, 1024), mode="bilinear"))[0, 0, :nh, :nw].numpy()
        res.append({"prob": cv2.resize(p, (img.shape[1], img.shape[0])).astype(np.float16)})
        print("segformer", r["ds"], flush=True)
        del net
        gc.collect()
    _save("segformer", res)


# ---------------------------------------------------------------- figure
def _unpack(b, shape):
    return np.unpackbits(b)[: shape[0] * shape[1]].reshape(shape).astype(bool)


def draw(rows, cfg, out="paper/figures/comparison"):
    W_IN = 5.16
    plt.rcParams.update({"font.size": 6.5, "font.family": "DejaVu Sans"})
    C = {k: torch.load(CACHE / f"{k}.pt", weights_only=False) for k in ("trashscribe", "clip", "gdino", "sam3", "lodo")
         if (CACHE / f"{k}.pt").exists()}
    seg = torch.load(CACHE / "segformer.pt", weights_only=False) if (CACHE / "segformer.pt").exists() else None
    has_seg = seg is not None and any(x is not None for x in seg)
    cols = ["Input, annotation\nand CLIP image label", "Grounding DINO\n→ SAM 2", "SAM 3\n(text prompt)"]
    if has_seg:
        cols.append("SegFormer-B2\n(dataset unseen)")
    cols += ["TrashScribe\n(dataset unseen)", "TrashScribe\n(all datasets)"]
    nc = len(cols)
    ratios = [r["sample"].image.shape[0] / r["sample"].image.shape[1] for r in rows]
    col_w = W_IN / nc
    Ht = col_w * sum(ratios) + 0.45
    fig, axes = plt.subplots(len(rows), nc, figsize=(W_IN, Ht),
                             gridspec_kw={"height_ratios": ratios, "wspace": 0.03, "hspace": 0.05})
    fig.subplots_adjust(left=0.035, right=0.998, bottom=0.2 / Ht, top=1 - 0.25 / Ht)
    badge = dict(fc="black", alpha=0.65, lw=0, pad=0.8)
    holdout = set(cfg.get("holdout") or [])
    _, atlas = load_vocab(cfg)
    record = []
    for ri, r in enumerate(rows):
        s = r["sample"]
        k_ = min(1.0, 700 / s.image.shape[1])
        dsz = (round(s.image.shape[1] * k_), round(s.image.shape[0] * k_))
        img = cv2.resize(s.image, dsz, interpolation=cv2.INTER_AREA)
        gt = cv2.resize(r["gt"], dsz, interpolation=cv2.INTER_NEAREST)
        ign = gt == IGNORE
        base = img.astype(np.float32)
        base[ign] = 0.35 * base[ign] + 0.65 * 255
        rec = {"dataset": r["ds"], "image_id": r["image_id"], "test_index": r["index"]}

        def rs(m):
            return cv2.resize(m.astype(np.uint8), dsz, interpolation=cv2.INTER_NEAREST) > 0

        def show(ax, mask_full, color, label_iou=True, key=None):
            ov = base.copy()
            m = rs(mask_full)
            ov[m] = 0.45 * ov[m] + 0.55 * np.array(color, np.float32)
            ax.imshow(ov.astype(np.uint8))
            outline(ax, gt == 1, "#00e000", 0.6)
            if label_iou:
                v = iou(mask_full, r["gt"])
                rec[key] = v
                ax.text(0.03, 0.95, f"IoU {v:.2f}", transform=ax.transAxes, va="top", fontsize=5.5,
                        color="white", bbox=badge)
        c = 0
        ax = axes[ri, c]
        ax.imshow(base.astype(np.uint8))
        outline(ax, gt == 1, "#00e000", 0.6)
        ax.set_ylabel(TITLES[r["ds"]], fontsize=6.5, fontweight="bold", labelpad=1.5)
        # image-level CLIP (zero-shot waste classification): one label for the whole picture
        if "clip" in C:
            top = C["clip"][ri]["image_top"]
            ax.text(0.97, 0.05, f"CLIP: {top[0][0]} {top[0][1]:.2f}", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=5.5, color="white", bbox=dict(fc="#1f3a93", alpha=0.85, lw=0, pad=0.8))
            rec["clip_image_top"] = top
        c += 1
        # Grounding DINO -> SAM 2: boxes and their masks
        ax = axes[ri, c]
        if "gdino" in C:
            g = C["gdino"][ri]
            ms = [_unpack(m, g["shape"]) for m in g["masks"]]
            un = np.any(ms, 0) if ms else np.zeros(g["shape"], bool)
            show(ax, un, (230, 30, 30), key="iou_gdino")
            for b in g["boxes"]:
                x0, y0, x1, y1 = b * k_
                ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#ffd400", lw=0.5))
            rec["gdino_n"] = len(ms)
        c += 1
        ax = axes[ri, c]
        if "sam3" in C:
            q = C["sam3"][ri]
            ms = [_unpack(m, q["shape"]) for m in q["masks"]]
            un = np.any(ms, 0) if ms else np.zeros(q["shape"], bool)
            show(ax, un, (230, 30, 30), key="iou_sam3")
            rec["sam3_n"] = len(ms)
        c += 1
        if has_seg:
            ax = axes[ri, c]
            if seg[ri] is not None:
                show(ax, seg[ri]["prob"].astype(np.float32) > 0.5, (230, 30, 30), key="iou_segformer")
            else:
                ax.imshow(base.astype(np.uint8))
            c += 1
        ax = axes[ri, c]
        if "lodo" in C:
            show(ax, C["lodo"][ri]["prob"].astype(np.float32) > 0.5, (230, 30, 30), key="iou_trashscribe_lodo")
        c += 1
        # TrashScribe: instances, numbered objects
        ax = axes[ri, c]
        listed = []
        if "trashscribe" in C:
            t = C["trashscribe"][ri]
            objs = sorted(t["objs"], key=lambda o: -o["area"])
            ov = base.copy()
            masks = []
            for k, o in enumerate(objs):
                m = rs(_unpack(o["mask"], o["shape"]))
                masks.append(m)
                ov[m] = 0.45 * ov[m] + 0.55 * PALETTE[k % len(PALETTE)]
            ax.imshow(ov.astype(np.uint8))
            outline(ax, gt == 1, "#00e000", 0.6)
            v = iou(t["prob"].astype(np.float32) > 0.5, r["gt"])
            rec["iou_trashscribe"] = v
            ax.text(0.03, 0.95, f"IoU {v:.2f}", transform=ax.transAxes, va="top", fontsize=5.5, color="white",
                    bbox=badge)
            # the numbered object: the largest one (same rule as the qualitative table, 1 per image)
            if objs:
                full_m = _unpack(objs[0]["mask"], objs[0]["shape"])
                best, ent = 0, None
                for gm, ge in zip(s.instances, s.entries):
                    inter = float((full_m & (gm > 0)).sum())
                    if inter > best:
                        best, ent = inter, ge
                ci = [i for i, o in enumerate(t["objs"]) if o is objs[0]][0]   # identity: masks are arrays
                listed.append((objs[0], ent, ci))
                m = masks[0]
                ys, xs = np.nonzero(m)
                W = m.shape[1]
                if len(xs) and xs.max() - xs.min() > 0.12 * W:
                    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
                    y, x = np.unravel_index(int(dt.argmax()), dt.shape)
                elif len(xs):
                    x = xs.max() + 0.06 * W
                    if x > 0.94 * W:
                        x = xs.min() - 0.06 * W
                    y = float(np.clip(ys.mean(), 0.1 * m.shape[0], 0.9 * m.shape[0]))
                if len(xs):
                    ax.text(x, y, MARK[0], color="black", fontsize=5.5, fontweight="bold", ha="center", va="center",
                            bbox=dict(boxstyle="circle,pad=0.1", fc="white", ec="black", lw=0.4))
        rec["listed"] = []
        for o, ent, ci in listed:
            clip_d = C["clip"][ri]["crop_attrs"][ci] if "clip" in C and C["clip"][ri]["crop_attrs"] else {}
            prof = {}
            if ent is not None and ent in atlas:
                for f in FIELDS:
                    v = atlas[ent].descriptors.get(f)
                    if v:
                        prof[f] = [(x, 1.0) for x in (v if isinstance(v, list) else [v])]
            rec["listed"].append({"gt": ent, "name": o["name"], "ts_desc": short_desc(o["desc"]),
                                  "clip_desc": short_desc(clip_d), "annotation": short_desc(prof) if prof else "--"})
        for cc in range(nc):
            axes[ri, cc].set_xticks([])
            axes[ri, cc].set_yticks([])
            for sp in axes[ri, cc].spines.values():
                sp.set_linewidth(0.4)
        record.append(rec)
    for cc, h in enumerate(cols):
        axes[0, cc].set_title(h, fontsize=6.3, pad=2.5)
    fig.legend(handles=[Patch(fc="none", ec="#00e000", label="annotated waste"),
                        Patch(fc=(0.9, 0.12, 0.12, 0.55), label="predicted waste"),
                        Patch(fc="none", ec="#ffd400", label="detected box"),
                        Patch(fc=(0.92, 0.92, 0.92), ec="0.6", label="not scored")],
               loc="lower center", ncol=4, frameon=False, fontsize=6.3, bbox_to_anchor=(0.5, -0.01),
               handlelength=1.3, columnspacing=1.0)
    out = Path(_p(out))
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01, dpi=300)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.01, dpi=170)
    out.with_suffix(".json").write_text(json.dumps({"rows": record, "queries_gdino_owl": QUERIES,
                                                    "sam3_prompts": SAM3_PROMPTS, "image_level_classes": TRASHNET,
                                                    "gdino_thresholds": {"box": 0.3, "text": 0.25, "max_box_area": 0.5},
                                                    "sam3_threshold": 0.5}, indent=1, default=float))
    tex = []
    for rec in record:
        for o in rec["listed"]:
            gtn = o["gt"] or ("waste (single-class dataset)" if rec["dataset"] in ("uavvaste", "mju") else "background")
            if o["gt"] in holdout:
                gtn += " (held out)"
            tex.append(f"{TITLES[rec['dataset']]} & {latex(gtn)} & {latex(o['annotation'])} & {latex(o['clip_desc'])} & "
                       f"{latex(o['name'])}: {latex(o['ts_desc'])}\\\\")
    out.with_name(out.name + "_objects.tex").write_text(
        "% generated by scripts/make_comparison.py -- do not edit\n" + "\n".join(tex) + "\n")
    print(json.dumps(record, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "draw"])
    ap.add_argument("method", nargs="?")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rows, cfg = samples()
    if a.cmd == "run":
        {"trashscribe": run_trashscribe, "clip": run_clip, "gdino": run_gdino, "sam3": run_sam3,
         "segformer": run_segformer, "lodo": run_lodo}[a.method](rows, cfg)
    else:
        draw(rows, cfg)


if __name__ == "__main__":
    main()
