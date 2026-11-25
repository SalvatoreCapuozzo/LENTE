"""SAM 3 on the device: latency, memory and power of the strongest zero-shot baseline, run exactly as for
its accuracy numbers (scripts/baseline_zeroshot.py --detector sam3 --sam3-prompt-set objects16): image
encoder once per frame, text embeddings of the 16 prompts computed once and kept, one detector/mask
decoder pass per prompt, instances with score >= 0.5, bfloat16 on the GPU, post-processing in float32.

    python scripts/bench_sam3_edge.py --images ~/data/waste/_edge_test --dtype bf16 \\
        --out runs/e6_edge_640/jetson_jp6/sam3_bf16 --status runs/e6_edge_640/jetson_jp6/status.log

Per frame and stage (CUDA synchronised between stages): pre (Sam3Processor: resize to 1008 px and
normalise, copy to the GPU), vision (image encoder), decode (all prompts), post (instance post-processing
at the frame size, all prompts), total. Also the one-prompt lower bound, pre + vision + one decode + its
post: no batching of the prompts can make a frame cheaper than that. Memory: the system-wide drop of
MemAvailable from before the model is loaded (CPU and GPU share it on a Jetson) and torch's peak.
Post-processing is that of Sam3ImageProcessorFast.post_process_instance_segmentation (score = sigmoid(logit)
x sigmoid(presence) > thr; mask = bilinear(sigmoid(mask logits), frame size) > 0.5), computed in chunks of
8 instances and kept boolean: the library makes int64 masks of all kept instances at once (8 bytes per
pixel, about 1 GB for 60 instances on a 1024 px frame), which the 8 GB board refused on a cluttered TACO
frame (24 Nov). --check-post compares both on the first frames.
If the model does not fit, the JSON says so (`fits: false`) and the script exits 0: that is a result.
With --status, PHASE lines around the timed loop let jetson_summary.py cut tegrastats into it.
Not an accuracy run: the edge images carry no labels.

Backends (A22): `torch` (default) is transformers' Sam3Model as above. `parts` runs the two modules of
scripts/sam3_parts.py in PyTorch (bit-identical to Sam3Model in the same dtype), `trt` their TensorRT engines
(export_sam3_onnx.py + trtexec), both with the stored FP32 text embeddings (no text encoder loaded); same
pre- and post-processing. --save-masks writes each frame's union mask; scripts/sam3_agreement.py compares runs.
    python scripts/bench_sam3_edge.py --backend trt --engines runs/e6_edge_640/export/jp6/sam3 ...
"""
import _bootstrap  # noqa: F401
import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

from types import SimpleNamespace

import cv2
import numpy as np
import torch

import sam3_parts as P
from sam3_parts import PROMPTS
from trashscribe import device as D

DT = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="facebook/sam3")
ap.add_argument("--images", required=True, help="folder of <dataset>/ subfolders (or one folder) of images")
ap.add_argument("--backend", default="torch", choices=["torch", "parts", "trt"])
ap.add_argument("--engines", default="runs/e6_edge_640/export/jp6/sam3",
                help="parts/trt: folder of export_sam3_onnx.py (text_<prompts>.pt, sam3_*_fp16.engine)")
ap.add_argument("--dtype", default="bf16", choices=list(DT), help="torch/parts; trt: the engines' precision")
ap.add_argument("--save-masks", action="store_true", help="union mask of every frame -> <out>/<tag>_masks.npz")
ap.add_argument("--prompts", default="objects16", choices=list(PROMPTS))
ap.add_argument("--thr", type=float, default=0.5)
ap.add_argument("--n", type=int, default=30, help="frames per dataset")
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--min-free-mb", type=float, default=None,
                help="do not even try below this MemAvailable (default: weights size + 1.5 GB)")
ap.add_argument("--out", required=True)
ap.add_argument("--status", help="status.log of the suite, for PHASE lines")
ap.add_argument("--check-post", type=int, default=0, help="compare with the library post-processing on N frames")
a = ap.parse_args()

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
tag = f"sam3_{a.dtype}" if a.backend == "torch" else f"sam3_{a.backend}_{a.dtype}"
eng = Path(a.engines)
dev = torch.device("cuda")
prompts = PROMPTS[a.prompts]
res = {"model": a.model, "tag": tag, "backend": a.backend, "dtype": a.dtype, "prompt_set": a.prompts, "prompts": prompts, "threshold": a.thr,
       "host": platform.node(), "jetson": D.is_jetson(), "torch": torch.__version__,
       "gpu": torch.cuda.get_device_name(0), "started": datetime.now().strftime("%F %T")}


def phase(name):
    if a.status:
        with open(a.status, "a") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} PHASE {name}\n")


def is_oom(e):
    s = str(e).lower()
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in s or "cudaerrormemoryallocation" in s \
        or "nvmap" in s or "cannot allocate memory" in s or "nvml_success" in s


def finish(code=0):
    res["memory"] = mw.stop()
    res["finished"] = datetime.now().strftime("%F %T")
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "frames"}, indent=1, default=float))
    raise SystemExit(code)


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


# frames: <images>/<dataset>/*.jpg, or the folder itself
root = Path(a.images).expanduser()
dirs = [d for d in sorted(root.iterdir()) if d.is_dir()] or [root]
sets = {}
for d in dirs:
    fs = [p for p in sorted(d.iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png")][:a.n]
    if fs:
        sets[d.name] = fs
res["datasets"] = {k: len(v) for k, v in sets.items()}
assert sets, f"no images under {root}"

# weights size from the local HF cache (HF_HUB_OFFLINE=1 in the suite)
from huggingface_hub import snapshot_download  # noqa: E402
snap = Path(snapshot_download(a.model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]))
wbytes = sum(p.stat().st_size for p in snap.glob("*.safetensors"))
res["checkpoint_gb"] = wbytes / 1e9
res["weights_in_dtype_gb"] = wbytes / 1e9 * (2 if a.dtype != "fp32" else 4) / 4   # checkpoint is fp32
if a.backend != "torch":      # no text encoder (353 M of the 848 M) and no video tracker on the device
    res["weights_in_dtype_gb"] *= (454.0 + 24.5) / 848.0
    if a.backend == "trt":
        res["engines_gb"] = sum(p.stat().st_size for p in eng.glob(f"sam3_*_{a.dtype}.engine")) / 1e9
        assert res["engines_gb"] > 0, f"no sam3_*_{a.dtype}.engine in {eng}"
need = a.min_free_mb if a.min_free_mb is not None else res["weights_in_dtype_gb"] * 1e3 + 1500
mw = D.MemWatch()
res["mem_available_before_load_mb"] = mw.base
if mw.base is not None and mw.base < need:
    res.update(fits="not attempted", reason=f"{mw.base:.0f} MB available, at least {need:.0f} MB needed "
               f"({res['weights_in_dtype_gb']:.2f} GB of weights in {a.dtype} + 1.5 GB)")
    finish(0)

from transformers import Sam3Model, Sam3Processor  # noqa: E402
t0 = time.perf_counter()
model = vis_m = dec_m = vis_e = dec_e = None
try:
    proc = Sam3Processor.from_pretrained(a.model)
    if a.backend == "torch":
        try:   # straight to the GPU: no second copy in (shared) memory
            model = Sam3Model.from_pretrained(a.model, dtype=DT[a.dtype], device_map="cuda").eval()
        except (ValueError, ImportError, TypeError):
            model = Sam3Model.from_pretrained(a.model, dtype=DT[a.dtype]).to(dev).eval()
    elif a.backend == "parts":
        cfg = P.config(snap)
        vis_m = P.load_vision(snap, cfg, dev, DT[a.dtype])
        dec_m = P.load_decoder(snap, cfg, dev, DT[a.dtype])
    else:
        st = torch.cuda.current_stream()      # one stream: the copies into the engines' inputs stay ordered
        shared = []                           # one activation buffer for the five engines (run in sequence)
        files = [eng / f"sam3_vision_{k}_{a.dtype}.engine" for k in range(len(P.CHUNKS))]
        vis_e = [P.TrtEngine(f, st, shared) for f in files]
        dec_e = P.TrtEngine(eng / f"sam3_decoder_b1_{a.dtype}.engine", st, shared)
        for prev, nxt in zip(vis_e, vis_e[1:]):   # each chunk reads the previous one's output in place
            nxt.bind("hidden_in", prev.bufs["hidden_out"])
        for n in ("fpn0", "fpn1", "fpn2"):    # the decoder reads the encoder's outputs in place
            dec_e.bind(n, vis_e[-1].bufs[n])
        scratch = P.share_memory(shared)
        res["engine_files"] = [str(f) for f in files] + [str(eng / f"sam3_decoder_b1_{a.dtype}.engine")]
        res["trt_activation_mb"] = scratch.numel() / 2**20
    torch.cuda.synchronize()
except Exception as e:  # noqa: BLE001
    if not is_oom(e):
        raise
    res.update(fits=False, stage="load", error=repr(e)[:500])
    finish(0)
res["load_s"] = time.perf_counter() - t0
if model is not None:
    res["parameters_M"] = sum(p.numel() for p in model.parameters()) / 1e6
elif vis_m is not None:
    res["parameters_M"] = sum(p.numel() for m in (vis_m, dec_m) for p in m.parameters()) / 1e6
res["memory_after_load"] = mw.result()      # includes the transient of reading the fp32 checkpoint
import gc  # noqa: E402
gc.collect()
torch.cuda.empty_cache()
time.sleep(2)
res["mem_available_settled_after_load_mb"] = D.mem_available_mb()


@torch.no_grad()
def text_embeddings():
    """As _sam3_text in baseline_zeroshot.py: the text encoder does not depend on the image.
    parts/trt: the FP32 embeddings stored by export_sam3_onnx.py (same prompts, checked)."""
    if model is None:
        t = torch.load(eng / f"text_{a.prompts}.pt")
        assert t["prompts"] == prompts, (t["prompts"], prompts)
        dt = DT[a.dtype] if a.backend == "parts" else torch.float32
        return [(t["embeds"][i:i + 1].to(dev, dt), t["mask"][i:i + 1].to(dev)) for i in range(len(prompts))]
    emb = []
    for q in prompts:
        t = proc(text=q, return_tensors="pt").to(dev)
        e = model.get_text_features(input_ids=t.input_ids, attention_mask=t.attention_mask, return_dict=True)
        emb.append((e.pooler_output, t.attention_mask))
    return emb


@torch.no_grad()
def vision(pv):
    """Image encoder; returns what decode() needs."""
    if a.backend == "torch":
        return model.get_vision_features(pixel_values=pv.to(model.dtype))
    if a.backend == "parts":
        return [f.to(DT[a.dtype]) for f in vis_m(pv)]
    vis_e[0].bufs["pixel_values"].copy_(pv)
    for e in vis_e:
        e(sync=e is vis_e[-1])


@torch.no_grad()
def decode(vis, te, am):
    """One prompt: an object with pred_logits, presence_logits, pred_masks (as Sam3Model's output)."""
    if a.backend == "torch":
        return model(vision_embeds=vis, text_embeds=te, attention_mask=am)
    if a.backend == "parts":
        lo, pr, mk = dec_m(*vis, te, am)
    else:
        dec_e.bufs["text_embeds"].copy_(te)
        dec_e.bufs["text_mask"].copy_(am)
        b = dec_e()
        lo, pr, mk = b["pred_logits"], b["presence_logits"], b["pred_masks"]
    return SimpleNamespace(pred_logits=lo, presence_logits=pr, pred_masks=mk)


@torch.no_grad()
def post_masks(o, h, w, chunk=8):
    """(number of instances, union mask (h, w) bool or None): the library's rule, chunked and boolean."""
    sc = o.pred_logits[0].float().sigmoid()
    if getattr(o, "presence_logits", None) is not None:
        sc = sc * o.presence_logits[0].float().sigmoid()
    keep = sc > a.thr
    if not bool(keep.any()):
        return 0, None
    m = o.pred_masks[0][keep]
    union = None
    for c in m.split(chunk):
        u = (torch.nn.functional.interpolate(c.float().sigmoid()[None], size=(h, w), mode="bilinear",
                                             align_corners=False)[0] > 0.5).any(0)
        union = u if union is None else (union | u)
    return int(keep.sum()), union


@torch.no_grad()
def frame(rgb, txt):
    h, w = rgb.shape[:2]
    t0 = sync()
    inp = proc(images=rgb, return_tensors="pt").to(dev)
    t1 = sync()
    vis = vision(inp.pixel_values)
    t2 = sync()
    dec, post, n_inst = [], [], 0
    union = None
    for te, am in txt:
        s = sync()
        o = decode(vis, te, am)
        m = sync()
        k, u = post_masks(o, h, w)                 # in fp32, as for the accuracy numbers
        n_inst += k
        if u is not None:
            union = u if union is None else (union | u)
        e = sync()
        dec.append(1e3 * (m - s))
        post.append(1e3 * (e - m))
    t3 = sync()
    return {"pre_ms": 1e3 * (t1 - t0), "vision_ms": 1e3 * (t2 - t1), "decode_ms": sum(dec), "post_ms": sum(post),
            "decode_ms_per_prompt": float(np.mean(dec)), "total_ms": 1e3 * (t3 - t0),
            "one_prompt_ms": 1e3 * (t2 - t0) + dec[0] + post[0], "instances": n_inst,
            "waste_fraction": float(union.float().mean()) if union is not None else 0.0}, union


def load_rgb(p):
    im = cv2.imread(str(p))
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


try:
    t0 = time.perf_counter()
    txt = text_embeddings()
    torch.cuda.synchronize()
    res["text_embeddings_once_ms"] = 1e3 * (time.perf_counter() - t0)
    first = load_rgb(next(iter(sets.values()))[0])
    if a.check_post and a.backend == "torch":
        chk = []
        for p in list(next(iter(sets.values())))[:a.check_post]:
            rgb = load_rgb(p)
            h, w = rgb.shape[:2]
            with torch.no_grad():
                inp = proc(images=rgb, return_tensors="pt").to(dev)
                vis = model.get_vision_features(pixel_values=inp.pixel_values.to(model.dtype))
                for te, am in txt[:4]:
                    o = model(vision_embeds=vis, text_embeds=te, attention_mask=am)
                    k, u = post_masks(o, h, w)
                    for kk, v in list(o.items()):
                        if torch.is_tensor(v) and v.is_floating_point():
                            o[kk] = v.float()
                    d3 = proc.post_process_instance_segmentation(o, threshold=a.thr, mask_threshold=0.5,
                                                                 target_sizes=[(h, w)])[0]
                    ref = d3["masks"].bool().any(0) if len(d3["masks"]) else None
                    same = (u is None and ref is None) or (u is not None and ref is not None and bool((u == ref).all()))
                    chk.append({"image": p.name, "instances": [k, len(d3["masks"])], "union_identical": same})
        res["check_post"] = chk
        print("check_post", chk, flush=True)
    for _ in range(a.warmup):
        frame(first, txt)
    masks = {}
    torch.cuda.reset_peak_memory_stats()
    frames = []
    mw.low = D.mem_available_mb()            # from here on: the minimum during inference only
    phase(f"{tag}_begin")
    for ds, fs in sets.items():
        for p in fs:
            rgb = load_rgb(p)
            r, u = frame(rgb, txt)
            r.update(dataset=ds, image=p.name)
            frames.append(r)
            if a.save_masks:     # after the timed frame
                um = np.zeros(rgb.shape[:2], bool) if u is None else u.cpu().numpy()
                masks[f"{ds}__{p.stem}"] = np.packbits(um)
                masks[f"{ds}__{p.stem}__shape"] = np.array(um.shape)
                masks[f"{ds}__{p.stem}__n"] = np.array(r["instances"])
            print(f"{ds} {p.name}: total {r['total_ms']:.0f} ms (vision {r['vision_ms']:.0f}, "
                  f"decode {r['decode_ms']:.0f}), {r['instances']} instances", flush=True)
    phase(f"{tag}_end")
    if a.save_masks:
        np.savez_compressed(out / f"{tag}_masks.npz", **masks)
        res["masks_file"] = str(out / f"{tag}_masks.npz")
    res["mem_available_min_during_inference_mb"] = mw.low
    res["footprint_inference_mb"] = mw.base - mw.low if mw.base is not None and mw.low is not None else None
except Exception as e:  # noqa: BLE001
    if not is_oom(e):
        raise
    phase(f"{tag}_end")
    res.update(fits=False, stage="inference", error=repr(e)[:500])
    finish(0)

res["fits"] = True
keys = ["pre_ms", "vision_ms", "decode_ms", "decode_ms_per_prompt", "post_ms", "total_ms", "one_prompt_ms"]


def summarise(rows):
    s = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    s.update(total_ms_p95=float(np.percentile([r["total_ms"] for r in rows], 95)),
             fps=1e3 / s["total_ms"], one_prompt_fps=1e3 / s["one_prompt_ms"],
             instances_per_frame=float(np.mean([r["instances"] for r in rows])), frames=len(rows))
    return s


res["per_dataset"] = {ds: summarise([r for r in frames if r["dataset"] == ds]) for ds in sets}
res["all"] = summarise(frames)
res["frames"] = frames
finish(0)
