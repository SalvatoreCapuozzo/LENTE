"""Latency / throughput / memory of TrashScribe on the current device.

    # workstation or Mac (PyTorch)
    python scripts/bench_edge.py --run runs/<run>
    # Jetson, TensorRT engine + accuracy check against ONNX Runtime FP32
    python3 scripts/bench_edge.py --engine x_fp16.engine --onnx x.onnx --cache atlas_cache.pt --check
    # any device, untrained graph (speed only)
    python scripts/bench_edge.py --untrained --size 640

Reports network-only and full-pipeline (pre + net + instances + naming) times.
Numbers from a laptop are pipeline checks, NOT edge results: only figures
measured on the Jetson in MAXN mode go in the paper.
"""
import _bootstrap  # noqa: F401
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from trashscribe import device as D
from trashscribe.edge_runtime import EdgePipeline, OrtBackend, TorchBackend, TrtBackend

ap = argparse.ArgumentParser()
ap.add_argument("--run")
ap.add_argument("--untrained", action="store_true")
ap.add_argument("--size", type=int, default=640)
ap.add_argument("--engine")
ap.add_argument("--onnx")
ap.add_argument("--cache")
ap.add_argument("--device", default="auto")
ap.add_argument("--half", action="store_true", help="fp16 PyTorch on CUDA")
ap.add_argument("--iters", type=int, default=100)
ap.add_argument("--warmup", type=int, default=20)
ap.add_argument("--check", action="store_true", help="compare engine vs ONNX Runtime FP32")
ap.add_argument("--images", help="folder of real images for the pipeline timing (else random)")
ap.add_argument("--out", default=None)
a = ap.parse_args()

cache = None
if a.engine:
    backend, size = TrtBackend(a.engine), None
    size = backend.bufs["image"].shape[-1]
    name = f"tensorrt:{Path(a.engine).name}"
elif a.onnx and not a.run:
    backend = OrtBackend(a.onnx)
    size = backend.s.get_inputs()[0].shape[-1]
    name = f"onnxruntime:{backend.s.get_providers()[0]}"
else:
    from trashscribe.engine import _p, build_model, load_config, load_run, load_vocab
    from trashscribe.text_bank import TextBank
    dev = D.pick_device(a.device)
    if a.untrained:
        cfg = load_config(_p("configs/base.yaml"), [f"model.image_size={a.size}"])
        schema, _ = load_vocab(cfg)
        model = build_model(cfg, schema, TextBank(cfg["text_model"], "cpu", cache_path=_p(cfg["text_cache"])), dev)
    else:
        model, cfg, *_ = load_run(a.run, device=a.device)
        c = Path(a.run) / "atlas_cache.pt"
        cache = c if c.exists() else None
    backend, size = TorchBackend(model, dev, half=a.half), model.cfg.image_size
    name = f"torch:{dev.type}{':fp16' if a.half and dev.type == 'cuda' else ''}"
cache = a.cache or cache

x = np.random.randn(1, 3, size, size).astype(np.float32)
sync = (lambda: torch.cuda.synchronize()) if torch.cuda.is_available() else (
    (lambda: torch.mps.synchronize()) if torch.backends.mps.is_available() else (lambda: None))
for _ in range(a.warmup):
    backend(x)
sync()
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
ts = []
for _ in range(a.iters):
    t = time.perf_counter()
    backend(x)
    sync()
    ts.append(1e3 * (time.perf_counter() - t))
res = {"backend": name, "size": size, "host": platform.node(), "machine": platform.machine(),
       "jetson": D.is_jetson(), "net_ms_mean": float(np.mean(ts)), "net_ms_p50": float(np.median(ts)),
       "net_ms_p95": float(np.percentile(ts, 95)), "net_fps": float(1e3 / np.mean(ts))}
if torch.cuda.is_available():
    res["cuda_peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
    res["gpu"] = torch.cuda.get_device_name(0)

if cache:
    from trashscribe.naming import AtlasCache
    pipe = EdgePipeline(backend, AtlasCache.load(cache), size)
    frames = []
    if a.images:
        import cv2
        for p in sorted(Path(a.images).glob("*"))[:50]:
            im = cv2.imread(str(p))
            if im is not None:
                frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    frames = frames or [np.random.randint(0, 255, (720, 1280, 3), np.uint8)]
    tt = []
    for i in range(min(a.iters, 50)):
        pipe(frames[i % len(frames)])
        tt.append(pipe.timing)
    res["pipeline_ms"] = {k: float(np.mean([t[k] for t in tt])) for k in tt[0]}

if a.check and a.engine and a.onnx:
    ref = OrtBackend(a.onnx, ["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    diffs, agree = [], []
    for _ in range(5):
        xi = rng.standard_normal((1, 3, size, size)).astype(np.float32)
        r, g = ref(xi), backend(xi)
        diffs.append([float((ri.float() - gi.cpu().float()).abs().max()) for ri, gi in zip(r, g)])
        pr, pg = r[0] > 0.5, g[0].cpu() > 0.5
        agree.append(float((pr == pg).float().mean()))
    res["check_vs_ort_fp32"] = {"max_abs_diff[waste,contour,desc]": np.max(diffs, 0).tolist(),
                                "waste_mask_pixel_agreement": float(np.mean(agree))}
print(json.dumps(res, indent=1))
out = Path(a.out or (Path(a.run) / "bench" if a.run else Path("runs/bench")))
out.mkdir(parents=True, exist_ok=True)
(out / f"bench_{name.replace(':', '_').replace('/', '_')}_{size}.json").write_text(json.dumps(res, indent=1))
