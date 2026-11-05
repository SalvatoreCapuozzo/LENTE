"""Offline inference runtime: no internet, no text encoder, no training code.

The network runs in one of three backends (PyTorch, ONNX Runtime, TensorRT);
everything after it (instances, pooling, scoring, naming, report) is small and
runs with PyTorch on the same device. The vocabulary and the atlas arrive as an
AtlasCache file computed once on the workstation.

    pipe = EdgePipeline(TrtBackend("trashscribe_640_fp16.engine"), AtlasCache.load("atlas_cache.pt"), size=640)
    report = pipe(rgb_uint8)
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import torch

from .data.canonical import IMAGENET_MEAN, IMAGENET_STD
from .monitoring import extract_instances, frame_report, pool_numpy, sentence
from .naming import AtlasCache, Namer


def preprocess(rgb: np.ndarray, size: int):
    h, w = rgb.shape[:2]
    k = size / max(h, w)
    nh, nw = round(h * k), round(w * k)
    canvas = np.zeros((size, size, 3), np.float32)
    canvas[:nh, :nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    x = (canvas / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1)[None].astype(np.float32), (nh, nw), k


class TorchBackend:
    def __init__(self, model, device=None, half=False):
        self.m = model.eval()
        self.dev = device or next(model.parameters()).device
        self.half = half and self.dev.type == "cuda"
        if self.half:
            self.m.half()

    @torch.no_grad()
    def __call__(self, x: np.ndarray):
        t = torch.from_numpy(x).to(self.dev)
        if self.half:
            t = t.half()
        o = self.m(t)
        return torch.sigmoid(o["waste"]).float(), torch.sigmoid(o["contour"]).float(), o["desc"].float()


class OrtBackend:
    def __init__(self, onnx_path, providers=None):
        import onnxruntime as ort
        avail = ort.get_available_providers()
        pref = providers or ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        self.s = ort.InferenceSession(str(onnx_path), providers=[p for p in pref if p in avail])
        self.dev = torch.device("cpu")

    def __call__(self, x):
        w, c, d = self.s.run(None, {"image": x})
        return torch.from_numpy(w), torch.from_numpy(c), torch.from_numpy(d)


class TrtBackend:
    """TensorRT 10 engine with torch CUDA tensors as I/O buffers (JetPack 6)."""

    def __init__(self, engine_path):
        import tensorrt as trt
        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.dev = torch.device("cuda")
        self.stream = torch.cuda.Stream()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.bufs = {}
        for n in self.names:
            shape = tuple(self.engine.get_tensor_shape(n))
            dt = {trt.float32: torch.float32, trt.float16: torch.float16}[self.engine.get_tensor_dtype(n)]
            self.bufs[n] = torch.empty(shape, dtype=dt, device=self.dev)
            self.ctx.set_tensor_address(n, self.bufs[n].data_ptr())

    def __call__(self, x):
        self.bufs["image"].copy_(torch.from_numpy(x).to(self.dev, self.bufs["image"].dtype))
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return (self.bufs["waste_prob"].float(), self.bufs["contour_prob"].float(), self.bufs["desc"].float())


class EdgePipeline:
    def __init__(self, backend, cache: AtlasCache, size: int, naming_cfg: dict | None = None, ring_cells=3,
                 thr=0.5, cthr=0.5):
        self.b, self.size, self.ring, self.thr, self.cthr = backend, size, ring_cells, thr, cthr
        nc = naming_cfg or {}
        self.namer = Namer(cache, nc.get("weights"), nc.get("conf_thr", 0.0), nc.get("margin_thr", 0.0))
        self.timing = {}

    def __call__(self, rgb: np.ndarray, describe=True) -> dict:
        t0 = time.perf_counter()
        x, (nh, nw), _ = preprocess(rgb, self.size)
        t1 = time.perf_counter()
        wp, cp, desc = self.b(x)
        t2 = time.perf_counter()
        s4 = self.size // 4
        valid = np.zeros((s4, s4), bool)
        valid[:max(1, nh // 4), :max(1, nw // 4)] = True
        w = wp[0, 0].cpu().numpy() * valid
        c = cp[0, 0].cpu().numpy()
        inst = extract_instances(w, c, self.thr, self.cthr)
        named = post = None
        if inst and describe:
            zi, zc = pool_numpy(desc[0].to(torch.float32).cpu(), inst, self.ring)
            named = self.namer.name(zi, zc)
            post = named["posterior"]
        rep = frame_report(w, valid.mean(), inst, named, self.namer, post)
        rep["lines"] = [sentence(o) for o in rep["objects"]]
        t3 = time.perf_counter()
        self.timing = {"pre_ms": 1e3 * (t1 - t0), "net_ms": 1e3 * (t2 - t1), "post_ms": 1e3 * (t3 - t2),
                       "total_ms": 1e3 * (t3 - t0)}
        rep["timing"] = self.timing
        rep["_maps"] = (w, c, valid, (nh, nw))
        return rep


def overlay(rgb: np.ndarray, rep: dict, size: int) -> np.ndarray:
    """Waste mask and object ids drawn on the input frame."""
    w, c, valid, (nh, nw) = rep["_maps"]
    h0, w0 = rgb.shape[:2]
    m = cv2.resize((w[:max(1, nh // 4), :max(1, nw // 4)] > 0.5).astype(np.uint8), (w0, h0),
                   interpolation=cv2.INTER_NEAREST)
    out = rgb.copy()
    out[m > 0] = (0.55 * out[m > 0] + 0.45 * np.array([255, 60, 30])).astype(np.uint8)
    sx, sy = w0 / max(1, nw // 4), h0 / max(1, nh // 4)
    for o in rep["objects"]:
        x0, y0, x1, y1 = o["bbox_s4"]
        p0, p1 = (int(x0 * sx), int(y0 * sy)), (int((x1 + 1) * sx), int((y1 + 1) * sy))
        cv2.rectangle(out, p0, p1, (255, 255, 0), 2)
        label = f"{o['id']}:{o.get('name') or 'set aside'}"
        cv2.putText(out, label[:40], (p0[0], max(12, p0[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return out
