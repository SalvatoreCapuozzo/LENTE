"""Device selection. CUDA first (workstation, Jetson), then Apple MPS, then CPU.

On CUDA, Ampere and newer (RTX 30xx/40xx, A100, Jetson Orin) use bfloat16
autocast with TF32 matmuls; older GPUs fall back to float16 with a GradScaler.
On MPS and CPU everything stays float32 (correctness over speed while
developing off-GPU).
"""
from __future__ import annotations

import contextlib
import os
import platform
import random

import numpy as np
import torch


def pick_device(pref: str = "auto") -> torch.device:
    if pref in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    dev = torch.device(pref)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return dev


def setup(device: torch.device, seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic


def amp_dtype(device: torch.device):
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast(device: torch.device, enabled: bool = True):
    dt = amp_dtype(device)
    if not enabled or dt is None:
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=dt)


def grad_scaler(device: torch.device, enabled: bool = True):
    use = enabled and amp_dtype(device) == torch.float16
    return torch.amp.GradScaler("cuda", enabled=use)


def is_jetson() -> bool:
    return platform.machine() == "aarch64" and os.path.exists("/etc/nv_tegra_release")


def mem_available_mb() -> float | None:
    """System-wide MemAvailable (MB). On a Jetson CPU and GPU share this memory."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


class MemWatch:
    """Footprint of this process on a unified-memory device: MemAvailable when the watch starts
    (before the model is loaded) minus the lowest MemAvailable seen while it runs, sampled every
    `period` s in a thread. Includes what torch's allocator does not see (CUDA context, TensorRT
    engine and workspace, cuDNN/cuBLAS handles); other processes add noise, so keep the desktop quiet.
    Also reports the process peak RSS (VmHWM) and, on CUDA, torch's peak allocation."""

    def __init__(self, period: float = 0.05):
        import threading
        self.period, self.base, self.low = period, mem_available_mb(), mem_available_mb()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop.wait(self.period):
            m = mem_available_mb()
            if m is not None and (self.low is None or m < self.low):
                self.low = m

    def result(self) -> dict:
        r = {"mem_available_start_mb": self.base, "mem_available_min_mb": self.low,
             "footprint_mb": (self.base - self.low) if self.base is not None and self.low is not None else None}
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        r["process_peak_rss_mb"] = int(line.split()[1]) / 1024
        except OSError:
            pass
        if torch.cuda.is_available():
            r["torch_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
            r["torch_peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 2 ** 20
        return r

    def stop(self) -> dict:
        self._stop.set()
        self._t.join()
        return self.result()


def describe(device: torch.device) -> dict:
    d = {"device": str(device), "torch": torch.__version__, "jetson": is_jetson()}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        d.update(gpu=p.name, vram_gb=round(p.total_memory / 2 ** 30, 1), sm=f"{p.major}.{p.minor}",
                 amp=str(amp_dtype(device)))
    return d
