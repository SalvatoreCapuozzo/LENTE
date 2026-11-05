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


def describe(device: torch.device) -> dict:
    d = {"device": str(device), "torch": torch.__version__, "jetson": is_jetson()}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        d.update(gpu=p.name, vram_gb=round(p.total_memory / 2 ** 30, 1), sm=f"{p.major}.{p.minor}",
                 amp=str(amp_dtype(device)))
    return d
