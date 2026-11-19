"""Record the software and clock state of the Jetson (for the paper's implementation details).

    python scripts/jetson_env.py --out runs/e6_edge_640/jetson/env.json
"""
import argparse
import json
import platform
import re
import subprocess
from pathlib import Path


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def read(p):
    try:
        return Path(p).read_text().strip()
    except Exception:  # noqa: BLE001
        return None


ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()

l4t = read("/etc/nv_tegra_release") or ""
m = re.search(r"R(\d+).*REVISION: ([\d.]+)", l4t)
l4t_ver = f"{m.group(1)}.{m.group(2)}" if m else None
jetpack = {"35.3.1": "5.1.1", "35.4.1": "5.1.2", "35.5.0": "5.1.3", "35.6.0": "5.1.4",
           "36.3.0": "6.0", "36.4.0": "6.1", "36.4.3": "6.2"}.get(l4t_ver)
gpu = "/sys/class/devfreq/17000000.ga10b"
env = {
    "model": (read("/proc/device-tree/model") or "").strip("\x00"),
    "l4t": l4t_ver, "jetpack": jetpack, "kernel": platform.release(), "python": platform.python_version(),
    "cuda": sh("/usr/local/cuda/bin/nvcc --version | tail -2 | head -1"),
    "nvpmodel": sh("nvpmodel -q 2>/dev/null | head -2 | tr '\\n' ' '"),
    "gpu_freq_hz": {k: read(f"{gpu}/{k}") for k in ("cur_freq", "min_freq", "max_freq")},
    "cpu_freq_khz": [read(f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq") for i in range(6)],
    "cpu_max_khz": read("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"),
    "mem": sh("free -m | sed -n 2,3p"),
}
env["gpu_clock_locked"] = env["gpu_freq_hz"]["min_freq"] == env["gpu_freq_hz"]["max_freq"]
try:
    import tensorrt
    env["tensorrt"] = tensorrt.__version__
except Exception:  # noqa: BLE001
    env["tensorrt"] = None
try:
    import torch
    env["torch"] = torch.__version__
    env["cudnn"] = torch.backends.cudnn.version()
except Exception:  # noqa: BLE001
    env["torch"] = None
try:
    import onnxruntime
    env["onnxruntime"] = onnxruntime.__version__
except Exception:  # noqa: BLE001
    pass
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(env, indent=1))
print(json.dumps(env, indent=1))
