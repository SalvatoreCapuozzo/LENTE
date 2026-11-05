"""Run experiment configs one after the other on the GPU machine, skipping runs
that already have test metrics. Logs go to runs/<run_name>.log.

    python scripts/run_grid.py configs/exp/e4_*.yaml configs/exp/e5_*.yaml
    python scripts/run_grid.py --dry configs/exp/*.yaml
"""
import _bootstrap  # noqa: F401
import argparse
import subprocess
import sys
import time
from pathlib import Path

from trashscribe.engine import _p, load_config

ap = argparse.ArgumentParser()
ap.add_argument("configs", nargs="+")
ap.add_argument("--dry", action="store_true")
ap.add_argument("--force", action="store_true")
ap.add_argument("overrides", nargs="*", default=[])
a = ap.parse_args()
for c in a.configs:
    cfg = load_config(c)
    run = _p(cfg.get("output_dir", "runs")) / cfg["run_name"]
    if (run / "test_metrics.json").exists() and not a.force:
        print("done   ", cfg["run_name"])
        continue
    cmd = [sys.executable, str(Path(__file__).parent / "train.py"), c, *a.overrides]
    print("run    ", cfg["run_name"], " ".join(cmd[2:]))
    if a.dry:
        continue
    run.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with open(run.parent / f"{cfg['run_name']}.log", "w") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f"        exit {rc} after {(time.time() - t) / 60:.1f} min")
