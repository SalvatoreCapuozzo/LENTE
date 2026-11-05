"""One table of every run: the numbers that go into the paper come from here,
never from memory.

    python scripts/list_runs.py [--md] [runs_dir]
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="?", default=str(Path(__file__).resolve().parents[1] / "runs"))
ap.add_argument("--md", action="store_true")
a = ap.parse_args()
rows = []
for d in sorted(Path(a.runs).iterdir()):
    f = d / "test_metrics.json"
    if not f.exists():
        continue
    s = json.loads(f.read_text()).get("summary", {})
    nm = s.get("naming") or {}
    au = s.get("naming_ausuc") or {}
    li = s.get("leak_intrinsic") or {}
    rows.append([d.name, s.get("waste_iou"), s.get("waste_bf1"), s.get("desc_map"), s.get("desc_floor"),
                 nm.get("seen"), nm.get("unseen"), nm.get("H"), au.get("ausuc"), li.get("probe_acc"),
                 li.get("majority")])
hdr = ["run", "IoU", "BF1", "desc mAP", "floor", "seen", "unseen", "H", "AUSUC", "env probe", "majority"]
fmt = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))  # noqa: E731
if a.md:
    print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
    for r in rows:
        print("| " + " | ".join(fmt(v) for v in r) + " |")
else:
    w = max(len(r[0]) for r in rows) if rows else 10
    print(f"{'run':<{w}} " + " ".join(f"{h:>9}" for h in hdr[1:]))
    for r in rows:
        print(f"{r[0]:<{w}} " + " ".join(f"{fmt(v):>9}" for v in r[1:]))
