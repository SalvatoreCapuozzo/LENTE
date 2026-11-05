"""Write the manuscript tables (paper/tables/*.tex) from runs/*/test_metrics.json.
Missing runs print as "--", so the paper always compiles and never carries a
number that was not produced by a run.

    python scripts/make_tables.py
"""
import _bootstrap  # noqa: F401
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS, OUT = ROOT / "runs", ROOT / "paper" / "tables"
DATASETS = ["taco", "zerowaste", "uavvaste", "mju", "bepli"]
NAMES = {"taco": "TACO", "zerowaste": "ZeroWaste-f", "uavvaste": "UAVVaste", "mju": "MJU-Waste", "bepli": "BePLi v2"}


def load(run):
    f = RUNS / run / "test_metrics.json"
    return json.loads(f.read_text()) if f.exists() else None


def g(d, *keys, pct=True):
    for k in keys:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return "--"
        d = d[k]
    if isinstance(d, (int, float)) and d == d:
        return f"{100 * d:.1f}" if pct else f"{d:.3f}"
    return "--"


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(body)
    print("wrote", OUT / f"{name}.tex")


def lodo():
    rows = []
    for d in DATASETS:
        v = load(f"e3_lodo_{d}")
        s = load(f"e3_lodo_{d}__segformer_mit-b2")
        z = load("e4_heldout_classes__zeroshot")
        i = load(f"e1_indomain_{d}")
        rows.append(f"{NAMES[d]} & {g(z, 'where_owlv2_sam2', d, 'iou')} & {g(s, 'seg', d, 'iou')} & "
                    f"{g(v, 'seg', d, 'iou')} & {g(v, 'seg', d, 'bf1')} & {g(i, 'seg', d, 'iou')} \\\\")
    write("tab_lodo", "\n".join(rows) + "\n")


def ablation():
    runs = [("Full TrashScribe", "e4_heldout_classes"), ("-- transplant", "e5_ablate_no_transplant"),
            ("-- adversary", "e5_ablate_no_adversary"), ("-- MixStyle", "e5_ablate_no_mixstyle"),
            ("-- all debiasing", "e5_ablate_no_debias"), ("-- dense supervision", "e5_ablate_no_dense"),
            ("LoRA $r{=}8$ instead of adapters", "e5_ablate_lora8"), ("Adapters attn+MLP", "e5_ablate_adapters_both"),
            ("+ geometric gate (depth)", "e5_ablate_geometry"), ("Closed-vocabulary probe", "e5_ablate_probe"),
            ("LENTE recipe only", "e5_ablate_lente_rgb"), ("Full, 640\\,px (edge)", "e6_edge_640")]
    rows = []
    for label, r in runs:
        m = load(r)
        s = (m or {}).get("summary", {})
        rows.append(f"{label} & {g(s, 'waste_iou')} & {g(s, 'desc_map')} & {g(s, 'naming', 'seen')} & "
                    f"{g(s, 'naming', 'unseen')} & {g(s, 'naming', 'H')} & {g(s, 'naming_ausuc', 'ausuc')} & "
                    f"{g(s, 'leak_intrinsic', 'probe_acc')} \\\\")
    write("tab_ablation", "\n".join(rows) + "\n")


def describe():
    m = load("e4_heldout_classes")
    p = load("e5_ablate_probe")
    z = load("e4_heldout_classes__zeroshot")
    rows = []
    fields = ["material", "rigidity", "optics", "shape", "surface", "color", "state", "size", "surroundings"]
    for f in fields:
        rows.append(f"{f} & {g(m, 'desc', f, 'floor', pct=False)} & {g(z, 'describe_raw_clip', f, 'map', pct=False)} & "
                    f"{g(p, 'desc', f, 'map', pct=False)} & {g(m, 'desc', f, 'map', pct=False)} \\\\")
    rows.append("\\midrule")
    rows.append(f"macro (intrinsic) & {g(m, 'desc', '_macro_intrinsic', 'floor', pct=False)} & "
                f"{g(z, 'describe_raw_clip', '_macro_intrinsic', 'map', pct=False)} & "
                f"{g(p, 'desc', '_macro_intrinsic', 'map', pct=False)} & "
                f"{g(m, 'desc', '_macro_intrinsic', 'map', pct=False)} \\\\")
    write("tab_describe", "\n".join(rows) + "\n")


def retrieval():
    m = load("e4_heldout_classes")
    rows = []
    for e, r in sorted(((m or {}).get("retrieval") or {}).items()):
        if e in ((m or {}).get("summary", {}).get("unseen_classes") or []):
            rows.append(f"{e} & {r['n_pos']} & {r['ap']:.3f} & [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}] & "
                        f"{r['lift']:.2f} & {r['p']:.3f} \\\\")
    write("tab_retrieval", "\n".join(rows) + "\n" if rows else "\\multicolumn{6}{c}{-- (run E4) --} \\\\\n")


def edge():
    rows = []
    for f in sorted((RUNS).glob("*/bench/*.json")) + sorted((RUNS / "bench").glob("*.json") if (RUNS / "bench").exists() else []):
        b = json.loads(f.read_text())
        if not b.get("jetson"):
            continue          # only on-device numbers go in the paper
        pm = b.get("pipeline_ms", {})
        rows.append(f"{b['backend'].replace('_', ' ')} & {b['size']} & {b['net_ms_mean']:.1f} & {b['net_fps']:.1f} & "
                    f"{pm.get('total_ms', float('nan')):.1f} & {b.get('cuda_peak_mem_mb', float('nan')):.0f} \\\\")
    write("tab_edge", "\n".join(rows) + "\n" if rows else "\\multicolumn{6}{c}{-- (measure on the Jetson) --} \\\\\n")


if __name__ == "__main__":
    lodo()
    ablation()
    describe()
    retrieval()
    edge()
