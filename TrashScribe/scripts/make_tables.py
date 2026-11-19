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
        gd = load("e4_heldout_classes__zeroshot_gdino")
        # SAM 3, 16 prompts, masks <= half the image: the stronger rule pooled (0.613 vs 0.579) and the
        # same rule as the Grounding DINO boxes and the E9 teacher labels
        s3 = load("e4_heldout_classes__zeroshot_sam3obj_area")
        i = load(f"e1_indomain_{d}")
        a = load("e4_heldout_classes")       # all five datasets in training: the seen-environment reference
        rows.append(f"{NAMES[d]} & {g(z, 'where_owlv2_sam2', d, 'iou')} & {g(gd, 'where_gdino_sam2', d, 'iou')} & "
                    f"{g(s3, 'where_sam3_area', d, 'iou')} & "
                    f"{g(s, 'seg', d, 'iou')} & "
                    f"{g(v, 'seg', d, 'iou')} & {g(a, 'seg', d, 'iou')} & "
                    f"{g(i, 'seg', d, 'iou')} \\\\")
    write("tab_lodo", "\n".join(rows) + "\n")


def ladder():
    """Edge model vs SAM 3 by the amount of data from the new site (17 Nov): SAM 3 zero-shot (fair rule)
    against TrashScribe with no site data (E3), generic negatives (E3b, MJU only), SAM 3 labels on the
    site's unlabelled images (E9), 10 / 50 labelled images (E10), and the site fully in training (E4).
    Cells at or above SAM 3 are bold."""
    s3 = load("e4_heldout_classes__zeroshot_sam3obj_area")
    a = load("e4_heldout_classes")
    rows = []
    for d in DATASETS:
        ref = (((s3 or {}).get("where_sam3_area") or {}).get(d) or {}).get("iou")
        cells = []
        for run in (f"e3_lodo_{d}", "e3b_lodo_mju_negatives" if d == "mju" else None, f"e9_sam3_{d}",
                    f"e10_k10_{d}", f"e10_k50_{d}", None):
            m = a if run is None and len(cells) == 5 else (load(run) if run else None)
            v = (((m or {}).get("seg") or {}).get(d) or {}).get("iou")
            txt = "--" if v is None else f"{100 * v:.1f}"
            if v is not None and ref is not None and v >= ref:
                txt = f"\\textbf{{{txt}}}"
            cells.append(txt)
        rows.append(f"{NAMES[d]} & {g(s3, 'where_sam3_area', d, 'iou')} & " + " & ".join(cells) + " \\\\")
    write("tab_ladder", "\n".join(rows) + "\n")


def ablation():
    """Ablations of E4 (19 Nov): the variants removed from the queue on 13 Nov are not listed; AUSUC is
    left out because runs evaluated before 13 Nov used an underestimating definition (see RESULTS.md).
    Naming and description are shown only where the variant can produce them."""
    runs = [("Full \\trashscribe{} (E4)", "e4_heldout_classes", "all"),
            ("-- object transplant", "e5_ablate_no_transplant", "all"),
            ("-- environment adversary", "e5_ablate_no_adversary", "all"),
            ("-- MixStyle", "e5_ablate_no_mixstyle", "all"),
            ("-- all three", "e5_ablate_no_debias", "all"),
            ("-- dense pixel loss", "e5_ablate_no_dense", "all"),
            ("closed-vocabulary probe head", "e5_ablate_probe", "desc"),
            ("no description (segmentation only)", "e5_ablate_lente_rgb", "seg"),
            ("Full, 640\\,px (deployed model, E6)", "e6_edge_640", "all")]
    rows = []
    for label, r, what in runs:
        m = load(r)
        s = (m or {}).get("summary", {})
        na = "n/a"
        desc = g(s, "desc_map") if what in ("all", "desc") else na
        nam = [g(s, "naming", k) for k in ("seen", "unseen", "H")] if what == "all" else [na] * 3
        rows.append(f"{label} & {g(s, 'waste_iou')} & {desc} & " + " & ".join(nam) +
                    f" & {g(s, 'leak_intrinsic', 'probe_acc')} \\\\")
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
                        f"{r['lift']:.1f} & {'$<$0.001' if r['p'] < 0.001 else format(r['p'], '.3f')} \\\\")
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
    ladder()
    ablation()
    describe()
    retrieval()
    edge()
