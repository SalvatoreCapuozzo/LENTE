"""Agreement of SAM 3 runs on the edge frames (A22): union-mask IoU per frame and instance counts.

    python scripts/sam3_agreement.py --ref <dir>/sam3_bf16_masks.npz \\
        --floor <dir>/sam3_parts_fp32_masks.npz --run <dir>/sam3_trt_fp16_masks.npz --out <dir>/sam3_agreement.json

Masks from bench_sam3_edge.py --save-masks: per frame, the union over the 16 prompts of the instances with score
>= 0.5 (mask > 0.5 at frame size). IoU of two empty masks = 1. Rule fixed before the TensorRT output was seen
(attempts/ATTEMPTS.md, A22): --run counts as SAM 3 if its mean IoU with --ref is at least that of --floor minus 0.02.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True, help="PyTorch bf16 (Sam3Model), the configuration of the accuracy numbers")
ap.add_argument("--floor", help="PyTorch FP32: its agreement with --ref is the noise floor of the rule")
ap.add_argument("--run", nargs="+", required=True)
ap.add_argument("--margin", type=float, default=0.02)
ap.add_argument("--out")
a = ap.parse_args()


def load(p):
    z = np.load(p)
    keys = sorted(k for k in z.files if not k.endswith(("__shape", "__n")))
    out = {}
    for k in keys:
        shp = tuple(z[k + "__shape"])
        out[k] = (np.unpackbits(z[k])[:int(np.prod(shp))].reshape(shp).astype(bool), int(z[k + "__n"]))
    return out


def compare(ref, run):
    common = sorted(set(ref) & set(run))
    rows = []
    for k in common:
        (ma, na), (mb, nb) = ref[k], run[k]
        inter, union = np.logical_and(ma, mb).sum(), np.logical_or(ma, mb).sum()
        rows.append({"frame": k, "dataset": k.split("__")[0], "iou": 1.0 if union == 0 else float(inter / union),
                     "instances_ref": na, "instances_run": nb, "area_ref": float(ma.mean()), "area_run": float(mb.mean())})

    def agg(rs):
        iou = np.array([r["iou"] for r in rs])
        d = np.array([r["instances_run"] - r["instances_ref"] for r in rs])
        return {"frames": len(rs), "iou_mean": float(iou.mean()), "iou_median": float(np.median(iou)),
                "iou_min": float(iou.min()), "frames_iou_below_0.9": int((iou < 0.9).sum()),
                "instances_ref_mean": float(np.mean([r["instances_ref"] for r in rs])),
                "instances_run_mean": float(np.mean([r["instances_run"] for r in rs])),
                "instances_abs_diff_mean": float(np.abs(d).mean())}
    res = {"all": agg(rows), "per_dataset": {ds: agg([r for r in rows if r["dataset"] == ds])
                                             for ds in sorted({r["dataset"] for r in rows})},
           "missing_frames": sorted(set(ref) ^ set(run)), "frames": rows}
    return res


ref = load(a.ref)
res = {"ref": a.ref, "margin": a.margin, "runs": {}}
if a.floor:
    res["floor"] = {"file": a.floor, **compare(ref, load(a.floor))}
for p in a.run:
    r = compare(ref, load(p))
    if a.floor:
        need = res["floor"]["all"]["iou_mean"] - a.margin
        r["rule"] = {"needed_iou_mean": need, "passes": r["all"]["iou_mean"] >= need}
    res["runs"][p] = r


def line(name, r):
    s = r["all"]
    return (f"{name}: IoU mean {s['iou_mean']:.4f} median {s['iou_median']:.4f} min {s['iou_min']:.3f}, "
            f"{s['frames_iou_below_0.9']} of {s['frames']} frames < 0.9; instances/frame {s['instances_run_mean']:.1f} "
            f"vs {s['instances_ref_mean']:.1f} (mean |diff| {s['instances_abs_diff_mean']:.2f})"
            + "".join(f"\n    {ds}: IoU {v['iou_mean']:.4f} (min {v['iou_min']:.3f})" for ds, v in r["per_dataset"].items()))


if a.floor:
    print(line("floor " + Path(a.floor).name, res["floor"]))
for p, r in res["runs"].items():
    print(line(Path(p).name, r) + (f"\n    rule: {'PASS' if r['rule']['passes'] else 'FAIL'} "
                                   f"(needs >= {r['rule']['needed_iou_mean']:.4f})" if "rule" in r else ""))
if a.out:
    Path(a.out).write_text(json.dumps(res, indent=1))
