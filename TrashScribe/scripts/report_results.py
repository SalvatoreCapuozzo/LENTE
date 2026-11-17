"""Regenerate the tables of RESULTS.md from runs/ (called by run_queue.py after
every job; safe to run any time). Only the block between the AUTO markers is
rewritten: the notes written by hand around it are kept.

    python scripts/report_results.py
"""
import _bootstrap  # noqa: F401
import json
import math
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "RESULTS.md"
QUEUE = ROOT / "configs" / "gpu_queue.yaml"
BEGIN, END = "<!-- AUTO:BEGIN (scripts/report_results.py) -->", "<!-- AUTO:END -->"
DS = ["taco", "zerowaste", "uavvaste", "mju", "bepli"]
DSN = {"taco": "TACO", "zerowaste": "ZeroWaste-f", "uavvaste": "UAVVaste", "mju": "MJU-Waste", "bepli": "BePLi v2"}


def jload(p):
    try:
        return json.loads(Path(p).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def get(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or d.get(k) is None:
            return None
        d = d[k]
    return d


def f3(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v:.3f}" if isinstance(v, (int, float)) else str(v)


def table(hdr, rows):
    out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def jobs():
    try:
        return yaml.safe_load(QUEUE.read_text())["jobs"]
    except Exception:
        return []


def status_section():
    st = jload(RUNS / "_queue_status.json") or {}
    cur = (st.get("current") or {}).get("name")
    hist = {}
    for h in st.get("history", []):
        hist.setdefault(h["name"], []).append(h)
    rows = []
    for i, j in enumerate(jobs(), 1):
        n = j["name"]
        h = jload(RUNS / n / "history.json") or []
        tm = (RUNS / n / "test_metrics.json").exists()
        att = hist.get(n, [])
        if tm:
            s = "done"
        elif n == cur:
            s = "**running**"
        elif att and not att[-1]["ok"]:
            s = f"failed ×{len(att)} (rc {att[-1]['rc']})"
        else:
            s = "queued"
        ep = best = ""
        if h:
            bi = max(range(len(h)), key=lambda k: h[k].get("val_score", -1e9) if "val_score" in h[k] else -1e9)
            ep = f"{len(h)}"
            if "val_score" in h[bi]:
                best = f"ep {h[bi]['epoch']}: IoU {f3(h[bi].get('val_iou'))}, mAP {f3(h[bi].get('val_desc_map'))}"
            elif "val_iou" in h[bi]:
                best = f"IoU {f3(max(r.get('val_iou', 0) for r in h))}"
        tr_h = sum(r.get("time_s", 0) for r in h) / 3600          # training time of the epochs kept
        if not h and att:
            tr_h = sum(a.get("minutes", 0) for a in att) / 60       # baselines: no history rows with time_s
        rows.append([i, f"`{n}`", s, ep, best, f"{tr_h:.1f} h" if tr_h else ""])
    return table(["#", "run", "status", "epochs", "best validation (epoch: pooled IoU, desc mAP)",
                  "training time"], rows)


def _status_md():
    try:
        import progress
        return progress.render_md(progress.snapshot())
    except Exception as e:                       # never let the report fail on the live part
        return status_section() + f"\n\n_(ETA unavailable: {e!r})_"


def summary(name):
    return (jload(RUNS / name / "test_metrics.json") or {}).get("summary") or {}


def main_section():
    rows = []
    for j in jobs():
        if j.get("kind", "train") != "train":
            continue
        n = j["name"]
        m = jload(RUNS / n / "test_metrics.json")
        if not m:
            continue
        s = m.get("summary", {})
        nm, n0, au = s.get("naming") or {}, s.get("naming_uncalibrated") or {}, s.get("naming_ausuc") or {}
        li = s.get("leak_intrinsic") or {}
        probe = "probe" in n
        nam = lambda d: "n/a" if probe else f"{f3(d.get('seen'))} / {f3(d.get('unseen'))} / {f3(d.get('H'))}"  # noqa: E731
        rows.append([f"`{n}`", f3(s.get("waste_iou")), f3(s.get("waste_bf1")),
                     f"{f3(s.get('desc_map'))} ({f3(s.get('desc_floor'))})", nam(n0), nam(nm),
                     "n/a" if probe else f3(au.get("ausuc")), f3(get(s, "calibration", "gamma")),
                     f"{f3(li.get('probe_acc'))} ({f3(li.get('majority'))})"])
    if not rows:
        return "_no finished run yet_"
    return table(["run", "IoU", "BF1", "desc mAP (floor)", "naming default: seen / unseen / H",
                  "naming val-calibrated: seen / unseen / H", "AUSUC", "γ", "env probe (majority)"], rows)


def per_dataset_section():
    rows = []
    for j in jobs():
        n = j["name"]
        m = jload(RUNS / n / "test_metrics.json")
        if not m:
            continue
        seg = m.get("seg") or {}
        if j.get("kind") == "zeroshot":
            seg = m.get("where_owlv2_sam2") or m.get("where_gdino_sam2") or m.get("where_sam3") or {}
            if not seg:          # SAM 3 pseudo-label jobs (E9 teachers) have no segmentation score
                continue
        cells = [f"{f3(get(seg, d, 'iou'))} / {f3(get(seg, d, 'bf1'))}" if d in seg else "" for d in DS]
        rows.append([f"`{n}`"] + cells + [f"{f3(get(seg, '_all', 'iou'))}"])
    if not rows:
        return "_no finished run yet_"
    return table(["run"] + [f"{DSN[d]} IoU / BF1" for d in DS] + ["pooled IoU"], rows)


def lodo_section():
    z = jload(RUNS / "e4_heldout_classes__zeroshot" / "test_metrics.json") or {}
    gd = jload(RUNS / "e4_heldout_classes__zeroshot_gdino" / "test_metrics.json") or {}
    s3 = jload(RUNS / "e4_heldout_classes__zeroshot_sam3" / "test_metrics.json") or {}
    s3o = jload(RUNS / "e4_heldout_classes__zeroshot_sam3obj" / "test_metrics.json") or {}
    s3a = jload(RUNS / "e4_heldout_classes__zeroshot_sam3obj_area" / "test_metrics.json") or {}
    a4 = jload(RUNS / "e4_heldout_classes" / "test_metrics.json") or {}
    rows = []
    for d in DS:
        v = jload(RUNS / f"e3_lodo_{d}" / "test_metrics.json") or {}
        sf = jload(RUNS / f"e3_lodo_{d}__segformer_mit-b2" / "test_metrics.json") or {}
        i = jload(RUNS / f"e1_indomain_{d}" / "test_metrics.json") or {}
        e9 = jload(RUNS / f"e9_sam3_{d}" / "test_metrics.json") or {}
        e10 = jload(RUNS / f"e10_k10_{d}" / "test_metrics.json") or {}
        rows.append([DSN[d], f3(get(z, "where_owlv2_sam2", d, "iou")), f3(get(gd, "where_gdino_sam2", d, "iou")),
                     f3(get(s3, "where_sam3", d, "iou")), f3(get(s3o, "where_sam3", d, "iou")),
                     f3(get(s3a, "where_sam3_area", d, "iou")),
                     f3(get(sf, "seg", d, "iou")), f3(get(v, "seg", d, "iou")), f3(get(v, "seg", d, "bf1")),
                     f3(get(e9, "seg", d, "iou")), f3(get(e10, "seg", d, "iou")), f3(get(a4, "seg", d, "iou")),
                     f3(get(i, "seg", d, "iou"))])
    return table(["held-out dataset", "OWLv2→SAM2 zero-shot", "Grounding DINO→SAM2 zero-shot",
                  "SAM 3 zero-shot (4 waste words)", "SAM 3 zero-shot (16 prompts)",
                  "SAM 3 zero-shot (16 prompts, masks ≤ half image)", "SegFormer-B2 (LODO)",
                  "TrashScribe LODO IoU", "TrashScribe LODO BF1", "+ SAM 3 teacher (E9) IoU", "+ 10 labelled images (E10) IoU", "E4 all five IoU",
                  "E1 in-domain IoU"], rows)


def heldout_section(run):
    m = jload(RUNS / run / "test_metrics.json")
    if not m:
        return f"_`{run}` not finished_"
    s = m.get("summary", {})
    unseen = set(s.get("unseen_classes") or [])
    rows = []
    for e, r in sorted((m.get("retrieval") or {}).items()):
        if e in unseen:
            ci = r.get("ci") or [None, None]
            rows.append([e, r.get("n_pos"), f3(r.get("ap")), f"[{f3(ci[0])}, {f3(ci[1])}]", f"{f3(r.get('prevalence'))}",
                         f3(r.get("lift")), f3(r.get("p"))])
    out = table(["held-out class", "n objects", "AP", "95% CI", "prevalence", "lift", "perm. p"], rows) if rows \
        else "_no held-out class in this test set_"
    st = s.get("naming_by_stream") or {}
    out += (f"\n\nUnseen-class top-1 by evidence stream alone (default point): descriptor agreement "
            f"{f3(st.get('dap'))}, text prototype {f3(st.get('proto'))}, prose {f3(st.get('prose'))}; "
            f"chance {f3(get(m, 'naming', 'chance_unseen'))}. AUSUC oracle best H {f3(get(s, 'naming_ausuc', 'best_H_oracle'))} "
            f"(seen {f3(get(s, 'naming_ausuc', 'at_seen'))}, unseen {f3(get(s, 'naming_ausuc', 'at_unseen'))}).")
    return out


def describe_section():
    m = jload(RUNS / "e4_heldout_classes" / "test_metrics.json") or {}
    p = jload(RUNS / "e5_ablate_probe" / "test_metrics.json") or {}
    z = jload(RUNS / "e4_heldout_classes__zeroshot" / "test_metrics.json") or {}
    fields = [k for k in (m.get("desc") or {}) if not k.startswith("_")] or \
        ["material", "rigidity", "optics", "shape", "surface", "color", "state", "size", "surroundings"]
    rows = []
    for f in fields + ["_macro_intrinsic"]:
        rows.append([f, get(m, "desc", f, "n") or "", f3(get(m, "desc", f, "floor")),
                     f3(get(z, "describe_raw_clip", f, "map")), f3(get(p, "desc", f, "map")),
                     f3(get(m, "desc", f, "map"))])
    return table(["field", "n objects", "prevalence floor", "raw CLIP crops", "closed probe (E5)",
                  "TrashScribe (E4)"], rows)


def zeroshot_section():
    z = jload(RUNS / "e4_heldout_classes__zeroshot" / "test_metrics.json")
    if not z:
        return "_not run yet_"
    nm = z.get("name_raw_clip_prose") or {}
    gd = jload(RUNS / "e4_heldout_classes__zeroshot_gdino" / "test_metrics.json") or {}
    s3 = jload(RUNS / "e4_heldout_classes__zeroshot_sam3" / "test_metrics.json") or {}
    return (f"OWLv2→SAM2 pooled IoU {f3(get(z, 'where_owlv2_sam2', '_all', 'iou'))}; Grounding DINO→SAM2 (boxes ≤ half "
            f"the image) {f3(get(gd, 'where_gdino_sam2', '_all', 'iou'))}; SAM 3 (score ≥ 0.5) with 4 waste words "
            f"{f3(get(s3, 'where_sam3', '_all', 'iou'))}, with 16 prompts "
            f"{f3(get(jload(RUNS / 'e4_heldout_classes__zeroshot_sam3obj' / 'test_metrics.json') or {}, 'where_sam3', '_all', 'iou'))}"
            f"; raw CLIP description macro mAP "
            f"{f3(get(z, 'describe_raw_clip', '_macro_intrinsic', 'map'))}; raw CLIP prose naming seen / unseen / H "
            f"{f3(nm.get('seen'))} / {f3(nm.get('unseen'))} / {f3(nm.get('H'))}.")


def build():
    parts = [BEGIN, f"_Tables regenerated {time.strftime('%Y-%m-%d %H:%M')} from `runs/*/test_metrics.json` "
                    "and `runs/*/history.json`. Numbers are fractions (0–1), test split unless stated._",
             "### Queue status", _status_md(),
             "### All finished TrashScribe runs (test split, pooled over the run's test datasets)",
             "Naming = generalised zero-shot top-1 over the atlas (seen classes / held-out classes / harmonic mean). "
             "\"Default\" uses no seen-class penalty; \"val-calibrated\" uses the penalty γ chosen on validation. "
             "Env probe = 5-fold linear probe predicting the surroundings from the intrinsic object embedding. \"–\" = not defined for that run, not a failure: UAVVaste and MJU-Waste have a single \"waste\" class with no atlas description, so runs tested only on them have no description, naming or probe score; runs without held-out classes in their test set have no unseen / H / AUSUC.",
             main_section(),
             "### Waste segmentation per test dataset (IoU / boundary F1)", per_dataset_section(),
             "### E3 leave-one-dataset-out (waste IoU on the dataset never seen in training or selection)",
             lodo_section(),
             "### E4 held-out classes: retrieval from the atlas entry alone", heldout_section("e4_heldout_classes"),
             "### E6 (640 px) held-out classes", heldout_section("e6_edge_640"),
             "### Description: descriptor mAP per field (E4 test objects)", describe_section(),
             "### Training-free references", zeroshot_section(), END]
    return "\n\n".join(parts)


def main():
    auto = build()
    if OUT.exists():
        txt = OUT.read_text()
        if BEGIN in txt and END in txt:
            a, rest = txt.split(BEGIN, 1)
            _, b = rest.split(END, 1)
            OUT.write_text(a + auto + b)
            return
        OUT.write_text(txt.rstrip() + "\n\n" + auto + "\n")
        return
    OUT.write_text("# TrashScribe — GPU results\n\n" + auto + "\n")


if __name__ == "__main__":
    main()
