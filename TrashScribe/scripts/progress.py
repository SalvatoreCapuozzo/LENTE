"""Live view of the experiment grid: what is done, what runs now, what comes next, ETAs.

    python scripts/progress.py              # one snapshot (every job listed)
    python scripts/progress.py --watch 30   # refresh every 30 s (Ctrl-C to quit); the job list is
                                            # trimmed to the terminal height, --all lists every job

The grid end date (upper bound, and "expected" once finished runs show how much early
stopping saves) is printed at the top and again at the bottom.

Reads configs/gpu_queue.yaml, runs/_queue_status.json and, per run, history.json,
progress.json and test_metrics.json. Estimates come from measured times: training
seconds per epoch and validation seconds per image of the runs so far (priors from
the first E4 measurements until then). A training run is assumed to use all its
epochs (upper bound): early stopping can only make it shorter.
"""
import _bootstrap  # noqa: F401
import argparse
import json
import locale
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

from trashscribe.engine import load_config

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
QUEUE = ROOT / "configs" / "gpu_queue.yaml"
UNIT = "trashscribe-queue"
# images per split (val after removing the E4 held-out TACO images: 114 of 150)
N_VAL = {"taco": 150, "zerowaste": 572, "uavvaste": 116, "mju": 248, "bepli": 744,
         "coco_neg": 49,                      # E3b negatives (scripts/make_negatives.py), no test split
         "mju_sam3": 248, "uavvaste_sam3": 116, "bepli_sam3": 744,   # E9: SAM 3-labelled val splits
         "taco_sam3": 150, "zerowaste_sam3": 572}
N_VAL_HOLDOUT = {"taco": 114}
N_TEST = {"taco": 300, "zerowaste": 929, "uavvaste": 116, "mju": 742, "bepli": 744}
ALL = list(N_TEST)
# priors (seconds), measured on E4 (1024 px, 4000 samples/epoch) on 5 Nov
PRIOR = {"train_epoch": {("train", 1024): 540.0, ("train", 640): 330.0, ("segformer", 1024): 690.0},
         "val_per_img": {("train", 1024): 0.05, ("train", 640): 0.03, ("segformer", 1024): 0.06},
         "retrieval_s": 600.0, "zeroshot_per_img": 1.0, "setup_s": 90.0}


def jload(p):
    try:
        return json.loads(Path(p).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def fmt_h(s):
    if s is None:
        return "?"
    s = max(0, s)
    if s >= 2 * 86400:
        return f"{int(s // 86400)} d {int(s % 86400 // 3600)} h"
    return f"{s / 3600:.1f} h" if s >= 5400 else f"{s / 60:.0f} min"


def fmt_t(ts, now):
    if ts is None:
        return "?"
    lt = time.localtime(ts)
    return time.strftime("%H:%M" if time.strftime("%j", lt) == time.strftime("%j", time.localtime(now))
                         else "%a %d %b %H:%M", lt)


def fmt_date(ts):
    """Full date of a finish time, in the system's date language (e.g. "sab 14 nov 2026, 15:53")."""
    return "?" if ts is None else time.strftime("%a %d %b %Y, %H:%M", time.localtime(ts))


def bar(frac, n=24):
    frac = min(1.0, max(0.0, frac))
    k = int(round(frac * n))
    return "[" + "#" * k + "-" * (n - k) + "]"


def job_info(job, common):
    kind = job.get("kind", "train")
    ov = list(common.get(kind, [])) + list(job.get("overrides", []))
    cfg = load_config(ROOT / job["config"], ov)
    tc = cfg["train"]
    size = int(cfg["model"].get("image_size", 1024))
    vals = cfg.get("val_datasets") or cfg.get("train_datasets") or ALL
    tests = cfg.get("test_datasets") or ALL
    hold = bool(cfg.get("holdout"))
    n_val = sum((N_VAL_HOLDOUT.get(d, N_VAL.get(d, 0)) if hold else N_VAL.get(d, 0)) for d in vals)
    n_test = sum(N_TEST.get(d, 0) for d in tests)
    return {"name": job["name"], "kind": kind, "size": size, "epochs": int(tc["epochs"]),
            "batch": int(tc["batch_size"]) * int(tc.get("accum_steps", 1)),
            "patience": int(tc.get("patience", 10)) if kind == "train" else None,
            "n_val": n_val, "n_test": n_test}


def measured(infos):
    """Median training s/epoch and validation s/image per (kind, size), from histories."""
    tr, va = {}, {}
    for j in infos:
        h = jload(RUNS / j["name"] / "history.json") or []
        key = (j["kind"], j["size"])
        for r in h:
            if j["kind"] == "segformer":
                tr.setdefault(key, []).append(r.get("time_s", 0))
            elif "time_s" in r:
                tr.setdefault(key, []).append(r["time_s"])
                if r.get("val_s"):
                    va.setdefault(key, []).append(r["val_s"] / max(j["n_val"], 1))
    pick = lambda d, k, prior: statistics.median(d[k][-20:]) if d.get(k) else prior  # noqa: E731
    return lambda key: (pick(tr, key, PRIOR["train_epoch"].get(key, 540.0)),
                        pick(va, key, PRIOR["val_per_img"].get(key, 0.05)))


def eval_seconds(j, per_img):
    if j["kind"] == "zeroshot":
        return PRIOR["zeroshot_per_img"] * j["n_test"]
    if j["kind"] == "segformer":
        return per_img * j["n_test"]
    return per_img * (j["n_val"] + 2 * j["n_test"]) + PRIOR["retrieval_s"]


def full_estimate(j, rate):
    """Seconds for a job from scratch (all epochs)."""
    if j["kind"] == "zeroshot":
        return PRIOR["setup_s"] + eval_seconds(j, 0)
    tr, vimg = rate((j["kind"], j["size"]))
    per_epoch = tr + (0 if j["kind"] == "segformer" else vimg * j["n_val"])
    return PRIOR["setup_s"] + j["epochs"] * per_epoch + eval_seconds(j, vimg)


def running_estimate(j, rate, now):
    """(seconds left upper bound, seconds left if early-stopped now, detail dict)."""
    p = jload(RUNS / j["name"] / "progress.json") or {}
    h = jload(RUNS / j["name"] / "history.json") or []
    tr, vimg = rate((j["kind"], j["size"]))
    val_s = 0 if j["kind"] == "segformer" else vimg * j["n_val"]
    phase = p.get("phase", "setup")
    d = {"phase": phase, "epochs_done": len(h), "p": p}
    ev = eval_seconds(j, vimg)
    if j["kind"] == "zeroshot":
        frac = p.get("batch", 0) / max(p.get("batches", 1), 1)
        stage = 0 if phase.startswith("where") else 1
        left = (1 - (stage + frac) / 2) * ev
        return left, left, d
    if phase in ("setup",) or not p:
        return full_estimate(j, rate), None, d
    if phase == "train":
        frac = p.get("step", 0) / max(p.get("steps", 1), 1)
        el = now - p.get("epoch_started", now)
        rest_epoch = (el / frac - el) if frac > 0.05 else tr * (1 - frac)
        cur_left = rest_epoch + val_s
        later = j["epochs"] - p.get("epoch", len(h) + 1)
    elif phase == "val":
        frac = p.get("batch", 0) / max(p.get("batches", 1), 1)
        cur_left = (1 - frac) * val_s
        later = j["epochs"] - p.get("epoch", len(h))
    elif j["kind"] == "segformer":            # "test" / "done"
        left = 0.0 if phase == "done" else ev
        return left, left, d
    else:                                      # train_done / evaluation phases / done
        stages = {"calibration": 0, "default": 1, "calibrated": 2, "retrieval": 3}
        st = next((v for k, v in stages.items() if k in phase), 3 if "retrieval" in phase else 0)
        if "retrieval" in phase:
            left = max(0.0, PRIOR["retrieval_s"] - (now - p.get("phase_started", now)))
        elif phase in ("done", "train_done"):
            left = ev if phase == "train_done" else 0.0
        else:
            n = [j["n_val"], j["n_test"], j["n_test"]][min(st, 2)]
            frac = p.get("batch", 0) / max(p.get("batches", 1), 1)
            rest = [j["n_val"], j["n_test"], j["n_test"]][st + 1:] if st < 2 else []
            left = vimg * ((1 - frac) * n + sum(rest)) + PRIOR["retrieval_s"]
        return left, left, d
    upper = cur_left + later * (tr + val_s) + ev
    early = None
    if j["kind"] == "train" and p.get("bad") is not None and j["patience"]:
        stop_in = max(0, j["patience"] - int(p.get("bad", 0)) - 1)       # further epochs without gain
        early = cur_left + min(later, stop_in) * (tr + val_s) + ev
    return upper, early, d


def _num(v):
    """'n/a' for metrics that do not exist for a run (None or NaN): single-class datasets (UAVVaste,
    MJU-Waste) have no described objects, so description and naming scores are undefined there."""
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"


def best_row(h):
    rows = [r for r in h if "val_score" in r]
    if rows:
        b = max(rows, key=lambda r: r["val_score"])
        m = b.get("val_desc_map")
        return f"ep {b['epoch']}: IoU {b.get('val_iou', 0):.3f}" + ("" if _num(m) == "n/a" else f", mAP {m:.3f}")
    rows = [r for r in h if "val_iou" in r]
    if rows:
        b = max(rows, key=lambda r: r["val_iou"])
        return f"ep {b['epoch']}: IoU {b['val_iou']:.3f}"
    return ""


def result_line(name, kind):
    m = jload(RUNS / name / "test_metrics.json") or {}
    s = m.get("summary") or {}
    if kind == "zeroshot":
        if "pseudo_sam3" in m:   # E9 teacher job: SAM 3 labels of a site, no segmentation score of its own
            q = m["pseudo_sam3"].get("quality_vs_labels_record_only", {})
            return "SAM 3 labels for E9 · IoU vs real labels " + " ".join(
                f"{k.split('/')[-1]} {_num(v.get('iou_vs_labels'))}" for k, v in q.items())
        for key, lab in (("where_sam3_area", "SAM 3, masks ≤ half image"), ("where_sam3", "SAM 3"),
                         ("where_gdino_sam2", "G-DINO→SAM2"), ("where_owlv2_sam2", "OWLv2→SAM2")):
            if key in m:
                return f"zero-shot {lab} IoU {_num(m[key].get('_all', {}).get('iou'))}"
        return "zero-shot (no segmentation score)"
    if kind == "segformer":
        return f"test IoU {_num(s.get('waste_iou'))}"
    nm = s.get("naming") or {}
    out = f"test IoU {_num(s.get('waste_iou'))}"
    if _num(s.get("desc_map")) == "n/a":
        return out + " · description/naming n/a (single-class test set)"
    return out + (f" · desc mAP {_num(s.get('desc_map'))} · naming seen/unseen "
                  f"{_num(nm.get('seen'))}/{_num(nm.get('unseen'))}")


def system():
    out = {}
    try:
        g = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,clocks.sm",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        u, m, t, c = [x.strip() for x in g.strip().split(",")]
        out["gpu"] = f"GPU {u}% · {int(m) / 1024:.1f} GB · {t} °C · {c} MHz"
    except Exception:
        out["gpu"] = "GPU ?"
    try:
        r = subprocess.run(["systemctl", "--user", "show", UNIT, "-p", "ActiveState", "-p", "MemoryCurrent",
                            "-p", "MemoryMax"], capture_output=True, text=True, timeout=5).stdout
        kv = dict(line.split("=", 1) for line in r.strip().splitlines() if "=" in line)
        mem = kv.get("MemoryCurrent", "")
        mx = kv.get("MemoryMax", "")
        gb = lambda v: f"{int(v) / 2**30:.1f}" if v.isdigit() else "?"  # noqa: E731
        out["service"] = f"service {kv.get('ActiveState', '?')} · RAM {gb(mem)}/{gb(mx)} GB"
        out["active"] = kv.get("ActiveState") == "active"
    except Exception:
        out["service"], out["active"] = "service ?", False
    return out


def snapshot():
    now = time.time()
    q = yaml.safe_load(QUEUE.read_text())
    common, jobs = q.get("common", {}), q["jobs"]
    infos = [job_info(j, common) for j in jobs]
    rate = measured(infos)
    st = jload(RUNS / "_queue_status.json") or {}
    cur = (st.get("current") or {}).get("name")
    hist = {}
    for h in st.get("history", []):
        hist.setdefault(h["name"], []).append(h)
    sysinfo = system()
    rows, t = [], now
    done_s = 0.0
    for j in infos:
        n = j["name"]
        att = hist.get(n, [])
        wall = sum(a.get("minutes", 0) for a in att) * 60
        r = {"info": j, "wall": wall}
        if (RUNS / n / "test_metrics.json").exists():
            r["status"] = "done"
            done_s += wall
        elif n == cur and sysinfo["active"]:
            up, early, det = running_estimate(j, rate, now)
            started = time.mktime(time.strptime(st["current"]["started"], "%Y-%m-%d %H:%M:%S"))
            hh = jload(RUNS / n / "history.json") or []
            pp = det["p"]
            spent = sum(x.get("time_s", 0) + x.get("val_s", 0) for x in hh)     # survives restarts
            if pp.get("phase") in ("train", "val") and pp.get("epoch", 0) > len(hh):
                spent += now - pp.get("epoch_started", now)
            spent = max(spent, now - started)
            r.update(status="running", left=up, early=early, det=det, end=now + up,
                     early_end=None if early is None else now + early, wall=spent)
            done_s += spent
            t = now + up
        else:
            fails = [a for a in att if not a.get("ok")]
            r["status"] = f"failed x{len(fails)}" if fails and len(fails) >= int(
                next((jj.get("max_attempts", 3) for jj in jobs if jj["name"] == n), 3)) else "queued"
            if r["status"] == "queued":
                est = full_estimate(j, rate)
                r.update(est=est, start=t, end=t + est)
                t += est
        rows.append(r)
    fr = []
    for r in rows:
        if r["status"] == "done" and r["info"]["kind"] == "train":
            h = jload(RUNS / r["info"]["name"] / "history.json") or []
            if h:
                fr.append(len(h) / r["info"]["epochs"])
    f = statistics.median(fr) if fr else None
    exp_end = None
    if f is not None and f < 0.999:
        cur = next((r for r in rows if r["status"] == "running"), None)
        te = now if cur is None else now + (cur["left"] if cur.get("early") is None
                                            else max(cur["early"], cur["left"] * f))
        for r in rows:
            if r["status"] == "queued":
                te += r["est"] * (f if r["info"]["kind"] == "train" else 1.0)
        exp_end = te
    return {"now": now, "rows": rows, "end": t, "done_s": done_s, "sys": sysinfo, "rate": rate,
            "exp_end": exp_end, "epoch_share": f, "n_finished_train": len(fr)}


def grid_end_lines(s, bold=False):
    now = s["now"]
    b, e = ("\x1b[1m", "\x1b[0m") if bold else ("", "")
    out = [f"{b}GRID END  ≤ {fmt_date(s['end'])}{e}   (in {fmt_h(s['end'] - now)}; every remaining run at "
           f"full length, early stops make it sooner)"]
    if s.get("exp_end"):
        out.append(f"          expected ~ {fmt_date(s['exp_end'])}   (finished runs used a median "
                   f"{100 * s['epoch_share']:.0f}% of their epochs, n={s['n_finished_train']})")
    return out


def render_text(s, max_lines=None, bold=False):
    now, rows = s["now"], s["rows"]
    n_done = sum(r["status"] == "done" for r in rows)
    left = s["end"] - now
    frac = s["done_s"] / max(s["done_s"] + left, 1)
    out = [f"TrashScribe grid — {time.strftime('%a %d %b %Y %H:%M:%S')}"] + grid_end_lines(s, bold) + [
           f"Overall   {bar(frac, 30)} {n_done}/{len(rows)} jobs done · {fmt_h(s['done_s'])} spent · "
           f"≤ {fmt_h(left)} left",
           f"          {s['sys']['service']} · {s['sys']['gpu']}", ""]
    for r in rows:
        if r["status"] != "running":
            continue
        j, d, p = r["info"], r["det"], r["det"]["p"]
        out.append(f"NOW   {j['name']}  ({j['kind']}, {j['size']} px) — phase: {d['phase']}")
        if d["phase"] == "train":
            fr = p.get("step", 0) / max(p.get("steps", 1), 1)
            el = now - p.get("epoch_started", now)
            ips = p.get("step", 0) * j["batch"] / el if el > 0 else 0
            out.append(f"      epoch {p.get('epoch')}/{j['epochs']} {bar(fr)} step {p.get('step')}/{p.get('steps')} "
                       f"· {ips:.1f} img/s")
        elif "batch" in p and d["phase"] != "setup":
            fr = p.get("batch", 0) / max(p.get("batches", 1), 1)
            out.append(f"      {bar(fr)} batch {p.get('batch')}/{p.get('batches')}")
        h = jload(RUNS / j["name"] / "history.json") or []
        if h:
            pat = f" · no gain for {p.get('bad', 0)}/{j['patience']} epochs" if j["patience"] else ""
            out.append(f"      best val {best_row(h)}{pat}")
            trail = " ".join(f"{x.get('val_iou', 0):.3f}" for x in h[-8:])
            out.append(f"      val IoU last epochs: {trail}")
        er = "" if r.get("early_end") is None else f" · earliest {fmt_t(r['early_end'], now)} if it early-stops"
        out.append(f"      run ends ≤ {fmt_t(r['end'], now)} ({fmt_h(r['left'])} left){er}")
        out.append("")
    done = [r for r in rows if r["status"] == "done"]
    if done:
        out.append("DONE")
        for r in done:
            out.append(f"  ✓ {r['info']['name']:<38} {fmt_h(r['wall']):>7}  {result_line(r['info']['name'], r['info']['kind'])}")
        out.append("")
    fails = [r for r in rows if r["status"].startswith("failed")]
    if fails:
        out.append("FAILED (see runs/<name>.log)")
        out += [f"  ✗ {r['info']['name']} ({r['status']})" for r in fails]
        out.append("")
    nxt = [r for r in rows if r["status"] == "queued"]
    footer = [""] + grid_end_lines(s, bold)
    if not s["sys"]["active"]:
        footer += [f"!! {UNIT} is not active — restart it with the command in RESULTS.md"]
    if nxt:
        out.append("NEXT                                         est.             start               end")
        room = len(nxt) if max_lines is None else max(3, max_lines - len(out) - len(footer) - 3)
        shown = nxt if room >= len(nxt) else nxt[:room - 1]
        for i, r in enumerate(shown, 1):
            out.append(f"  {i:>2} {r['info']['name']:<38} {fmt_h(r['est']):>7}  {fmt_t(r['start'], now):>16}  "
                       f"{fmt_t(r['end'], now):>16}")
        if len(shown) < len(nxt):
            out.append(f"     … {len(nxt) - len(shown)} more, the last ({nxt[-1]['info']['name']}) ends "
                       f"≤ {fmt_t(nxt[-1]['end'], now)}   (--all to list them)")
    return "\n".join(out + footer)


def render_md(s):
    """Status table for RESULTS.md."""
    now = s["now"]
    lines = [f"Snapshot {time.strftime('%Y-%m-%d %H:%M')} · {s['sys']['service']} · {s['sys']['gpu']} · "
             f"grid ends ≤ **{fmt_date(s['end'])}** (all jobs at full length; early stops make it sooner"
             + (f"; expected ~{fmt_date(s['exp_end'])}" if s.get("exp_end") else "") + "). "
             "Live view: `conda activate trashscribe && python scripts/progress.py --watch 30`.", "",
             "| # | run | status | epochs | best validation (pooled IoU, desc mAP) | time | ends |",
             "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(s["rows"], 1):
        j = r["info"]
        h = jload(RUNS / j["name"] / "history.json") or []
        ep = f"{len(h)}/{j['epochs']}" if j["kind"] != "zeroshot" else ""
        if r["status"] == "done":
            stt, tm, end = "done", fmt_h(r["wall"]), ""
        elif r["status"] == "running":
            stt, tm, end = f"**running** ({r['det']['phase']})", fmt_h(r["wall"]), f"≤ {fmt_t(r['end'], now)}"
        elif r["status"] == "queued":
            stt, tm, end = "queued", f"~{fmt_h(r['est'])}", f"≤ {fmt_t(r['end'], now)}"
        else:
            stt, tm, end = r["status"], fmt_h(r["wall"]), ""
        lines.append(f"| {i} | `{j['name']}` | {stt} | {ep} | {best_row(h)} | {tm} | {end} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=float, default=0, help="refresh period in seconds (0 = once)")
    ap.add_argument("--all", action="store_true", help="list every queued job even if it overflows the terminal")
    a = ap.parse_args()
    try:
        locale.setlocale(locale.LC_TIME, "")        # dates in the system's language (e.g. "sab 14 nov")
    except locale.Error:
        pass
    tty = sys.stdout.isatty()
    if not a.watch:
        print(render_text(snapshot(), bold=tty))
        return
    try:
        while True:
            rows = shutil.get_terminal_size((120, 40)).lines - 2
            txt = render_text(snapshot(), max_lines=None if a.all else rows, bold=tty)
            sys.stdout.write("\x1b[2J\x1b[H" + txt + f"\n\n(refresh every {a.watch:.0f} s · Ctrl-C to quit)\n")
            sys.stdout.flush()
            time.sleep(a.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    main()
