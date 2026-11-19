"""Collect the outputs of scripts/jetson_suite.sh into summary.md / summary.json (Table tab:edge).

    python scripts/jetson_summary.py runs/e6_edge_640/jetson

Sources: env_*.json, trtexec build logs (export/*_build.log), bench_<prec>/<dataset>/*.json,
power_<prec>/*.json and, if the owner ran `sudo tegrastats --logfile <out>/tegrastats.log`, the
power rails averaged over the PHASE windows written in status.log.
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

out = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/e6_edge_640/jetson")
exp = out.parent / "export"
out.mkdir(parents=True, exist_ok=True)
S ={"generated": datetime.now().strftime("%Y-%m-%d %H:%M")}


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001
        return None


env = load(out / "env_bench.json") or load(out / "env_start.json") or {}
S["env"] = env

# trtexec (network only, CUDA graph, measured by trtexec after the build)
S["trtexec"] = {}
for prec in ("fp16", "fp32"):
    log = exp / f"trashscribe_640_{prec}_build.log"
    if not log.exists():
        continue
    t = log.read_text(errors="ignore")
    r = {}
    m = re.search(r"Throughput: ([\d.]+) qps", t)
    if m:
        r["throughput_qps"] = float(m.group(1))
    for key, lab in (("GPU Compute Time", "gpu_compute_ms"), ("Latency", "latency_ms")):
        m = re.search(key + r": min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, median = ([\d.]+) ms"
                      r", percentile\(90%\) = ([\d.]+) ms, percentile\(95%\) = ([\d.]+) ms, percentile\(99%\) = ([\d.]+) ms", t)
        if m:
            r[lab] = dict(zip(("min", "max", "mean", "median", "p90", "p95", "p99"), map(float, m.groups())))
    r["passed"] = "PASSED" in t
    r["tactics_skipped_for_memory"] = t.count("Skipping tactic")
    S["trtexec"][prec] = r

# python benchmarks (network + full pipeline on real images, check vs ORT fp32)
S["bench"] = {}
for prec in ("fp16", "fp32"):
    for d in sorted((out / f"bench_{prec}").glob("*")):
        js = sorted(d.glob("bench_*.json"))
        if js:
            S["bench"].setdefault(prec, {})[d.name] = load(js[0])
S["power_runs"] = {p: load(next(iter(sorted((out / f"power_{p}").glob("bench_*.json"))), None))
                   for p in ("fp16", "fp32") if (out / f"power_{p}").exists()
                   and list((out / f"power_{p}").glob("bench_*.json"))}

# power from tegrastats over the PHASE windows of status.log
phases = {}
st = out / "status.log"
if st.exists():
    for line in st.read_text().splitlines():
        m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) PHASE (\w+)_(begin|end)", line)
        if m:
            phases.setdefault(m.group(2), {})[m.group(3)] = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
S["power"] = {}
tl = out / "tegrastats.log"
if tl.exists() and phases:
    samples = []
    for line in tl.read_text(errors="ignore").splitlines():
        m = re.match(r"(\d\d-\d\d-\d{4} \d\d:\d\d:\d\d)", line)
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%m-%d-%Y %H:%M:%S")
        rails = {k: int(v) for k, v in re.findall(r"(VDD_\w+|POM_\w+) (\d+)mW/\d+mW", line)}
        ram = re.search(r"RAM (\d+)/(\d+)MB", line)
        gpu = re.search(r"GR3D_FREQ (\d+)%", line)
        samples.append((ts, rails, int(ram.group(1)) if ram else None, int(gpu.group(1)) if gpu else None))
    for ph, w in phases.items():
        if "begin" not in w or "end" not in w:
            continue
        sel = [s for s in samples if w["begin"] <= s[0] <= w["end"]]
        if not sel:
            continue
        rails = sorted({k for s in sel for k in s[1]})
        S["power"][ph] = {"samples": len(sel),
                          "mean_mW": {k: round(sum(s[1].get(k, 0) for s in sel) / len(sel)) for k in rails},
                          "ram_max_MB": max((s[2] or 0) for s in sel),
                          "gpu_util_mean_pct": round(sum((s[3] or 0) for s in sel) / len(sel), 1)}

(out / "summary.json").write_text(json.dumps(S, indent=1, default=str))

# markdown
L = [f"# Jetson measurements (generated {S['generated']})", ""]
if env:
    L += [f"* Device: {env.get('model')}; JetPack {env.get('jetpack')} (L4T {env.get('l4t')}); TensorRT {env.get('tensorrt')}; "
          f"{env.get('cuda')}; torch {env.get('torch')}; Python {env.get('python')}",
          f"* Power mode: {env.get('nvpmodel')}; GPU clock locked (jetson_clocks): **{env.get('gpu_clock_locked')}** "
          f"(GPU {env.get('gpu_freq_hz')})", ""]
if not env.get("gpu_clock_locked", False):
    L += ["> WARNING: GPU clocks were not locked during the benchmarks: not paper numbers. "
          "Run `sudo jetson_clocks`, delete the `.done_bench_*` / `.done_power` markers and rerun.", ""]
if S["trtexec"]:
    L += ["## Network only, trtexec (CUDA graph, 640 px, batch 1)", "",
          "| precision | GPU compute mean / median / p99 (ms) | end-to-end latency mean (ms) | throughput (FPS) | tactics skipped (memory) |",
          "|---|---|---|---|---|"]
    for p, r in S["trtexec"].items():
        g, la = r.get("gpu_compute_ms", {}), r.get("latency_ms", {})
        L.append(f"| {p} | {g.get('mean', '?')} / {g.get('median', '?')} / {g.get('p99', '?')} | {la.get('mean', '?')} "
                 f"| {r.get('throughput_qps', '?')} | {r.get('tactics_skipped_for_memory', '?')} |")
    L.append("")
if S["bench"]:
    L += ["## Python runtime (scripts/bench_edge.py): network and full pipeline on real test images", "",
          "| precision | dataset | net mean / p95 (ms) | net FPS | pre / net / post / total (ms) | pipeline FPS | objects/frame | "
          "check vs ORT fp32: max abs diff [waste, contour, desc] | waste-mask IoU vs fp32 | torch I/O buffers (MB; TRT memory: see tegrastats RAM) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for p, dd in S["bench"].items():
        for ds, r in dd.items():
            if not r:
                continue
            pm, ck = r.get("pipeline_ms", {}), r.get("check_vs_ort_fp32", {})
            md = ck.get("max_abs_diff[waste,contour,desc]")
            L.append(f"| {p} | {ds} | {r['net_ms_mean']:.1f} / {r['net_ms_p95']:.1f} | {r['net_fps']:.1f} | "
                     f"{pm.get('pre_ms', 0):.1f} / {pm.get('net_ms', 0):.1f} / {pm.get('post_ms', 0):.1f} / {pm.get('total_ms', 0):.1f} | "
                     f"{1e3 / pm['total_ms']:.1f} | {r.get('pipeline_mean_objects') or 0:.1f} | "
                     f"{', '.join(f'{v:.2g}' for v in md) if md else '-'} | "
                     f"{ck.get('waste_mask_iou_engine_vs_fp32', float('nan')):.4f} | {r.get('cuda_peak_mem_mb', 0):.0f} |")
    L.append("")
if S["power"]:
    L += ["## Power (sudo tegrastats, mean over each phase)", "", "| phase | samples | rails (mW) | RAM max (MB) | GPU util (%) |",
          "|---|---|---|---|---|"]
    for ph, r in S["power"].items():
        L.append(f"| {ph} | {r['samples']} | {', '.join(f'{k} {v}' for k, v in r['mean_mW'].items())} | {r['ram_max_MB']} "
                 f"| {r['gpu_util_mean_pct']} |")
    idle = S["power"].get("idle", {}).get("mean_mW", {}).get("VDD_IN")
    for p in ("fp16", "fp32"):
        ld = S["power"].get(f"load_{p}", {}).get("mean_mW", {}).get("VDD_IN")
        run = S["power_runs"].get(p)
        if ld and run:
            L.append(f"\n{p}: board input {ld / 1e3:.2f} W under load"
                     + (f" ({(ld - idle) / 1e3:.2f} W above idle)" if idle else "")
                     + f", {ld * run['net_ms_mean'] / 1e6:.3f} J per frame (network only)")
    L.append("")
elif phases:
    L += ["## Power", "", "No `tegrastats.log` in this folder: power rails need root. Run "
          "`sudo tegrastats --interval 500 --logfile <this folder>/tegrastats.log` during the `power` step.", ""]
L += ["Not measured here: SAM 3 on the device (needs Python >= 3.10 / transformers 5.2, not available on JetPack 5.1); "
      "the PyTorch backend (same reason)."]
(out / "summary.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
