"""Collect the outputs of scripts/jetson_suite.sh into summary.md / summary.json (Table tab:edge).

    python scripts/jetson_summary.py runs/e6_edge_640/jetson                     # JetPack 5.1.1 run
    python scripts/jetson_summary.py runs/e6_edge_640/jetson_jp6 runs/e6_edge_640/export/jp6

Sources: env_*.json, trtexec build logs (<engine dir>/*_build.log, default <run>/export),
bench_<prec>/<dataset>/*.json (TensorRT), bench_torch_fp16/<dataset>/*.json (PyTorch backend),
power_<prec>/*.json, sam3_<dtype>/sam3_<dtype>.json (SAM 3 on the device) and <out>/tegrastats.log
(the suite runs tegrastats itself on JetPack 6; `sudo tegrastats` on JetPack 5), averaged over the
PHASE windows written in status.log.
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

out = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/e6_edge_640/jetson")
exp = Path(sys.argv[2]) if len(sys.argv) > 2 else out.parent / "export"
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
for prec in ("fp16", "fp32", "torch_fp16"):
    for d in sorted((out / f"bench_{prec}").glob("*")):
        js = sorted(d.glob("bench_*.json"))
        if js:
            S["bench"].setdefault(prec, {})[d.name] = load(js[0])
S["sam3"] = {}
for f in sorted(out.glob("sam3_*/sam3_*.json")):
    r = load(f)
    if r:
        r.pop("frames", None)
        ag = f.parent / "agreement.json"      # sam3_agreement.py (A22): agreement with PyTorch bf16
        if ag.exists():
            r["agreement"] = load(ag)
        # key = PHASE name: sam3_bf16 (PyTorch, as before A22), sam3_trt_fp16, ...
        S["sam3"][r.get("tag") or f"sam3_{r.get('dtype', f.stem)}"] = r
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
          f"{env.get('cuda')}; torch {env.get('torch')}; transformers {env.get('transformers', '-')}; Python {env.get('python')}",
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
def fmt(v, f=".1f"):
    return "-" if v is None else format(v, f)


if S["bench"]:
    L += ["## Python runtime (scripts/bench_edge.py): network and full pipeline on real test images", "",
          "| precision | dataset | net mean / p95 (ms) | net FPS | pre / net / post / total (ms) | pipeline FPS | objects/frame | "
          "check vs ORT fp32: max abs diff [waste, contour, desc] | waste-mask IoU vs fp32 | torch peak (MB) | "
          "footprint, network only (MB) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
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
                     f"{ck.get('waste_mask_iou_engine_vs_fp32', float('nan')):.4f} | {r.get('cuda_peak_mem_mb', 0):.0f} | "
                     f"{fmt((r.get('memory_network') or {}).get('footprint_mb'), '.0f')} |")
    L.append("")
if S["power"]:
    L += ["## Power (tegrastats, mean over each phase; board input = VDD_IN)", "", "| phase | samples | rails (mW) | RAM max (MB) | GPU util (%) |",
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
    for key, r in S["sam3"].items():
        ld = S["power"].get(key, {}).get("mean_mW", {}).get("VDD_IN")
        if ld and r.get("fits") is True:
            L.append(f"\nSAM 3 {key[5:]}: board input {ld / 1e3:.2f} W"
                     + (f" ({(ld - idle) / 1e3:.2f} W above idle)" if idle else "")
                     + f", {ld * r['all']['total_ms'] / 1e6:.2f} J per frame (16 prompts, whole frame)")
    L.append("")
elif phases:
    L += ["## Power", "", "No `tegrastats.log` in this folder: power rails need root. Run "
          "`sudo tegrastats --interval 500 --logfile <this folder>/tegrastats.log` during the `power` step.", ""]
if S["sam3"]:
    L += ["## SAM 3 on the device (scripts/bench_sam3_edge.py, 16 prompts, as for its accuracy numbers)", "",
          "bf16 = PyTorch Sam3Model; trt_fp16 = TensorRT engines of scripts/sam3_parts.py (A22), stored FP32 text "
          "embeddings, same pre- and post-processing.", ""]
    for key, r in S["sam3"].items():
        dt = key[5:]
        m = r.get("memory") or {}
        if r.get("fits") is not True:
            L += [f"* {dt}: **does not run** ({r.get('fits')}; {r.get('stage', '')} {r.get('reason') or r.get('error', '')})",
                  f"  memory available before loading: {fmt(r.get('mem_available_before_load_mb'), '.0f')} MB; "
                  f"footprint reached: {fmt(m.get('footprint_mb'), '.0f')} MB", ""]
            continue
        size = (f"engines {r['engines_gb']:.2f} GB" if r.get("engines_gb") else
                f"{r.get('parameters_M', 0):.0f} M parameters, checkpoint {r.get('checkpoint_gb', 0):.2f} GB")
        L += [f"* {dt}: {size}, loaded in "
              f"{r.get('load_s', 0):.0f} s; footprint {fmt(m.get('footprint_mb'), '.0f')} MB (MemAvailable drop), torch peak "
              f"{fmt(m.get('torch_peak_allocated_mb'), '.0f')} MB, process peak RSS {fmt(m.get('process_peak_rss_mb'), '.0f')} MB", "",
              "| dataset | frames | pre / vision / decode (16) / post (ms) | total mean / p95 (ms) | FPS | one prompt (ms, FPS) | "
              "instances/frame |", "|---|---|---|---|---|---|---|"]
        for ds, q in list(r.get("per_dataset", {}).items()) + [("all", r.get("all", {}))]:
            L.append(f"| {ds} | {q['frames']} | {q['pre_ms']:.0f} / {q['vision_ms']:.0f} / {q['decode_ms']:.0f} / {q['post_ms']:.0f} | "
                     f"{q['total_ms']:.0f} / {q['total_ms_p95']:.0f} | {q['fps']:.2f} | {q['one_prompt_ms']:.0f}, "
                     f"{q['one_prompt_fps']:.2f} | {q['instances_per_frame']:.1f} |")
        ag = r.get("agreement")
        if ag:
            fl = ag.get("floor", {}).get("all", {})
            for run, x in ag.get("runs", {}).items():
                q, rule = x["all"], x.get("rule", {})
                L.append(f"\nAgreement with PyTorch bf16 on the same frames (union mask of the 16 prompts): IoU mean "
                         f"{q['iou_mean']:.4f}, min {q['iou_min']:.3f}, {q['frames_iou_below_0.9']} of {q['frames']} frames "
                         f"below 0.9; instances/frame {q['instances_run_mean']:.1f} vs {q['instances_ref_mean']:.1f}"
                         + (f". PyTorch FP32 vs bf16 (noise floor): {fl['iou_mean']:.4f}, min {fl['iou_min']:.3f}; rule "
                            f"(>= floor - {ag.get('margin', 0.02)}): **{'pass' if rule.get('passes') else 'FAIL'}**"
                            if fl else ""))
        L.append("")
else:
    L += ["Not measured here: SAM 3 on the device (JetPack 5.1: needs Python >= 3.10 / transformers 5.2; JetPack 6: "
          "see the sam3_* step in status.log)."]
(out / "summary.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
