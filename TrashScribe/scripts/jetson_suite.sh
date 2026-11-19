#!/usr/bin/env bash
# All on-device measurements of docs/JETSON.md, in sequence, resumable (a finished step leaves
# $OUT/.done_<step>; delete it to redo the step). Run it detached, not in the VS Code terminal:
#
#   systemd-run --user --unit=trashscribe-jetson --collect --working-directory=$PWD \
#     -p StandardOutput=append:$PWD/runs/e6_edge_640/jetson/suite.log \
#     -p StandardError=append:$PWD/runs/e6_edge_640/jetson/suite.log bash scripts/jetson_suite.sh
#
# Before the timed steps it waits (up to $WAIT_CLOCKS s) for `sudo jetson_clocks`. Power needs
# root: run `sudo tegrastats --interval 500 --logfile runs/e6_edge_640/jetson/tegrastats.log`
# alongside; jetson_summary.py cuts it into the phases recorded in status.log.
# Progress: bash scripts/jetson_progress.sh [--watch]
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:-runs/e6_edge_640}
EXP=$RUN/export
OUT=${OUT:-$RUN/jetson}
IMAGES=${IMAGES:-$HOME/data/waste/_edge_test}
WAIT_CLOCKS=${WAIT_CLOCKS:-5400}
MIN_MEM_MB=${MIN_MEM_MB:-3000}
WAIT_MEM=${WAIT_MEM:-900}
PY=${PYTHON:-$PWD/.venv/bin/python}
ONNX=$EXP/trashscribe_640.onnx
CACHE=$EXP/atlas_cache.pt
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p "$OUT"

status() { echo "$(date '+%F %T') $*" | tee -a "$OUT/status.log"; }
step() {          # step <name> <command...>: skipped if done, log in $OUT/<name>.log
  local n=$1; shift
  if [ -f "$OUT/.done_$n" ]; then status "skip  $n (done before)"; return 0; fi
  status "START $n"
  if "$@" > "$OUT/$n.log" 2>&1; then touch "$OUT/.done_$n"; status "DONE  $n"
  else status "FAIL  $n (exit $?, see $OUT/$n.log)"; FAILED=1; fi
}
gpu_locked() {
  local d=/sys/class/devfreq/17000000.ga10b
  [ "$(cat $d/min_freq)" = "$(cat $d/max_freq)" ]
}
FAILED=0
for f in "$ONNX" "$CACHE"; do [ -f "$f" ] || { status "MISSING $f (copy it from the workstation, docs/JETSON.md)"; exit 1; }; done
status "suite start (pid $$)"

step env "$PY" scripts/jetson_env.py --out "$OUT/env_start.json"

# Preflight. TensorRT picks tactics by timing them, so the build already needs locked clocks; and on the
# 8 GB shared memory it skips the fastest tactics of the big fused attention regions (they ask 1.4-1.7 GB)
# when the desktop holds the memory (first build, 18 Nov: 81 tactics skipped, 120 ms).
mem_avail() { awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo; }
if ! gpu_locked; then
  status "WAIT  GPU clocks not locked: run 'sudo jetson_clocks' (waiting up to ${WAIT_CLOCKS}s)"
  t=0; while ! gpu_locked && [ $t -lt "$WAIT_CLOCKS" ]; do sleep 30; t=$((t + 30)); done
  gpu_locked && status "clocks locked, continuing" || status "WARN  clocks still not locked: results flagged in summary"
fi
if [ "$(mem_avail)" -lt "$MIN_MEM_MB" ]; then
  status "WAIT  only $(mem_avail) MB available: close Firefox / other apps (waiting up to ${WAIT_MEM}s for ${MIN_MEM_MB} MB)"
  t=0; while [ "$(mem_avail)" -lt "$MIN_MEM_MB" ] && [ $t -lt "$WAIT_MEM" ]; do sleep 15; t=$((t + 15)); done
  status "memory available: $(mem_avail) MB, continuing"
fi
step env_bench "$PY" scripts/jetson_env.py --out "$OUT/env_bench.json"
step build_fp16 bash scripts/build_trt.sh "$ONNX" fp16
step build_fp32 bash scripts/build_trt.sh "$ONNX" fp32
for P in fp16 fp32; do
  n=$(grep -c "Skipping tactic" "$EXP/trashscribe_640_${P}_build.log" 2>/dev/null || true)
  [ "${n:-0}" -gt 0 ] && status "WARN  $P build skipped ${n} tactics for lack of memory"
done

for P in fp16 fp32; do
  ENG=$EXP/trashscribe_640_$P.engine
  [ -f "$ENG" ] || { status "FAIL  no engine $ENG, $P benchmarks skipped"; FAILED=1; continue; }
  for D in "$IMAGES"/*/; do
    D=${D%/}; ds=$(basename "$D")
    ls "$D"/*.jpg > /dev/null 2>&1 || continue
    step "bench_${P}_$ds" "$PY" scripts/bench_edge.py --engine "$ENG" --onnx "$ONNX" --cache "$CACHE" \
      --images "$D" --iters 300 --warmup 30 --check --out "$OUT/bench_$P/$ds"
  done
done

# sustained load for the power figure (phases marked in status.log; tegrastats run by the owner)
if [ ! -f "$OUT/.done_power" ]; then
  status "PHASE idle_begin"; sleep 60; status "PHASE idle_end"
  for P in fp16 fp32; do
    ENG=$EXP/trashscribe_640_$P.engine; [ -f "$ENG" ] || continue
    status "PHASE load_${P}_begin"
    "$PY" scripts/bench_edge.py --engine "$ENG" --iters 3000 --warmup 50 --out "$OUT/power_$P" \
      > "$OUT/power_$P.log" 2>&1 || status "FAIL  power $P (see $OUT/power_$P.log)"
    status "PHASE load_${P}_end"
    sleep 20
  done
  touch "$OUT/.done_power"; status "DONE  power"
fi

# qualitative output on real images (overlays + written reports): sanity check and paper figure material
for D in "$IMAGES"/*/; do
  D=${D%/}; ds=$(basename "$D")
  ls "$D"/*.jpg > /dev/null 2>&1 || continue
  step "infer_$ds" "$PY" scripts/infer.py --engine "$EXP/trashscribe_640_fp16.engine" --cache "$CACHE" \
    --source "$D" --out "$OUT/infer/$ds"
done

"$PY" scripts/jetson_summary.py "$OUT" > "$OUT/summary.log" 2>&1 && status "DONE  summary -> $OUT/summary.md" \
  || status "FAIL  summary (see $OUT/summary.log)"
status "suite end ($([ $FAILED = 0 ] && echo ok || echo 'with failures'))"
