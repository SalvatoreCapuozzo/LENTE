#!/usr/bin/env bash
# All on-device measurements of docs/JETSON.md, in sequence, resumable (a finished step leaves
# $OUT/.done_<step>; delete it to redo the step). Run it detached, not in the VS Code terminal:
#
#   systemd-run --user --unit=trashscribe-jetson --collect --working-directory=$PWD \
#     --setenv=OUT=runs/e6_edge_640/jetson_jp6 --setenv=ENG_DIR=runs/e6_edge_640/export/jp6 \
#     -p StandardOutput=append:$PWD/runs/e6_edge_640/jetson_jp6/suite.log \
#     -p StandardError=append:$PWD/runs/e6_edge_640/jetson_jp6/suite.log bash scripts/jetson_suite.sh
#
# Before the timed steps it waits (up to $WAIT_CLOCKS s) for `sudo jetson_clocks`. Power: on JetPack 6
# the INA3221 rails are readable without root, so the suite runs `tegrastats` itself into
# $OUT/tegrastats.log (TEGRASTATS=0 to leave it to a `sudo tegrastats` started by the owner, as on
# JetPack 5); jetson_summary.py cuts it into the PHASE windows recorded in status.log.
# SAM 3 (last step) needs the gated facebook/sam3 weights in the HF cache; the suite waits for them
# (up to $WAIT_SAM3 s) and for $MIN_MEM_SAM3_MB MB of free memory. SAM3_FP32=1 also tries FP32.
# SAM3_TRT=1: SAM 3 in TensorRT FP16 (A22): export, engines, timing and the agreement rule (about 1 h the first time,
# most of it the PyTorch FP32 reference masks).
# Progress: OUT=<out> bash scripts/jetson_progress.sh [--watch]
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:-runs/e6_edge_640}
EXP=$RUN/export
OUT=${OUT:-$RUN/jetson}
# engines and their trtexec logs: one folder per JetPack so the engines of another TensorRT stay put
# (an engine only runs on the TensorRT that built it). Default: next to the ONNX, as before.
ENG_DIR=${ENG_DIR:-$EXP}
IMAGES=${IMAGES:-$HOME/data/waste/_edge_test}
WAIT_CLOCKS=${WAIT_CLOCKS:-5400}
MIN_MEM_MB=${MIN_MEM_MB:-3000}
WAIT_MEM=${WAIT_MEM:-900}
WAIT_SAM3=${WAIT_SAM3:-21600}
MIN_MEM_SAM3_MB=${MIN_MEM_SAM3_MB:-4000}
TEGRASTATS=${TEGRASTATS:-1}
PY=${PYTHON:-$PWD/.venv/bin/python}
ONNX=$EXP/trashscribe_640.onnx
CACHE=$EXP/atlas_cache.pt
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p "$OUT" "$ENG_DIR"

status() { echo "$(date '+%F %T') $*" | tee -a "$OUT/status.log"; }
step() {          # step <name> <command...>: skipped if done, log in $OUT/<name>.log
  local n=$1; shift
  if [ -f "$OUT/.done_$n" ]; then status "skip  $n (done before)"; return 0; fi
  status "START $n"
  if "$@" > "$OUT/$n.log" 2>&1; then touch "$OUT/.done_$n"; status "DONE  $n"
  else status "FAIL  $n (exit $?, see $OUT/$n.log)"; FAILED=1; fi
}
# GPU devfreq node: 17000000.ga10b on JetPack 5, 17000000.gpu on JetPack 6 (with a missing node the old
# test compared two empty strings and reported locked clocks)
GPU_DF=$(ls -d /sys/class/devfreq/17000000.gpu /sys/class/devfreq/17000000.ga10b 2>/dev/null | head -1)
gpu_locked() {
  [ -n "$GPU_DF" ] && [ -n "$(cat $GPU_DF/min_freq)" ] && [ "$(cat $GPU_DF/min_freq)" = "$(cat $GPU_DF/max_freq)" ]
}
mem_avail() { awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo; }
wait_mem() {      # wait_mem <MB> <max s>
  if [ "$(mem_avail)" -lt "$1" ]; then
    status "WAIT  only $(mem_avail) MB available: close Firefox / other apps (waiting up to ${2}s for $1 MB)"
    local t=0; while [ "$(mem_avail)" -lt "$1" ] && [ $t -lt "$2" ]; do sleep 15; t=$((t + 15)); done
    status "memory available: $(mem_avail) MB, continuing"
  fi
}
FAILED=0
for f in "$ONNX" "$CACHE"; do [ -f "$f" ] || { status "MISSING $f (copy it from the workstation, docs/JETSON.md)"; exit 1; }; done
# the engine build reads the graph from ENG_DIR so that build_trt.sh writes engines and logs there
BUILD_ONNX=$ENG_DIR/trashscribe_640.onnx
if [ "$ENG_DIR" != "$EXP" ] && [ ! -e "$BUILD_ONNX" ]; then ln -s "$(realpath "$ONNX")" "$BUILD_ONNX"; fi
status "suite start (pid $$; out $OUT; engines $ENG_DIR)"

TS_PID=
if [ "$TEGRASTATS" = 1 ] && command -v tegrastats > /dev/null; then
  # (not `tegrastats | head | grep -q`: with pipefail the SIGPIPE'd tegrastats fails the whole test)
  ts_line=$(timeout 3 tegrastats --interval 1000 2>/dev/null | head -1)
  if [[ "$ts_line" == *VDD_IN* ]]; then
    tegrastats --interval 500 --logfile "$OUT/tegrastats.log" & TS_PID=$!
    status "tegrastats (user, power rails readable) -> $OUT/tegrastats.log (pid $TS_PID)"
  else
    status "WARN  tegrastats shows no power rails without root: run 'sudo tegrastats --interval 500 --logfile $OUT/tegrastats.log'"
  fi
fi
trap '[ -n "$TS_PID" ] && kill $TS_PID 2>/dev/null' EXIT

step env "$PY" scripts/jetson_env.py --out "$OUT/env_start.json"

# Preflight. TensorRT picks tactics by timing them, so the build already needs locked clocks; and on the
# 8 GB shared memory it skips the fastest tactics of the big fused attention regions (they ask 1.4-1.7 GB)
# when the desktop holds the memory (first build, 18 Nov: 81 tactics skipped, 120 ms).
if ! gpu_locked; then
  status "WAIT  GPU clocks not locked: run 'sudo jetson_clocks' (waiting up to ${WAIT_CLOCKS}s)"
  t=0; while ! gpu_locked && [ $t -lt "$WAIT_CLOCKS" ]; do sleep 30; t=$((t + 30)); done
  gpu_locked && status "clocks locked, continuing" || status "WARN  clocks still not locked: results flagged in summary"
fi
wait_mem "$MIN_MEM_MB" "$WAIT_MEM"
step env_bench "$PY" scripts/jetson_env.py --out "$OUT/env_bench.json"
step build_fp16 bash scripts/build_trt.sh "$BUILD_ONNX" fp16
step build_fp32 bash scripts/build_trt.sh "$BUILD_ONNX" fp32
for P in fp16 fp32; do
  n=$(grep -c "Skipping tactic" "$ENG_DIR/trashscribe_640_${P}_build.log" 2>/dev/null || true)
  [ "${n:-0}" -gt 0 ] && status "WARN  $P build skipped ${n} tactics for lack of memory"
done

for P in fp16 fp32; do
  ENG=$ENG_DIR/trashscribe_640_$P.engine
  [ -f "$ENG" ] || { status "FAIL  no engine $ENG, $P benchmarks skipped"; FAILED=1; continue; }
  for D in "$IMAGES"/*/; do
    D=${D%/}; ds=$(basename "$D")
    ls "$D"/*.jpg > /dev/null 2>&1 || continue
    step "bench_${P}_$ds" "$PY" scripts/bench_edge.py --engine "$ENG" --onnx "$ONNX" --cache "$CACHE" \
      --images "$D" --iters 300 --warmup 30 --check --out "$OUT/bench_$P/$ds"
  done
done

# the same model in PyTorch FP16 (no TensorRT): the like-for-like reference for SAM 3, which runs in PyTorch.
# Needs the HF caches of facebook/sam2.1-hiera-tiny and openai/clip-vit-base-patch16 (text cache) or skips.
if "$PY" -c "from huggingface_hub import snapshot_download as s; s('facebook/sam2.1-hiera-tiny')" > /dev/null 2>&1; then
  for D in "$IMAGES"/*/; do
    D=${D%/}; ds=$(basename "$D")
    ls "$D"/*.jpg > /dev/null 2>&1 || continue
    step "bench_torch16_$ds" "$PY" scripts/bench_edge.py --run "$RUN" --half --images "$D" --iters 300 --warmup 30 \
      --out "$OUT/bench_torch_fp16/$ds"
  done
else
  status "skip  PyTorch backend (facebook/sam2.1-hiera-tiny not in the HF cache)"
fi

# sustained load for the power figure (phases marked in status.log)
if [ ! -f "$OUT/.done_power" ]; then
  status "PHASE idle_begin"; sleep 60; status "PHASE idle_end"
  for P in fp16 fp32; do
    ENG=$ENG_DIR/trashscribe_640_$P.engine; [ -f "$ENG" ] || continue
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
  step "infer_$ds" "$PY" scripts/infer.py --engine "$ENG_DIR/trashscribe_640_fp16.engine" --cache "$CACHE" \
    --source "$D" --out "$OUT/infer/$ds"
done

# SAM 3 on the same device, run as for its accuracy numbers (bf16, 16 prompts). Last: it needs the gated
# weights (hf auth login + hf download facebook/sam3, or the workstation's HF cache) and ~3.5 GB.
sam3_cached() { "$PY" -c "
import glob, sys
from huggingface_hub import snapshot_download as s
p = s('facebook/sam3', allow_patterns=['*.json', '*.safetensors', '*.txt'])  # sam3.pt (Meta's own checkpoint) not needed
sys.exit(0 if glob.glob(p + '/*.safetensors') else 1)" > /dev/null 2>&1; }
if [ ! -f "$OUT/.done_sam3_bf16" ]; then
  if ! sam3_cached; then
    status "WAIT  SAM 3 weights not in the HF cache: 'hf auth login' then 'hf download facebook/sam3' (waiting up to ${WAIT_SAM3}s)"
    t=0; while ! sam3_cached && [ $t -lt "$WAIT_SAM3" ]; do sleep 60; t=$((t + 60)); done
  fi
  if sam3_cached; then
    gpu_locked || status "WARN  clocks not locked for SAM 3: results flagged in summary"
    wait_mem "$MIN_MEM_SAM3_MB" "$WAIT_MEM"
    "$PY" scripts/jetson_env.py --out "$OUT/env_sam3.json" > /dev/null 2>&1
    sleep 20   # let the board settle after the last step (power baseline)
    step sam3_bf16 "$PY" scripts/bench_sam3_edge.py --images "$IMAGES" --dtype bf16 --out "$OUT/sam3_bf16" \
      --status "$OUT/status.log"
  else
    status "FAIL  SAM 3 weights never appeared in the HF cache: SAM 3 not measured"; FAILED=1
  fi
fi
if [ "${SAM3_FP32:-0}" = 1 ] && sam3_cached; then
  wait_mem 6000 "$WAIT_MEM"
  step sam3_fp32 "$PY" scripts/bench_sam3_edge.py --images "$IMAGES" --dtype fp32 --n 10 --out "$OUT/sam3_fp32" \
    --status "$OUT/status.log"
fi

# SAM 3 in TensorRT (A22, SAM3_TRT=1): export the image encoder and the per-prompt detector/mask decoder,
# build FP16 engines, time them (PHASE sam3_trt_fp16), then the agreement rule fixed in attempts/ATTEMPTS.md:
# union masks of PyTorch bf16 (reference), of the same parts in PyTorch FP32 (noise floor) and of the engines.
# The mask runs write no PHASE lines and live in agreement/ (not sam3_*/), so the timings above stay untouched.
if [ "${SAM3_TRT:-0}" = 1 ] && sam3_cached; then
  S3=$ENG_DIR/sam3
  AG=$OUT/agreement
  wait_mem 3000 "$WAIT_MEM"
  [ -f "$S3/sam3_vision_3.onnx" ] && [ -f "$S3/sam3_decoder_b1.onnx" ] && [ -f "$S3/text_objects16.pt" ] \
    || step sam3_export "$PY" scripts/export_sam3_onnx.py --out "$S3"
  # one trtexec at a time: each build takes what memory is left (MemAvailable fell to 0.3 GB on 24 Nov) and skips
  # the tactics that need more (1.76 GB); existing engines are kept
  step sam3_build bash scripts/build_sam3_trt.sh "$S3" fp16 all
  wait_mem 3000 "$WAIT_MEM"
  gpu_locked || status "WARN  clocks not locked for SAM 3 TensorRT: results flagged in summary"
  "$PY" scripts/jetson_env.py --out "$OUT/env_sam3_trt.json" > /dev/null 2>&1
  sleep 20   # let the board settle (power baseline)
  step sam3_trt_fp16 "$PY" scripts/bench_sam3_edge.py --images "$IMAGES" --backend trt --dtype fp16 --engines "$S3" \
    --out "$OUT/sam3_trt_fp16" --status "$OUT/status.log"
  step sam3_masks_trt_fp16 "$PY" scripts/bench_sam3_edge.py --images "$IMAGES" --backend trt --dtype fp16 \
    --engines "$S3" --warmup 1 --save-masks --out "$AG/trt_fp16"
  wait_mem "$MIN_MEM_SAM3_MB" "$WAIT_MEM"
  step sam3_masks_bf16 "$PY" scripts/bench_sam3_edge.py --images "$IMAGES" --dtype bf16 --warmup 1 --save-masks \
    --out "$AG/bf16"
  step sam3_masks_parts_fp32 "$PY" scripts/bench_sam3_edge.py --images "$IMAGES" --backend parts --dtype fp32 \
    --engines "$S3" --warmup 1 --save-masks --out "$AG/parts_fp32"
  step sam3_agreement "$PY" scripts/sam3_agreement.py --ref "$AG/bf16/sam3_bf16_masks.npz" \
    --floor "$AG/parts_fp32/sam3_parts_fp32_masks.npz" --run "$AG/trt_fp16/sam3_trt_fp16_masks.npz" \
    --out "$OUT/sam3_trt_fp16/agreement.json"
fi

"$PY" scripts/jetson_summary.py "$OUT" "$ENG_DIR" > "$OUT/summary.log" 2>&1 && status "DONE  summary -> $OUT/summary.md" \
  || status "FAIL  summary (see $OUT/summary.log)"
status "suite end ($([ $FAILED = 0 ] && echo ok || echo 'with failures'))"
