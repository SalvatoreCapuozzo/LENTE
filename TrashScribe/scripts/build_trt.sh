#!/usr/bin/env bash
# Build TensorRT engines ON THE TARGET DEVICE (Jetson Orin Nano 8GB).
# Works with JetPack 6.x (TensorRT 10) and JetPack 5.1.x (TensorRT 8.5: the graph is first
# rewritten by scripts/onnx_for_trt85.py, which decomposes LayerNormalization and folds the
# shape plumbing of the HF mask decoder; outputs checked against ONNX Runtime).
# Engines are tied to the GPU, the TensorRT version and the JetPack: never copy
# an engine between machines, copy the .onnx and rebuild.
#
#   bash scripts/build_trt.sh runs/<run>/export/trashscribe_640.onnx [fp16|fp32|best]
#
# Python: $PYTHON, else the repo's .venv, else python3 (needs tensorrt + onnx).
# Before benchmarking, put the board in its max-performance mode:
#   sudo nvpmodel -m 0 && sudo jetson_clocks   (15 W = MAXN on JetPack 5; MAXN SUPER on JetPack 6.2)
set -euo pipefail
ONNX=${1:?onnx path}
PREC=${2:-fp16}
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -n "${PYTHON:-}" ]; then PY=$PYTHON
elif [ -x "$HERE/../.venv/bin/python" ]; then PY="$HERE/../.venv/bin/python"
else PY=python3; fi
BASE=${ONNX%.onnx}

TRT_VER=$("$PY" -c "import tensorrt; print(tensorrt.__version__)" 2>/dev/null || echo "?")
echo "TensorRT $TRT_VER, python $PY"

if [ "$TRT_VER" != "?" ] && "$PY" -c "import sys; v=tuple(map(int,'$TRT_VER'.split('.')[:2])); sys.exit(0 if v < (8,6) else 1)"; then
  # 1a) TensorRT < 8.6: decompose LayerNormalization + constant-fold (also does what onnxsim would)
  PREP=${BASE}_trt85.onnx
  if [ ! -f "$PREP" ] || [ "$PREP" -ot "$ONNX" ]; then
    "$PY" "$HERE/onnx_for_trt85.py" "$ONNX" "$PREP"
  fi
  ONNX=$PREP
elif "$PY" -c "import onnxsim" 2>/dev/null; then
  # 1b) simplify shapes (removes the dynamic-shape plumbing of the HF graph)
  "$PY" -m onnxsim "$ONNX" "${BASE}_sim.onnx" && ONNX=${BASE}_sim.onnx
else
  echo "onnxsim not installed (pip install onnxsim): building from the raw graph"
fi

FLAGS=""
case $PREC in
  fp16) FLAGS="--fp16" ;;
  fp32) FLAGS="" ;;
  best) FLAGS="--best" ;;        # lets TRT pick int8/fp16 per layer (needs calibration for accuracy)
esac
# --builderOptimizationLevel exists from TensorRT 8.6 on
if "$TRTEXEC" --help 2>&1 | grep -q builderOptimizationLevel; then FLAGS="$FLAGS --builderOptimizationLevel=3"; fi

# 2) build. 8 GB is shared CPU/GPU memory on the Orin Nano: cap the workspace.
"$TRTEXEC" --onnx="$ONNX" --saveEngine="${BASE}_${PREC}.engine" $FLAGS \
  --memPoolSize=workspace:2048 \
  --profilingVerbosity=detailed --exportLayerInfo="${BASE}_${PREC}_layers.json" \
  --exportProfile="${BASE}_${PREC}_profile.json" --separateProfileRun \
  --useCudaGraph --warmUp=500 --iterations=200 --avgRuns=50 2>&1 | tee "${BASE}_${PREC}_build.log"

echo "engine: ${BASE}_${PREC}.engine"
echo "next:   $PY scripts/bench_edge.py --engine ${BASE}_${PREC}.engine --onnx ${1} --cache $(dirname "$1")/atlas_cache.pt"
# FP16 note: NanoSAM reports FP16 errors on some SAM-family encoders; bench_edge.py
# --check compares this engine against ONNX Runtime FP32 and flags any drift.
