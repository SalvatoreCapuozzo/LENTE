#!/usr/bin/env bash
# Build TensorRT engines ON THE TARGET DEVICE (Jetson Orin Nano 8GB, JetPack 6.x).
# Engines are tied to the GPU, the TensorRT version and the JetPack: never copy
# an engine between machines, copy the .onnx and rebuild.
#
#   bash scripts/build_trt.sh runs/<run>/export/trashscribe_640.onnx [fp16|fp32|best]
#
# Before benchmarking, put the board in its max-performance mode:
#   sudo nvpmodel -m 0 && sudo jetson_clocks          (MAXN SUPER on JetPack 6.2)
set -euo pipefail
ONNX=${1:?onnx path}
PREC=${2:-fp16}
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
BASE=${ONNX%.onnx}
SIMP=${BASE}_sim.onnx

# 1) simplify shapes (removes the dynamic-shape plumbing of the HF graph)
if python3 -c "import onnxsim" 2>/dev/null; then
  python3 -m onnxsim "$ONNX" "$SIMP" && ONNX=$SIMP
else
  echo "onnxsim not installed (pip install onnxsim): building from the raw graph"
fi

FLAGS=""
case $PREC in
  fp16) FLAGS="--fp16" ;;
  fp32) FLAGS="" ;;
  best) FLAGS="--best" ;;        # lets TRT pick int8/fp16 per layer (needs calibration for accuracy)
esac

# 2) build. 8 GB is shared CPU/GPU memory on the Orin Nano: cap the workspace.
"$TRTEXEC" --onnx="$ONNX" --saveEngine="${BASE}_${PREC}.engine" $FLAGS \
  --memPoolSize=workspace:2048 --builderOptimizationLevel=3 \
  --profilingVerbosity=detailed --exportLayerInfo="${BASE}_${PREC}_layers.json" \
  --exportProfile="${BASE}_${PREC}_profile.json" --separateProfileRun \
  --useCudaGraph --warmUp=500 --iterations=200 --avgRuns=50 2>&1 | tee "${BASE}_${PREC}_build.log"

echo "engine: ${BASE}_${PREC}.engine"
echo "next:   python3 scripts/bench_edge.py --engine ${BASE}_${PREC}.engine --onnx ${1} --cache $(dirname "$1")/atlas_cache.pt"
# FP16 note: NanoSAM reports FP16 errors on some SAM-family encoders; bench_edge.py
# --check compares this engine against ONNX Runtime FP32 and flags any drift.
