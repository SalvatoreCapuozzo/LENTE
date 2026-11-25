#!/usr/bin/env bash
# TensorRT engines of SAM 3 (A22) from the graphs of export_sam3_onnx.py, with the flags of build_trt.sh.
#   bash scripts/build_sam3_trt.sh runs/e6_edge_640/export/jp6/sam3 [fp16] [vision|vision_<k>|decoder_b1|all]
# (existing engines are kept; FORCE=1 rebuilds)
# No graph surgery: the exporter already folds constants, and the TrashScribe-specific preparation of
# build_trt.sh (OneHot, hypernetwork outputs) does not apply. Run one build at a time on the 8 GB board.
set -euo pipefail
DIR=${1:?folder of export_sam3_onnx.py}
PREC=${2:-fp16}
WHICH=${3:-all}
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
WS=${WS:-1024}   # MiB of tactic workspace: 2048 (build_trt.sh) got the 454 M encoder build OOM-killed on 24 Nov
FLAGS=""
[ "$PREC" = fp16 ] && FLAGS="--fp16"
"$TRTEXEC" --help 2>&1 | grep -q builderOptimizationLevel && FLAGS="$FLAGS --builderOptimizationLevel=3"
for ONNX in "$DIR"/sam3_vision_[0-9].onnx "$DIR"/sam3_decoder_b1.onnx; do
  B=${ONNX%.onnx}; P=$(basename "$B"); P=${P#sam3_}       # vision_0 .. vision_3, decoder_b1
  [ "$WHICH" = all ] || [ "$WHICH" = "$P" ] || [ "$WHICH" = "${P%_[0-9]}" ] || continue
  [ -f "${B}_${PREC}.engine" ] && [ "${FORCE:-0}" != 1 ] && { echo "== ${B}_${PREC}.engine exists"; continue; }
  echo "== $B.onnx -> ${B}_${PREC}.engine ($(date '+%F %T'))"
  "$TRTEXEC" --onnx="$B.onnx" --saveEngine="${B}_${PREC}.engine" $FLAGS \
    --memPoolSize=workspace:$WS \
    --profilingVerbosity=detailed --exportLayerInfo="${B}_${PREC}_layers.json" \
    --exportProfile="${B}_${PREC}_profile.json" --separateProfileRun \
    --useCudaGraph --warmUp=500 --iterations=50 --avgRuns=10 2>&1 | tee "${B}_${PREC}_build.log"
done
