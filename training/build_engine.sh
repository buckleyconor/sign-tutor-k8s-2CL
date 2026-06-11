#!/usr/bin/env bash
# Build a TensorRT engine from an ONNX model for the cluster GPU (sm_120).
#
# Engines are hardware-specific (sm_120 = RTX PRO 6000 Blackwell). The ONNX is
# the portable artefact; rebuild the engine on each non-Blackwell target.
#
# Usage: build_engine.sh <model.onnx> <model.plan>
set -euo pipefail

ONNX="${1:-model.onnx}"
PLAN="${2:-model.plan}"

trtexec \
    --onnx="${ONNX}" \
    --saveEngine="${PLAN}" \
    --minShapes=input:1x63 \
    --optShapes=input:32x63 \
    --maxShapes=input:64x63 \
    --useCudaGraph

# Note: no --fp16. For a ~12k-parameter MLP, FP16 shifts logits enough to change
# the top-1 class on many inputs; the latency difference is negligible (both
# sub-millisecond). Use FP32.
