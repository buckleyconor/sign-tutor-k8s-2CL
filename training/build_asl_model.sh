#!/usr/bin/env bash
# Produce the ASL model end-to-end: extract landmarks -> quality gate ->
# train -> export ONNX. Run inside the tutor-app container (has MediaPipe +
# torch). The TensorRT engine build is a separate, GPU-bound step
# (training/build_engine.sh), run where an sm_120 GPU is available.
#
#   bash training/build_asl_model.sh
set -euo pipefail

SRC="${1:-datasets/asl_dataset2/asl_alphabet_train/asl_alphabet_train}"
LANDMARKS="languages/asl/landmarks.csv"
CKPT_DIR="checkpoints/asl"

echo "==> 1/4 Extract landmarks from ${SRC}"
# Fresh CSV each run (extract appends).
rm -f "${LANDMARKS}"
python -m training.extract_landmarks \
    --src "${SRC}" \
    --dst "${LANDMARKS}" \
    --hands 1 --source-tag kaggle_asl

echo "==> 2/4 Data-quality gate"
python training/check_quality.py "${LANDMARKS}" 63 || true   # warnings are non-fatal

echo "==> 3/4 Train classifier"
python training/train_classifier.py \
    --dataset "${LANDMARKS}" \
    --epochs 50 \
    --checkpoint-dir "${CKPT_DIR}" \
    --csv-file "${CKPT_DIR}/train_log.csv"

echo "==> 4/4 Export ONNX"
python training/export_onnx.py \
    --checkpoint "${CKPT_DIR}/best.pt" \
    --output languages/asl/model.onnx

# Stage into the chart so the deploy is self-contained (Helm .Files.Get reads
# from inside the chart dir). Both copies are committed (we ship ASL pre-trained).
mkdir -p k8s/chart/files/asl
cp languages/asl/model.onnx k8s/chart/files/asl/model.onnx

echo "Done. ASL ONNX ready at languages/asl/model.onnx (and staged in k8s/chart/files/asl/)."
echo "Next: build the TensorRT engine on an sm_120 GPU:"
echo "  bash training/build_engine.sh languages/asl/model.onnx /models/asl_classifier/1/model.plan"
