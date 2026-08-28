#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! "$PYTHON_BIN" -c 'import cv2, timm, torch' >/dev/null 2>&1; then
  echo "PYTHON_BIN must provide OpenCV, timm, and PyTorch." >&2
  exit 1
fi

"$PYTHON_BIN" prepare_stereo_segmentation_dataset.py
"$PYTHON_BIN" train_rgb_segmenter.py \
  --dataset ../../datasets/training/workpiece-seg-stereo-v1 \
  --output results/rgb_segmenter_stereo_v1 \
  --rgbd-initialization results/rgb_segmenter_grouped_v3/best.pt \
  --batch-size 2 --workers 4
"$PYTHON_BIN" prepare_ablation_manifest.py --split val \
  --output inputs/grouped_v3_val.csv
"$PYTHON_BIN" prepare_ablation_manifest.py --split test \
  --output inputs/grouped_v3_test.csv
"$PYTHON_BIN" evaluate_ablation.py \
  --manifest inputs/grouped_v3_test.csv \
  --segmenter results/rgb_segmenter_stereo_v1/best.pt \
  --guidance-weight 2.0 \
  --output results/ablation_stereo_v1
