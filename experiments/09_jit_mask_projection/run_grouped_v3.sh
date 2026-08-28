#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/uestc/mount_2T/uestc/.conda/envs/liteanystereo/bin/python}"
DATASET="$PROJECT_ROOT/datasets/training/workpiece-seg-grouped-v3"
BASE_CHECKPOINT="$PROJECT_ROOT/experiments/08_lightweight_rgbd/results/student_base_grouped_v3/best.pt"
COARSE_DIR="$EXPERIMENT_DIR/results/coarse_predictions_grouped_v3"
RUN_DIR="$EXPERIMENT_DIR/results/clean_mask_projector_grouped_v3"

cd "$EXPERIMENT_DIR"

echo "[1/4] Cache 317 frozen V3 Base predictions"
"$PYTHON_BIN" prepare_coarse_predictions.py \
    --dataset "$DATASET" \
    --checkpoint "$BASE_CHECKPOINT" \
    --output "$COARSE_DIR" \
    --batch-size 2 \
    --workers 4

echo "[2/4] Train grouped V3 clean-mask projector"
if [[ -f "$RUN_DIR/best.pt" ]]; then
    echo "Existing grouped V3 checkpoint found; training skipped."
else
    "$PYTHON_BIN" train_clean_mask_projector.py \
        --dataset "$DATASET" \
        --coarse-dir "$COARSE_DIR" \
        --output "$RUN_DIR" \
        --batch-size 8 \
        --epochs 50 \
        --patience 10 \
        --workers 4
fi

echo "[3/4] Evaluate the frozen 46-image V3 test split"
"$PYTHON_BIN" evaluate_grouped_v3.py \
    --dataset "$DATASET" \
    --coarse-dir "$COARSE_DIR" \
    --checkpoint "$RUN_DIR/best.pt"

echo "[4/4] Complete"
echo "Results: $EXPERIMENT_DIR/results/comparison_grouped_v3"
