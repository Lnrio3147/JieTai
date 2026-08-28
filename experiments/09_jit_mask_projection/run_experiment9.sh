#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/uestc/mount_2T/uestc/.conda/envs/liteanystereo/bin/python}"
cd "$EXPERIMENT_DIR"

echo "[1/4] Cache frozen Exp8 Base predictions"
"$PYTHON_BIN" prepare_coarse_predictions.py

echo "[2/4] Train direct clean-mask projector"
if [[ -f results/clean_mask_projector/best.pt ]]; then
    echo "Existing projector checkpoint found; training skipped."
else
    "$PYTHON_BIN" train_clean_mask_projector.py
fi

echo "[3/4] Evaluate frozen 21-image test split"
"$PYTHON_BIN" evaluate.py

echo "[4/4] Complete"
echo "Results: $EXPERIMENT_DIR/results/comparison"
