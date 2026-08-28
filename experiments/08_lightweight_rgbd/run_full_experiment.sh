#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/uestc/mount_2T/uestc/.conda/envs/liteanystereo/bin/python}"

cd "$EXPERIMENT_DIR"

echo "[1/7] Checking environment and data"
"$PYTHON_BIN" - <<'PY'
import importlib
from pathlib import Path
import config_experiment8 as cfg

for module in ("torch", "timm", "cv2", "numpy", "pandas", "thop", "onnx"):
    importlib.import_module(module)
for path in (
    cfg.DATASET_DIR / "index/train.csv",
    cfg.DATASET_DIR / "index/val.csv",
    cfg.DATASET_DIR / "index/test.csv",
    cfg.TEACHER_CHECKPOINT,
):
    if not Path(path).exists():
        raise FileNotFoundError(path)
print("Environment/data check passed")
PY

echo "[2/7] Preparing frozen Exp7.2/Exp7.1 teacher masks"
"$PYTHON_BIN" prepare_teacher_targets.py

echo "[3/7] Training student without distillation"
if [[ -f results/student_base/best.pt ]]; then
    echo "Existing Base checkpoint found; training skipped."
else
    "$PYTHON_BIN" train_distillation.py --mode base
fi

echo "[4/7] Training distilled student"
if [[ -f results/student_distilled/best.pt ]]; then
    echo "Existing Distilled checkpoint found; training skipped."
else
    "$PYTHON_BIN" train_distillation.py --mode distilled
fi

echo "[5/7] Evaluating frozen 21-image comparison set"
"$PYTHON_BIN" evaluate.py

echo "[6/7] Exporting ONNX"
"$PYTHON_BIN" export_rknn.py --stage onnx

echo "[7/7] RKNN conversion (when rknn-toolkit2 is available)"
if "$PYTHON_BIN" -c 'from rknn.api import RKNN' >/dev/null 2>&1; then
    "$PYTHON_BIN" export_rknn.py --stage hybrid-all
else
    echo "rknn-toolkit2 is not installed; ONNX export completed, RKNN conversion skipped."
fi

echo "Experiment 8 complete. Results: $EXPERIMENT_DIR/results/comparison"
