"""Configuration for Experiment 9: JiT-inspired clean-mask projection."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "experiments/09_jit_mask_projection"
EXP8_DIR = ROOT / "experiments/08_lightweight_rgbd"
DATASET_DIR = ROOT / "datasets/training/workpiece-seg-isat-v2"

RESULTS_DIR = EXPERIMENT_DIR / "results"
COARSE_DIR = RESULTS_DIR / "coarse_predictions"
RUN_DIR = RESULTS_DIR / "clean_mask_projector"
COMPARISON_DIR = RESULTS_DIR / "comparison"

EXP8_BASE_CHECKPOINT = EXP8_DIR / "results/student_base/best.pt"
EXP8_DISTILLED_CHECKPOINT = EXP8_DIR / "results/student_distilled/best.pt"
EXP8_COMPARISON_DIR = EXP8_DIR / "results/comparison"
EXP7_OVERFLOW_DIR = ROOT / "experiments/07_rgbd_fusion/results/rgbd_fusion_v1_geometry_overflow"

# Exp8 performs RGB-D inference at 1024 x 576. The projector works on the
# original annotated 512 x 288 grid, on which final metrics/post-processing run.
BASE_INPUT_WIDTH = 576
BASE_INPUT_HEIGHT = 1024
PROJECTOR_WIDTH = 288
PROJECTOR_HEIGHT = 512
PROJECTOR_INPUT_CHANNELS = 9
PROJECTOR_CHANNELS = (16, 24, 32, 48)
PATCH_SIZE = 4
BOTTLENECK_CHANNELS = 16

BATCH_SIZE = 8
EPOCHS = 50
PATIENCE = 10
WORKERS = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 20260824
CATEGORY_BALANCE_POWER = 0.25

CORRUPTION_PROBABILITY = 0.70
MAX_MORPH_RADIUS = 6
MAX_STRUCTURED_OPERATIONS = 3

BCE_WEIGHT = 0.35
TVERSKY_WEIGHT = 0.55
BOUNDARY_WEIGHT = 0.10
TVERSKY_ALPHA = 0.30
TVERSKY_BETA = 0.70
BOUNDARY_TOLERANCE = 2
THRESHOLD_CANDIDATES = tuple(round(value / 100.0, 2) for value in range(10, 91, 2))
THRESHOLD_RECALL_FLOOR = 0.98

MAX_EXTRA_PARAMS = 500_000
MAX_TOTAL_GFLOPS = 12.0
