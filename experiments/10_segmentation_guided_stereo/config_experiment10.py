"""Configuration for Experiment 10: segmentation-guided stereo."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "experiments/10_segmentation_guided_stereo"
RESULTS_DIR = EXPERIMENT_DIR / "results"
DATASET_DIR = ROOT / "datasets/training/workpiece-seg-grouped-v3"
RIGHT_MANUAL_DATASET = (
    ROOT / "datasets/evaluation/workpiece-right-manual-isat-v1"
)

LAS_ROOT = ROOT / "projects/LiteAnyStereo"
LAS_CHECKPOINT = LAS_ROOT / "checkpoints/LiteAnyStereo.pth"
RGBD_INITIALIZATION = (
    ROOT
    / "experiments/08_lightweight_rgbd/results/student_base_grouped_v3/best.pt"
)

SEGMENTER_RUN_DIR = RESULTS_DIR / "rgb_segmenter_grouped_v3"
SEGMENTER_CHECKPOINT = SEGMENTER_RUN_DIR / "best.pt"
ABLATION_DIR = RESULTS_DIR / "ablation_grouped_v3"

MODEL_NAME = "mobilenetv4_conv_small"
IMAGE_WIDTH = 576
IMAGE_HEIGHT = 1024
EVALUATION_WIDTH = 288
EVALUATION_HEIGHT = 512

SEED = 20260825
BATCH_SIZE = 4
EPOCHS = 40
PATIENCE = 8
WORKERS = 4
LEARNING_RATE = 8e-4
ENCODER_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
CATEGORY_BALANCE_POWER = 0.25
BOUNDARY_TOLERANCE = 2

BCE_WEIGHT = 0.35
TVERSKY_WEIGHT = 0.55
BOUNDARY_WEIGHT = 0.10
TVERSKY_ALPHA = 0.30
TVERSKY_BETA = 0.70

MASK_THRESHOLD = 0.50
THRESHOLD_CANDIDATES = tuple(value / 100.0 for value in range(5, 96, 2))
THRESHOLD_RECALL_FLOOR = 0.95
# Selected on the 11-scene FDJYP-3 validation subset.  Foundation Stereo is
# an engineering reference, not human dense ground truth; see README.md.
MASK_GUIDANCE_WEIGHT = 2.0
ROI_MARGIN = 48
ROI_MIN_FOREGROUND_PIXELS = 64
ROI_MIN_AREA_RATIO = 0.04
ROI_MAX_AREA_RATIO = 0.92
ROI_STRIDE = 32
LAS_MAX_DISPARITY = 192
