#!/usr/bin/env python3
"""Single-image RGB + LiteAnyStereo disparity inference pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import config_experiment8 as config
from models.student_network import create_student
from utils.data import RGB_MEAN, RGB_STD, geometry_channels, robust_normalize_disparity
from utils.postprocess import mask_stats, refine_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--disparity", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=config.DISTILLED_RUN_DIR / "best.pt"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Foreground threshold; defaults to the validation-calibrated value",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def preprocess(image_bgr: np.ndarray, disparity: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(
        image,
        (config.IMAGE_WIDTH, config.IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32) / 255.0
    normalized, valid = robust_normalize_disparity(disparity)
    normalized = cv2.resize(
        normalized,
        (config.IMAGE_WIDTH, config.IMAGE_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )
    valid = cv2.resize(
        valid,
        (config.IMAGE_WIDTH, config.IMAGE_HEIGHT),
        interpolation=cv2.INTER_NEAREST,
    )
    rgb = ((image - RGB_MEAN) / RGB_STD).transpose(2, 0, 1)
    geometry = geometry_channels(normalized, valid).transpose(2, 0, 1)
    return (
        torch.from_numpy(rgb.astype(np.float32))[None],
        torch.from_numpy(geometry.astype(np.float32))[None],
    )


@torch.inference_mode()
def run_inference(
    rgb_path: Path,
    disparity_path: Path,
    checkpoint: Path,
    threshold: float | None,
    output: Path,
    no_amp: bool = False,
) -> dict:
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(rgb_path)
    disparity = np.load(disparity_path, allow_pickle=False).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not no_amp
    model = create_student(pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    rgb, geometry = preprocess(image, disparity)
    with torch.autocast(device_type=device.type, enabled=amp):
        logits, boundary_logits = model(rgb.to(device), geometry.to(device))
    probability = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
    boundary = torch.sigmoid(boundary_logits)[0, 0].float().cpu().numpy()
    probability = cv2.resize(
        probability, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    boundary = cv2.resize(
        boundary, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    if threshold is None:
        summary_path = config.COMPARISON_DIR / "summary.json"
        threshold = 0.5
        if summary_path.is_file():
            comparison = json.loads(summary_path.read_text(encoding="utf-8"))
            method = (
                "student_base"
                if checkpoint.resolve() == (config.BASE_RUN_DIR / "best.pt").resolve()
                else "student_distilled"
            )
            threshold = float(comparison["selected_thresholds"][method])
    mask, diagnostics = refine_prediction(
        probability,
        disparity,
        threshold,
        enable_topology_repair=config.TOPOLOGY_REPAIR,
        topology_smooth_sigma=config.TOPOLOGY_SMOOTH_SIGMA,
        topology_smooth_threshold=config.TOPOLOGY_SMOOTH_THRESHOLD,
        topology_envelope_min_added_fraction=(
            config.TOPOLOGY_ENVELOPE_MIN_ADDED_FRACTION
        ),
        topology_envelope_max_added_fraction=(
            config.TOPOLOGY_ENVELOPE_MAX_ADDED_FRACTION
        ),
        topology_envelope_closing_radius=(
            config.TOPOLOGY_ENVELOPE_CLOSING_RADIUS
        ),
    )
    stats = mask_stats(mask)
    output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output / "mask.png"), mask.astype(np.uint8) * 255)
    cv2.imwrite(
        str(output / "boundary.png"), np.clip(boundary * 255.0, 0, 255).astype(np.uint8)
    )
    np.save(output / "probability.npy", probability.astype(np.float16))
    result = {
        "rgb": str(rgb_path.resolve()),
        "disparity": str(disparity_path.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "threshold": threshold,
        **stats,
        "postprocess": diagnostics,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run_inference(
                args.rgb,
                args.disparity,
                args.checkpoint,
                args.threshold,
                args.output,
                args.no_amp,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
