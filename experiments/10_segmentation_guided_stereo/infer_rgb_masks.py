#!/usr/bin/env python3
"""Predict left/right foreground probabilities before stereo inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import config_experiment10 as config
from utils.segmentation import RGBSegmenterPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=config.SEGMENTER_CHECKPOINT)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    left = cv2.imread(str(args.left), cv2.IMREAD_COLOR)
    right = cv2.imread(str(args.right), cv2.IMREAD_COLOR)
    if left is None:
        raise FileNotFoundError(args.left)
    if right is None:
        raise FileNotFoundError(args.right)
    predictor = RGBSegmenterPredictor(
        args.checkpoint,
        config.IMAGE_WIDTH,
        config.IMAGE_HEIGHT,
        torch.device(args.device),
        args.no_amp,
    )
    left_probability, right_probability, left_boundary, right_boundary = (
        predictor.predict_pair(left, right)
    )
    threshold = predictor.threshold if args.threshold is None else args.threshold
    args.output.mkdir(parents=True, exist_ok=True)
    for side, probability, boundary in (
        ("left", left_probability, left_boundary),
        ("right", right_probability, right_boundary),
    ):
        np.save(args.output / f"{side}_probability.npy", probability.astype(np.float16))
        cv2.imwrite(
            str(args.output / f"{side}_mask.png"),
            (probability >= threshold).astype(np.uint8) * 255,
        )
        cv2.imwrite(
            str(args.output / f"{side}_boundary.png"),
            np.clip(boundary * 255.0, 0, 255).astype(np.uint8),
        )
    summary = {
        "left": str(args.left.resolve()),
        "right": str(args.right.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "threshold": float(threshold),
        "left_foreground_fraction": float((left_probability >= threshold).mean()),
        "right_foreground_fraction": float((right_probability >= threshold).mean()),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
