#!/usr/bin/env python3
"""Compare RK3588 outputs with the saved FP32 Experiment 1-4 reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-disparity", type=Path, required=True)
    parser.add_argument("--candidate-disparity", type=Path, required=True)
    parser.add_argument("--reference-probability", type=Path)
    parser.add_argument("--candidate-probability", type=Path)
    parser.add_argument("--reference-mask", type=Path)
    parser.add_argument("--candidate-mask", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_npy(path: Path) -> np.ndarray:
    value = np.load(path.expanduser().resolve(), allow_pickle=False)
    return np.asarray(value)


def mask_iou(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    if reference.shape != candidate.shape:
        raise ValueError(f"Mask shape mismatch: {reference.shape} vs {candidate.shape}")
    union = np.logical_or(reference, candidate).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(reference, candidate).sum() / union)


def disparity_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    reference = np.squeeze(np.asarray(reference, dtype=np.float32))
    candidate = np.squeeze(np.asarray(candidate, dtype=np.float32))
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Disparity shape mismatch: {reference.shape} vs {candidate.shape}"
        )
    valid = np.isfinite(reference) & np.isfinite(candidate)
    if not valid.any():
        raise ValueError("No jointly finite disparity pixels")
    difference = np.abs(reference[valid] - candidate[valid])
    return {
        "joint_finite_pixels": int(valid.sum()),
        "mae_px": float(difference.mean()),
        "rmse_px": float(np.sqrt(np.mean(np.square(difference)))),
        "p95_abs_px": float(np.percentile(difference, 95)),
        "max_abs_px": float(difference.max()),
        "bad_1px_fraction": float(np.mean(difference > 1.0)),
        "bad_3px_fraction": float(np.mean(difference > 3.0)),
    }


def probability_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    reference = np.squeeze(np.asarray(reference, dtype=np.float32))
    candidate = np.squeeze(np.asarray(candidate, dtype=np.float32))
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Probability shape mismatch: {reference.shape} vs {candidate.shape}"
        )
    difference = np.abs(reference - candidate)
    return {
        "mae": float(difference.mean()),
        "p95_abs": float(np.percentile(difference, 95)),
        "max_abs": float(difference.max()),
        "threshold_0_5_mask_iou": mask_iou(reference >= 0.5, candidate >= 0.5),
        "threshold_0_5_changed_fraction": float(
            np.mean((reference >= 0.5) != (candidate >= 0.5))
        ),
    }


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path.expanduser().resolve()), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask > 0


def paired(first, second, label: str) -> bool:
    if (first is None) != (second is None):
        raise ValueError(f"Both reference and candidate {label} paths are required")
    return first is not None


def main() -> None:
    args = parse_args()
    report = {
        "disparity": disparity_metrics(
            load_npy(args.reference_disparity), load_npy(args.candidate_disparity)
        )
    }
    if paired(args.reference_probability, args.candidate_probability, "probability"):
        report["probability"] = probability_metrics(
            load_npy(args.reference_probability),
            load_npy(args.candidate_probability),
        )
    if paired(args.reference_mask, args.candidate_mask, "mask"):
        reference_mask = load_mask(args.reference_mask)
        candidate_mask = load_mask(args.candidate_mask)
        report["refined_mask"] = {
            "iou": mask_iou(reference_mask, candidate_mask),
            "changed_pixels": int(np.sum(reference_mask != candidate_mask)),
            "changed_fraction": float(np.mean(reference_mask != candidate_mask)),
        }

    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
