#!/usr/bin/env python3
"""Evaluate baseline, post-mask, ROI, and guided stereo on one manifest."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

import config_experiment10 as config
from guided_stereo import colorize_disparity, infer_las, load_las, run_method
from guided_stereo import sample_right_probability
from utils.metrics import (
    add_boundary_metrics,
    boundary_counts,
    confusion_counts,
    metrics_from_confusion,
)
from utils.segmentation import RGBSegmenterPredictor


METHODS = ("baseline", "post_mask", "roi", "guided")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=config.EXPERIMENT_DIR / "inputs/grouped_v3_test.csv",
    )
    parser.add_argument("--segmenter", type=Path, default=config.SEGMENTER_CHECKPOINT)
    parser.add_argument("--las-checkpoint", type=Path, default=config.LAS_CHECKPOINT)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--guidance-weight", type=float, default=config.MASK_GUIDANCE_WEIGHT)
    parser.add_argument("--roi-margin", type=int, default=config.ROI_MARGIN)
    parser.add_argument("--max-disparity", type=int, default=config.LAS_MAX_DISPARITY)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--category", action="append", default=None)
    parser.add_argument(
        "--method",
        action="append",
        choices=METHODS,
        default=None,
        help="repeatable; defaults to all four methods",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-arrays", action="store_true")
    parser.add_argument("--output", type=Path, default=config.ABLATION_DIR)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def crop_from_text(value: str) -> tuple[slice, slice] | None:
    if not value:
        return None
    y0, y1, x0, x1 = (int(item) for item in value.split(","))
    return slice(y0, y1), slice(x0, x1)


def reference_metrics(
    disparity: np.ndarray,
    reference_path: str,
    crop_text: str,
    subject_mask: np.ndarray,
) -> dict:
    if not reference_path:
        return {}
    reference = np.load(reference_path, allow_pickle=False).astype(np.float32)
    crop = crop_from_text(crop_text)
    prediction = disparity[crop] if crop is not None else disparity
    mask = subject_mask[crop] if crop is not None else subject_mask
    if prediction.shape != reference.shape:
        raise ValueError(
            f"Reference/prediction shape mismatch: {reference.shape} vs {prediction.shape}"
        )
    valid = (
        np.isfinite(reference)
        & (reference > 0)
        & np.isfinite(prediction)
        & (prediction > 0)
        & mask
    )
    if not np.any(valid):
        return {"reference_valid_pixels": 0}
    error = np.abs(prediction[valid] - reference[valid])
    return {
        "reference_valid_pixels": int(valid.sum()),
        "reference_epe": float(error.mean()),
        "reference_bad_1": float((error > 1.0).mean()),
        "reference_bad_2": float((error > 2.0).mean()),
        "reference_bad_3": float((error > 3.0).mean()),
    }


def right_segmentation_metrics(
    right_gt: np.ndarray, right_prediction: np.ndarray
) -> dict:
    confusion = confusion_counts(right_gt, right_prediction)
    boundary = boundary_counts(
        right_gt, right_prediction, config.BOUNDARY_TOLERANCE
    )
    metrics = metrics_from_confusion(confusion)
    add_boundary_metrics(metrics, boundary)
    return {f"right_segmentation_{key}": value for key, value in metrics.items()}


def human_mask_correspondence_metrics(
    disparity: np.ndarray,
    left_gt: np.ndarray,
    right_gt: np.ndarray,
) -> dict:
    valid = np.isfinite(disparity) & (disparity > 0) & left_gt
    support = sample_right_probability(right_gt.astype(np.float32), disparity)
    violation = valid & (support < 0.5)
    return {
        "human_gt_valid_subject_pixels": int(valid.sum()),
        "human_right_mask_violation_pixels": int(violation.sum()),
        "human_right_mask_violation_rate": float(
            violation.sum() / max(valid.sum(), 1)
        ),
        "mean_human_right_mask_support": (
            float(support[valid].mean()) if np.any(valid) else None
        ),
    }


def save_method_outputs(
    directory: Path,
    disparity: np.ndarray,
    subject: np.ndarray,
    save_arrays: bool,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if save_arrays:
        np.save(directory / "disparity.npy", disparity)
        np.save(directory / "subject_disparity.npy", subject)
    cv2.imwrite(str(directory / "disparity.png"), colorize_disparity(disparity))
    cv2.imwrite(
        str(directory / "subject_disparity.png"), colorize_disparity(subject)
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    output = {}
    for method, method_rows in grouped.items():
        keys = sorted(
            {
                key
                for row in method_rows
                for key, value in row.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        )
        output[method] = {
            "scene_count": len(method_rows),
            **{
                f"mean_{key}": float(
                    np.mean([row[key] for row in method_rows if row.get(key) is not None])
                )
                for key in keys
                if any(row.get(key) is not None for row in method_rows)
            },
        }
    return output


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest.resolve())
    if args.category:
        selected = set(args.category)
        rows = [row for row in rows if row["category"] in selected]
        if not rows:
            raise ValueError(f"No manifest rows match categories: {sorted(selected)}")
    if args.limit is not None:
        rows = rows[: args.limit]
    selected_methods = tuple(args.method) if args.method else METHODS
    if len(set(selected_methods)) != len(selected_methods):
        raise ValueError("Duplicate --method values are not allowed")
    device = torch.device(args.device)
    predictor = RGBSegmenterPredictor(
        args.segmenter,
        config.IMAGE_WIDTH,
        config.IMAGE_HEIGHT,
        device,
        args.no_amp,
    )
    threshold = predictor.threshold if args.threshold is None else args.threshold
    model = load_las(args.las_checkpoint.resolve(), device)

    # Exclude one-time CUDA/cuDNN kernel setup from every reported method.  The
    # first smoke run otherwise makes the baseline appear hundreds of times
    # slower than the immediately following variants.
    warm_left = cv2.imread(rows[0]["left"], cv2.IMREAD_COLOR)
    warm_right = cv2.imread(rows[0]["right"], cv2.IMREAD_COLOR)
    if warm_left is None or warm_right is None:
        raise FileNotFoundError("Cannot read the first manifest pair for warm-up")
    predictor.predict_pair(warm_left, warm_right)
    infer_las(
        model,
        warm_left,
        warm_right,
        device,
        args.max_disparity,
    )

    result_rows = []
    for index, row in enumerate(rows, start=1):
        left = cv2.imread(row["left"], cv2.IMREAD_COLOR)
        right = cv2.imread(row["right"], cv2.IMREAD_COLOR)
        gt_image = cv2.imread(row["left_gt_mask"], cv2.IMREAD_GRAYSCALE)
        right_gt_image = (
            cv2.imread(row["right_gt_mask"], cv2.IMREAD_GRAYSCALE)
            if row.get("right_gt_mask")
            else None
        )
        if left is None or right is None:
            raise FileNotFoundError(f"Invalid pair {row['left']} / {row['right']}")
        if gt_image is None:
            raise FileNotFoundError(row["left_gt_mask"])
        gt = cv2.resize(
            gt_image,
            (left.shape[1], left.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 127
        right_gt = None
        if row.get("right_gt_mask"):
            if right_gt_image is None:
                raise FileNotFoundError(row["right_gt_mask"])
            right_gt = cv2.resize(
                right_gt_image,
                (right.shape[1], right.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ) > 127
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        segment_started = time.perf_counter()
        left_probability, right_probability, _, _ = predictor.predict_pair(left, right)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        segmentation_seconds = time.perf_counter() - segment_started
        scene_root = output / "scenes" / row["name"]
        scene_root.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(scene_root / "left_mask.png"),
            (left_probability >= threshold).astype(np.uint8) * 255,
        )
        cv2.imwrite(
            str(scene_root / "right_mask.png"),
            (right_probability >= threshold).astype(np.uint8) * 255,
        )

        baseline_disparity, baseline_subject, _, baseline_diagnostics = run_method(
            "baseline",
            model,
            left,
            right,
            left_probability,
            right_probability,
            device,
            threshold=threshold,
            max_disparity=args.max_disparity,
            guidance_weight=args.guidance_weight,
            roi_margin=args.roi_margin,
        )
        method_outputs = {}
        if "baseline" in selected_methods:
            method_outputs["baseline"] = (
                baseline_disparity,
                baseline_subject,
                baseline_diagnostics,
            )
        if "post_mask" in selected_methods:
            method_outputs["post_mask"] = (
                baseline_disparity.copy(),
                baseline_subject.copy(),
                {
                    **baseline_diagnostics,
                    "method": "post_mask",
                    "reused_baseline_inference": True,
                },
            )
        for method in ("roi", "guided"):
            if method not in selected_methods:
                continue
            disparity, subject, _, diagnostics = run_method(
                method,
                model,
                left,
                right,
                left_probability,
                right_probability,
                device,
                threshold=threshold,
                max_disparity=args.max_disparity,
                guidance_weight=args.guidance_weight,
                roi_margin=args.roi_margin,
            )
            method_outputs[method] = (disparity, subject, diagnostics)

        right_prediction = right_probability >= threshold
        right_metrics = (
            right_segmentation_metrics(right_gt, right_prediction)
            if right_gt is not None
            else {}
        )
        for method in selected_methods:
            disparity, subject, diagnostics = method_outputs[method]
            save_method_outputs(
                scene_root / method, disparity, subject, args.save_arrays
            )
            comparable = (
                gt
                & np.isfinite(disparity)
                & np.isfinite(baseline_disparity)
                & (disparity > 0)
                & (baseline_disparity > 0)
            )
            relative_change = (
                float(np.abs(disparity[comparable] - baseline_disparity[comparable]).mean())
                if np.any(comparable)
                else None
            )
            flat = {
                "name": row["name"],
                "category": row["category"],
                "method": method,
                "segmentation_seconds": segmentation_seconds,
                "stereo_seconds": diagnostics["stereo_seconds"],
                "computed_pixel_fraction": diagnostics["computed_pixel_fraction"],
                "roi_used": diagnostics["roi"]["used"],
                "roi_reason": diagnostics["roi"]["reason"],
                "valid_subject_pixels": diagnostics["valid_subject_pixels"],
                "output_valid_pixel_fraction": float(
                    (
                        np.isfinite(disparity if method == "baseline" else subject)
                        & ((disparity if method == "baseline" else subject) > 0)
                    ).mean()
                ),
                "right_mask_violation_rate": diagnostics[
                    "right_mask_violation_rate"
                ],
                "mean_right_mask_support": diagnostics["mean_right_mask_support"],
                "mean_subject_photometric_error": diagnostics[
                    "mean_subject_photometric_error"
                ],
                "mean_abs_change_from_baseline": relative_change,
                "right_gt_available": right_gt is not None,
                **right_metrics,
                **(
                    human_mask_correspondence_metrics(disparity, gt, right_gt)
                    if right_gt is not None
                    else {}
                ),
                **reference_metrics(
                    disparity,
                    row.get("reference_disparity", ""),
                    row.get("reference_crop_y0_y1_x0_x1", ""),
                    gt,
                ),
            }
            result_rows.append(flat)
        print(f"[{index}/{len(rows)}] {row['name']}", flush=True)

    write_csv(output / "per_scene.csv", result_rows)
    summary = {
        "completed": True,
        "manifest": str(args.manifest.resolve()),
        "scene_count": len(rows),
        "mask_threshold": float(threshold),
        "guidance_weight": float(args.guidance_weight),
        "selected_methods": list(selected_methods),
        "methods": aggregate(result_rows),
        "human_right_gt_note": (
            "When right_gt_mask is present, human_right_mask_violation_rate "
            "measures valid human-left foreground correspondences landing outside "
            "the independently annotated right foreground."
        ),
        "reference_note": (
            "FDJYP-3 reference is Foundation Stereo engineering reference, not human GT."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
