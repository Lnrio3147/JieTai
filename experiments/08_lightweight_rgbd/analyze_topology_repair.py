#!/usr/bin/env python3
"""Search deployable repairs for holes and broken subject boundaries.

The script consumes the cached FDJYP-0 predictions produced by
``evaluate_fdjyp0.py``.  It deliberately separates single-model candidates
from two-student consensus candidates so deployment cost remains explicit.
Ground truth is used only for evaluation and never by a repair function.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

import config_experiment8 as config
from evaluate_fdjyp0 import ANNOTATION_DATASET, read_fdjyp0_records
from utils.metrics import boundary_counts, confusion_counts, metrics_from_confusion
from utils.postprocess import largest_component


DEFAULT_SOURCE = config.RESULTS_DIR / "fdjyp0_unseen_82_20260823"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output",
        type=Path,
        default=config.RESULTS_DIR / "fdjyp0_topology_search_20260824.csv",
    )
    return parser.parse_args()


def read_mask(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise FileNotFoundError(path)
    return value > 127


def fill_external(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    output = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(output, contours, -1, 1, thickness=-1)
    return largest_component(output > 0)


def close_and_fill(mask: np.ndarray, radius: int) -> np.ndarray:
    value = mask.astype(np.uint8)
    if radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        value = cv2.morphologyEx(value, cv2.MORPH_CLOSE, kernel)
    return fill_external(value > 0)


def smooth_and_fill(mask: np.ndarray, sigma: float, threshold: float) -> np.ndarray:
    field = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    return fill_external(field >= threshold)


def convex_envelope(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.zeros_like(mask, dtype=bool)
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    output = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(output, [hull], -1, 1, thickness=-1)
    return output > 0


def span_envelope(mask: np.ndarray, axis: int) -> np.ndarray:
    """Fill foreground spans along rows (axis=1) or columns (axis=0)."""
    output = mask.copy()
    if axis == 1:
        for row in range(mask.shape[0]):
            indices = np.flatnonzero(mask[row])
            if indices.size > 1:
                output[row, indices[0] : indices[-1] + 1] = True
    else:
        for column in range(mask.shape[1]):
            indices = np.flatnonzero(mask[:, column])
            if indices.size > 1:
                output[indices[0] : indices[-1] + 1, column] = True
    return fill_external(output)


def restricted_rescue(
    seed: np.ndarray,
    support: np.ndarray,
    dilation_radius: int,
    closing_radius: int,
) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * dilation_radius + 1, 2 * dilation_radius + 1),
    )
    allowed = cv2.dilate(seed.astype(np.uint8), kernel) > 0
    candidate = seed | (support & allowed)
    return close_and_fill(candidate, closing_radius)


def evaluate_candidate(
    predictions: list[np.ndarray], ground_truth: list[np.ndarray]
) -> dict[str, float]:
    per_scene = []
    hole_counts = []
    for prediction, gt in zip(predictions, ground_truth):
        metrics = metrics_from_confusion(confusion_counts(gt, prediction))
        counts = boundary_counts(gt, prediction, config.BOUNDARY_TOLERANCE)
        boundary_precision = counts[0] / max(int(counts[1]), 1)
        boundary_recall = counts[2] / max(int(counts[3]), 1)
        metrics["boundary_f1"] = (
            2.0
            * boundary_precision
            * boundary_recall
            / max(boundary_precision + boundary_recall, 1e-12)
        )
        per_scene.append(metrics)
        inverse = (~prediction).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(inverse, 8)
        height, width = prediction.shape
        holes = 0
        for index in range(1, count):
            x, y, w, h, _ = (int(v) for v in stats[index])
            if x > 0 and y > 0 and x + w < width and y + h < height:
                holes += 1
        hole_counts.append(holes)
    return {
        "macro_iou": float(np.mean([v["foreground_iou"] for v in per_scene])),
        "macro_precision": float(np.mean([v["precision"] for v in per_scene])),
        "macro_recall": float(np.mean([v["recall"] for v in per_scene])),
        "macro_boundary_f1": float(np.mean([v["boundary_f1"] for v in per_scene])),
        "images_with_holes": int(sum(value > 0 for value in hole_counts)),
        "mean_holes": float(np.mean(hole_counts)),
    }


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    records = read_fdjyp0_records()
    names = [record["name"] for record in records]
    ground_truth = [
        read_mask(ANNOTATION_DATASET / record["mask"]) for record in records
    ]
    masks = {
        method: [read_mask(source / "masks" / method / f"{name}.png") for name in names]
        for method in ("teacher_7_2", "student_base", "student_distilled")
    }
    probabilities = {
        method: [
            np.load(source / "probabilities" / method / f"{name}.npy").astype(
                np.float32
            )
            for name in names
        ]
        for method in ("teacher_7_2", "student_base", "student_distilled")
    }

    candidates: dict[str, list[np.ndarray]] = {
        "base_current": masks["student_base"],
        "base_fill_holes": [fill_external(value) for value in masks["student_base"]],
    }
    for radius in (4, 8, 12, 16, 20, 24, 30, 40):
        candidates[f"base_close_fill_r{radius}"] = [
            close_and_fill(value, radius) for value in masks["student_base"]
        ]
    for sigma in (2.0, 3.0, 4.0, 5.0, 6.0):
        candidates[f"base_smooth_fill_s{sigma:g}_t60"] = [
            smooth_and_fill(value, sigma, 0.60) for value in masks["student_base"]
        ]
    candidates["base_convex"] = [convex_envelope(v) for v in masks["student_base"]]
    row = [span_envelope(v, 1) for v in masks["student_base"]]
    column = [span_envelope(v, 0) for v in masks["student_base"]]
    candidates["base_row_span"] = row
    candidates["base_column_span"] = column
    candidates["base_span_intersection"] = [r & c for r, c in zip(row, column)]
    candidates["base_span_union"] = [fill_external(r | c) for r, c in zip(row, column)]

    consensus_supports = {
        "distill": masks["student_distilled"],
        "teacher": masks["teacher_7_2"],
        "distill_teacher": [
            a & b for a, b in zip(masks["student_distilled"], masks["teacher_7_2"])
        ],
    }
    for support_name, supports in consensus_supports.items():
        for dilation_radius in (10, 20, 30, 40, 60):
            candidates[f"base_{support_name}_d{dilation_radius}_c8"] = [
                restricted_rescue(seed, support, dilation_radius, 8)
                for seed, support in zip(masks["student_base"], supports)
            ]

    for low_threshold in (0.02, 0.05, 0.08, 0.10, 0.12, 0.16, 0.20):
        supports = [value >= low_threshold for value in probabilities["student_base"]]
        for dilation_radius in (10, 20, 30, 40):
            candidates[f"base_hyst_t{low_threshold:.2f}_d{dilation_radius}_c8"] = [
                restricted_rescue(seed, support, dilation_radius, 8)
                for seed, support in zip(masks["student_base"], supports)
            ]

    rows = []
    for name, predictions in candidates.items():
        family = "single_model"
        if "distill" in name:
            family = "two_students"
        elif "teacher" in name:
            family = "student_teacher"
        rows.append({"candidate": name, "family": family, **evaluate_candidate(predictions, ground_truth)})
    rows.sort(key=lambda row: row["macro_iou"], reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows[:25], indent=2))
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
