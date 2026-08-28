"""Segmentation metrics and validation-only threshold calibration."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def confusion_counts(gt: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    encoded = gt.astype(np.uint8).ravel() * 2 + prediction.astype(np.uint8).ravel()
    return np.bincount(encoded, minlength=4).reshape(2, 2).astype(np.int64)


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = (int(value) for value in confusion.ravel())
    return {
        "foreground_iou": tp / max(tp + fp + fn, 1),
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def boundary_counts(gt: np.ndarray, prediction: np.ndarray, tolerance: int) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    gt_edge = cv2.morphologyEx(gt.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    pred_edge = cv2.morphologyEx(
        prediction.astype(np.uint8), cv2.MORPH_GRADIENT, kernel
    ) > 0
    band = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=np.uint8)
    gt_band = cv2.dilate(gt_edge.astype(np.uint8), band) > 0
    pred_band = cv2.dilate(pred_edge.astype(np.uint8), band) > 0
    return np.asarray(
        [
            int((pred_edge & gt_band).sum()),
            int(pred_edge.sum()),
            int((gt_edge & pred_band).sum()),
            int(gt_edge.sum()),
        ],
        dtype=np.int64,
    )


def add_boundary_metrics(metrics: dict, counts: np.ndarray) -> None:
    matched_pred, pred_total, matched_gt, gt_total = (int(value) for value in counts)
    precision = matched_pred / max(pred_total, 1)
    recall = matched_gt / max(gt_total, 1)
    metrics["boundary_precision"] = precision
    metrics["boundary_recall"] = recall
    metrics["boundary_f1"] = 2 * precision * recall / max(
        precision + recall, 1e-12
    )


def aggregate_metrics(
    dataset: Path,
    records: list[dict[str, str]],
    predictions: dict[str, np.ndarray],
    boundary_tolerance: int,
) -> dict:
    total_confusion = np.zeros((2, 2), dtype=np.int64)
    total_boundary = np.zeros(4, dtype=np.int64)
    category_confusion = defaultdict(lambda: np.zeros((2, 2), dtype=np.int64))
    category_boundary = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    per_scene = {}
    for record in records:
        gt_path = dataset / record["mask"]
        gt_image = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt_image is None:
            raise FileNotFoundError(gt_path)
        gt = gt_image > 127
        prediction = np.asarray(predictions[record["name"]], dtype=bool)
        if prediction.shape != gt.shape:
            prediction = cv2.resize(
                prediction.astype(np.uint8),
                (gt.shape[1], gt.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        confusion = confusion_counts(gt, prediction)
        boundary = boundary_counts(gt, prediction, boundary_tolerance)
        total_confusion += confusion
        total_boundary += boundary
        category_confusion[record["category"]] += confusion
        category_boundary[record["category"]] += boundary
        values = metrics_from_confusion(confusion)
        add_boundary_metrics(values, boundary)
        per_scene[record["name"]] = values

    overall = metrics_from_confusion(total_confusion)
    add_boundary_metrics(overall, total_boundary)
    per_category = {}
    for category in sorted(category_confusion):
        values = metrics_from_confusion(category_confusion[category])
        add_boundary_metrics(values, category_boundary[category])
        per_category[category] = values
    overall["macro_category_iou"] = float(
        np.mean([value["foreground_iou"] for value in per_category.values()])
    )
    return {"overall": overall, "per_category": per_category, "per_scene": per_scene}


def select_threshold(
    dataset: Path,
    records: list[dict[str, str]],
    probabilities: dict[str, np.ndarray],
    candidates: tuple[float, ...],
    recall_floor: float,
    boundary_tolerance: int,
) -> tuple[float, list[dict]]:
    rows = []
    for threshold in candidates:
        predictions = {
            name: probability >= threshold
            for name, probability in probabilities.items()
        }
        metrics = aggregate_metrics(
            dataset, records, predictions, boundary_tolerance
        )["overall"]
        rows.append({"threshold": float(threshold), **metrics})
    eligible = [row for row in rows if row["recall"] >= recall_floor]
    pool = eligible or rows
    best = max(
        pool,
        key=lambda row: (
            row["foreground_iou"],
            row["boundary_f1"],
            row["recall"],
        ),
    )
    return float(best["threshold"]), rows
