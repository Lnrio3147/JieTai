"""Pixel and boundary metrics matching Experiments 6/7."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def confusion_counts(gt: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    encoded = gt.astype(np.uint8).reshape(-1) * 2 + prediction.astype(np.uint8).reshape(-1)
    return np.bincount(encoded, minlength=4).reshape(2, 2).astype(np.int64)


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = (int(value) for value in confusion.reshape(-1))
    return {
        "foreground_iou": tp / max(tp + fp + fn, 1),
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "pixel_accuracy": (tp + tn) / max(int(confusion.sum()), 1),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def boundary_counts(gt: np.ndarray, prediction: np.ndarray, tolerance: int) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    gt_edge = cv2.morphologyEx(gt.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    pred_edge = (
        cv2.morphologyEx(prediction.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    )
    band_kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=np.uint8)
    gt_band = cv2.dilate(gt_edge.astype(np.uint8), band_kernel) > 0
    pred_band = cv2.dilate(pred_edge.astype(np.uint8), band_kernel) > 0
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
    metrics.update(
        {
            "boundary_precision": precision,
            "boundary_recall": recall,
            "boundary_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        }
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
        gt = cv2.imread(str(dataset / record["mask"]), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise FileNotFoundError(dataset / record["mask"])
        gt = gt > 127
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
        scene_metrics = metrics_from_confusion(confusion)
        add_boundary_metrics(scene_metrics, boundary)
        per_scene[record["name"]] = scene_metrics

    overall = metrics_from_confusion(total_confusion)
    add_boundary_metrics(overall, total_boundary)
    per_category = {}
    for category in sorted(category_confusion):
        values = metrics_from_confusion(category_confusion[category])
        add_boundary_metrics(values, category_boundary[category])
        per_category[category] = values
    overall["macro_category_iou"] = float(
        np.mean([values["foreground_iou"] for values in per_category.values()])
    )
    return {"overall": overall, "per_category": per_category, "per_scene": per_scene}


def label_panel(image: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(
        output,
        label,
        (5, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def mask_panel(mask: np.ndarray, label: str) -> np.ndarray:
    panel = cv2.cvtColor(np.asarray(mask, dtype=np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    return label_panel(panel, label)
