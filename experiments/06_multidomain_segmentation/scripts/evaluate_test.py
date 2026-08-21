#!/usr/bin/env python3
"""Evaluate a frozen BiSeNetV2 model on the frozen stratified test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-pb", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root / "datasets/training/workpiece-seg-isat-v2",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--foreground-threshold",
        type=float,
        default=0.5,
        help="Foreground probability threshold used for the binary mask.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_records(dataset: Path, split: str) -> list[dict]:
    with (dataset / "index/{}.csv".format(split)).open(
        encoding="utf-8", newline=""
    ) as stream:
        records = list(csv.DictReader(stream))
    if not records:
        raise ValueError("{} index is empty".format(split))
    return records


def confusion_counts(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    encoded = gt.astype(np.uint8).reshape(-1) * 2 + pred.astype(np.uint8).reshape(-1)
    return np.bincount(encoded, minlength=4).reshape(2, 2).astype(np.int64)


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = (int(value) for value in confusion.reshape(-1))
    foreground_union = tp + fp + fn
    background_union = tn + fp + fn
    return {
        "foreground_iou": tp / max(foreground_union, 1),
        "background_iou": tn / max(background_union, 1),
        "mean_iou": 0.5 * (
            tp / max(foreground_union, 1) + tn / max(background_union, 1)
        ),
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "pixel_accuracy": (tp + tn) / max(confusion.sum(), 1),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def boundary(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    source = mask.astype(np.uint8)
    return cv2.morphologyEx(source, cv2.MORPH_GRADIENT, kernel) > 0


def boundary_counts(gt: np.ndarray, pred: np.ndarray, tolerance: int) -> tuple[int, int, int, int]:
    gt_boundary = boundary(gt)
    pred_boundary = boundary(pred)
    size = 2 * tolerance + 1
    kernel = np.ones((size, size), dtype=np.uint8)
    gt_dilated = cv2.dilate(gt_boundary.astype(np.uint8), kernel) > 0
    pred_dilated = cv2.dilate(pred_boundary.astype(np.uint8), kernel) > 0
    matched_pred = int(np.count_nonzero(pred_boundary & gt_dilated))
    matched_gt = int(np.count_nonzero(gt_boundary & pred_dilated))
    return matched_pred, int(pred_boundary.sum()), matched_gt, int(gt_boundary.sum())


def boundary_f1(counts: tuple[int, int, int, int]) -> dict[str, float]:
    matched_pred, pred_total, matched_gt, gt_total = counts
    precision = matched_pred / max(pred_total, 1)
    recall = matched_gt / max(gt_total, 1)
    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def make_comparison(
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    title: str,
) -> np.ndarray:
    gt_u8 = gt.astype(np.uint8) * 255
    pred_u8 = pred.astype(np.uint8) * 255
    error = np.zeros_like(image)
    error[(pred == 1) & (gt == 0)] = (0, 0, 255)  # false positive: red
    error[(pred == 0) & (gt == 1)] = (255, 0, 0)  # false negative: blue
    panels = [
        image,
        cv2.cvtColor(gt_u8, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(pred_u8, cv2.COLOR_GRAY2BGR),
        error,
    ]
    labels = ("image", "ground truth", "prediction", "FP red / FN blue")
    for panel, label in zip(panels, labels):
        cv2.putText(
            panel,
            label,
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    comparison = np.hstack(panels)
    cv2.putText(
        comparison,
        title,
        (5, comparison.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return comparison


def save_contact_sheet(comparisons: list[np.ndarray], output: Path) -> None:
    tiles = [
        cv2.resize(comparison, (720, 256), interpolation=cv2.INTER_AREA)
        for comparison in comparisons
    ]
    blank = np.zeros_like(tiles[0])
    rows = []
    for start in range(0, len(tiles), 2):
        row = tiles[start : start + 2]
        row.extend([blank] * (2 - len(row)))
        rows.append(np.hstack(row))
    cv2.imwrite(str(output), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> None:
    args = parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    model_pb = args.model_pb.resolve()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if not 0.0 < args.foreground_threshold < 1.0:
        raise ValueError("--foreground-threshold must be in (0, 1)")
    if output.exists():
        raise FileExistsError("Use a new result version; output exists: {}".format(output))
    records = read_records(dataset, args.split)
    (output / "masks").mkdir(parents=True, exist_ok=False)
    (output / "probabilities").mkdir(parents=True, exist_ok=False)
    (output / "comparisons").mkdir(parents=True, exist_ok=False)

    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(model_pb.read_bytes())
        tf.import_graph_def(graph_def, name="")
    input_tensor = graph.get_tensor_by_name("input_tensor:0")
    probability_tensor = graph.get_tensor_by_name("final_probability:0")
    output_tensor = graph.get_tensor_by_name("final_output:0")

    overall_confusion = np.zeros((2, 2), dtype=np.int64)
    category_confusions = defaultdict(lambda: np.zeros((2, 2), dtype=np.int64))
    overall_boundary = np.zeros(4, dtype=np.int64)
    category_boundaries = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    metric_rows = []
    comparisons = []
    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=session_config) as session:
        for index, record in enumerate(records, start=1):
            image = cv2.imread(str(dataset / record["image"]), cv2.IMREAD_COLOR)
            gt_u8 = cv2.imread(str(dataset / record["mask"]), cv2.IMREAD_GRAYSCALE)
            if image is None or gt_u8 is None:
                raise FileNotFoundError(record["name"])
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            batch = ((rgb - 0.5) / 0.5)[None]
            probabilities = session.run(
                probability_tensor,
                feed_dict={input_tensor: batch},
            )
            gt = gt_u8 > 0
            foreground_probability = probabilities[0, :, :, 1].astype(np.float32)
            pred = foreground_probability >= args.foreground_threshold
            confusion = confusion_counts(gt, pred)
            boundaries = np.asarray(
                boundary_counts(gt, pred, args.boundary_tolerance), dtype=np.int64
            )
            metrics = metrics_from_confusion(confusion)
            metrics.update(boundary_f1(tuple(int(value) for value in boundaries)))
            metrics.update(
                {
                    "name": record["name"],
                    "category": record["category"],
                    "gt_foreground_fraction": float(gt.mean()),
                    "pred_foreground_fraction": float(pred.mean()),
                }
            )
            metric_rows.append(metrics)
            overall_confusion += confusion
            category_confusions[record["category"]] += confusion
            overall_boundary += boundaries
            category_boundaries[record["category"]] += boundaries

            mask_path = output / "masks" / "{}.png".format(record["name"])
            probability_path = output / "probabilities" / "{}.npy".format(record["name"])
            comparison_path = output / "comparisons" / "{}.jpg".format(record["name"])
            cv2.imwrite(str(mask_path), pred.astype(np.uint8) * 255)
            np.save(str(probability_path), foreground_probability, allow_pickle=False)
            comparison = make_comparison(
                image.copy(), gt, pred, "{} [{}] IoU={:.3f}".format(
                    record["name"], record["category"], metrics["foreground_iou"]
                )
            )
            cv2.imwrite(str(comparison_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 94])
            comparisons.append(comparison)
            print(
                "{}/{} {} [{}] IoU={:.4f} recall={:.4f}".format(
                    index,
                    len(records),
                    record["name"],
                    record["category"],
                    metrics["foreground_iou"],
                    metrics["recall"],
                ),
                flush=True,
            )

    metric_fieldnames = [
        "name", "category", "foreground_iou", "background_iou", "mean_iou",
        "dice", "precision", "recall", "pixel_accuracy", "boundary_precision",
        "boundary_recall", "boundary_f1", "gt_foreground_fraction",
        "pred_foreground_fraction", "tn", "fp", "fn", "tp",
    ]
    metrics_name = "{}_metrics.csv".format(args.split)
    with (output / metrics_name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in metric_fieldnames} for row in metric_rows)

    overall = metrics_from_confusion(overall_confusion)
    overall.update(boundary_f1(tuple(int(value) for value in overall_boundary)))
    per_category = {}
    for category in sorted(category_confusions):
        category_metrics = metrics_from_confusion(category_confusions[category])
        category_metrics.update(
            boundary_f1(tuple(int(value) for value in category_boundaries[category]))
        )
        category_metrics["count"] = sum(
            row["category"] == category for row in metric_rows
        )
        per_category[category] = category_metrics
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_pb": str(model_pb),
        "model_pb_sha256": sha256_file(model_pb),
        "dataset": str(dataset),
        "dataset_metadata_sha256": sha256_file(dataset / "metadata.json"),
        "split": (
            "frozen_stratified_test" if args.split == "test" else "model_selection_validation"
        ),
        "count": len(records),
        "boundary_tolerance_pixels": args.boundary_tolerance,
        "foreground_threshold": args.foreground_threshold,
        "overall": overall,
        "per_category": per_category,
        "hole_evaluation_warning": (
            "Human labels contain no explicit background-hole polygons. Metrics measure the "
            "annotated outer subject mask and do not establish hole extraction accuracy."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_contact_sheet(
        comparisons, output / "{}_contact_sheet.jpg".format(args.split)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
