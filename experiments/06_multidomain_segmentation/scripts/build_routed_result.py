#!/usr/bin/env python3
"""Build a category-routed test result using validation-only model selection."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from evaluate_test import (
    boundary_counts,
    boundary_f1,
    confusion_counts,
    make_comparison,
    metrics_from_confusion,
    save_contact_sheet,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    results = root / "experiments/06_multidomain_segmentation/results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=root / "datasets/training/workpiece-seg-isat-v2"
    )
    parser.add_argument("--old-validation", type=Path, default=results / "val_old_model")
    parser.add_argument("--new-validation", type=Path, default=results / "val_balanced_v2")
    parser.add_argument("--old-test", type=Path, default=results / "baseline_old_model")
    parser.add_argument("--new-test", type=Path, default=results / "test_balanced_v2")
    parser.add_argument("--output", type=Path, default=results / "routed_v3")
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    return parser.parse_args()


def load_summary(path: Path) -> dict:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a new result version; output exists: {}".format(output))
    old_validation = load_summary(args.old_validation.resolve())
    new_validation = load_summary(args.new_validation.resolve())
    old_test = load_summary(args.old_test.resolve())
    new_test = load_summary(args.new_test.resolve())
    categories = sorted(old_validation["per_category"])
    if categories != sorted(new_validation["per_category"]):
        raise ValueError("Validation category mismatch")

    route = {}
    route_evidence = {}
    for category in categories:
        old_score = old_validation["per_category"][category]["foreground_iou"]
        new_score = new_validation["per_category"][category]["foreground_iou"]
        selected = "balanced_v2" if new_score > old_score else "old_model"
        route[category] = selected
        route_evidence[category] = {
            "metric": "validation_foreground_iou",
            "old_model": old_score,
            "balanced_v2": new_score,
            "selected": selected,
        }

    with (dataset / "index/test.csv").open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    (output / "masks").mkdir(parents=True, exist_ok=False)
    (output / "probabilities").mkdir(parents=True, exist_ok=False)
    (output / "comparisons").mkdir(parents=True, exist_ok=False)

    source_dirs = {
        "old_model": args.old_test.resolve(),
        "balanced_v2": args.new_test.resolve(),
    }
    overall_confusion = np.zeros((2, 2), dtype=np.int64)
    category_confusions = defaultdict(lambda: np.zeros((2, 2), dtype=np.int64))
    overall_boundary = np.zeros(4, dtype=np.int64)
    category_boundaries = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    metric_rows = []
    comparisons = []
    for record in records:
        category = record["category"]
        selected = route[category]
        source_dir = source_dirs[selected]
        source_mask = source_dir / "masks" / "{}.png".format(record["name"])
        source_probability = source_dir / "probabilities" / "{}.npy".format(record["name"])
        image = cv2.imread(str(dataset / record["image"]), cv2.IMREAD_COLOR)
        gt_u8 = cv2.imread(str(dataset / record["mask"]), cv2.IMREAD_GRAYSCALE)
        pred_u8 = cv2.imread(str(source_mask), cv2.IMREAD_GRAYSCALE)
        if image is None or gt_u8 is None or pred_u8 is None:
            raise FileNotFoundError(record["name"])
        gt = gt_u8 > 0
        pred = pred_u8 > 0
        confusion = confusion_counts(gt, pred)
        boundaries = np.asarray(
            boundary_counts(gt, pred, args.boundary_tolerance), dtype=np.int64
        )
        metrics = metrics_from_confusion(confusion)
        metrics.update(boundary_f1(tuple(int(value) for value in boundaries)))
        metrics.update(
            {
                "name": record["name"],
                "category": category,
                "selected_model": selected,
                "gt_foreground_fraction": float(gt.mean()),
                "pred_foreground_fraction": float(pred.mean()),
            }
        )
        metric_rows.append(metrics)
        overall_confusion += confusion
        category_confusions[category] += confusion
        overall_boundary += boundaries
        category_boundaries[category] += boundaries

        target_mask = output / "masks" / source_mask.name
        target_probability = output / "probabilities" / source_probability.name
        shutil.copy2(source_mask, target_mask)
        shutil.copy2(source_probability, target_probability)
        comparison = make_comparison(
            image.copy(),
            gt,
            pred,
            "{} [{} -> {}] IoU={:.3f}".format(
                record["name"], category, selected, metrics["foreground_iou"]
            ),
        )
        cv2.imwrite(
            str(output / "comparisons" / "{}.jpg".format(record["name"])),
            comparison,
            [cv2.IMWRITE_JPEG_QUALITY, 94],
        )
        comparisons.append(comparison)

    fieldnames = [
        "name", "category", "selected_model", "foreground_iou", "background_iou",
        "mean_iou", "dice", "precision", "recall", "pixel_accuracy",
        "boundary_precision", "boundary_recall", "boundary_f1",
        "gt_foreground_fraction", "pred_foreground_fraction", "tn", "fp", "fn", "tp",
    ]
    with (output / "test_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in metric_rows)

    overall = metrics_from_confusion(overall_confusion)
    overall.update(boundary_f1(tuple(int(value) for value in overall_boundary)))
    per_category = {}
    for category in sorted(category_confusions):
        metrics = metrics_from_confusion(category_confusions[category])
        metrics.update(
            boundary_f1(tuple(int(value) for value in category_boundaries[category]))
        )
        metrics["count"] = sum(row["category"] == category for row in metric_rows)
        metrics["selected_model"] = route[category]
        per_category[category] = metrics
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "filename_category_route_selected_only_on_validation_iou",
        "route": route,
        "route_evidence": route_evidence,
        "model_sources": {
            "old_model": old_test["model_pb"],
            "old_model_sha256": old_test["model_pb_sha256"],
            "balanced_v2": new_test["model_pb"],
            "balanced_v2_sha256": new_test["model_pb_sha256"],
        },
        "dataset": str(dataset),
        "split": "development_comparison_test",
        "count": len(records),
        "overall": overall,
        "per_category": per_category,
        "priority": "subject_recall",
        "test_reuse_warning": (
            "The first test result was inspected before V2 was trained. These numbers are a "
            "development comparison, not a pristine final generalization estimate."
        ),
        "hole_evaluation_warning": (
            "Human labels cover the outer subject and contain no explicit background-hole "
            "polygons. Hole extraction must be evaluated separately after disparity refinement."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_contact_sheet(comparisons, output / "test_contact_sheet.jpg")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
