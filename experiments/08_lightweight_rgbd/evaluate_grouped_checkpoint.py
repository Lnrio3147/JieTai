#!/usr/bin/env python3
"""Calibrate one student on V3 validation and evaluate the frozen V3 test set."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import config_experiment8 as config
from evaluate import load_model, postprocess_all, predict_probabilities
from utils.data import (
    WorkpieceStudentDataset,
    disparity_path,
    read_disparity,
)
from utils.metrics import aggregate_metrics, label_panel, mask_panel


DEFAULT_DATASET = config.ROOT / "datasets/training/workpiece-seg-grouped-v3"
DEFAULT_CHECKPOINT = config.RESULTS_DIR / "student_base_grouped_v3/best.pt"
DEFAULT_OUTPUT = config.RESULTS_DIR / "student_base_grouped_v3/evaluation"
CALIBRATION_THRESHOLDS = tuple(
    sorted(
        set(config.THRESHOLD_CANDIDATES)
        | {round(value / 100.0, 2) for value in range(5, 20)}
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=config.WORKERS)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def load_disparities(
    dataset: Path, records: list[dict[str, str]]
) -> dict[str, np.ndarray]:
    output = {}
    for record in records:
        if record.get("disparity"):
            path = dataset / record["disparity"]
        else:
            path = disparity_path(config.ROOT, record)
        output[record["name"]] = read_disparity(path)
    return output


def calibrate_threshold(
    dataset: Path,
    records: list[dict[str, str]],
    probabilities: dict[str, np.ndarray],
    disparities: dict[str, np.ndarray],
) -> tuple[float, list[dict]]:
    table = []
    for threshold in CALIBRATION_THRESHOLDS:
        masks, _ = postprocess_all(records, probabilities, threshold, disparities)
        metrics = aggregate_metrics(
            dataset, records, masks, config.BOUNDARY_TOLERANCE
        )["overall"]
        table.append({"threshold": threshold, **metrics})
    eligible = [
        row for row in table if row["recall"] >= config.THRESHOLD_RECALL_FLOOR
    ]
    selection_pool = eligible or table
    selected = max(
        selection_pool,
        key=lambda row: (
            row["foreground_iou"],
            row["macro_category_iou"],
            row["boundary_f1"],
        ),
    )
    return float(selected["threshold"]), table


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(path)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_contact_sheet(
    dataset: Path,
    records: list[dict[str, str]],
    masks: dict[str, np.ndarray],
    output: Path,
    model_label: str,
) -> None:
    sample_width, sample_height = 144, 256
    columns = 4
    rows = int(math.ceil(len(records) / columns))
    scene_width = sample_width * 3
    canvas = np.zeros(
        (rows * sample_height, columns * scene_width, 3), dtype=np.uint8
    )
    for index, record in enumerate(records):
        image = cv2.imread(str(dataset / record["image"]), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(dataset / record["mask"]), cv2.IMREAD_GRAYSCALE)
        if image is None or gt is None:
            raise FileNotFoundError(record["name"])
        image = cv2.resize(
            image, (sample_width, sample_height), interpolation=cv2.INTER_AREA
        )
        gt = cv2.resize(
            gt, (sample_width, sample_height), interpolation=cv2.INTER_NEAREST
        ) > 127
        prediction = cv2.resize(
            masks[record["name"]].astype(np.uint8),
            (sample_width, sample_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        label = f"{record['category']} {record['name'][-9:]}"
        panel = np.hstack(
            [
                label_panel(image, label),
                mask_panel(gt, "GT"),
                mask_panel(prediction, model_label),
            ]
        )
        row, column = divmod(index, columns)
        y, x = row * sample_height, column * scene_width
        canvas[y : y + sample_height, x : x + scene_width] = panel
    cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 91])


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    common = {
        "root": config.ROOT,
        "dataset": dataset,
        "width": config.IMAGE_WIDTH,
        "height": config.IMAGE_HEIGHT,
        "augment": False,
        "seed": config.SEED,
    }
    datasets = {
        split: WorkpieceStudentDataset(split=split, **common)
        for split in ("val", "test")
    }
    loaders = {
        split: DataLoader(
            value,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        for split, value in datasets.items()
    }
    model = load_model(checkpoint, device)
    probabilities = {
        split: predict_probabilities(model, loaders[split], device, amp)
        for split in ("val", "test")
    }
    disparities = {
        split: load_disparities(dataset, datasets[split].records)
        for split in ("val", "test")
    }
    threshold, threshold_table = calibrate_threshold(
        dataset,
        datasets["val"].records,
        probabilities["val"],
        disparities["val"],
    )
    test_masks, diagnostics = postprocess_all(
        datasets["test"].records,
        probabilities["test"],
        threshold,
        disparities["test"],
    )
    evaluation = aggregate_metrics(
        dataset,
        datasets["test"].records,
        test_masks,
        config.BOUNDARY_TOLERANCE,
    )

    mask_dir = output / "test_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for name, mask in test_masks.items():
        cv2.imwrite(str(mask_dir / f"{name}.png"), mask.astype(np.uint8) * 255)
    write_csv(output / "threshold_sweep.csv", threshold_table)
    write_csv(
        output / "per_category.csv",
        [
            {"category": category, **metrics}
            for category, metrics in evaluation["per_category"].items()
        ],
    )
    write_csv(
        output / "per_image.csv",
        [
            {
                "name": record["name"],
                "category": record["category"],
                **evaluation["per_scene"][record["name"]],
            }
            for record in datasets["test"].records
        ],
    )
    save_contact_sheet(
        dataset,
        datasets["test"].records,
        test_masks,
        output / "test_contact_sheet.jpg",
        "V3 Distilled" if "distilled" in checkpoint.parent.name else "V3 Base",
    )

    component_counts = [
        diagnostics[record["name"]]["connected_components"]
        for record in datasets["test"].records
    ]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "checkpoint": str(checkpoint),
        "validation_count": len(datasets["val"]),
        "test_count": len(datasets["test"]),
        "threshold_selection": (
            "validation IoU maximum subject to Recall >= "
            f"{config.THRESHOLD_RECALL_FLOOR:.2f}"
        ),
        "selected_threshold": threshold,
        "test_metrics": evaluation,
        "continuity": {
            "all_single_component": all(count == 1 for count in component_counts),
            "mean_components": float(np.mean(component_counts)),
            "max_components": int(max(component_counts)),
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected_threshold": threshold,
                "test_overall": evaluation["overall"],
                "continuity": summary["continuity"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
