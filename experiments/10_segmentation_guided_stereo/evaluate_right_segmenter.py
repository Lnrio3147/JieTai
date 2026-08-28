#!/usr/bin/env python3
"""Evaluate RGB segmenters on manually reviewed right-view val/test masks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch

import config_experiment10 as config
from utils.data import read_records
from utils.metrics import aggregate_metrics, select_threshold
from utils.segmentation import RGBSegmenterPredictor


DEFAULT_MODELS = {
    "left_only": config.RESULTS_DIR / "rgb_segmenter_grouped_v3/best.pt",
    "pseudo_stereo": config.RESULTS_DIR / "rgb_segmenter_stereo_v1/best.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=config.RIGHT_MANUAL_DATASET,
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        metavar="NAME=CHECKPOINT",
        help="repeatable; defaults to left_only and pseudo_stereo",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=config.RESULTS_DIR / "right_manual_evaluation_v2",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_models(values: list[str] | None) -> dict[str, Path]:
    if values is None:
        return {name: path.resolve() for name, path in DEFAULT_MODELS.items()}
    models: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=CHECKPOINT, got {value!r}")
        name, raw_path = value.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"Invalid model name: {name!r}")
        if name in models:
            raise ValueError(f"Duplicate model name: {name}")
        models[name] = Path(raw_path).resolve()
    if not models:
        raise ValueError("At least one model is required")
    return models


def predict_split(
    predictor: RGBSegmenterPredictor,
    dataset: Path,
    records: list[dict[str, str]],
) -> dict[str, np.ndarray]:
    probabilities = {}
    for record in records:
        image_path = dataset / record["image"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        probability, _ = predictor.predict_image(image)
        probabilities[record["name"]] = probability
    return probabilities


def predictions_at(
    probabilities: dict[str, np.ndarray], threshold: float
) -> dict[str, np.ndarray]:
    return {name: value >= threshold for name, value in probabilities.items()}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    output = image.copy()
    active = mask > 0
    output[active] = (
        output[active].astype(np.float32) * 0.55
        + np.asarray(color, dtype=np.float32) * 0.45
    ).astype(np.uint8)
    contours, _ = cv2.findContours(
        active.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(output, contours, -1, (0, 0, 255), 2)
    return output


def make_contact_sheet(
    path: Path,
    dataset: Path,
    records: list[dict[str, str]],
    predictions: dict[str, np.ndarray],
) -> None:
    tiles = []
    for record in records:
        image = cv2.imread(str(dataset / record["image"]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(dataset / record["mask"]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(record["image"] if image is None else record["mask"])
        prediction = predictions[record["name"]].astype(np.uint8)
        if prediction.shape != mask.shape:
            prediction = cv2.resize(
                prediction,
                (mask.shape[1], mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        target_height = 280
        scale = target_height / image.shape[0]
        size = (max(1, round(image.shape[1] * scale)), target_height)
        views = (
            image,
            overlay(image, mask > 127, (0, 255, 0)),
            overlay(image, prediction, (255, 0, 0)),
        )
        tile = np.hstack(
            [cv2.resize(view, size, interpolation=cv2.INTER_AREA) for view in views]
        )
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 27), (0, 0, 0), -1)
        cv2.putText(
            tile,
            f"{record['category']} {record['name']}",
            (5, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    columns = 3
    rows = []
    blank = np.zeros_like(tiles[0])
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        rows.append(np.hstack(row + [blank] * (columns - len(row))))
    if not cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise IOError(f"Failed to write {path}")


def flatten_per_scene(
    model_name: str,
    threshold_source: str,
    split: str,
    metrics: dict,
) -> list[dict]:
    return [
        {
            "model": model_name,
            "threshold_source": threshold_source,
            "split": split,
            "name": name,
            **values,
        }
        for name, values in metrics["per_scene"].items()
    ]


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    models = parse_models(args.model)
    for checkpoint in models.values():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    val_records = read_records(dataset, "val")
    test_records = read_records(dataset, "test")
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    summary_models = {}
    per_scene_rows = []

    for model_index, (model_name, checkpoint) in enumerate(models.items(), start=1):
        predictor = RGBSegmenterPredictor(
            checkpoint,
            config.IMAGE_WIDTH,
            config.IMAGE_HEIGHT,
            device,
            args.no_amp,
        )
        val_probabilities = predict_split(predictor, dataset, val_records)
        test_probabilities = predict_split(predictor, dataset, test_records)
        frozen_threshold = predictor.threshold
        calibrated_threshold, sweep = select_threshold(
            dataset,
            val_records,
            val_probabilities,
            config.THRESHOLD_CANDIDATES,
            config.THRESHOLD_RECALL_FLOOR,
            config.BOUNDARY_TOLERANCE,
        )
        frozen_validation = aggregate_metrics(
            dataset,
            val_records,
            predictions_at(val_probabilities, frozen_threshold),
            config.BOUNDARY_TOLERANCE,
        )
        frozen_test = aggregate_metrics(
            dataset,
            test_records,
            predictions_at(test_probabilities, frozen_threshold),
            config.BOUNDARY_TOLERANCE,
        )
        calibrated_validation = aggregate_metrics(
            dataset,
            val_records,
            predictions_at(val_probabilities, calibrated_threshold),
            config.BOUNDARY_TOLERANCE,
        )
        calibrated_test_predictions = predictions_at(
            test_probabilities, calibrated_threshold
        )
        calibrated_test = aggregate_metrics(
            dataset,
            test_records,
            calibrated_test_predictions,
            config.BOUNDARY_TOLERANCE,
        )
        model_output = output / "models" / model_name
        model_output.mkdir(parents=True, exist_ok=False)
        write_csv(model_output / "right_val_threshold_sweep.csv", sweep)
        make_contact_sheet(
            model_output / "test_checkpoint_threshold_contact_sheet.jpg",
            dataset,
            test_records,
            predictions_at(test_probabilities, frozen_threshold),
        )
        make_contact_sheet(
            model_output / "test_right_calibrated_threshold_contact_sheet.jpg",
            dataset,
            test_records,
            calibrated_test_predictions,
        )
        for source, validation, test in (
            ("checkpoint_left_validation", frozen_validation, frozen_test),
            ("manual_right_validation", calibrated_validation, calibrated_test),
        ):
            per_scene_rows.extend(
                flatten_per_scene(model_name, source, "val", validation)
            )
            per_scene_rows.extend(
                flatten_per_scene(model_name, source, "test", test)
            )
        summary_models[model_name] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_threshold": frozen_threshold,
            "manual_right_validation_threshold": calibrated_threshold,
            "checkpoint_threshold_metrics": {
                "validation": frozen_validation,
                "frozen_test": frozen_test,
            },
            "manual_right_calibrated_metrics": {
                "validation": calibrated_validation,
                "frozen_test": calibrated_test,
            },
        }
        print(f"[{model_index}/{len(models)}] evaluated {model_name}", flush=True)

    write_csv(output / "per_scene.csv", per_scene_rows)
    summary = {
        "completed": True,
        "dataset": str(dataset),
        "dataset_annotation_manifest_sha256": json.loads(
            (dataset / "metadata.json").read_text(encoding="utf-8")
        )["annotation_manifest_sha256"],
        "counts": {"val": len(val_records), "test": len(test_records)},
        "selection_policy": (
            "model checkpoints remain frozen; optional threshold is selected only "
            "on the 15-image manual right validation split and evaluated on the "
            "17-image manual right test split"
        ),
        "models": summary_models,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
