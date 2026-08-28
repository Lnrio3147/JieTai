#!/usr/bin/env python3
"""Create a stratified ISAT task pack for manual right-view mask correction.

The generated polygons are disparity-projected prelabels only.  They must be
reviewed by a person before being used as right-view ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import config_experiment10 as config
from prepare_ablation_manifest import right_image_for
from prepare_stereo_segmentation_dataset import (
    project_mask_to_right,
    read_disparity,
)
from utils.data import read_records


ISAT_YAML = """label:
- color: '#000000'
  name: __background__
- color: '#00ff00'
  name: workpiece
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=config.DATASET_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=config.ROOT / "datasets/annotations/workpiece-right-isat-v1",
    )
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument(
        "--splits", nargs="+", choices=("val", "test"), default=("val", "test")
    )
    return parser.parse_args()


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    """Select deterministic low/median/high ranks without duplicates."""

    if length <= 0 or count <= 0:
        return []
    count = min(length, count)
    if count == 1:
        return [length // 2]
    return sorted(
        {
            int(round(value))
            for value in np.linspace(0, length - 1, count, dtype=np.float64)
        }
    )


def mask_to_isat_objects(mask: np.ndarray) -> list[dict]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    objects = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = float(cv2.contourArea(contour))
        if area < 25.0:
            continue
        epsilon = max(1.0, 0.001 * cv2.arcLength(contour, closed=True))
        polygon = cv2.approxPolyDP(contour, epsilon, closed=True).reshape(-1, 2)
        if len(polygon) < 3:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        objects.append(
            {
                "category": "workpiece",
                "group": 1,
                "segmentation": polygon.astype(float).tolist(),
                "area": area,
                "layer": float(len(objects) + 1),
                "bbox": [
                    float(x) - 0.5,
                    float(y) - 0.5,
                    float(x + width) - 0.5,
                    float(y + height) - 0.5,
                ],
                "iscrowd": False,
                "note": "disparity-projected prelabel; manual correction required",
            }
        )
    return objects


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    color = np.zeros_like(image)
    color[..., 1] = 255
    foreground = mask > 0
    overlay[foreground] = cv2.addWeighted(
        image[foreground], 0.55, color[foreground], 0.45, 0.0
    )
    contours, _ = cv2.findContours(
        foreground.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    return overlay


def write_contact_sheet(path: Path, rows: list[dict], columns: int = 4) -> None:
    cell_width, image_height, label_height = 240, 320, 42
    cell_height = image_height + label_height
    sheet_rows = (len(rows) + columns - 1) // columns
    canvas = np.full(
        (sheet_rows * cell_height, columns * cell_width, 3), 245, dtype=np.uint8
    )
    for index, row in enumerate(rows):
        preview = cv2.imread(row["preview"], cv2.IMREAD_COLOR)
        if preview is None:
            raise FileNotFoundError(row["preview"])
        scale = min(cell_width / preview.shape[1], image_height / preview.shape[0])
        resized = cv2.resize(
            preview,
            (max(1, round(preview.shape[1] * scale)), max(1, round(preview.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        row_index, column_index = divmod(index, columns)
        x0 = column_index * cell_width + (cell_width - resized.shape[1]) // 2
        y0 = row_index * cell_height + (image_height - resized.shape[0]) // 2
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        label = (
            f"{row['split']}/{row['category']} "
            f"q={float(row['projection_quality']):.2f}"
        )
        cv2.putText(
            canvas,
            label,
            (column_index * cell_width + 5, row_index * cell_height + image_height + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            row["source_name"][-22:],
            (column_index * cell_width + 5, row_index * cell_height + image_height + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(path), canvas):
        raise IOError(f"Failed to write {path}")


def main() -> None:
    args = parse_args()
    if args.per_category <= 0:
        raise ValueError("--per-category must be positive")
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    candidates = []
    for split in args.splits:
        for record in read_records(dataset, split):
            left_image = Path(record["source_image"]).resolve()
            right_image = right_image_for(left_image)
            source_mask = Path(record["source_mask"]).resolve()
            disparity_path = Path(record["source_disparity"]).resolve()
            image = cv2.imread(str(left_image), cv2.IMREAD_COLOR)
            mask_image = cv2.imread(str(source_mask), cv2.IMREAD_GRAYSCALE)
            if image is None or mask_image is None:
                raise FileNotFoundError(left_image if image is None else source_mask)
            height, width = image.shape[:2]
            mask = cv2.resize(
                mask_image, (width, height), interpolation=cv2.INTER_NEAREST
            ) > 127
            projected, diagnostics = project_mask_to_right(
                mask, read_disparity(disparity_path)
            )
            quality = min(
                diagnostics["valid_fraction"], diagnostics["inside_fraction"]
            )
            candidates.append(
                {
                    "record": record,
                    "split": split,
                    "right_image": right_image,
                    "projected": projected,
                    "diagnostics": diagnostics,
                    "projection_quality": float(quality),
                }
            )

    grouped = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate["split"], candidate["record"]["category"])].append(
            candidate
        )
    selected = []
    for key in sorted(grouped):
        ranked = sorted(
            grouped[key],
            key=lambda item: (
                item["projection_quality"],
                item["record"]["capture_group"],
                item["record"]["name"],
            ),
        )
        selected.extend(
            ranked[index]
            for index in evenly_spaced_indices(len(ranked), args.per_category)
        )

    rows = []
    for candidate in selected:
        record = candidate["record"]
        split = candidate["split"]
        category = record["category"]
        task_name = f"{record['name']}_right"
        image_dir = output / "images" / split / category
        prelabel_dir = output / "prelabels" / split / category
        preview_dir = output / "previews" / split / category
        for directory in (image_dir, prelabel_dir, preview_dir):
            directory.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{task_name}.png"
        shutil.copy2(candidate["right_image"], image_path)
        mask_path = prelabel_dir / f"{task_name}.png"
        cv2.imwrite(str(mask_path), candidate["projected"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        preview_path = preview_dir / f"{task_name}.jpg"
        cv2.imwrite(str(preview_path), make_overlay(image, candidate["projected"]))
        payload = {
            "info": {
                "description": "ISAT",
                "folder": str(image_dir),
                "name": image_path.name,
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "depth": int(image.shape[2]),
                "note": (
                    "right-view disparity-projected prelabel; manual correction "
                    "required before evaluation"
                ),
            },
            "objects": mask_to_isat_objects(candidate["projected"]),
        }
        annotation_path = image_path.with_suffix(".json")
        annotation_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (image_dir / "isat.yaml").write_text(ISAT_YAML, encoding="utf-8")
        rows.append(
            {
                "task_name": task_name,
                "split": split,
                "category": category,
                "capture_group": record["capture_group"],
                "source_name": record["name"],
                "right_image": str(image_path),
                "isat_annotation": str(annotation_path),
                "prelabel_mask": str(mask_path),
                "preview": str(preview_path),
                "manual_status": "pending",
                "projection_quality": candidate["projection_quality"],
                **candidate["diagnostics"],
            }
        )

    write_csv(output / "tasks.csv", rows)
    write_contact_sheet(output / "task_contact_sheet.jpg", rows)
    category_counts = defaultdict(int)
    split_counts = defaultdict(int)
    for row in rows:
        category_counts[row["category"]] += 1
        split_counts[row["split"]] += 1
    metadata = {
        "name": "workpiece-right-isat-v1",
        "purpose": "manual right-view correction and held-out stereo-mask evaluation",
        "source_dataset": str(dataset),
        "task_count": len(rows),
        "per_category_limit_per_split": args.per_category,
        "split_counts": dict(sorted(split_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "selection": (
            "deterministic low/median/high disparity-projection quality ranks "
            "within each split and category"
        ),
        "warning": "Every JSON polygon is a prelabel; manual_status starts as pending.",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
