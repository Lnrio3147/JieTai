#!/usr/bin/env python3
"""Validate and export reviewed right-view ISAT polygons as an evaluation set.

The source task pack contains disparity-projected prelabels.  Passing
``--annotator-confirmed`` records the external fact that a person reviewed all
tasks; geometric validation alone cannot establish semantic correctness.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

import config_experiment10 as config


FOREGROUND_CATEGORY = "workpiece"
BACKGROUND_CATEGORY = "__background__"
ALLOWED_CATEGORIES = {FOREGROUND_CATEGORY, BACKGROUND_CATEGORY}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=config.ROOT / "datasets/annotations/workpiece-right-isat-v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.RIGHT_MANUAL_DATASET,
    )
    parser.add_argument(
        "--annotator-confirmed",
        action="store_true",
        help="declare that every source JSON was manually reviewed in ISAT",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def resolve_task_path(source: Path, row: dict[str, str], kind: str) -> Path:
    """Resolve a task path without trusting stale absolute workspace prefixes."""

    declared = Path(row[kind])
    if declared.is_file():
        return declared.resolve()
    suffixes = {
        "right_image": ("images", ".png"),
        "isat_annotation": ("images", ".json"),
        "prelabel_mask": ("prelabels", ".png"),
    }
    directory, extension = suffixes[kind]
    repaired = (
        source
        / directory
        / row["split"]
        / row["category"]
        / f"{row['task_name']}{extension}"
    )
    if not repaired.is_file():
        raise FileNotFoundError(repaired)
    return repaired.resolve()


def polygon_points(
    item: dict,
    annotation_path: Path,
    object_index: int,
    width: int,
    height: int,
) -> np.ndarray:
    segmentation = item.get("segmentation")
    if not isinstance(segmentation, list) or len(segmentation) < 3:
        raise ValueError(
            f"Invalid polygon in {annotation_path}, object {object_index}"
        )
    points = np.asarray(segmentation, dtype=np.float64)
    if points.shape != (len(segmentation), 2) or not np.isfinite(points).all():
        raise ValueError(
            f"Invalid polygon points in {annotation_path}, object {object_index}"
        )
    # ISAT stores border vertices at subpixel coordinates such as 0.1 and may
    # place them up to width/height - 0.5.  Anything farther out is malformed.
    if (
        (points[:, 0] < -0.5).any()
        or (points[:, 0] > width - 0.5).any()
        or (points[:, 1] < -0.5).any()
        or (points[:, 1] > height - 0.5).any()
    ):
        raise ValueError(
            f"Out-of-bounds polygon in {annotation_path}, object {object_index}"
        )
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    polygon = np.rint(points).astype(np.int32)
    if cv2.contourArea(polygon) <= 0:
        raise ValueError(
            f"Degenerate polygon in {annotation_path}, object {object_index}"
        )
    return polygon


def rasterize_isat(
    annotation: dict,
    annotation_path: Path,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict]:
    """Rasterize ISAT layers, including explicit background erase polygons."""

    info = annotation.get("info", {})
    if (info.get("width"), info.get("height")) != (width, height):
        raise ValueError(
            f"Annotation/image size mismatch in {annotation_path}: "
            f"annotation={info.get('width')}x{info.get('height')}, "
            f"image={width}x{height}"
        )
    objects = annotation.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError(f"No annotated objects in {annotation_path}")
    ordered = sorted(
        enumerate(objects), key=lambda pair: (float(pair[1].get("layer", 0)), pair[0])
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    category_counts: Counter[str] = Counter()
    point_count = 0
    for object_index, item in ordered:
        category = item.get("category")
        if category not in ALLOWED_CATEGORIES:
            raise ValueError(
                f"Unexpected category {category!r} in {annotation_path}, "
                f"object {object_index}"
            )
        polygon = polygon_points(
            item, annotation_path, object_index, width, height
        )
        cv2.fillPoly(
            mask,
            [polygon],
            color=255 if category == FOREGROUND_CATEGORY else 0,
        )
        category_counts[category] += 1
        point_count += len(polygon)
    if category_counts[FOREGROUND_CATEGORY] == 0 or not np.any(mask):
        raise ValueError(f"No non-empty workpiece region in {annotation_path}")
    components = int(cv2.connectedComponents((mask > 0).astype(np.uint8))[0] - 1)
    return mask, {
        "object_count": len(objects),
        "foreground_object_count": category_counts[FOREGROUND_CATEGORY],
        "background_erase_object_count": category_counts[BACKGROUND_CATEGORY],
        "point_count": point_count,
        "foreground_fraction": float(np.mean(mask > 0)),
        "connected_components": components,
    }


def prelabel_metrics(mask: np.ndarray, prelabel: np.ndarray) -> dict[str, float]:
    human = mask > 0
    projected = prelabel > 127
    intersection = int((human & projected).sum())
    union = int((human | projected).sum())
    return {
        "prelabel_iou": intersection / max(union, 1),
        "changed_pixel_fraction": float(np.mean(human != projected)),
    }


def qa_tile(image: np.ndarray, mask: np.ndarray, prelabel: np.ndarray, label: str) -> np.ndarray:
    overlay = image.copy()
    foreground = mask > 0
    overlay[foreground] = (
        0.55 * overlay[foreground]
        + 0.45 * np.asarray([0, 255, 0], dtype=np.float32)
    ).astype(np.uint8)
    human_contours, _ = cv2.findContours(
        foreground.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    projected_contours, _ = cv2.findContours(
        (prelabel > 127).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, projected_contours, -1, (0, 255, 255), 2)
    cv2.drawContours(overlay, human_contours, -1, (0, 0, 255), 2)
    target_height = 320
    scale = target_height / image.shape[0]
    size = (max(1, round(image.shape[1] * scale)), target_height)
    left = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    right = cv2.resize(overlay, size, interpolation=cv2.INTER_AREA)
    tile = np.hstack((left, right))
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        tile,
        label,
        (5, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def write_contact_sheet(path: Path, tiles: list[np.ndarray], columns: int = 4) -> None:
    if not tiles:
        raise ValueError("Cannot make an empty QA contact sheet")
    rows = []
    blank = np.zeros_like(tiles[0])
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        rows.append(np.hstack(row + [blank] * (columns - len(row))))
    if not cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise IOError(f"Failed to write {path}")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames = fields or sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)


def main() -> None:
    args = parse_args()
    if not args.annotator_confirmed:
        raise ValueError(
            "Refusing to label prelabels as human truth without --annotator-confirmed"
        )
    source = args.source.resolve()
    output = args.output.resolve()
    tasks_path = source / "tasks.csv"
    with tasks_path.open(encoding="utf-8", newline="") as stream:
        tasks = list(csv.DictReader(stream))
    if not tasks:
        raise ValueError(f"Empty task index: {tasks_path}")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    prepared: dict[str, list[dict]] = {"val": [], "test": []}
    qa_tiles: dict[str, list[np.ndarray]] = {"val": [], "test": []}
    try:
        for split in prepared:
            for directory in ("images", "masks", "annotations"):
                (output / directory / split).mkdir(parents=True, exist_ok=False)
            (output / "qa").mkdir(parents=True, exist_ok=split != "val")
        (output / "index").mkdir(parents=True, exist_ok=False)

        seen_names: set[str] = set()
        for task in tasks:
            split = task["split"]
            if split not in prepared:
                raise ValueError(f"Unexpected task split {split!r}")
            task_name = task["task_name"]
            if task_name in seen_names:
                raise ValueError(f"Duplicate task name: {task_name}")
            seen_names.add(task_name)
            image_path = resolve_task_path(source, task, "right_image")
            annotation_path = resolve_task_path(source, task, "isat_annotation")
            prelabel_path = resolve_task_path(source, task, "prelabel_mask")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            prelabel = cv2.imread(str(prelabel_path), cv2.IMREAD_GRAYSCALE)
            if image is None or prelabel is None:
                raise FileNotFoundError(image_path if image is None else prelabel_path)
            if prelabel.shape != image.shape[:2]:
                raise ValueError(f"Prelabel/image size mismatch for {task_name}")
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            if annotation.get("info", {}).get("name") != image_path.name:
                raise ValueError(f"Annotation image name mismatch in {annotation_path}")
            mask, stats = rasterize_isat(
                annotation,
                annotation_path,
                image.shape[1],
                image.shape[0],
            )
            comparison = prelabel_metrics(mask, prelabel)

            category = task["category"]
            destinations = {
                "image": output / "images" / split / category / image_path.name,
                "mask": output / "masks" / split / category / image_path.name,
                "annotation": (
                    output / "annotations" / split / category / annotation_path.name
                ),
            }
            for destination in destinations.values():
                destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, destinations["image"])
            shutil.copy2(annotation_path, destinations["annotation"])
            if not cv2.imwrite(str(destinations["mask"]), mask):
                raise IOError(f"Failed to write {destinations['mask']}")
            qa_tiles[split].append(
                qa_tile(
                    image,
                    mask,
                    prelabel,
                    f"{split}/{category} {task_name}",
                )
            )
            prepared[split].append(
                {
                    "name": task_name,
                    "source_name": task["source_name"],
                    "capture_group": task["capture_group"],
                    "category": category,
                    "split": split,
                    "view": "right_human",
                    "label_source": "human_isat_polygon",
                    "manual_status": "completed",
                    "image": str(destinations["image"].relative_to(output)),
                    "mask": str(destinations["mask"].relative_to(output)),
                    "annotation": str(destinations["annotation"].relative_to(output)),
                    "source_image": str(image_path),
                    "source_annotation": str(annotation_path),
                    "image_sha256": sha256_file(image_path),
                    "annotation_sha256": sha256_file(annotation_path),
                    "mask_sha256": sha256_file(destinations["mask"]),
                    **stats,
                    **comparison,
                }
            )

        all_records = prepared["val"] + prepared["test"]
        for split, records in prepared.items():
            records.sort(key=lambda row: (row["category"], row["name"]))
            write_csv(output / "index" / f"{split}.csv", records)
            write_contact_sheet(output / "qa" / f"{split}_contact_sheet.jpg", qa_tiles[split])
        write_contact_sheet(
            output / "qa" / "all_contact_sheet.jpg",
            qa_tiles["val"] + qa_tiles["test"],
        )
        write_csv(output / "annotation_audit.csv", all_records)
        manifest_path = output / "index" / "annotations.sha256"
        with manifest_path.open("w", encoding="utf-8") as stream:
            for record in sorted(all_records, key=lambda row: row["name"]):
                stream.write(
                    f"{record['annotation_sha256']}  {record['annotation']}\n"
                )
        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "name": "workpiece-right-manual-isat-v1",
            "purpose": "held-out right-view segmentation and stereo-mask evaluation",
            "source_task_pack": str(source),
            "annotator_confirmation_supplied": True,
            "geometric_qa_passed": True,
            "semantic_qa_note": (
                "Manual completion was declared by the annotator; this exporter "
                "checks structure and geometry, not semantic boundary accuracy."
            ),
            "class_mapping": {BACKGROUND_CATEGORY: 0, FOREGROUND_CATEGORY: 255},
            "layer_policy": "ascending ISAT layer; __background__ erases foreground",
            "counts": {split: len(records) for split, records in prepared.items()},
            "category_counts": dict(
                sorted(Counter(row["category"] for row in all_records).items())
            ),
            "object_count": int(sum(row["object_count"] for row in all_records)),
            "background_erase_object_count": int(
                sum(row["background_erase_object_count"] for row in all_records)
            ),
            "foreground_fraction": summarize(
                [row["foreground_fraction"] for row in all_records]
            ),
            "prelabel_iou": summarize([row["prelabel_iou"] for row in all_records]),
            "changed_pixel_fraction": summarize(
                [row["changed_pixel_fraction"] for row in all_records]
            ),
            "annotation_manifest": str(manifest_path.relative_to(output)),
            "annotation_manifest_sha256": sha256_file(manifest_path),
        }
        (output / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
