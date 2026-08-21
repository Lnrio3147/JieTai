#!/usr/bin/env python3
"""Prepare a reproducible stratified binary segmentation dataset from ISAT JSON."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


CATEGORIES = ("fdjyp3", "luowen", "general", "scale", "jop1")
BACKGROUND = "__background__"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "datasets/annotations/workpiece_isat_v2/images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "datasets/training/workpiece-seg-isat-v2",
    )
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--width", type=int, default=288)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--qa-samples", type=int, default=30)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def category_from_name(name: str) -> str:
    category = name.split("_", 1)[0].lower()
    if category not in CATEGORIES:
        raise ValueError("Unknown filename category: {}".format(name))
    return category


def stratified_split(paths: list[Path], seed: int, val_ratio: float, test_ratio: float):
    if not (0.0 < val_ratio < 0.5 and 0.0 < test_ratio < 0.5):
        raise ValueError("Validation and test ratios must be between 0 and 0.5")
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        grouped[category_from_name(path.name)].append(path)
    if set(grouped) != set(CATEGORIES):
        raise ValueError("Missing categories: {}".format(sorted(set(CATEGORIES) - set(grouped))))

    rng = random.Random(seed)
    assignments = {}
    counts = {}
    for category in CATEGORIES:
        category_paths = sorted(grouped[category])
        rng.shuffle(category_paths)
        count = len(category_paths)
        test_count = max(1, int(math.ceil(count * test_ratio)))
        val_count = max(1, int(math.ceil(count * val_ratio)))
        if test_count + val_count >= count:
            raise ValueError("Category {} is too small to form three splits".format(category))
        for path in category_paths[:test_count]:
            assignments[path] = "test"
        for path in category_paths[test_count : test_count + val_count]:
            assignments[path] = "val"
        for path in category_paths[test_count + val_count :]:
            assignments[path] = "train"
        counts[category] = {
            "train": count - test_count - val_count,
            "val": val_count,
            "test": test_count,
        }
    return assignments, counts


def validate_points(segmentation, path: Path, object_index: int, width: int, height: int):
    points = np.asarray(segmentation, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("Invalid polygon in {} object {}".format(path, object_index))
    if not np.isfinite(points).all():
        raise ValueError("Non-finite polygon in {} object {}".format(path, object_index))
    # iSAT may store a boundary vertex at -0.5 or width-0.5. Clipping is the
    # same convention used for the earlier manually labelled training set.
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    return np.rint(points).astype(np.int32)


def rasterize(annotation: dict, path: Path, width: int, height: int):
    info = annotation.get("info", {})
    if (info.get("width"), info.get("height")) != (width, height):
        raise ValueError("Image/JSON size mismatch: {}".format(path))
    objects = annotation.get("objects", [])
    if not objects:
        raise ValueError("Empty annotation: {}".format(path))

    ordered = sorted(
        enumerate(objects),
        key=lambda pair: (float(pair[1].get("layer", pair[0] + 1)), pair[0]),
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    category_counts = Counter()
    point_count = 0
    for object_index, item in ordered:
        category = item.get("category")
        if category not in set(CATEGORIES) | {BACKGROUND}:
            raise ValueError(
                "Unexpected category {!r} in {} object {}".format(
                    category, path, object_index
                )
            )
        polygon = validate_points(
            item.get("segmentation"), path, object_index, width, height
        )
        cv2.fillPoly(mask, [polygon], 0 if category == BACKGROUND else 255)
        category_counts[category] += 1
        point_count += len(polygon)
    return mask, category_counts, point_count


def qa_tile(image: np.ndarray, mask: np.ndarray, title: str) -> np.ndarray:
    overlay = image.copy()
    foreground = mask > 0
    overlay[foreground] = (
        0.45 * overlay[foreground]
        + 0.55 * np.array([0, 0, 255], dtype=np.float32)
    ).astype(np.uint8)
    tile = np.hstack([image, overlay, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
    cv2.putText(
        tile,
        title,
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return cv2.resize(tile, (432, 256), interpolation=cv2.INTER_AREA)


def save_contact_sheet(records: list[dict], output: Path, sample_count: int) -> None:
    count = min(sample_count, len(records))
    indices = np.linspace(0, len(records) - 1, num=count, dtype=np.int32)
    tiles = []
    for index in indices:
        record = records[int(index)]
        image = cv2.imread(record["prepared_image"], cv2.IMREAD_COLOR)
        mask = cv2.imread(record["prepared_mask"], cv2.IMREAD_GRAYSCALE)
        tiles.append(qa_tile(image, mask, "{} [{}]".format(record["name"], record["category"])))
    rows = []
    blank = np.zeros_like(tiles[0])
    for start in range(0, len(tiles), 3):
        row = tiles[start : start + 3]
        row.extend([blank] * (3 - len(row)))
        rows.append(np.hstack(row))
    cv2.imwrite(str(output), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a new dataset version; output exists: {}".format(output))

    image_paths = sorted(source.glob("*/*/*.png"))
    json_paths = sorted(source.glob("*/*/*.json"))
    if len(image_paths) != 130 or len(json_paths) != 130:
        raise RuntimeError(
            "Expected 130 images and JSON files, found {} and {}".format(
                len(image_paths), len(json_paths)
            )
        )
    image_stems = {path.relative_to(source).with_suffix("") for path in image_paths}
    json_stems = {path.relative_to(source).with_suffix("") for path in json_paths}
    if image_stems != json_stems:
        raise RuntimeError("Image/annotation pairing mismatch")

    assignments, split_counts = stratified_split(
        image_paths, args.seed, args.val_ratio, args.test_ratio
    )
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=False)
        (output / "masks" / split).mkdir(parents=True, exist_ok=False)
        (output / "annotations" / split).mkdir(parents=True, exist_ok=False)
    (output / "index").mkdir(parents=True, exist_ok=False)
    (output / "qa").mkdir(parents=True, exist_ok=False)

    records_by_split = {"train": [], "val": [], "test": []}
    object_categories = Counter()
    for image_path in image_paths:
        category = category_from_name(image_path.name)
        split = assignments[image_path]
        annotation_path = image_path.with_suffix(".json")
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        if annotation.get("info", {}).get("name") != image_path.name:
            raise ValueError("JSON filename mismatch: {}".format(annotation_path))
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        source_height, source_width = image.shape[:2]
        mask, category_counts, point_count = rasterize(
            annotation, annotation_path, source_width, source_height
        )
        foreground_categories = set(category_counts) - {BACKGROUND}
        if foreground_categories != {category}:
            raise ValueError(
                "Filename/object category mismatch in {}: {}".format(
                    annotation_path, sorted(foreground_categories)
                )
            )
        object_categories.update(category_counts)

        prepared_image = cv2.resize(
            image, (args.width, args.height), interpolation=cv2.INTER_AREA
        )
        prepared_mask = cv2.resize(
            mask, (args.width, args.height), interpolation=cv2.INTER_NEAREST
        )
        if not set(np.unique(prepared_mask).tolist()) <= {0, 255}:
            raise ValueError("Non-binary resized mask: {}".format(annotation_path))

        image_output = output / "images" / split / image_path.name
        mask_output = output / "masks" / split / image_path.name
        annotation_output = output / "annotations" / split / annotation_path.name
        if not cv2.imwrite(str(image_output), prepared_image):
            raise OSError(image_output)
        if not cv2.imwrite(str(mask_output), prepared_mask):
            raise OSError(mask_output)
        shutil.copy2(annotation_path, annotation_output)
        source_relative = image_path.relative_to(source).as_posix()
        record = {
            "name": image_path.stem,
            "category": category,
            "split": split,
            "image": image_output.relative_to(output).as_posix(),
            "mask": mask_output.relative_to(output).as_posix(),
            "annotation": annotation_output.relative_to(output).as_posix(),
            "source_image": str(image_path),
            "source_annotation": str(annotation_path),
            "source_relative": source_relative,
            "image_sha256": sha256_file(image_path),
            "annotation_sha256": sha256_file(annotation_path),
            "mask_sha256": sha256_file(mask_output),
            "object_count": sum(category_counts.values()),
            "background_object_count": category_counts[BACKGROUND],
            "point_count": point_count,
            "foreground_fraction": "{:.8f}".format(float(np.mean(prepared_mask > 0))),
            "prepared_image": str(image_output),
            "prepared_mask": str(mask_output),
        }
        records_by_split[split].append(record)

    fieldnames = [
        "name", "category", "split", "image", "mask", "annotation",
        "source_image", "source_annotation", "source_relative", "image_sha256",
        "annotation_sha256", "mask_sha256", "object_count",
        "background_object_count", "point_count", "foreground_fraction",
    ]
    for split, records in records_by_split.items():
        records.sort(key=lambda record: (record["category"], record["name"]))
        with (output / "index" / "{}.csv".format(split)).open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(
                {key: record[key] for key in fieldnames} for record in records
            )
        save_contact_sheet(
            records, output / "qa" / "{}_contact_sheet.jpg".format(split), args.qa_samples
        )

    all_records = sum(records_by_split.values(), [])
    manifest = output / "index" / "annotations.sha256"
    with manifest.open("w", encoding="utf-8") as stream:
        for record in sorted(all_records, key=lambda record: record["source_relative"]):
            stream.write("{}  {}\n".format(record["annotation_sha256"], record["source_relative"]))
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "seed": args.seed,
        "split_policy": "stratified_random_by_filename_category",
        "val_ratio_requested": args.val_ratio,
        "test_ratio_requested": args.test_ratio,
        "split_counts_by_category": split_counts,
        "counts": {split: len(records) for split, records in records_by_split.items()},
        "categories": list(CATEGORIES),
        "class_mapping": {"background": 0, "workpiece": 255},
        "image_size": {"width": args.width, "height": args.height},
        "source_image_size": {"width": 720, "height": 1280},
        "rasterization": "paint polygons by ascending ISAT layer; background polygons erase foreground",
        "explicit_background_polygon_count": object_categories[BACKGROUND],
        "hole_label_scope": (
            "No explicit __background__ polygons are present because annotation covers the "
            "outer subject only. Internal-hole extraction is outside this supervised target."
            if object_categories[BACKGROUND] == 0
            else None
        ),
        "object_category_counts": dict(sorted(object_categories.items())),
        "foreground_fraction": summarize(
            [float(record["foreground_fraction"]) for record in all_records]
        ),
        "annotation_manifest": manifest.relative_to(output).as_posix(),
        "annotation_manifest_sha256": sha256_file(manifest),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
