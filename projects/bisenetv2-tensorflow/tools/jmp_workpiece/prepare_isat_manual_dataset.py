#!/usr/bin/env python3
"""Convert ISAT workpiece polygons into a versioned binary segmentation dataset."""

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


METHOD_NAME = "isat_polygon_rasterization_v1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source_dir",
        required=True,
        help="directory containing same-name ISAT .png and .json files",
    )
    parser.add_argument("--output_dir", required=True, help="new versioned dataset directory")
    parser.add_argument("--foreground_category", default="jinshu")
    parser.add_argument(
        "--val_groups",
        nargs="+",
        default=["fdjyp_0_2", "fdjyp_2_3"],
        help="complete capture groups reserved for validation",
    )
    parser.add_argument("--width", type=int, default=288)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--qa_samples", type=int, default=24)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_group(name):
    parts = name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unable to derive capture group from {name!r}")
    return "_".join(parts[:3])


def rasterize_isat(annotation, annotation_path, width, height, category):
    info = annotation.get("info", {})
    if (info.get("width"), info.get("height")) != (width, height):
        raise ValueError(
            f"Unexpected annotation size in {annotation_path}: "
            f"{info.get('width')}x{info.get('height')}"
        )

    objects = annotation.get("objects", [])
    if not objects:
        raise ValueError(f"No annotated objects in {annotation_path}")

    mask = np.zeros((height, width), dtype=np.uint8)
    point_count = 0
    annotated_area = 0.0
    for object_index, item in enumerate(objects):
        object_category = item.get("category")
        if object_category != category:
            raise ValueError(
                f"Unexpected category {object_category!r} in {annotation_path}, "
                f"object {object_index}; expected {category!r}"
            )
        segmentation = item.get("segmentation")
        if not isinstance(segmentation, list) or len(segmentation) < 3:
            raise ValueError(f"Invalid polygon in {annotation_path}, object {object_index}")
        points = np.asarray(segmentation, dtype=np.float64)
        if points.shape != (len(segmentation), 2) or not np.isfinite(points).all():
            raise ValueError(f"Invalid polygon points in {annotation_path}, object {object_index}")
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        polygon = np.rint(points).astype(np.int32)
        cv2.fillPoly(mask, [polygon], color=255)
        point_count += len(polygon)
        annotated_area += float(item.get("area", cv2.contourArea(polygon)))

    return mask, {
        "object_count": len(objects),
        "point_count": point_count,
        "annotated_area": annotated_area,
        "foreground_fraction": float(np.mean(mask == 255)),
    }


def make_qa_tile(image, mask, name):
    overlay = image.copy()
    foreground = mask > 0
    overlay[foreground] = (
        0.45 * overlay[foreground]
        + 0.55 * np.array([0, 0, 255], dtype=np.float32)
    ).astype(np.uint8)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    tile = np.hstack([image, overlay, mask_bgr])
    cv2.putText(
        tile,
        name,
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return tile


def save_contact_sheet(records, output_path, sample_count):
    count = min(sample_count, len(records))
    indices = np.linspace(0, len(records) - 1, num=count, dtype=np.int32)
    tiles = []
    for index in indices:
        record = records[int(index)]
        image = cv2.imread(record["prepared_image"], cv2.IMREAD_COLOR)
        mask = cv2.imread(record["prepared_mask"], cv2.IMREAD_GRAYSCALE)
        tile = make_qa_tile(image, mask, record["name"])
        tiles.append(cv2.resize(tile, (432, 256), interpolation=cv2.INTER_AREA))

    columns = 3
    rows = []
    for start in range(0, len(tiles), columns):
        row_tiles = tiles[start:start + columns]
        while len(row_tiles) < columns:
            row_tiles.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row_tiles))
    cv2.imwrite(str(output_path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def main():
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    val_groups = set(args.val_groups)

    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}. Use a new version to preserve reproducibility."
        )

    annotation_paths = sorted(source_dir.glob("*.json"))
    image_names = {path.stem for path in source_dir.glob("*.png")}
    annotation_names = {path.stem for path in annotation_paths}
    missing_annotations = sorted(image_names - annotation_names)
    missing_images = sorted(annotation_names - image_names)
    if missing_annotations or missing_images:
        raise ValueError(
            f"Image/annotation mismatch: missing_annotations={missing_annotations}, "
            f"missing_images={missing_images}"
        )
    if not annotation_paths:
        raise ValueError(f"No ISAT JSON annotations found in {source_dir}")

    all_groups = {capture_group(path.stem) for path in annotation_paths}
    unknown_val_groups = sorted(val_groups - all_groups)
    if unknown_val_groups:
        raise ValueError(f"Validation groups not present: {unknown_val_groups}")
    if val_groups == all_groups:
        raise ValueError("Validation groups consume the entire dataset")

    prepared = {"train": [], "val": []}
    for split in prepared:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=False)
        (output_dir / "masks" / split).mkdir(parents=True, exist_ok=False)
        (output_dir / "annotations" / split).mkdir(parents=True, exist_ok=False)
    (output_dir / "index").mkdir(parents=True, exist_ok=False)
    (output_dir / "qa").mkdir(parents=True, exist_ok=False)

    try:
        for annotation_path in annotation_paths:
            name = annotation_path.stem
            image_path = source_dir / f"{name}.png"
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            if annotation.get("info", {}).get("name") != image_path.name:
                raise ValueError(f"Annotation image name mismatch in {annotation_path}")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (args.height, args.width):
                raise ValueError(
                    f"Unexpected image shape for {image_path}: "
                    f"{None if image is None else image.shape}"
                )
            mask, stats = rasterize_isat(
                annotation,
                annotation_path,
                args.width,
                args.height,
                args.foreground_category,
            )

            group = capture_group(name)
            split = "val" if group in val_groups else "train"
            prepared_image = output_dir / "images" / split / image_path.name
            prepared_mask = output_dir / "masks" / split / image_path.name
            prepared_annotation = output_dir / "annotations" / split / annotation_path.name
            shutil.copy2(image_path, prepared_image)
            shutil.copy2(annotation_path, prepared_annotation)
            if not cv2.imwrite(str(prepared_mask), mask):
                raise OSError(f"Failed to write {prepared_mask}")

            prepared[split].append(
                {
                    "name": name,
                    "image": str(prepared_image.relative_to(output_dir)),
                    "mask": str(prepared_mask.relative_to(output_dir)),
                    "annotation": str(prepared_annotation.relative_to(output_dir)),
                    "capture_group": group,
                    "label_source": "human_isat_polygon",
                    "source_image": str(image_path),
                    "source_annotation": str(annotation_path),
                    "image_sha256": sha256_file(image_path),
                    "annotation_sha256": sha256_file(annotation_path),
                    "mask_sha256": sha256_file(prepared_mask),
                    "prepared_image": str(prepared_image),
                    "prepared_mask": str(prepared_mask),
                    **stats,
                }
            )

        fieldnames = [
            "name",
            "image",
            "mask",
            "annotation",
            "capture_group",
            "label_source",
            "source_image",
            "source_annotation",
            "image_sha256",
            "annotation_sha256",
            "mask_sha256",
            "object_count",
            "point_count",
            "annotated_area",
            "foreground_fraction",
        ]
        for split, records in prepared.items():
            index_path = output_dir / "index" / f"{split}.csv"
            with index_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for record in records:
                    writer.writerow({key: record[key] for key in fieldnames})
            save_contact_sheet(records, output_dir / "qa" / f"{split}_contact_sheet.jpg", args.qa_samples)

        all_records = prepared["train"] + prepared["val"]
        annotation_manifest = output_dir / "index" / "annotations.sha256"
        with annotation_manifest.open("w", encoding="utf-8") as handle:
            for record in sorted(all_records, key=lambda item: item["name"]):
                handle.write(f"{record['annotation_sha256']}  {record['annotation']}\n")

        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "method": METHOD_NAME,
            "source_dir": str(source_dir),
            "foreground_category": args.foreground_category,
            "class_mapping": {"background": 0, "workpiece": 255},
            "image_size": {"width": args.width, "height": args.height},
            "split_policy": "hold_out_complete_capture_groups",
            "validation_groups": sorted(val_groups),
            "all_group_counts": dict(
                sorted(Counter(record["capture_group"] for record in all_records).items())
            ),
            "counts": {split: len(records) for split, records in prepared.items()},
            "object_count": int(sum(record["object_count"] for record in all_records)),
            "foreground_fraction": summarize(
                [record["foreground_fraction"] for record in all_records]
            ),
            "annotation_manifest": str(annotation_manifest.relative_to(output_dir)),
            "annotation_manifest_sha256": sha256_file(annotation_manifest),
        }
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

        print(json.dumps(metadata, ensure_ascii=False, indent=2))
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
