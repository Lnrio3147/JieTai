#!/usr/bin/env python3
"""Prepare a deterministic binary workpiece pseudo-segmentation dataset.

The source JMP manifest contains stereo-valid masks, not semantic masks.  This
script deliberately creates weak labels from the rectified left RGB image with
Otsu thresholding followed by GrabCut.  The resulting labels are suitable only
for a reproducible baseline experiment and must not be treated as human ground
truth.
"""

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


METHOD_NAME = "otsu_grabcut_v1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="LiteAnyStereo JMP manifest.csv")
    parser.add_argument("--output_dir", required=True, help="new prepared dataset directory")
    parser.add_argument("--width", type=int, default=288)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--source_prefix", default="fdjyp_", help="only include matching scene names")
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--grabcut_iters", type=int, default=4)
    parser.add_argument("--min_component_area", type=int, default=500)
    parser.add_argument("--qa_samples", type=int, default=24)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_pseudo_mask(image, grabcut_iters, min_component_area, seed):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    otsu_threshold, otsu_mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    seed_mask = cv2.morphologyEx(
        otsu_mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8)
    )
    dilated = cv2.dilate(seed_mask, np.ones((21, 21), dtype=np.uint8))
    eroded = cv2.erode(seed_mask, np.ones((5, 5), dtype=np.uint8))

    grabcut_mask = np.full(gray.shape, cv2.GC_BGD, dtype=np.uint8)
    grabcut_mask[dilated > 0] = cv2.GC_PR_FGD
    grabcut_mask[eroded > 0] = cv2.GC_FGD
    grabcut_mask[(gray < 20) & (eroded == 0)] = cv2.GC_BGD

    cv2.setRNGSeed(int(seed % (2**31 - 1)))
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            image,
            grabcut_mask,
            None,
            background_model,
            foreground_model,
            grabcut_iters,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        grabcut_mask = np.where(seed_mask > 0, cv2.GC_FGD, cv2.GC_BGD).astype(np.uint8)

    mask = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    cleaned = np.zeros_like(mask)
    kept_components = 0
    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area >= min_component_area:
            cleaned[labels == component_id] = 255
            kept_components += 1

    return cleaned, {
        "otsu_threshold": float(otsu_threshold),
        "foreground_fraction": float(np.mean(cleaned > 0)),
        "kept_components": kept_components,
    }


def make_qa_tile(image, mask, name):
    overlay = image.copy()
    foreground = mask > 0
    overlay[foreground] = (
        0.45 * overlay[foreground] + 0.55 * np.array([0, 0, 255], dtype=np.float32)
    ).astype(np.uint8)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    tile = np.hstack([image, overlay, mask_bgr])
    cv2.putText(tile, name, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
    return tile


def save_contact_sheet(records, output_path, sample_count):
    if not records:
        return
    count = min(sample_count, len(records))
    indices = np.linspace(0, len(records) - 1, num=count, dtype=np.int32)
    tiles = []
    for index in indices:
        record = records[int(index)]
        image = cv2.imread(record["prepared_image"], cv2.IMREAD_COLOR)
        mask = cv2.imread(record["prepared_mask"], cv2.IMREAD_GRAYSCALE)
        tile = make_qa_tile(image, mask, record["name"])
        tile = cv2.resize(tile, (432, 256), interpolation=cv2.INTER_AREA)
        tiles.append(tile)

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
    manifest_path = Path(args.manifest).resolve()
    source_root = manifest_path.parent
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}. Use a new versioned directory to preserve reproducibility."
        )

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        source_records = list(csv.DictReader(handle))

    selected = [record for record in source_records if record["name"].startswith(args.source_prefix)]
    excluded = [record for record in source_records if not record["name"].startswith(args.source_prefix)]
    if not selected:
        raise ValueError(f"No manifest entries match prefix {args.source_prefix!r}")

    prepared = {"train": [], "val": []}
    for split in prepared:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=False)
        (output_dir / "masks" / split).mkdir(parents=True, exist_ok=False)
    (output_dir / "index").mkdir(parents=True, exist_ok=False)
    (output_dir / "qa").mkdir(parents=True, exist_ok=False)

    try:
        for item_index, record in enumerate(selected):
            split = record["split"]
            if split not in prepared:
                raise ValueError(f"Unsupported split {split!r} for {record['name']}")
            source_image = (source_root / record["left"]).resolve()
            image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Unable to read source image: {source_image}")
            image = cv2.resize(image, (args.width, args.height), interpolation=cv2.INTER_AREA)
            mask, stats = create_pseudo_mask(
                image=image,
                grabcut_iters=args.grabcut_iters,
                min_component_area=args.min_component_area,
                seed=args.seed + item_index,
            )

            image_path = output_dir / "images" / split / f"{record['name']}.png"
            mask_path = output_dir / "masks" / split / f"{record['name']}.png"
            if not cv2.imwrite(str(image_path), image):
                raise OSError(f"Failed to write {image_path}")
            if not cv2.imwrite(str(mask_path), mask):
                raise OSError(f"Failed to write {mask_path}")

            prepared[split].append({
                "name": record["name"],
                "image": str(image_path.relative_to(output_dir)),
                "mask": str(mask_path.relative_to(output_dir)),
                "source_image": str(source_image),
                "prepared_image": str(image_path),
                "prepared_mask": str(mask_path),
                **stats,
            })

        fieldnames = [
            "name", "image", "mask", "source_image", "otsu_threshold",
            "foreground_fraction", "kept_components",
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
        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "method": METHOD_NAME,
            "warning": "Weak Otsu+GrabCut pseudo labels; not human segmentation ground truth.",
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": sha256_file(manifest_path),
            "source_prefix": args.source_prefix,
            "excluded_names": [record["name"] for record in excluded],
            "image_size": {"width": args.width, "height": args.height},
            "label_values": {"background": 0, "workpiece": 255},
            "parameters": {
                "seed": args.seed,
                "grabcut_iters": args.grabcut_iters,
                "min_component_area": args.min_component_area,
            },
            "counts": {split: len(records) for split, records in prepared.items()},
            "foreground_fraction": summarize([record["foreground_fraction"] for record in all_records]),
            "otsu_threshold": summarize([record["otsu_threshold"] for record in all_records]),
        }
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
