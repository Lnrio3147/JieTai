#!/usr/bin/env python3
"""Add disparity-projected right-view pseudo masks to the RGB training split.

Validation and test remain left-view human labels.  Right pseudo masks are a
bootstrap source for view invariance, not ground truth; a manually corrected
right-view subset is still required for a publication-grade correspondence
evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

import config_experiment10 as config
from prepare_ablation_manifest import right_image_for
from utils.data import read_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=config.DATASET_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=config.ROOT / "datasets/training/workpiece-seg-stereo-v1",
    )
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.70)
    return parser.parse_args()


def resize_disparity(disparity: np.ndarray, width: int, height: int) -> np.ndarray:
    if disparity.shape == (height, width):
        return disparity.astype(np.float32, copy=False)
    scale_x = width / float(disparity.shape[1])
    resized = cv2.resize(
        disparity.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR
    )
    return resized * scale_x


def read_disparity(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".pfm":
        with path.open("rb") as stream:
            header = stream.readline().decode("ascii").strip()
            if header not in ("PF", "Pf"):
                raise ValueError(f"Invalid PFM header in {path}: {header}")
            color = header == "PF"
            width_text, height_text = stream.readline().decode("ascii").split()
            width, height = int(width_text), int(height_text)
            scale = float(stream.readline().decode("ascii").strip())
            dtype = "<f4" if scale < 0 else ">f4"
            channels = 3 if color else 1
            value = np.fromfile(
                stream, dtype=dtype, count=width * height * channels
            )
        if value.size != width * height * channels:
            raise ValueError(f"Truncated PFM file: {path}")
        shape = (height, width, channels) if color else (height, width)
        value = np.flipud(value.reshape(shape))
    else:
        raise ValueError(f"Unsupported disparity format: {path}")
    if value.ndim == 3:
        value = value[..., 0]
    return np.asarray(value, dtype=np.float32)


def project_mask_to_right(
    left_mask: np.ndarray, disparity: np.ndarray
) -> tuple[np.ndarray, dict]:
    height, width = left_mask.shape
    disparity = resize_disparity(disparity, width, height)
    valid = left_mask & np.isfinite(disparity) & (disparity > 0)
    ys, xs = np.nonzero(valid)
    right_x = np.rint(xs.astype(np.float32) - disparity[ys, xs]).astype(np.int32)
    inside = (right_x >= 0) & (right_x < width)
    right = np.zeros((height, width), dtype=np.uint8)
    right[ys[inside], right_x[inside]] = 255
    right = cv2.morphologyEx(
        right, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)
    )
    return right, {
        "left_foreground_pixels": int(left_mask.sum()),
        "valid_left_foreground_pixels": int(valid.sum()),
        "projected_inside_pixels": int(inside.sum()),
        "right_foreground_pixels": int((right > 0).sum()),
        "valid_fraction": float(valid.sum() / max(left_mask.sum(), 1)),
        "inside_fraction": float(inside.sum() / max(valid.sum(), 1)),
    }


def write_index(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")
    (output / "index").mkdir(parents=True, exist_ok=True)
    (output / "masks/right_pseudo/train").mkdir(parents=True, exist_ok=True)
    audit_rows = []
    split_counts = {}
    for split in ("train", "val", "test"):
        output_rows = []
        records = read_records(source, split)
        for record in records:
            left_image = Path(record["source_image"]).resolve()
            left_mask = Path(record["source_mask"]).resolve()
            left_row = {
                "name": f"{record['name']}_left",
                "category": record["category"],
                "capture_group": record["capture_group"],
                "view": "left_human",
                "image": str(left_image),
                "mask": str(left_mask),
                "source_name": record["name"],
            }
            output_rows.append(left_row)
            if split != "train":
                continue
            right_image = right_image_for(left_image)
            disparity_path = Path(record["source_disparity"]).resolve()
            image = cv2.imread(str(left_image), cv2.IMREAD_COLOR)
            mask_image = cv2.imread(str(left_mask), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(left_image)
            if mask_image is None:
                raise FileNotFoundError(left_mask)
            if not disparity_path.is_file():
                raise FileNotFoundError(disparity_path)
            height, width = image.shape[:2]
            left_binary = cv2.resize(
                mask_image, (width, height), interpolation=cv2.INTER_NEAREST
            ) > 127
            disparity = read_disparity(disparity_path)
            right_mask, diagnostics = project_mask_to_right(left_binary, disparity)
            accepted = (
                diagnostics["valid_fraction"] >= args.minimum_valid_fraction
                and diagnostics["inside_fraction"] >= args.minimum_valid_fraction
            )
            mask_path = output / "masks/right_pseudo/train" / f"{record['name']}.png"
            if accepted:
                cv2.imwrite(str(mask_path), right_mask)
                output_rows.append(
                    {
                        "name": f"{record['name']}_right",
                        "category": record["category"],
                        "capture_group": record["capture_group"],
                        "view": "right_pseudo",
                        "image": str(right_image),
                        "mask": str(mask_path),
                        "source_name": record["name"],
                    }
                )
            audit_rows.append(
                {
                    "name": record["name"],
                    "category": record["category"],
                    "accepted": accepted,
                    **diagnostics,
                }
            )
        write_index(output / "index" / f"{split}.csv", output_rows)
        split_counts[split] = len(output_rows)
    write_index(output / "right_projection_audit.csv", audit_rows)
    metadata = {
        "name": "workpiece-seg-stereo-v1",
        "source": str(source),
        "purpose": "RGB-only pre-stereo segmentation with right-view pseudo-label augmentation",
        "split_counts": split_counts,
        "right_pseudo_candidates": len(audit_rows),
        "right_pseudo_accepted": sum(row["accepted"] for row in audit_rows),
        "minimum_valid_fraction": args.minimum_valid_fraction,
        "warning": (
            "Right masks are projected with cached LAS disparity and are not human GT. "
            "Validation/test remain left-view human labels."
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
