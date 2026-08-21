#!/usr/bin/env python3
"""Validate a custom stereo CSV manifest before starting a training run."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.data import load_manifest_samples, read_disparity, read_rgb, read_valid_mask


def parse_args():
    parser = argparse.ArgumentParser(description="Check custom stereo paths, shapes, masks, and disparity units.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--max_disp", type=float, default=192.0)
    parser.add_argument("--max_samples", type=int, default=0, help="0 checks every sample")
    parser.add_argument("--report", default=None, help="optional JSON report path")
    return parser.parse_args()


def finite_or_none(value):
    return float(value) if math.isfinite(float(value)) else None


def inspect_sample(sample, max_disp):
    left = read_rgb(sample.left)
    right = read_rgb(sample.right)
    disparity = read_disparity(sample.disparity, sample.disparity_scale)
    if left.shape != right.shape:
        raise ValueError(f"left/right shape mismatch: {left.shape} vs {right.shape}")
    if left.shape[:2] != disparity.shape:
        raise ValueError(f"image/disparity shape mismatch: {left.shape[:2]} vs {disparity.shape}")

    if sample.valid is None:
        external_valid = np.ones(disparity.shape, dtype=bool)
    else:
        external_valid = read_valid_mask(sample.valid)
        if external_valid.shape != disparity.shape:
            raise ValueError(f"disparity/mask shape mismatch: {disparity.shape} vs {external_valid.shape}")

    finite = np.isfinite(disparity)
    positive = finite & (disparity > 0)
    valid = external_valid & positive & (disparity < max_disp)
    values = disparity[valid]
    pixels = int(disparity.size)
    return {
        "name": sample.name,
        "height": int(disparity.shape[0]),
        "width": int(disparity.shape[1]),
        "pixels": pixels,
        "finite_pixels": int(finite.sum()),
        "positive_pixels": int(positive.sum()),
        "valid_pixels": int(valid.sum()),
        "over_max_pixels": int((positive & (disparity >= max_disp)).sum()),
        "minimum": finite_or_none(np.min(values)) if values.size else None,
        "median": finite_or_none(np.median(values)) if values.size else None,
        "p99": finite_or_none(np.percentile(values, 99)) if values.size else None,
        "maximum": finite_or_none(np.max(values)) if values.size else None,
    }


def main():
    args = parse_args()
    if args.max_disp <= 0 or args.max_samples < 0:
        raise ValueError("--max_disp must be positive and --max_samples cannot be negative")

    splits = [args.split] if args.split != "all" else ["train", "val"]
    samples_by_split = {}
    load_errors = []
    for split in splits:
        try:
            samples_by_split[split] = load_manifest_samples(args.manifest, split)
        except Exception as error:
            load_errors.append(f"{split}: {error}")

    rows = []
    errors = list(load_errors)
    split_counts = {}
    duplicate_names = {}
    for split, samples in samples_by_split.items():
        split_counts[split] = len(samples)
        name_counts = Counter(sample.name for sample in samples)
        duplicates = sorted(name for name, count in name_counts.items() if count > 1)
        if duplicates:
            duplicate_names[split] = duplicates
        selected = samples[: args.max_samples] if args.max_samples else samples
        for sample in selected:
            try:
                row = inspect_sample(sample, args.max_disp)
                row["split"] = split
                rows.append(row)
            except Exception as error:
                errors.append(f"{split}/{sample.name}: {error}")

    total_pixels = sum(row["pixels"] for row in rows)
    total_valid = sum(row["valid_pixels"] for row in rows)
    total_over = sum(row["over_max_pixels"] for row in rows)
    valid_medians = [row["median"] for row in rows if row["median"] is not None]
    shapes = Counter(f"{row['height']}x{row['width']}" for row in rows)
    warnings = []
    if duplicate_names:
        warnings.append("Duplicate sample names were found; unique names make result tracing easier.")
    if total_pixels and total_valid / total_pixels < 0.01:
        warnings.append("Fewer than 1% of pixels are valid; check disparity units, masks, and --max_disp.")
    if total_over:
        warnings.append("Some positive disparities are excluded by --max_disp; raise it only if the model supports it.")
    if len(shapes) > 1:
        warnings.append("Image sizes vary; training cropping supports this and validation uses batch size 1.")

    report = {
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "max_disp": args.max_disp,
        "declared_samples": split_counts,
        "checked_samples": len(rows),
        "errors": errors,
        "warnings": warnings,
        "duplicate_names": duplicate_names,
        "image_shapes": dict(sorted(shapes.items())),
        "valid_pixel_fraction": total_valid / total_pixels if total_pixels else 0.0,
        "over_max_pixel_fraction": total_over / total_pixels if total_pixels else 0.0,
        "median_of_sample_medians": float(np.median(valid_medians)) if valid_medians else None,
        "samples": rows,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    print(rendered)
    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered + "\n", encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
