#!/usr/bin/env python3
"""Build a stereo-pair manifest from the grouped V3 segmentation split."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import config_experiment10 as config
from utils.data import read_records


REFERENCE_CROP = (234, 1052, 126, 638)  # y0, y1, x0, x1


def right_image_for(left: Path) -> Path:
    replacements = {"im0.png": "im1.png", "left.png": "right.png"}
    if left.name not in replacements:
        raise ValueError(f"Unknown rectified-left filename: {left}")
    right = left.with_name(replacements[left.name])
    if not right.is_file():
        raise FileNotFoundError(right)
    return right


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=config.DATASET_DIR)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--right-ground-truth-dataset",
        type=Path,
        default=None,
        help="optional prepared right-view dataset with source_name in its index",
    )
    parser.add_argument(
        "--only-with-right-ground-truth",
        action="store_true",
        help="keep only rows represented in --right-ground-truth-dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.EXPERIMENT_DIR / "inputs/grouped_v3_test.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    if args.only_with_right_ground_truth and args.right_ground_truth_dataset is None:
        raise ValueError(
            "--only-with-right-ground-truth requires --right-ground-truth-dataset"
        )
    right_ground_truth = {}
    right_dataset = None
    if args.right_ground_truth_dataset is not None:
        right_dataset = args.right_ground_truth_dataset.resolve()
        for record in read_records(right_dataset, args.split):
            source_name = record.get("source_name")
            if not source_name:
                raise ValueError(
                    f"Missing source_name in {right_dataset}/index/{args.split}.csv"
                )
            if source_name in right_ground_truth:
                raise ValueError(f"Duplicate right GT source_name: {source_name}")
            right_ground_truth[source_name] = record
    rows = []
    for record in read_records(dataset, args.split):
        right_record = right_ground_truth.get(record["name"])
        if args.only_with_right_ground_truth and right_record is None:
            continue
        left = Path(record["source_image"]).resolve()
        right = right_image_for(left)
        mask = Path(record["source_mask"]).resolve()
        if not mask.is_file():
            raise FileNotFoundError(mask)
        reference = ""
        reference_crop = ""
        if record["category"] == "fdjyp3":
            candidate = (
                config.ROOT
                / "datasets/tradition_raw/FDJYP-3"
                / left.parent.name
                / "disp_cropped.npy"
            )
            if candidate.is_file():
                reference = str(candidate.resolve())
                reference_crop = ",".join(str(value) for value in REFERENCE_CROP)
        output_row = {
            "name": record["name"],
            "category": record["category"],
            "left": str(left),
            "right": str(right),
            "left_gt_mask": str(mask),
            "reference_disparity": reference,
            "reference_crop_y0_y1_x0_x1": reference_crop,
        }
        if right_dataset is not None:
            output_row.update(
                {
                    "right_gt_mask": (
                        str((right_dataset / right_record["mask"]).resolve())
                        if right_record is not None
                        else ""
                    ),
                    "right_gt_annotation": (
                        str((right_dataset / right_record["annotation"]).resolve())
                        if right_record is not None
                        else ""
                    ),
                }
            )
        rows.append(output_row)
    if not rows:
        raise ValueError("No rows selected for the ablation manifest")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
