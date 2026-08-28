#!/usr/bin/env python3
"""Build a leakage-resistant grouped split from all 317 human-labelled scenes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

import config_experiment8 as config
from evaluate_all_datasets import inventory


DEFAULT_OUTPUT = config.ROOT / "datasets/training/workpiece-seg-grouped-v3"
CATEGORIES = (
    "fdjyp0",
    "fdjyp2",
    "fdjyp3",
    "luowen",
    "general",
    "scale",
    "jop1",
)
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()


def capture_group(record: dict) -> tuple[str, str]:
    """Return a conservative grouping key and the source of that grouping."""
    name = record["name"]
    category = record["group"]
    if category in ("fdjyp0", "fdjyp2"):
        match = re.match(r"(fdjyp_[02]_\d+)_", name)
        if match is None:
            raise ValueError(name)
        return match.group(1), "explicit_capture_group"
    if category == "luowen":
        match = re.search(r"_(\d{4})$", name)
        if match is None:
            raise ValueError(name)
        frame = int(match.group(1))
        block = 1 + (frame - 1) // 6
        return f"luowen_contiguous_{block:02d}", "contiguous_six_frame_proxy"
    match = re.search(r"_(\d{12})_\d{4}$", name)
    if match is None:
        raise ValueError(name)
    minute = match.group(1)
    return f"{category}_{minute}", "timestamp_minute_proxy"


def grouped_records() -> list[dict]:
    records = [record.copy() for record in inventory() if record["gt"]]
    if len(records) != 317:
        raise ValueError(f"Expected 317 labelled records, got {len(records)}")
    for record in records:
        group, definition = capture_group(record)
        record["capture_group"] = group
        record["group_definition"] = definition
    return records


def optimized_category_assignment(
    category: str,
    groups: dict[str, list[dict]],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """Assign intact groups while minimizing per-category count-ratio error."""
    values = [(name, len(records)) for name, records in sorted(groups.items())]
    if len(values) < 3:
        raise ValueError(f"{category} has fewer than three capture groups")
    rng = random.Random(f"{seed}:{category}")
    rng.shuffle(values)
    # DP state is (validation image count, test image count). The stored tuple
    # gives 0/1/2 assignments for train/val/test in the shuffled group order.
    states: dict[tuple[int, int], tuple[int, ...]] = {(0, 0): ()}
    for _, size in values:
        updated = {}
        for (val_count, test_count), assignment in states.items():
            options = (
                ((val_count, test_count), assignment + (0,)),
                ((val_count + size, test_count), assignment + (1,)),
                ((val_count, test_count + size), assignment + (2,)),
            )
            for state, candidate in options:
                current = updated.get(state)
                if current is None or candidate < current:
                    updated[state] = candidate
        states = updated

    total = sum(size for _, size in values)
    targets = {split: total * ratios[split] for split in SPLITS}
    candidates = []
    for (val_count, test_count), assignment in states.items():
        train_count = total - val_count - test_count
        counts = {"train": train_count, "val": val_count, "test": test_count}
        if min(counts.values()) <= 0 or set(assignment) != {0, 1, 2}:
            continue
        score = sum(
            ((counts[split] - targets[split]) / max(total, 1)) ** 2
            for split in SPLITS
        )
        # Prefer comparable validation/test sizes when the ratio error ties.
        tie = abs(val_count - test_count) / max(total, 1)
        candidates.append((score, tie, assignment, counts))
    if not candidates:
        raise ValueError(f"No valid grouped split for {category}")
    _, _, selected, _ = min(candidates)
    split_names = {0: "train", 1: "val", 2: "test"}
    return {
        group_name: split_names[assignment]
        for (group_name, _), assignment in zip(values, selected)
    }


def assign_splits(
    records: list[dict], ratios: dict[str, float], seed: int
) -> tuple[list[dict], list[dict]]:
    by_category_group = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_category_group[record["group"]][record["capture_group"]].append(record)
    assignments = {}
    group_rows = []
    for category in CATEGORIES:
        category_groups = by_category_group[category]
        selected = optimized_category_assignment(
            category, category_groups, ratios, seed
        )
        for group, split in selected.items():
            key = (category, group)
            assignments[key] = split
            records_in_group = category_groups[group]
            group_rows.append(
                {
                    "category": category,
                    "capture_group": group,
                    "group_definition": records_in_group[0]["group_definition"],
                    "split": split,
                    "image_count": len(records_in_group),
                    "first_name": min(item["name"] for item in records_in_group),
                    "last_name": max(item["name"] for item in records_in_group),
                }
            )
    for record in records:
        record["split"] = assignments[(record["group"], record["capture_group"])]
    records.sort(key=lambda item: (SPLITS.index(item["split"]), item["group"], item["name"]))
    group_rows.sort(
        key=lambda item: (
            SPLITS.index(item["split"]),
            CATEGORIES.index(item["category"]),
            item["capture_group"],
        )
    )
    return records, group_rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(target.resolve(), link.parent.resolve()))


def make_contact_sheet(records: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cell_width, image_height, label_height = 144, 256, 34
    columns = 8
    rows = int(math.ceil(len(records) / columns))
    canvas = np.zeros(
        (rows * (image_height + label_height), columns * cell_width, 3),
        dtype=np.uint8,
    )
    for index, record in enumerate(records):
        image = cv2.imread(record["source_image"], cv2.IMREAD_COLOR)
        mask = cv2.imread(record["source_mask"], cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(record["name"])
        image = cv2.resize(image, (cell_width, image_height), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(
            mask, (cell_width, image_height), interpolation=cv2.INTER_NEAREST
        ) > 127
        overlay = image.copy()
        overlay[mask] = np.round(0.65 * image[mask] + 0.35 * np.asarray([0, 255, 0])).astype(
            np.uint8
        )
        row, column = divmod(index, columns)
        y = row * (image_height + label_height)
        x = column * cell_width
        canvas[y : y + image_height, x : x + cell_width] = overlay
        label = f"{record['category']} {record['capture_group'].split('_')[-1]}"
        cv2.putText(
            canvas,
            label[:23],
            (x + 3, y + image_height + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.31,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            record["name"][-9:],
            (x + 3, y + image_height + 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(
    output: Path,
    records: list[dict],
    group_rows: list[dict],
    ratios: dict[str, float],
    seed: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".workpiece-seg-grouped-v3-", dir=output.parent))
    try:
        index_rows = []
        for record in records:
            split = record["split"]
            name = record["name"]
            image_link = temporary / "images" / split / f"{name}.png"
            mask_link = temporary / "masks" / split / f"{name}.png"
            disparity_suffix = Path(record["disparity"]).suffix.lower()
            disparity_link = temporary / "disparities" / split / f"{name}{disparity_suffix}"
            relative_symlink(Path(record["image"]), image_link)
            relative_symlink(Path(record["gt"]), mask_link)
            relative_symlink(Path(record["disparity"]), disparity_link)
            index_rows.append(
                {
                    "name": name,
                    "category": record["group"],
                    "capture_group": record["capture_group"],
                    "group_definition": record["group_definition"],
                    "split": split,
                    "image": str(image_link.relative_to(temporary)),
                    "mask": str(mask_link.relative_to(temporary)),
                    "disparity": str(disparity_link.relative_to(temporary)),
                    "source_image": str(Path(record["image"]).resolve()),
                    "source_mask": str(Path(record["gt"]).resolve()),
                    "source_disparity": str(Path(record["disparity"]).resolve()),
                    "source_annotation_dataset": record["annotation_source"],
                    "source_annotation_split": record["annotation_split"],
                    "previous_experiment8_role": record["experiment8_training_relation"],
                    "image_sha256": sha256(Path(record["image"])),
                    "mask_sha256": sha256(Path(record["gt"])),
                }
            )
        fields = list(index_rows[0])
        for split in SPLITS:
            split_rows = [row for row in index_rows if row["split"] == split]
            write_csv(temporary / "index" / f"{split}.csv", split_rows, fields)
            make_contact_sheet(split_rows, temporary / "qa" / f"{split}_contact_sheet.jpg")
        write_csv(temporary / "index/all.csv", index_rows, fields)
        write_csv(temporary / "qa/group_assignments.csv", group_rows)

        split_counts = Counter(row["split"] for row in index_rows)
        category_counts = {
            category: {
                split: sum(
                    row["category"] == category and row["split"] == split
                    for row in index_rows
                )
                for split in SPLITS
            }
            for category in CATEGORIES
        }
        group_sets = {
            split: {
                (row["category"], row["capture_group"])
                for row in index_rows
                if row["split"] == split
            }
            for split in SPLITS
        }
        hash_sets = {
            split: {
                row["image_sha256"] for row in index_rows if row["split"] == split
            }
            for split in SPLITS
        }
        group_overlap = {
            f"{first}_{second}": sorted(group_sets[first] & group_sets[second])
            for index, first in enumerate(SPLITS)
            for second in SPLITS[index + 1 :]
        }
        hash_overlap = {
            f"{first}_{second}": sorted(hash_sets[first] & hash_sets[second])
            for index, first in enumerate(SPLITS)
            for second in SPLITS[index + 1 :]
        }
        if any(group_overlap.values()) or any(hash_overlap.values()):
            raise ValueError(
                f"Leakage audit failed: groups={group_overlap}, hashes={hash_overlap}"
            )
        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "name": "workpiece-seg-grouped-v3",
            "purpose": "group-isolated combined human-labelled RGB-D segmentation split",
            "seed": seed,
            "requested_ratios": ratios,
            "actual_counts": dict(split_counts),
            "actual_ratios": {
                split: split_counts[split] / len(index_rows) for split in SPLITS
            },
            "category_split_counts": category_counts,
            "scene_count": len(index_rows),
            "capture_group_count": len(
                {(row["category"], row["capture_group"]) for row in index_rows}
            ),
            "source_counts": dict(
                Counter(row["source_annotation_dataset"] for row in index_rows)
            ),
            "grouping_policy": {
                "fdjyp0_fdjyp2": "explicit capture_group encoded in filename",
                "fdjyp3_general_scale_jop1": "same category and acquisition minute",
                "luowen": "contiguous blocks of six sequence frames; proxy because capture metadata is absent",
            },
            "assignment_policy": "per-category dynamic programming minimizes 70/15/15 image-count error while preserving intact groups and all categories in all splits",
            "leakage_audit": {
                "capture_group_overlap": group_overlap,
                "exact_source_image_sha256_overlap": hash_overlap,
                "passed": True,
            },
            "evaluation_warning": (
                "All 317 labels and previous model outputs have already been inspected. "
                "The test split is frozen for future engineering regression, but is not "
                "a never-seen publication-grade final test set."
            ),
            "excluded_unlabelled_groups": {
                "de0548": 7,
                "jxp": 15,
                "gongjian_test": 8,
                "other_test": 6,
            },
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        readme = f"""# Workpiece segmentation grouped V3

This dataset combines all 317 current human foreground masks and splits intact
capture groups with seed `{seed}`. It is implemented with relative symbolic
links, so source images, masks and disparities are not duplicated.

| Split | Images | Ratio |
|---|---:|---:|
| train | {split_counts['train']} | {split_counts['train'] / len(index_rows):.3f} |
| val | {split_counts['val']} | {split_counts['val'] / len(index_rows):.3f} |
| test | {split_counts['test']} | {split_counts['test'] / len(index_rows):.3f} |

No capture group or exact source-image hash crosses splits. See `metadata.json`
and `qa/group_assignments.csv` for the full audit. Because the existing 317
labels have already informed development, `test` is a frozen engineering
regression set, not a never-observed publication final test set.
"""
        (temporary / "README.md").write_text(readme, encoding="utf-8")
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary)
        raise


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    ratios = {
        "train": float(args.train_ratio),
        "val": float(args.val_ratio),
        "test": float(args.test_ratio),
    }
    if abs(sum(ratios.values()) - 1.0) > 1e-9 or min(ratios.values()) <= 0:
        raise ValueError(f"Ratios must be positive and sum to one: {ratios}")
    records, group_rows = assign_splits(grouped_records(), ratios, args.seed)
    build_dataset(output, records, group_rows, ratios, args.seed)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
