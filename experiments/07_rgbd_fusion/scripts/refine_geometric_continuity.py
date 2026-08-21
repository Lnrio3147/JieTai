#!/usr/bin/env python3
"""Apply the Experiment 7.1 continuous-solid geometry prior.

The input is the recall-priority RGB-D + Experiment 4 mask.  The refinement is
label-free at inference: smooth the binary field, close narrow cracks, fill the
external contour, and retain one connected solid.  Large enclosed holes already
preserved by Experiment 4 are restored so a real through-hole is not erased.
Parameters were selected on the fixed validation split and are applied unchanged
to the comparison split.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


TRAIN_SCRIPT = Path(__file__).resolve().with_name("train_rgbd_fusion.py")
spec = importlib.util.spec_from_file_location("rgbd_training", TRAIN_SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(TRAIN_SCRIPT)
RGBD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RGBD)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run",
        type=Path,
        default=RGBD.EXPERIMENT / "results/rgbd_fusion_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RGBD.EXPERIMENT / "results/rgbd_fusion_v1_geometry",
    )
    parser.add_argument("--gaussian-sigma", type=float, default=3.0)
    parser.add_argument("--binary-threshold", type=float, default=0.60)
    parser.add_argument("--closing-radius", type=int, default=6)
    parser.add_argument("--preserve-hole-area", type=int, default=256)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    return parser.parse_args()


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.asarray(mask, dtype=np.uint8), 8
    )
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    selected = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == selected


def geometric_refine(
    mask: np.ndarray,
    gaussian_sigma: float,
    binary_threshold: float,
    closing_radius: int,
    preserve_hole_area: int,
) -> np.ndarray:
    """Create one smooth solid while retaining large Experiment 4 holes."""
    field = cv2.GaussianBlur(
        np.asarray(mask, dtype=np.float32), (0, 0), gaussian_sigma
    )
    solid = field >= binary_threshold
    if closing_radius > 0:
        size = 2 * closing_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        solid = cv2.morphologyEx(
            solid.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)
    # RETR_EXTERNAL deliberately matches the current human target: an outer
    # workpiece contour without supervised internal background polygons.
    contours, _ = cv2.findContours(
        solid.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros_like(solid, dtype=np.uint8)
    cv2.drawContours(filled, contours, -1, 1, thickness=-1)
    refined = largest_component(filled > 0)

    # Experiment 4 has already compared an enclosed region with its disparity
    # ring.  Preserve only its larger surviving holes; tiny black speckles remain
    # filled by the geometric prior.
    inverse = (~np.asarray(mask, dtype=bool)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, 8)
    height, width = mask.shape
    for index in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[index])
        enclosed = x > 0 and y > 0 and x + w < width and y + h < height
        if enclosed and area > preserve_hole_area:
            refined[labels == index] = False
    return refined


def mask_stats(mask: np.ndarray) -> dict[str, float]:
    foreground = np.asarray(mask, dtype=bool)
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8), 8
    )
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.asarray([])
    inverse = (~foreground).astype(np.uint8)
    hole_count, _, hole_stats, _ = cv2.connectedComponentsWithStats(inverse, 8)
    height, width = foreground.shape
    holes = []
    for index in range(1, hole_count):
        x, y, w, h, area = (int(value) for value in hole_stats[index])
        if x > 0 and y > 0 and x + w < width and y + h < height:
            holes.append(area)
    contours, _ = cv2.findContours(
        foreground.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    return {
        "foreground_components": float(len(areas)),
        "foreground_pixels_outside_largest": float(
            areas.sum() - areas.max() if areas.size else 0
        ),
        "enclosed_holes": float(len(holes)),
        "enclosed_hole_pixels": float(sum(holes)),
        "external_perimeter": float(
            sum(cv2.arcLength(contour, True) for contour in contours)
        ),
    }


def mean_stats(masks: dict[str, np.ndarray]) -> dict[str, float]:
    values = [mask_stats(mask) for mask in masks.values()]
    return {
        key: float(np.mean([record[key] for record in values])) for key in values[0]
    }


def load_masks(directory: Path, records: list[dict]) -> dict[str, np.ndarray]:
    masks = {}
    for record in records:
        path = directory / f"{record['name']}.png"
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(path)
        masks[record["name"]] = mask > 127
    return masks


def save_split(
    output: Path,
    split: str,
    dataset: Path,
    records: list[dict],
    v4_masks: dict[str, np.ndarray],
    base_masks: dict[str, np.ndarray],
    geometry_masks: dict[str, np.ndarray],
) -> None:
    split_output = output / split
    mask_output = split_output / "masks"
    comparison_output = split_output / "comparisons"
    mask_output.mkdir(parents=True)
    comparison_output.mkdir(parents=True)
    rows = []
    panels = []
    for record in records:
        name = record["name"]
        cv2.imwrite(
            str(mask_output / f"{name}.png"), geometry_masks[name].astype(np.uint8) * 255
        )
        image = cv2.imread(str(dataset / record["image"]), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(dataset / record["mask"]), cv2.IMREAD_GRAYSCALE) > 127
        row = {"name": name, "category": record["category"]}
        for method, masks in (
            ("v4_1", v4_masks),
            ("rgbd_exp4", base_masks),
            ("geometry", geometry_masks),
        ):
            metrics = RGBD.sample_metrics(gt, masks[name])
            for key in ("foreground_iou", "precision", "recall", "dice"):
                row[f"{method}_{key}"] = metrics[key]
        rows.append(row)
        panel = np.hstack(
            [
                RGBD.EVAL.label_panel(image, "image"),
                RGBD.EVAL.mask_panel(gt, "human outer contour"),
                RGBD.EVAL.mask_panel(v4_masks[name], "V4.1"),
                RGBD.EVAL.mask_panel(base_masks[name], "RGB-D + Exp4"),
                RGBD.EVAL.mask_panel(geometry_masks[name], "Exp7.1 geometry"),
            ]
        )
        cv2.putText(
            panel,
            name,
            (5, panel.shape[0] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        comparison = cv2.resize(panel, (900, 320), interpolation=cv2.INTER_AREA)
        cv2.imwrite(
            str(comparison_output / f"{name}.jpg"),
            comparison,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
        panels.append(comparison)
    with (split_output / "per_scene.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cv2.imwrite(
        str(split_output / "contact_sheet.jpg"),
        np.vstack(panels),
        [cv2.IMWRITE_JPEG_QUALITY, 91],
    )


def main() -> None:
    args = parse_args()
    source_run = args.source_run.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a new geometry result version: {output}")
    output.mkdir(parents=True)
    run_config = json.loads((source_run / "run_config.json").read_text(encoding="utf-8"))
    dataset = Path(run_config["dataset"])
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "7.1 continuous-solid geometric refinement",
        "source_run": str(source_run),
        "inference_requires_ground_truth": False,
        "parameters": {
            "gaussian_sigma": args.gaussian_sigma,
            "binary_threshold": args.binary_threshold,
            "closing_radius": args.closing_radius,
            "fill_external_contour": True,
            "keep_largest_component": True,
            "preserve_experiment4_holes_larger_than": args.preserve_hole_area,
        },
        "selection": {
            "split": "validation only",
            "candidate_gaussian_sigma": [0, 0.75, 1, 1.5, 2, 3],
            "candidate_binary_threshold": [0.4, 0.5, 0.6],
            "candidate_closing_radius": [0, 2, 4, 6],
            "candidate_fill_external_contour": [False, True],
            "constraint": "validation recall >= 0.9905",
            "large_hole_policy": "preserve Experiment 4 holes larger than 256 px for point-cloud safety",
        },
        "splits": {},
    }
    for split in ("val", "test"):
        records = RGBD.read_records(dataset, split)
        metric_records = [{**record, "dataset": str(dataset)} for record in records]
        base_directory = (
            source_run / "validation/masks/rgbd_exp4"
            if split == "val"
            else source_run / "test_masks/rgbd_exp4"
        )
        baseline_directory = (
            RGBD.BASELINE_EXPERIMENT
            / f"results/tune01_jop_reflective_rescue_{split}_v2/masks/jop_reflective_rescue"
        )
        base_masks = load_masks(base_directory, records)
        v4_masks = load_masks(baseline_directory, records)
        geometry_masks = {
            name: geometric_refine(
                mask,
                args.gaussian_sigma,
                args.binary_threshold,
                args.closing_radius,
                args.preserve_hole_area,
            )
            for name, mask in base_masks.items()
        }
        gt_masks = {
            record["name"]: cv2.imread(
                str(dataset / record["mask"]), cv2.IMREAD_GRAYSCALE
            )
            > 127
            for record in records
        }
        summary["splits"][split] = {
            "count": len(records),
            "metrics": {
                "v4_1": RGBD.aggregate(
                    metric_records, v4_masks, args.boundary_tolerance
                ),
                "rgbd_exp4": RGBD.aggregate(
                    metric_records, base_masks, args.boundary_tolerance
                ),
                "geometry": RGBD.aggregate(
                    metric_records, geometry_masks, args.boundary_tolerance
                ),
            },
            "continuity": {
                "human_outer_contour": mean_stats(gt_masks),
                "rgbd_exp4": mean_stats(base_masks),
                "geometry": mean_stats(geometry_masks),
            },
        }
        save_split(
            output,
            split,
            dataset,
            records,
            v4_masks,
            base_masks,
            geometry_masks,
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    test = summary["splits"]["test"]["metrics"]
    print(
        json.dumps(
            {
                method: values["overall"]
                for method, values in test.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
