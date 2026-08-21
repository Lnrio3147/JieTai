#!/usr/bin/env python3
"""Experiment 7.2: rescue confident subjects from foreground overflow.

The global recall-priority threshold is retained for normal scenes.  A scene is
switched to a high-confidence component only when the probability topology shows
all of the following without using ground truth:

1. raising the threshold from the global value to 0.70 removes an unusually
   large amount of foreground;
2. the remaining component still has an almost full-frame bounding box;
3. a small threshold increase near the high-confidence end causes an abrupt,
   spatially compact component split.

The split component is closed, filled by its external contour, and reduced to a
single connected solid.  This targets the failure where background and subject
are joined at low confidence; ordinary Experiment 7.1 masks remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


GEOMETRY_SCRIPT = Path(__file__).resolve().with_name(
    "refine_geometric_continuity.py"
)
spec = importlib.util.spec_from_file_location("exp7_geometry", GEOMETRY_SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(GEOMETRY_SCRIPT)
GEO = importlib.util.module_from_spec(spec)
spec.loader.exec_module(GEO)
RGBD = GEO.RGBD


@dataclass(frozen=True)
class Component:
    mask: np.ndarray
    area_fraction: float
    bbox_fraction: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run",
        type=Path,
        default=RGBD.EXPERIMENT / "results/rgbd_fusion_v1",
    )
    parser.add_argument(
        "--geometry-run",
        type=Path,
        default=RGBD.EXPERIMENT / "results/rgbd_fusion_v1_geometry",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RGBD.EXPERIMENT / "results/rgbd_fusion_v1_geometry_overflow",
    )
    parser.add_argument("--reference-threshold", type=float, default=0.70)
    parser.add_argument("--search-start", type=float, default=0.90)
    parser.add_argument("--search-stop", type=float, default=0.97)
    parser.add_argument("--search-step", type=float, default=0.001)
    parser.add_argument("--max-reference-area-ratio", type=float, default=0.80)
    parser.add_argument("--min-reference-bbox-fraction", type=float, default=0.90)
    parser.add_argument("--max-step-area-ratio", type=float, default=0.85)
    parser.add_argument("--max-bbox-contraction-ratio", type=float, default=0.80)
    parser.add_argument("--min-candidate-area-fraction", type=float, default=0.03)
    parser.add_argument("--closing-radius", type=int, default=6)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    return parser.parse_args()


def largest_component_description(mask: np.ndarray) -> Component:
    foreground = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    empty = np.zeros_like(foreground, dtype=bool)
    if count <= 1:
        return Component(empty, 0.0, 0.0)
    selected = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, width, height, area = (
        int(value) for value in stats[selected]
    )
    del x, y
    pixels = float(foreground.size)
    return Component(
        labels == selected,
        area / pixels,
        width * height / pixels,
    )


def solidify(mask: np.ndarray, closing_radius: int) -> np.ndarray:
    solid = np.asarray(mask, dtype=np.uint8)
    if closing_radius > 0:
        size = 2 * closing_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros_like(solid)
    cv2.drawContours(filled, contours, -1, 1, thickness=-1)
    return largest_component_description(filled).mask


def find_overflow_rescue(
    probability: np.ndarray,
    low_threshold: float,
    reference_threshold: float,
    search_start: float,
    search_stop: float,
    search_step: float,
    max_reference_area_ratio: float,
    min_reference_bbox_fraction: float,
    max_step_area_ratio: float,
    max_bbox_contraction_ratio: float,
    min_candidate_area_fraction: float,
    closing_radius: int,
) -> tuple[np.ndarray | None, dict[str, float | bool | None]]:
    low = largest_component_description(probability >= low_threshold)
    reference = largest_component_description(probability >= reference_threshold)
    reference_area_ratio = reference.area_fraction / max(low.area_fraction, 1e-9)

    event_threshold = None
    event = None
    previous = largest_component_description(probability >= search_start)
    thresholds = np.arange(
        search_start + search_step,
        search_stop + search_step * 0.5,
        search_step,
    )
    for raw_threshold in thresholds:
        current = largest_component_description(probability >= raw_threshold)
        step_area_ratio = current.area_fraction / max(previous.area_fraction, 1e-9)
        bbox_contraction = current.bbox_fraction / max(
            reference.bbox_fraction, 1e-9
        )
        if (
            current.area_fraction >= min_candidate_area_fraction
            and step_area_ratio <= max_step_area_ratio
            and bbox_contraction <= max_bbox_contraction_ratio
        ):
            event_threshold = float(round(raw_threshold, 3))
            event = (current, step_area_ratio, bbox_contraction)
            break
        previous = current

    precondition = (
        reference_area_ratio < max_reference_area_ratio
        and reference.bbox_fraction > min_reference_bbox_fraction
    )
    triggered = precondition and event is not None
    decision: dict[str, float | bool | None] = {
        "triggered": triggered,
        "low_area_fraction": low.area_fraction,
        "reference_area_fraction": reference.area_fraction,
        "reference_area_ratio": reference_area_ratio,
        "reference_bbox_fraction": reference.bbox_fraction,
        "event_threshold": event_threshold,
        "event_step_area_ratio": None if event is None else event[1],
        "event_bbox_contraction_ratio": None if event is None else event[2],
    }
    if not triggered or event is None:
        return None, decision
    return solidify(event[0].mask, closing_radius), decision


def probability_directory(source_run: Path, split: str) -> Path:
    if split == "test":
        return source_run / "test_probabilities"
    return source_run / "validation_with_probabilities/probabilities"


def save_split(
    output: Path,
    split: str,
    dataset: Path,
    records: list[dict],
    v4_masks: dict[str, np.ndarray],
    base_masks: dict[str, np.ndarray],
    geometry_masks: dict[str, np.ndarray],
    adaptive_masks: dict[str, np.ndarray],
    decisions: dict[str, dict],
) -> None:
    split_output = output / split
    mask_output = split_output / "masks"
    comparison_output = split_output / "comparisons"
    mask_output.mkdir(parents=True)
    comparison_output.mkdir(parents=True)
    metric_rows = []
    decision_rows = []
    panels = []
    for record in records:
        name = record["name"]
        cv2.imwrite(
            str(mask_output / f"{name}.png"),
            adaptive_masks[name].astype(np.uint8) * 255,
        )
        image = cv2.imread(str(dataset / record["image"]), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(dataset / record["mask"]), cv2.IMREAD_GRAYSCALE) > 127
        row = {"name": name, "category": record["category"]}
        for method, masks in (
            ("v4_1", v4_masks),
            ("rgbd_exp4", base_masks),
            ("geometry_7_1", geometry_masks),
            ("overflow_7_2", adaptive_masks),
        ):
            metrics = RGBD.sample_metrics(gt, masks[name])
            for key in ("foreground_iou", "precision", "recall", "dice"):
                row[f"{method}_{key}"] = metrics[key]
        metric_rows.append(row)
        decision_rows.append(
            {"name": name, "category": record["category"], **decisions[name]}
        )
        panel = np.hstack(
            [
                RGBD.EVAL.label_panel(image, "image"),
                RGBD.EVAL.mask_panel(gt, "human outer contour"),
                RGBD.EVAL.mask_panel(v4_masks[name], "V4.1"),
                RGBD.EVAL.mask_panel(geometry_masks[name], "Exp7.1 geometry"),
                RGBD.EVAL.mask_panel(adaptive_masks[name], "Exp7.2 overflow"),
            ]
        )
        status = "rescued" if decisions[name]["triggered"] else "unchanged"
        cv2.putText(
            panel,
            f"{name} [{status}]",
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
    for filename, rows in (
        ("per_scene.csv", metric_rows),
        ("decisions.csv", decision_rows),
    ):
        with (split_output / filename).open(
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
    geometry_run = args.geometry_run.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a new Experiment 7.2 result version: {output}")
    output.mkdir(parents=True)
    config = json.loads((source_run / "run_config.json").read_text(encoding="utf-8"))
    training_summary = json.loads(
        (source_run / "summary.json").read_text(encoding="utf-8")
    )
    dataset = Path(config["dataset"])
    low_threshold = float(training_summary["selected_probability_threshold"])
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "7.2 adaptive foreground-overflow rescue",
        "source_run": str(source_run),
        "geometry_run": str(geometry_run),
        "inference_requires_ground_truth": False,
        "parameters": {
            "low_threshold": low_threshold,
            "reference_threshold": args.reference_threshold,
            "search_range": [args.search_start, args.search_stop, args.search_step],
            "max_reference_area_ratio": args.max_reference_area_ratio,
            "min_reference_bbox_fraction": args.min_reference_bbox_fraction,
            "max_step_area_ratio": args.max_step_area_ratio,
            "max_bbox_contraction_ratio": args.max_bbox_contraction_ratio,
            "min_candidate_area_fraction": args.min_candidate_area_fraction,
            "closing_radius": args.closing_radius,
            "fill_external_contour": True,
            "keep_single_component": True,
        },
        "selection_note": (
            "The 0.70 reference threshold is supported by the fixed validation "
            "sweep. Overflow trigger inequalities were developed after inspecting "
            "the user-identified 0031/0033 comparison failures; Experiment 7.2 "
            "therefore remains an engineering comparison, not an untouched test."
        ),
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
        base_masks = GEO.load_masks(base_directory, records)
        v4_masks = GEO.load_masks(baseline_directory, records)
        geometry_masks = GEO.load_masks(geometry_run / split / "masks", records)
        adaptive_masks = {}
        decisions = {}
        probability_root = probability_directory(source_run, split)
        for record in records:
            name = record["name"]
            probability = np.load(probability_root / f"{name}.npy").astype(np.float32)
            rescue, decision = find_overflow_rescue(
                probability,
                low_threshold,
                args.reference_threshold,
                args.search_start,
                args.search_stop,
                args.search_step,
                args.max_reference_area_ratio,
                args.min_reference_bbox_fraction,
                args.max_step_area_ratio,
                args.max_bbox_contraction_ratio,
                args.min_candidate_area_fraction,
                args.closing_radius,
            )
            decisions[name] = decision
            adaptive_masks[name] = geometry_masks[name] if rescue is None else rescue
        gt_masks = {
            record["name"]: cv2.imread(
                str(dataset / record["mask"]), cv2.IMREAD_GRAYSCALE
            )
            > 127
            for record in records
        }
        summary["splits"][split] = {
            "count": len(records),
            "triggered_count": sum(
                bool(decision["triggered"]) for decision in decisions.values()
            ),
            "triggered_scenes": [
                name for name, decision in decisions.items() if decision["triggered"]
            ],
            "metrics": {
                "v4_1": RGBD.aggregate(
                    metric_records, v4_masks, args.boundary_tolerance
                ),
                "rgbd_exp4": RGBD.aggregate(
                    metric_records, base_masks, args.boundary_tolerance
                ),
                "geometry_7_1": RGBD.aggregate(
                    metric_records, geometry_masks, args.boundary_tolerance
                ),
                "overflow_7_2": RGBD.aggregate(
                    metric_records, adaptive_masks, args.boundary_tolerance
                ),
            },
            "continuity": {
                "human_outer_contour": GEO.mean_stats(gt_masks),
                "geometry_7_1": GEO.mean_stats(geometry_masks),
                "overflow_7_2": GEO.mean_stats(adaptive_masks),
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
            adaptive_masks,
            decisions,
        )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "triggered": {
                    split: values["triggered_scenes"]
                    for split, values in summary["splits"].items()
                },
                "test": {
                    method: values["overall"]
                    for method, values in summary["splits"]["test"]["metrics"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
