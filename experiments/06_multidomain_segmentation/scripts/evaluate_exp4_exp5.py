#!/usr/bin/env python3
"""Evaluate the trained multi-domain segmenter with Experiment 4/5 post-processing.

Experiment 4 and Experiment 5 are not separately trainable neural networks.  This
script keeps the semantic probability map fixed and compares, on exactly the same
annotated split:

* raw semantic thresholding;
* Experiment 4 topology and disparity-continuity refinement;
* Experiment 5 V1 semantic/disparity soft fusion;
* Experiment 4 + 5 recall-priority fusion V2.

All disparity arrays are existing LiteAnyStereo predictions.  Ground-truth masks
are used only for metrics, never by any refinement method.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/06_multidomain_segmentation"
DATASET_DEFAULT = ROOT / "datasets/training/workpiece-seg-isat-v2"
RESULTS = EXPERIMENT / "results"
REC_STEREO = (
    ROOT / "experiments/01_stereo_comparison/rec_img_set/results/final_203/outputs"
)
JOP1_STEREO = (
    ROOT / "experiments/01_stereo_comparison/jop1/results/final_9/liteanystereo"
)
FUSION_SCRIPT = (
    ROOT
    / "experiments/05_disparity_guided_segmentation/scripts/run_experiment.py"
)
METHODS = ("semantic", "experiment4", "experiment5", "experiment4_plus_5")


def load_fusion_module():
    spec = importlib.util.spec_from_file_location("experiment5_fusion", FUSION_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load {}".format(FUSION_SCRIPT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FUSION = load_fusion_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--semantic-source",
        choices=("routed_v3", "balanced_v2", "old_model"),
        default="routed_v3",
    )
    parser.add_argument(
        "--probability-dir",
        type=Path,
        default=None,
        help="Optional custom probability directory; overrides --semantic-source.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "exp4_exp5_routed_v3_test_v1",
    )
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.5,
        help="Foreground threshold used by semantic and recall-preservation branches.",
    )
    parser.add_argument(
        "--category-threshold",
        action="append",
        default=[],
        metavar="CATEGORY=VALUE",
        help=(
            "Optional category-specific foreground threshold. May be repeated; "
            "for example --category-threshold jop1=0.03."
        ),
    )
    parser.add_argument(
        "--jop-reflective-rescue",
        action="store_true",
        help=(
            "Add an adaptive Jop1 branch that grows uncertain reflective subjects "
            "only when semantic size and disparity reliability indicate a safe rescue."
        ),
    )
    parser.add_argument("--rescue-low-threshold", type=float, default=0.01)
    parser.add_argument("--rescue-depth-interpolation", type=float, default=0.10)
    parser.add_argument("--rescue-min-depth-reliability", type=float, default=0.70)
    parser.add_argument("--rescue-min-base-fraction", type=float, default=0.70)
    return parser.parse_args()


def parse_category_thresholds(values: list[str]) -> dict[str, float]:
    thresholds = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                "--category-threshold must use CATEGORY=VALUE, got {!r}".format(value)
            )
        category, raw_threshold = value.split("=", 1)
        category = category.strip()
        if not category:
            raise ValueError("Category name cannot be empty")
        if category in thresholds:
            raise ValueError("Duplicate category threshold: {}".format(category))
        threshold = float(raw_threshold)
        if not 0.0 < threshold < 1.0:
            raise ValueError(
                "Category threshold for {} must be in (0, 1)".format(category)
            )
        thresholds[category] = threshold
    return thresholds


def read_records(dataset: Path, split: str) -> list[dict]:
    with (dataset / "index/{}.csv".format(split)).open(
        encoding="utf-8", newline=""
    ) as stream:
        records = list(csv.DictReader(stream))
    if not records:
        raise ValueError("Empty split: {}".format(split))
    return records


def route_definition() -> dict[str, str]:
    summary_path = RESULTS / "routed_v3/summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))["route"]


def probability_path(
    record: dict,
    split: str,
    source: str,
    custom_directory: Path | None = None,
) -> tuple[Path, str]:
    name = record["name"]
    if custom_directory is not None:
        return custom_directory / "{}.npy".format(name), custom_directory.parent.name
    if source == "balanced_v2":
        directory = "val_balanced_v2" if split == "val" else "test_balanced_v2"
        return RESULTS / directory / "probabilities/{}.npy".format(name), source
    if source == "old_model":
        directory = "val_old_model" if split == "val" else "baseline_old_model"
        return RESULTS / directory / "probabilities/{}.npy".format(name), source

    selected = route_definition()[record["category"]]
    if split == "test":
        return RESULTS / "routed_v3/probabilities/{}.npy".format(name), selected
    directory = "val_balanced_v2" if selected == "balanced_v2" else "val_old_model"
    return RESULTS / directory / "probabilities/{}.npy".format(name), selected


def scene_name(record: dict) -> str:
    category = record["category"]
    prefix = category + "_"
    if not record["name"].startswith(prefix):
        raise ValueError("Unexpected sample name: {}".format(record["name"]))
    suffix = record["name"][len(prefix) :]
    if category in ("general", "scale", "jop1"):
        return suffix.replace("_", "-")
    stem, frame = suffix.rsplit("_", 1)
    return "{}-{}".format(stem, frame)


def disparity_path(record: dict) -> Path:
    category = record["category"]
    scene = scene_name(record)
    if category == "jop1":
        return JOP1_STEREO / scene / "disp.npy"
    group = {"general": "general_1221", "scale": "scale_1221"}.get(
        category, category
    )
    return REC_STEREO / group / scene / "liteanystereo/disp_full.npy"


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.asarray(mask, dtype=np.uint8), 8
    )
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    selected = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == selected


def enclosed_holes(mask: np.ndarray):
    inverse = (~np.asarray(mask, dtype=bool)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, 8)
    height, width = mask.shape
    for label in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[label])
        if x == 0 or y == 0 or x + w == width or y + h == height:
            continue
        yield labels == label, area


def experiment4_refine(
    probability: np.ndarray, disparity: np.ndarray, threshold: float = 0.5
) -> tuple[np.ndarray, dict]:
    """Resolution-scaled form of Experiment 4's fixed refinement rules."""
    config = FUSION.scaled_recall_priority_config(probability.shape)
    raw = np.asarray(probability, dtype=np.float32) >= threshold
    radius = int(config["closing_radius"])
    if radius:
        size = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        closed = cv2.morphologyEx(
            raw.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)
    else:
        closed = raw.copy()
    subject = largest_component(closed)
    max_fill_area = int(
        round(float(config["max_ambiguous_fill_fraction"]) * subject.size)
    )
    valid = np.isfinite(disparity) & (disparity > 0)
    decisions = []
    ring_radius = int(config["hole_ring_radius"])
    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * ring_radius + 1, 2 * ring_radius + 1)
    )
    for hole, area in list(enclosed_holes(subject)):
        fill = False
        reason = "area_limit"
        details = {"area": int(area)}
        if area <= max_fill_area:
            ring = cv2.dilate(hole.astype(np.uint8), ring_kernel).astype(bool)
            ring &= subject
            hole_values = disparity[hole & valid]
            ring_values = disparity[ring & valid]
            support = int(config["minimum_support_pixels"])
            if hole_values.size >= support and ring_values.size >= support:
                hole_median = float(np.median(hole_values))
                ring_median = float(np.median(ring_values))
                ring_mad = float(np.median(np.abs(ring_values - ring_median)))
                difference = abs(hole_median - ring_median)
                tolerance = max(
                    float(config["background_absolute_difference_px"]),
                    float(config["background_mad_scale"]) * ring_mad,
                )
                fill = difference <= tolerance
                reason = "disparity_continuous" if fill else "disparity_discontinuous"
                details.update(
                    {
                        "hole_median_disparity": hole_median,
                        "ring_median_disparity": ring_median,
                        "median_disparity_difference": difference,
                        "allowed_disparity_difference": tolerance,
                    }
                )
            elif area <= int(config["small_hole_area"]):
                fill = True
                reason = "small_without_disparity_support"
            else:
                reason = "insufficient_disparity_support"
        if fill:
            subject[hole] = True
        decisions.append({"fill": bool(fill), "reason": reason, **details})
    refined = largest_component(subject)
    return refined, {
        "changed_fraction": float(np.mean(refined != raw)),
        "removed_fraction": float(np.mean(raw & ~refined)),
        "added_fraction": float(np.mean(refined & ~raw)),
        "hole_count": len(decisions),
        "filled_hole_count": sum(item["fill"] for item in decisions),
        "decisions": decisions,
    }


def jop_reflective_rescue(
    category: str,
    probability: np.ndarray,
    normalized_disparity: np.ndarray,
    experiment4: np.ndarray,
    experiment5: np.ndarray,
    experiment5_diagnostics: dict,
    low_threshold: float,
    depth_interpolation: float,
    minimum_depth_reliability: float,
    minimum_base_fraction: float,
) -> tuple[np.ndarray, dict]:
    """Grow a reflective Jop1 subject through low-confidence, depth-consistent pixels.

    The trigger and growth use only model/disparity outputs. Ground truth is not read.
    Non-Jop1 samples and unreliable/small-base Jop1 samples keep Experiment 4 exactly.
    """
    base = np.asarray(experiment4, dtype=bool)
    diagnostics = {
        "triggered": False,
        "category": category,
        "base_fraction": float(np.mean(base)),
        "depth_reliability": float(
            experiment5_diagnostics.get("depth_reliability", 0.0)
        ),
        "low_threshold": float(low_threshold),
        "depth_interpolation": float(depth_interpolation),
        "minimum_depth_reliability": float(minimum_depth_reliability),
        "minimum_base_fraction": float(minimum_base_fraction),
        "changed_fraction": 0.0,
    }
    if category != "jop1":
        diagnostics["reason"] = "non_jop1"
        return base.copy(), diagnostics
    if diagnostics["depth_reliability"] < minimum_depth_reliability:
        diagnostics["reason"] = "insufficient_depth_reliability"
        return base.copy(), diagnostics
    if diagnostics["base_fraction"] < minimum_base_fraction:
        diagnostics["reason"] = "base_subject_too_small"
        return base.copy(), diagnostics

    foreground_median = float(
        experiment5_diagnostics["foreground_depth_median"]
    )
    background_median = float(
        experiment5_diagnostics["background_depth_median"]
    )
    depth_cutoff = background_median + depth_interpolation * (
        foreground_median - background_median
    )
    if foreground_median >= background_median:
        depth_gate = normalized_disparity >= depth_cutoff
        direction = "higher"
    else:
        depth_gate = normalized_disparity <= depth_cutoff
        direction = "lower"
    low_confidence_candidate = probability >= low_threshold
    rescued = np.asarray(experiment5, dtype=bool) | (
        low_confidence_candidate & depth_gate
    )
    diagnostics.update(
        {
            "triggered": True,
            "reason": "reflective_depth_rescue",
            "foreground_depth_median": foreground_median,
            "background_depth_median": background_median,
            "depth_cutoff": depth_cutoff,
            "foreground_depth_direction": direction,
            "changed_fraction": float(np.mean(rescued != base)),
            "added_fraction": float(np.mean(rescued & ~base)),
            "removed_fraction": float(np.mean(base & ~rescued)),
        }
    )
    return rescued, diagnostics


def confusion_counts(gt: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    encoded = gt.astype(np.uint8).reshape(-1) * 2 + prediction.astype(np.uint8).reshape(-1)
    return np.bincount(encoded, minlength=4).reshape(2, 2).astype(np.int64)


def metrics_from_confusion(confusion: np.ndarray) -> dict:
    tn, fp, fn, tp = (int(value) for value in confusion.reshape(-1))
    return {
        "foreground_iou": tp / max(tp + fp + fn, 1),
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "pixel_accuracy": (tp + tn) / max(int(confusion.sum()), 1),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def boundary_counts(
    gt: np.ndarray, prediction: np.ndarray, tolerance: int
) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    gt_edge = cv2.morphologyEx(gt.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    pred_edge = (
        cv2.morphologyEx(prediction.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
        > 0
    )
    band_kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=np.uint8)
    gt_band = cv2.dilate(gt_edge.astype(np.uint8), band_kernel) > 0
    pred_band = cv2.dilate(pred_edge.astype(np.uint8), band_kernel) > 0
    return np.asarray(
        [
            int((pred_edge & gt_band).sum()),
            int(pred_edge.sum()),
            int((gt_edge & pred_band).sum()),
            int(gt_edge.sum()),
        ],
        dtype=np.int64,
    )


def add_boundary_metrics(metrics: dict, counts: np.ndarray) -> None:
    matched_pred, pred_total, matched_gt, gt_total = (int(value) for value in counts)
    precision = matched_pred / max(pred_total, 1)
    recall = matched_gt / max(gt_total, 1)
    metrics.update(
        {
            "boundary_precision": precision,
            "boundary_recall": recall,
            "boundary_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        }
    )


def label_panel(panel: np.ndarray, label: str) -> np.ndarray:
    output = panel.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(
        output,
        label,
        (5, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def mask_panel(mask: np.ndarray, label: str) -> np.ndarray:
    return label_panel(cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR), label)


def make_comparison(image: np.ndarray, gt: np.ndarray, outputs: dict) -> np.ndarray:
    panels = [
        label_panel(image, "image"),
        mask_panel(gt, "human outer contour"),
        mask_panel(outputs["semantic"], "semantic"),
        mask_panel(outputs["experiment4"], "experiment 4"),
        mask_panel(outputs["experiment5"], "experiment 5 soft"),
        mask_panel(outputs["experiment4_plus_5"], "experiment 4+5 recall"),
    ]
    if "jop_reflective_rescue" in outputs:
        panels.append(
            mask_panel(outputs["jop_reflective_rescue"], "Jop reflective rescue")
        )
    rows = []
    for start in range(0, len(panels), 3):
        row = panels[start : start + 3]
        while len(row) < 3:
            row.append(np.zeros_like(panels[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def save_contact_sheet(paths: list[Path], output: Path) -> None:
    tiles = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        tiles.append(cv2.resize(image, (432, 512), interpolation=cv2.INTER_AREA))
    blank = np.zeros_like(tiles[0])
    rows = []
    for start in range(0, len(tiles), 3):
        row = tiles[start : start + 3]
        row.extend([blank] * (3 - len(row)))
        rows.append(np.hstack(row))
    cv2.imwrite(str(output), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    category_thresholds = parse_category_thresholds(args.category_threshold)
    if not 0.0 < args.semantic_threshold < 1.0:
        raise ValueError("--semantic-threshold must be in (0, 1)")
    if not 0.0 < args.rescue_low_threshold < 1.0:
        raise ValueError("--rescue-low-threshold must be in (0, 1)")
    if not 0.0 <= args.rescue_depth_interpolation <= 1.0:
        raise ValueError("--rescue-depth-interpolation must be in [0, 1]")
    if not 0.0 <= args.rescue_min_depth_reliability <= 1.0:
        raise ValueError("--rescue-min-depth-reliability must be in [0, 1]")
    if not 0.0 <= args.rescue_min_base_fraction <= 1.0:
        raise ValueError("--rescue-min-base-fraction must be in [0, 1]")
    if output.exists():
        raise FileExistsError("Use a new result version; output exists: {}".format(output))
    records = read_records(dataset, args.split)
    unknown_categories = set(category_thresholds) - {
        record["category"] for record in records
    }
    if unknown_categories:
        raise ValueError(
            "Category thresholds do not match this split: {}".format(
                ", ".join(sorted(unknown_categories))
            )
        )
    methods = METHODS + (("jop_reflective_rescue",) if args.jop_reflective_rescue else ())
    for method in methods:
        (output / "masks" / method).mkdir(parents=True, exist_ok=False)
    (output / "comparisons").mkdir(parents=True, exist_ok=False)

    overall_confusions = {method: np.zeros((2, 2), dtype=np.int64) for method in methods}
    overall_boundaries = {method: np.zeros(4, dtype=np.int64) for method in methods}
    category_confusions = defaultdict(
        lambda: {method: np.zeros((2, 2), dtype=np.int64) for method in methods}
    )
    category_boundaries = defaultdict(
        lambda: {method: np.zeros(4, dtype=np.int64) for method in methods}
    )
    metric_rows = []
    comparison_paths = []
    source_models = defaultdict(int)

    for index, record in enumerate(records, start=1):
        image = cv2.imread(str(dataset / record["image"]), cv2.IMREAD_COLOR)
        gt_u8 = cv2.imread(str(dataset / record["mask"]), cv2.IMREAD_GRAYSCALE)
        prob_path, selected_model = probability_path(
            record,
            args.split,
            args.semantic_source,
            args.probability_dir.resolve() if args.probability_dir else None,
        )
        disp_path = disparity_path(record)
        if image is None or gt_u8 is None:
            raise FileNotFoundError(record["name"])
        if not prob_path.is_file():
            raise FileNotFoundError(prob_path)
        if not disp_path.is_file():
            raise FileNotFoundError(disp_path)
        gt = gt_u8 > 0
        probability = np.load(prob_path, allow_pickle=False).astype(np.float32)
        disparity_full = np.load(disp_path, allow_pickle=False).astype(np.float32)
        if probability.shape != gt.shape:
            raise ValueError("Probability/GT mismatch for {}".format(record["name"]))
        disparity = cv2.resize(
            disparity_full,
            (gt.shape[1], gt.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        normalized, _, _, _ = FUSION.normalize_disparity(disparity_full, gt.shape)
        record_threshold = category_thresholds.get(
            record["category"], args.semantic_threshold
        )
        experiment4, exp4_diagnostics = experiment4_refine(
            probability, disparity, threshold=record_threshold
        )
        experiment5, exp5_diagnostics = FUSION.soft_fuse_semantic_and_disparity(
            probability, normalized
        )
        _, combined, combined_diagnostics = FUSION.combine_soft_fusion_recall_priority(
            probability >= record_threshold,
            experiment5,
            disparity,
            fusion_diagnostics=exp5_diagnostics,
            config=FUSION.scaled_recall_priority_config(probability.shape),
        )
        outputs = {
            "semantic": probability >= record_threshold,
            "experiment4": experiment4,
            "experiment5": experiment5,
            "experiment4_plus_5": combined,
        }
        rescue_diagnostics = None
        if args.jop_reflective_rescue:
            rescue, rescue_diagnostics = jop_reflective_rescue(
                record["category"],
                probability,
                normalized,
                experiment4,
                experiment5,
                exp5_diagnostics,
                args.rescue_low_threshold,
                args.rescue_depth_interpolation,
                args.rescue_min_depth_reliability,
                args.rescue_min_base_fraction,
            )
            outputs["jop_reflective_rescue"] = rescue
        source_models[selected_model] += 1
        row = {
            "name": record["name"],
            "category": record["category"],
            "selected_model": selected_model,
            "semantic_threshold": record_threshold,
            "experiment4_changed_fraction": exp4_diagnostics["changed_fraction"],
            "experiment5_changed_fraction": exp5_diagnostics["changed_fraction"],
            "combined_changed_fraction": combined_diagnostics["changed_fraction"],
        }
        if rescue_diagnostics is not None:
            row.update(
                {
                    "jop_rescue_triggered": rescue_diagnostics["triggered"],
                    "jop_rescue_reason": rescue_diagnostics["reason"],
                    "jop_rescue_changed_fraction": rescue_diagnostics[
                        "changed_fraction"
                    ],
                }
            )
        for method, prediction in outputs.items():
            confusion = confusion_counts(gt, prediction)
            boundaries = boundary_counts(gt, prediction, args.boundary_tolerance)
            metrics = metrics_from_confusion(confusion)
            add_boundary_metrics(metrics, boundaries)
            for key in ("foreground_iou", "precision", "recall", "boundary_f1"):
                row["{}_{}".format(method, key)] = metrics[key]
            overall_confusions[method] += confusion
            overall_boundaries[method] += boundaries
            category_confusions[record["category"]][method] += confusion
            category_boundaries[record["category"]][method] += boundaries
            cv2.imwrite(
                str(output / "masks" / method / "{}.png".format(record["name"])),
                prediction.astype(np.uint8) * 255,
            )
        metric_rows.append(row)
        comparison_path = output / "comparisons/{}.jpg".format(record["name"])
        cv2.imwrite(
            str(comparison_path),
            make_comparison(image, gt, outputs),
            [cv2.IMWRITE_JPEG_QUALITY, 94],
        )
        comparison_paths.append(comparison_path)
        print(
            "{}/{} {} [{} -> {}] semantic={:.4f} exp4={:.4f} exp5={:.4f} combined={:.4f}".format(
                index,
                len(records),
                record["name"],
                record["category"],
                selected_model,
                row["semantic_foreground_iou"],
                row["experiment4_foreground_iou"],
                row["experiment5_foreground_iou"],
                row["experiment4_plus_5_foreground_iou"],
            ),
            flush=True,
        )

    fieldnames = list(metric_rows[0])
    with (output / "per_scene.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    overall = {}
    for method in methods:
        metrics = metrics_from_confusion(overall_confusions[method])
        add_boundary_metrics(metrics, overall_boundaries[method])
        overall[method] = metrics
    per_category = {}
    for category in sorted(category_confusions):
        per_category[category] = {}
        for method in methods:
            metrics = metrics_from_confusion(category_confusions[category][method])
            add_boundary_metrics(metrics, category_boundaries[category][method])
            per_category[category][method] = metrics

    method_descriptions = {
        "semantic": (
            "raw foreground probability >= category threshold when configured, "
            "otherwise the global threshold"
        ),
        "experiment4": "closing, largest component, disparity-continuous hole repair",
        "experiment5": "V1 semantic probability + disparity soft fusion",
        "experiment4_plus_5": "recall-priority soft fusion with selective disparity holes",
    }
    if args.jop_reflective_rescue:
        method_descriptions["jop_reflective_rescue"] = (
            "Experiment 4 by default; for large, depth-reliable Jop1 masks, grow "
            "Experiment 5 through low-confidence pixels on the foreground side of "
            "an interpolated semantic foreground/background disparity cutoff"
        )
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "trained multi-domain BiSeNetV2 with Experiment 4/5 processing",
        "dataset": str(dataset),
        "split": "development_comparison_{}".format(args.split),
        "count": len(records),
        "semantic_source": (
            "custom_probability_dir"
            if args.probability_dir is not None
            else args.semantic_source
        ),
        "custom_probability_dir": (
            str(args.probability_dir.resolve()) if args.probability_dir else None
        ),
        "semantic_threshold": args.semantic_threshold,
        "category_thresholds": category_thresholds,
        "jop_reflective_rescue": {
            "enabled": args.jop_reflective_rescue,
            "low_threshold": args.rescue_low_threshold,
            "depth_interpolation": args.rescue_depth_interpolation,
            "minimum_depth_reliability": args.rescue_min_depth_reliability,
            "minimum_base_fraction": args.rescue_min_base_fraction,
        },
        "source_model_counts": dict(source_models),
        "methods": method_descriptions,
        "overall": overall,
        "per_category": per_category,
        "ground_truth_usage": "metrics only; no method reads GT",
        "outer_contour_warning": (
            "The human masks contain only outer contours and no background-hole polygons. "
            "Metrics reward filled internal holes and cannot measure true through-hole quality."
        ),
        "test_reuse_warning": (
            "This test split has already informed earlier model development and is a "
            "development comparison, not a pristine final generalization test."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_contact_sheet(comparison_paths, output / "contact_sheet.jpg")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
