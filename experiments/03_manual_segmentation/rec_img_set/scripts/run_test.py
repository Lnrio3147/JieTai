#!/usr/bin/env python3
"""Evaluate frozen BiSeNetV2 post-masking on non-FDJYP3 rec_img_set groups.

The script consumes saved BiSeNetV2 probabilities and the already verified
full-resolution IGEV++ RT / LiteAnyStereo disparities.  It does not run either
stereo network again.  FDJYP-0 human masks are used for an in-domain regression
check; the other groups have no segmentation ground truth and are reported as
qualitative/diagnostic only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT_ROOT.parents[2]
TRADITION_ROOT = ROOT / "projects/tradition_stereo"
LAS_ROOT = ROOT / "projects/LiteAnyStereo"
REFINEMENT_SCRIPT = LAS_ROOT / "tools/refine_bisenet_subject_masks.py"
CROP = (234, 1052, 126, 638)


@dataclass(frozen=True)
class GroupSpec:
    key: str
    source_dir: str


GROUPS = (
    GroupSpec("fdjyp0", "FDJYP-0-rectified_images"),
    GroupSpec("luowen", "luowen_rectified_images"),
    GroupSpec("general_1221", "rectified_images"),
    GroupSpec("scale_1221", "rectified_images_刻度"),
)
GROUP_BY_KEY = {group.key: group for group in GROUPS}


def load_refinement_module():
    spec = importlib.util.spec_from_file_location(
        "refine_bisenet_subject_masks", REFINEMENT_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load refinement code: {REFINEMENT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REFINEMENT = load_refinement_module()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, default=ROOT / "datasets/rec_img_set"
    )
    parser.add_argument(
        "--stereo-results-root",
        type=Path,
        default=ROOT / "experiments/01_stereo_comparison/rec_img_set/results/final_203",
    )
    parser.add_argument("--bisenet-raw-root", type=Path, required=True)
    parser.add_argument(
        "--manual-seg-root",
        type=Path,
        default=ROOT / "datasets/annotations/JMP-workpiece-seg-manual-isat-v1",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-disp", type=float, default=192.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--closing-radius", type=int, default=3)
    parser.add_argument("--hole-ring-radius", type=int, default=7)
    parser.add_argument("--hole-absolute-tolerance", type=float, default=1.5)
    parser.add_argument("--hole-mad-scale", type=float, default=1.0)
    parser.add_argument("--max-fill-hole-fraction", type=float, default=0.025)
    parser.add_argument("--small-hole-area", type=int, default=1000)
    parser.add_argument("--contact-sheet-samples", type=int, default=18)
    return parser.parse_args()


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(base: Path, value: str):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def scene_from_manual_name(name: str):
    parts = name.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Cannot map manual-mask name to scene: {name!r}")
    return f"{parts[-2]}-{parts[-1]}"


def load_manual_masks(root: Path):
    records = {}
    for split in ("train", "val"):
        index_path = root / "index" / f"{split}.csv"
        with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if not row["name"].startswith("fdjyp_0_"):
                    continue
                scene = scene_from_manual_name(row["name"])
                if scene in records:
                    raise ValueError(f"Duplicate manual mask for {scene}")
                records[scene] = {
                    "split": split,
                    "name": row["name"],
                    "mask": resolve(root, row["mask"]),
                }
    return records


def load_csv_by_key(path: Path, key_fields):
    records = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = tuple(row[field] for field in key_fields)
            records[key] = row
    return records


def discover(source_root: Path, probability_dir: Path):
    samples = []
    for probability_path in sorted(probability_dir.glob("*.npy")):
        if "__" not in probability_path.stem:
            raise ValueError(f"Expected <group>__<scene>.npy: {probability_path}")
        group_key, scene = probability_path.stem.split("__", 1)
        if group_key not in GROUP_BY_KEY:
            raise ValueError(f"Unknown group in {probability_path.name}: {group_key}")
        group = GROUP_BY_KEY[group_key]
        scene_root = source_root / group.source_dir / scene
        left_path = scene_root / "im0.png"
        right_path = scene_root / "im1.png"
        if not left_path.is_file() or not right_path.is_file():
            raise FileNotFoundError(scene_root)
        samples.append((group, scene, left_path, right_path, probability_path))
    if not samples:
        raise FileNotFoundError(f"No probability maps in {probability_dir}")
    return samples


def binary_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError(f"Mask shape mismatch: {prediction.shape} vs {target.shape}")
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    tn = int(np.count_nonzero(~prediction & ~target))

    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else 1.0

    foreground_iou = ratio(tp, tp + fp + fn)
    background_iou = ratio(tn, tn + fp + fn)
    return {
        "foreground_iou": foreground_iou,
        "background_iou": background_iou,
        "mean_iou": 0.5 * (foreground_iou + background_iou),
        "dice": ratio(2 * tp, 2 * tp + fp + fn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "pixel_accuracy": ratio(tp + tn, tp + fp + fn + tn),
    }


def difference_metrics(first, second, region=None):
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    if first.shape != second.shape:
        raise ValueError(f"Disparity shape mismatch: {first.shape} vs {second.shape}")
    valid = np.isfinite(first) & np.isfinite(second)
    if region is not None:
        valid &= np.asarray(region, dtype=bool)
    count = int(valid.sum())
    if not count:
        return {"pixels": 0, "mae_px": None, "median_px": None, "p95_px": None, "bad3_pct": None}
    difference = np.abs(first[valid] - second[valid])
    return {
        "pixels": count,
        "mae_px": float(difference.mean()),
        "median_px": float(np.median(difference)),
        "p95_px": float(np.percentile(difference, 95)),
        "bad3_pct": float((difference > 3.0).mean() * 100.0),
    }


def colorize(values, maximum, valid=None):
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    normalized = np.clip(values / float(maximum), 0.0, 1.0)
    normalized = np.where(finite, normalized, 0.0)
    result = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    mask = finite if valid is None else finite & np.asarray(valid, dtype=bool)
    result[~mask] = 0
    return result


def overlay(image, mask, color):
    result = image.copy()
    mask = np.asarray(mask, dtype=bool)
    result[mask] = (
        0.45 * result[mask] + 0.55 * np.asarray(color, dtype=np.float32)
    ).astype(np.uint8)
    return result


def label_panel(image, text):
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        result, text, (7, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    return result


def save_comparison(
    path, left, raw_mask, refined_mask, las, igev, max_disp, manual_mask=None
):
    subject = np.where(refined_mask, las, np.nan)
    if manual_mask is not None:
        last_panel = label_panel(
            overlay(left, manual_mask, (255, 0, 255)), "FDJYP-0 human mask (in-domain)"
        )
    else:
        disagreement = np.abs(las - igev)
        last_panel = label_panel(
            colorize(disagreement, 20.0, refined_mask),
            "LAS/IGEV subject difference [0,20] (not error)",
        )
    panels = [
        label_panel(left, "Rectified left"),
        label_panel(overlay(left, raw_mask, (0, 0, 255)), "Raw BiSeNetV2 mask"),
        label_panel(overlay(left, refined_mask, (0, 255, 0)), "Refined one-component mask"),
        label_panel(colorize(las, max_disp), "LAS1 complete disparity"),
        label_panel(colorize(subject, max_disp), "LAS1 subject disparity"),
        last_panel,
    ]
    comparison = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])])
    if not cv2.imwrite(str(path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 91]):
        raise OSError(f"Failed to write {path}")


def save_contact_sheet(paths, path, sample_count):
    count = min(sample_count, len(paths))
    indices = np.linspace(0, len(paths) - 1, num=count, dtype=np.int32)
    tiles = []
    for index in indices:
        image = cv2.imread(str(paths[int(index)]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(paths[int(index)])
        width = 480
        height = int(round(image.shape[0] * width / image.shape[1]))
        tiles.append(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
    rows = []
    for start in range(0, len(tiles), 3):
        row = tiles[start : start + 3]
        while len(row) < 3:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    if not cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 91]):
        raise OSError(f"Failed to write {path}")


def add_prefixed(record, prefix, values):
    for key, value in values.items():
        record[f"{prefix}_{key}"] = value


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def summarize_rows(rows, prefix, keys):
    result = {}
    for key in keys:
        values = [row[f"{prefix}_{key}"] for row in rows if row.get(f"{prefix}_{key}") is not None]
        if values:
            result[key] = summarize(values)
    return result


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    stereo_root = args.stereo_results_root.expanduser().resolve()
    raw_root = args.bisenet_raw_root.expanduser().resolve()
    manual_root = args.manual_seg_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    probability_dir = raw_root / "probabilities"
    for path in (source_root, stereo_root, raw_root, probability_dir, manual_root):
        if not path.exists():
            raise FileNotFoundError(path)
    summary_path = output_root / "metrics/summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Completed output already exists: {summary_path}")
    outputs_dir = output_root / "outputs"
    metrics_dir = output_root / "metrics"
    assets_dir = output_root / "report_assets"
    if outputs_dir.exists() or metrics_dir.exists() or assets_dir.exists():
        raise FileExistsError(f"Partial evaluation output already exists: {output_root}")
    outputs_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)
    assets_dir.mkdir(parents=True)

    samples = discover(source_root, probability_dir)
    expected_counts = {"fdjyp0": 82, "luowen": 37, "general_1221": 6, "scale_1221": 5}
    actual_counts = Counter(group.key for group, *_ in samples)
    if dict(actual_counts) != expected_counts:
        raise RuntimeError(f"Unexpected group counts: {dict(actual_counts)}")
    prediction_rows = load_csv_by_key(raw_root / "predictions.csv", ("name",))
    stereo_rows = load_csv_by_key(stereo_root / "metrics/per_scene.csv", ("group", "scene"))
    manual_masks = load_manual_masks(manual_root)
    if len(manual_masks) != 82:
        raise RuntimeError(f"Expected 82 FDJYP-0 human masks, found {len(manual_masks)}")

    y0, y1, x0, x1 = CROP
    rows = []
    comparisons_by_group = defaultdict(list)
    hole_decisions = {}
    for index, (group, scene, left_path, right_path, probability_path) in enumerate(samples, start=1):
        key_name = f"{group.key}__{scene}"
        left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left is None or right is None or left.shape != right.shape:
            raise ValueError(f"Invalid stereo pair: {group.key}/{scene}")
        if left.shape[:2] != (1280, 720):
            raise ValueError(f"Unexpected shape in {group.key}/{scene}: {left.shape}")
        stereo_scene = stereo_root / "outputs" / group.key / scene
        las_path = stereo_scene / "liteanystereo/disp_full.npy"
        igev_path = stereo_scene / "igev_rt/disp_full.npy"
        las = np.load(las_path, allow_pickle=False).astype(np.float32)
        igev = np.load(igev_path, allow_pickle=False).astype(np.float32)
        probability = np.load(probability_path, allow_pickle=False).astype(np.float32)
        if las.shape != left.shape[:2] or igev.shape != left.shape[:2]:
            raise ValueError(f"Saved disparity shape mismatch: {group.key}/{scene}")

        height, width = left.shape[:2]
        raw_mask, refined_mask, stats = REFINEMENT.refine_mask(
            probability,
            (height, width),
            las,
            threshold=args.threshold,
            closing_radius=args.closing_radius,
            hole_ring_radius=args.hole_ring_radius,
            hole_absolute_tolerance=args.hole_absolute_tolerance,
            hole_mad_scale=args.hole_mad_scale,
            max_fill_hole_fraction=args.max_fill_hole_fraction,
            small_hole_area=args.small_hole_area,
            crop=(0, height, 0, width),
        )
        if stats["final_foreground_components"] != 1:
            raise AssertionError(f"Final mask is not one component: {group.key}/{scene}")
        scene_dir = outputs_dir / group.key / scene
        scene_dir.mkdir(parents=True)
        cv2.imwrite(str(scene_dir / "raw_mask.png"), raw_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(scene_dir / "foreground_mask.png"), refined_mask.astype(np.uint8) * 255)
        subject_full = np.where(refined_mask, las, np.nan).astype(np.float32)
        subject_crop = subject_full[y0:y1, x0:x1]
        np.save(scene_dir / "disp_subject_full.npy", subject_full, allow_pickle=False)
        np.save(scene_dir / "disp_subject_crop.npy", subject_crop, allow_pickle=False)
        cv2.imwrite(
            str(scene_dir / "disp_subject_crop_color.png"), colorize(subject_crop, args.max_disp)
        )

        manual = manual_masks.get(scene) if group.key == "fdjyp0" else None
        manual_full = None
        manual_split = None
        raw_segmentation = {}
        refined_segmentation = {}
        if manual is not None:
            manual_small = cv2.imread(str(manual["mask"]), cv2.IMREAD_GRAYSCALE)
            raw_small_path = raw_root / "masks" / f"{key_name}.png"
            raw_small = cv2.imread(str(raw_small_path), cv2.IMREAD_GRAYSCALE)
            if manual_small is None or raw_small is None:
                raise FileNotFoundError(manual["mask"])
            manual_bool = manual_small > 0
            raw_segmentation = binary_metrics(raw_small > 0, manual_bool)
            refined_small = cv2.resize(
                refined_mask.astype(np.uint8),
                (manual_small.shape[1], manual_small.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            refined_segmentation = binary_metrics(refined_small, manual_bool)
            manual_full = cv2.resize(
                manual_small,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
            manual_split = manual["split"]

        comparison_path = scene_dir / "comparison.jpg"
        save_comparison(
            comparison_path,
            left,
            raw_mask,
            refined_mask,
            las,
            igev,
            args.max_disp,
            manual_mask=manual_full,
        )
        comparisons_by_group[group.key].append(comparison_path)

        roi_mask = refined_mask[y0:y1, x0:x1]
        las_crop = las[y0:y1, x0:x1]
        igev_crop = igev[y0:y1, x0:x1]
        all_difference = difference_metrics(las_crop, igev_crop)
        subject_difference = difference_metrics(las_crop, igev_crop, roi_mask)
        background_difference = difference_metrics(las_crop, igev_crop, ~roi_mask)
        prediction_record = prediction_rows[(key_name,)]
        stereo_record = stereo_rows[(group.key, scene)]
        hole_decisions[f"{group.key}/{scene}"] = stats.pop("hole_decisions")
        record = {
            "group": group.key,
            "source_dir": group.source_dir,
            "scene": scene,
            "manual_split": manual_split,
            "geometry_status": stereo_record["geometry_status"],
            "median_vertical_residual_px": float(stereo_record["median_vertical_residual_px"]),
            "mean_confidence": float(prediction_record["mean_confidence"]),
            "uncertain_fraction": float(prediction_record["uncertain_fraction"]),
            "raw_foreground_fraction": float(raw_mask.mean()),
            "refined_foreground_fraction": float(refined_mask.mean()),
            "roi_foreground_fraction": float(roi_mask.mean()),
            **stats,
        }
        add_prefixed(record, "all_difference", all_difference)
        add_prefixed(record, "subject_difference", subject_difference)
        add_prefixed(record, "background_difference", background_difference)
        add_prefixed(record, "raw_segmentation", raw_segmentation)
        add_prefixed(record, "refined_segmentation", refined_segmentation)
        rows.append(record)
        print(
            f"[{index:03d}/{len(samples):03d}] {group.key}/{scene} "
            f"fg={refined_mask.mean():.3f} conf={record['mean_confidence']:.3f} "
            f"LAS/IGEV subject MAE={subject_difference['mae_px']:.3f}",
            flush=True,
        )

    write_csv(metrics_dir / "per_scene.csv", rows)
    (metrics_dir / "hole_decisions.json").write_text(
        json.dumps(hole_decisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for group_key, paths in comparisons_by_group.items():
        save_contact_sheet(
            paths,
            assets_dir / f"overview_{group_key}.jpg",
            args.contact_sheet_samples,
        )

    group_summaries = {}
    for group in GROUPS:
        group_rows = [row for row in rows if row["group"] == group.key]
        group_summary = {
            "count": len(group_rows),
            "geometry_status": dict(Counter(row["geometry_status"] for row in group_rows)),
            "mean_confidence": summarize([row["mean_confidence"] for row in group_rows]),
            "uncertain_fraction": summarize([row["uncertain_fraction"] for row in group_rows]),
            "raw_foreground_fraction": summarize(
                [row["raw_foreground_fraction"] for row in group_rows]
            ),
            "refined_foreground_fraction": summarize(
                [row["refined_foreground_fraction"] for row in group_rows]
            ),
            "roi_foreground_fraction": summarize(
                [row["roi_foreground_fraction"] for row in group_rows]
            ),
            "topology_totals": {
                "initial_components_after_closing": int(
                    sum(row["initial_components_after_closing"] for row in group_rows)
                ),
                "removed_island_pixels": int(
                    sum(row["removed_island_pixels"] for row in group_rows)
                ),
                "holes": int(sum(row["hole_count"] for row in group_rows)),
                "filled_holes": int(sum(row["filled_hole_count"] for row in group_rows)),
                "filled_hole_pixels": int(
                    sum(row["filled_hole_pixels"] for row in group_rows)
                ),
                "preserved_holes": int(
                    sum(row["preserved_hole_count"] for row in group_rows)
                ),
                "final_masks_with_exactly_one_component": int(
                    sum(row["final_foreground_components"] == 1 for row in group_rows)
                ),
            },
            "las_igev_difference": {
                "all": summarize_rows(
                    group_rows, "all_difference", ("mae_px", "median_px", "p95_px", "bad3_pct")
                ),
                "subject": summarize_rows(
                    group_rows, "subject_difference", ("mae_px", "median_px", "p95_px", "bad3_pct")
                ),
                "background": summarize_rows(
                    group_rows, "background_difference", ("mae_px", "median_px", "p95_px", "bad3_pct")
                ),
                "interpretation": "Inter-model difference is diagnostic disagreement, not error.",
            },
            "highest_uncertainty_scenes": [
                {"scene": row["scene"], "uncertain_fraction": row["uncertain_fraction"]}
                for row in sorted(group_rows, key=lambda item: item["uncertain_fraction"], reverse=True)[:5]
            ],
        }
        manual_rows = [row for row in group_rows if row["manual_split"]]
        if manual_rows:
            group_summary["manual_segmentation_regression"] = {
                "scope_warning": (
                    "FDJYP-0 labels participated in training or model selection; these are not independent test metrics."
                ),
                "all_82": {
                    "raw": summarize_rows(
                        manual_rows,
                        "raw_segmentation",
                        ("foreground_iou", "mean_iou", "dice", "precision", "recall", "pixel_accuracy"),
                    ),
                    "refined": summarize_rows(
                        manual_rows,
                        "refined_segmentation",
                        ("foreground_iou", "mean_iou", "dice", "precision", "recall", "pixel_accuracy"),
                    ),
                },
                "train_64": {
                    "raw": summarize_rows(
                        [row for row in manual_rows if row["manual_split"] == "train"],
                        "raw_segmentation",
                        ("foreground_iou", "dice", "precision", "recall"),
                    ),
                    "refined": summarize_rows(
                        [row for row in manual_rows if row["manual_split"] == "train"],
                        "refined_segmentation",
                        ("foreground_iou", "dice", "precision", "recall"),
                    ),
                },
                "validation_18": {
                    "raw": summarize_rows(
                        [row for row in manual_rows if row["manual_split"] == "val"],
                        "raw_segmentation",
                        ("foreground_iou", "dice", "precision", "recall"),
                    ),
                    "refined": summarize_rows(
                        [row for row in manual_rows if row["manual_split"] == "val"],
                        "refined_segmentation",
                        ("foreground_iou", "dice", "precision", "recall"),
                    ),
                },
            }
        group_summaries[group.key] = group_summary

    bisenet_metadata_path = raw_root / "metadata.json"
    source_stereo_summary = stereo_root / "metrics/summary.json"
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "130 unique non-FDJYP3 rec_img_set scenes; kedu duplicate excluded",
        "scene_count": len(rows),
        "group_counts": dict(actual_counts),
        "excluded": {
            "fdjyp3": "already evaluated in the dedicated 73-scene pipeline",
            "kedu": "byte-identical duplicate of scale_1221",
        },
        "pipeline": (
            "frozen BiSeNetV2 -> probability upsample -> one-component/disparity-continuity refinement; "
            "saved LAS1 full disparity is post-masked only"
        ),
        "models": {
            "bisenet_metadata": str(bisenet_metadata_path),
            "bisenet_metadata_sha256": sha256_file(bisenet_metadata_path),
            "bisenet_pb": json.loads(bisenet_metadata_path.read_text(encoding="utf-8"))["model_pb"],
            "bisenet_pb_sha256": json.loads(
                bisenet_metadata_path.read_text(encoding="utf-8")
            )["model_pb_sha256"],
            "saved_stereo_summary": str(source_stereo_summary),
            "saved_stereo_summary_sha256": sha256_file(source_stereo_summary),
        },
        "refinement": {
            "ground_truth_used": False,
            "parameters": {
                "threshold": args.threshold,
                "closing_radius": args.closing_radius,
                "hole_ring_radius": args.hole_ring_radius,
                "hole_absolute_tolerance": args.hole_absolute_tolerance,
                "hole_mad_scale": args.hole_mad_scale,
                "max_fill_hole_fraction": args.max_fill_hole_fraction,
                "small_hole_area": args.small_hole_area,
            },
        },
        "groups": group_summaries,
        "limitations": [
            "Only FDJYP-0 has manual segmentation masks, and those images participated in training/model selection.",
            "Luowen, general_1221, and scale_1221 have no segmentation ground truth; confidence and coverage are not accuracy.",
            "LAS/IGEV disparity differences are disagreement diagnostics, not error without ground truth.",
            "General_1221 and part of luowen have known epipolar-geometry risks from the prior feasibility audit.",
            "The one-component prior may be invalid if a future scene contains multiple legitimate visible workpieces.",
        ],
        "outputs": {
            "per_scene_csv": str(metrics_dir / "per_scene.csv"),
            "hole_decisions": str(metrics_dir / "hole_decisions.json"),
            "report_assets": str(assets_dir),
            "scene_outputs": str(outputs_dir),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
