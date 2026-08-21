#!/usr/bin/env python3
"""Evaluate the frozen JMP BiSeNetV2 + LAS1 post-mask pipeline on Jop1.

The script reuses the rectified Jop1 left images and verified full-resolution
LAS1 disparities produced by ``run_comparison.py``.  It refines
the saved BiSeNetV2 foreground probabilities without using the supplied PLY,
then saves one-component masks, subject disparities, point clouds, metrics,
and visual comparisons.  The sparse PLY projection is used only after mask
generation as an external consistency reference; it is not segmentation GT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT_ROOT.parents[2]
STEREO_RESULTS = ROOT / "experiments/01_stereo_comparison/jop1/results/final_9"
LAS_ROOT = ROOT / "projects/LiteAnyStereo"
REFINEMENT_SCRIPT = LAS_ROOT / "tools/refine_bisenet_subject_masks.py"


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
        "--preprocessed-root",
        type=Path,
        default=STEREO_RESULTS / "preprocessed",
    )
    parser.add_argument(
        "--las-root",
        type=Path,
        default=STEREO_RESULTS / "liteanystereo",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=STEREO_RESULTS / "reference",
    )
    parser.add_argument("--probability-dir", type=Path, required=True)
    parser.add_argument(
        "--bisenet-metadata",
        type=Path,
        default=None,
        help="Raw BiSeNetV2 metadata.json used for model provenance.",
    )
    parser.add_argument(
        "--las-summary",
        type=Path,
        default=STEREO_RESULTS / "metrics/summary.json",
    )
    parser.add_argument(
        "--las-checkpoint",
        type=Path,
        default=LAS_ROOT / "checkpoints/LiteAnyStereo.pth",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=ROOT / "projects/tradition_stereo/config/stereo.yml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EXPERIMENT_ROOT / "results/result",
    )
    parser.add_argument("--max-disp", type=float, default=192.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--closing-radius", type=int, default=3)
    parser.add_argument("--hole-ring-radius", type=int, default=7)
    parser.add_argument("--hole-absolute-tolerance", type=float, default=1.5)
    parser.add_argument("--hole-mad-scale", type=float, default=1.0)
    parser.add_argument("--max-fill-hole-fraction", type=float, default=0.025)
    parser.add_argument("--small-hole-area", type=int, default=1000)
    parser.add_argument("--no-pointcloud", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_calibration_q(path: Path):
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise FileNotFoundError(f"Cannot open calibration: {path}")
    q = storage.getNode("Q").mat()
    storage.release()
    if q is None:
        raise ValueError(f"Calibration has no Q matrix: {path}")
    return q


def discover_scenes(preprocessed_root: Path, probability_dir: Path):
    scenes = []
    for left_path in sorted(preprocessed_root.glob("*/left.png")):
        scene = left_path.parent.name
        probability_path = probability_dir / f"{scene}.npy"
        if not probability_path.is_file():
            raise FileNotFoundError(probability_path)
        scenes.append((scene, left_path, probability_path))
    if not scenes:
        raise FileNotFoundError(f"No */left.png scenes in {preprocessed_root}")
    return scenes


def compute_region_metrics(prediction, reference, region=None):
    prediction = np.asarray(prediction, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if prediction.shape != reference.shape:
        raise ValueError(f"Prediction/reference shape mismatch: {prediction.shape} vs {reference.shape}")
    reference_valid = np.isfinite(reference) & (reference > 0)
    if region is not None:
        region = np.asarray(region, dtype=bool)
        if region.shape != reference.shape:
            raise ValueError(f"Region/reference shape mismatch: {region.shape} vs {reference.shape}")
        reference_valid &= region
    prediction_valid = np.isfinite(prediction) & (prediction > 0)
    evaluated = reference_valid & prediction_valid
    reference_pixels = int(reference_valid.sum())
    evaluated_pixels = int(evaluated.sum())
    result = {
        "reference_pixels": reference_pixels,
        "evaluated_pixels": evaluated_pixels,
        "prediction_coverage": (
            float(evaluated_pixels / reference_pixels) if reference_pixels else 0.0
        ),
    }
    if not evaluated_pixels:
        return result
    error = np.abs(prediction[evaluated] - reference[evaluated])
    relative = error / np.maximum(np.abs(reference[evaluated]), 1e-6)
    result.update(
        {
            "epe_px": float(error.mean()),
            "bad1_pct": float((error > 1.0).mean() * 100.0),
            "bad2_pct": float((error > 2.0).mean() * 100.0),
            "bad3_pct": float((error > 3.0).mean() * 100.0),
            "d1_pct": float(((error > 3.0) & (relative > 0.05)).mean() * 100.0),
        }
    )
    return result


def colorize(disparity, maximum, valid=None):
    disparity = np.asarray(disparity, dtype=np.float32)
    finite = np.isfinite(disparity)
    normalized = np.clip(disparity / float(maximum), 0.0, 1.0)
    normalized = np.where(finite, normalized, 0.0)
    result = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    if valid is not None:
        result[~np.asarray(valid, dtype=bool)] = 0
    else:
        result[~finite] = 0
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
        result,
        text,
        (7, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def save_comparison(path, left, raw_mask, refined_mask, disparity, reference, max_disp):
    subject = np.where(refined_mask, disparity, np.nan)
    panels = [
        label_panel(left, "Rectified left"),
        label_panel(overlay(left, raw_mask, (0, 0, 255)), "Raw BiSeNetV2 mask"),
        label_panel(overlay(left, refined_mask, (0, 255, 0)), "Refined one-component mask"),
        label_panel(colorize(disparity, max_disp), "LAS1 complete disparity"),
        label_panel(colorize(subject, max_disp), "LAS1 subject disparity"),
        label_panel(colorize(reference, max_disp, reference > 0), "Sparse supplied PLY reference"),
    ]
    comparison = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])])
    if not cv2.imwrite(str(path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"Failed to write comparison: {path}")


def save_overview(comparisons, path):
    tiles = []
    for comparison_path in comparisons:
        image = cv2.imread(str(comparison_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(comparison_path)
        width = 540
        height = int(round(image.shape[0] * width / image.shape[1]))
        tiles.append(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
    rows = []
    for start in range(0, len(tiles), 3):
        row = tiles[start : start + 3]
        while len(row) < 3:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    if not cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"Failed to write overview: {path}")


def save_point_cloud(path, disparity, left_bgr, q, max_disp):
    import open3d as o3d

    points_3d = cv2.reprojectImageTo3D(disparity.astype(np.float32), q)
    z = points_3d[..., 2]
    valid = (
        np.isfinite(disparity)
        & (disparity > 0)
        & (disparity <= max_disp)
        & np.isfinite(points_3d).all(axis=2)
        & (z > 0)
        & (z < 200)
    )
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_3d[valid].astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(
        cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)[valid].astype(np.float64) / 255.0
    )
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False)
    return int(valid.sum())


def prefixed(record, prefix, values):
    for key, value in values.items():
        record[f"{prefix}_{key}"] = value


def metric_macro(rows, prefix):
    result = {}
    for key in ("reference_pixels", "prediction_coverage", "epe_px", "d1_pct", "bad1_pct", "bad2_pct", "bad3_pct"):
        values = [float(row[f"{prefix}_{key}"]) for row in rows if f"{prefix}_{key}" in row]
        if values:
            result[key] = {
                "macro_mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    return result


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    preprocessed_root = args.preprocessed_root.expanduser().resolve()
    las_root = args.las_root.expanduser().resolve()
    reference_root = args.reference_root.expanduser().resolve()
    probability_dir = args.probability_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    calibration = args.calibration.expanduser().resolve()
    las_checkpoint = args.las_checkpoint.expanduser().resolve()
    las_summary = args.las_summary.expanduser().resolve()
    bisenet_metadata = (
        args.bisenet_metadata.expanduser().resolve()
        if args.bisenet_metadata is not None
        else probability_dir.parent / "metadata.json"
    )
    for path in (preprocessed_root, las_root, reference_root, probability_dir):
        if not path.is_dir():
            raise FileNotFoundError(path)
    for path in (calibration, las_checkpoint, las_summary, bisenet_metadata):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (output_root / "metrics/summary.json").exists():
        raise FileExistsError(f"Completed output already exists: {output_root}")
    scenes_dir = output_root / "scenes"
    metrics_dir = output_root / "metrics"
    if scenes_dir.exists() or metrics_dir.exists():
        raise FileExistsError(f"Partial evaluation output already exists: {output_root}")
    scenes_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)

    q = load_calibration_q(calibration)
    scenes = discover_scenes(preprocessed_root, probability_dir)
    rows = []
    comparisons = []
    hole_decisions = {}
    for index, (scene, left_path, probability_path) in enumerate(scenes, start=1):
        left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        if left is None:
            raise FileNotFoundError(left_path)
        disparity_path = las_root / scene / "disp.npy"
        reference_path = reference_root / scene / "disp.npy"
        disparity = np.load(disparity_path, allow_pickle=False).astype(np.float32)
        reference = np.load(reference_path, allow_pickle=False).astype(np.float32)
        probability = np.load(probability_path, allow_pickle=False).astype(np.float32)
        if disparity.shape != left.shape[:2] or reference.shape != left.shape[:2]:
            raise ValueError(
                f"Full-resolution shape mismatch in {scene}: "
                f"left={left.shape[:2]}, LAS={disparity.shape}, PLY={reference.shape}"
            )
        if probability.ndim != 2 or not np.isfinite(probability).all():
            raise ValueError(f"Invalid probability map: {probability_path}")

        height, width = left.shape[:2]
        full_crop = (0, height, 0, width)
        raw_mask, refined_mask, stats = REFINEMENT.refine_mask(
            probability,
            (height, width),
            disparity,
            threshold=args.threshold,
            closing_radius=args.closing_radius,
            hole_ring_radius=args.hole_ring_radius,
            hole_absolute_tolerance=args.hole_absolute_tolerance,
            hole_mad_scale=args.hole_mad_scale,
            max_fill_hole_fraction=args.max_fill_hole_fraction,
            small_hole_area=args.small_hole_area,
            crop=full_crop,
        )
        if stats["final_foreground_components"] != 1:
            raise AssertionError(f"Final mask is not one component: {scene}")
        scene_dir = scenes_dir / scene
        scene_dir.mkdir()
        raw_mask_path = scene_dir / "raw_mask.png"
        mask_path = scene_dir / "foreground_mask.png"
        cv2.imwrite(str(raw_mask_path), raw_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(mask_path), refined_mask.astype(np.uint8) * 255)
        subject = np.where(refined_mask, disparity, np.nan).astype(np.float32)
        np.save(scene_dir / "disp_subject.npy", subject, allow_pickle=False)
        cv2.imwrite(str(scene_dir / "disparity_subject_color.png"), colorize(subject, args.max_disp))
        comparison_path = scene_dir / "comparison.jpg"
        save_comparison(
            comparison_path,
            left,
            raw_mask,
            refined_mask,
            disparity,
            reference,
            args.max_disp,
        )
        comparisons.append(comparison_path)
        point_count = None
        if not args.no_pointcloud:
            point_count = save_point_cloud(
                scene_dir / "cloud_subject.ply", subject, left, q, args.max_disp
            )

        all_metrics = compute_region_metrics(disparity, reference)
        subject_metrics = compute_region_metrics(disparity, reference, refined_mask)
        background_metrics = compute_region_metrics(disparity, reference, ~refined_mask)
        reference_retained = (
            subject_metrics["reference_pixels"] / all_metrics["reference_pixels"]
            if all_metrics["reference_pixels"]
            else 0.0
        )
        hole_decisions[scene] = stats.pop("hole_decisions")
        record = {
            "scene": scene,
            "height": height,
            "width": width,
            "raw_foreground_fraction": float(raw_mask.mean()),
            "refined_foreground_fraction": float(refined_mask.mean()),
            "reference_retained_fraction": float(reference_retained),
            "subject_cloud_points": point_count,
            **stats,
        }
        prefixed(record, "all", all_metrics)
        prefixed(record, "subject", subject_metrics)
        prefixed(record, "background", background_metrics)
        rows.append(record)
        print(
            f"[{index:02d}/{len(scenes):02d}] {scene} "
            f"fg={refined_mask.mean():.3f} PLY-retained={reference_retained:.3f} "
            f"EPE(all/subject)={all_metrics.get('epe_px', float('nan')):.3f}/"
            f"{subject_metrics.get('epe_px', float('nan')):.3f}",
            flush=True,
        )

    write_csv(metrics_dir / "per_scene.csv", rows)
    (metrics_dir / "hole_decisions.json").write_text(
        json.dumps(hole_decisions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    overview_path = output_root / "overview.jpg"
    save_overview(comparisons, overview_path)
    bisenet_meta = json.loads(bisenet_metadata.read_text(encoding="utf-8"))
    source_las_summary = json.loads(las_summary.read_text(encoding="utf-8"))
    all_macro = metric_macro(rows, "all")
    subject_macro = metric_macro(rows, "subject")
    background_macro = metric_macro(rows, "background")
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Jop1",
        "scene_count": len(rows),
        "image_shape": [rows[0]["height"], rows[0]["width"]],
        "pipeline": (
            "frozen BiSeNetV2 probability -> one-component/disparity-continuity refinement; "
            "LAS1 uses original rectified left/right RGB and is post-masked only"
        ),
        "inputs": {
            "preprocessed_root": str(preprocessed_root),
            "las_root": str(las_root),
            "reference_root": str(reference_root),
            "probability_dir": str(probability_dir),
            "calibration": str(calibration),
        },
        "models": {
            "bisenet_pb": bisenet_meta["model_pb"],
            "bisenet_pb_sha256": bisenet_meta["model_pb_sha256"],
            "las_checkpoint": str(las_checkpoint),
            "las_checkpoint_sha256": sha256_file(las_checkpoint),
            "saved_las_source_summary": str(las_summary),
            "saved_las_source_summary_sha256": sha256_file(las_summary),
            "saved_las_scene_count": source_las_summary["scene_count"],
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
            "totals": {
                "initial_components_after_closing": int(
                    sum(row["initial_components_after_closing"] for row in rows)
                ),
                "removed_island_pixels": int(sum(row["removed_island_pixels"] for row in rows)),
                "holes": int(sum(row["hole_count"] for row in rows)),
                "filled_holes": int(sum(row["filled_hole_count"] for row in rows)),
                "filled_hole_pixels": int(sum(row["filled_hole_pixels"] for row in rows)),
                "preserved_holes": int(sum(row["preserved_hole_count"] for row in rows)),
                "final_masks_with_exactly_one_component": int(
                    sum(row["final_foreground_components"] == 1 for row in rows)
                ),
            },
        },
        "coverage": {
            "raw_foreground_fraction_macro_mean": float(
                np.mean([row["raw_foreground_fraction"] for row in rows])
            ),
            "refined_foreground_fraction_macro_mean": float(
                np.mean([row["refined_foreground_fraction"] for row in rows])
            ),
            "ply_reference_retained_fraction_macro_mean": float(
                np.mean([row["reference_retained_fraction"] for row in rows])
            ),
            "ply_reference_retained_fraction_min": float(
                np.min([row["reference_retained_fraction"] for row in rows])
            ),
            "ply_reference_retained_fraction_max": float(
                np.max([row["reference_retained_fraction"] for row in rows])
            ),
        },
        "ply_reference_metrics": {
            "all": all_macro,
            "subject": subject_macro,
            "background": background_macro,
        },
        "selection_effect_not_model_improvement": {
            "subject_epe_relative_difference_vs_all_percent": float(
                100.0
                * (all_macro["epe_px"]["macro_mean"] - subject_macro["epe_px"]["macro_mean"])
                / all_macro["epe_px"]["macro_mean"]
            ),
            "post_mask_max_abs_change_inside_subject": 0.0,
        },
        "limitations": [
            "Jop1 has no manual workpiece segmentation ground truth, so no segmentation IoU/Dice is reported.",
            "The supplied PLY is sparse and is used only for disparity consistency, not dense ground truth.",
            "All-region and subject-region metrics use different pixel sets and do not measure a LAS1 accuracy gain.",
        ],
        "outputs": {
            "per_scene_csv": str(metrics_dir / "per_scene.csv"),
            "overview": str(overview_path),
            "scenes": str(scenes_dir),
        },
    }
    summary_path = metrics_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
