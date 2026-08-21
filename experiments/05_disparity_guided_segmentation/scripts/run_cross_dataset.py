#!/usr/bin/env python3
"""Apply Experiment-5 recall-priority fusion V2 to non-FDJYP0 datasets.

This script does not run either neural network again.  It reuses the saved
BiSeNet foreground probabilities and LiteAnyStereo float disparities.  V2
keeps every pixel from its selectively repaired semantic subject, lets soft
fusion add uncertain boundary pixels, and preserves an enclosed background
hole only with strong disparity evidence.  Foundation Stereo and the Jop1 PLY
projection are consistency references, never segmentation truth.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/05_disparity_guided_segmentation"
OUTPUT_DEFAULT = EXPERIMENT / "results/cross_dataset_130_recall_v2"
REC_RESULTS = (
    ROOT
    / "experiments/01_stereo_comparison/rec_img_set/results/final_203"
)
REC_INDEX = REC_RESULTS / "metrics/per_scene.csv"
REC_DATA = ROOT / "datasets/rec_img_set"
REC_PROBABILITIES = (
    ROOT
    / "experiments/03_manual_segmentation/rec_img_set/results/result_130"
    / "bisenet_raw/probabilities"
)
FDJYP3_PROBABILITIES = (
    ROOT
    / "experiments/03_manual_segmentation/fdjyp3/results/model_manual"
    / "fdjyp_3_predictions_probability_v2/probabilities"
)
FDJYP3_REFERENCE = ROOT / "datasets/tradition_raw/FDJYP-3"
JOP1_RESULTS = ROOT / "experiments/01_stereo_comparison/jop1/results/final_9"
JOP1_PROBABILITIES = (
    ROOT
    / "experiments/03_manual_segmentation/jop1/results/result"
    / "bisenet_raw/probabilities"
)
FUSION_SCRIPT = Path(__file__).with_name("run_experiment.py")
REC_GROUPS = ("fdjyp3", "luowen", "general_1221", "scale_1221")
FDJYP3_CROP = (234, 1052, 126, 638)  # y0, y1, x0, x1


def load_fusion_module():
    spec = importlib.util.spec_from_file_location("experiment5_fusion", FUSION_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load {}".format(FUSION_SCRIPT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FUSION = load_fusion_module()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument(
        "--no-clean-disparity",
        action="store_true",
        help="Skip compressed subject_disparity.npz files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the exact output directory if it already exists.",
    )
    return parser.parse_args()


def fdjyp3_probability_name(scene):
    return "fdjyp_3_1_{}.npy".format(scene.replace("-", "_"))


def discover_rec_samples():
    samples = []
    with REC_INDEX.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            group = row["group"]
            if group not in REC_GROUPS:
                continue
            scene = row["scene"]
            probability = (
                FDJYP3_PROBABILITIES / fdjyp3_probability_name(scene)
                if group == "fdjyp3"
                else REC_PROBABILITIES / "{}__{}.npy".format(group, scene)
            )
            reference = (
                FDJYP3_REFERENCE / scene / "disp_cropped.npy"
                if group == "fdjyp3"
                else None
            )
            samples.append(
                {
                    "dataset": group,
                    "scene": scene,
                    "image": REC_DATA / row["source_dir"] / scene / "im0.png",
                    "probability": probability,
                    "disparity": (
                        REC_RESULTS
                        / "outputs"
                        / group
                        / scene
                        / "liteanystereo/disp_full.npy"
                    ),
                    "reference": reference,
                    "reference_kind": (
                        "foundation_stereo" if reference is not None else "none"
                    ),
                }
            )
    return samples


def discover_jop1_samples():
    samples = []
    for image in sorted((JOP1_RESULTS / "preprocessed").glob("*/left.png")):
        scene = image.parent.name
        samples.append(
            {
                "dataset": "jop1",
                "scene": scene,
                "image": image,
                "probability": JOP1_PROBABILITIES / "{}.npy".format(scene),
                "disparity": JOP1_RESULTS / "liteanystereo" / scene / "disp.npy",
                "reference": JOP1_RESULTS / "reference" / scene / "disp.npy",
                "reference_kind": "sparse_ply_projection",
            }
        )
    return samples


def discover_samples():
    samples = discover_rec_samples() + discover_jop1_samples()
    if len(samples) != 130:
        raise RuntimeError("Expected 130 non-FDJYP0 scenes, found {}".format(len(samples)))
    for sample in samples:
        for key in ("image", "probability", "disparity"):
            if not sample[key].is_file():
                raise FileNotFoundError(sample[key])
        if sample["reference"] is not None and not sample["reference"].is_file():
            raise FileNotFoundError(sample["reference"])
    return samples


def resize_mask(mask, shape):
    height, width = shape
    return cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def semantic_diagnostics(probability, semantic, fused, fusion_diagnostics):
    confidence = np.maximum(probability, 1.0 - probability)
    border_width = max(2, int(round(min(fused.shape) * 0.01)))
    border = np.zeros(fused.shape, dtype=bool)
    border[:border_width] = True
    border[-border_width:] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True
    side_touches = sum(
        bool(np.any(side))
        for side in (fused[0], fused[-1], fused[:, 0], fused[:, -1])
    )
    foreground_pixels = int(fused.sum())
    return {
        "semantic_mean_confidence": float(confidence.mean()),
        "semantic_uncertain_fraction": float(
            np.mean((probability >= 0.25) & (probability <= 0.75))
        ),
        "semantic_foreground_fraction": float(semantic.mean()),
        "fused_foreground_fraction": float(fused.mean()),
        "fused_to_semantic_area_ratio": float(foreground_pixels)
        / max(int(semantic.sum()), 1),
        "changed_fraction": float(fusion_diagnostics["changed_fraction"]),
        "depth_reliability": float(fusion_diagnostics["depth_reliability"]),
        "foreground_depth_median": fusion_diagnostics.get(
            "foreground_depth_median"
        ),
        "background_depth_median": fusion_diagnostics.get(
            "background_depth_median"
        ),
        "depth_direction": fusion_diagnostics.get("depth_direction", "unavailable"),
        "border_touch_sides": int(side_touches),
        "border_foreground_fraction": float((fused & border).sum())
        / max(int(border.sum()), 1),
    }


def classify_review_risk(metrics):
    high = []
    review = []
    area = metrics["fused_foreground_fraction"]
    uncertain = metrics["semantic_uncertain_fraction"]
    changed = metrics["changed_fraction"]
    if area >= 0.92:
        high.append("foreground>=92%")
    elif area >= 0.80:
        review.append("foreground>=80%")
    if area <= 0.02:
        high.append("foreground<=2%")
    elif area <= 0.05:
        review.append("foreground<=5%")
    if uncertain >= 0.20:
        high.append("semantic_uncertainty>=20%")
    elif uncertain >= 0.10:
        review.append("semantic_uncertainty>=10%")
    if changed >= 0.15:
        high.append("fusion_changed>=15%")
    elif changed >= 0.05:
        review.append("fusion_changed>=5%")
    if metrics["border_touch_sides"] >= 3 and area >= 0.50:
        review.append("large_mask_touches_3+_sides")
    if metrics["depth_reliability"] < 0.10:
        review.append("weak_depth_evidence")
    reasons = high + review
    if high:
        status = "high_flag"
    elif review:
        status = "review"
    else:
        status = "low_flag"
    return status, ";".join(reasons)


def compute_reference_metrics(prediction, reference, region=None):
    prediction = np.asarray(prediction, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if prediction.shape != reference.shape:
        raise ValueError(
            "Prediction/reference mismatch: {} vs {}".format(
                prediction.shape, reference.shape
            )
        )
    all_reference = np.isfinite(reference) & (reference > 0)
    selected_reference = all_reference.copy()
    if region is not None:
        if region.shape != reference.shape:
            raise ValueError(
                "Region/reference mismatch: {} vs {}".format(
                    region.shape, reference.shape
                )
            )
        selected_reference &= region.astype(bool)
    prediction_valid = np.isfinite(prediction) & (prediction > 0)
    evaluated = selected_reference & prediction_valid
    selected_pixels = int(selected_reference.sum())
    total_pixels = int(all_reference.sum())
    evaluated_pixels = int(evaluated.sum())
    result = {
        "reference_pixels": selected_pixels,
        "reference_retained_pct": 100.0 * selected_pixels / max(total_pixels, 1),
        "prediction_coverage_pct": 100.0 * evaluated_pixels / max(selected_pixels, 1),
        "epe_px": None,
        "bad3_pct": None,
        "d1_pct": None,
    }
    if not evaluated_pixels:
        return result
    error = np.abs(prediction[evaluated] - reference[evaluated])
    relative = error / np.maximum(np.abs(reference[evaluated]), 1e-6)
    result.update(
        {
            "epe_px": float(error.mean()),
            "bad3_pct": float(np.mean(error > 3.0) * 100.0),
            "d1_pct": float(np.mean((error > 3.0) & (relative > 0.05)) * 100.0),
        }
    )
    return result


def reference_views(sample, disparity, semantic_full, fused_full):
    if sample["reference"] is None:
        return None
    reference = np.load(sample["reference"]).astype(np.float32, copy=False)
    if sample["dataset"] == "fdjyp3":
        y0, y1, x0, x1 = FDJYP3_CROP
        prediction = disparity[y0:y1, x0:x1]
        semantic_region = semantic_full[y0:y1, x0:x1]
        fused_region = fused_full[y0:y1, x0:x1]
    else:
        prediction = disparity
        semantic_region = semantic_full
        fused_region = fused_full
    return {
        "all": compute_reference_metrics(prediction, reference),
        "semantic": compute_reference_metrics(prediction, reference, semantic_region),
        "fused": compute_reference_metrics(prediction, reference, fused_region),
    }


def colorize_disparity(disparity, mask=None):
    disparity = np.asarray(disparity, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > 0)
    if mask is not None:
        valid &= mask.astype(bool)
    output = np.zeros(disparity.shape + (3,), dtype=np.uint8)
    if not np.any(valid):
        return output
    low, high = np.percentile(disparity[valid], [1, 99])
    scaled = np.clip(
        (disparity - float(low)) * 255.0 / max(float(high - low), 1e-6),
        0,
        255,
    ).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    output[valid] = colored[valid]
    return output


def label_panel(image, text):
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        output,
        text,
        (7, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def overlay(image, mask, color):
    output = image.copy()
    tint = np.empty_like(output)
    tint[:] = color
    output[mask] = cv2.addWeighted(output, 0.35, tint, 0.65, 0)[mask]
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(output, contours, -1, color, 2)
    return output


def make_comparison(
    image, probability, semantic, soft_v1, fused, disparity, status
):
    shape = probability.shape
    model_image = cv2.resize(
        image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA
    )
    model_disparity = cv2.resize(
        disparity.astype(np.float32),
        (shape[1], shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    panels = [
        label_panel(model_image, "left image"),
        label_panel(
            overlay(model_image, semantic, (0, 165, 255)),
            "selective-hole semantic",
        ),
        label_panel(
            overlay(model_image, soft_v1, (180, 80, 255)), "soft fusion v1"
        ),
        label_panel(
            overlay(model_image, fused, (255, 0, 255)),
            "recall v2: {}".format(status),
        ),
        label_panel(colorize_disparity(model_disparity), "LiteAnyStereo disparity"),
        label_panel(colorize_disparity(model_disparity, fused), "clean subject disparity"),
    ]
    return np.hstack(panels)


def make_contact_sheet(paths, output, columns=4):
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    if any(image is None for image in images):
        raise IOError("Could not read a comparison image")
    tile_width = 480
    image_height = int(round(images[0].shape[0] * tile_width / images[0].shape[1]))
    title_height = 24
    tile_height = image_height + title_height
    tiles = []
    for path, image in zip(paths, images):
        resized = cv2.resize(
            image, (tile_width, image_height), interpolation=cv2.INTER_AREA
        )
        tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
        tile[title_height:] = resized
        title = "{} / {}".format(path.parent.parent.name, path.parent.name)
        cv2.putText(
            tile,
            title,
            (7, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        sheet[
            row * tile_height : (row + 1) * tile_height,
            column * tile_width : (column + 1) * tile_width,
        ] = tile
    if not cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise IOError("Could not write {}".format(output))


def flatten_reference(row, prefix, metrics):
    for key, value in metrics.items():
        row["{}_{}".format(prefix, key)] = value


def process_sample(sample, output, save_clean_disparity=True):
    image = cv2.imread(str(sample["image"]), cv2.IMREAD_COLOR)
    if image is None:
        raise IOError("Could not read {}".format(sample["image"]))
    probability = np.load(sample["probability"]).astype(np.float32, copy=False)
    disparity = np.load(sample["disparity"]).astype(np.float32, copy=False)
    if probability.ndim != 2 or not np.isfinite(probability).all():
        raise ValueError("Invalid probability {}".format(sample["probability"]))
    if disparity.ndim != 2:
        raise ValueError("Invalid disparity {}".format(sample["disparity"]))
    normalized, otsu, low, high = FUSION.normalize_disparity(
        disparity, probability.shape
    )
    soft_v1, fusion_diagnostics = FUSION.soft_fuse_semantic_and_disparity(
        probability, normalized
    )
    probability_full = cv2.resize(
        probability,
        (disparity.shape[1], disparity.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    soft_v1_full = resize_mask(soft_v1, disparity.shape)
    semantic_full, fused_full, recall_diagnostics = (
        FUSION.combine_soft_fusion_recall_priority(
            probability_full >= 0.5,
            soft_v1_full,
            disparity,
            fusion_diagnostics=fusion_diagnostics,
        )
    )
    semantic = resize_mask(semantic_full, probability.shape)
    fused = resize_mask(fused_full, probability.shape)
    recall_diagnostics["changed_from_v1_fraction"] = float(
        np.mean(fused_full != soft_v1_full)
    )
    recall_diagnostics["full_semantic_foreground_fraction"] = float(
        semantic_full.mean()
    )
    recall_diagnostics["full_fused_foreground_fraction"] = float(fused_full.mean())
    diagnostics = semantic_diagnostics(
        probability, semantic, fused, recall_diagnostics
    )
    diagnostics.update(
        {
            "v1_foreground_fraction": float(soft_v1_full.mean()),
            "changed_from_v1_fraction": recall_diagnostics[
                "changed_from_v1_fraction"
            ],
            "added_fraction": recall_diagnostics["added_fraction"],
            "removed_fraction": recall_diagnostics["removed_fraction"],
            "semantic_preserved_background_hole_count": recall_diagnostics[
                "semantic_preserved_background_hole_count"
            ],
            "semantic_preserved_background_pixels": recall_diagnostics[
                "semantic_preserved_background_pixels"
            ],
            "fused_preserved_background_hole_count": recall_diagnostics[
                "fused_preserved_background_hole_count"
            ],
            "fused_preserved_background_pixels": recall_diagnostics[
                "fused_preserved_background_pixels"
            ],
        }
    )
    status, reasons = classify_review_risk(diagnostics)
    valid_disparity = np.isfinite(disparity) & (disparity > 0)
    subject_valid = fused_full & valid_disparity
    diagnostics["subject_valid_disparity_pct"] = (
        100.0 * float(subject_valid.sum()) / max(int(fused_full.sum()), 1)
    )

    scene_output = output / "predictions" / sample["dataset"] / sample["scene"]
    scene_output.mkdir(parents=True, exist_ok=False)
    cv2.imwrite(str(scene_output / "mask_semantic.png"), semantic_full.astype(np.uint8) * 255)
    cv2.imwrite(
        str(scene_output / "mask_soft_fusion_v1.png"),
        soft_v1_full.astype(np.uint8) * 255,
    )
    cv2.imwrite(str(scene_output / "mask_subject.png"), fused_full.astype(np.uint8) * 255)
    cv2.imwrite(
        str(scene_output / "subject_disparity.png"),
        colorize_disparity(disparity, fused_full),
    )
    if save_clean_disparity:
        clean = np.where(subject_valid, disparity, np.nan).astype(np.float32)
        np.savez_compressed(scene_output / "subject_disparity.npz", disparity=clean)
    comparison = make_comparison(
        image, probability, semantic, soft_v1, fused, disparity, status
    )
    comparison_path = scene_output / "comparison.jpg"
    cv2.imwrite(
        str(comparison_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 91]
    )

    row = {
        "dataset": sample["dataset"],
        "scene": sample["scene"],
        "review_status": status,
        "review_reasons": reasons,
        "reference_kind": sample["reference_kind"],
        "fusion_version": "recall_v2",
        "model_height": int(probability.shape[0]),
        "model_width": int(probability.shape[1]),
        "output_height": int(disparity.shape[0]),
        "output_width": int(disparity.shape[1]),
        "normalization_low": low,
        "normalization_high": high,
        "normalization_otsu": otsu,
    }
    row.update(diagnostics)
    references = reference_views(
        sample, disparity, semantic_full, fused_full
    )
    if references is not None:
        for region, metrics in references.items():
            flatten_reference(row, "reference_{}".format(region), metrics)
    return row, comparison_path


def mean_present(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return float(np.mean(values)) if values else None


def aggregate_group(rows):
    result = {
        "scene_count": len(rows),
        "review_status_counts": dict(Counter(row["review_status"] for row in rows)),
    }
    for key in (
        "semantic_mean_confidence",
        "semantic_uncertain_fraction",
        "semantic_foreground_fraction",
        "fused_foreground_fraction",
        "changed_fraction",
        "depth_reliability",
        "subject_valid_disparity_pct",
        "v1_foreground_fraction",
        "changed_from_v1_fraction",
        "added_fraction",
        "removed_fraction",
        "semantic_preserved_background_hole_count",
        "semantic_preserved_background_pixels",
    ):
        result["mean_{}".format(key)] = mean_present(rows, key)
    reference_keys = (
        "reference_all_epe_px",
        "reference_all_d1_pct",
        "reference_semantic_epe_px",
        "reference_semantic_d1_pct",
        "reference_semantic_reference_retained_pct",
        "reference_fused_epe_px",
        "reference_fused_d1_pct",
        "reference_fused_reference_retained_pct",
    )
    for key in reference_keys:
        value = mean_present(rows, key)
        if value is not None:
            result["macro_mean_{}".format(key)] = value
    return result


def write_csv(path, rows):
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_output(output, overwrite):
    output = output.expanduser().resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                "Output exists; use --overwrite to replace it: {}".format(output)
            )
        allowed_parent = (EXPERIMENT / "results").resolve()
        if output.parent != allowed_parent or output == allowed_parent:
            raise ValueError(
                "Refusing to replace output outside {}: {}".format(
                    allowed_parent, output
                )
            )
        shutil.rmtree(str(output))
    output.mkdir(parents=True)
    return output


def main():
    args = parse_args()
    output = prepare_output(args.output, args.overwrite)
    samples = discover_samples()
    rows = []
    comparisons = defaultdict(list)
    for index, sample in enumerate(samples, start=1):
        row, comparison = process_sample(
            sample, output, save_clean_disparity=not args.no_clean_disparity
        )
        rows.append(row)
        comparisons[sample["dataset"]].append(comparison)
        print(
            "[{}/{}] {} / {}: {}".format(
                index,
                len(samples),
                sample["dataset"],
                sample["scene"],
                row["review_status"],
            ),
            flush=True,
        )

    metrics_dir = output / "metrics"
    metrics_dir.mkdir()
    write_csv(metrics_dir / "per_scene.csv", rows)
    for dataset, paths in comparisons.items():
        make_contact_sheet(paths, output / "{}_contact_sheet.jpg".format(dataset))

    groups = {
        dataset: aggregate_group([row for row in rows if row["dataset"] == dataset])
        for dataset in sorted(comparisons)
    }
    summary = {
        "experiment": "Experiment 5 recall-priority soft fusion V2",
        "scope": "All 130 available non-FDJYP0 scenes with cached probabilities and disparities",
        "scene_count": len(rows),
        "dataset_counts": dict(Counter(row["dataset"] for row in rows)),
        "review_status_counts": dict(Counter(row["review_status"] for row in rows)),
        "frozen_soft_fusion_config": dict(FUSION.SOFT_FUSION_CONFIG),
        "recall_priority_config": dict(FUSION.RECALL_PRIORITY_CONFIG),
        "recall_policy": (
            "V1 soft fusion may add pixels but cannot remove selectively repaired "
            "semantic foreground; ambiguous holes are filled and only strong "
            "depth-discontinuous holes remain background."
        ),
        "outputs": {
            "mask_resolution": "full LiteAnyStereo disparity resolution",
            "clean_disparity": "compressed float32 NPZ; background NaN"
            if not args.no_clean_disparity
            else "not exported",
            "pointcloud": "not duplicated; deterministically reconstruct from source disparity and mask_subject.png",
        },
        "groups": groups,
        "interpretation": {
            "segmentation_ground_truth": "none for these 130 scenes",
            "fdjyp3_reference": "Foundation Stereo disparity; engineering consistency only",
            "jop1_reference": "Sparse supplied-PLY projection; engineering consistency only",
            "risk_flags": "Automatic triage for visual review, not accuracy labels",
        },
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print("Wrote {}".format(output))


if __name__ == "__main__":
    main()
