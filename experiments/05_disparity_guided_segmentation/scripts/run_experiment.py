#!/usr/bin/env python3
"""Experiment 5: extract the workpiece with LiteAnyStereo disparity.

V1 soft-fusion constants were selected only on the 64 FDJYP-0 ``train.csv``
rows.  Recall V2 inherits Experiment 4's selective-hole constants and adds a
user-specified no-semantic-deletion policy.  No V2 constant was selected from
the 18 FDJYP-0 ``val.csv`` masks; their numbers are an engineering comparison,
not a newly untouched test after this iterative project history.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/05_disparity_guided_segmentation"
ANNOTATIONS = ROOT / "datasets/annotations/JMP-workpiece-seg-manual-isat-v1"
LAS_OUTPUTS = (
    ROOT
    / "experiments/01_stereo_comparison/rec_img_set/results/final_203/outputs/fdjyp0"
)
CALIBRATION = ROOT / "projects/tradition_stereo/config/stereo_gongjian.yml"
DEFAULT_OUTPUT = EXPERIMENT / "results/fdjyp0_holdout"
DEFAULT_SEMANTIC_MASKS = EXPERIMENT / "results/bisenet_reference/masks"
DEFAULT_SEMANTIC_PROBABILITIES = (
    EXPERIMENT / "results/bisenet_reference/probabilities"
)
DEFAULT_TRAIN_SEMANTIC_PROBABILITIES = (
    EXPERIMENT / "results/bisenet_train_reference/probabilities"
)

# Frozen after inspecting only the 64 FDJYP-0 training masks.
GEOMETRY_CONFIG = {
    "lower_percentile": 10.0,
    "upper_percentile": 90.0,
    "otsu_offset": -30.0,
    "center_penalty": 8.0,
}
REFINEMENT_CONFIG = {
    "probable_foreground_offset": -40.0,
    "sure_foreground_offset": 20.0,
    "sure_background_offset": -45.0,
    "grabcut_iterations": 1,
}
SOFT_FUSION_CONFIG = {
    "sure_foreground_probability": 0.9,
    "sure_background_probability": 0.1,
    "depth_score_clip": 3.0,
    "minimum_depth_scale": 0.05,
    "reliability_offset": 0.05,
    "depth_weight": 1.0,
    "semantic_logit_bias": 0.25,
}
RECALL_PRIORITY_CONFIG = {
    # These hole tests inherit Experiment 4.  Recall priority is enforced by
    # never subtracting semantic foreground and filling ambiguous small holes.
    "closing_radius": 3,
    "hole_ring_radius": 7,
    "small_hole_area": 1000,
    "max_ambiguous_fill_fraction": 0.025,
    "minimum_support_pixels": 32,
    "background_absolute_difference_px": 1.5,
    "background_mad_scale": 1.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--las-outputs", type=Path, default=LAS_OUTPUTS)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--semantic-mask-dir",
        type=Path,
        default=DEFAULT_SEMANTIC_MASKS,
        help=(
            "Fallback BiSeNet mask directory used when probabilities are unavailable."
        ),
    )
    parser.add_argument(
        "--semantic-probability-dir",
        type=Path,
        default=DEFAULT_SEMANTIC_PROBABILITIES,
        help="BiSeNet foreground probabilities for the 18-row holdout split.",
    )
    parser.add_argument(
        "--train-semantic-probability-dir",
        type=Path,
        default=DEFAULT_TRAIN_SEMANTIC_PROBABILITIES,
        help="BiSeNet probabilities used to select the soft-fusion constants.",
    )
    parser.add_argument(
        "--no-pointcloud",
        action="store_true",
        help="Skip subject_cloud.ply generation (masks and disparity are still written).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing result directory.",
    )
    return parser.parse_args()


def scene_from_name(name):
    parts = name.split("_")
    if len(parts) < 5 or parts[0] != "fdjyp" or parts[1] != "0":
        raise ValueError("Unsupported annotation name: {}".format(name))
    return "{}-{}".format(parts[3], parts[4])


def load_rows(annotation_root, las_outputs, split):
    index_path = annotation_root / "index/{}.csv".format(split)
    rows = []
    with index_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row["capture_group"].startswith("fdjyp_0_"):
                continue
            row = dict(row)
            row["split"] = split
            row["scene"] = scene_from_name(row["name"])
            row["image_path"] = annotation_root / row["image"]
            row["mask_path"] = annotation_root / row["mask"]
            row["disparity_path"] = (
                las_outputs / row["scene"] / "liteanystereo/disp_full.npy"
            )
            for key in ("image_path", "mask_path", "disparity_path"):
                if not row[key].is_file():
                    raise FileNotFoundError(row[key])
            rows.append(row)
    if not rows:
        raise RuntimeError("No FDJYP-0 rows found in {}".format(index_path))
    return rows


def normalize_disparity(disparity, shape):
    height, width = shape
    resized = cv2.resize(
        disparity.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_AREA,
    )
    valid = np.isfinite(resized) & (resized > 0)
    if not np.any(valid):
        raise ValueError("Disparity contains no finite positive pixels")
    low, high = np.percentile(
        resized[valid],
        [GEOMETRY_CONFIG["lower_percentile"], GEOMETRY_CONFIG["upper_percentile"]],
    )
    scale = max(float(high - low), 1e-6)
    normalized = np.clip((resized - low) * 255.0 / scale, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(
        normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return normalized, float(otsu), float(low), float(high)


def choose_centered_component(mask, center_penalty=None):
    if center_penalty is None:
        center_penalty = GEOMETRY_CONFIG["center_penalty"]
    height, width = mask.shape
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    if count <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    dx = (centroids[1:, 0] - width / 2.0) / (width / 2.0)
    dy = (centroids[1:, 1] - height / 2.0) / (height / 2.0)
    scores = np.log(areas + 1.0) - float(center_penalty) * (dx * dx + dy * dy)
    selected = 1 + int(np.argmax(scores))
    return labels == selected


def fill_holes(mask):
    inverse = (~mask.astype(bool)).astype(np.uint8)
    _, labels, _, _ = cv2.connectedComponentsWithStats(inverse, 8)
    border_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    holes = (inverse > 0) & (~np.isin(labels, border_labels))
    return mask.astype(bool) | holes


def enclosed_holes(mask):
    """Yield enclosed background components as boolean masks."""
    inverse = (~np.asarray(mask, dtype=bool)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, 8)
    height, width = mask.shape
    for label in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[label])
        touches_border = x == 0 or y == 0 or x + w == width or y + h == height
        if not touches_border:
            yield labels == label, area


def repair_holes_recall_priority(mask, disparity, config=None):
    """Fill ambiguous holes and preserve only strong depth-background holes.

    This is the recall-priority counterpart of Experiment 4's selective hole
    repair.  Existing foreground pixels are never removed by a hole decision.
    A hole remains background only when it has enough valid disparity support
    and its median is separated from the surrounding subject by both a strict
    absolute threshold and a robust-MAD threshold.
    """
    if config is None:
        config = RECALL_PRIORITY_CONFIG
    mask = np.asarray(mask, dtype=bool)
    disparity = np.asarray(disparity, dtype=np.float32)
    if mask.shape != disparity.shape:
        raise ValueError(
            "Mask/disparity shape mismatch: {} vs {}".format(
                mask.shape, disparity.shape
            )
        )
    radius = int(config["closing_radius"])
    if radius > 0:
        size = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        closed = cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)
    else:
        closed = mask.copy()
    result = choose_centered_component(closed)
    preserved_background = np.zeros_like(result)
    decisions = []
    ring_radius = int(config["hole_ring_radius"])
    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * ring_radius + 1, 2 * ring_radius + 1)
    )
    valid = np.isfinite(disparity) & (disparity > 0)
    max_ambiguous_fill_area = int(
        round(float(config["max_ambiguous_fill_fraction"]) * mask.size)
    )
    for hole, area in list(enclosed_holes(result)):
        reason = "ambiguous_fill_for_recall"
        preserve = False
        details = {"area": int(area)}
        if area > max_ambiguous_fill_area:
            preserve = True
            reason = "large_background_hole"
        elif area > int(config["small_hole_area"]):
            ring = cv2.dilate(hole.astype(np.uint8), ring_kernel).astype(bool)
            ring &= result
            hole_values = disparity[hole & valid]
            ring_values = disparity[ring & valid]
            details.update(
                {
                    "hole_disparity_pixels": int(hole_values.size),
                    "ring_disparity_pixels": int(ring_values.size),
                }
            )
            support = int(config["minimum_support_pixels"])
            if hole_values.size >= support and ring_values.size >= support:
                hole_median = float(np.median(hole_values))
                ring_median = float(np.median(ring_values))
                ring_mad = float(np.median(np.abs(ring_values - ring_median)))
                difference = abs(hole_median - ring_median)
                threshold = max(
                    float(config["background_absolute_difference_px"]),
                    float(config["background_mad_scale"]) * ring_mad,
                )
                preserve = difference > threshold
                reason = (
                    "strong_depth_background"
                    if preserve
                    else "depth_ambiguous_fill_for_recall"
                )
                details.update(
                    {
                        "hole_median_disparity": hole_median,
                        "ring_median_disparity": ring_median,
                        "ring_mad_disparity": ring_mad,
                        "median_disparity_difference": difference,
                        "required_difference": threshold,
                    }
                )
            else:
                reason = "insufficient_depth_support_fill_for_recall"
        else:
            reason = "small_hole_fill_for_recall"
        if preserve:
            preserved_background |= hole
        else:
            result[hole] = True
        details.update({"preserve_background": bool(preserve), "reason": reason})
        decisions.append(details)
    diagnostics = {
        "hole_count": len(decisions),
        "filled_hole_count": sum(not item["preserve_background"] for item in decisions),
        "filled_hole_pixels": sum(
            item["area"] for item in decisions if not item["preserve_background"]
        ),
        "preserved_background_hole_count": sum(
            item["preserve_background"] for item in decisions
        ),
        "preserved_background_pixels": int(preserved_background.sum()),
        "decisions": decisions,
    }
    return result, preserved_background, diagnostics


def scaled_recall_priority_config(shape, reference_shape=(1280, 720)):
    """Scale pixel/area morphology constants to a different mask resolution."""
    height, width = shape
    reference_height, reference_width = reference_shape
    linear_scale = np.sqrt(
        float(height * width) / float(reference_height * reference_width)
    )
    area_scale = float(height * width) / float(
        reference_height * reference_width
    )
    config = dict(RECALL_PRIORITY_CONFIG)
    config["closing_radius"] = max(
        1, int(round(config["closing_radius"] * linear_scale))
    )
    config["hole_ring_radius"] = max(
        1, int(round(config["hole_ring_radius"] * linear_scale))
    )
    config["small_hole_area"] = max(
        1, int(round(config["small_hole_area"] * area_scale))
    )
    config["minimum_support_pixels"] = max(
        8, int(round(config["minimum_support_pixels"] * area_scale))
    )
    return config


def combine_soft_fusion_recall_priority(
    semantic_candidate,
    soft_fusion_v1,
    disparity,
    fusion_diagnostics=None,
    config=None,
):
    """Combine V1 soft fusion with selective holes and a no-deletion guard."""
    semantic, semantic_background, semantic_holes = repair_holes_recall_priority(
        semantic_candidate, disparity, config=config
    )
    candidate = choose_centered_component(
        np.asarray(soft_fusion_v1, dtype=bool) | semantic
    )
    # A strong depth-discontinuous hole identified from the semantic contour
    # vetoes V1's unconditional fill.
    candidate[semantic_background] = False
    fused, fused_background, fused_holes = repair_holes_recall_priority(
        candidate, disparity, config=config
    )
    fused |= semantic
    fused[semantic_background | fused_background] = False
    diagnostics = dict(fusion_diagnostics or {})
    diagnostics.update(
        {
            "changed_fraction": float(np.mean(fused != semantic)),
            "added_fraction": float(np.mean(fused & ~semantic)),
            "removed_fraction": float(np.mean(semantic & ~fused)),
            "semantic_hole_count": semantic_holes["hole_count"],
            "semantic_filled_hole_count": semantic_holes["filled_hole_count"],
            "semantic_preserved_background_hole_count": semantic_holes[
                "preserved_background_hole_count"
            ],
            "semantic_preserved_background_pixels": semantic_holes[
                "preserved_background_pixels"
            ],
            "fused_hole_count": fused_holes["hole_count"],
            "fused_filled_hole_count": fused_holes["filled_hole_count"],
            "fused_preserved_background_hole_count": fused_holes[
                "preserved_background_hole_count"
            ],
            "fused_preserved_background_pixels": fused_holes[
                "preserved_background_pixels"
            ],
        }
    )
    if diagnostics["removed_fraction"] != 0.0:
        raise AssertionError("Recall-priority fusion removed semantic foreground")
    return semantic, fused, diagnostics


def soft_fuse_recall_priority(probability, normalized, disparity):
    """Run V1 soft fusion, then add recall guard and selective hole handling."""
    soft_v1, diagnostics = soft_fuse_semantic_and_disparity(
        probability, normalized
    )
    return combine_soft_fusion_recall_priority(
        np.asarray(probability) >= 0.5,
        soft_v1,
        disparity,
        fusion_diagnostics=diagnostics,
        config=scaled_recall_priority_config(np.asarray(probability).shape),
    )


def geometry_from_normalized(normalized, otsu, offset=None):
    if offset is None:
        offset = GEOMETRY_CONFIG["otsu_offset"]
    threshold = float(np.clip(otsu + offset, 0, 255))
    candidate = normalized > threshold
    return fill_holes(choose_centered_component(candidate))


def refine_with_color(image, normalized, otsu, fallback):
    config = REFINEMENT_CONFIG
    probable_threshold = np.clip(
        otsu + config["probable_foreground_offset"], 0, 255
    )
    foreground_threshold = np.clip(
        otsu + config["sure_foreground_offset"], 0, 255
    )
    background_threshold = np.clip(
        otsu + config["sure_background_offset"], 0, 255
    )

    probable = fill_holes(
        choose_centered_component(normalized > probable_threshold)
    )
    sure_foreground = choose_centered_component(
        (normalized > foreground_threshold) & probable
    )
    if not np.any(sure_foreground):
        return fallback.copy()

    grabcut_mask = np.full(image.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
    grabcut_mask[probable] = cv2.GC_PR_FGD
    grabcut_mask[normalized <= background_threshold] = cv2.GC_BGD
    grabcut_mask[sure_foreground] = cv2.GC_FGD
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            image,
            grabcut_mask,
            None,
            background_model,
            foreground_model,
            int(config["grabcut_iterations"]),
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return fallback.copy()
    refined = (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD)
    refined = fill_holes(choose_centered_component(refined))
    return refined if np.any(refined) else fallback.copy()


def soft_fuse_semantic_and_disparity(probability, normalized):
    """Use scene-adaptive depth evidence only around uncertain semantic pixels."""
    config = SOFT_FUSION_CONFIG
    probability = np.asarray(probability, dtype=np.float32)
    depth = normalized.astype(np.float32) / 255.0
    sure_foreground = probability > config["sure_foreground_probability"]
    sure_background = probability < config["sure_background_probability"]
    semantic = fill_holes(choose_centered_component(probability >= 0.5))
    if sure_foreground.sum() < 32 or sure_background.sum() < 32:
        return semantic, {
            "depth_reliability": 0.0,
            "foreground_depth_median": None,
            "background_depth_median": None,
            "changed_fraction": 0.0,
        }

    foreground_values = depth[sure_foreground]
    background_values = depth[sure_background]
    foreground_median = float(np.median(foreground_values))
    background_median = float(np.median(background_values))
    foreground_mad = float(
        1.4826 * np.median(np.abs(foreground_values - foreground_median))
    )
    background_mad = float(
        1.4826 * np.median(np.abs(background_values - background_median))
    )
    separation = abs(foreground_median - background_median)
    reliability = float(
        np.clip(
            separation
            / (
                foreground_mad
                + background_mad
                + config["reliability_offset"]
            ),
            0.0,
            1.0,
        )
    )
    direction = 1.0 if foreground_median >= background_median else -1.0
    midpoint = 0.5 * (foreground_median + background_median)
    scale = max(0.5 * separation, config["minimum_depth_scale"])
    depth_score = direction * (depth - midpoint) / scale
    depth_score = (
        np.clip(
            depth_score,
            -config["depth_score_clip"],
            config["depth_score_clip"],
        )
        * reliability
    )

    clipped_probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
    semantic_logit = np.log(clipped_probability / (1.0 - clipped_probability))
    fused_logit = (
        semantic_logit
        + config["depth_weight"] * depth_score
        + config["semantic_logit_bias"]
    )
    fused = fill_holes(choose_centered_component(fused_logit >= 0.0))
    diagnostics = {
        "depth_reliability": reliability,
        "foreground_depth_median": foreground_median,
        "background_depth_median": background_median,
        "depth_direction": "foreground_higher"
        if direction > 0
        else "foreground_lower",
        "changed_fraction": float(np.mean(fused != semantic)),
    }
    return fused, diagnostics


def boundary_f1(prediction, target, tolerance=2):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pred_u8 = prediction.astype(np.uint8)
    target_u8 = target.astype(np.uint8)
    pred_edge = pred_u8 ^ cv2.erode(pred_u8, kernel)
    target_edge = target_u8 ^ cv2.erode(target_u8, kernel)
    if not np.any(pred_edge) and not np.any(target_edge):
        return 1.0
    radius = 2 * int(tolerance) + 1
    tolerance_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius, radius)
    )
    target_band = cv2.dilate(target_edge, tolerance_kernel) > 0
    pred_band = cv2.dilate(pred_edge, tolerance_kernel) > 0
    pred_count = max(int(pred_edge.sum()), 1)
    target_count = max(int(target_edge.sum()), 1)
    precision = float((pred_edge.astype(bool) & target_band).sum()) / pred_count
    recall = float((target_edge.astype(bool) & pred_band).sum()) / target_count
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def mask_metrics(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    true_positive = int((prediction & target).sum())
    false_positive = int((prediction & ~target).sum())
    false_negative = int((~prediction & target).sum())
    true_negative = int((~prediction & ~target).sum())
    union = true_positive + false_positive + false_negative
    return {
        "iou": true_positive / max(union, 1),
        "dice": 2.0 * true_positive
        / max(2 * true_positive + false_positive + false_negative, 1),
        "precision": true_positive / max(true_positive + false_positive, 1),
        "recall": true_positive / max(true_positive + false_negative, 1),
        "accuracy": (true_positive + true_negative) / prediction.size,
        "boundary_f1_2px": boundary_f1(prediction, target, tolerance=2),
        "predicted_fraction": float(prediction.mean()),
        "target_fraction": float(target.mean()),
        "area_ratio": int(prediction.sum()) / max(int(target.sum()), 1),
    }


def evaluate_row(row):
    image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
    target = cv2.imread(str(row["mask_path"]), cv2.IMREAD_GRAYSCALE)
    disparity = np.load(row["disparity_path"]).astype(np.float32)
    if image is None or target is None:
        raise IOError("Could not read image or mask for {}".format(row["name"]))
    target = target > 0
    if image.shape[:2] != target.shape:
        image = cv2.resize(
            image, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_AREA
        )
    normalized, otsu, low, high = normalize_disparity(disparity, target.shape)
    geometry = geometry_from_normalized(normalized, otsu)
    refined = refine_with_color(image, normalized, otsu, geometry)
    target_values = disparity[
        cv2.resize(
            target.astype(np.uint8),
            (disparity.shape[1], disparity.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        > 0
    ]
    background_values = disparity[
        cv2.resize(
            target.astype(np.uint8),
            (disparity.shape[1], disparity.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        == 0
    ]
    valid_target = target_values[np.isfinite(target_values) & (target_values > 0)]
    valid_background = background_values[
        np.isfinite(background_values) & (background_values > 0)
    ]
    polarity = (
        "foreground_higher"
        if np.median(valid_target) > np.median(valid_background)
        else "foreground_lower"
    )
    evaluation = {
        "image": image,
        "target": target,
        "disparity": disparity,
        "normalized": normalized,
        "otsu": otsu,
        "normalization_low": low,
        "normalization_high": high,
        "geometry": geometry,
        "refined": refined,
        "geometry_metrics": mask_metrics(geometry, target),
        "refined_metrics": mask_metrics(refined, target),
        "polarity": polarity,
    }
    probability_path = row.get("semantic_probability_path")
    semantic_path = row.get("semantic_mask_path")
    probability = None
    if probability_path is not None:
        probability = np.load(probability_path).astype(np.float32)
        if probability.shape != target.shape or not np.isfinite(probability).all():
            raise ValueError("Invalid semantic probability: {}".format(probability_path))
        semantic = fill_holes(choose_centered_component(probability >= 0.5))
    elif semantic_path is not None:
        semantic = cv2.imread(str(semantic_path), cv2.IMREAD_GRAYSCALE)
        if semantic is None:
            raise FileNotFoundError(semantic_path)
        if semantic.shape != target.shape:
            semantic = cv2.resize(
                semantic,
                (target.shape[1], target.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        semantic = fill_holes(choose_centered_component(semantic > 0))
    else:
        semantic = None
    if semantic is not None:
        semantic_depth_intersection = semantic & refined
        update = {
            "semantic": semantic,
            "semantic_depth_intersection": semantic_depth_intersection,
            "semantic_metrics": mask_metrics(semantic, target),
            "semantic_depth_intersection_metrics": mask_metrics(
                semantic_depth_intersection, target
            ),
        }
        if probability is not None:
            soft_fusion, fusion_diagnostics = soft_fuse_semantic_and_disparity(
                probability, normalized
            )
            model_disparity = cv2.resize(
                disparity.astype(np.float32),
                (target.shape[1], target.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            (
                recall_priority_semantic,
                recall_priority_fusion,
                recall_priority_diagnostics,
            ) = soft_fuse_recall_priority(
                probability, normalized, model_disparity
            )
            update.update(
                {
                    "soft_fusion": soft_fusion,
                    "soft_fusion_metrics": mask_metrics(soft_fusion, target),
                    "fusion_diagnostics": fusion_diagnostics,
                    "recall_priority_semantic": recall_priority_semantic,
                    "recall_priority_fusion": recall_priority_fusion,
                    "recall_priority_fusion_metrics": mask_metrics(
                        recall_priority_fusion, target
                    ),
                    "recall_priority_diagnostics": recall_priority_diagnostics,
                }
            )
        evaluation.update(update)
    return evaluation


def colorize_subject_disparity(disparity, mask):
    valid = mask & np.isfinite(disparity) & (disparity > 0)
    output = np.zeros((*disparity.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return output
    low, high = np.percentile(disparity[valid], [1, 99])
    scaled = np.clip((disparity - low) * 255.0 / max(high - low, 1e-6), 0, 255)
    colored = cv2.applyColorMap(scaled.astype(np.uint8), cv2.COLORMAP_TURBO)
    output[valid] = colored[valid]
    return output


def load_q(path):
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise FileNotFoundError(path)
    matrix = storage.getNode("Q").mat()
    storage.release()
    if matrix is None or matrix.shape != (4, 4):
        raise ValueError("Invalid Q matrix in {}".format(path))
    return matrix.astype(np.float32)


def write_binary_ply(path, points, colors):
    vertex_type = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertices = np.empty(len(points), dtype=vertex_type)
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Experiment 5 LiteAnyStereo disparity-guided subject cloud\n"
        "element vertex {}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).format(len(vertices))
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertices.tofile(handle)


def save_subject_cloud(path, disparity, image, mask, q):
    full_mask = cv2.resize(
        mask.astype(np.uint8),
        (disparity.shape[1], disparity.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    full_image = cv2.resize(
        image,
        (disparity.shape[1], disparity.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    filtered = cv2.bilateralFilter(
        disparity.astype(np.float32), d=5, sigmaColor=50, sigmaSpace=50
    )
    points_3d = cv2.reprojectImageTo3D(filtered, q, handleMissingValues=True)
    z = points_3d[..., 2]
    valid = (
        full_mask
        & np.isfinite(filtered)
        & (filtered >= 5.0)
        & (filtered <= 192.0)
        & np.isfinite(points_3d).all(axis=2)
        & (z > 0.0)
        & (z < 200.0)
    )
    points = points_3d[valid].astype(np.float32)
    colors = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)[valid]
    write_binary_ply(path, points, colors)
    return {
        "cloud_points": int(valid.sum()),
        "cloud_coverage_pct": 100.0 * float(valid.mean()),
        "cloud_z_p01": float(np.percentile(points[:, 2], 1)) if len(points) else None,
        "cloud_z_median": float(np.median(points[:, 2])) if len(points) else None,
        "cloud_z_p99": float(np.percentile(points[:, 2], 99)) if len(points) else None,
    }


def label_panel(image, text):
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        panel,
        text,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def overlay(image, mask, color):
    output = image.copy()
    tint = np.zeros_like(output)
    tint[:] = color
    output[mask] = cv2.addWeighted(output, 0.35, tint, 0.65, 0)[mask]
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(output, contours, -1, color, 2)
    return output


def make_comparison(evaluation):
    image = evaluation["image"]
    panels = [
        label_panel(image, "left image"),
        label_panel(overlay(image, evaluation["target"], (0, 0, 255)), "human GT"),
        label_panel(
            overlay(image, evaluation["geometry"], (255, 255, 0)),
            "geometry IoU {:.3f}".format(evaluation["geometry_metrics"]["iou"]),
        ),
        label_panel(
            overlay(image, evaluation["refined"], (0, 255, 0)),
            "refined IoU {:.3f}".format(evaluation["refined_metrics"]["iou"]),
        ),
    ]
    if "semantic" in evaluation:
        panels.append(
            label_panel(
                overlay(image, evaluation["semantic"], (0, 165, 255)),
                "semantic+LAS IoU {:.3f}".format(
                    evaluation["semantic_metrics"]["iou"]
                ),
            )
        )
    if "soft_fusion" in evaluation:
        panels.append(
            label_panel(
                overlay(image, evaluation["soft_fusion"], (255, 0, 255)),
                "soft fusion IoU {:.3f}".format(
                    evaluation["soft_fusion_metrics"]["iou"]
                ),
            )
        )
    if "recall_priority_fusion" in evaluation:
        panels.append(
            label_panel(
                overlay(
                    image, evaluation["recall_priority_fusion"], (255, 80, 80)
                ),
                "recall v2 IoU {:.3f}".format(
                    evaluation["recall_priority_fusion_metrics"]["iou"]
                ),
            )
        )
    return np.hstack(panels)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flatten_metrics(prefix, metrics):
    return {"{}_{}".format(prefix, key): value for key, value in metrics.items()}


def aggregate(rows, prefix):
    keys = (
        "iou",
        "dice",
        "precision",
        "recall",
        "accuracy",
        "boundary_f1_2px",
        "predicted_fraction",
        "target_fraction",
        "area_ratio",
    )
    result = {}
    for key in keys:
        values = np.asarray([row["{}_{}".format(prefix, key)] for row in rows])
        result["macro_{}".format(key)] = float(values.mean())
        result["median_{}".format(key)] = float(np.median(values))
    return result


def evaluate_split(rows, output_root=None, q=None, write_cloud=True):
    metric_rows = []
    gallery = []
    for index, row in enumerate(rows, 1):
        evaluation = evaluate_row(row)
        result = {
            "split": row["split"],
            "capture_group": row["capture_group"],
            "name": row["name"],
            "scene": row["scene"],
            "polarity": evaluation["polarity"],
            "otsu_threshold": evaluation["otsu"],
            "normalization_low": evaluation["normalization_low"],
            "normalization_high": evaluation["normalization_high"],
            **flatten_metrics("geometry", evaluation["geometry_metrics"]),
            **flatten_metrics("refined", evaluation["refined_metrics"]),
        }
        if "semantic" in evaluation:
            result.update(flatten_metrics("semantic", evaluation["semantic_metrics"]))
            result.update(
                flatten_metrics(
                    "semantic_depth_intersection",
                    evaluation["semantic_depth_intersection_metrics"],
                )
            )
        if "soft_fusion" in evaluation:
            result.update(
                flatten_metrics("soft_fusion", evaluation["soft_fusion_metrics"])
            )
            result.update(evaluation["fusion_diagnostics"])
        if "recall_priority_fusion" in evaluation:
            result.update(
                flatten_metrics(
                    "recall_priority_fusion",
                    evaluation["recall_priority_fusion_metrics"],
                )
            )
            result.update(
                {
                    "recall_priority_{}".format(key): value
                    for key, value in evaluation[
                        "recall_priority_diagnostics"
                    ].items()
                    if key != "decisions"
                }
            )

        if output_root is not None:
            scene_root = output_root / "scenes" / row["scene"]
            scene_root.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(scene_root / "mask_geometry.png"),
                evaluation["geometry"].astype(np.uint8) * 255,
            )
            cv2.imwrite(
                str(scene_root / "mask_refined.png"),
                evaluation["refined"].astype(np.uint8) * 255,
            )
            subject_mask = evaluation.get(
                "recall_priority_fusion",
                evaluation.get(
                    "soft_fusion", evaluation.get("semantic", evaluation["refined"])
                ),
            )
            if "recall_priority_fusion" in evaluation:
                result["subject_mask_source"] = "recall_priority_fusion_v2"
            elif "soft_fusion" in evaluation:
                result["subject_mask_source"] = "soft_fusion"
            elif "semantic" in evaluation:
                result["subject_mask_source"] = "semantic"
            else:
                result["subject_mask_source"] = "refined"
            cv2.imwrite(
                str(scene_root / "mask_subject.png"),
                subject_mask.astype(np.uint8) * 255,
            )
            if "semantic" in evaluation:
                cv2.imwrite(
                    str(scene_root / "mask_semantic.png"),
                    evaluation["semantic"].astype(np.uint8) * 255,
                )
                cv2.imwrite(
                    str(scene_root / "mask_semantic_depth_intersection.png"),
                    evaluation["semantic_depth_intersection"].astype(np.uint8) * 255,
                )
            if "soft_fusion" in evaluation:
                cv2.imwrite(
                    str(scene_root / "mask_soft_fusion.png"),
                    evaluation["soft_fusion"].astype(np.uint8) * 255,
                )
            if "recall_priority_fusion" in evaluation:
                cv2.imwrite(
                    str(scene_root / "mask_recall_priority_fusion.png"),
                    evaluation["recall_priority_fusion"].astype(np.uint8) * 255,
                )
                cv2.imwrite(
                    str(scene_root / "mask_recall_priority_semantic.png"),
                    evaluation["recall_priority_semantic"].astype(np.uint8) * 255,
                )
            comparison = make_comparison(evaluation)
            cv2.imwrite(str(scene_root / "comparison.jpg"), comparison)
            gallery.append(
                cv2.resize(comparison, (864, 256), interpolation=cv2.INTER_AREA)
            )

            full_mask = cv2.resize(
                subject_mask.astype(np.uint8),
                (evaluation["disparity"].shape[1], evaluation["disparity"].shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            subject_disparity = evaluation["disparity"].copy()
            subject_disparity[~full_mask] = np.nan
            np.save(scene_root / "subject_disparity.npy", subject_disparity)
            cv2.imwrite(
                str(scene_root / "subject_disparity.png"),
                colorize_subject_disparity(evaluation["disparity"], full_mask),
            )
            if write_cloud:
                cloud_stats = save_subject_cloud(
                    scene_root / "subject_cloud.ply",
                    evaluation["disparity"],
                    evaluation["image"],
                    subject_mask,
                    q,
                )
                result.update(cloud_stats)
        metric_rows.append(result)
        message = "[{}/{}] {} geometry={:.3f}, refined={:.3f}".format(
            index,
            len(rows),
            row["scene"],
            result["geometry_iou"],
            result["refined_iou"],
        )
        if "soft_fusion_iou" in result:
            message += ", soft_fusion={:.3f}".format(result["soft_fusion_iou"])
        if "recall_priority_fusion_iou" in result:
            message += ", recall_v2={:.3f}".format(
                result["recall_priority_fusion_iou"]
            )
        print(message, flush=True)

    if output_root is not None and gallery:
        columns = 2
        rows_of_tiles = []
        blank = np.zeros_like(gallery[0])
        for start in range(0, len(gallery), columns):
            tiles = gallery[start : start + columns]
            rows_of_tiles.append(np.hstack(tiles + [blank] * (columns - len(tiles))))
        cv2.imwrite(str(output_root / "contact_sheet.jpg"), np.vstack(rows_of_tiles))
    return metric_rows


def prepare_output(path, overwrite):
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            "{} already exists; pass --overwrite to replace generated files".format(path)
        )
    if path.exists() and overwrite:
        # Only remove the exact generated result tree, never a broad or unresolved path.
        import shutil

        shutil.rmtree(str(path))
    path.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    train_rows = load_rows(args.annotations, args.las_outputs, "train")
    holdout_rows = load_rows(args.annotations, args.las_outputs, "val")
    if len(train_rows) != 64 or len(holdout_rows) != 18:
        raise RuntimeError(
            "Expected 64 FDJYP-0 train and 18 holdout rows, got {} and {}".format(
                len(train_rows), len(holdout_rows)
            )
        )
    holdout_probability_available = args.semantic_probability_dir.is_dir()
    train_probability_available = args.train_semantic_probability_dir.is_dir()
    semantic_available = holdout_probability_available or args.semantic_mask_dir.is_dir()
    if semantic_available:
        for row in holdout_rows:
            if holdout_probability_available:
                probability_path = args.semantic_probability_dir / "{}.npy".format(
                    row["name"]
                )
                if not probability_path.is_file():
                    raise FileNotFoundError(probability_path)
                row["semantic_probability_path"] = probability_path
            else:
                semantic_path = args.semantic_mask_dir / "{}.png".format(row["name"])
                if not semantic_path.is_file():
                    raise FileNotFoundError(semantic_path)
                row["semantic_mask_path"] = semantic_path
    else:
        print(
            "Semantic masks not found; running geometry-only: {}".format(
                args.semantic_mask_dir
            ),
            flush=True,
        )
    if train_probability_available:
        for row in train_rows:
            probability_path = args.train_semantic_probability_dir / "{}.npy".format(
                row["name"]
            )
            if not probability_path.is_file():
                raise FileNotFoundError(probability_path)
            row["semantic_probability_path"] = probability_path
    prepare_output(args.output, args.overwrite)
    q = None if args.no_pointcloud else load_q(args.calibration)

    print("Evaluating the 64-row parameter-selection split...")
    train_metrics = evaluate_split(train_rows)
    print("Evaluating the frozen 18-row holdout test...")
    holdout_metrics = evaluate_split(
        holdout_rows,
        output_root=args.output,
        q=q,
        write_cloud=not args.no_pointcloud,
    )
    write_csv(args.output / "train_metrics.csv", train_metrics)
    write_csv(args.output / "holdout_metrics.csv", holdout_metrics)

    holdout_summary = {
        "geometry": aggregate(holdout_metrics, "geometry"),
        "refined": aggregate(holdout_metrics, "refined"),
        "polarity": {
            "foreground_higher": sum(
                row["polarity"] == "foreground_higher" for row in holdout_metrics
            ),
            "foreground_lower": sum(
                row["polarity"] == "foreground_lower" for row in holdout_metrics
            ),
        },
    }
    if semantic_available:
        holdout_summary["semantic_reference"] = aggregate(
            holdout_metrics, "semantic"
        )
        holdout_summary["semantic_depth_intersection"] = aggregate(
            holdout_metrics, "semantic_depth_intersection"
        )
    if holdout_probability_available:
        holdout_summary["soft_fusion"] = aggregate(
            holdout_metrics, "soft_fusion"
        )
        holdout_summary["recall_priority_fusion_v2"] = aggregate(
            holdout_metrics, "recall_priority_fusion"
        )

    train_summary = {
        "geometry": aggregate(train_metrics, "geometry"),
        "refined": aggregate(train_metrics, "refined"),
        "polarity": {
            "foreground_higher": sum(
                row["polarity"] == "foreground_higher" for row in train_metrics
            ),
            "foreground_lower": sum(
                row["polarity"] == "foreground_lower" for row in train_metrics
            ),
        },
    }
    if train_probability_available:
        train_summary["semantic_reference"] = aggregate(train_metrics, "semantic")
        train_summary["soft_fusion"] = aggregate(train_metrics, "soft_fusion")
        train_summary["recall_priority_fusion_v2"] = aggregate(
            train_metrics, "recall_priority_fusion"
        )

    summary = {
        "experiment": "05_disparity_guided_segmentation",
        "purpose": "geometry-guided workpiece extraction for clean disparity and point cloud",
        "semantic_claim": (
            "The geometry method extracts a foreground subject; it does not assign a semantic "
            "class. The refined method adds image-color boundary fitting but no learned semantics. "
            "BiSeNet supplies the workpiece class; soft fusion lets reliable LAS depth modify only "
            "the uncertain semantic boundary."
        ),
        "split_policy": {
            "parameter_selection": "64 FDJYP-0 rows from annotation index/train.csv",
            "frozen_holdout_test": (
                "18 FDJYP-0 rows from annotation index/val.csv (capture group fdjyp_0_2)"
            ),
            "leakage_control": (
                "No holdout mask was used to change a geometry or soft-fusion parameter."
            ),
            "semantic_reference_warning": (
                "The BiSeNet checkpoint was historically selected with this dataset's val.csv. "
                "Its score is an engineering reference, not an independent model test."
            ),
        },
        "counts": {"train": len(train_rows), "holdout": len(holdout_rows)},
        "geometry_config": GEOMETRY_CONFIG,
        "refinement_config": REFINEMENT_CONFIG,
        "soft_fusion_config": SOFT_FUSION_CONFIG,
        "recall_priority_config": RECALL_PRIORITY_CONFIG,
        "train": train_summary,
        "holdout": holdout_summary,
        "inputs": {
            "annotations": str(args.annotations.resolve()),
            "las_outputs": str(args.las_outputs.resolve()),
            "calibration": str(args.calibration.resolve()),
            "semantic_masks": (
                str(args.semantic_mask_dir.resolve()) if semantic_available else None
            ),
            "holdout_semantic_probabilities": (
                str(args.semantic_probability_dir.resolve())
                if holdout_probability_available
                else None
            ),
            "train_semantic_probabilities": (
                str(args.train_semantic_probability_dir.resolve())
                if train_probability_available
                else None
            ),
        },
        "outputs": {
            "holdout_metrics": str((args.output / "holdout_metrics.csv").resolve()),
            "train_metrics": str((args.output / "train_metrics.csv").resolve()),
            "contact_sheet": str((args.output / "contact_sheet.jpg").resolve()),
            "per_scene": str((args.output / "scenes").resolve()),
            "pointcloud_generated": not args.no_pointcloud,
            "subject_mask_source": (
                "BiSeNet + LiteAnyStereo recall-priority soft fusion v2"
                if holdout_probability_available
                else (
                    "BiSeNet semantic mask"
                    if semantic_available
                    else "disparity-guided refinement"
                )
            ),
        },
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary["holdout"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
