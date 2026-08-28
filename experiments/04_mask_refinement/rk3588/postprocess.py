"""CPU post-processing used by the Experiment 1-4 RK3588 pipeline.

This is the deployable subset of
``projects/LiteAnyStereo/tools/refine_bisenet_subject_masks.py``.  Keep the
defaults in sync with the frozen Experiment 4 report.
"""

from __future__ import annotations

import cv2
import numpy as np


TRADITION_CROP = (234, 1052, 126, 638)  # y0, y1, x0, x1


def keep_largest_component(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Keep one 8-connected foreground component."""
    mask = np.asarray(mask, dtype=bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return np.zeros_like(mask), max(count - 1, 0), 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    result = labels == largest_label
    return result, count - 1, int(mask.sum() - result.sum())


def enclosed_holes(mask: np.ndarray):
    """Yield enclosed background components that do not touch an image edge."""
    inverse = (~np.asarray(mask, dtype=bool)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverse, connectivity=8
    )
    height, width = mask.shape
    for label in range(1, count):
        x, y, component_width, component_height, area = (
            int(value) for value in stats[label]
        )
        touches_border = (
            x == 0
            or y == 0
            or x + component_width == width
            or y + component_height == height
        )
        if not touches_border:
            yield label, labels == label, area


def hole_disparity_decision(
    hole: np.ndarray,
    foreground: np.ndarray,
    disparity_crop: np.ndarray,
    crop: tuple[int, int, int, int],
    ring_radius: int,
    absolute_tolerance: float,
    mad_scale: float,
    max_fill_area: int,
    small_hole_area: int,
) -> tuple[bool, dict]:
    area = int(hole.sum())
    if area > max_fill_area:
        return False, {"reason": "area_limit", "area": area}

    y0, y1, x0, x1 = crop
    hole_roi = hole[y0:y1, x0:x1]
    if not hole_roi.any():
        fill = area <= small_hole_area
        return fill, {
            "reason": (
                "small_outside_disparity_roi" if fill else "outside_disparity_roi"
            ),
            "area": area,
        }

    kernel_size = 2 * ring_radius + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    ring = cv2.dilate(hole.astype(np.uint8), kernel).astype(bool)
    ring &= foreground
    ring_roi = ring[y0:y1, x0:x1]
    finite = np.isfinite(disparity_crop)
    if finite.shape != hole_roi.shape:
        raise ValueError(
            "disparity_crop shape must match the configured crop: "
            f"{finite.shape} vs {hole_roi.shape}"
        )
    hole_values = disparity_crop[hole_roi & finite]
    ring_values = disparity_crop[ring_roi & finite]
    if hole_values.size < 32 or ring_values.size < 32:
        fill = area <= small_hole_area
        return fill, {
            "reason": (
                "small_without_disparity_support"
                if fill
                else "insufficient_disparity_support"
            ),
            "area": area,
            "hole_disparity_pixels": int(hole_values.size),
            "ring_disparity_pixels": int(ring_values.size),
        }

    hole_median = float(np.median(hole_values))
    ring_median = float(np.median(ring_values))
    ring_mad = float(np.median(np.abs(ring_values - ring_median)))
    median_difference = abs(hole_median - ring_median)
    tolerance = max(absolute_tolerance, mad_scale * ring_mad)
    fill = median_difference <= tolerance
    return fill, {
        "reason": "disparity_continuous" if fill else "disparity_discontinuous",
        "area": area,
        "hole_median_disparity": hole_median,
        "ring_median_disparity": ring_median,
        "ring_mad_disparity": ring_mad,
        "median_disparity_difference": median_difference,
        "allowed_disparity_difference": tolerance,
        "hole_disparity_pixels": int(hole_values.size),
        "ring_disparity_pixels": int(ring_values.size),
    }


def refine_mask(
    probability: np.ndarray,
    full_shape: tuple[int, int],
    disparity_crop: np.ndarray,
    threshold: float = 0.5,
    closing_radius: int = 3,
    hole_ring_radius: int = 7,
    hole_absolute_tolerance: float = 1.5,
    hole_mad_scale: float = 1.0,
    max_fill_hole_fraction: float = 0.025,
    small_hole_area: int = 1000,
    crop: tuple[int, int, int, int] = TRADITION_CROP,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply the frozen Experiment 4 rules to one probability map."""
    full_height, full_width = full_shape
    probability_full = cv2.resize(
        np.asarray(probability, dtype=np.float32),
        (full_width, full_height),
        interpolation=cv2.INTER_LINEAR,
    )
    raw = probability_full >= threshold
    if closing_radius > 0:
        size = 2 * closing_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        closed = cv2.morphologyEx(
            raw.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)
    else:
        closed = raw.copy()

    main, initial_components, removed_island_pixels = keep_largest_component(closed)
    if not main.any():
        raise ValueError("No foreground remains after connected-component filtering")

    decisions = []
    max_fill_area = int(round(max_fill_hole_fraction * full_height * full_width))
    for label, hole, _ in list(enclosed_holes(main)):
        fill, details = hole_disparity_decision(
            hole,
            main,
            disparity_crop,
            crop,
            hole_ring_radius,
            hole_absolute_tolerance,
            hole_mad_scale,
            max_fill_area,
            small_hole_area,
        )
        if fill:
            main[hole] = True
        decisions.append({"label": label, "fill": bool(fill), **details})

    refined, final_components_before_filter, final_removed_pixels = (
        keep_largest_component(main)
    )
    final_count, _, _, _ = cv2.connectedComponentsWithStats(
        refined.astype(np.uint8), connectivity=8
    )
    stats = {
        "raw_foreground_pixels": int(raw.sum()),
        "refined_foreground_pixels": int(refined.sum()),
        "foreground_pixel_change": int(refined.sum() - raw.sum()),
        "initial_components_after_closing": initial_components,
        "removed_island_pixels": removed_island_pixels + final_removed_pixels,
        "hole_count": len(decisions),
        "filled_hole_count": sum(item["fill"] for item in decisions),
        "filled_hole_pixels": sum(
            item["area"] for item in decisions if item["fill"]
        ),
        "preserved_hole_count": sum(not item["fill"] for item in decisions),
        "final_components_before_filter": final_components_before_filter,
        "final_foreground_components": final_count - 1,
        "hole_decisions": decisions,
    }
    return raw, refined, stats


def crop_array(array: np.ndarray, crop=TRADITION_CROP) -> np.ndarray:
    """Crop a full-resolution HxW array with the Experiment 4 ROI."""
    y0, y1, x0, x1 = crop
    if y0 < 0 or x0 < 0 or y1 > array.shape[0] or x1 > array.shape[1]:
        raise ValueError(f"Crop {crop} is outside array shape {array.shape}")
    return array[y0:y1, x0:x1]
