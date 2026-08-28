"""Self-contained Experiment 4 + 7.1 + 7.2 post-processing."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


RECALL_PRIORITY_CONFIG = {
    "closing_radius": 3,
    "hole_ring_radius": 7,
    "small_hole_area": 1000,
    "max_ambiguous_fill_fraction": 0.025,
    "minimum_support_pixels": 32,
    "background_absolute_difference_px": 1.5,
    "background_mad_scale": 1.0,
}


def scaled_recall_priority_config(
    shape: tuple[int, int], reference_shape: tuple[int, int] = (1280, 720)
) -> dict:
    height, width = shape
    reference_height, reference_width = reference_shape
    area_scale = float(height * width) / float(reference_height * reference_width)
    linear_scale = np.sqrt(area_scale)
    config = dict(RECALL_PRIORITY_CONFIG)
    config["closing_radius"] = max(1, int(round(config["closing_radius"] * linear_scale)))
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
    probability: np.ndarray, disparity: np.ndarray, threshold: float
) -> tuple[np.ndarray, dict]:
    config = scaled_recall_priority_config(probability.shape)
    raw = np.asarray(probability, dtype=np.float32) >= threshold
    radius = int(config["closing_radius"])
    size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    closed = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    subject = largest_component(closed)
    max_fill_area = int(round(config["max_ambiguous_fill_fraction"] * subject.size))
    valid = np.isfinite(disparity) & (disparity > 0)
    ring_radius = int(config["hole_ring_radius"])
    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * ring_radius + 1, 2 * ring_radius + 1)
    )
    decisions = []
    for hole, area in list(enclosed_holes(subject)):
        fill = False
        reason = "area_limit"
        if area <= max_fill_area:
            ring = cv2.dilate(hole.astype(np.uint8), ring_kernel).astype(bool) & subject
            hole_values = disparity[hole & valid]
            ring_values = disparity[ring & valid]
            support = int(config["minimum_support_pixels"])
            if hole_values.size >= support and ring_values.size >= support:
                hole_median = float(np.median(hole_values))
                ring_median = float(np.median(ring_values))
                ring_mad = float(np.median(np.abs(ring_values - ring_median)))
                tolerance = max(
                    float(config["background_absolute_difference_px"]),
                    float(config["background_mad_scale"]) * ring_mad,
                )
                fill = abs(hole_median - ring_median) <= tolerance
                reason = "disparity_continuous" if fill else "disparity_discontinuous"
            elif area <= int(config["small_hole_area"]):
                fill = True
                reason = "small_without_disparity_support"
            else:
                reason = "insufficient_disparity_support"
        if fill:
            subject[hole] = True
        decisions.append({"area": int(area), "fill": bool(fill), "reason": reason})
    refined = largest_component(subject)
    return refined, {
        "changed_fraction": float(np.mean(refined != raw)),
        "hole_count": len(decisions),
        "filled_hole_count": sum(item["fill"] for item in decisions),
        "decisions": decisions,
    }


def geometric_refine(
    mask: np.ndarray,
    gaussian_sigma: float = 3.0,
    binary_threshold: float = 0.60,
    closing_radius: int = 6,
    preserve_hole_area: int = 256,
) -> np.ndarray:
    field = cv2.GaussianBlur(np.asarray(mask, dtype=np.float32), (0, 0), gaussian_sigma)
    solid = field >= binary_threshold
    if closing_radius > 0:
        size = 2 * closing_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        solid = cv2.morphologyEx(solid.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    contours, _ = cv2.findContours(
        solid.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros_like(solid, dtype=np.uint8)
    cv2.drawContours(filled, contours, -1, 1, thickness=-1)
    refined = largest_component(filled > 0)
    for hole, area in enclosed_holes(np.asarray(mask, dtype=bool)):
        if area > preserve_hole_area:
            refined[hole] = False
    return refined


@dataclass(frozen=True)
class Component:
    mask: np.ndarray
    area_fraction: float
    bbox_fraction: float


def largest_component_description(mask: np.ndarray) -> Component:
    foreground = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    empty = np.zeros_like(foreground, dtype=bool)
    if count <= 1:
        return Component(empty, 0.0, 0.0)
    selected = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    _, _, width, height, area = (int(value) for value in stats[selected])
    pixels = float(foreground.size)
    return Component(labels == selected, area / pixels, width * height / pixels)


def solidify(mask: np.ndarray, closing_radius: int) -> np.ndarray:
    solid = np.asarray(mask, dtype=np.uint8)
    if closing_radius > 0:
        size = 2 * closing_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(solid)
    cv2.drawContours(filled, contours, -1, 1, thickness=-1)
    return largest_component(filled)


def span_envelope(mask: np.ndarray, axis: int) -> np.ndarray:
    """Fill foreground spans along rows (axis=1) or columns (axis=0)."""
    value = np.asarray(mask, dtype=bool)
    output = value.copy()
    if axis == 1:
        for row in range(value.shape[0]):
            indices = np.flatnonzero(value[row])
            if indices.size > 1:
                output[row, indices[0] : indices[-1] + 1] = True
    elif axis == 0:
        for column in range(value.shape[1]):
            indices = np.flatnonzero(value[:, column])
            if indices.size > 1:
                output[indices[0] : indices[-1] + 1, column] = True
    else:
        raise ValueError(f"axis must be 0 or 1, got {axis}")
    return output


def topology_repair(
    mask: np.ndarray,
    smooth_sigma: float = 2.0,
    smooth_threshold: float = 0.60,
    envelope_min_added_fraction: float = 0.10,
    envelope_max_added_fraction: float = 0.30,
    envelope_closing_radius: int = 7,
    reference_shape: tuple[int, int] = (512, 288),
) -> tuple[np.ndarray, dict]:
    """Make the foreground solid while preserving ordinary concave edges.

    ``solidify`` fills only the external contour, so all enclosed false holes
    disappear.  The intersection of horizontal and vertical span envelopes is
    an orthogonally convex shape prior.  It is used only when its added area
    indicates a substantial but still recoverable break in the prediction.
    """
    original = largest_component(np.asarray(mask, dtype=bool))
    before_holes = list(enclosed_holes(original))
    reference_height, reference_width = reference_shape
    linear_scale = np.sqrt(
        float(original.shape[0] * original.shape[1])
        / float(reference_height * reference_width)
    )
    effective_sigma = max(float(smooth_sigma) * linear_scale, 0.01)
    effective_closing_radius = max(
        1, int(round(float(envelope_closing_radius) * linear_scale))
    )
    field = cv2.GaussianBlur(
        original.astype(np.float32), (0, 0), effective_sigma
    )
    smoothed = solidify(field >= float(smooth_threshold), closing_radius=0)
    horizontal = span_envelope(original, axis=1)
    vertical = span_envelope(original, axis=0)
    envelope = solidify(horizontal & vertical, closing_radius=0)
    added = envelope & ~smoothed
    added_fraction = float(added.sum()) / max(float(smoothed.sum()), 1.0)
    use_envelope = (
        float(envelope_min_added_fraction)
        <= added_fraction
        <= float(envelope_max_added_fraction)
    )
    repaired = solidify(
        envelope if use_envelope else smoothed,
        closing_radius=effective_closing_radius if use_envelope else 0,
    )
    return repaired, {
        "method": "orthogonal_envelope" if use_envelope else "smooth_external_fill",
        "envelope_added_fraction": added_fraction,
        "holes_before": len(before_holes),
        "hole_pixels_before": int(sum(area for _, area in before_holes)),
        "holes_after": len(list(enclosed_holes(repaired))),
        "changed_fraction": float(np.mean(repaired != original)),
        "effective_smooth_sigma": effective_sigma,
        "effective_envelope_closing_radius": effective_closing_radius,
    }


def find_overflow_rescue(
    probability: np.ndarray,
    low_threshold: float,
    reference_threshold: float = 0.70,
    search_start: float = 0.90,
    search_stop: float = 0.97,
    search_step: float = 0.001,
    max_reference_area_ratio: float = 0.80,
    min_reference_bbox_fraction: float = 0.90,
    max_step_area_ratio: float = 0.85,
    max_bbox_contraction_ratio: float = 0.80,
    min_candidate_area_fraction: float = 0.03,
    closing_radius: int = 6,
) -> tuple[np.ndarray | None, dict]:
    low = largest_component_description(probability >= low_threshold)
    reference = largest_component_description(probability >= reference_threshold)
    reference_area_ratio = reference.area_fraction / max(low.area_fraction, 1e-9)
    event_threshold = None
    event = None
    previous = largest_component_description(probability >= search_start)
    for raw_threshold in np.arange(
        search_start + search_step, search_stop + search_step * 0.5, search_step
    ):
        current = largest_component_description(probability >= raw_threshold)
        step_area_ratio = current.area_fraction / max(previous.area_fraction, 1e-9)
        bbox_contraction = current.bbox_fraction / max(reference.bbox_fraction, 1e-9)
        if (
            current.area_fraction >= min_candidate_area_fraction
            and step_area_ratio <= max_step_area_ratio
            and bbox_contraction <= max_bbox_contraction_ratio
        ):
            event_threshold = float(round(raw_threshold, 3))
            event = current
            break
        previous = current
    triggered = (
        reference_area_ratio < max_reference_area_ratio
        and reference.bbox_fraction > min_reference_bbox_fraction
        and event is not None
    )
    diagnostics = {
        "triggered": triggered,
        "reference_area_ratio": reference_area_ratio,
        "reference_bbox_fraction": reference.bbox_fraction,
        "event_threshold": event_threshold,
    }
    if not triggered or event is None:
        return None, diagnostics
    return solidify(event.mask, closing_radius), diagnostics


def refine_prediction(
    probability: np.ndarray,
    raw_disparity: np.ndarray,
    threshold: float,
    **kwargs,
) -> tuple[np.ndarray, dict]:
    disparity = cv2.resize(
        np.asarray(raw_disparity, dtype=np.float32),
        (probability.shape[1], probability.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    exp4, exp4_diagnostics = experiment4_refine(probability, disparity, threshold)
    geometry = geometric_refine(
        exp4,
        gaussian_sigma=float(kwargs.get("gaussian_sigma", 3.0)),
        binary_threshold=float(kwargs.get("binary_threshold", 0.60)),
        closing_radius=int(kwargs.get("geometry_closing_radius", 6)),
        preserve_hole_area=int(kwargs.get("preserve_hole_area", 256)),
    )
    rescue, overflow_diagnostics = find_overflow_rescue(
        probability,
        low_threshold=threshold,
        reference_threshold=float(kwargs.get("reference_threshold", 0.70)),
        search_start=float(kwargs.get("search_start", 0.90)),
        search_stop=float(kwargs.get("search_stop", 0.97)),
        search_step=float(kwargs.get("search_step", 0.001)),
        max_reference_area_ratio=float(kwargs.get("max_reference_area_ratio", 0.80)),
        min_reference_bbox_fraction=float(
            kwargs.get("min_reference_bbox_fraction", 0.90)
        ),
        max_step_area_ratio=float(kwargs.get("max_step_area_ratio", 0.85)),
        max_bbox_contraction_ratio=float(
            kwargs.get("max_bbox_contraction_ratio", 0.80)
        ),
        min_candidate_area_fraction=float(
            kwargs.get("min_candidate_area_fraction", 0.03)
        ),
        closing_radius=int(kwargs.get("overflow_closing_radius", 6)),
    )
    final = geometry if rescue is None else rescue
    topology_diagnostics = {"enabled": False}
    if bool(kwargs.get("enable_topology_repair", True)):
        final, topology_values = topology_repair(
            final,
            smooth_sigma=float(kwargs.get("topology_smooth_sigma", 2.0)),
            smooth_threshold=float(kwargs.get("topology_smooth_threshold", 0.60)),
            envelope_min_added_fraction=float(
                kwargs.get("topology_envelope_min_added_fraction", 0.10)
            ),
            envelope_max_added_fraction=float(
                kwargs.get("topology_envelope_max_added_fraction", 0.30)
            ),
            envelope_closing_radius=int(
                kwargs.get("topology_envelope_closing_radius", 7)
            ),
        )
        topology_diagnostics = {"enabled": True, **topology_values}
    return final, {
        "experiment4": exp4_diagnostics,
        "overflow": overflow_diagnostics,
        "topology": topology_diagnostics,
        "connected_components": mask_stats(final)["connected_components"],
    }


def mask_stats(mask: np.ndarray) -> dict[str, float | int]:
    foreground = np.asarray(mask, dtype=np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    components = int(count - 1)
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return {
        "connected_components": components,
        "foreground_area": int(foreground.sum()),
        "perimeter": float(sum(cv2.arcLength(contour, True) for contour in contours)),
        "largest_component_area": int(
            stats[1:, cv2.CC_STAT_AREA].max() if components else 0
        ),
    }
