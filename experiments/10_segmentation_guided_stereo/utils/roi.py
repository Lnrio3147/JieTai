"""Confidence-gated common ROI selection for rectified stereo pairs."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class StereoROI:
    x0: int
    y0: int
    x1: int
    y1: int
    used: bool
    reason: str
    area_ratio: float
    left_foreground_pixels: int
    right_foreground_pixels: int

    def slices(self) -> tuple[slice, slice]:
        return slice(self.y0, self.y1), slice(self.x0, self.x1)

    def to_dict(self) -> dict:
        return asdict(self)


def _foreground_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _full_roi(
    width: int,
    height: int,
    reason: str,
    left_pixels: int,
    right_pixels: int,
) -> StereoROI:
    return StereoROI(
        0,
        0,
        width,
        height,
        False,
        reason,
        1.0,
        left_pixels,
        right_pixels,
    )


def select_common_roi(
    left_probability: np.ndarray,
    right_probability: np.ndarray,
    *,
    threshold: float,
    margin: int,
    max_disparity: int,
    stride: int,
    min_foreground_pixels: int,
    min_area_ratio: float,
    max_area_ratio: float,
) -> StereoROI:
    left_probability = np.asarray(left_probability, dtype=np.float32)
    right_probability = np.asarray(right_probability, dtype=np.float32)
    if left_probability.shape != right_probability.shape:
        raise ValueError(
            "Left/right probability shape mismatch: "
            f"{left_probability.shape} vs {right_probability.shape}"
        )
    if left_probability.ndim != 2:
        raise ValueError("Mask probabilities must be two-dimensional")
    height, width = left_probability.shape
    left_mask = left_probability >= threshold
    right_mask = right_probability >= threshold
    left_pixels = int(left_mask.sum())
    right_pixels = int(right_mask.sum())
    left_bbox = _foreground_bbox(left_mask)
    right_bbox = _foreground_bbox(right_mask)
    left_valid = left_pixels >= min_foreground_pixels and left_bbox is not None
    right_valid = right_pixels >= min_foreground_pixels and right_bbox is not None
    if not left_valid and not right_valid:
        return _full_roi(
            width, height, "no_confident_foreground", left_pixels, right_pixels
        )

    boxes = [box for box, valid in ((left_bbox, left_valid), (right_bbox, right_valid)) if valid]
    assert boxes
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[2] for box in boxes)
    y1 = max(box[3] for box in boxes)
    if left_valid and not right_valid:
        x0 -= max_disparity
    elif right_valid and not left_valid:
        x1 += max_disparity
    x0 -= margin
    y0 -= margin
    x1 += margin
    y1 += margin

    x0 = max(0, (x0 // stride) * stride)
    y0 = max(0, (y0 // stride) * stride)
    x1 = min(width, ((x1 + stride - 1) // stride) * stride)
    y1 = min(height, ((y1 + stride - 1) // stride) * stride)
    if x1 <= x0 or y1 <= y0:
        return _full_roi(width, height, "empty_roi", left_pixels, right_pixels)
    area_ratio = (x1 - x0) * (y1 - y0) / float(width * height)
    if area_ratio < min_area_ratio:
        return _full_roi(
            width, height, "roi_too_small_for_safe_context", left_pixels, right_pixels
        )
    if area_ratio > max_area_ratio:
        return _full_roi(
            width, height, "roi_savings_too_small", left_pixels, right_pixels
        )
    return StereoROI(
        x0,
        y0,
        x1,
        y1,
        True,
        "confident_joint_foreground",
        float(area_ratio),
        left_pixels,
        right_pixels,
    )
