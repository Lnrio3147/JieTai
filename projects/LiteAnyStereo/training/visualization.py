"""Qualitative disparity visualizations for validation predictions."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .data import TRADITION_CROP


def _to_numpy_2d(value):
    if torch.is_tensor(value):
        value = value.detach().float().cpu().numpy()
    value = np.asarray(value)
    while value.ndim > 2:
        value = value[0]
    if value.ndim != 2:
        raise ValueError(f"Expected a two-dimensional map, got {value.shape}")
    return value


def _to_rgb_image(value):
    if torch.is_tensor(value):
        value = value.detach().float().cpu().numpy()
    value = np.asarray(value)
    while value.ndim > 3:
        value = value[0]
    if value.ndim != 3:
        raise ValueError(f"Expected an RGB tensor/image, got {value.shape}")
    if value.shape[0] in {1, 3, 4}:
        value = np.moveaxis(value, 0, -1)
    if value.shape[-1] == 1:
        value = np.repeat(value, 3, axis=-1)
    return np.clip(value[..., :3], 0, 255).astype(np.uint8)


def colorize_map(values, *, minimum, maximum, valid=None):
    """Colorize a scalar map with a fixed range so scenes remain comparable."""
    values = np.asarray(values, dtype=np.float32)
    if maximum <= minimum:
        raise ValueError("Visualization maximum must be greater than minimum")
    finite = np.isfinite(values)
    normalized = np.clip((values - minimum) / (maximum - minimum), 0.0, 1.0)
    normalized = np.where(finite, normalized, 0.0)
    color = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    mask = finite if valid is None else finite & np.asarray(valid, dtype=bool)
    color[~mask] = 0
    return color


def _auto_colorize(values):
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros((*values.shape, 3), dtype=np.uint8)
    minimum = float(values[finite].min())
    maximum = float(values[finite].max())
    if maximum <= minimum:
        maximum = minimum + 1.0
    return colorize_map(values, minimum=minimum, maximum=maximum, valid=finite)


def _labeled_panel(image, title):
    header_height = 38
    panel = np.zeros((image.shape[0] + header_height, image.shape[1], 3), dtype=np.uint8)
    panel[header_height:] = image
    cv2.putText(
        panel,
        title,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def save_validation_vis(
    path,
    *,
    left,
    prediction,
    target,
    valid,
    evaluation_protocol,
    traditional=None,
    traditional_label="Previous algorithm",
    disparity_max=192.0,
    error_max=20.0,
):
    """Save prediction views and optional LAS/traditional/reference comparison."""
    left_image = _to_rgb_image(left)
    prediction_map = _to_numpy_2d(prediction)
    target_map = _to_numpy_2d(target)
    valid_map = _to_numpy_2d(valid).astype(bool)
    traditional_map = _to_numpy_2d(traditional) if traditional is not None else None

    if evaluation_protocol == "tradition":
        y0, y1, x0, x1 = TRADITION_CROP
        region = np.s_[y0:y1, x0:x1]
        left_image = left_image[region]
        prediction_map = prediction_map[region]
        target_map = target_map[region]
        valid_map = valid_map[region]
        if traditional_map is not None:
            traditional_map = traditional_map[region]

    prediction_color = colorize_map(
        prediction_map,
        minimum=0.0,
        maximum=float(disparity_max),
        valid=np.isfinite(prediction_map),
    )
    target_color = colorize_map(
        target_map,
        minimum=0.0,
        maximum=float(disparity_max),
        valid=valid_map,
    )
    absolute_error = np.abs(prediction_map - target_map)
    error_color = colorize_map(
        absolute_error,
        minimum=0.0,
        maximum=float(error_max),
        valid=valid_map,
    )
    panels = [
        _labeled_panel(left_image, "Left ROI"),
        _labeled_panel(prediction_color, f"LiteAnyStereo disparity [0,{disparity_max:g}] px"),
        _labeled_panel(target_color, f"Reference disparity [0,{disparity_max:g}] px"),
        _labeled_panel(error_color, f"Absolute error [0,{error_max:g}] px"),
    ]
    montage = np.concatenate(
        [np.concatenate(panels[:2], axis=1), np.concatenate(panels[2:], axis=1)],
        axis=0,
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the floating-point prediction used to build the visualization.
    # This allows later metric recalculation without inferring the model again
    # or attempting to recover disparity values from a coloured PNG.
    np.save(output_path.with_name("disp.npy"), prediction_map.astype(np.float32))
    Image.fromarray(_auto_colorize(prediction_map)).save(output_path)
    Image.fromarray(prediction_color).save(output_path.with_name("vis_fixed.png"))
    Image.fromarray(montage).save(output_path.with_name("comparison.png"))
    if traditional_map is not None:
        traditional_valid = np.isfinite(traditional_map) & (traditional_map > 0.0)
        traditional_color = colorize_map(
            traditional_map,
            minimum=0.0,
            maximum=float(disparity_max),
            valid=traditional_valid,
        )
        traditional_error = np.abs(traditional_map - target_map)
        traditional_error_color = colorize_map(
            traditional_error,
            minimum=0.0,
            maximum=float(error_max),
            # Keep finite zero-disparity holes in the error view: the unified
            # metrics use the reference mask, so missing traditional output is
            # an error rather than an ignored pixel.
            valid=valid_map & np.isfinite(traditional_map),
        )
        comparison_panels = [
            _labeled_panel(left_image, "Left ROI"),
            _labeled_panel(traditional_color, f"{traditional_label} disparity [0,{disparity_max:g}] px"),
            _labeled_panel(prediction_color, f"LiteAnyStereo disparity [0,{disparity_max:g}] px"),
            _labeled_panel(target_color, f"Reference disparity [0,{disparity_max:g}] px"),
            _labeled_panel(traditional_error_color, f"{traditional_label} absolute error [0,{error_max:g}] px"),
            _labeled_panel(error_color, f"LiteAnyStereo absolute error [0,{error_max:g}] px"),
        ]
        tradition_montage = np.concatenate(
            [
                np.concatenate(comparison_panels[:3], axis=1),
                np.concatenate(comparison_panels[3:], axis=1),
            ],
            axis=0,
        )
        Image.fromarray(tradition_montage).save(
            output_path.with_name("traditional_comparison.png")
        )
    return output_path


def save_inference_vis(
    directory,
    *,
    left,
    right,
    prediction,
    disparity_max=192.0,
):
    """Save raw-inference visualizations when no reference disparity exists."""
    left_image = _to_rgb_image(left)
    right_image = _to_rgb_image(right)
    prediction_map = _to_numpy_2d(prediction)
    if left_image.shape != right_image.shape:
        raise ValueError(
            f"Left/right visualization shape mismatch: {left_image.shape} vs {right_image.shape}"
        )
    if left_image.shape[:2] != prediction_map.shape:
        raise ValueError(
            f"Image/prediction visualization shape mismatch: "
            f"{left_image.shape[:2]} vs {prediction_map.shape}"
        )

    prediction_auto = _auto_colorize(prediction_map)
    prediction_fixed = colorize_map(
        prediction_map,
        minimum=0.0,
        maximum=float(disparity_max),
        valid=np.isfinite(prediction_map),
    )
    panels = [
        _labeled_panel(left_image, "Left input"),
        _labeled_panel(right_image, "Right input"),
        _labeled_panel(prediction_auto, "LiteAnyStereo disparity [auto scale]"),
        _labeled_panel(prediction_fixed, f"LiteAnyStereo disparity [0,{disparity_max:g}] px"),
    ]
    montage = np.concatenate(
        [np.concatenate(panels[:2], axis=1), np.concatenate(panels[2:], axis=1)],
        axis=0,
    )
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(prediction_auto).save(output_dir / "vis.png")
    Image.fromarray(prediction_fixed).save(output_dir / "vis_fixed.png")
    Image.fromarray(montage).save(output_dir / "comparison.png")
    return output_dir


def save_algorithm_comparison_vis(
    path,
    *,
    left,
    traditional,
    prediction,
    disparity_max=192.0,
):
    """Save left/previous/current disparity comparison without a reference map."""
    left_image = _to_rgb_image(left)
    traditional_map = _to_numpy_2d(traditional)
    prediction_map = _to_numpy_2d(prediction)
    expected_shape = left_image.shape[:2]
    if traditional_map.shape != expected_shape or prediction_map.shape != expected_shape:
        raise ValueError(
            "Comparison maps must match the left image: "
            f"image={expected_shape}, traditional={traditional_map.shape}, "
            f"prediction={prediction_map.shape}"
        )
    traditional_valid = np.isfinite(traditional_map) & (traditional_map > 0.0)
    traditional_auto = _auto_colorize(
        np.where(traditional_valid, traditional_map, np.nan)
    )
    prediction_auto = _auto_colorize(prediction_map)
    traditional_fixed = colorize_map(
        traditional_map,
        minimum=0.0,
        maximum=float(disparity_max),
        valid=traditional_valid,
    )
    prediction_fixed = colorize_map(
        prediction_map,
        minimum=0.0,
        maximum=float(disparity_max),
        valid=np.isfinite(prediction_map),
    )
    panels = [
        _labeled_panel(left_image, "Left input"),
        _labeled_panel(traditional_auto, "Previous algorithm disparity [auto scale]"),
        _labeled_panel(prediction_auto, "LiteAnyStereo disparity [auto scale]"),
        _labeled_panel(traditional_fixed, f"Previous algorithm disparity [0,{disparity_max:g}] px"),
        _labeled_panel(prediction_fixed, f"LiteAnyStereo disparity [0,{disparity_max:g}] px"),
        _labeled_panel(
            colorize_map(
                np.abs(prediction_map - traditional_map),
                minimum=0.0,
                maximum=20.0,
                valid=traditional_valid & np.isfinite(prediction_map),
            ),
            "Absolute difference [0,20] px (not error)",
        ),
    ]
    montage = np.concatenate(
        [
            np.concatenate(panels[:3], axis=1),
            np.concatenate(panels[3:], axis=1),
        ],
        axis=0,
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(montage).save(output_path)
    return output_path
