#!/usr/bin/env python3
"""Fixed-ROI point-cloud reconstruction for the FDJYP-3 stereo setup."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from postprocess import TRADITION_CROP


FDJYP3_Q = np.array(
    [
        [1.0, 0.0, 0.0, -2.2227048492431641e02],
        [0.0, 1.0, 0.0, -7.2185147857666016e02],
        [0.0, 0.0, 0.0, 9.4420949981369597e02],
        [0.0, 0.0, 3.9456987572407826e-01, 0.0],
    ],
    dtype=np.float32,
)


def adjusted_q_for_crop(
    q_matrix: np.ndarray = FDJYP3_Q,
    crop: tuple[int, int, int, int] = TRADITION_CROP,
) -> np.ndarray:
    """Return Q expressed in the coordinate system of the fixed image crop."""
    y0, _, x0, _ = crop
    adjusted = np.asarray(q_matrix, dtype=np.float32).copy()
    if adjusted.shape != (4, 4):
        raise ValueError(f"Q matrix must be 4x4, got {adjusted.shape}")
    adjusted[0, 3] += x0
    adjusted[1, 3] += y0
    return adjusted


def reconstruct_point_cloud(
    subject_disparity: np.ndarray,
    left_bgr_crop: np.ndarray,
    *,
    q_matrix: np.ndarray = FDJYP3_Q,
    crop: tuple[int, int, int, int] = TRADITION_CROP,
    min_disparity: float = 5.0,
    max_disparity: float = 300.0,
    min_depth: float = 0.0,
    max_depth: float = 200.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproject subject disparity and return XYZ, RGB, and the valid-pixel mask."""
    disparity = np.asarray(subject_disparity, dtype=np.float32)
    image = np.asarray(left_bgr_crop)
    if disparity.ndim != 2:
        raise ValueError(f"Expected 2D disparity, got {disparity.shape}")
    if image.shape != (*disparity.shape, 3):
        raise ValueError(
            f"Image/disparity shape mismatch: image={image.shape}, "
            f"disparity={disparity.shape}"
        )

    points_image = cv2.reprojectImageTo3D(
        disparity,
        adjusted_q_for_crop(q_matrix, crop),
        handleMissingValues=True,
    )
    depth = points_image[..., 2]
    valid = (
        np.isfinite(disparity)
        & np.isfinite(points_image).all(axis=2)
        & (disparity >= min_disparity)
        & (disparity <= max_disparity)
        & (depth > min_depth)
        & (depth < max_depth)
    )
    points = np.ascontiguousarray(points_image[valid], dtype=np.float32)
    rgb = np.ascontiguousarray(image[..., ::-1][valid], dtype=np.uint8)
    return points, rgb, valid


def write_binary_ply(path: Path, points: np.ndarray, rgb: np.ndarray) -> None:
    """Write compact binary little-endian XYZRGB PLY without Open3D."""
    xyz = np.asarray(points, dtype="<f4")
    colors = np.asarray(rgb, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got {xyz.shape}")
    if colors.shape != xyz.shape:
        raise ValueError(f"Point/color shape mismatch: {xyz.shape} vs {colors.shape}")

    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertices = np.empty(xyz.shape[0], dtype=vertex_dtype)
    vertices["x"], vertices["y"], vertices["z"] = xyz.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {xyz.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)
