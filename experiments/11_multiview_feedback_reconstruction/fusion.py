"""Observation-aware voxel fusion for registered subject point clouds."""

from __future__ import annotations

import copy

import numpy as np
import open3d as o3d


def transformed_arrays(
    clouds: list[o3d.geometry.PointCloud], poses: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(clouds) != len(poses):
        raise ValueError("Cloud and pose counts must match")
    points_parts = []
    color_parts = []
    view_parts = []
    for view_index, (cloud, pose) in enumerate(zip(clouds, poses)):
        transformed = copy.deepcopy(cloud)
        transformed.transform(pose)
        points = np.asarray(transformed.points, dtype=np.float64)
        if len(points) == 0:
            continue
        if transformed.has_colors():
            colors = np.asarray(transformed.colors, dtype=np.float64)
        else:
            colors = np.full(points.shape, 0.7, dtype=np.float64)
        valid = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
        points_parts.append(points[valid])
        color_parts.append(np.clip(colors[valid], 0.0, 1.0))
        view_parts.append(np.full(int(valid.sum()), view_index, dtype=np.int32))
    if not points_parts:
        raise ValueError("No finite points are available for fusion")
    return (
        np.concatenate(points_parts),
        np.concatenate(color_parts),
        np.concatenate(view_parts),
    )


def fuse_clouds(
    clouds: list[o3d.geometry.PointCloud],
    poses: list[np.ndarray],
    voxel_size: float,
) -> tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray]:
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    points, colors, view_ids = transformed_arrays(clouds, poses)
    voxel_keys = np.floor(points / voxel_size).astype(np.int64)
    unique_keys, inverse = np.unique(voxel_keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(unique_keys)).astype(np.float64)
    point_sum = np.zeros((len(unique_keys), 3), dtype=np.float64)
    color_sum = np.zeros_like(point_sum)
    np.add.at(point_sum, inverse, points)
    np.add.at(color_sum, inverse, colors)
    fused_points = point_sum / counts[:, None]
    fused_colors = color_sum / counts[:, None]
    voxel_view_pairs = np.unique(np.column_stack((inverse, view_ids)), axis=0)
    support = np.bincount(
        voxel_view_pairs[:, 0], minlength=len(unique_keys)
    ).astype(np.int32)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(fused_points)
    cloud.colors = o3d.utility.Vector3dVector(np.clip(fused_colors, 0.0, 1.0))
    return cloud, support, unique_keys


def low_support_cloud(
    fused: o3d.geometry.PointCloud,
    support: np.ndarray,
    minimum_views: int,
) -> o3d.geometry.PointCloud:
    indices = np.flatnonzero(support < minimum_views)
    return fused.select_by_index(indices.tolist())


def incremental_coverage(
    clouds: list[o3d.geometry.PointCloud],
    poses: list[np.ndarray],
    voxel_size: float,
) -> list[dict]:
    observed: set[tuple[int, int, int]] = set()
    rows = []
    for index, (cloud, pose) in enumerate(zip(clouds, poses)):
        transformed = copy.deepcopy(cloud)
        transformed.transform(pose)
        points = np.asarray(transformed.points, dtype=np.float64)
        keys = {
            tuple(value)
            for value in np.floor(points[np.isfinite(points).all(axis=1)] / voxel_size)
            .astype(np.int64)
            .tolist()
        }
        new_keys = keys - observed
        previous = len(observed)
        observed.update(keys)
        rows.append(
            {
                "view_index": index,
                "view_voxels": len(keys),
                "new_voxels": len(new_keys),
                "total_voxels": len(observed),
                "new_coverage_gain": (
                    1.0 if previous == 0 else len(new_keys) / previous
                ),
            }
        )
    return rows
