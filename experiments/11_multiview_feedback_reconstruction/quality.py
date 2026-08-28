"""Reconstruction coverage and optional reference-cloud metrics."""

from __future__ import annotations

import numpy as np
import open3d as o3d

import config_experiment11 as config


def coverage_metrics(support: np.ndarray, view_count: int) -> dict:
    support = np.asarray(support, dtype=np.int32)
    if support.ndim != 1 or support.size == 0:
        raise ValueError("support must be a non-empty vector")
    return {
        "surface_voxels": int(support.size),
        "view_count": int(view_count),
        "mean_observations_per_voxel": float(support.mean()),
        "single_view_voxel_fraction": float((support == 1).mean()),
        "supported_by_2_views_fraction": float((support >= 2).mean()),
        "supported_by_3_views_fraction": float((support >= 3).mean()),
        "maximum_observations": int(support.max()),
    }


def registration_metrics(edge_rows: list[dict]) -> dict:
    accepted = [row for row in edge_rows if row["accepted"]]
    return {
        "edge_count": len(edge_rows),
        "accepted_edge_count": len(accepted),
        "rejected_edge_count": len(edge_rows) - len(accepted),
        "mean_fitness": (
            float(np.mean([row["fitness"] for row in accepted])) if accepted else None
        ),
        "mean_inlier_rmse": (
            float(np.mean([row["inlier_rmse"] for row in accepted]))
            if accepted
            else None
        ),
        "maximum_inlier_rmse": (
            float(np.max([row["inlier_rmse"] for row in accepted]))
            if accepted
            else None
        ),
    }


def reference_metrics(
    fused: o3d.geometry.PointCloud,
    reference: o3d.geometry.PointCloud,
    voxel_size: float,
) -> dict:
    fused_down = fused.voxel_down_sample(voxel_size)
    reference_down = reference.voxel_down_sample(voxel_size)
    if len(fused_down.points) == 0 or len(reference_down.points) == 0:
        raise ValueError("Fused/reference cloud is empty after downsampling")
    fused_to_reference = np.asarray(
        fused_down.compute_point_cloud_distance(reference_down), dtype=np.float64
    )
    reference_to_fused = np.asarray(
        reference_down.compute_point_cloud_distance(fused_down), dtype=np.float64
    )
    threshold = config.REFERENCE_F_SCORE_DISTANCE_FACTOR * voxel_size
    precision = float((fused_to_reference <= threshold).mean())
    recall = float((reference_to_fused <= threshold).mean())
    return {
        "threshold": threshold,
        "fused_to_reference_mean": float(fused_to_reference.mean()),
        "fused_to_reference_p95": float(np.percentile(fused_to_reference, 95.0)),
        "reference_to_fused_mean": float(reference_to_fused.mean()),
        "reference_to_fused_p95": float(np.percentile(reference_to_fused, 95.0)),
        "precision": precision,
        "recall": recall,
        "f_score": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def should_stop_acquisition(incremental: list[dict]) -> dict:
    needed = config.STOP_CONSECUTIVE_VIEWS
    recent = incremental[-needed:]
    stop = len(recent) == needed and all(
        row["new_coverage_gain"] < config.STOP_NEW_COVERAGE_GAIN for row in recent
    )
    return {
        "stop_recommended": stop,
        "gain_threshold": config.STOP_NEW_COVERAGE_GAIN,
        "consecutive_views_required": needed,
        "recent_gains": [row["new_coverage_gain"] for row in recent],
    }
