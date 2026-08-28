#!/usr/bin/env python3
"""Audit whether a point-cloud sequence supports reconstruction claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

import config_experiment11 as config
from manifest import load_manifest
from pose_graph import relative_source_to_target
from registration import register_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--registration", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def summarize_cloud(cloud: o3d.geometry.PointCloud, voxel_size: float) -> dict:
    points = np.asarray(cloud.points)
    finite = np.isfinite(points).all(axis=1)
    finite_points = points[finite]
    if len(finite_points) < 20:
        raise ValueError("Point cloud has fewer than 20 finite points")
    low, high = np.percentile(finite_points, (1.0, 99.0), axis=0)
    robust_extent = high - low
    robust_diagonal = float(np.linalg.norm(robust_extent))
    clean = o3d.geometry.PointCloud()
    clean.points = o3d.utility.Vector3dVector(finite_points)
    sampled = clean.voxel_down_sample(max(voxel_size * 0.5, 1e-9))
    distances = np.asarray(sampled.compute_nearest_neighbor_distance())
    return {
        "raw_points": int(len(points)),
        "finite_points": int(len(finite_points)),
        "finite_fraction": float(finite.mean()),
        "qa_downsampled_points": int(len(sampled.points)),
        "robust_extent": robust_extent.tolist(),
        "robust_diagonal": robust_diagonal,
        "median_nearest_neighbor_distance": (
            float(np.median(distances)) if distances.size else None
        ),
        "p95_nearest_neighbor_distance": (
            float(np.percentile(distances, 95.0)) if distances.size else None
        ),
    }


def viewpoint_diversity(manifest: dict, diagonal: float) -> dict:
    poses = [view["pose"] for view in manifest["views"]]
    translations = []
    rotations = []
    if all(pose is not None for pose in poses):
        anchor = poses[0]
        assert anchor is not None
        for pose in poses[1:]:
            assert pose is not None
            relative = np.linalg.inv(anchor) @ pose
            translations.append(float(np.linalg.norm(relative[:3, 3])))
            rotations.append(rotation_angle_deg(relative[:3, :3]))
    azimuths = [
        float(view["azimuth_deg"])
        for view in manifest["views"]
        if view.get("azimuth_deg") is not None
    ]
    elevations = [
        float(view["elevation_deg"])
        for view in manifest["views"]
        if view.get("elevation_deg") is not None
    ]
    azimuth_span = max(azimuths) - min(azimuths) if len(azimuths) >= 2 else None
    elevation_span = (
        max(elevations) - min(elevations) if len(elevations) >= 2 else None
    )
    maximum_translation = max(translations, default=0.0)
    maximum_rotation = max(rotations, default=0.0)
    diverse = (
        maximum_translation >= 0.05 * max(diagonal, 1e-9)
        or maximum_rotation >= 5.0
        or (azimuth_span is not None and azimuth_span >= 10.0)
        or (elevation_span is not None and elevation_span >= 10.0)
    )
    return {
        "sufficient_for_coverage_claim": bool(diverse),
        "maximum_camera_translation": maximum_translation,
        "maximum_camera_rotation_deg": maximum_rotation,
        "azimuth_span_deg": azimuth_span,
        "elevation_span_deg": elevation_span,
    }


def assess_claims(
    manifest: dict,
    cloud_summaries: list[dict],
    registration_rows: list[dict],
) -> dict:
    acquisition = manifest.get("acquisition", {})
    blockers = []
    warnings = []
    if acquisition.get("same_static_object_confirmed") is not True:
        blockers.append("same_static_object_not_confirmed")
    if acquisition.get("common_calibration_confirmed") is not True:
        blockers.append("common_calibration_not_confirmed")
    if str(manifest.get("units", "")).lower() in ("", "unknown", "unspecified"):
        blockers.append("point_cloud_units_not_confirmed")
    diagonals = [row["robust_diagonal"] for row in cloud_summaries]
    scale_ratio = max(diagonals) / max(min(diagonals), 1e-12)
    if scale_ratio > 3.0:
        blockers.append("cloud_scale_ratio_above_3")
    elif scale_ratio > 2.0:
        warnings.append("cloud_scale_ratio_above_2")
    if any(row["finite_fraction"] < 0.999 for row in cloud_summaries):
        warnings.append("nonfinite_points_present")
    required_edge_count = len(manifest["views"]) - 1
    failed_edges = [row for row in registration_rows if not row["accepted"]]
    if len(registration_rows) < required_edge_count:
        blockers.append("required_registration_not_completed")
    elif failed_edges:
        blockers.append("required_registration_edge_failed")
    diversity = viewpoint_diversity(manifest, float(np.median(diagonals)))
    if not diversity["sufficient_for_coverage_claim"]:
        warnings.append("viewpoint_diversity_not_demonstrated")
    reconstruction_ready = not blockers
    coverage_ready = reconstruction_ready and diversity[
        "sufficient_for_coverage_claim"
    ]
    return {
        "reconstruction_ready": reconstruction_ready,
        "coverage_claim_ready": coverage_ready,
        "blockers": blockers,
        "warnings": warnings,
        "robust_diagonal_scale_ratio": float(scale_ratio),
        "viewpoint_diversity": diversity,
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = load_manifest(args.manifest)
    voxel_size = manifest["voxel_size"]
    raw_clouds = [o3d.io.read_point_cloud(str(view["cloud"])) for view in manifest["views"]]
    if voxel_size is None:
        first_points = np.asarray(raw_clouds[0].points)
        extent = np.ptp(first_points[np.isfinite(first_points).all(axis=1)], axis=0)
        voxel_size = float(np.linalg.norm(extent) * config.VOXEL_SIZE_FRACTION)
    summaries = [summarize_cloud(cloud, voxel_size) for cloud in raw_clouds]
    registration_rows = []
    if args.registration:
        poses = [view["pose"] for view in manifest["views"]]
        for index in range(len(raw_clouds) - 1):
            source = raw_clouds[index].voxel_down_sample(voxel_size * 0.5)
            target = raw_clouds[index + 1].voxel_down_sample(voxel_size * 0.5)
            initial = None
            if all(pose is not None for pose in poses):
                initial = relative_source_to_target(poses[index], poses[index + 1])
            try:
                result = register_pair(
                    source,
                    target,
                    voxel_size,
                    initial=initial,
                    min_fitness=float(
                        manifest.get("registration", {}).get(
                            "min_fitness", config.MIN_REGISTRATION_FITNESS
                        )
                    ),
                    max_rmse_factor=float(
                        manifest.get("registration", {}).get(
                            "max_rmse_factor", config.MAX_REGISTRATION_RMSE_FACTOR
                        )
                    ),
                )
                registration_rows.append(
                    {
                        "source_id": manifest["views"][index]["id"],
                        "target_id": manifest["views"][index + 1]["id"],
                        **result.to_dict(),
                    }
                )
            except Exception as error:  # keep the full preflight report actionable
                registration_rows.append(
                    {
                        "source_id": manifest["views"][index]["id"],
                        "target_id": manifest["views"][index + 1]["id"],
                        "accepted": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    payload = {
        "completed": True,
        "manifest": str(manifest["manifest_path"]),
        "voxel_size": float(voxel_size),
        "clouds": [
            {"id": view["id"], "cloud": str(view["cloud"]), **summary}
            for view, summary in zip(manifest["views"], summaries)
        ],
        "registration": registration_rows,
        "assessment": assess_claims(manifest, summaries, registration_rows),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["assessment"], indent=2))


if __name__ == "__main__":
    main()
