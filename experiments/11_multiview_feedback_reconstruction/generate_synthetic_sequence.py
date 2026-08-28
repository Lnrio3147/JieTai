#!/usr/bin/env python3
"""Generate a deterministic asymmetric multi-view sequence for regression tests."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import open3d as o3d

import config_experiment11 as config
from next_best_view import look_at_pose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=config.EXPERIMENT_DIR / "inputs/synthetic_sequence",
    )
    parser.add_argument("--views", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--omit-poses",
        action="store_true",
        help="Hide generated camera poses to exercise FPFH/RANSAC initialization.",
    )
    return parser.parse_args()


def make_reference(seed: int) -> o3d.geometry.PointCloud:
    np.random.seed(seed)
    box = o3d.geometry.TriangleMesh.create_box(width=42.0, height=28.0, depth=16.0)
    box.translate((-21.0, -14.0, -8.0))
    wedge = o3d.geometry.TriangleMesh.create_cone(radius=9.0, height=20.0, resolution=32)
    wedge.rotate(
        o3d.geometry.get_rotation_matrix_from_xyz((0.0, np.pi / 2.0, 0.0)),
        center=(0.0, 0.0, 0.0),
    )
    wedge.translate((21.0, 6.0, -4.0))
    mesh = box + wedge
    mesh.compute_vertex_normals()
    cloud = mesh.sample_points_uniformly(number_of_points=12000)
    points = np.asarray(cloud.points)
    low = points.min(axis=0)
    scale = np.maximum(points.max(axis=0) - low, 1e-6)
    cloud.colors = o3d.utility.Vector3dVector((points - low) / scale)
    return cloud


def main() -> None:
    args = parse_args()
    if args.views < 2:
        raise ValueError("--views must be at least two")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    reference = make_reference(args.seed)
    reference_path = output / "reference.ply"
    o3d.io.write_point_cloud(str(reference_path), reference, write_ascii=False)
    global_points = np.asarray(reference.points)
    global_colors = np.asarray(reference.colors)
    rng = np.random.default_rng(args.seed)
    views = []
    first_pose = None
    center = np.asarray((0.0, 0.0, 0.0))
    for index in range(args.views):
        azimuth = 2.0 * np.pi * index / args.views
        position = np.asarray(
            (110.0 * np.cos(azimuth), 110.0 * np.sin(azimuth), 45.0)
        )
        pose = look_at_pose(position, center)
        if first_pose is None:
            first_pose = pose.copy()
        global_to_camera = np.linalg.inv(pose)
        homogeneous = np.column_stack((global_points, np.ones(len(global_points))))
        local_points = (global_to_camera @ homogeneous.T).T[:, :3]
        keep = rng.random(len(local_points)) > 0.08
        local_points = local_points[keep]
        local_points += rng.normal(0.0, 0.04, size=local_points.shape)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(local_points)
        cloud.colors = o3d.utility.Vector3dVector(global_colors[keep])
        cloud_path = output / f"view_{index:03d}.ply"
        o3d.io.write_point_cloud(str(cloud_path), cloud, write_ascii=False)
        views.append(
            {
                "id": f"view_{index:03d}",
                "cloud": cloud_path.name,
                "left_image": None,
                "pose_camera_to_world": None if args.omit_poses else pose.tolist(),
                "azimuth_deg": float(np.degrees(azimuth)),
                "elevation_deg": 22.0,
            }
        )
    if args.omit_poses:
        assert first_pose is not None
        reference_first_camera = copy.deepcopy(reference)
        reference_first_camera.transform(np.linalg.inv(first_pose))
        reference_manifest_path = output / "reference_first_camera.ply"
        o3d.io.write_point_cloud(
            str(reference_manifest_path), reference_first_camera, write_ascii=False
        )
    else:
        reference_manifest_path = reference_path
    manifest = {
        "sequence_id": "synthetic_asymmetric_workpiece",
        "units": "mm",
        "voxel_size": 0.8,
        "reference_cloud": reference_manifest_path.name,
        "acquisition": {
            "same_static_object_confirmed": True,
            "common_calibration_confirmed": True,
            "calibration_id": "synthetic_exact_v1",
            "intended_use": "multiview_coverage",
            "shared_camera_frame": False,
        },
        "views": views,
        "candidate_views": [],
        "registration": {
            "all_pairs": False,
            "loop_closure": True,
            "min_fitness": 0.5,
            "max_rmse_factor": 2.0,
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
