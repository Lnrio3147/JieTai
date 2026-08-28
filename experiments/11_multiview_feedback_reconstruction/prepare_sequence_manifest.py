#!/usr/bin/env python3
"""Build a validated reconstruction manifest from an ordered cloud list."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import config_experiment11 as config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--cloud", type=Path, action="append", required=True)
    parser.add_argument("--left-image", type=Path, action="append", default=None)
    parser.add_argument("--pose-json", type=Path, default=None)
    parser.add_argument(
        "--shared-camera-frame",
        action="store_true",
        help="Set every pose to identity; only valid for a fixed calibrated camera.",
    )
    parser.add_argument("--units", required=True)
    parser.add_argument("--voxel-size", type=float, required=True)
    parser.add_argument("--calibration-id", default="")
    parser.add_argument("--same-static-object-confirmed", action="store_true")
    parser.add_argument("--common-calibration-confirmed", action="store_true")
    parser.add_argument(
        "--intended-use",
        choices=("multiview_coverage", "repeated_view_stability", "legacy_audit"),
        default="multiview_coverage",
    )
    parser.add_argument("--azimuth", type=float, action="append", default=None)
    parser.add_argument("--elevation", type=float, action="append", default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def relative_path(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve())


def load_pose_map(path: Path | None) -> dict[str, list] | None:
    if path is None:
        return None
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--pose-json must contain an object mapping view IDs to poses")
    return payload


def main() -> None:
    args = parse_args()
    if len(args.cloud) < 2:
        raise ValueError("At least two --cloud arguments are required")
    if args.voxel_size <= 0:
        raise ValueError("--voxel-size must be positive")
    for values, name in (
        (args.left_image, "--left-image"),
        (args.azimuth, "--azimuth"),
        (args.elevation, "--elevation"),
    ):
        if values is not None and len(values) != len(args.cloud):
            raise ValueError(f"{name} count must match --cloud count")
    if args.pose_json is not None and args.shared_camera_frame:
        raise ValueError("Choose either --pose-json or --shared-camera-frame")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pose_map = load_pose_map(args.pose_json)
    ids = []
    for index, cloud in enumerate(args.cloud):
        cloud = cloud.resolve()
        if not cloud.is_file():
            raise FileNotFoundError(cloud)
        candidate = cloud.parent.name or cloud.stem
        view_id = candidate if candidate not in ids else f"{candidate}_{index:03d}"
        ids.append(view_id)
    views = []
    for index, (view_id, cloud) in enumerate(zip(ids, args.cloud)):
        if args.shared_camera_frame:
            pose = np.eye(4, dtype=float).tolist()
        elif pose_map is not None:
            if view_id not in pose_map:
                raise KeyError(f"Missing pose for {view_id}")
            pose = pose_map[view_id]
        else:
            pose = None
        left_image = None
        if args.left_image is not None:
            image = args.left_image[index].resolve()
            if not image.is_file():
                raise FileNotFoundError(image)
            left_image = relative_path(image, output.parent)
        views.append(
            {
                "id": view_id,
                "cloud": relative_path(cloud, output.parent),
                "left_image": left_image,
                "pose_camera_to_world": pose,
                "azimuth_deg": (
                    None if args.azimuth is None else float(args.azimuth[index])
                ),
                "elevation_deg": (
                    None if args.elevation is None else float(args.elevation[index])
                ),
            }
        )
    payload = {
        "sequence_id": args.sequence_id,
        "units": args.units,
        "voxel_size": float(args.voxel_size),
        "reference_cloud": None,
        "acquisition": {
            "same_static_object_confirmed": args.same_static_object_confirmed,
            "common_calibration_confirmed": args.common_calibration_confirmed,
            "calibration_id": args.calibration_id or None,
            "intended_use": args.intended_use,
            "shared_camera_frame": args.shared_camera_frame,
        },
        "views": views,
        "candidate_views": [],
        "registration": {
            "all_pairs": False,
            "loop_closure": True,
            "min_fitness": config.MIN_REGISTRATION_FITNESS,
            "max_rmse_factor": config.MAX_REGISTRATION_RMSE_FACTOR,
        },
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
