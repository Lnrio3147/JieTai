#!/usr/bin/env python3
"""Register, optimize, fuse, evaluate, and recommend the next acquisition view."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import open3d as o3d

import config_experiment11 as config
from fusion import fuse_clouds, incremental_coverage, low_support_cloud
from manifest import load_manifest
from next_best_view import recommend_next_view
from pose_graph import register_sequence
from quality import (
    coverage_metrics,
    reference_metrics,
    registration_metrics,
    should_stop_acquisition,
)
from registration import estimate_voxel_size, load_cloud


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--all-pairs", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--loop-closure", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--min-fitness", type=float, default=None)
    parser.add_argument("--max-rmse-factor", type=float, default=None)
    parser.add_argument(
        "--preflight-report",
        type=Path,
        default=None,
        help="Require a matching preflight report with reconstruction_ready=true.",
    )
    return parser.parse_args()


def write_edge_csv(path: Path, rows: list[dict]) -> None:
    compact = [
        {key: value for key, value in row.items() if key not in ("transformation", "information")}
        for row in rows
    ]
    fields = sorted({key for row in compact for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(compact)


def normalized_candidates(manifest: dict) -> list[dict]:
    candidates = manifest["candidate_views"]
    if not candidates:
        return []
    poses = [view["pose"] for view in manifest["views"]]
    if any(pose is None for pose in poses):
        raise ValueError(
            "Manifest candidate views require camera-to-world poses for all acquired views"
        )
    anchor_inverse = np.linalg.inv(poses[0])
    return [
        {**candidate, "pose": anchor_inverse @ candidate["pose"]}
        for candidate in candidates
    ]


def validate_preflight_report(manifest_path: Path, report_path: Path) -> dict:
    report_path = report_path.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("completed"):
        raise ValueError(f"Incomplete preflight report: {report_path}")
    reported_manifest = Path(str(report.get("manifest", ""))).resolve()
    if reported_manifest != manifest_path.resolve():
        raise ValueError(
            f"Preflight manifest mismatch: {reported_manifest} != {manifest_path.resolve()}"
        )
    assessment = report.get("assessment", {})
    if assessment.get("reconstruction_ready") is not True:
        blockers = assessment.get("blockers", ["unspecified"])
        raise RuntimeError(f"Preflight blocked reconstruction: {blockers}")
    return report


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    preflight = None
    if args.preflight_report is not None:
        preflight = validate_preflight_report(
            manifest["manifest_path"], args.preflight_report
        )
    sequence_id = str(manifest.get("sequence_id", args.manifest.stem))
    output = (
        args.output.resolve()
        if args.output is not None
        else (config.RESULTS_DIR / sequence_id).resolve()
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    views = manifest["views"]
    view_ids = [view["id"] for view in views]
    clouds = [load_cloud(view["cloud"]) for view in views]
    voxel_size = args.voxel_size or manifest["voxel_size"]
    if voxel_size is None:
        voxel_size = estimate_voxel_size(clouds[0])
    voxel_size = float(voxel_size)
    registration_config = manifest.get("registration", {})
    all_pairs = (
        bool(registration_config.get("all_pairs", False))
        if args.all_pairs is None
        else args.all_pairs
    )
    loop_closure = (
        bool(registration_config.get("loop_closure", True))
        if args.loop_closure is None
        else args.loop_closure
    )
    min_fitness = float(
        args.min_fitness
        if args.min_fitness is not None
        else registration_config.get("min_fitness", config.MIN_REGISTRATION_FITNESS)
    )
    max_rmse_factor = float(
        args.max_rmse_factor
        if args.max_rmse_factor is not None
        else registration_config.get(
            "max_rmse_factor", config.MAX_REGISTRATION_RMSE_FACTOR
        )
    )

    poses, edges = register_sequence(
        clouds,
        view_ids,
        voxel_size,
        manifest_poses=[view["pose"] for view in views],
        all_pairs=all_pairs,
        loop_closure=loop_closure,
        min_fitness=min_fitness,
        max_rmse_factor=max_rmse_factor,
    )
    edge_rows = [edge.to_dict(view_ids) for edge in edges]
    fused, support, voxel_keys = fuse_clouds(clouds, poses, voxel_size)
    low_support = low_support_cloud(
        fused, support, config.MIN_SUPPORTED_VIEW_COUNT
    )
    increments = incremental_coverage(clouds, poses, voxel_size)
    fused_path = output / "fused_cloud.ply"
    low_support_path = output / "low_support_cloud.ply"
    if not o3d.io.write_point_cloud(str(fused_path), fused, write_ascii=False):
        raise IOError(f"Failed to write {fused_path}")
    low_support.paint_uniform_color((1.0, 0.1, 0.1))
    if not o3d.io.write_point_cloud(
        str(low_support_path), low_support, write_ascii=False
    ):
        raise IOError(f"Failed to write {low_support_path}")
    np.save(output / "voxel_support.npy", support)
    np.save(output / "voxel_keys.npy", voxel_keys)

    pose_payload = {
        "coordinate_frame": "first_camera",
        "poses_camera_to_global": {
            view_id: pose.tolist() for view_id, pose in zip(view_ids, poses)
        },
    }
    (output / "optimized_poses.json").write_text(
        json.dumps(pose_payload, indent=2), encoding="utf-8"
    )
    (output / "registration_edges.json").write_text(
        json.dumps(edge_rows, indent=2), encoding="utf-8"
    )
    write_edge_csv(output / "registration_edges.csv", edge_rows)

    quality = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sequence_id": sequence_id,
        "units": manifest.get("units", "unspecified"),
        "voxel_size": voxel_size,
        "registration": registration_metrics(edge_rows),
        "coverage": coverage_metrics(support, len(views)),
        "incremental_coverage": increments,
        "acquisition_stop": should_stop_acquisition(increments),
        "reference": None,
    }
    if manifest["reference_cloud"] is not None:
        reference = load_cloud(manifest["reference_cloud"])
        if all(view["pose"] is not None for view in views):
            # Optimized output is anchored in the first-camera frame.  A scan
            # stored in the manifest world frame must be normalized identically.
            reference.transform(np.linalg.inv(views[0]["pose"]))
        quality["reference"] = reference_metrics(fused, reference, voxel_size)
    (output / "quality.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )

    next_view = recommend_next_view(
        np.asarray(fused.points),
        np.asarray(low_support.points),
        poses,
        normalized_candidates(manifest),
    )
    next_view["acquisition_stop"] = quality["acquisition_stop"]
    (output / "next_view.json").write_text(
        json.dumps(next_view, indent=2), encoding="utf-8"
    )
    summary = {
        "completed": True,
        "manifest": str(manifest["manifest_path"]),
        "output": str(output),
        "view_count": len(views),
        "fused_points": len(fused.points),
        "low_support_points": len(low_support.points),
        "voxel_size": voxel_size,
        "quality": quality,
        "recommended_next_view": next_view["recommended"],
        "preflight_report": (
            None if preflight is None else str(args.preflight_report.resolve())
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
