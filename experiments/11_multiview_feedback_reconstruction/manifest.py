"""Manifest loading and validation for multi-view sequences."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def resolve_path(base: Path, value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def parse_pose(value, name: str) -> np.ndarray | None:
    if value is None:
        return None
    pose = np.asarray(value, dtype=np.float64)
    if pose.size == 16:
        pose = pose.reshape(4, 4)
    if pose.shape != (4, 4):
        raise ValueError(f"{name} must contain a 4x4 pose")
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
        raise ValueError(f"{name} rotation is not orthonormal")
    return pose


def load_manifest(path: Path) -> dict:
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    views = data.get("views")
    if not isinstance(views, list) or len(views) < 2:
        raise ValueError("A reconstruction sequence requires at least two views")
    ids = [str(view.get("id", "")) for view in views]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Every view needs a unique non-empty id")
    base = path.parent
    normalized_views = []
    for index, view in enumerate(views):
        cloud = resolve_path(base, view.get("cloud"))
        if cloud is None or not cloud.is_file():
            raise FileNotFoundError(cloud or f"views[{index}].cloud")
        normalized = dict(view)
        normalized["id"] = ids[index]
        normalized["cloud"] = cloud
        normalized["left_image"] = resolve_path(base, view.get("left_image"))
        normalized["pose"] = parse_pose(
            view.get("pose_camera_to_world"),
            f"views[{index}].pose_camera_to_world",
        )
        normalized_views.append(normalized)
    candidates = []
    for index, candidate in enumerate(data.get("candidate_views", [])):
        normalized = dict(candidate)
        normalized["id"] = str(candidate.get("id", f"candidate_{index:03d}"))
        normalized["pose"] = parse_pose(
            candidate.get("pose_camera_to_world"),
            f"candidate_views[{index}].pose_camera_to_world",
        )
        if normalized["pose"] is None:
            raise ValueError(f"candidate_views[{index}] requires a pose")
        candidates.append(normalized)
    output = dict(data)
    output["manifest_path"] = path
    output["views"] = normalized_views
    output["candidate_views"] = candidates
    output["reference_cloud"] = resolve_path(base, data.get("reference_cloud"))
    voxel_size = data.get("voxel_size")
    if voxel_size is not None and float(voxel_size) <= 0:
        raise ValueError("voxel_size must be positive")
    output["voxel_size"] = None if voxel_size is None else float(voxel_size)
    return output
