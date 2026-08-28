"""Rank candidate camera poses against currently under-observed voxels."""

from __future__ import annotations

import math

import numpy as np

import config_experiment11 as config


def look_at_pose(
    position: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray((0.0, 0.0, 1.0) if up is None else up, dtype=np.float64)
    forward = target - position
    norm = float(np.linalg.norm(forward))
    if norm <= 1e-12:
        raise ValueError("Camera position and look-at target cannot coincide")
    forward /= norm
    if abs(float(np.dot(forward, up / np.linalg.norm(up)))) > 0.98:
        up = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.column_stack((right, down, forward))
    pose[:3, 3] = position
    return pose


def generate_candidate_views(
    object_points: np.ndarray,
    existing_poses: list[np.ndarray],
) -> list[dict]:
    points = np.asarray(object_points, dtype=np.float64)
    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    existing_distances = [
        float(np.linalg.norm(pose[:3, 3] - center)) for pose in existing_poses
    ]
    radius = (
        float(np.median(existing_distances))
        if existing_distances and np.median(existing_distances) > diagonal
        else config.AUTO_CANDIDATE_RADIUS_FACTOR * diagonal
    )
    candidates = []
    for elevation_deg in config.AUTO_CANDIDATE_ELEVATIONS:
        elevation = math.radians(elevation_deg)
        for azimuth_deg in config.AUTO_CANDIDATE_AZIMUTHS:
            azimuth = math.radians(azimuth_deg)
            offset = radius * np.asarray(
                (
                    math.cos(elevation) * math.cos(azimuth),
                    math.cos(elevation) * math.sin(azimuth),
                    math.sin(elevation),
                )
            )
            pose = look_at_pose(center + offset, center)
            candidates.append(
                {
                    "id": f"auto_az{azimuth_deg:03d}_el{elevation_deg:02d}",
                    "pose": pose,
                    "azimuth_deg": float(azimuth_deg),
                    "elevation_deg": float(elevation_deg),
                }
            )
    return candidates


def visible_fraction(
    points: np.ndarray,
    camera_to_global: np.ndarray,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
) -> tuple[float, int]:
    points = np.asarray(points, dtype=np.float64)
    global_to_camera = np.linalg.inv(camera_to_global)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    camera = (global_to_camera @ homogeneous.T).T[:, :3]
    valid = camera[:, 2] > 1e-9
    horizontal = np.abs(np.arctan2(camera[:, 0], camera[:, 2]))
    vertical = np.abs(np.arctan2(camera[:, 1], camera[:, 2]))
    valid &= horizontal <= math.radians(horizontal_fov_deg) / 2.0
    valid &= vertical <= math.radians(vertical_fov_deg) / 2.0
    return float(valid.mean()) if len(valid) else 0.0, int(valid.sum())


def angular_novelty(candidate: np.ndarray, existing: list[np.ndarray]) -> float:
    if not existing:
        return 1.0
    direction = candidate[:3, 2] / np.linalg.norm(candidate[:3, 2])
    angles = []
    for pose in existing:
        other = pose[:3, 2] / np.linalg.norm(pose[:3, 2])
        angles.append(math.acos(float(np.clip(np.dot(direction, other), -1.0, 1.0))))
    return min(angles) / math.pi


def rank_candidates(
    low_support_points: np.ndarray,
    candidates: list[dict],
    existing_poses: list[np.ndarray],
    object_diagonal: float,
) -> list[dict]:
    if len(low_support_points) == 0:
        return []
    last_position = existing_poses[-1][:3, 3] if existing_poses else None
    rows = []
    for candidate in candidates:
        pose = np.asarray(candidate["pose"], dtype=np.float64)
        h_fov = float(candidate.get("horizontal_fov_deg", config.DEFAULT_HORIZONTAL_FOV_DEG))
        v_fov = float(candidate.get("vertical_fov_deg", config.DEFAULT_VERTICAL_FOV_DEG))
        visible, visible_count = visible_fraction(
            low_support_points, pose, h_fov, v_fov
        )
        novelty = angular_novelty(pose, existing_poses)
        motion = (
            float(np.linalg.norm(pose[:3, 3] - last_position))
            / max(object_diagonal, 1e-12)
            if last_position is not None
            else 0.0
        )
        score = (
            config.NBV_VISIBLE_WEIGHT * visible
            + config.NBV_NOVELTY_WEIGHT * novelty
            - config.NBV_MOTION_WEIGHT * motion
        )
        rows.append(
            {
                "id": candidate["id"],
                "score": float(score),
                "visible_low_support_fraction": visible,
                "visible_low_support_voxels": visible_count,
                "angular_novelty": novelty,
                "normalized_motion_cost": motion,
                "pose_camera_to_global": pose.tolist(),
                "azimuth_deg": candidate.get("azimuth_deg"),
                "elevation_deg": candidate.get("elevation_deg"),
            }
        )
    return sorted(rows, key=lambda row: row["score"], reverse=True)


def recommend_next_view(
    fused_points: np.ndarray,
    low_support_points: np.ndarray,
    existing_poses: list[np.ndarray],
    candidates: list[dict],
) -> dict:
    fused_points = np.asarray(fused_points, dtype=np.float64)
    low_support_points = np.asarray(low_support_points, dtype=np.float64)
    diagonal = float(np.linalg.norm(fused_points.max(axis=0) - fused_points.min(axis=0)))
    if not candidates:
        candidates = generate_candidate_views(fused_points, existing_poses)
        source = "auto_spherical_candidates"
    else:
        source = "manifest_candidates"
    ranked = rank_candidates(
        low_support_points, candidates, existing_poses, diagonal
    )
    return {
        "candidate_source": source,
        "under_observed_voxels": int(len(low_support_points)),
        "recommended": ranked[0] if ranked else None,
        "ranked_candidates": ranked,
        "limitations": (
            "Visibility uses a pinhole field-of-view test without collision or "
            "self-occlusion; a robot controller must perform final safety checks."
        ),
    }
