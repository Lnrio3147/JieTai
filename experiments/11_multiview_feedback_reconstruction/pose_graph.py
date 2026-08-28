"""Pair selection, registration, and global pose-graph optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

import config_experiment11 as config
from registration import RegistrationResult, register_pair


@dataclass
class RegisteredEdge:
    source: int
    target: int
    uncertain: bool
    result: RegistrationResult

    def to_dict(self, view_ids: list[str]) -> dict:
        return {
            "source_index": self.source,
            "target_index": self.target,
            "source_id": view_ids[self.source],
            "target_id": view_ids[self.target],
            "uncertain": self.uncertain,
            **self.result.to_dict(),
        }


def relative_source_to_target(
    source_camera_to_global: np.ndarray,
    target_camera_to_global: np.ndarray,
) -> np.ndarray:
    return np.linalg.inv(target_camera_to_global) @ source_camera_to_global


def candidate_edges(
    view_count: int, *, all_pairs: bool, loop_closure: bool
) -> list[tuple[int, int, bool]]:
    if view_count < 2:
        raise ValueError("At least two views are required")
    edges: list[tuple[int, int, bool]] = [
        (index, index + 1, False) for index in range(view_count - 1)
    ]
    if all_pairs:
        existing = {(source, target) for source, target, _ in edges}
        edges.extend(
            (source, target, True)
            for source in range(view_count)
            for target in range(source + 1, view_count)
            if (source, target) not in existing
        )
    elif loop_closure and view_count >= 4:
        edges.append((0, view_count - 1, True))
    return edges


def normalize_manifest_poses(poses: list[np.ndarray | None]) -> list[np.ndarray] | None:
    available = [pose is not None for pose in poses]
    if not any(available):
        return None
    if not all(available):
        raise ValueError("Either all views or no views must provide camera-to-world poses")
    typed = [np.asarray(pose, dtype=np.float64) for pose in poses if pose is not None]
    anchor_inverse = np.linalg.inv(typed[0])
    return [anchor_inverse @ pose for pose in typed]


def register_sequence(
    clouds: list[o3d.geometry.PointCloud],
    view_ids: list[str],
    voxel_size: float,
    *,
    manifest_poses: list[np.ndarray | None],
    all_pairs: bool,
    loop_closure: bool,
    min_fitness: float,
    max_rmse_factor: float,
) -> tuple[list[np.ndarray], list[RegisteredEdge]]:
    if len(clouds) != len(view_ids) or len(clouds) != len(manifest_poses):
        raise ValueError("Cloud, ID, and pose counts must match")
    normalized_poses = normalize_manifest_poses(manifest_poses)
    node_poses: list[np.ndarray | None] = [None] * len(clouds)
    node_poses[0] = np.eye(4) if normalized_poses is None else normalized_poses[0]
    registered: list[RegisteredEdge] = []

    pairs = candidate_edges(
        len(clouds), all_pairs=all_pairs, loop_closure=loop_closure
    )
    # Sequential edges are intentionally processed first so they can initialize
    # every node when robot/turntable poses are unavailable.
    for source, target, uncertain in pairs:
        if normalized_poses is not None:
            initial = relative_source_to_target(
                normalized_poses[source], normalized_poses[target]
            )
        elif uncertain:
            # Loop closures must be independently verified by feature matching;
            # using accumulated odometry as the only prior would hide drift.
            initial = None
        else:
            initial = None
        result = register_pair(
            clouds[source],
            clouds[target],
            voxel_size,
            initial=initial,
            min_fitness=min_fitness,
            max_rmse_factor=max_rmse_factor,
        )
        edge = RegisteredEdge(source, target, uncertain, result)
        registered.append(edge)
        if not result.accepted:
            if not uncertain:
                raise RuntimeError(
                    f"Required sequential registration {view_ids[source]} -> "
                    f"{view_ids[target]} failed: fitness={result.fitness:.4f}, "
                    f"rmse={result.inlier_rmse:.6g}"
                )
            continue
        if normalized_poses is None and not uncertain:
            assert node_poses[source] is not None
            node_poses[target] = (
                np.asarray(node_poses[source]) @ np.linalg.inv(result.transformation)
            )

    if normalized_poses is not None:
        node_poses = [pose.copy() for pose in normalized_poses]
    if any(pose is None for pose in node_poses):
        missing = [view_ids[index] for index, pose in enumerate(node_poses) if pose is None]
        raise RuntimeError(f"Could not initialize poses for: {missing}")
    typed_poses = [np.asarray(pose, dtype=np.float64) for pose in node_poses]

    graph = o3d.pipelines.registration.PoseGraph()
    for pose in typed_poses:
        graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(pose))
    for edge in registered:
        if not edge.result.accepted:
            continue
        graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                edge.source,
                edge.target,
                edge.result.transformation,
                edge.result.information,
                uncertain=edge.uncertain,
            )
        )
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=1.5 * voxel_size,
        edge_prune_threshold=0.25,
        preference_loop_closure=1.0,
        reference_node=0,
    )
    o3d.pipelines.registration.global_optimization(
        graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )
    optimized = [np.asarray(node.pose, dtype=np.float64) for node in graph.nodes]
    return optimized, registered
