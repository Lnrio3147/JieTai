"""FPFH/RANSAC coarse registration followed by multi-scale GICP."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import open3d as o3d

import config_experiment11 as config


@dataclass
class RegistrationResult:
    transformation: np.ndarray
    information: np.ndarray
    fitness: float
    inlier_rmse: float
    accepted: bool
    coarse_method: str
    source_points: int
    target_points: int

    def to_dict(self) -> dict:
        return {
            "transformation": self.transformation.tolist(),
            "information": self.information.tolist(),
            "fitness": self.fitness,
            "inlier_rmse": self.inlier_rmse,
            "accepted": self.accepted,
            "coarse_method": self.coarse_method,
            "source_points": self.source_points,
            "target_points": self.target_points,
        }


def clean_cloud(cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    points = np.asarray(cloud.points)
    if points.size == 0:
        raise ValueError("Point cloud is empty")
    valid = np.isfinite(points).all(axis=1)
    cloud = cloud.select_by_index(np.flatnonzero(valid).tolist())
    if len(cloud.points) < 20:
        raise ValueError("Point cloud has fewer than 20 finite points")
    if len(cloud.points) >= 100:
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
    if len(cloud.points) < 20:
        raise ValueError("Point cloud became too small after cleaning")
    return cloud


def cloud_diagonal(cloud: o3d.geometry.PointCloud) -> float:
    extent = np.asarray(cloud.get_axis_aligned_bounding_box().get_extent())
    diagonal = float(np.linalg.norm(extent))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("Point cloud has an invalid bounding-box diagonal")
    return diagonal


def estimate_voxel_size(cloud: o3d.geometry.PointCloud) -> float:
    return cloud_diagonal(cloud) * config.VOXEL_SIZE_FRACTION


def estimate_normals(cloud: o3d.geometry.PointCloud, radius: float) -> None:
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=40)
    )
    cloud.normalize_normals()


def downsample_with_features(
    cloud: o3d.geometry.PointCloud, voxel_size: float
) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    down = cloud.voxel_down_sample(voxel_size)
    if len(down.points) < 20:
        raise ValueError(
            f"Only {len(down.points)} points remain at voxel size {voxel_size:g}"
        )
    estimate_normals(down, config.NORMAL_RADIUS_FACTOR * voxel_size)
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=config.FPFH_RADIUS_FACTOR * voxel_size, max_nn=100
        ),
    )
    return down, feature


def ransac_coarse_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size: float,
) -> np.ndarray:
    source_down, source_feature = downsample_with_features(source, voxel_size)
    target_down, target_feature = downsample_with_features(target, voxel_size)
    distance = config.RANSAC_DISTANCE_FACTOR * voxel_size
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_feature,
        target_feature,
        True,
        distance,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.90),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return np.asarray(result.transformation, dtype=np.float64)


def multiscale_gicp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size: float,
    initial: np.ndarray,
) -> o3d.pipelines.registration.RegistrationResult:
    transformation = np.asarray(initial, dtype=np.float64)
    result = None
    for voxel_factor, distance_factor in zip(
        config.GICP_VOXEL_FACTORS, config.GICP_DISTANCE_FACTORS
    ):
        scale_voxel = max(voxel_size * voxel_factor, voxel_size * 0.25)
        source_down = source.voxel_down_sample(scale_voxel)
        target_down = target.voxel_down_sample(scale_voxel)
        estimate_normals(source_down, config.NORMAL_RADIUS_FACTOR * scale_voxel)
        estimate_normals(target_down, config.NORMAL_RADIUS_FACTOR * scale_voxel)
        result = o3d.pipelines.registration.registration_generalized_icp(
            source_down,
            target_down,
            max(voxel_size * distance_factor, scale_voxel),
            transformation,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=80,
            ),
        )
        transformation = np.asarray(result.transformation, dtype=np.float64)
    assert result is not None
    return result


def register_pair(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size: float,
    *,
    initial: np.ndarray | None = None,
    min_fitness: float = config.MIN_REGISTRATION_FITNESS,
    max_rmse_factor: float = config.MAX_REGISTRATION_RMSE_FACTOR,
) -> RegistrationResult:
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    source = clean_cloud(copy.deepcopy(source))
    target = clean_cloud(copy.deepcopy(target))
    if initial is None:
        initial = ransac_coarse_registration(source, target, voxel_size)
        coarse_method = "fpfh_ransac"
    else:
        initial = np.asarray(initial, dtype=np.float64)
        if initial.shape != (4, 4):
            raise ValueError("initial transformation must be 4x4")
        coarse_method = "manifest_pose"
    refined = multiscale_gicp(source, target, voxel_size, initial)
    transformation = np.asarray(refined.transformation, dtype=np.float64)
    information = (
        o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source,
            target,
            max_correspondence_distance=2.0 * voxel_size,
            transformation=transformation,
        )
    )
    fitness = float(refined.fitness)
    rmse = float(refined.inlier_rmse)
    accepted = fitness >= min_fitness and rmse <= max_rmse_factor * voxel_size
    return RegistrationResult(
        transformation=transformation,
        information=np.asarray(information, dtype=np.float64),
        fitness=fitness,
        inlier_rmse=rmse,
        accepted=accepted,
        coarse_method=coarse_method,
        source_points=len(source.points),
        target_points=len(target.points),
    )


def load_cloud(path) -> o3d.geometry.PointCloud:
    cloud = o3d.io.read_point_cloud(str(path))
    if len(cloud.points) == 0:
        raise ValueError(f"Empty or unreadable point cloud: {path}")
    return clean_cloud(cloud)
