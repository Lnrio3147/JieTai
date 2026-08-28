from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from fusion import fuse_clouds, incremental_coverage  # noqa: E402
from quality import coverage_metrics  # noqa: E402


def cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    value = o3d.geometry.PointCloud()
    value.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    value.colors = o3d.utility.Vector3dVector(
        np.full(points.shape, 0.5, dtype=np.float64)
    )
    return value


class FusionQualityTest(unittest.TestCase):
    def test_support_counts_unique_views(self) -> None:
        global_points = np.asarray(
            [(x, y, z) for x in range(4) for y in range(3) for z in range(2)],
            dtype=np.float64,
        )
        pose0 = np.eye(4)
        pose1 = np.eye(4)
        pose1[0, 3] = 10.0
        local1 = global_points.copy()
        local1[:, 0] -= 10.0
        fused, support, _ = fuse_clouds(
            [cloud(global_points), cloud(local1)], [pose0, pose1], voxel_size=0.5
        )
        self.assertEqual(len(fused.points), len(global_points))
        self.assertTrue(np.all(support == 2))
        metrics = coverage_metrics(support, 2)
        self.assertEqual(metrics["supported_by_2_views_fraction"], 1.0)

    def test_incremental_coverage_detects_repeated_surface(self) -> None:
        points = np.arange(30, dtype=np.float64).reshape(10, 3)
        rows = incremental_coverage(
            [cloud(points), cloud(points)], [np.eye(4), np.eye(4)], voxel_size=0.1
        )
        self.assertEqual(rows[0]["new_voxels"], 10)
        self.assertEqual(rows[1]["new_voxels"], 0)
        self.assertEqual(rows[1]["new_coverage_gain"], 0.0)


if __name__ == "__main__":
    unittest.main()
