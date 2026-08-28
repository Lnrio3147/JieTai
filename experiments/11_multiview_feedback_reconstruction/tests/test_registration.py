from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from pose_graph import relative_source_to_target  # noqa: E402
from registration import register_pair  # noqa: E402


class RegistrationTest(unittest.TestCase):
    def test_relative_pose_convention(self) -> None:
        source_pose = np.eye(4)
        source_pose[0, 3] = 2.0
        target_pose = np.eye(4)
        target_pose[0, 3] = 5.0
        relative = relative_source_to_target(source_pose, target_pose)
        self.assertAlmostEqual(relative[0, 3], -3.0)

    def test_gicp_refines_known_initial_transform(self) -> None:
        rng = np.random.default_rng(7)
        points = rng.normal(size=(1200, 3))
        points[:, 0] *= 2.0
        points[:, 2] += 0.2 * points[:, 0] ** 2
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(points)
        target = copy.deepcopy(source)
        transform = np.eye(4)
        angle = 0.12
        transform[:3, :3] = np.asarray(
            (
                (np.cos(angle), -np.sin(angle), 0.0),
                (np.sin(angle), np.cos(angle), 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        transform[:3, 3] = (0.3, -0.1, 0.2)
        target.transform(transform)
        initial = transform.copy()
        initial[0, 3] += 0.03
        result = register_pair(
            source,
            target,
            voxel_size=0.12,
            initial=initial,
            min_fitness=0.5,
            max_rmse_factor=2.0,
        )
        self.assertTrue(result.accepted)
        self.assertGreater(result.fitness, 0.9)
        np.testing.assert_allclose(result.transformation, transform, atol=0.03)


if __name__ == "__main__":
    unittest.main()
