#!/usr/bin/env python3
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("run_experiment.py")
SPEC = importlib.util.spec_from_file_location("run_experiment", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DisparityGuidedSegmentationTests(unittest.TestCase):
    def test_scene_name_mapping(self):
        self.assertEqual(
            MODULE.scene_from_name("fdjyp_0_2_202506261657_0011"),
            "202506261657-0011",
        )

    def test_fill_holes_preserves_outer_background(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[3:17, 3:17] = True
        mask[8:12, 8:12] = False
        filled = MODULE.fill_holes(mask)
        self.assertTrue(filled[9, 9])
        self.assertFalse(filled[0, 0])

    def test_centered_component_rejects_remote_blob(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True
        mask[0:30, 0:30] = True
        chosen = MODULE.choose_centered_component(mask)
        self.assertTrue(chosen[50, 50])
        self.assertFalse(chosen[10, 10])

    def test_mask_metrics_perfect_prediction(self):
        target = np.zeros((20, 20), dtype=bool)
        target[5:15, 5:15] = True
        metrics = MODULE.mask_metrics(target, target)
        for key in ("iou", "dice", "precision", "recall", "accuracy"):
            self.assertAlmostEqual(metrics[key], 1.0)

    def test_soft_fusion_recovers_depth_consistent_uncertain_boundary(self):
        probability = np.full((40, 40), 0.01, dtype=np.float32)
        probability[:, 21:] = 0.99
        probability[:, 20] = 0.45
        normalized = np.full((40, 40), 40, dtype=np.uint8)
        normalized[:, 20:] = 220
        semantic = MODULE.fill_holes(
            MODULE.choose_centered_component(probability >= 0.5)
        )
        fused, diagnostics = MODULE.soft_fuse_semantic_and_disparity(
            probability, normalized
        )
        self.assertFalse(semantic[20, 20])
        self.assertTrue(fused[20, 20])
        self.assertGreater(diagnostics["depth_reliability"], 0.0)
        self.assertGreater(diagnostics["changed_fraction"], 0.0)

    def test_recall_priority_fills_ambiguous_hole(self):
        mask = np.zeros((60, 60), dtype=bool)
        mask[5:55, 5:55] = True
        mask[26:34, 26:34] = False
        disparity = np.full((60, 60), 20.0, dtype=np.float32)
        disparity[26:34, 26:34] = 19.0
        repaired, preserved, diagnostics = (
            MODULE.repair_holes_recall_priority(mask, disparity)
        )
        self.assertTrue(repaired[30, 30])
        self.assertFalse(preserved[30, 30])
        self.assertEqual(diagnostics["filled_hole_count"], 1)

    def test_recall_priority_preserves_strong_depth_hole(self):
        mask = np.zeros((90, 90), dtype=bool)
        mask[5:85, 5:85] = True
        mask[25:65, 25:65] = False
        disparity = np.full((90, 90), 30.0, dtype=np.float32)
        disparity[25:65, 25:65] = 10.0
        repaired, preserved, diagnostics = (
            MODULE.repair_holes_recall_priority(mask, disparity)
        )
        self.assertFalse(repaired[45, 45])
        self.assertTrue(preserved[45, 45])
        self.assertEqual(
            diagnostics["preserved_background_hole_count"], 1
        )

    def test_recall_priority_preserves_large_semantic_background(self):
        mask = np.ones((100, 100), dtype=bool)
        mask[20:80, 45:55] = False
        disparity = np.full((100, 100), 30.0, dtype=np.float32)
        repaired, preserved, diagnostics = (
            MODULE.repair_holes_recall_priority(mask, disparity)
        )
        self.assertFalse(repaired[50, 50])
        self.assertTrue(preserved[50, 50])
        self.assertEqual(
            diagnostics["decisions"][0]["reason"], "large_background_hole"
        )

    def test_recall_priority_fusion_never_removes_semantic(self):
        semantic = np.zeros((80, 80), dtype=bool)
        semantic[10:70, 10:70] = True
        soft_v1 = semantic.copy()
        soft_v1[10:20, 10:20] = False
        soft_v1[5:10, 30:50] = True
        disparity = np.full((80, 80), 30.0, dtype=np.float32)
        repaired_semantic, fused, diagnostics = (
            MODULE.combine_soft_fusion_recall_priority(
                semantic, soft_v1, disparity
            )
        )
        self.assertTrue(np.all(fused[repaired_semantic]))
        self.assertAlmostEqual(diagnostics["removed_fraction"], 0.0)
        self.assertGreater(diagnostics["added_fraction"], 0.0)

    def test_binary_ply_header_and_size(self):
        points = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
        colors = np.asarray([[10, 20, 30]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cloud.ply"
            MODULE.write_binary_ply(path, points, colors)
            data = path.read_bytes()
        self.assertTrue(data.startswith(b"ply\nformat binary_little_endian 1.0\n"))
        self.assertIn(b"element vertex 1\n", data)
        self.assertTrue(data.endswith(points[0].tobytes() + colors[0].tobytes()))


if __name__ == "__main__":
    unittest.main()
