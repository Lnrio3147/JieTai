import importlib.util
import unittest
from pathlib import Path

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "tools/refine_bisenet_subject_masks.py"
SPEC = importlib.util.spec_from_file_location("refine_bisenet_subject_masks", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RefineBiSeNetSubjectMasksTest(unittest.TestCase):
    def test_keep_largest_component_removes_island(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[2:15, 2:15] = True
        mask[17:19, 17:19] = True
        result, component_count, removed_pixels = MODULE.keep_largest_component(mask)
        self.assertEqual(component_count, 2)
        self.assertEqual(removed_pixels, 4)
        self.assertTrue(result[5, 5])
        self.assertFalse(result[17, 17])

    def test_disparity_continuous_hole_is_filled(self):
        probability = np.ones((40, 40), dtype=np.float32)
        probability[15:25, 15:25] = 0.0
        disparity = np.full((40, 40), 30.0, dtype=np.float32)
        raw, refined, stats = MODULE.refine_mask(
            probability,
            (40, 40),
            disparity,
            closing_radius=0,
            crop=(0, 40, 0, 40),
            max_fill_hole_fraction=0.2,
        )
        self.assertFalse(raw[20, 20])
        self.assertTrue(refined[20, 20])
        self.assertEqual(stats["filled_hole_count"], 1)
        self.assertEqual(stats["final_foreground_components"], 1)

    def test_disparity_discontinuous_hole_is_preserved(self):
        probability = np.ones((40, 40), dtype=np.float32)
        probability[15:25, 15:25] = 0.0
        disparity = np.full((40, 40), 30.0, dtype=np.float32)
        disparity[15:25, 15:25] = 45.0
        _, refined, stats = MODULE.refine_mask(
            probability,
            (40, 40),
            disparity,
            closing_radius=0,
            crop=(0, 40, 0, 40),
            max_fill_hole_fraction=0.2,
        )
        self.assertFalse(refined[20, 20])
        self.assertEqual(stats["preserved_hole_count"], 1)
        count, _, _, _ = cv2.connectedComponentsWithStats(
            refined.astype(np.uint8), connectivity=8
        )
        self.assertEqual(count - 1, 1)


if __name__ == "__main__":
    unittest.main()
