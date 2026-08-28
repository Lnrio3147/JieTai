from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from prepare_right_annotation_tasks import (  # noqa: E402
    evenly_spaced_indices,
    mask_to_isat_objects,
)


class RightAnnotationTaskTest(unittest.TestCase):
    def test_evenly_spaced_selection_includes_range_endpoints(self) -> None:
        self.assertEqual(evenly_spaced_indices(10, 3), [0, 4, 9])
        self.assertEqual(evenly_spaced_indices(2, 5), [0, 1])

    def test_binary_mask_becomes_valid_isat_polygon(self) -> None:
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[4:16, 7:25] = 255
        objects = mask_to_isat_objects(mask)
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["category"], "workpiece")
        self.assertGreaterEqual(len(objects[0]["segmentation"]), 4)
        self.assertGreater(objects[0]["area"], 100.0)


if __name__ == "__main__":
    unittest.main()
