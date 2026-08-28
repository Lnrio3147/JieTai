from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from evaluate_ablation import human_mask_correspondence_metrics  # noqa: E402


class HumanMaskCorrespondenceTest(unittest.TestCase):
    def test_disparity_shift_is_checked_against_right_ground_truth(self) -> None:
        disparity = np.full((3, 8), np.nan, dtype=np.float32)
        disparity[1, 4:6] = 2.0
        left = np.zeros((3, 8), dtype=bool)
        left[1, 4:6] = True
        right = np.zeros((3, 8), dtype=bool)
        right[1, 2:4] = True
        metrics = human_mask_correspondence_metrics(disparity, left, right)
        self.assertEqual(metrics["human_gt_valid_subject_pixels"], 2)
        self.assertEqual(metrics["human_right_mask_violation_rate"], 0.0)

    def test_background_correspondence_is_a_violation(self) -> None:
        disparity = np.full((3, 8), np.nan, dtype=np.float32)
        disparity[1, 4:6] = 1.0
        left = np.zeros((3, 8), dtype=bool)
        left[1, 4:6] = True
        right = np.zeros((3, 8), dtype=bool)
        metrics = human_mask_correspondence_metrics(disparity, left, right)
        self.assertEqual(metrics["human_right_mask_violation_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
