from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from utils.roi import select_common_roi  # noqa: E402


def select(left: np.ndarray, right: np.ndarray, **overrides):
    options = {
        "threshold": 0.5,
        "margin": 8,
        "max_disparity": 32,
        "stride": 16,
        "min_foreground_pixels": 8,
        "min_area_ratio": 0.02,
        "max_area_ratio": 0.90,
    }
    options.update(overrides)
    return select_common_roi(left, right, **options)


class StereoROITest(unittest.TestCase):
    def test_joint_roi_contains_both_views_and_is_stride_aligned(self) -> None:
        left = np.zeros((100, 200), dtype=np.float32)
        right = np.zeros_like(left)
        left[20:80, 60:100] = 1.0
        right[20:80, 40:80] = 1.0
        roi = select(left, right)
        self.assertTrue(roi.used)
        self.assertLessEqual(roi.x0, 40)
        self.assertGreaterEqual(roi.x1, 100)
        self.assertLessEqual(roi.y0, 20)
        self.assertGreaterEqual(roi.y1, 80)
        self.assertEqual(roi.x0 % 16, 0)
        self.assertEqual(roi.y0 % 16, 0)

    def test_no_foreground_falls_back_to_full_frame(self) -> None:
        probability = np.zeros((50, 80), dtype=np.float32)
        roi = select(probability, probability)
        self.assertFalse(roi.used)
        self.assertEqual(roi.reason, "no_confident_foreground")
        self.assertEqual((roi.x0, roi.y0, roi.x1, roi.y1), (0, 0, 80, 50))

    def test_single_left_mask_reserves_disparity_search_to_the_left(self) -> None:
        left = np.zeros((80, 160), dtype=np.float32)
        right = np.zeros_like(left)
        left[24:56, 80:112] = 1.0
        roi = select(left, right)
        self.assertTrue(roi.used)
        self.assertLessEqual(roi.x0, 80 - 32 - 8)

    def test_nearly_full_mask_uses_full_frame(self) -> None:
        probability = np.ones((100, 200), dtype=np.float32)
        roi = select(probability, probability)
        self.assertFalse(roi.used)
        self.assertEqual(roi.reason, "roi_savings_too_small")


if __name__ == "__main__":
    unittest.main()
