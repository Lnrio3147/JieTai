from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
LAS_ROOT = ROOT / "projects/LiteAnyStereo"
sys.path.insert(0, str(LAS_ROOT))

from core.submodule import build_stereo_mask_guidance  # noqa: E402


class MaskGuidanceTest(unittest.TestCase):
    def test_rectified_shift_selects_expected_disparity(self) -> None:
        left = torch.zeros(1, 1, 1, 6)
        right = torch.zeros(1, 1, 1, 6)
        left[0, 0, 0, 4] = 1.0
        right[0, 0, 0, 2] = 1.0
        guidance = build_stereo_mask_guidance(left, right, 4, (1, 6))
        self.assertEqual(tuple(guidance.shape), (1, 4, 1, 6))
        self.assertEqual(float(guidance[0, 2, 0, 4]), 1.0)
        self.assertEqual(float(guidance[0, 0, 0, 4]), 0.0)
        self.assertEqual(float(guidance[0, 1, 0, 4]), 0.0)
        self.assertEqual(float(guidance[0, 3, 0, 4]), 0.0)

    def test_background_is_neutral(self) -> None:
        left = torch.zeros(1, 1, 2, 5)
        right = torch.rand(1, 1, 2, 5)
        guidance = build_stereo_mask_guidance(left, right, 3, (2, 5))
        torch.testing.assert_close(guidance, torch.ones_like(guidance))

    def test_probability_shape_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "right_probability"):
            build_stereo_mask_guidance(torch.ones(1, 1, 2, 2), None, 2, (2, 2))


if __name__ == "__main__":
    unittest.main()
