from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from utils.data import preprocess_rgb  # noqa: E402


class SegmentationPreprocessTest(unittest.TestCase):
    def test_preprocess_returns_requested_tensor_shape(self) -> None:
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        tensor = preprocess_rgb(image, width=12, height=8)
        self.assertEqual(tuple(tensor.shape), (3, 8, 12))
        self.assertTrue(np.isfinite(tensor.numpy()).all())

    def test_preprocess_rejects_grayscale(self) -> None:
        with self.assertRaisesRegex(ValueError, "BGR image"):
            preprocess_rgb(np.zeros((20, 30), dtype=np.uint8), 12, 8)


if __name__ == "__main__":
    unittest.main()
