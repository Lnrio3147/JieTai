from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from prepare_stereo_segmentation_dataset import (  # noqa: E402
    project_mask_to_right,
    read_disparity,
)


class StereoDatasetProjectionTest(unittest.TestCase):
    def test_forward_projection_uses_left_minus_disparity(self) -> None:
        mask = np.zeros((7, 8), dtype=bool)
        mask[3, 4:6] = True
        disparity = np.full((7, 8), 2.0, dtype=np.float32)
        projected, diagnostics = project_mask_to_right(mask, disparity)
        self.assertTrue(np.all(projected[3, 2:4] == 255))
        self.assertEqual(int((projected > 0).sum()), 2)
        self.assertEqual(diagnostics["valid_fraction"], 1.0)
        self.assertEqual(diagnostics["inside_fraction"], 1.0)

    def test_little_endian_pfm_is_read_and_flipped(self) -> None:
        expected = np.asarray(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "disparity.pfm"
            with path.open("wb") as stream:
                stream.write(b"Pf\n3 2\n-1.0\n")
                np.flipud(expected).astype("<f4").tofile(stream)
            np.testing.assert_allclose(read_disparity(path), expected)


if __name__ == "__main__":
    unittest.main()
