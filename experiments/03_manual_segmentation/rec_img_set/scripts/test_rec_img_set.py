import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parent / "run_test.py"
SPEC = importlib.util.spec_from_file_location("run_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RecImgSetBiSeNetLas1Test(unittest.TestCase):
    def test_scene_from_manual_name(self):
        self.assertEqual(
            MODULE.scene_from_manual_name("fdjyp_0_2_202506261657_0011"),
            "202506261657-0011",
        )

    def test_binary_metrics_match_known_masks(self):
        prediction = np.array([[1, 1], [0, 0]], dtype=bool)
        target = np.array([[1, 0], [1, 0]], dtype=bool)
        metrics = MODULE.binary_metrics(prediction, target)
        self.assertAlmostEqual(metrics["foreground_iou"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["dice"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)

    def test_difference_metrics_respect_region(self):
        first = np.array([[1.0, 10.0], [3.0, 4.0]], dtype=np.float32)
        second = np.array([[2.0, 2.0], [3.0, 8.0]], dtype=np.float32)
        region = np.array([[True, False], [True, False]])
        metrics = MODULE.difference_metrics(first, second, region)
        self.assertEqual(metrics["pixels"], 2)
        self.assertAlmostEqual(metrics["mae_px"], 0.5)

    def test_comparison_writes_six_panels(self):
        with tempfile.TemporaryDirectory() as directory:
            left = np.zeros((40, 24, 3), dtype=np.uint8)
            mask = np.zeros((40, 24), dtype=bool)
            mask[4:36, 3:21] = True
            las = np.full((40, 24), 20.0, dtype=np.float32)
            igev = np.full((40, 24), 22.0, dtype=np.float32)
            output = Path(directory) / "comparison.jpg"
            MODULE.save_comparison(output, left, mask, mask, las, igev, 192.0)
            image = cv2.imread(str(output), cv2.IMREAD_COLOR)
            self.assertIsNotNone(image)
            self.assertEqual(image.shape[:2], (80, 72))


if __name__ == "__main__":
    unittest.main()
