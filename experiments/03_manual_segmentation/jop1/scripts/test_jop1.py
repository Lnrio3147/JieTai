import importlib.util
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parent / "run_test.py"
SPEC = importlib.util.spec_from_file_location("run_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class JopBiSeNetLas1Test(unittest.TestCase):
    def test_region_metrics_use_selected_reference_pixels(self):
        prediction = np.array([[1.0, 8.0], [4.0, 6.0]], dtype=np.float32)
        reference = np.array([[1.0, 2.0], [3.0, 6.0]], dtype=np.float32)
        region = np.array([[True, False], [True, False]])
        metrics = MODULE.compute_region_metrics(prediction, reference, region)
        self.assertEqual(metrics["reference_pixels"], 2)
        self.assertEqual(metrics["evaluated_pixels"], 2)
        self.assertAlmostEqual(metrics["prediction_coverage"], 1.0)
        self.assertAlmostEqual(metrics["epe_px"], 0.5)

    def test_region_metrics_report_missing_prediction_coverage(self):
        prediction = np.array([[1.0, np.nan]], dtype=np.float32)
        reference = np.array([[1.0, 2.0]], dtype=np.float32)
        metrics = MODULE.compute_region_metrics(prediction, reference)
        self.assertEqual(metrics["reference_pixels"], 2)
        self.assertEqual(metrics["evaluated_pixels"], 1)
        self.assertAlmostEqual(metrics["prediction_coverage"], 0.5)

    def test_comparison_writes_six_full_resolution_panels(self):
        with tempfile.TemporaryDirectory() as directory:
            left = np.zeros((40, 24, 3), dtype=np.uint8)
            raw = np.zeros((40, 24), dtype=bool)
            raw[5:35, 4:20] = True
            refined = raw.copy()
            disparity = np.full((40, 24), 30.0, dtype=np.float32)
            reference = np.full((40, 24), 29.0, dtype=np.float32)
            output = Path(directory) / "comparison.jpg"
            MODULE.save_comparison(
                output, left, raw, refined, disparity, reference, 192.0
            )
            image = cv2.imread(str(output), cv2.IMREAD_COLOR)
            self.assertIsNotNone(image)
            self.assertEqual(image.shape[:2], (80, 72))


if __name__ == "__main__":
    unittest.main()
