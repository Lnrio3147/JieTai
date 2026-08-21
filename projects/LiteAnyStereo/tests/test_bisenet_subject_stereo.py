import importlib.util
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "tools/evaluate_bisenet_subject_stereo.py"
SPEC = importlib.util.spec_from_file_location("evaluate_bisenet_subject_stereo", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BiSeNetSubjectStereoTest(unittest.TestCase):
    def test_scene_id_matches_tradition_name(self):
        self.assertEqual(
            MODULE.scene_id("fdjyp_3_1_202506281603_0001"),
            "202506281603-0001",
        )

    def test_subject_metrics_use_only_selected_pixels(self):
        prediction = np.array([[1.0, 10.0], [3.0, 4.0]], dtype=np.float32)
        reference = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32)
        subject = np.array([[True, False], [True, False]])
        metrics = MODULE.compute_metrics(prediction, reference, subject)
        self.assertEqual(metrics["valid_pixels"], 2)
        self.assertAlmostEqual(metrics["epe"], 0.5)
        self.assertAlmostEqual(metrics["bad1"], 0.0)

    def test_comparison_visualization_writes_six_panels(self):
        with tempfile.TemporaryDirectory() as directory:
            left = np.zeros((32, 24, 3), dtype=np.uint8)
            prediction = np.full((32, 24), 20.0, dtype=np.float32)
            reference = np.full((32, 24), 18.0, dtype=np.float32)
            subject = np.ones((32, 24), dtype=bool)
            output = Path(directory) / "comparison.jpg"
            MODULE.save_comparison(
                output,
                left,
                prediction,
                reference,
                subject,
                192,
            )
            image = cv2.imread(str(output), cv2.IMREAD_COLOR)
            self.assertIsNotNone(image)
            self.assertEqual(image.shape[:2], (818, 768))


if __name__ == "__main__":
    unittest.main()
