import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("run_cross_dataset.py")
SPEC = importlib.util.spec_from_file_location("run_cross_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CrossDatasetTest(unittest.TestCase):
    def test_fdjyp3_probability_name(self):
        self.assertEqual(
            MODULE.fdjyp3_probability_name("202506281603-0001"),
            "fdjyp_3_1_202506281603_0001.npy",
        )

    def test_reference_metrics_respect_region(self):
        prediction = np.array([[1.0, 5.0], [4.0, 8.0]], dtype=np.float32)
        reference = np.array([[1.0, 2.0], [4.0, 0.0]], dtype=np.float32)
        region = np.array([[True, False], [True, False]])
        metrics = MODULE.compute_reference_metrics(prediction, reference, region)
        self.assertEqual(metrics["reference_pixels"], 2)
        self.assertAlmostEqual(metrics["reference_retained_pct"], 200.0 / 3.0)
        self.assertAlmostEqual(metrics["epe_px"], 0.0)
        self.assertAlmostEqual(metrics["bad3_pct"], 0.0)

    def test_risk_flags_are_triage(self):
        base = {
            "fused_foreground_fraction": 0.5,
            "semantic_uncertain_fraction": 0.02,
            "changed_fraction": 0.01,
            "border_touch_sides": 0,
            "depth_reliability": 0.8,
        }
        self.assertEqual(MODULE.classify_review_risk(base)[0], "low_flag")
        review = dict(base, fused_foreground_fraction=0.85)
        self.assertEqual(MODULE.classify_review_risk(review)[0], "review")
        high = dict(base, fused_foreground_fraction=0.95)
        self.assertEqual(MODULE.classify_review_risk(high)[0], "high_flag")

    def test_resize_mask_is_nearest_and_boolean(self):
        mask = np.array([[False, True], [False, True]])
        resized = MODULE.resize_mask(mask, (4, 4))
        self.assertEqual(resized.dtype, np.bool_)
        self.assertTrue(np.all(resized[:, :2] == 0))
        self.assertTrue(np.all(resized[:, 2:] == 1))

    def test_prepare_output_only_replaces_results_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            outside = Path(temporary) / "existing"
            outside.mkdir()
            with self.assertRaises(ValueError):
                MODULE.prepare_output(outside, overwrite=True)


if __name__ == "__main__":
    unittest.main()
