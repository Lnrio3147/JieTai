from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from preflight_sequence import assess_claims  # noqa: E402


def cloud_summary(diagonal: float = 10.0) -> dict:
    return {"robust_diagonal": diagonal, "finite_fraction": 1.0}


class PreflightAssessmentTest(unittest.TestCase):
    def test_unconfirmed_legacy_data_is_blocked(self) -> None:
        manifest = {
            "units": "cm",
            "acquisition": {
                "same_static_object_confirmed": False,
                "common_calibration_confirmed": True,
            },
            "views": [
                {"pose": None, "azimuth_deg": None, "elevation_deg": None},
                {"pose": None, "azimuth_deg": None, "elevation_deg": None},
            ],
        }
        result = assess_claims(
            manifest,
            [cloud_summary(), cloud_summary()],
            [{"accepted": True}],
        )
        self.assertFalse(result["reconstruction_ready"])
        self.assertIn("same_static_object_not_confirmed", result["blockers"])

    def test_confirmed_diverse_views_allow_coverage_claim(self) -> None:
        manifest = {
            "units": "mm",
            "acquisition": {
                "same_static_object_confirmed": True,
                "common_calibration_confirmed": True,
            },
            "views": [
                {"pose": None, "azimuth_deg": 0.0, "elevation_deg": 20.0},
                {"pose": None, "azimuth_deg": 30.0, "elevation_deg": 20.0},
            ],
        }
        result = assess_claims(
            manifest,
            [cloud_summary(), cloud_summary()],
            [{"accepted": True}],
        )
        self.assertTrue(result["reconstruction_ready"])
        self.assertTrue(result["coverage_claim_ready"])

    def test_skipped_registration_is_not_reconstruction_ready(self) -> None:
        manifest = {
            "units": "mm",
            "acquisition": {
                "same_static_object_confirmed": True,
                "common_calibration_confirmed": True,
            },
            "views": [
                {"pose": None, "azimuth_deg": 0.0, "elevation_deg": 0.0},
                {"pose": None, "azimuth_deg": 20.0, "elevation_deg": 0.0},
            ],
        }
        result = assess_claims(manifest, [cloud_summary(), cloud_summary()], [])
        self.assertIn("required_registration_not_completed", result["blockers"])


if __name__ == "__main__":
    unittest.main()
