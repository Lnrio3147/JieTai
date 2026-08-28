from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from reconstruct import validate_preflight_report  # noqa: E402


class ReconstructPreflightTest(unittest.TestCase):
    def test_blocked_report_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            report = root / "preflight.json"
            report.write_text(
                json.dumps(
                    {
                        "completed": True,
                        "manifest": str(manifest),
                        "assessment": {
                            "reconstruction_ready": False,
                            "blockers": ["same_static_object_not_confirmed"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "same_static_object"):
                validate_preflight_report(manifest, report)

    def test_ready_matching_report_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            report = root / "preflight.json"
            report.write_text(
                json.dumps(
                    {
                        "completed": True,
                        "manifest": str(manifest),
                        "assessment": {"reconstruction_ready": True},
                    }
                ),
                encoding="utf-8",
            )
            result = validate_preflight_report(manifest, report)
            self.assertTrue(result["assessment"]["reconstruction_ready"])


if __name__ == "__main__":
    unittest.main()
