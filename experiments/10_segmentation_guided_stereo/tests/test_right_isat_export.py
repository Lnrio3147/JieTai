from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from export_right_isat_annotations import rasterize_isat  # noqa: E402


def polygon(category: str, layer: float, points: list[list[float]]) -> dict:
    return {"category": category, "layer": layer, "segmentation": points}


class RightIsatExportTest(unittest.TestCase):
    def test_background_layer_erases_foreground(self) -> None:
        annotation = {
            "info": {"width": 20, "height": 20},
            "objects": [
                polygon("workpiece", 1, [[1, 1], [18, 1], [18, 18], [1, 18]]),
                polygon("__background__", 2, [[7, 7], [12, 7], [12, 12], [7, 12]]),
            ],
        }
        mask, stats = rasterize_isat(annotation, Path("sample.json"), 20, 20)
        self.assertEqual(int(mask[3, 3]), 255)
        self.assertEqual(int(mask[9, 9]), 0)
        self.assertEqual(stats["foreground_object_count"], 1)
        self.assertEqual(stats["background_erase_object_count"], 1)

    def test_layers_are_sorted_before_rendering(self) -> None:
        annotation = {
            "info": {"width": 20, "height": 20},
            "objects": [
                polygon("__background__", 2, [[7, 7], [12, 7], [12, 12], [7, 12]]),
                polygon("workpiece", 1, [[1, 1], [18, 1], [18, 18], [1, 18]]),
            ],
        }
        mask, _ = rasterize_isat(annotation, Path("sample.json"), 20, 20)
        self.assertEqual(int(mask[9, 9]), 0)

    def test_rejects_unknown_category(self) -> None:
        annotation = {
            "info": {"width": 20, "height": 20},
            "objects": [
                polygon("unknown", 1, [[1, 1], [18, 1], [18, 18], [1, 18]])
            ],
        }
        with self.assertRaisesRegex(ValueError, "Unexpected category"):
            rasterize_isat(annotation, Path("sample.json"), 20, 20)


if __name__ == "__main__":
    unittest.main()
