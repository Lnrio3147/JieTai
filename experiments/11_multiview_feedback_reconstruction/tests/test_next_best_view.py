from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from next_best_view import look_at_pose, rank_candidates, visible_fraction  # noqa: E402


class NextBestViewTest(unittest.TestCase):
    def test_look_at_pose_places_points_in_front(self) -> None:
        pose = look_at_pose(np.asarray((0.0, 0.0, -10.0)), np.zeros(3))
        fraction, count = visible_fraction(
            np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 0.0))), pose, 70.0, 55.0
        )
        self.assertEqual(count, 2)
        self.assertEqual(fraction, 1.0)

    def test_candidate_looking_away_scores_lower(self) -> None:
        points = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        good = look_at_pose(np.asarray((0.0, 0.0, -10.0)), np.zeros(3))
        bad = good.copy()
        bad[:3, :3] *= -1.0
        bad[3, 3] = 1.0
        ranked = rank_candidates(
            points,
            [{"id": "bad", "pose": bad}, {"id": "good", "pose": good}],
            [],
            object_diagonal=1.0,
        )
        self.assertEqual(ranked[0]["id"], "good")


if __name__ == "__main__":
    unittest.main()
