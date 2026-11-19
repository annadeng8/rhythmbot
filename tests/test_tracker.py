"""Slide path from the vision model's active slot; run in an environment with Drake and torch."""
import unittest

import numpy as np

from rhythmbot.brain import RhythmBrain
from rhythmbot.chart import SLIDE, Chart, Note
from rhythmbot.config import DEFAULT
from rhythmbot.game import FrameRenderer
from rhythmbot.perception import DetectedNote
from rhythmbot.planner import TrackedNote


class ActivePathTests(unittest.TestCase):
    def setUp(self):
        # Lanes 0 -> 2 -> 5 over one second, from t = 1.
        self.note = Note(SLIDE, 0, 1.0, 1.0, [(0.0, 0.0), (0.5, 2.0), (1.0, 5.0)])
        self.chart = Chart([self.note])

    def test_rest_lanes_label_what_is_left(self):
        (d,) = FrameRenderer(DEFAULT).label(self.chart, 1.5)
        self.assertTrue(d["active"])
        self.assertEqual(d["lane_now"], 2.0)
        np.testing.assert_allclose(d["rest_lanes"], [2.0, 2.75, 3.5, 4.25, 5.0])

    def test_active_targets_are_offsets_from_the_slot_lane(self):
        from rhythmbot.train import K_SLOTS, build_targets

        notes = FrameRenderer(DEFAULT).label(self.chart, 1.5)
        tgt = build_targets(notes, 6, DEFAULT.layout.approach_time)
        np.testing.assert_allclose(tgt[2, K_SLOTS, 3:8] * 5, [0.0, 0.75, 1.5, 2.25, 3.0], atol=1e-6)

    def test_tracker_takes_the_rest_of_the_path_from_the_active_slot(self):
        # The forecast from up the highway ends two lanes short.
        n = TrackedNote(0, SLIDE, 0.0, 1.0, 1.0, waypoints=[(0.0, 0.0), (0.5, 1.5), (1.0, 3.0)])
        # At t = 1.6 the active slot sees 0.38 s left (the end 20 ms early).
        d = DetectedNote(SLIDE, 3.0, 1.6, 0.38, active=True,
                         waypoints=[(0.0, 2.6), (0.095, 3.2), (0.19, 3.8), (0.285, 4.4), (0.38, 5.0)])
        RhythmBrain._replace_rest(n, d, 1.6)
        self.assertEqual(n.waypoints[:2], [(0.0, 0.0), (0.5, 1.5)])
        self.assertAlmostEqual(n.waypoints[2][0], 0.6)
        self.assertAlmostEqual(n.waypoints[-1][0], n.duration)
        self.assertEqual(n.lane_at_offset(n.duration), 5.0)


if __name__ == "__main__":
    unittest.main()
