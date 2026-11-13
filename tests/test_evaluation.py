"""Evaluation framework: failure causes and report-consistency warnings."""

import unittest

from rhythmbot.chart import HOLD, TAP, Chart, Note
from rhythmbot.config import DEFAULT, two_arm_config
from rhythmbot.evaluate import consistency_notes
from rhythmbot.failures import ChartDemands


def demands(notes, cfg=DEFAULT):
    contacts = cfg.gripper.contacts_per_arm * cfg.arms.n
    chart = Chart(notes, name="t")
    return chart, ChartDemands(chart, cfg, contacts)


class BeyondTheRobotTests(unittest.TestCase):
    def test_single_tap_is_playable(self):
        chart, d = demands([Note(TAP, 2, 1.0)])
        self.assertIsNone(d.reason(chart.notes[0]))

    def test_three_lanes_at_once_exceed_two_fingertips(self):
        chart, d = demands([Note(TAP, 1, 1.0), Note(TAP, 2, 1.0), Note(TAP, 3, 1.0)])
        self.assertIn("needs 3 lanes", d.reason(chart.notes[0]))

    def test_tap_during_another_lanes_hold_counts_the_hold(self):
        chart, d = demands([Note(HOLD, 1, 1.0, 1.0), Note(TAP, 2, 1.5), Note(TAP, 3, 1.5)])
        tap = [n for n in chart.notes if n.lane == 2][0]
        self.assertIn("needs 3 lanes", d.reason(tap))

    def test_adjacent_chord_fits_one_gripper(self):
        chart, d = demands([Note(TAP, 1, 1.0), Note(TAP, 2, 1.0)])
        self.assertIsNone(d.reason(chart.notes[0]))

    def test_chord_wider_than_the_gripper(self):
        # Lanes 0 and 5 are 25 cm apart; one gripper opens to 13 cm.
        chart, d = demands([Note(TAP, 0, 1.0), Note(TAP, 5, 1.0)])
        self.assertEqual(d.reason(chart.notes[0]), "chord wider than the gripper opens")

    def test_fast_lane_change(self):
        chart, d = demands([Note(TAP, 1, 1.0), Note(TAP, 4, 1.1)])
        second = [n for n in chart.notes if n.lane == 4][0]
        self.assertIn("lane change", d.reason(second))

    def test_slow_lane_change_is_playable(self):
        chart, d = demands([Note(TAP, 1, 1.0), Note(TAP, 4, 1.5)])
        second = [n for n in chart.notes if n.lane == 4][0]
        self.assertIsNone(d.reason(second))

    def test_two_arms_have_four_fingertips(self):
        cfg = two_arm_config()
        chart, d = demands([Note(TAP, 0, 1.0), Note(TAP, 1, 1.0), Note(TAP, 4, 1.0)], cfg)
        self.assertNotIn("needs", d.reason(chart.notes[0]) or "")


class ConsistencyTests(unittest.TestCase):
    def prov(self, tol=0.35, commit="abc", model="m1"):
        return {"provenance": {"judge": {"slide_tolerance_lanes": tol}, "commit": commit,
                               "uncommitted_code": False, "model_sha256": model}}

    def test_same_settings_no_warning(self):
        self.assertEqual(consistency_notes([self.prov(), self.prov()]), [])

    def test_mixed_judges_are_flagged(self):
        notes = consistency_notes([self.prov(0.35), self.prov(0.6)])
        self.assertTrue(any("judge settings" in n for n in notes))

    def test_reports_without_provenance_are_flagged(self):
        notes = consistency_notes([self.prov(), {"chart": "x"}])
        self.assertTrue(any("no provenance" in n for n in notes))

    def test_different_commits_and_models_are_flagged(self):
        notes = consistency_notes([self.prov(commit="a", model="m1"), self.prov(commit="b", model="m2")])
        self.assertTrue(any("commits" in n for n in notes))
        self.assertTrue(any("models" in n for n in notes))


if __name__ == "__main__":
    unittest.main()
