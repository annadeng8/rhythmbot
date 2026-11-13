"""Behavioral regression tests; run in an environment with Drake."""
import unittest
import numpy as np
from rhythmbot.chart import Chart, Note, SLIDE
from rhythmbot.config import two_arm_config
from rhythmbot.judge import Judge
from rhythmbot.handoff import hand_notes
from rhythmbot.planner import TrackedNote, ClassicalPlanner


class SlideTests(unittest.TestCase):
    def run_slide(self, contact, end=2.01):
        cfg=two_arm_config()
        note=Note(SLIDE,1,1,1,[(0,1),(.5,3),(1,4)])
        chart=Chart([note]);j=Judge(chart,cfg);r=j.results[0]
        j.down_since[0]=1.
        start=[(True,np.array([cfg.layout.lane_x(1),cfg.layout.hit_y,cfg.layout.screen_z]))]
        j._judge_start(note,r,1.,start)
        self.assertIsNone(r.judgment)
        for t in np.arange(1.,end,1/cfg.judge_hz):
            down,lane=contact(float(t),note.lane_at(float(t)))
            touches=[(down,np.array([cfg.layout.lane_x(lane),cfg.layout.hit_y,cfg.layout.screen_z]))]
            j._track_body(note,r,float(t),touches)
            if r.completed is not None: break
        return j,r

    def test_good_path_only_scores_after_tail(self):
        j,r=self.run_slide(lambda t,l:(True,l),end=1.99)
        self.assertIsNone(r.judgment);self.assertIsNone(r.completed)
        j,r=self.run_slide(lambda t,l:(True,l))
        self.assertTrue(r.completed);self.assertEqual(r.judgment,'perfect')

    def test_head_only_is_miss_and_zero_score(self):
        j,r=self.run_slide(lambda t,l:(t<1.02,l))
        self.assertFalse(r.completed);self.assertEqual(r.judgment,'miss')
        self.assertEqual(j.report()['score'],0)

    def test_contact_break_fails_even_with_correct_tail(self):
        _,r=self.run_slide(lambda t,l:(not 1.3<t<1.4,l))
        self.assertFalse(r.completed);self.assertEqual(r.failure_reason,'contact_break')

    def test_missing_tail_fails(self):
        _,r=self.run_slide(lambda t,l:(t<1.96,l))
        self.assertFalse(r.completed);self.assertEqual(r.failure_reason,'tail_missing')

    def test_wrong_tail_lane_fails(self):
        _,r=self.run_slide(lambda t,l:(True,l if t<1.96 else 0))
        self.assertFalse(r.completed);self.assertEqual(r.failure_reason,'tail_missing')

    def test_truncated_run_does_not_award_perfect(self):
        j,r=self.run_slide(lambda t,l:(True,l),end=1.5)
        j.finalize(1.5)
        self.assertEqual(r.judgment,'miss');self.assertEqual(r.failure_reason,'truncated')

    def test_both_crossing_directions_stay_reachable(self):
        cfg=two_arm_config()
        for path in ([(0,1),(.5,3),(1,4)],[(0,5),(.6,3),(1.2,1),(1.6,2)]):
            n=TrackedNote(1,SLIDE,path[0][1],3,path[-1][0],waypoints=path)
            hands=[hand_notes([n],h,cfg) for h in range(2)]
            self.assertTrue(all(hands))
            for h in range(2):
                planner=ClassicalPlanner(cfg,h)
                for segment in hands[h]:
                    self.assertTrue(planner._reachable(planner._single(segment)))
            for t in np.linspace(0,n.duration,200):
                target=n.lane_at_offset(t)
                distances=[abs(s.lane_at_offset(t+3-s.hit_time)-target)
                    for group in hands for s in group if s.hit_time<=t+3<=s.end_time]
                self.assertTrue(distances)
                self.assertLessEqual(min(distances),cfg.timing.slide_tolerance_lanes)

if __name__=='__main__': unittest.main()
