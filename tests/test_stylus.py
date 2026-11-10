import unittest
import numpy as np
from pydrake.all import RigidTransform, RotationMatrix
from rhythmbot.config import stylus_config
from rhythmbot.robot import touch_width
from rhythmbot.judge import Judge
from rhythmbot.chart import Chart, Note, TAP
from rhythmbot.planner import ClassicalPlanner, TrackedNote, R_DOWN


class StylusTests(unittest.TestCase):
    def test_exactly_two_contacts(self):
        cfg=stylus_config()
        self.assertEqual(touch_width(cfg),8)
        judge=Judge(Chart([]),cfg)
        self.assertEqual(len(judge.down_since),2)

    def test_single_tip_has_no_chord_action(self):
        cfg=stylus_config();planner=ClassicalPlanner(cfg,0)
        notes=[TrackedNote(i,TAP,lane,1) for i,lane in enumerate([0,1])]
        self.assertTrue(planner._options(notes))
        self.assertTrue(all(len(a.notes)==1 for a in planner._options(notes)))
        self.assertTrue(all(a.width==0 and a.side==0 for a in planner._options(notes)))

    def test_three_simultaneous_notes_cannot_all_score(self):
        cfg=stylus_config();chart=Chart([Note(TAP,l,1) for l in [1,2,4]])
        judge=Judge(chart,cfg);judge.down_since=[1.,1.]
        touches=[(True,np.array([cfg.layout.lane_x(l),cfg.layout.hit_y,.1])) for l in [1,4]]
        for n in chart.notes: judge._judge_start(n,judge.results[n.id],1.,touches)
        judge.finalize(2.)
        self.assertEqual(sum(bool(r.completed) for r in judge.results.values()),2)

    def test_fixed_tilt_does_not_accumulate_on_replan(self):
        cfg=stylus_config()
        for hand in (0,1):
            p=ClassicalPlanner(cfg,hand)
            R=RotationMatrix.MakeYRotation(np.deg2rad(cfg.gripper.tilt_degrees)*(-1 if hand==0 else 1))@R_DOWN
            X=RigidTransform(R,[p.rest_x(),p.hit_y,p.hover_z])
            for i in range(10):
                plan=p.plan([],i*.1,X,0)
                X=plan.tip_pose(i*.1+.05)
                np.testing.assert_allclose(X.rotation().matrix(),R.matrix(),atol=1e-12)
                self.assertEqual(plan.width_at(i*.1),0)

if __name__=='__main__': unittest.main()
