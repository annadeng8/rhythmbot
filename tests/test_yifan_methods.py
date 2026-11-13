import unittest
import numpy as np
from pydrake.all import RigidTransform,RotationMatrix,Simulator
from rhythmbot.config import stylus_config
from rhythmbot.cap.api import RobotAPI
from rhythmbot.planner import R_DOWN
from rhythmbot.chart import Chart
from rhythmbot.play import build

class Hover:
    def __init__(self,cfg,hand):self.api=RobotAPI(cfg,hand)
    def plan(self,notes,t,X,w):
        k=self.api.current_keyframe(t,X,w)
        return self.api.make_plan([k,self.api.keyframe(t+.1,k.x,k.z)])

class MethodTests(unittest.TestCase):
    def test_cap_enforces_single_tip_geometry(self):
        cfg=stylus_config()
        for hand in (0,1):
            api=RobotAPI(cfg,hand)
            plan=api.make_plan([api.keyframe(0,.6,.4,.1,ref=1,rise=.01)])
            self.assertEqual(plan.width_at(0),0)
            self.assertLessEqual(plan.tip_pose(0).translation()[2],api.screen_z+.1)
            expected=RotationMatrix.MakeYRotation(api.tilt)@R_DOWN
            np.testing.assert_allclose(plan.tip_pose(0).rotation().matrix(),expected.matrix())

    def test_two_independent_custom_policies_build_and_run(self):
        cfg=stylus_config();policies=[Hover(cfg,k) for k in (0,1)]
        diagram,robot,brain,judge=build(Chart([]),cfg,'oracle',None,None,policy=policies)
        sim=Simulator(diagram);robot.initialize(sim.get_mutable_context());sim.AdvanceTo(.05)
        self.assertIs(brain.policies[0],policies[0]);self.assertIs(brain.policies[1],policies[1])
        self.assertEqual(len(judge.down_since),2)

    def test_bc_single_tip_plan_has_fixed_tilt(self):
        from rhythmbot.bc.models import BCModel
        from rhythmbot.bc.policy import BCPolicy
        cfg=stylus_config()
        for hand in (0,1):
            p=BCPolicy(BCModel('state','mlp',16,32),cfg,hand=hand)
            api=RobotAPI(cfg,hand)
            X=RigidTransform(RotationMatrix.MakeYRotation(api.tilt)@R_DOWN,[-.18 if hand==0 else .18,api.hit_y,.14])
            plan=p.plan([],0,X,0)
            self.assertEqual(plan.width_at(.1),0)
            np.testing.assert_allclose(plan.tip_pose(.1).rotation().matrix(),X.rotation().matrix(),atol=1e-12)

if __name__=='__main__':unittest.main()
