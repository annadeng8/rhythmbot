import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rhythmbot.profiles import resolve, config
from rhythmbot.evaluate import _run_one, consistency_notes, evaluate
from rhythmbot import splits

class HandoffTests(unittest.TestCase):
    def test_complete_preset(self):
        self.assertEqual(resolve('sekai-camera'), (2,'camera',('concert',),True))
        c=config({'preset':'sekai-camera'})
        self.assertEqual(c.arms.n,2)
        self.assertEqual(c.gripper.contacts_per_arm,1)
        self.assertEqual((c.camera.width,c.camera.height),(960,720))

    def test_no_silent_override(self):
        for kwargs in [dict(arms=1),dict(detectors=('oracle',)),dict(view='screen')]:
            with self.assertRaises(ValueError):resolve('sekai-camera',**kwargs)

    def test_committed_chart_sets(self):
        self.assertEqual(len(splits.charts('bimanual-test')),8)
        self.assertEqual(len(splits.charts('bimanual-smoke')),2)

    def test_two_custom_policy_instances_reach_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            job=dict(policy='cap:unused.py',chart=splits.SPLITS['bimanual-smoke'][0],
                split='bimanual-smoke',out=str(Path(tmp)/'report.json'),preset='sekai-camera',
                view='camera',arms=2,stylus=True,detector='concert',time_limit=0)
            with patch('rhythmbot.policies.make_policy',side_effect=lambda *a,**kw:object()) as make, \
                 patch('rhythmbot.play.run',return_value={'notes':[]}) as run:
                rep=_run_one(job)
                self.assertEqual([c.kwargs['hand'] for c in make.call_args_list],[0,1])
                ps=run.call_args.kwargs['policy']
                self.assertIsNot(ps[0],ps[1])
                self.assertEqual(len(rep['policy_stats_by_hand']),2)

    def test_mixed_environments_warn(self):
        a={'provenance':{'judge':{},'view':'camera','contacts_per_arm':1}}
        b={'provenance':{'judge':{},'view':'screen','contacts_per_arm':2}}
        notes=consistency_notes([a,b])
        self.assertTrue(any('view' in x for x in notes))
        self.assertTrue(any('contacts_per_arm' in x for x in notes))

    def test_output_environment_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('rhythmbot.evaluate._run_one',return_value={}):
                evaluate(['classical'],'bimanual-smoke',tmp,workers=1,preset='sekai-camera')
                with self.assertRaisesRegex(ValueError,'another environment'):
                    evaluate(['classical'],'bimanual-smoke',tmp,workers=1,preset='sekai-screen')

if __name__=='__main__': unittest.main()
