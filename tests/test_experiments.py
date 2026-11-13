import tempfile
import unittest
from pathlib import Path
from rhythmbot.experiments import make_chart,prepare
from rhythmbot.config import stylus_config
from rhythmbot.handoff import hand_notes
from rhythmbot.planner import TrackedNote
from rhythmbot.chart import SLIDE
from dataclasses import replace

class ExperimentTests(unittest.TestCase):
    def test_chart_reproducible_and_two_tool_pairs_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            m=prepare(Path(tmp));self.assertEqual(len(m['jobs']),61)
            for job in m['jobs']:
                if job['tool']=='two_finger':
                    match=[j for j in m['jobs'] if j['chart']==job['chart'] and j['tool']=='stylus' and j['detector']=='oracle' and j['handoff']]
                    self.assertEqual(len(match),1)
                    self.assertEqual(match[0]['chart_hash'],job['chart_hash'])
        a=make_chart('crossing',101,.4);b=make_chart('crossing',101,.4)
        self.assertEqual([n.to_dict() for n in a.notes],[n.to_dict() for n in b.notes])

    def test_no_handoff_ablation_only_assigns_start_hand(self):
        cfg=stylus_config();cfg=replace(cfg,arms=replace(cfg.arms,slide_handoff=False))
        note=TrackedNote(0,SLIDE,1,3,1,waypoints=[(0,1),(1,4)])
        self.assertEqual(hand_notes([note],0,cfg),[note])
        self.assertEqual(hand_notes([note],1,cfg),[])

if __name__=='__main__': unittest.main()
