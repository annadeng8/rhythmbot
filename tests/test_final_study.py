import tempfile
import unittest
from pathlib import Path
from collections import Counter
from rhythmbot.final_study import prepare, cfg_and_input, sha
from rhythmbot.chart import Chart

class FinalStudyTests(unittest.TestCase):
    def test_split_counts_and_disjoint_semantic_charts(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);p=prepare(out)
            counts=Counter(e['split'] for e in p['entries'])
            self.assertEqual(counts,dict(train=48,validation=6,test_iid=6,test_speed=2,test_song=3))
            self.assertEqual(len(p['cap_feedback_charts']),6)
            self.assertTrue(all('/train/' in x for x in p['cap_feedback_charts']))
            self.assertTrue(all(sha(out/e['path'])==e['sha256'] for e in p['entries']))
            self.assertEqual(prepare(out),p)
            for e in p['entries']:
                if e['split']=='test_song':
                    self.assertLessEqual(Chart.load(out/e['path']).end_time,32.5)

    def test_condition_changes_observation_not_robot(self):
        a,_,_=cfg_and_input('oracle')
        for c in ('screen','camera'):
            b,det,view=cfg_and_input(c)
            self.assertEqual(a,b)
            self.assertEqual(det,'concert')
            self.assertEqual(view,c)

if __name__=='__main__': unittest.main()
