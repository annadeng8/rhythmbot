"""How far v7's read-off rest of a slide's path is from the true path.

    python scripts/diagnose_slide_path.py results/yifan-final-v1/charts/test_iid/holds-s71000-x1.1287.json

Plays the chart on two gripper arms (plain game screen, v7) and logs the path
the tracker holds after each `_replace_rest`, then prints the lane error at
the current moment and 33/100/200 ms ahead.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from rhythmbot import brain
from rhythmbot.chart import Chart
from rhythmbot.config import two_arm_config
from rhythmbot.play import build
from pydrake.all import Simulator

chart = Chart.load(sys.argv[1])
log = []
orig = brain.RhythmBrain._replace_rest
def logged(n, d, t):
    orig(n, d, t)
    log.append((t, n.hit_time, list(n.waypoints), list(d.waypoints), d.duration, n.end_time))
brain.RhythmBrain._replace_rest = staticmethod(logged)
cfg = two_arm_config()
diagram, robot, b, judge = build(chart, cfg, 'torch', 'results/note_detector_v7.pt', None, None)
sim = Simulator(diagram); robot.initialize(sim.get_mutable_context()); sim.Initialize()
sim.AdvanceTo(chart.end_time + 0.5)
slides = [n for n in chart.notes if n.kind == 'slide']
def true_note(hit):
    return min(slides, key=lambda n: abs(n.time - hit))
def lane(wps, off):
    o = [w[0] for w in wps]; l = [w[1] for w in wps]
    return float(np.interp(off, o, l))
rows = []
for t, hit, wps, raw, dur, end in log:
    n = true_note(hit)
    errs = {}
    for la in (0.0, 0.033, 0.1, 0.2):
        tt = t + la
        if tt > n.end_time: continue
        errs[la] = abs(lane(wps, tt - hit) - n.lane_at(tt))
    rows.append(dict(t=round(t, 3), id=n.id, errs=errs, end_err=abs(wps[-1][1] - n.lane_at(n.end_time)),
                     raw=[(round(a, 2), round(b, 2)) for a, b in raw], dur=round(dur, 2), true_left=round(n.end_time - t, 2)))
for r in rows[::6]: print(r)
for la in (0.0, 0.033, 0.1, 0.2):
    e = [r['errs'][la] for r in rows if la in r['errs']]
    print(f'lookahead {la}s: mean lane err {np.mean(e):.2f}, p90 {np.quantile(e, .9):.2f}, max {max(e):.2f}')
