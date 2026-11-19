"""Export native Meshcat frames from a frozen camera-policy Drake rollout.

The recorded controller receives camera RGB, matching the quantitative run.
Frame HTML is an intermediate; use Chrome to rasterize it without replacing
Meshcat's robot materials or lighting. Run outside the benchmark worker pool.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from pydrake.all import Meshcat, Simulator
from rhythmbot.chart import Chart
from rhythmbot.final_study import cfg_and_input
from rhythmbot.play import build
from rhythmbot.policies import make_policy

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--chart',type=Path,required=True)
ap.add_argument('--policy',required=True)
ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--fps',type=int,default=8)
a=ap.parse_args()
if a.policy.startswith('bc:'):
    import torch
    torch.set_num_threads(1);torch.manual_seed(0)
chart=Chart.load(a.chart);cfg,detector,view=cfg_and_input('camera')
meshcat=Meshcat()
meshcat.SetCamera(Meshcat.PerspectiveCamera(fov=50))
center=cfg.layout.screen_center()
meshcat.SetCameraPose(center+np.array([1.05,-1.4,1.05]),center+np.array([0,.18,.22]))
policies=[make_policy(a.policy,cfg,hand=k) for k in (0,1)]
diagram,robot,brain,judge=build(chart,cfg,detector,None,meshcat,policies,view=view)
sim=Simulator(diagram);context=sim.get_mutable_context();robot.initialize(context)
a.out.mkdir(parents=True,exist_ok=True)
end=chart.end_time+1.
# Capture the whole chart, including approach and completion.
times=np.arange(0,end,1/a.fps).tolist()+[end]
for i,t in enumerate(times):
    sim.AdvanceTo(t);diagram.ForcedPublish(context);meshcat.Flush()
    html=meshcat.StaticHtml().replace('</head>','<style>.dg { display:none !important; }</style></head>')
    (a.out/f'{i:04d}.html').write_text(html)
    if i%16==0:print(i,len(times),flush=True)
judge.finalize(end)
report=judge.report(extra={'config':asdict(cfg),'sim_seconds':end,'detector':detector,'view':view})
report['policy_stats_by_hand']=[p.stats() if hasattr(p,'stats') else {} for p in policies]
report.update(policy=a.policy,fps=a.fps,times=times,chart_sha256=hashlib.sha256(a.chart.read_bytes()).hexdigest(),
    observer='native Meshcat/default materials and lighting/minimal Sekai screen',
    control_input='Drake camera RGB with color/shape detector; same as camera benchmark')
if ':' in a.policy:report['policy_sha256']=hashlib.sha256(Path(a.policy.split(':',1)[1]).read_bytes()).hexdigest()
(a.out/'replay.json').write_text(json.dumps(report,indent=2)+'\n')
