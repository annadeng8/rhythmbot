"""Save observed RGB and rectified network input from a real dual-arm rollout.

A four-second illustration, excluded from full-chart accuracy and timing counts.
"""
import argparse,json
from dataclasses import asdict
from pathlib import Path
import torch
from PIL import Image
from pydrake.all import Simulator
from rhythmbot.chart import Chart
from rhythmbot.profiles import config
from rhythmbot.play import build
from rhythmbot.policies import make_policy
from rhythmbot.experiments import source_hash

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(1);torch.manual_seed(0)
cfg=config({'preset':'sekai-camera'})
chart=Chart.load('results/yifan-final-v1/charts/test_iid/crossing-s71000-0.8s-x1.1287.json')
policies=[make_policy('classical',cfg,hand=k) for k in range(2)]
diagram,robot,brain,judge=build(chart,cfg,'concert-torch','results/note_detector_v7.pt',None,policies,view='camera')
sim=Simulator(diagram);robot.initialize(sim.get_mutable_context());sim.Initialize();sim.AdvanceTo(4.)
a.out.mkdir(parents=True,exist_ok=True)
Image.fromarray(brain.detector.last_rgb).save(a.out/'camera-rgb.png')
Image.fromarray(brain.detector.last_rectified).save(a.out/'network-input.png')
(a.out/'snapshot.json').write_text(json.dumps(dict(time=4.,source_sha256=source_hash(),config=asdict(cfg),
 vision=brain.detector.stats(),detections=[vars(n) for n in brain.detector.last_detections],
 note='Truncated diagnostic rollout, excluded from held-out scores and timing.'),indent=2)+'\n')
print(a.out)
