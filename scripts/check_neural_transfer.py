"""Small predeclared diagnostic probes, separate from held-out policy scores.

The network sees either original straight-lane RGB, rectified minimal-Sekai
RGB, or zero RGB. The targets are used only when writing diagnostic results.
No retraining or hyperparameter selection is performed from these probes.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from rhythmbot.camera_vision import NeuralCameraDetector
from rhythmbot.perception import TorchDetector
from rhythmbot.profiles import config
from rhythmbot.chart import Chart,Note,TAP,HOLD,SLIDE
from rhythmbot.game import FrameRenderer
from rhythmbot.sekai_screen import SekaiScreen
from PIL import Image

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
cfg=config({'preset':'sekai-camera'});model='results/note_detector_v7.pt'
native=TorchDetector(model,cfg);rectified=NeuralCameraDetector(model,cfg,camera=False)
script=native.model.state_dict()
state=torch.load('results/note_detector_v7.state.pt',map_location='cpu',weights_only=True)
assert set(script)==set(state)
assert all(torch.equal(script[k],state[k]) for k in state)
rows=[]; inputs=[]; predictions=[]
for kind in [TAP,HOLD,SLIDE]:
 chart=Chart([Note(kind,1,3.,duration=0 if kind==TAP else .8,
                   waypoints=[(0,1),(.8,4)] if kind==SLIDE else [])])
 for t in [2.,2.6,3.2]:
  for mode in ['native','sekai','black']:
   if mode=='native':rgb=FrameRenderer(cfg).render(chart,t);d=native
   elif mode=='sekai':rgb=SekaiScreen(cfg).render(chart,t);d=rectified
   else:rgb=np.zeros((384,256,3),np.uint8);d=native
   found=d.detect(rgb,t)
   actual=rectified.rectifier.rectify(rgb) if mode=='sekai' else rgb
   inputs.append(actual)
   with torch.no_grad():
    predictions.append(d.model(torch.from_numpy(actual.copy()).permute(2,0,1).float().div(255)[None]).numpy())
   rows.append(dict(target_kind=kind,time=t,input=mode,detections=[vars(n) for n in found]))
   if mode!='black' and t==2.6:
    shown=rectified.rectifier.rectify(rgb) if mode=='sekai' else rgb
    Image.fromarray(shown).save(a.out/f'{mode}-{kind}.png')
np.savez_compressed(a.out/'model-outputs.npz', inputs=np.stack(inputs), predictions=np.stack(predictions))
(a.out/'probes.json').write_text(json.dumps(dict(loaded_weights_exact=True,tensors_checked=len(state),torch_version=torch.__version__,
    backend=type(native.model).__name__,
    note='Probes are not held-out task results. Native direct RGB is a positive control, not a camera ablation.',rows=rows),indent=2)+'\n')
