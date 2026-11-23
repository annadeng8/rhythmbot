"""Check the older-Torch eager fallback against the original TorchScript export.

Run on the newer-Torch host after sequential timing trials have finished.
Inputs and fallback predictions are produced by check_neural_transfer.py on
its older-Torch host. No study model or detector settings are changed.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from rhythmbot.train import NoteNet

p=argparse.ArgumentParser();p.add_argument('probes',type=Path);a=p.parse_args()
torch.set_num_threads(1)
model=Path('results/note_detector_v7.pt')
script=torch.jit.load(str(model),map_location='cpu').eval()
state=torch.load(model.with_suffix('.state.pt'),map_location='cpu',weights_only=True)
assert set(script.state_dict())==set(state)
assert all(torch.equal(script.state_dict()[k],state[k]) for k in state)
meta=json.loads(model.with_suffix('.json').read_text())
eager=NoteNet(meta['n_lanes'],meta['rows'],meta['k_slots']).eval()
eager.load_state_dict(state)
z=np.load(a.probes/'model-outputs.npz')
errors=[];local_errors=[]
with torch.no_grad():
 for rgb,remote in zip(z['inputs'],z['predictions']):
  x=torch.from_numpy(rgb.copy()).permute(2,0,1).float().div(255)[None]
  expected=script(x).numpy();current=eager(x).numpy()
  np.testing.assert_allclose(current,expected,atol=2e-5,rtol=2e-5)
  np.testing.assert_allclose(remote,expected,atol=2e-5,rtol=2e-5)
  errors.append(float(np.max(np.abs(remote-expected))))
  local_errors.append(float(np.max(np.abs(current-expected))))
result=dict(tensors_exact=True,tensors=len(state),frames=len(errors),
 torch_version=torch.__version__,max_cross_host_abs_error=max(errors),
 max_local_eager_script_abs_error=max(local_errors),atol=2e-5,rtol=2e-5,
 note='Checks fixed diagnostic inputs, not a proof of equality for every possible image.')
(a.probes/'backend-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
