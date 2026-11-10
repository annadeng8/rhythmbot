"""Exercise exactly the BC operations used by the pilot, before training."""
import json
import torch
from rhythmbot.bc.models import BCModel
from rhythmbot.bc.features import OBS_DIM

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(.12)
p=torch.cuda.get_device_properties(0)
result={'gpu':p.name,'vram_bytes':p.total_memory,'shared_memory_per_block_bytes':p.shared_memory_per_block,'kernels':'PyTorch eager MLP/diffusion; no torch.compile/Triton','checks':[]}
for head in ('mlp','diffusion'):
    model=BCModel('state',head,16,256).cuda()
    batch={'obs':torch.randn(256,OBS_DIM,device='cuda'),'act':torch.randn(256,16,4,device='cuda')}
    loss=model.loss(batch);loss.backward();action=model.act(batch)
    assert torch.isfinite(loss) and torch.isfinite(action).all()
    result['checks'].append({'head':head,'loss':float(loss),'action_shape':list(action.shape)})
    del model,batch,loss,action
    torch.cuda.empty_cache()
result['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
print(json.dumps(result,indent=2))
