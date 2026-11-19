"""Match native Meshcat replays against their frozen camera evaluation reports."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image
ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);a=ap.parse_args()
rows=[]
for stem,label in [('classical','classical'),('cap-feedback-2','cap-v2'),('bc-diffusion-seed-0','bc-diffusion-s0')]:
    media=json.loads((a.out/f'replays/{stem}.json').read_text())
    r=media['source_report']
    if label!='classical':
        freeze=json.loads((a.out/'freeze.json').read_text())
        assert r['policy_sha256']==freeze['artifacts'][label]['sha256']
    benchmark=a.out/f'eval/test/{label}/camera/{r["chart"]}.json'
    b=json.loads(benchmark.read_text())
    assert r['chart_sha256']==b['chart_sha256']
    assert r['config']==b['config']
    assert r['view']==b['view']=='camera'
    assert r['detector']==b['detector']=='concert'
    assert len(r['notes'])==len(b['notes'])
    for x,y in zip(r['notes'],b['notes']):
        for k in ('id','completed','judgment','failure_reason','contact_ticks','active_ticks','tail_ticks','tail_contact_ticks'):
            assert x[k]==y[k],(stem,k,x[k],y[k])
        for k in ('timing_error','max_gap_seconds','slide_rms_m','maintenance'):
            assert (x[k] is None and y[k] is None) or (x[k] is not None and y[k] is not None and abs(x[k]-y[k])<1e-9),(stem,k,x[k],y[k])
    gif=a.out/f'replays/{stem}.gif'
    with Image.open(gif) as im:
        assert im.n_frames>1 and im.size==(960,675)
        frame_count=im.n_frames
        duration=0
        for i in range(im.n_frames):im.seek(i);duration+=im.info['duration']
    assert duration==sum(media['frame_durations_ms'])
    rows.append(dict(controller=label,chart=r['chart'],completed=sum(n['completed'] for n in r['notes']),
        notes=len(r['notes']),matches_frozen_note_metrics=True,encoded_frames=frame_count,duration_ms=duration,
        gif_sha256=hashlib.sha256(gif.read_bytes()).hexdigest(),benchmark=str(benchmark),source_sha256=b['source_sha256']))
(a.out/'replay-verification.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
