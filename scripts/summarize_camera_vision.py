"""Verify the camera experiment and compare matched charts to the prior study."""
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from camera_vision_study import ROOT, OUT, OLD, config
from rhythmbot.experiments import source_hash,digest,write_json

freeze=json.loads((OUT/'freeze.json').read_text())
assert freeze['source_hash']==source_hash()
assert freeze['runner_sha256']==hashlib.sha256((ROOT/'scripts/camera_vision_study.py').read_bytes()).hexdigest()
prior=json.loads((OLD/'freeze.json').read_text())
for label,artifact in freeze['policies'].items():
    assert hashlib.sha256(Path(artifact['path']).read_bytes()).hexdigest()==artifact['sha256']
    assert artifact['sha256']==prior['artifacts'][label]['sha256'], 'Controller changed'
reports=[json.loads(p.read_text()) for p in (OUT/'eval').rglob('*.json')]
assert len(reports)==64
assert sorted([r['job'] for r in reports],key=lambda j:j['out'])==sorted(freeze['jobs'],key=lambda j:j['out'])
checks=reports+[json.loads(p.read_text()) for phase in ['warmup','negative'] for p in (OUT/phase).rglob('*.json')]
assert len(checks)==70
for r in checks:
    j=r['job'];artifact=freeze['policies'].get(j['label'])
    identity=digest(dict(job=j,source_hash=freeze['source_hash'],
        chart_sha256=hashlib.sha256(Path(j['chart']).read_bytes()).hexdigest(),
        policy_sha256=artifact['sha256'] if artifact else None,config=asdict(config())))
    assert r['fingerprint']==identity,j['out']
    if r['execution_status']=='ok':
        v=r['vision'];assert v['frames']>0 and not v['chart_access'] and not v['label_input'] and not v['depth_input']
    if j.get('blackout'):
        assert r['execution_status']=='ok' and r['n_detections']==0 and r['completion_rate']==0
policies=['classical','cap-v1','bc-mlp','bc-diffusion']
groups=defaultdict(list)
for r in reports:groups[(r['job']['label'],'concert-'+r['job']['view'])].append(r)
for label in policies:
    for detector in ['oracle','color']:
        for p in (OLD/'eval/test'/label/detector).glob('*.json'):
            r=json.loads(p.read_text())
            # Same chart bytes, not just matching names.
            baseline_chart=OLD/('charts/'+r['study_job']['chart'].split('/charts/',1)[1])
            matches=[j for j in freeze['jobs'] if j['label']==label and j['view']=='camera' and Path(j['chart']).name==baseline_chart.name]
            assert len(matches)==1
            assert Path(matches[0]['chart']).read_bytes()==baseline_chart.read_bytes()
            groups[(label,'prior-'+detector)].append(r)
rows=[]
for (label,view),rs in sorted(groups.items()):
    assert len(rs)==8
    ok=[r for r in rs if r['execution_status']=='ok'];notes=[n for r in ok for n in r['notes']]
    complete=sum(n['completed'] is True for n in notes)
    rows.append(dict(policy=label,input=view,charts=len(rs),execution_errors=len(rs)-len(ok),
        completed=complete,notes=len(notes),completion=complete/len(notes) if notes and len(ok)==len(rs) else None,
        chart_mean_completion=float(np.mean([r['completion_rate'] for r in ok])) if len(ok)==len(rs) else None,
        by_kind={k:dict(completed=sum(n['completed'] is True for n in notes if n['kind']==k),notes=sum(n['kind']==k for n in notes)) for k in ['tap','hold','slide']},
        failure_reasons=dict(Counter(n.get('failure_reason') or 'head_miss' for n in notes if not n['completed'])),
        policy_errors=sum(s.get('plan_errors',0) for r in ok for s in r.get('policy_stats_by_hand',[]))))
errors=[r['job']['out'] for r in checks if r['execution_status']!='ok']
write_json(OUT/'verification.json',dict(fingerprints_verified=len(checks),evaluation_runs=64,training_warmups=4,
    blackout_controls=2,execution_errors=errors,policy_runtime_errors=sum(s.get('plan_errors',0) for r in checks for s in r.get('policy_stats_by_hand',[])),
    frozen_controllers_match_previous_study=True,blackout_detections=0,blackout_completed_notes=0))
write_json(OUT/'summary.json',dict(rows=rows,source_hash=freeze['source_hash']))
lookup={(r['policy'],r['input']):r for r in rows}
inputs=['prior-oracle','prior-color','concert-screen','concert-camera']
lines=['# RGB camera comparison','',
 'Eight identical held-out charts; 78 notes per method/input. Controllers are unchanged. Prior color uses the calibrated flat display; concert-screen and concert-camera use the same concert artwork and the new RGB detector. The direct-concert vs camera-concert pair isolates the camera path (perspective sampling, noise and occlusion).','',
 '| Policy | Previous oracle | Previous color | Direct concert RGB | Drake camera RGB |',
 '|---|---:|---:|---:|---:|']
for policy in policies:
    vals=[lookup[policy,v] for v in inputs]
    cells=[f"{r['completed']}/{r['notes']} ({r['completion']:.1%})" if r['completion'] is not None else 'execution error' for r in vals]
    lines.append(f"| {policy} | {' | '.join(cells)} |")
lines+=['','| Policy | Input | Tap | Hold | Slide | Equal-chart mean |','|---|---|---:|---:|---:|---:|']
for policy in policies:
    for view in inputs[-2:]:
        r=lookup[policy,view];kinds=[f"{r['by_kind'][k]['completed']}/{r['by_kind'][k]['notes']}" for k in ['tap','hold','slide']]
        mean='unavailable' if r['chart_mean_completion'] is None else f"{r['chart_mean_completion']:.1%}"
        lines.append(f"| {policy} | {view} | {' | '.join(kinds)} | {mean} |")
lines+=['','The RGB blackout control uses nonempty training charts in full dual-arm camera simulation: zero detections and zero completions. No clean texture, depth, render labels or chart state is available to the detector. The simulator uses render labels only to paint the dynamic display without overwriting occluding robot pixels. This is an occlusion-correct camera composite, not a native animated Drake material or a real-world camera.','']
(OUT/'summary.md').write_text('\n'.join(lines))
fig,axs=plt.subplots(1,2,figsize=(13,4.6),layout='constrained')
x=np.arange(4)
for i,(view,color) in enumerate(zip(inputs,['#a5b0be','#7687aa','#d19c3f','#128b91'])):
    vals=[lookup[p,view]['completion'] for p in policies]
    axs[0].bar(x+(i-1.5)*.2,[v*100 if v is not None else np.nan for v in vals],.19,label=view,color=color)
axs[0].set(xticks=x,xticklabels=policies,ylim=(0,105),ylabel='Completed notes (%)',title='Same frozen controllers / same 8 charts')
axs[0].legend(fontsize=8)
for i,(view,color) in enumerate(zip(inputs[-2:],['#d19c3f','#128b91'])):
    vals=[lookup[p,view]['by_kind']['slide'] for p in policies]
    axs[1].bar(x+(i-.5)*.32,[100*v['completed']/v['notes'] for v in vals],.31,label=view,color=color)
axs[1].set(xticks=x,xticklabels=policies,ylim=(0,105),ylabel='Completed slides (%)',title='Strict sustained contact and tail scoring');axs[1].legend(fontsize=8)
for ax in axs:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.tick_params(axis='x',rotation=15)
fig.savefig(OUT/'comparison.png',dpi=160);fig.savefig(OUT/'comparison.pdf')
print(json.dumps(dict(verified=len(checks),execution_errors=errors)))
