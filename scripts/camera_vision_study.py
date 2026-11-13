"""Paired camera-vs-direct concert RGB study with frozen existing controllers."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace, asdict
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
from rhythmbot.chart import Chart
from rhythmbot.config import stylus_config, Camera
from rhythmbot.experiments import source_hash, digest, write_json

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'results/camera-vision-v1'
OLD=ROOT/'results/yifan-study-v1'
CHECKPOINT_ROOT=Path('/data/YifanKang/rhythmbot_yifan_study_v1/results/yifan-study-v1/checkpoints')

def config():
    return replace(stylus_config(),camera=Camera(width=960,height=720,fov_y_deg=48,
        eye_offset=(.35,-.50,1.15),target_offset=(0.,.05,0.),noise_std=3.))

def policies():
    return {'classical':'classical','cap-v1':'cap:'+str(OLD/'cap/v1/policy.py'),
            'bc-mlp':'bc:'+str(CHECKPOINT_ROOT/'bc-mlp.pt'),
            'bc-diffusion':'bc:'+str(CHECKPOINT_ROOT/'bc-diffusion.pt')}

def frame_panel(detector, judge, t, label):
    from PIL import Image, ImageDraw, ImageFont
    canvas=Image.new('RGB',(1440,780),(17,21,35))
    canvas.paste(Image.fromarray(detector.last_rgb),(0,60))
    rect=Image.fromarray(detector.last_rectified).resize((384,576))
    canvas.paste(rect,(1008,115))
    draw=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype('DejaVuSans.ttf',20)
    except OSError:font=ImageFont.load_default()
    complete=sum(r.completed is True for r in judge.results.values())
    draw.text((18,15),f'{label} / CAMERA RGB CONTROL / RGB t={detector.last_time:.2f}s',font=font,fill='white')
    draw.text((1000,15),f'{complete}/{len(judge.results)} complete',font=font,fill='white')
    draw.text((990,78),'RECTIFIED OBSERVED PIXELS',font=font,fill=(130,240,210))
    draw.text((980,715),'No chart / depth / label input',font=font,fill='white')
    draw.text((980,750),f'{len(detector.last_detections)} visual detections',font=font,fill='white')
    return canvas

def evaluate(job):
    from pydrake.all import Simulator
    from rhythmbot.play import build
    from rhythmbot.policies import make_policy
    from rhythmbot.judge import Judge
    if job['label'].startswith('bc'):
        import torch
        torch.set_num_threads(1);torch.manual_seed(0)
    artifact=job['policy'].split(':',1)[1] if ':' in job['policy'] else None
    identity=digest(dict(job=job,source_hash=source_hash(),
        chart_sha256=hashlib.sha256(Path(job['chart']).read_bytes()).hexdigest(),
        policy_sha256=hashlib.sha256(Path(artifact).read_bytes()).hexdigest() if artifact else None,
        config=asdict(config())))
    out=Path(job['out']);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():
        report=json.loads(out.read_text())
        if report['fingerprint'] != identity:raise RuntimeError('Evaluation fingerprint mismatch: '+str(out))
        return report
    start=time.time()
    try:
        cfg=config();chart=Chart.load(job['chart'])
        ps=None if job['policy']=='classical' else [make_policy(job['policy'],cfg,hand=k) for k in (0,1)]
        diagram,robot,brain,judge=build(chart,cfg,'concert',None,None,policy=ps,view=job['view'])
        if job.get('blackout'):
            detect = brain.detector.detect
            brain.detector.detect = lambda frame, t: detect(np.zeros_like(frame), t)
        sim=Simulator(diagram);context=sim.get_mutable_context();robot.initialize(context)
        end=chart.end_time+1
        if job.get('capture'):
            frames=Path(job['capture']);frames.mkdir(parents=True,exist_ok=True)
            for i,t in enumerate(np.arange(.1,end+.0001,.1)):
                sim.AdvanceTo(float(t))
                frame_panel(brain.detector,judge,float(t),job['label']).save(frames/f'{i:06d}.png')
            if context.get_time() < end: sim.AdvanceTo(end)
        else:sim.AdvanceTo(end)
        judge.finalize(end)
        report=judge.report(extra=dict(execution_status='ok',view=job['view'],detector='concert',
            vision=brain.detector.stats(),sim_seconds=end,wall_seconds=time.time()-start,
            n_detections=brain.last_state.n_detections,config=asdict(cfg),
            policy_stats_by_hand=[p.stats() if hasattr(p,'stats') else {} for p in (ps or [])]))
    except Exception:
        report=dict(execution_status='error',error=traceback.format_exc(),wall_seconds=time.time()-start)
    report.update(job=job,fingerprint=identity);write_json(out,report)
    return report

def jobs(warmup=False):
    protocol=json.loads((OLD/'protocol.json').read_text())
    entries=[e for e in protocol['entries'] if e['split'].startswith('test')]
    if warmup:
        entries=[e for e in protocol['entries'] if e['split']=='train' and e['family'] in ('crossing','holds')][:2]
    result=[]
    for label,spec in policies().items():
        if warmup and label!='cap-v1':continue
        for view in ('screen','camera'):
            for e in entries:
                name=Path(e['path']).stem
                result.append(dict(label=label,policy=spec,chart=str(OLD/e['path']),view=view,split=e['split'],
                    out=str(OUT/('warmup' if warmup else 'eval')/label/view/(name+'.json')),
                    capture=str(OUT/'frames'/label/('warmup-'+name if warmup else name)) if view=='camera' and
                    ((warmup and e['family']=='crossing') or (not warmup and label in ('cap-v1','bc-mlp') and name=='crossing-s31000-0.6s')) else None))
    return result

def freeze():
    if (OUT/'freeze.json').exists():raise RuntimeError('Already frozen')
    if list((OUT/'eval').rglob('*.json')):raise RuntimeError('Test reports already exist')
    warm=list((OUT/'warmup').rglob('*.json'))
    negative=list((OUT/'negative').rglob('*.json'))
    assert len(warm)==4 and len(negative)==2, 'Warmup and blackout controls required'
    for p in warm+negative:
        r=json.loads(p.read_text());assert r['execution_status']=='ok' and r['vision']['frames']>0
        if p in negative:assert r['completion_rate']==0 and r['n_detections']==0, 'Blackout leaked note information'
    records={}
    for name,spec in policies().items():
        if ':' in spec:
            p=Path(spec.split(':',1)[1]);records[name]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    write_json(OUT/'freeze.json',dict(source_hash=source_hash(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        config=asdict(config()),policies=records,jobs=jobs(),frozen_at=time.time(),
        protocol='8 unchanged held-out charts x 4 frozen policies x camera/direct concert RGB; no retraining or test tuning'))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['warmup','negative','freeze','eval'])
    a=ap.parse_args()
    if a.action=='freeze':freeze()
    else:
        work=jobs(a.action in ('warmup','negative'))
        if a.action=='negative':
            work=[dict(j,blackout=True,capture=None,out=j['out'].replace('/warmup/','/negative/')) for j in work if j['view']=='camera']
        if a.action=='eval':
            frozen=json.loads((OUT/'freeze.json').read_text());assert frozen['source_hash']==source_hash()
            assert frozen['jobs']==work
        with ProcessPoolExecutor(max_workers=4) as pool:
            for i,f in enumerate(as_completed([pool.submit(evaluate,j) for j in work]),1):
                r=f.result();print(i,len(work),r['job']['label'],r['job']['view'],r.get('completion_rate'),r['execution_status'],flush=True)
