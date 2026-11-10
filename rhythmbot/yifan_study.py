"""Frozen train/validation/test pilot for Yifan's bimanual CAP and BC study.

Prepare locally, collect classical commanded-target demonstrations on CPU,
train state MLP/diffusion BC, and evaluate all policies in full Drake physics.
The shared router splits cross-hand notes for every policy. Policies only
receive their local tracked segments, never test-chart ground truth.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
from .chart import Chart
from .experiments import make_chart, write_json


def prepare(out):
    path=out/'protocol.json'
    if path.exists(): return json.loads(path.read_text())
    entries=[]
    for split,seeds in [('train',range(11000,11016)),('validation',range(21000,21002)),('test_iid',range(31000,31002))]:
        for seed in seeds:
            for family in ('crossing','holds','chords'):
                speed=(.5,.6,.8)[seed%3] if family=='crossing' else None
                chart=make_chart(family,seed,speed)
                rel=f'charts/{split}/{chart.name}.json';(out/rel).parent.mkdir(parents=True,exist_ok=True);chart.save(out/rel)
                entries.append(dict(split=split,family=family,seed=seed,path=rel,sha256=hashlib.sha256((out/rel).read_bytes()).hexdigest()))
    for seed,speed in [(41000,.3),(41001,.4)]:
        chart=make_chart('crossing',seed,speed);rel=f'charts/test_speed/{chart.name}.json'
        (out/rel).parent.mkdir(parents=True,exist_ok=True);chart.save(out/rel)
        entries.append(dict(split='test_speed',family='crossing',seed=seed,path=rel,sha256=hashlib.sha256((out/rel).read_bytes()).hexdigest()))
    protocol=dict(entries=entries,bc_seed=0,bc_epochs=30,bc_train_charts=48,bc_validation_charts=6,
                  cap_feedback_charts=[e['path'] for e in entries if e['split']=='train'][:6],
                  cap_revisions=3,test_charts=8,detectors=['oracle','color'],
                  notes='Pilot: one BC seed per head and one 3-revision CAP chain; shared classical handoff routing and tool constraints. Color sees calibrated frames, not camera/concert art. Test accessed only after freezing models and programs.')
    write_json(path,protocol);return protocol


def collect_chart(job):
    from pydrake.all import DiagramBuilder, RigidTransform, RotationMatrix, Simulator
    from .config import stylus_config
    from .brain import GameScreen,RhythmBrain
    from .perception import make_detector
    from .bc.train import _Recorder
    from .planner import ClassicalPlanner
    cfg=stylus_config();chart=Chart.load(job['path']);recorders=[];initial=[]
    for hand in (0,1):
        expert=ClassicalPlanner(cfg,hand)
        rec=_Recorder(cfg);rec.expert=expert;recorders.append(rec)
        R=RotationMatrix.MakeYRotation(np.deg2rad(cfg.gripper.tilt_degrees)*(-1 if hand==0 else 1))
        from .planner import R_DOWN
        X=RigidTransform(R@R_DOWN,[expert.rest_x(),expert.hit_y,expert.hover_z])
        initial.append(X@RigidTransform([0,-cfg.gripper.tip_offset,0]))
    builder=DiagramBuilder();screen=builder.AddSystem(GameScreen(chart,cfg))
    brain=builder.AddSystem(RhythmBrain(make_detector('oracle',chart,cfg,None),recorders,cfg,initial))
    builder.Connect(screen.get_output_port(),brain.GetInputPort('frame'))
    sim=Simulator(builder.Build());sim.AdvanceTo(chart.end_time+1)
    return dict(obs=np.concatenate([np.array(r.obs,dtype=np.float32) for r in recorders]),
                act=np.concatenate([np.array(r.act,dtype=np.float32) for r in recorders]))


def collect(out,workers):
    protocol=prepare(out)
    for split in ('train','validation'):
        dest=out/f'{split}.npz'
        if dest.exists():continue
        entries=[e for e in protocol['entries'] if e['split']==split]
        jobs=[dict(path=str(out/e['path'])) for e in entries]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts=list(pool.map(collect_chart,jobs))
        data={k:np.concatenate([p[k] for p in parts]) for k in ('obs','act')}
        np.savez_compressed(dest,**data)
        write_json(out/f'{split}-data.json',dict(charts=len(jobs),samples=len(data['obs']),
            observation_shape=list(data['obs'].shape),action_shape=list(data['act'].shape),
            type='commanded-target demonstrations through shared brain+router; no physics during collection'))
        print(split,data['obs'].shape,flush=True)


def fit(out,head,device,epochs):
    import torch
    from .bc.models import BCModel
    torch.set_num_threads(2);torch.manual_seed(0)
    if device.startswith("cuda"): torch.cuda.set_per_process_memory_fraction(.12)
    train=np.load(out/'train.npz');val=np.load(out/'validation.npz')
    model=BCModel('state',head,16,256)
    act=train['act'].reshape(len(train['act']),-1)
    model.act_mean.copy_(torch.from_numpy(act.mean(0)));model.act_std.copy_(torch.from_numpy(act.std(0)+1e-3))
    model.encoder.obs_mean.copy_(torch.from_numpy(train['obs'].mean(0)))
    model.encoder.obs_std.copy_(torch.from_numpy(train['obs'].std(0)+1e-3));model.to(device)
    O=torch.from_numpy(train['obs']).to(device);A=torch.from_numpy(train['act']).to(device)
    V=torch.from_numpy(val['obs']).to(device);VA=torch.from_numpy(val['act']).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    best=float('inf');hist=[];bs=256
    path=out/'checkpoints'/f'bc-{head}.pt';path.parent.mkdir(parents=True,exist_ok=True)
    for ep in range(epochs):
        model.train();idx=torch.randperm(len(O),device=device);total=0;start=time.time()
        for ids in idx.split(bs):
            loss=model.loss(dict(obs=O[ids],act=A[ids]));opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();total+=float(loss)*len(ids)
        model.eval();vl=0
        # Fixed validation randomness for diffusion; no test-chart access.
        with torch.random.fork_rng(devices=[torch.device(device).index or 0] if device.startswith('cuda') else []):
            torch.manual_seed(991)
            with torch.no_grad():
                for i in range(0,len(V),bs):vl+=float(model.loss(dict(obs=V[i:i+bs],act=VA[i:i+bs])))*len(V[i:i+bs])
        vl/=len(V)
        hist.append(dict(epoch=ep+1,train_loss=total/len(O),validation_loss=vl,wall_seconds=time.time()-start))
        if vl<best:
            best=vl
            torch.save(dict(arch=model.arch,state_dict=model.state_dict(),tool_mode='bimanual_stylus',
                exec_steps=1 if head=='mlp' else 4,seed=0,epoch=ep+1,config='stylus_config',validation_loss=vl),path)
        write_json(out/f'bc-{head}-training.json',dict(head=head,epochs=hist,best_validation_loss=best,
            checkpoint=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seed=0))
        print(head,hist[-1],flush=True)


def evaluate_job(job):
    os.environ['HEADLESS']='1'
    from .config import stylus_config
    from .play import run
    from .policies import make_policy
    cfg=stylus_config();out=Path(job['out']);out.parent.mkdir(parents=True,exist_ok=True)
    from .experiments import source_hash,digest
    artifact=job['policy'].split(':',1)[1] if ':' in job['policy'] else None
    identity=digest(dict(job=job,source_hash=source_hash(),
        chart_sha256=hashlib.sha256(Path(job['chart']).read_bytes()).hexdigest(),
        policy_sha256=hashlib.sha256(Path(artifact).read_bytes()).hexdigest() if artifact else None))
    if out.exists():
        cached=json.loads(out.read_text())
        if cached.get('fingerprint')==identity:return cached
        raise RuntimeError('Mismatched evaluation cache: '+str(out))
    try:
        if job['policy'].startswith('bc:'):
            import torch
            torch.set_num_threads(1);torch.manual_seed(0)
        policies=None if job['policy']=='classical' else [make_policy(job['policy'],cfg,hand=k) for k in (0,1)]
        rep=run(Chart.load(job['chart']),cfg,detector=job['detector'],realtime=False,quiet=True,
                policy=policies,policy_name=job['label'],out_path=out)
        rep['execution_status']='ok'
        if policies:rep['policy_stats_by_hand']=[p.stats() if hasattr(p,'stats') else {} for p in policies]
    except Exception:rep=dict(execution_status='error',error=traceback.format_exc())
    rep['study_job']=job;rep['fingerprint']=identity;write_json(out,rep);return rep


def evaluate(out,specs,labels,split,detectors,workers):
    protocol=prepare(out)
    if split=='feedback': entries=[e for e in protocol['entries'] if e['path'] in protocol['cap_feedback_charts']]
    else: entries=[e for e in protocol['entries'] if e['split'].startswith(split)]
    jobs=[]
    for spec,label in zip(specs,labels):
        for det in detectors:
            for e in entries:
                chart=out/e['path']
                jobs.append(dict(policy=spec,label=label,detector=det,chart=str(chart),split=e['split'],
                    out=str(out/'eval'/split/label/det/f'{chart.stem}.json')))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(evaluate_job,j) for j in jobs]
        for i,f in enumerate(as_completed(fs),1):
            r=f.result();print(i,len(fs),r['study_job']['label'],r.get('completion_rate'),r['execution_status'],flush=True)
    return [json.loads(Path(j['out']).read_text()) for j in jobs]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['prepare','collect','fit','eval'])
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--workers',type=int,choices=[1,2],default=2)
    ap.add_argument('--head',choices=['mlp','diffusion'],default='mlp')
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--epochs',type=int,default=30)
    ap.add_argument('--policy',nargs='+',default=['classical']);ap.add_argument('--label',nargs='+',default=['classical'])
    ap.add_argument('--split',choices=['feedback','validation','test'],default='test')
    ap.add_argument('--detector',nargs='+',choices=['oracle','color'],default=['oracle','color'])
    a=ap.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    for var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(var,'1')
    if a.action=='prepare':print(prepare(a.out)['notes'])
    elif a.action=='collect':collect(a.out,a.workers)
    elif a.action=='fit':fit(a.out,a.head,a.device,a.epochs)
    else:
        if len(a.policy)!=len(a.label):ap.error('--policy and --label counts must match')
        evaluate(a.out,a.policy,a.label,a.split,a.detector,a.workers)

if __name__=='__main__':main()
