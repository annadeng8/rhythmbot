"""Paired bimanual benchmark, resumable at individual simulation boundaries.

python -m rhythmbot.experiments --out results/bimanual-suite --prepare
HEADLESS=1 python -m rhythmbot.experiments --out results/bimanual-suite --workers 2
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import traceback
import numpy as np
from .chart import Chart, Note, TAP, HOLD, SLIDE

ROOT=Path(__file__).resolve().parent.parent
SEEDS=(101,202,303)
SPEEDS=(.3,.4,.5,.6,.8)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def source_hash():
    return digest({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted((ROOT/'rhythmbot').rglob('*.py')) if not p.name.startswith('._')})


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def make_chart(family,seed,speed=None):
    rng=np.random.default_rng(seed)
    left=float(rng.uniform(.85,1.15));right=float(rng.uniform(3.85,4.15))
    if rng.random()<.5: left,right=right,left
    notes=[]
    if family=='crossing':
        # Four crossings in one sustain; then two short-gap opposite slides.
        notes.append(Note(SLIDE,left,3,waypoints=[(i*speed,left if i%2==0 else right) for i in range(5)]))
        t=3+4*speed+.6
        notes += [Note(SLIDE,left,t,waypoints=[(0,left),(2*speed,right)]),
                  Note(SLIDE,right,t+2*speed+.2,waypoints=[(0,right),(2*speed,left)])]
    elif family=='holds':
        a=int(rng.integers(0,3));b=int(rng.integers(3,6))
        notes=[Note(HOLD,a,3,2),Note(HOLD,b,3,2),
               Note(HOLD,1,5.6,1.6),Note(SLIDE,3,5.6,waypoints=[(0,3),(.4,5),(.8,3),(1.2,5),(1.6,4)]),
               Note(HOLD,4,8,1.6),Note(SLIDE,2,8,waypoints=[(0,2),(.4,0),(.8,2),(1.2,0),(1.6,1)])]
    elif family=='chords':
        # 2-note paired controls then deliberate 3-note capacity stress.
        t=3.
        for gap in (.5,.35,.25):
            for _ in range(3):
                notes += [Note(TAP,int(rng.integers(0,3)),t),Note(TAP,int(rng.integers(3,6)),t)]
                t+=gap
            t+=.5
        for _ in range(3):
            lanes=([0,1,4] if rng.random()<.5 else [1,4,5])
            notes += [Note(TAP,l,t) for l in lanes];t+=.7
    else: raise ValueError(family)
    return Chart(notes,name=f'{family}-s{seed}'+(f'-{speed:.1f}s' if speed else ''),bpm=150)


def prepare(out):
    if (out/'manifest.json').exists():
        manifest=json.loads((out/'manifest.json').read_text())
        if manifest['source_hash']!=source_hash():
            raise RuntimeError('Source changed: use a new output directory for comparable results')
        return manifest
    jobs=[]
    for family in ('crossing','holds','chords'):
        for speed in SPEEDS if family=='crossing' else (None,):
            for seed in SEEDS:
                chart=make_chart(family,seed,speed)
                chartfile=out/'charts'/f'{chart.name}.json'
                chartfile.parent.mkdir(parents=True,exist_ok=True);chart.save(chartfile)
                modes=[('stylus','oracle',True),('two_finger','oracle',True)]
                # Paired calibrated-image perception comparison on all family types.
                if family!='crossing' or speed==.6: modes.append(('stylus','color',True))
                # Isolate coordination, leaving the tool and strict judge fixed.
                if family=='crossing' and speed in (.4,.6): modes.append(('stylus','oracle',False))
                for tool,detector,handoff in modes:
                    name=f'{chart.name}-{tool}-{detector}'+('' if handoff else '-no_handoff')
                    jobs.append(dict(id=name,family=family,speed=speed,seed=seed,
                        tool=tool,detector=detector,handoff=handoff,chart=str(chartfile.relative_to(out)),
                        chart_hash=hashlib.sha256(chartfile.read_bytes()).hexdigest()))
    # Original demonstrations retained as fixed anchor controls.
    for file in ('demo','stylus-hard-handoffs'):
        chart=Chart.load(ROOT/'charts'/f'{file}.json');path=out/'charts'/f'{file}.json';chart.save(path)
        for tool in ('stylus','two_finger'):
            jobs.append(dict(id=f'anchor-{file}-{tool}',family='anchor-'+file,speed=None,seed=None,
                tool=tool,detector='oracle',handoff=True,chart=str(path.relative_to(out)),
                chart_hash=hashlib.sha256(path.read_bytes()).hexdigest()))
    try: rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except (OSError,subprocess.CalledProcessError): rev='unavailable'
    manifest=dict(schema=1,created_utc=datetime.now(timezone.utc).isoformat(),source_revision=rev,
        source_hash=source_hash(),seeds=SEEDS,jobs=jobs,
        protocol='Fixed strict judge; paired identical charts; seeds vary charts, not stochastic repeats. Color uses calibrated FrameRenderer, NOT the concert art or external camera.',
        resource_budget=dict(workers=2,threads_per_worker=1,gpu=0,ram_gb=8,disk_gb=2))
    write_json(out/'manifest.json',manifest)
    return manifest


def run_job(out_str,job,code_hash):
    os.environ['HEADLESS']='1'
    from .config import stylus_config,two_arm_config
    from .play import run
    out=Path(out_str);path=out/'runs'/f"{job['id']}.json"
    fingerprint=digest(dict(job=job,source_hash=code_hash))
    if path.exists():
        report=json.loads(path.read_text())
        if report.get('fingerprint')!=fingerprint: raise RuntimeError('Mismatched resume file: '+str(path))
        return job['id'],report['execution_status'],'cached'
    chart_path=out/job['chart']
    if hashlib.sha256(chart_path.read_bytes()).hexdigest()!=job['chart_hash']:
        raise RuntimeError('Chart changed: '+str(chart_path))
    try:
        cfg=stylus_config() if job['tool']=='stylus' else two_arm_config()
        cfg=replace(cfg,arms=replace(cfg.arms,slide_handoff=job['handoff']))
        chart=Chart.load(chart_path)
        report=run(chart,cfg,detector=job['detector'],realtime=False,quiet=True,
                   out_path=out/'raw'/f"{job['id']}.json")
        report.update(execution_status='ok',fingerprint=fingerprint,experiment=job)
    except Exception:
        report=dict(execution_status='error',fingerprint=fingerprint,experiment=job,error=traceback.format_exc())
    write_json(path,report)
    return job['id'],report['execution_status'],'new'


def summarize(out,manifest):
    groups=defaultdict(list);errors=[];missing=[]
    for job in manifest['jobs']:
        path=out/'runs'/f"{job['id']}.json"
        if not path.exists():missing.append(job['id']);continue
        r=json.loads(path.read_text())
        if r['execution_status']!='ok':errors.append(job['id']);continue
        key=(job['family'],job['speed'],job['tool'],job['detector'],job['handoff'])
        groups[key].append(r)
    rows=[]
    for key,reports in groups.items():
        ns=[n for r in reports for n in r['notes']]
        kinds={k:dict(completed=sum(n['completed'] is True for n in ns if n['kind']==k),
                      total=sum(n['kind']==k for n in ns)) for k in (TAP,HOLD,SLIDE)}
        failures=Counter(n['failure_reason'] or 'head_miss' for n in ns if not n['completed'])
        rows.append(dict(family=key[0],speed=key[1],tool=key[2],detector=key[3],handoff=key[4],
            charts=len(reports),completed=sum(n['completed'] is True for n in ns),notes=len(ns),
            micro_completion=sum(n['completed'] is True for n in ns)/len(ns),
            macro_completion=float(np.mean([r['completion_rate'] for r in reports])),
            by_kind=kinds,failures=dict(failures),
            max_gap_s=max(n['max_gap_seconds'] for n in ns),
            wall_seconds=sum(r['wall_seconds'] for r in reports)))
    summary=dict(expected=len(manifest['jobs']),finished=sum(len(rs) for rs in groups.values()),errors=errors,missing=missing,rows=rows)
    write_json(out/'summary.json',summary)
    lines=['# Bimanual experiment results','',f"Finished simulations: {summary['finished']}/{summary['expected']}; errors: {len(errors)}; missing: {len(missing)}.",'',manifest['protocol'],'',
           '| Family | Cross time | Tool | Detector | Handoff | Charts | Complete | Slides |',
           '|---|---:|---|---|---|---:|---:|---:|']
    for r in rows:
        sl=r['by_kind']['slide']
        lines.append(f"| {r['family']} | {r['speed'] if r['speed'] else '—'} | {r['tool']} | {r['detector']} | {r['handoff']} | {r['charts']} | {r['completed']}/{r['notes']} ({r['micro_completion']:.1%}) | {sl['completed']}/{sl['total']} |")
    lines += ['', 'Errors and missing runs are never counted as successful simulations. Three chart seeds are paired procedural variants, not independent repeats or a confidence interval. Full per-note failures and configs are in runs/.', '']
    (out/'summary.md').write_text('\n'.join(lines))
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--summarize',action='store_true')
    ap.add_argument('--workers',type=int,choices=(1,2),default=2)
    args=ap.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    manifest=prepare(out)
    if args.prepare: print(f"Prepared {len(manifest['jobs'])} simulations");return
    if not args.summarize:
        for var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[var]='1'
        os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HEADLESS']='1'
        versions={name:importlib.metadata.version(name) for name in ('drake','numpy','scipy')}
        write_json(out/'runtime.json',dict(versions=versions,workers=args.workers,started_utc=datetime.now(timezone.utc).isoformat()))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(run_job,str(out),job,manifest['source_hash']) for job in manifest['jobs']]
            for i,f in enumerate(as_completed(futures),1):
                print(f'{i}/{len(futures)} {f.result()}',flush=True)
                summarize(out,manifest)
    s=summarize(out,manifest)
    print(f"Complete: {s['finished']}, errors {len(s['errors'])}, missing {len(s['missing'])}")
    if s['errors'] or s['missing']: raise SystemExit(1)

if __name__=='__main__':main()
