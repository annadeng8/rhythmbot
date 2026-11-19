"""Summarize the frozen study, keeping seed and observation conditions explicit."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics as st


def metrics(reports):
    ns = [n for r in reports for n in r['notes']]
    def family(r):return 'song_excerpt' if r['chart'].startswith('unknown-') else r['chart'].split('-')[0]
    timing = [abs(n['timing_error'])*1000 for n in ns if n['timing_error'] is not None]
    slides = [n for n in ns if n['kind']=='slide']
    rms = [n['slide_rms_m']*1000 for n in slides if n['slide_rms_m'] is not None]
    contact = [n['maintenance'] for n in ns if n['kind']!='tap' and n['maintenance'] is not None]
    chord_notes=defaultdict(list)
    for report in reports:
        taps=[n for n in report['notes'] if n['kind']=='tap']
        multiplicity=Counter(round(n['time'],6) for n in taps)
        for n in taps:chord_notes[multiplicity[round(n['time'],6)]].append(n)
    return dict(tap_simultaneity={str(k):dict(notes=len(v),completed=sum(bool(n['completed']) for n in v)) for k,v in sorted(chord_notes.items())},charts=len(reports), notes=len(ns), completed=sum(bool(n['completed']) for n in ns),
        completion=st.mean(bool(n['completed']) for n in ns),
        chart_mean_completion=st.mean(r['completion_rate'] for r in reports),
        by_family={k:dict(completed=sum(bool(n['completed']) for r in reports if family(r)==k for n in r['notes']), notes=sum(len(r['notes']) for r in reports if family(r)==k)) for k in sorted({family(r) for r in reports})},
        by_kind={k:dict(completed=sum(bool(n['completed']) for n in ns if n['kind']==k),
                       notes=sum(n['kind']==k for n in ns)) for k in ('tap','hold','slide')},
        head_timing_abs_mean_ms=st.mean(timing) if timing else None, timing_sample_count=len(timing),
        slide_contact_rms_mean_mm=st.mean(rms) if rms else None, slide_rms_sample_count=len(rms),
        sustain_maintenance_mean=st.mean(contact) if contact else None, sustain_maintenance_sample_count=len(contact),
        failures=dict(Counter(n['failure_reason'] or 'head_miss' for n in ns if not n['completed'])),
        policy_errors=sum(s.get('plan_errors',0) for r in reports for s in r.get('policy_stats_by_hand',[])))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('out',type=Path);a=ap.parse_args()
    p=json.loads((a.out/'protocol.json').read_text())
    split_by_name={Path(e['path']).stem:e['split'] for e in p['entries']}
    groups=defaultdict(list);test=[];all_reports=[]
    for path in sorted((a.out/'eval').rglob('*.json')):
        r=json.loads(path.read_text());assert r['execution_status']=='ok',path
        phase=path.relative_to(a.out/'eval').parts[0]
        split=split_by_name[Path(r['job']['chart']).stem]
        for scope in (split,'all'):
            groups[(phase,r['job']['label'],r['job']['condition'],scope)].append(r)
        all_reports.append(r)
        if phase=='test':test.append(r)
    assert len(test)==330, f'Incomplete frozen evaluation: {len(test)}/330'
    assert len(all_reports)==366, f'Expected 36 feedback + 330 test: {len(all_reports)}'
    rows=[dict(phase=k[0],policy=k[1],condition=k[2],split=k[3],**metrics(v)) for k,v in sorted(groups.items())]
    seed_summary=[]
    for head in ('mlp','diffusion'):
        for condition in ('oracle','screen','camera'):
            for split in ('all','test_iid','test_speed','test_song'):
                rs=[r for r in rows if r['phase']=='test' and r['policy'].startswith('bc-'+head+'-s') and r['condition']==condition and r['split']==split]
                assert len(rs)==3
                seed_summary.append(dict(head=head,condition=condition,split=split,
                    mean=st.mean(r['completion'] for r in rs),sd=st.stdev(r['completion'] for r in rs),
                    min=min(r['completion'] for r in rs),max=max(r['completion'] for r in rs)))
    demo_groups=defaultdict(list)
    for path in (a.out/'demos').rglob('*.json'):
        r=json.loads(path.read_text());assert r['execution_status']=='ok'
        parts=path.relative_to(a.out/'demos').parts
        demo_groups[(parts[0],r['job']['condition'])].append(r)
    demos=[dict(split=k[0],condition=k[1],**metrics(v)) for k,v in sorted(demo_groups.items())]
    summary=dict(test_runs=len(test),feedback_runs=len(all_reports)-len(test),rows=rows,bc_seed_summary=seed_summary,demonstrations=demos)
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Dual-arm CAP / BC study','',
        'All runs use the fixed minimal Sekai display, two Drake iiwa arms, one stylus per arm, and the strict touchdown/body/tail judge. CAP means **Code as Policy**: Codex writes a trajectory program from the public API and training-only simulation feedback. Initial, feedback 1, and feedback 2 denote successive programs in one generation chain. BC means behavior cloning of full-physics classical demonstrations.','',
        'Oracle receives chart-derived observations; screen RGB uses the direct game image; camera RGB uses the Drake camera image with RGB-only color/shape detection. The latter is not a trained neural detector. The router and physical environment are shared.','',
        'Training: 48 charts × oracle/camera; validation: 6 separate charts × oracle/camera. MLP and diffusion each train for 60 epochs with three seeds. Each seed is selected by validation loss. All policies are frozen before 11 held-out charts × 3 observation conditions × 10 controllers = 330 test runs. The test includes 6 synthetic IID charts, 2 faster crossing charts, and 3 predeclared 30-second song excerpts. Chord charts include deliberate three-note capacity stress for two styli.','',
        '| Controller | Input | Complete notes | Equal-weight chart mean | Policy errors |',
        '|---|---|---:|---:|---:|']
    camera={r['policy']:r for r in rows if r['phase']=='test' and r['condition']=='camera' and r['split']=='all'}
    cap_slide=[camera[f'cap-v{i}']['by_kind']['slide'] for i in range(3)]
    headline=(f"**Main result:** camera classical completion is {camera['classical']['completion']:.1%}. "
        f"CAP initial / feedback 1 / feedback 2 achieve "
        + ' / '.join(f"{camera[f'cap-v{i}']['completion']:.1%}" for i in range(3))
        + ". The final feedback revision does not improve overall camera completion over the initial program; slide completion falls from "
        + ' to '.join(f"{r['completed']}/{r['notes']}" for r in cap_slide)
        + ", while all three CAP programs complete every held-out hold. This aggregate score hides a tap/slide tradeoff.")
    at=next(i for i,line in enumerate(lines) if line.startswith('| Controller'))
    lines[at:at]=[headline,'']
    labels={'cap-v0':'CAP initial','cap-v1':'CAP feedback 1','cap-v2':'CAP feedback 2'}
    for r in rows:
        if r['phase']=='test' and r['split']=='all':
            lines.append(f"| {labels.get(r['policy'],r['policy'])} | {r['condition']} | {r['completed']}/{r['notes']} ({r['completion']:.1%}) | {r['chart_mean_completion']:.1%} | {r['policy_errors']} |")
    lines+=['','BC variation across three training seeds (sample SD, not a confidence interval):','',
            '| Head | Input | Mean ± SD | Min–max |','|---|---|---:|---:|']
    for r in seed_summary:
        if r['split']=='all':lines.append(f"| {r['head']} | {r['condition']} | {r['mean']:.1%} ± {r['sd']:.1%} | {r['min']:.1%}–{r['max']:.1%} |")
    lines+=['','Camera outcomes by note type (BC pools three training seeds):','',
        '| Controller | Tap | Hold | Slide |','|---|---:|---:|---:|']
    for label,keys in [('Classical',['classical']),('CAP initial',['cap-v0']),('CAP feedback 1',['cap-v1']),('CAP feedback 2',['cap-v2']),('BC MLP (3 seeds)',[f'bc-mlp-s{i}' for i in range(3)]),('BC diffusion (3 seeds)',[f'bc-diffusion-s{i}' for i in range(3)])]:
        counts=[]
        for kind in ('tap','hold','slide'):
            ks=[camera[k]['by_kind'][kind] for k in keys]
            counts.append(f"{sum(k['completed'] for k in ks)}/{sum(k['notes'] for k in ks)}")
        lines.append('| '+label+' | '+' | '.join(counts)+' |')
    lines+=['','Per-family, per-note-type, failure, timing and tracking metrics are in `summary.json`; raw per-note reports are under `eval/`. Timing is conditional on a detected head contact; slide RMS is conditional on available down-contact samples and does not penalize missing contact, so interpret it alongside completion/contact failures.','',
        'Limitations: one CAP generation chain; only three BC seeds; short song excerpts rather than full-song evaluation; commanded hand state as the policy input; fixed shared handoff routing rather than learned arm assignment; no validated robot–robot collision avoidance. New results should not be pooled with earlier pilots that used different scoring and charts.','']
    (a.out/'summary.md').write_text('\n'.join(lines))
    fields=['phase','policy','condition','split','charts','notes','completed','completion','chart_mean_completion','policy_errors']
    with (a.out/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
    print(json.dumps(dict(test_runs=len(test),feedback_runs=36,policy_errors=sum(s.get('plan_errors',0) for r in test for s in r.get('policy_stats_by_hand',[])))))

if __name__=='__main__':main()
