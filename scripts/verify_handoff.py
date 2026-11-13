"""Acceptance checks for the unified-entry handoff's 16 real Drake runs."""
import hashlib
import json
from pathlib import Path
from rhythmbot import PROJECT_ROOT, splits
from rhythmbot.experiments import source_hash

out=PROJECT_ROOT/'results/handoff-20261113'
source=source_hash()
judge=hashlib.sha256((PROJECT_ROOT/'rhythmbot/judge.py').read_bytes()).hexdigest()[:16]
old=json.loads((PROJECT_ROOT/'results/yifan-study-v1/freeze.json').read_text())
rows=[]
for preset in ('sekai-camera','sekai-screen'):
    reports=list((out/preset).glob('*/concert/*.json'))
    assert len(reports)==8,(preset,len(reports))
    manifest=json.loads((out/preset/'environment.json').read_text())
    assert manifest['source_sha256']==source
    for path in reports:
        r=json.loads(path.read_text());p=r['provenance']
        assert 'load_error' not in r and 'run_error' not in r,path
        assert (p['arms'],p['contacts_per_arm'],p['single_tip'],p['display'])==(2,1,True,'concert')
        assert p['preset']==preset and p['view']==preset.split('-')[1]
        assert p['source_sha256']==source and p['judge_sha256']==judge
        assert p['camera']['width']==960 and p['camera']['height']==720
        c=splits.load(r['chart_entry'])
        assert r['chart_sha256']==hashlib.sha256(json.dumps([n.to_dict() for n in c.notes],sort_keys=True).encode()).hexdigest()
        v=r['vision'];assert v['frames']>0 and r['n_detections']>0
        assert not any(v[k] for k in ('chart_access','depth_input','label_input'))
        if r['policy']!='classical':
            assert len(r['policy_stats_by_hand'])==2
            assert sum(x.get('plan_errors',0) for x in r['policy_stats_by_hand'])==0
            artifact=r['policy'].split(':',1)[1]
            actual=hashlib.sha256((PROJECT_ROOT/artifact).read_bytes()).hexdigest()
            assert actual[:16]==p['policy_sha256']
            name='cap-v1' if r['policy'].startswith('cap:') else Path(artifact).stem
            assert actual==old['artifacts'][name]['sha256']
        rows.append(dict(preset=preset,policy=r['policy'],label=('classical' if r['policy']=='classical' else name),chart=r['chart'],
            completed=sum(n['completed'] is True for n in r['notes']),notes=r['n_notes']))
assert 'Ran 39 tests' in (out/'tests.log').read_text() and (out/'tests.log').read_text().rstrip().endswith('OK')
record=dict(tests_passed=39,base_commit='5c644a7715624d7f08161f84db0f473e4e76ce76',source_sha256=source,
    judge_sha256=judge,runs_verified=len(rows),execution_errors=0,policy_errors=0,
    scope='2 charts x 4 methods x 2 presets: entry-point validation, not a complete benchmark',rows=rows)
(out/'validation.json').write_text(json.dumps(record,indent=2)+'\n')
lines=['# Unified handoff validation','',record['scope'],
       '',f"16 real Drake runs verified against source `{source}` and judge `{judge}`. No execution/load/policy errors.",
       '','| Preset | Policy | Chart | Complete |','|---|---|---|---:|']
for r in rows:lines.append(f"| {r['preset']} | {r['label']} | {r['chart']} | {r['completed']}/{r['notes']} |")
lines+=['','These runs use Anna\'s stricter touch-landing judge. They do not replace the historical 64-run table or establish a new full-suite completion rate. Reports from the remote source snapshot may have no git commit field; the full source hash and base revision above identify what was tested.','']
(out/'validation.md').write_text('\n'.join(lines))
print(json.dumps({k:v for k,v in record.items() if k!='rows'}))
