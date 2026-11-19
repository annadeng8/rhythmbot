"""Check protocol coverage, artifact identity and completed training before release."""
import argparse
from collections import Counter
import hashlib
import json
import re
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('out',type=Path);a=ap.parse_args();o=a.out
p=read(o/'protocol.json');f=read(o/'freeze.json')
assert f['protocol_sha256']==sha(o/'protocol.json')
manifest=read(o/'source-manifest.json')
root=Path(__file__).resolve().parents[1]
assert all(sha(root/path)==digest for path,digest in manifest.items())
assert hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()==f['source_sha256']
for e in p['entries']:assert sha(o/e['path'])==e['sha256']
checked_artifacts=[];missing_local_checkpoints=[]
for name,artifact in f['artifacts'].items():
    relative=Path('cap')/name.removeprefix('cap-')/'policy.py' if name.startswith('cap-') else Path('checkpoints')/(name+'.pt')
    local=o/relative
    if local.exists():
        assert sha(local)==artifact['sha256']
        checked_artifacts.append(name)
    elif not name.startswith('bc-'):raise AssertionError(local)
    else:missing_local_checkpoints.append(name)
for head in p['bc_heads']:
    for seed in p['bc_seeds']:
        name=f'bc-{head}-s{seed}';r=read(o/(name+'-training.json'))
        assert len(r['epochs'])==p['bc_epochs']
        assert r['checkpoint_sha256']==f['artifacts'][name]['sha256']
        assert r['training_sha256']==read(o/'train-data.json')['sha256']
        assert r['validation_sha256']==read(o/'validation-data.json')['sha256']
labels={'classical',*f['artifacts']}
expected={(label,c,Path(e['path']).stem) for label in labels for c in p['conditions'] for e in p['entries'] if e['split'].startswith('test')}
actual=set();reports=[];configs=set()
for path in (o/'eval/test').rglob('*.json'):
    r=read(path);j=r['job'];key=(j['label'],j['condition'],Path(j['chart']).stem)
    assert key not in actual,key
    actual.add(key);reports.append(r)
    assert r['execution_status']=='ok'
    assert r['source_sha256']==f['source_sha256']
    configs.add(json.dumps(r['config'],sort_keys=True))
    assert [r['detector'],r['view']]==p['conditions'][j['condition']]
    identity=dict(source=f['source_sha256'],chart=r['chart_sha256'],condition=j['condition'],
        policy='classical' if j['label']=='classical' else f['artifacts'][j['label']]['sha256'],seed=0,mode='eval')
    assert hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()==r['identity']
    chart=next(e for e in p['entries'] if Path(e['path']).stem==key[2])
    assert r['chart_sha256']==chart['sha256']
    assert all(n['completed'] is not None and n['judgment'] in ('perfect','great','good','miss') for n in r['notes'])
assert len(configs)==1
assert actual==expected,(len(actual),len(expected),expected-actual)
assert len(actual)==f['expected_test_runs']==330
for i in range(3):
    folder=o/f'cap/v{i}';m=read(folder/'metadata.json')
    assert m['provider']=='Codex CLI' and m['tool_calls_observed']==0
    assert m['policy_sha256']==sha(folder/'policy.py')
    assert m['prompt_sha256']==sha(folder/'prompt.txt')
    body=(folder/'response.txt').read_text()
    fenced=re.search(r'```(?:python)?\s*\n(.*?)```',body,re.S)
    code=fenced.group(1) if fenced else body
    assert code.strip()+'\n'==(folder/'policy.py').read_text()
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines() if line.strip()]
    assert any(e['type']=='turn.completed' for e in events)
    assert all(e['item']['type'] in ('reasoning','agent_message') for e in events if 'item' in e)
    feedback=read(folder/'feedback.json');assert len(feedback)==12
    feedback_names={Path(x).stem for x in p['cap_feedback_charts']}
    assert all(r['chart'] in feedback_names for r in feedback)
    if i:
        assert m['previous_sha256']==sha(o/f'cap/v{i-1}/policy.py')
        assert m['feedback_sha256']==sha(o/f'cap/v{i-1}/feedback.json')
        prompt=(folder/'prompt.txt').read_text()
        assert (o/f'cap/v{i-1}/policy.py').read_text() in prompt
        assert (o/f'cap/v{i-1}/feedback.json').read_text() in prompt
result=dict(test_runs=len(actual),training_rollouts=read(o/'train-data.json')['rollouts'],
    validation_rollouts=read(o/'validation-data.json')['rollouts'],bc_checkpoints=6,
    cap_training_feedback_runs=36,source_sha256=f['source_sha256'],
    execution_errors=0,policy_errors=sum(s.get('plan_errors',0) for r in reports for s in r.get('policy_stats_by_hand',[])),
    artifact_files_checked=checked_artifacts,missing_local_checkpoints=missing_local_checkpoints,
    checkpoint_location='gpu156: /data/YifanKang/rhythmbot_final_20261118/results/yifan-final-v1/checkpoints/')
(o/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
