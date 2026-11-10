"""Check completeness and every evaluation fingerprint against frozen artifacts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from rhythmbot.experiments import digest,source_hash

ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('out',type=Path);a=ap.parse_args()
out=a.out.resolve();freeze=json.loads((out/'freeze.json').read_text())
assert source_hash()==freeze['source_hash'], 'Source changed since freeze'
assert hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest()==freeze['protocol_sha256']
for name,item in freeze['artifacts'].items():
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'],name
protocol=json.loads((out/'protocol.json').read_text())
for e in protocol['entries']:
    assert hashlib.sha256((out/e['path']).read_bytes()).hexdigest()==e['sha256']
counts=Counter();errors=[];policy_errors=0;cells=Counter()
for p in (out/'eval').rglob('*.json'):
    r=json.loads(p.read_text());j=r['study_job'];phase=p.relative_to(out/'eval').parts[0]
    counts[phase]+=1;cells[(phase,j['label'],j['detector'])]+=1
    h=None if j['policy']=='classical' else freeze['artifacts'][j['label']]['sha256']
    expected=digest(dict(job=j,source_hash=freeze['source_hash'],chart_sha256=hashlib.sha256(Path(j['chart']).read_bytes()).hexdigest(),policy_sha256=h))
    assert r['fingerprint']==expected,str(p)
    if r['execution_status']!='ok':errors.append(str(p))
    policy_errors+=sum(s.get('plan_errors',0) for s in r.get('policy_stats_by_hand',[]))
assert dict(counts)==freeze['expected_runs'],dict(counts)
for label in ['classical','bc-mlp','bc-diffusion','cap-v0','cap-v1','cap-v2']:
    for detector in ['oracle','color']:assert cells[('test',label,detector)]==8
record=dict(counts=dict(counts),fingerprints_verified=sum(counts.values()),execution_errors=errors,policy_runtime_errors=policy_errors)
(out/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
