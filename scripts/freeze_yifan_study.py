"""Record immutable policy/checkpoint identities before opening held-out tests."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from rhythmbot.experiments import source_hash

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('out',type=Path)
a=ap.parse_args();out=a.out.resolve()
if (out/'freeze.json').exists():
    raise SystemExit('Already frozen; do not replace the record.')
if list((out/'eval'/'test').rglob('*.json')):
    raise SystemExit('Cannot claim pre-test freeze: test reports already exist.')
protocol=json.loads((out/'protocol.json').read_text())
for entry in protocol['entries']:
    assert hashlib.sha256((out/entry['path']).read_bytes()).hexdigest()==entry['sha256']
artifacts={}
for name,rel in [(f'cap-v{i}',f'cap/v{i}/policy.py') for i in range(3)]+[(f'bc-{h}',f'checkpoints/bc-{h}.pt') for h in ('mlp','diffusion')]:
    p=out/rel
    artifacts[name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
for i in range(3):
    reports=list((out/'eval'/'feedback'/f'cap-v{i}'/'oracle').glob('*.json'))
    assert len(reports)==6, (i,len(reports))
record={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'source_hash':source_hash(),
        'protocol_sha256':hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest(),
        'artifacts':artifacts,'selection':'BC best validation loss; all CAP revisions evaluated; no test selection',
        'expected_runs':{'validation':18,'feedback':18,'test':96}}
(out/'freeze.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
