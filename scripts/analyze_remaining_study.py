"""Additional descriptive diagnostics; never changes the frozen trial selection."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import statistics

p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args()
groups=defaultdict(list)
for f in (a.out/'runs').rglob('*.json'):
 r=json.loads(f.read_text());assert r['execution_status']=='ok'
 j=r['job'];groups[(j['phase'],j['detector'],j['policy']['label'])].append(r)
result=[]
for (phase,detector,policy),rs in sorted(groups.items()):
 notes=[n for r in rs for n in r['notes']]
 frames=sum(r['vision']['frames'] for r in rs)
 by_family={}
 for family in sorted({r['job']['chart']['family'] for r in rs}):
  ns=[n for r in rs if r['job']['chart']['family']==family for n in r['notes']]
  by_family[family]=dict(notes=len(ns),completed=sum(n['completed'] for n in ns))
 gaps=[n['max_gap_seconds'] for n in notes if n['kind']!='tap']
 rms=[n['slide_rms_m'] for n in notes if n['slide_rms_m'] is not None]
 result.append(dict(phase=phase,detector=detector,policy=policy,
  chart_macro_completion=statistics.mean(r['completion_rate'] for r in rs),
  failures=dict(Counter(n['failure_reason'] or n['status'] for n in notes if not n['completed'])),
  by_family=by_family,frames=frames,
  blank_frame_fraction=sum(r['vision'].get('blank_detection_frames',0) for r in rs)/frames,
  ticks=sum(len(r['tick_seconds']) for r in rs),
  ticks_over_budget=sum(r['ticks_over_budget'] for r in rs),
  mean_sustain_max_gap_seconds=statistics.mean(gaps) if gaps else None,
  mean_slide_rms_m=statistics.mean(rms) if rms else None,
  browser_connections_max=max(r['browser_connections_end'] for r in rs)))
(a.out/'diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
print('Descriptive diagnostics:',len(result),'groups')
