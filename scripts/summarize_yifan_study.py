"""Aggregate frozen CAP/BC pilot results without selecting on held-out scores."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('out',type=Path);a=ap.parse_args()
protocol=json.loads((a.out/'protocol.json').read_text())
reports=[json.loads(p.read_text()) for p in (a.out/'eval').rglob('*.json')]
groups=defaultdict(list)
for r in reports:
 j=r['study_job'];phase=Path(j['out']).parts[-4]
 groups[(phase,j['label'],j['detector'],j['split'])].append(r)
rows=[]
for key,rs in sorted(groups.items()):
 ok=[r for r in rs if r['execution_status']=='ok'];ns=[n for r in ok for n in r['notes']]
 errors=len(rs)-len(ok);done=sum(n['completed'] is True for n in ns)
 rows.append(dict(phase=key[0],policy=key[1],detector=key[2],split=key[3],charts=len(rs),execution_errors=errors,
  complete=done,notes=len(ns),completion=done/len(ns) if ns and not errors else None,
  chart_mean_completion=float(np.mean([r['completion_rate'] for r in ok])) if ok and not errors else None,
  by_kind={k:dict(complete=sum(n['completed'] is True for n in ns if n['kind']==k),total=sum(n['kind']==k for n in ns)) for k in ['tap','hold','slide']},
  failure_reasons=dict(Counter(n.get('failure_reason') or 'head_miss' for n in ns if not n['completed'])),
  mean_abs_timing_ms=float(np.mean([abs(n['timing_error'])*1000 for n in ns if n['timing_error'] is not None])) if any(n['timing_error'] is not None for n in ns) else None,
  policy_runtime_errors=sum(p.get('plan_errors',0) for r in ok for p in r.get('policy_stats_by_hand',[]))))
summary=dict(protocol=protocol['notes'],rows=rows,total_runs=len(reports),execution_errors=sum(r['execution_errors'] for r in rows))
(a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Yifan: bimanual CAP and BC pilot','',protocol['notes'],'',
 '| Phase | Policy | Input | Split | Charts | Complete | Execution / policy errors |',
 '|---|---|---|---|---:|---:|---:|']
for r in rows:
 rate='unavailable' if r['completion'] is None else f"{r['completion']:.1%}"
 lines.append(f"| {r['phase']} | {r['policy']} | {r['detector']} | {r['split']} | {r['charts']} | {r['complete']}/{r['notes']} ({rate}) | {r['execution_errors']} / {r['policy_runtime_errors']} |")
lines+=['','Completion above is note-weighted; chord charts contain more notes than crossing charts. The table below exposes sustain performance and gives each chart equal weight.','',
 '| Policy | Input | Tap | Hold | Slide | Mean chart completion |',
 '|---|---|---:|---:|---:|---:|']
for policy in ['classical','bc-mlp','bc-diffusion','cap-v0','cap-v1','cap-v2']:
 for detector in ['oracle','color']:
  subset=[r for r in rows if r['phase']=='test' and r['policy']==policy and r['detector']==detector]
  if not subset:continue
  scores=[]
  for kind in ['tap','hold','slide']:
   done=sum(r['by_kind'][kind]['complete'] for r in subset);total=sum(r['by_kind'][kind]['total'] for r in subset)
   scores.append(f'{done}/{total}')
  mean='unavailable' if any(r['chart_mean_completion'] is None for r in subset) else f"{sum(r['chart_mean_completion']*r['charts'] for r in subset)/sum(r['charts'] for r in subset):.1%}"
  lines.append(f"| {policy} | {detector} | {' | '.join(scores)} | {mean} |")
lines+=['','Invalid runs are reported separately; no complete-rate claim is made for cells with execution errors. CAP runtime exceptions are counted even when guarded fallback permits the physics run to finish. Full per-note metrics remain in eval/.','']
(a.out/'summary.md').write_text('\n'.join(lines))
fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
policies=['classical','bc-mlp','bc-diffusion','cap-v0','cap-v1','cap-v2']
x=np.arange(len(policies))
for offset,det,color in [(-.18,'oracle','#138b95'),(.18,'color','#d59632')]:
 vals=[]
 for p in policies:
  subset=[r for r in rows if r['phase']=='test' and r['policy']==p and r['detector']==det]
  vals.append(100*sum(r['complete'] for r in subset)/sum(r['notes'] for r in subset) if subset and not any(r['execution_errors'] for r in subset) else np.nan)
 axs[0].bar(x+offset,vals,.35,label=det,color=color)
axs[0].set(xticks=x,xticklabels=policies,ylim=(0,105),ylabel='Completed notes (%)',title='Held-out full Drake simulation')
axs[0].tick_params(axis='x',rotation=25);axs[0].legend()
for phase,label,color in [('feedback','Training feedback','#138b95'),('test','Held-out oracle','#8c55be')]:
 vals=[]
 for version in range(3):
  selected=[r for r in rows if r['phase']==phase and r['policy']==f'cap-v{version}' and r['detector']=='oracle']
  vals.append(100*sum(r['complete'] for r in selected)/sum(r['notes'] for r in selected) if selected and not any(r['execution_errors'] for r in selected) else np.nan)
 axs[1].plot(range(3),vals,'o-',label=label,color=color)
axs[1].set(xticks=range(3),xticklabels=['v0','v1','v2'],ylim=(0,105),ylabel='Completed notes (%)',title='CAP revisions (one generation chain)');axs[1].legend()
for ax in axs:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.savefig(a.out/'overview.png',dpi=160);fig.savefig(a.out/'overview.pdf')
print(json.dumps({'runs':len(reports),'errors':summary['execution_errors']}))
