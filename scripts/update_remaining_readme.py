"""Insert verified metrics into the study README without hand-transcribed numbers."""
import json
from pathlib import Path
import statistics

out=Path('results/remaining-study-v1')
v=json.loads((out/'verification.json').read_text());assert v['execution_errors']==0
s=json.loads((out/'summary.json').read_text())
def group(phase,detector,label):
 return [r for r in s if r['phase']==phase and r['detector']==detector and
  (r['policy']==label or r['policy'].startswith(label+'-s'))]
def score(det,label):
 vals=[100*r['completion'] for r in group('compare',det,label)]
 return (f'{statistics.mean(vals):.1f} ± {statistics.stdev(vals):.1f}%' if len(vals)>1 else f'{vals[0]:.1f}%')
rows=['<!-- VERIFIED RESULTS -->',
 f"**Verified: {v['compare_runs']} paired-vision runs and {v['live_runs']} live BC runs; zero execution or CAP runtime errors.**",
 '', '| Controller | Color/shape completion | Neural completion |', '|---|---:|---:|']
for label,name in [('classical','Classical'),('cap-v0','CAP initial'),('cap-v1','CAP feedback 1'),('cap-v2','CAP feedback 2'),('bc-mlp','MLP BC'),('bc-diffusion','Diffusion BC')]:
 rows.append(f'| {name} | {score("concert",label)} | {score("concert-torch",label)} |')
rows+=['','BC values are mean ± sample SD across three training seeds. All policies use the same 11 charts.','',
 '| Live controller | Completion (640×480) | Wall seconds / simulated second | One-frame lag passes | Worst accumulated lag |', '|---|---:|---:|---:|---:|']
for label in ('bc-mlp','bc-diffusion'):
 rs=group('live','concert',label)
 rows.append(f"| {label} | {100*statistics.mean(r['completion'] for r in rs):.1f}% | {statistics.mean(r['wall_per_sim'] for r in rs):.3f} | {sum(r['realtime_pass_runs'] for r in rs)}/33 | {max(r['max_lag_seconds'] for r in rs):.3f} s |")
rows+=['','![Paired accuracy and live timing](overview.png)','<!-- END VERIFIED RESULTS -->']
p=out/'README.md';text=p.read_text();start='<!-- VERIFIED RESULTS -->';end='<!-- END VERIFIED RESULTS -->'
if start in text:
 a=text.index(start);b=text.index(end)+len(end);text=text[:a]+'\n'.join(rows)+text[b:]
else:text=text.replace('## Results and audit\n','## Results and audit\n\n'+'\n'.join(rows)+'\n')
p.write_text(text)
