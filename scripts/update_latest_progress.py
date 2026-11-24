"""Build the concise November 24 report from committed study summaries."""
from pathlib import Path
import json
import statistics
root=Path(__file__).resolve().parents[1]
old=json.loads((root/'results/remaining-study-v1/summary.json').read_text())
new=json.loads((root/'results/slide-path-v1/sekai-camera/summary.json').read_text())
ft=json.loads((root/'results/sekai-finetune-v1/summary.json').read_text())
rows=[]
for policy,label in [('classical','Classical'),('cap-v0','CAP initial'),('cap-v1','CAP feedback 1'),('cap-v2','CAP feedback 2'),('bc-mlp','MLP BC'),('bc-diffusion','Diffusion BC')]:
 vals=[]
 for det in ['concert','concert-torch']:
  group=[r for r in old if r['phase']=='compare' and r['detector']==det and (r['policy']==policy or r['policy'].startswith(policy+'-s'))]
  assert len(group)==(3 if policy.startswith('bc-') else 1)
  vals.append(f"{statistics.mean(r['completion'] for r in group)*100:.1f}\\%")
 for source in [ft,new]:
  group=[r for r in source if r['policy']==policy and r['detector']=='v7-fine-tuned']
  if group:
   assert len(group)==1 and group[0]['charts']==11 and group[0]['notes']==323
   vals.append(f"{100*group[0]['completed']/group[0]['notes']:.1f}\\%")
  else:vals.append('---')
 rows.append(label+' & '+' & '.join(vals)+r'\\')
source=root/'output/pdf/rhythmbot-progress-source/progress.tex'
oldtex=source.read_text()
fig=oldtex[oldtex.index(r'\begin{figure}'):oldtex.index(r'\end{figure}')+len(r'\end{figure}')]
text=r'''\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[hidelinks]{hyperref}
\title{RhythmBot: Project Progress Report}
\author{Anna Deng \and Yifan Kang}
\date{November 24, 2026}
\begin{document}
\maketitle
\section*{Current progress}
We have built single-arm and dual-arm Drake environments for taps, holds, and
continuous slides. Our dual-arm setup uses two KUKA iiwa arms with one stylus
each, a minimal Project SEKAI-inspired display, and simulated camera RGB with
robot occlusion. We compare classical planning, Code as Policy (CAP), and
behavior cloning (BC). CAP uses Codex-generated Python controllers; initial,
feedback 1, and feedback 2 denote successive programs revised using training
feedback. BC uses either a multilayer perceptron (MLP) or a diffusion action
model, with three training seeds each. We completed the original 330-run study,
220 camera-detector comparisons, and 66 live BC tests. We then fine-tuned our
original neural detector (v7) on 29,112 camera frames from training charts and
ran 132 further comparisons. A subsequent tracker change combines slide-path
predictions across frames; 44 camera runs evaluate that change. Perception
receives pixels and time, never ground-truth note state.
FIGURE
\newpage
\section*{Results and next steps}
We find that camera fine-tuning improves classical completion from 84.5\% to
87.0\%; combining slide predictions raises it to 87.6\%, still below the
89.2\% color/shape baseline. Classical slide completion rises from 0/16 to
7/16 and then 9/16, versus color's 10/16. CAP feedback 1 reaches 80.2\% with
the adapted detector and tracker, compared with 78.6\% using color detection.
Thus neural adaptation helps, but its benefit depends on the controller and
slide tracking. Earlier frozen-detector results improved MLP BC from 37.3\%
to 55.3\%, while diffusion remained near 56\%. BC has not yet been evaluated
with the fine-tuned detector. Live MLP averages approximately real-time speed,
but only 17/33 runs meet our one-frame lag limit; diffusion meets it on 0/33.
Next we will evaluate adapted perception with BC, reduce latency, and test the
slide fix on fresh charts. The follow-up comparisons reuse the original test
set after failure diagnosis, so independent confirmation remains necessary.
\begin{table}[ht]
\centering
\small
\begin{tabular}{lrrrr}
\toprule
Controller & Color/shape & Original neural & Fine-tuned & + Path merge\\
\midrule
ROWS
\bottomrule
\end{tabular}
\caption{Dual-arm camera note completion on 11 charts (323 notes), including
sustain contact and tails. Original neural means the frozen detector without
camera adaptation; fine-tuned uses camera training data; path merge additionally
combines successive slide predictions. BC values average three training seeds;
dashes indicate experiments not yet run. Original columns retain the earlier
study results; the later columns come from the follow-up evaluations.}
\end{table}
\begin{table}[ht]
\centering
\begin{tabular}{lcc}
\toprule
Live controller & Wall / simulation time & One-frame lag passes\\
\midrule
MLP BC & 1.002 & 17 / 33\\
Diffusion BC & 1.262 & 0 / 33\\
\bottomrule
\end{tabular}
\caption{Earlier sequential Apple M5 tests with both arms, color/shape vision,
a 640$\times$480 camera, and Meshcat publishing. The lag limit is 33.3\,ms.
Startup and browser latency are excluded. Accuracy comparisons above use
960$\times$720; the later tracker change has not been retested for live timing.}
\end{table}
\end{document}
'''
source.write_text(text.replace('FIGURE',fig).replace('ROWS','\n'.join(rows)))
print(source)
