"""Update the existing plain two-page LaTeX report from verified study results."""
import json
from pathlib import Path
import shutil
import statistics

root=Path(__file__).resolve().parents[1]
out=root/'results/remaining-study-v1'
verified=json.loads((out/'verification.json').read_text())
assert (verified['compare_runs'],verified['live_runs'],verified['execution_errors'])==(220,66,0)
s=json.loads((out/'summary.json').read_text())
def group(phase,detector,label):
 return [r for r in s if r['phase']==phase and r['detector']==detector and
  (r['policy']==label or r['policy'].startswith(label+'-s'))]
def mean(phase,detector,label,key):
 return statistics.mean(r[key] for r in group(phase,detector,label))
def pct(detector,label):return f"{100*mean('compare',detector,label,'completion'):.1f}"
def live(label):
 rs=group('live','concert',label)
 return f"{statistics.mean(r['wall_per_sim'] for r in rs):.3f}",sum(r['realtime_pass_runs'] for r in rs)
mlp,mp=live('bc-mlp');diff,dp=live('bc-diffusion')
source=root/'output/pdf/rhythmbot-progress-source'
shutil.copy2(out/'overview.pdf',source/'assets/remaining-results.pdf')
text=r'''\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[hidelinks]{hyperref}
\title{RhythmBot: Project Progress Report}
\author{Anna Deng \and Yifan Kang}
\date{November 23, 2026}
\begin{document}
\maketitle
\section*{Current progress}
We have built single-arm and dual-arm Drake environments for taps, holds, and
continuous slides. We compare classical planning, Code as Policy (CAP), and
behavior cloning (BC). CAP generates executable Python controllers with Codex;
``initial,'' ``feedback 1,'' and ``feedback 2'' denote successive programs revised
using training feedback. BC learns either a multilayer perceptron (MLP) or a
diffusion action model from demonstrations. Our two-arm setup uses one stylus
per arm, a minimal Project SEKAI-inspired screen, and simulated camera RGB
with robot occlusion. We have completed the original 330-run dual-arm study,
then added 220 matched color/shape versus neural-vision runs and 66 sequential
live BC runs. We use 11 held-out charts and three training seeds per BC model.
The frozen neural detector now receives geometrically rectified camera pixels,
including its predicted remaining slide path; it receives no chart state.
\begin{figure}[ht]
\centering
\begin{minipage}[b]{.49\linewidth}
\centering
\includegraphics[width=\linewidth,height=2.7in,keepaspectratio]{assets/single-arm.png}\\[6pt]
(a) Single-arm environment
\end{minipage}\hfill
\begin{minipage}[b]{.49\linewidth}
\centering
\includegraphics[width=\linewidth,height=2.7in,keepaspectratio]{assets/camera.png}\\[6pt]
(b) Dual-arm environment
\end{minipage}
\caption{Our Drake environments with native Meshcat robot rendering. The dual-arm
screen has only lanes, notes, and the judgment line. These are observer views;
control uses a separate fixed simulated RGB camera.}
\end{figure}
\newpage
\section*{Results and next steps}
We completed all registered runs without execution errors, while retaining strict
slide continuity and tail judgments. On the matched 960$\times$720 camera test,
classical completion is COLORCLASS\% with color/shape vision and NEURALCLASS\%
with the frozen neural detector; MLP BC reaches COLORMLP\% and NEURALMLP\%,
and diffusion BC COLORDIFF\% and NEURALDIFF\%, respectively. The neural model
was trained on a different display: this is zero-shot transfer, not retraining
on the new screen. In separate 640$\times$480 live tests with color/shape vision,
Meshcat publishing, and the original ten-step diffusion sampler, MLP and
diffusion use MLPWALL and DIFFWALL wall seconds per simulated second.
Only PASSES of 66 runs meet our maximum-lag limit of one 30\,Hz frame.
These measurements exclude startup and browser display latency. Our next steps
are adapting neural perception using training-only camera data and reducing
controller latency; completed experiments do not imply reliable real-time play.
\begin{figure}[ht]
\centering
\includegraphics[width=\linewidth]{assets/remaining-results.pdf}
\caption{Dual-arm results on the same 11 held-out charts. BC error bars show sample
standard deviation across three training seeds. CAP rounds 1 and 2 are
training-feedback revisions of one initial program. Live points aggregate 11 charts per seed. Accuracy uses full note
completion, including sustain contact and tails.}
\end{figure}
\begin{table}[ht]
\centering
\begin{tabular}{lcc}
\toprule
Live controller & Wall / simulation time & One-frame lag passes\\
\midrule
MLP BC & MLPWALL & MLPPASS / 33\\
Diffusion BC & DIFFWALL & DIFFPASS / 33\\
\bottomrule
\end{tabular}
\caption{Sequential Apple M5 measurements with two Drake arms and Meshcat.
A near-1$\times$ average does not guarantee that individual frames meet deadlines.}
\end{table}
\end{document}
'''
values={'COLORCLASS':pct('concert','classical'),'NEURALCLASS':pct('concert-torch','classical'),
 'COLORMLP':pct('concert','bc-mlp'),'NEURALMLP':pct('concert-torch','bc-mlp'),
 'COLORDIFF':pct('concert','bc-diffusion'),'NEURALDIFF':pct('concert-torch','bc-diffusion'),
 'MLPWALL':mlp,'DIFFWALL':diff,'PASSES':str(mp+dp),'MLPPASS':str(mp),'DIFFPASS':str(dp)}
for key,value in values.items():text=text.replace(key,value)
(source/'progress.tex').write_text(text)
print(source/'progress.tex')
