"""Regenerate the progress-report figure from committed single-arm results.

Run from any directory with Python and matplotlib.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
rows={}
for line in (ROOT / 'results/eval-v2/summary.md').read_text().splitlines():
    c=[s.strip() for s in line.strip('|').split('|')]
    if len(c)>6 and c[0] in ('classical','cap-policy_v7','cap-policy_v1') and c[2] in ('heldout (all)','heldout-song'):
        rows[c[0],c[1],c[2]]=float(c[5].strip('%'))
plt.rcParams.update({'font.family':'serif','font.size':10})
fig,axs=plt.subplots(1,2,figsize=(8,3.3),sharey=True)
keys=['classical','cap-policy_v7','cap-policy_v1']
for ax,split,title in zip(axs,['heldout (all)','heldout-song'],['All 24 charts (1,810 notes)','Unseen song (1,106 notes)']):
    x=np.arange(3)
    for shift,det,label,color in [(-.18,'oracle','Oracle','#555555'),(.18,'torch','Trained vision','#a6b9cf')]:
        vals=[rows[k,det,split] for k in keys]
        bars=ax.bar(x+shift,vals,.35,label=label,color=color,edgecolor='black',linewidth=.4)
        ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=3)
    ax.set_xticks(x,['Classical','CAP: feedback','CAP: self-test'])
    ax.set_ylim(0,112); ax.set_yticks([0,20,40,60,80,100]); ax.set_title(title,fontsize=11)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
axs[0].set_ylabel('Note completion (%)')
axs[1].legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=2,frameon=False,fontsize=9)
fig.tight_layout()
fig.savefig(ROOT / 'output/pdf/rhythmbot-progress-source/assets/evaluation-results.pdf',bbox_inches='tight')
