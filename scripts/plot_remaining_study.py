"""Plot all registered policies, including failed note outcomes and timing misses."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args()
s=json.loads((a.out/'summary.json').read_text())
labels=['classical','cap-v0','cap-v1','cap-v2','bc-mlp','bc-diffusion']
names=['Classical','CAP\ninitial','CAP\nround 1','CAP\nround 2','BC\nMLP','BC\ndiffusion']
def group(phase,detector,label):
 return [r for r in s if r['phase']==phase and r['detector']==detector and
         (r['policy']==label or r['policy'].startswith(label+'-s'))]
plt.rcParams.update({'font.size':11,'axes.titlesize':11,'axes.labelsize':11})
fig,axes=plt.subplots(1,2,figsize=(9,4.2),layout='constrained')
x=np.arange(len(labels))
for offset,det,color,name in [(-.19,'concert','#777777','Color/shape'),(.19,'concert-torch','#2878a0','Frozen neural detector')]:
 vals=[];err=[]
 for label in labels:
  values=[r['completion']*100 for r in group('compare',det,label)]
  vals.append(np.mean(values));err.append(np.std(values,ddof=1) if len(values)>1 else 0)
 axes[0].bar(x+offset,vals,.36,color=color,label=name,yerr=err,capsize=2)
axes[0].set(xticks=x,xticklabels=names,ylim=(0,105),ylabel='Completed notes (%)',title='Dual-arm camera: 11 held-out charts')
axes[0].tick_params(axis='x',labelsize=10);axes[0].legend(fontsize=10)
for i,head in enumerate(['bc-mlp','bc-diffusion']):
 rs=group('live','concert',head)
 axes[1].scatter([i-.12,i,i+.12],[r['wall_per_sim'] for r in sorted(rs,key=lambda r:r['policy'])],label=head.removeprefix('bc-'),s=45)
axes[1].axhline(1,color='black',linestyle='--',linewidth=1,label='1× playback')
axes[1].set(xticks=[0,1],xticklabels=['MLP (3 seeds)','Diffusion (3 seeds)'],ylabel='Wall seconds / simulated second',title='Sequential live runs with Meshcat')
axes[1].legend(fontsize=10)
for ax in axes:ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
fig.savefig(a.out/'overview.png',dpi=180)
fig.savefig(a.out/'overview.pdf')
