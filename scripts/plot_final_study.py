"""Plot held-out completion and training-feedback evolution without test selection."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);a=ap.parse_args()
s=json.loads((a.out/'summary.json').read_text());rows=s['rows']
fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
controllers=['classical','cap-v0','cap-v1','cap-v2','bc-mlp','bc-diffusion']
names=['Classical','CAP\ninitial','CAP\nfeedback 1','CAP\nfeedback 2','BC\nMLP','BC\ndiffusion']
x=np.arange(len(controllers))
for offset,c,color in [(-.24,'oracle','#63666b'),(0,'screen','#439da8'),(.24,'camera','#bf763d')]:
    values=[];errors=[]
    for controller in controllers:
        if controller.startswith('bc-'):
            r=next(r for r in s['bc_seed_summary'] if r['head']==controller[3:] and r['condition']==c and r['split']=='all')
            values.append(r['mean']*100);errors.append(r['sd']*100)
        else:
            r=next(r for r in rows if r['phase']=='test' and r['policy']==controller and r['condition']==c and r['split']=='all')
            values.append(r['completion']*100);errors.append(0)
    axes[0].bar(x+offset,values,.23,yerr=errors,capsize=2,label=c,color=color)
axes[0].set(xticks=x,xticklabels=names,ylabel='Completed notes (%)',title='All held-out charts; BC mean ± seed SD',ylim=(0,105))
axes[0].tick_params(axis='x',labelsize=8);axes[0].legend(fontsize=8)
for policy,name,color in [('classical','Classical','#63666b'),('cap-v2','CAP feedback 2','#439da8'),('bc-diffusion','BC diffusion','#bf763d')]:
    values=[]
    for split in ('test_iid','test_speed','test_song'):
        if policy.startswith('bc-'):
            r=next(r for r in s['bc_seed_summary'] if r['head']==policy[3:] and r['condition']=='camera' and r['split']==split)
            values.append(r['mean']*100)
        else:values.append(next(r['completion']*100 for r in rows if r['phase']=='test' and r['policy']==policy and r['condition']=='camera' and r['split']==split))
    axes[1].plot(range(3),values,'o-',label=name,color=color)
axes[1].set(xticks=range(3),xticklabels=['IID','Faster slides','Song excerpts'],ylim=(0,105),title='Camera RGB generalization',ylabel='Completed notes (%)');axes[1].legend(fontsize=8)
for kind,label,color in [('tap','Tap','#63666b'),('hold','Hold','#439da8'),('slide','Slide','#bf763d')]:
    values=[]
    for i in range(3):
        r=next(r for r in rows if r['phase']=='test' and r['policy']==f'cap-v{i}' and r['condition']=='camera' and r['split']=='all')
        k=r['by_kind'][kind];values.append(100*k['completed']/k['notes'])
    axes[2].plot(range(3),values,'o-',label=label,color=color)
axes[2].set(xticks=range(3),xticklabels=['Initial','Feedback 1','Feedback 2'],ylim=(0,105),ylabel='Completed notes (%)',title='CAP camera: outcomes by note type');axes[2].legend(fontsize=8)
for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.savefig(a.out/'overview.png',dpi=180);fig.savefig(a.out/'overview.pdf')
