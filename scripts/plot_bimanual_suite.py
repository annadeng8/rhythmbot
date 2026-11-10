"""Plot the paired benchmark summary without recomputing or filtering outcomes."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('suite',type=Path)
args=ap.parse_args()
s=json.loads((args.suite/'summary.json').read_text())
if s['errors'] or s['missing']:raise SystemExit('Refusing to present an incomplete suite as finished')
rows=s['rows']
def row(family,tool='stylus',detector='oracle',handoff=True,speed=None):
    return next(r for r in rows if (r['family'],r['tool'],r['detector'],r['handoff'],r['speed'])==(family,tool,detector,handoff,speed))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
colors=['#138b95','#8c55be']
speeds=[.3,.4,.5,.6,.8]
for tool,color,label in zip(['stylus','two_finger'],colors,['Single-tip stylus','Two-finger WSG']):
    rates=[]
    for t in speeds:
        selected=[]
        for path in (args.suite/'runs').glob('*.json'):
            report=json.loads(path.read_text());job=report['experiment']
            if (job['family'],job['speed'],job['tool'],job['detector'],job['handoff'])==('crossing',t,tool,'oracle',True):
                selected.append(report['notes'][0]['completed'])
        rates.append(100*sum(selected)/len(selected))
    axs[0,0].plot(speeds,rates,'o-' if tool=='stylus' else 's--',color=color,label=label,markerfacecolor=color if tool=='stylus' else 'none')
axs[0,0].set(title='A  Four-crossing zigzag only',xlabel='Seconds per crossing (slower →)',ylabel='Completed notes (%)')
axs[0,0].legend(frameon=False)
x=np.arange(2)
for offset,on,color in [(-.18,True,colors[0]),(.18,False,'#999999')]:
    vals=[row('crossing',handoff=on,speed=t)['micro_completion']*100 for t in [.4,.6]]
    axs[0,1].bar(x+offset,vals,.35,color=color,label='Handoff on' if on else 'Handoff off')
axs[0,1].set(title='B  Coordination ablation (stylus)',xticks=x,xticklabels=['0.4 s','0.6 s'],ylabel='Completed notes (%)')
axs[0,1].legend(frameon=False)
for offset,tool,color in [(-.18,'stylus',colors[0]),(.18,'two_finger',colors[1])]:
    vals=[row(f,tool)['micro_completion']*100 for f in ['holds','chords']]
    axs[1,0].bar(x+offset,vals,.35,color=color,label=tool)
axs[1,0].set(title='C  Parallel holds and chord capacity',xticks=x,xticklabels=['Holds + local slides','Dense/chord stress'],ylabel='Completed notes (%)')
x=np.arange(3)
for offset,det,color in [(-.18,'oracle',colors[0]),(.18,'color','#df9a36')]:
    vals=[row(f,detector=det,speed=t)['micro_completion']*100 for f,t in [('crossing',.6),('holds',None),('chords',None)]]
    axs[1,1].bar(x+offset,vals,.35,color=color,label=det)
axs[1,1].set(title='D  Calibrated-frame perception (stylus)',xticks=x,xticklabels=['Crossing 0.6 s','Holds','Chords'],ylabel='Completed notes (%)')
axs[1,1].legend(loc='lower left',frameon=True,facecolor='white')
axs[1,0].legend(loc='lower left',frameon=True,facecolor='white')
for ax in axs.flat:
    ax.set_ylim(0,105);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('Drake bimanual benchmark • 3 paired chart variants per condition',fontsize=15)
fig.savefig(args.suite/'overview.png',dpi=160)
fig.savefig(args.suite/'overview.pdf')
