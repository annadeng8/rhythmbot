"""Verify learned-policy deployment on validation charts before freezing tests."""
import json
import os
from pathlib import Path
from rhythmbot.final_study import prepare,run_jobs
def main():
    out=Path(os.environ['STUDY_OUT']);p=prepare(out)
    chart=out/next(e['path'] for e in p['entries'] if e['split']=='validation')
    jobs=[]
    for head in ('mlp','diffusion'):
        training=json.loads((out/f'bc-{head}-s0-training.json').read_text())
        assert len(training['epochs'])==p['bc_epochs']
        jobs.append(dict(mode='eval',chart=str(chart),condition='camera',
            policy=f'bc:{out}/checkpoints/bc-{head}-s0.pt',label=f'bc-{head}-s0',
            out=str(out/f'preflight/deployment-warmup/{head}.json')))
    run_jobs(jobs,2)

if __name__=="__main__":main()
