"""Fine-tuned v7 on the dual-arm Sekai camera, against zero-shot v7 and color.

    python scripts/eval_sekai_finetune.py --model results/note_detector_v7_sekai.pt \
        --out results/sekai-finetune-v1 --workers 5

Runs the 11 held-out `remaining-study-v1` charts with the same settings as
its `compare` phase (960x720 camera, unpaced, no Meshcat, inference seed 0,
`realtime.run`) for every controller available locally: classical and the
three frozen CAP programs. The BC checkpoints live on gpu156 and are not
run. Three detectors per controller: `concert` (color/shape), `concert-torch`
with the original v7 (zero-shot) and with the fine-tuned model. The first
two repeat the frozen study on this machine, so the fine-tuned runs are
compared against same-machine, same-code baselines. Completed reports are
kept, so a rerun resumes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROTOCOL = ROOT / 'results/remaining-study-v1/protocol.json'
V7 = 'results/note_detector_v7.pt'


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def execute(job):
    import torch
    from rhythmbot.chart import Chart
    from rhythmbot.realtime import run
    dest = Path(job['out'])
    if dest.exists():
        saved = json.loads(dest.read_text())
        if saved['execution_status'] == 'ok':
            return saved
    torch.set_num_threads(1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        rep = run(Chart.load(ROOT / job['chart']), dest.with_suffix('.partial.json'),
                  policy=job['spec'], detector=job['detector'], model_path=job['model'],
                  camera_width=960, paced=False, meshcat=False, seed=0)
        rep['execution_status'] = 'ok'
    except Exception:
        rep = dict(execution_status='error', error=traceback.format_exc())
    rep.update(job=job, total_wall_seconds=time.time() - started)
    dest.write_text(json.dumps(rep, indent=2) + '\n')
    dest.with_suffix('.partial.json').unlink(missing_ok=True)
    return rep


def summarize(out: Path, labels):
    groups = defaultdict(list)
    for f in sorted((out / 'runs').rglob('*.json')):
        if f.name.endswith('.partial.json'):
            continue
        r = json.loads(f.read_text())
        if r['execution_status'] != 'ok':
            continue
        groups[(r['job']['policy'], r['job']['label'])].append(r)
    rows = []
    for (policy, label), rs in sorted(groups.items()):
        notes = [n for r in rs for n in r['notes']]
        kinds = {k: (sum(n['kind'] == k and bool(n['completed']) for n in notes),
                     sum(n['kind'] == k for n in notes)) for k in ('tap', 'hold', 'slide')}
        rows.append(dict(policy=policy, detector=label, charts=len(rs), notes=len(notes),
                         completed=sum(bool(n['completed']) for n in notes), by_kind=kinds,
                         compute_per_sim=sum(r['compute_seconds'] for r in rs) / sum(r['sim_seconds'] for r in rs)))
    (out / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    lines = ['| Controller | Detector | Charts | Completed | Taps | Holds | Slides | Compute / sim s |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    order = {l: k for k, l in enumerate(labels)}
    for r in sorted(rows, key=lambda r: (r['policy'], order.get(r['detector'], 99))):
        k = r['by_kind']
        lines.append(f"| {r['policy']} | {r['detector']} | {r['charts']} | "
                     f"{r['completed']}/{r['notes']} ({r['completed'] / r['notes']:.1%}) | "
                     + ' | '.join(f'{a}/{b}' for a, b in k.values()) + f" | {r['compute_per_sim']:.2f} |")
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, help='fine-tuned TorchScript detector')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--policies', nargs='*', help='labels to run (default: classical and CAP)')
    p.add_argument('--detectors', nargs='*', help='detector labels to run (default: all three)')
    a = p.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    policies = [q for q in protocol['policies'] if not q['label'].startswith('bc-')]
    if a.policies:
        policies = [q for q in policies if q['label'] in a.policies]
    detectors = [('color', 'concert', None), ('v7-zero-shot', 'concert-torch', V7),
                 ('v7-fine-tuned', 'concert-torch', a.model)]
    run_labels = a.detectors or [l for l, _, _ in detectors]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'setup.json').write_text(json.dumps(dict(
        charts=[c['path'] for c in protocol['charts']], policies=policies,
        detectors=[dict(label=l, detector=d, model=m, sha256=m and (ROOT / m).exists() and sha(m))
                   for l, d, m in detectors],
        settings=dict(camera_width=960, paced=False, meshcat=False, seed=0, runner='rhythmbot.realtime.run'),
        note='Local Apple M5 rerun; BC checkpoints are on gpu156 and not included.'), indent=2) + '\n')
    jobs = []
    for q in policies:
        for c in protocol['charts']:
            assert sha(c['path']) == c['sha256'], c['path']
            for label, detector, model in detectors:
                if label not in run_labels:
                    continue
                key = f"{label}/{q['label']}/{Path(c['path']).stem}"
                jobs.append(dict(key=key, policy=q['label'], spec=q['spec'], chart=c['path'], label=label,
                                 detector=detector, model=model, out=str(a.out / 'runs' / (key + '.json'))))
    failed = 0
    with ProcessPoolExecutor(a.workers, max_tasks_per_child=4,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(execute, j) for j in jobs]
        for n, f in enumerate(as_completed(futures), 1):
            r = f.result()
            failed += r['execution_status'] != 'ok'
            print(f"{n}/{len(jobs)} ({failed} errors) {r['job']['key']}: "
                  f"{r.get('score', '-')}/{r.get('max_score', '-')}", flush=True)
    summarize(a.out, [l for l, _, _ in detectors])


if __name__ == '__main__':
    main()
