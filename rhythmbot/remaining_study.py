"""Frozen completion study: paired neural camera evaluation and live BC trials.

prepare freezes the source, original held-out charts and existing weights.
compare runs 11 charts x 10 controllers x 2 camera detectors on gpu156.
live runs all six BC checkpoints on all 11 charts, sequentially with Meshcat.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import time
import traceback

from .experiments import source_hash, write_json, digest

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'results/yifan-final-v1'
MODEL = ROOT / 'results/note_detector_v7.pt'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out):
    original = json.loads((OLD / 'protocol.json').read_text())
    frozen = json.loads((OLD / 'freeze.json').read_text())
    charts = [dict(path=str((OLD / e['path']).relative_to(ROOT)), sha256=e['sha256'],
                   split=e['split'], family=e['family'])
              for e in original['entries'] if e['split'].startswith('test')]
    policies = [dict(label='classical', spec='classical', sha256=None)]
    for label, item in frozen['artifacts'].items():
        rel = (Path('checkpoints') / (label + '.pt') if label.startswith('bc-') else
               Path('cap') / label.removeprefix('cap-') / 'policy.py')
        path = OLD / rel
        assert sha(path) == item['sha256'], path
        policies.append(dict(label=label, spec=('bc:' if label.startswith('bc-') else 'cap:') +
                             str(path.relative_to(ROOT)), sha256=item['sha256']))
    for c in charts:
        assert sha(ROOT / c['path']) == c['sha256']
    protocol = dict(source_sha256=source_hash(), charts=charts, policies=policies,
        neural_model=str(MODEL.relative_to(ROOT)), neural_sha256=sha(MODEL),
        neural_sidecar_sha256=sha(MODEL.with_suffix('.json')),
        neural_state_sha256=sha(MODEL.with_suffix('.state.pt')),
        inference_seed=0, compare=dict(camera_width=960, meshcat=False, paced=False,
            detectors=['concert', 'concert-torch'], expected_runs=220, host='gpu156'),
        live=dict(camera_width=640, meshcat=True, paced=True, detector='concert',
                  expected_runs=66, host='local Mac', controllers='all six frozen BC checkpoints'),
        design='Zero-shot frozen Anna v7; no palette reconstruction, fine-tuning or test selection. '
               'Same calibrated RGB warp, camera, 1ms physics and strict judge for both comparison detectors. '
               'Diffusion uses original 10 sampling steps and 4-tick execution, CPU eager inference. '
               'Three training seeds per BC head, one fixed inference seed per run. '
               'Live throughput runs are sequential; comparison worker timing is not real-time evidence.')
    path = out / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol, 'Existing protocol differs; use a new output directory'
    else:
        write_json(path, protocol)
    return protocol


def jobs(protocol, phase, out):
    result = []
    for policy in protocol['policies']:
        if phase == 'live' and not policy['label'].startswith('bc-'):
            continue
        for chart in protocol['charts']:
            for detector in (protocol['compare']['detectors'] if phase == 'compare' else ['concert']):
                key = f"{phase}/{detector}/{policy['label']}/{Path(chart['path']).stem}"
                result.append(dict(key=key, phase=phase, policy=policy, chart=chart,
                    detector=detector, out=str(out / 'runs' / (key + '.json'))))
    return result


def execute(job, protocol):
    from .chart import Chart
    from .realtime import run
    identity = digest(dict(protocol=digest(protocol), key=job['key']))
    dest = Path(job['out'])
    if dest.exists():
        saved = json.loads(dest.read_text())
        assert saved['identity'] == identity and saved['execution_status'] == 'ok', dest
        return saved
    assert source_hash() == protocol['source_sha256']
    assert sha(ROOT / job['chart']['path']) == job['chart']['sha256']
    policy = job['policy']
    if policy['sha256'] is not None:
        assert sha(ROOT / policy['spec'].partition(':')[2]) == policy['sha256']
    assert sha(MODEL) == protocol['neural_sha256']
    assert sha(MODEL.with_suffix('.json')) == protocol['neural_sidecar_sha256']
    assert sha(MODEL.with_suffix('.state.pt')) == protocol['neural_state_sha256']
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix('.partial.json')
    started = time.time()
    def timeout(signum, frame):
        raise TimeoutError('Simulation exceeded 900 seconds')
    old_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(900)
    try:
        settings = protocol[job['phase']]
        rep = run(Chart.load(ROOT / job['chart']['path']), partial,
            policy=policy['spec'], detector=job['detector'], model_path=str(MODEL),
            camera_width=settings['camera_width'], paced=settings['paced'],
            meshcat=settings['meshcat'], seed=protocol['inference_seed'])
        rep['execution_status'] = 'ok'
    except Exception:
        rep = dict(execution_status='error', error=traceback.format_exc())
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    rep.update(identity=identity, job=job, protocol_sha256=digest(protocol),
               total_wall_seconds=time.time() - started,
               completed_at=datetime.now(timezone.utc).isoformat())
    write_json(dest, rep)
    partial.unlink(missing_ok=True)
    return rep


def run_phase(out, phase, workers):
    protocol = json.loads((out / 'protocol.json').read_text())
    assert source_hash() == protocol['source_sha256']
    if phase == 'live' and workers != 1:
        raise ValueError('Live timing requires exactly one simulator worker')
    tasks = jobs(protocol, phase, out)
    # Recycle processes to bound PyTorch/Drake retained allocations.
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=4,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(execute, task, protocol) for task in tasks]
        failed = 0
        for n, future in enumerate(as_completed(futures), 1):
            report = future.result()
            failed += report['execution_status'] != 'ok'
            print(f"{phase}: {n}/{len(tasks)} complete, {failed} errors; {report['job']['key']}", flush=True)
    if failed:
        raise RuntimeError(f'{failed} failed runs; inspect reports, do not count as passed')


def verify(out):
    p = json.loads((out / 'protocol.json').read_text())
    assert prepare(out) == p  # Recheck chart, policy, model and sidecar hashes.
    assert source_hash() == p['source_sha256']
    rows = []
    configs = {}
    for phase in ('compare', 'live'):
        expected = jobs(p, phase, out)
        assert len(expected) == p[phase]['expected_runs']
        actual = list((out / 'runs' / phase).rglob('*.json'))
        assert len(actual) == len(expected), (phase, len(actual), len(expected))
        for j in expected:
            r = json.loads(Path(j['out']).read_text())
            assert r['job']['key'] == j['key'] and r['job']['chart'] == j['chart']
            assert r['execution_status'] == 'ok', j['key']
            assert r['identity'] == digest(dict(protocol=digest(p), key=j['key']))
            assert r['source_sha256'] == p['source_sha256']
            assert len(r['executed_arms']) == 2 and r['config']['gripper']['single_tip']
            assert r['view'] == 'camera' and r['detector'] == j['detector']
            assert r['vision']['chart_access'] is False and r['vision']['label_input'] is False
            assert r['vision']['frames'] > 0 and r['seed'] == 0
            assert all(n['completed'] is not None for n in r['notes'])
            assert all(s.get('plan_errors', 0) == 0 for s in r['policy_stats'])
            assert r['paced'] == p[phase]['paced'] and r['meshcat'] == p[phase]['meshcat']
            cfg = json.dumps(r['config'], sort_keys=True)
            if phase in configs:
                assert configs[phase] == cfg, 'Unequal physical/camera configuration'
            configs[phase] = cfg
            if j['detector'] == 'concert-torch':
                assert r['vision']['model_sha256'] == p['neural_sha256']
                assert r['vision']['active_path'] and not r['vision']['fine_tuned']
            rows.append(r)
    result = dict(execution_errors=0, policy_errors=0, compare_runs=220, live_runs=66,
        source_sha256=p['source_sha256'], protocol_sha256=digest(p),
        note='Completed experiment coverage, not universal note success or a real-time pass.')
    write_json(out / 'verification.json', result)
    return rows


def summarize(out):
    from collections import defaultdict
    rows = verify(out)
    groups = defaultdict(list)
    for r in rows:
        j = r['job']
        groups[(j['phase'], j['detector'], j['policy']['label'])].append(r)
    summary = []
    for (phase, detector, label), rs in sorted(groups.items()):
        notes = [n for r in rs for n in r['notes']]
        sim = sum(r['sim_seconds'] for r in rs)
        summary.append(dict(phase=phase, detector=detector, policy=label, charts=len(rs),
            notes=len(notes), completed=sum(bool(n['completed']) for n in notes),
            completion=sum(bool(n['completed']) for n in notes)/len(notes),
            score=sum(r['score'] for r in rs), max_score=sum(r['max_score'] for r in rs),
            wall_per_sim=sum(r['wall_seconds'] for r in rs)/sim,
            compute_per_sim=sum(r['compute_seconds'] for r in rs)/sim,
            realtime_pass_runs=sum(r['realtime_pass'] for r in rs),
            max_lag_seconds=max(r['max_lag_seconds'] for r in rs),
            by_kind={k:dict(n=sum(n['kind']==k for n in notes),
                completed=sum(n['kind']==k and bool(n['completed']) for n in notes)) for k in ('tap','hold','slide')}))
    write_json(out / 'summary.json', summary)
    lines = ['| Phase | Detector | Policy | Charts | Completed | Wall/sim | Compute/sim | One-frame lag passes | Max lag s |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['phase']} | {r['detector']} | {r['policy']} | {r['charts']} | "
            f"{r['completed']}/{r['notes']} ({r['completion']:.1%}) | {r['wall_per_sim']:.3f} | "
            f"{r['compute_per_sim']:.3f} | {r['realtime_pass_runs']}/{r['charts']} | {r['max_lag_seconds']:.3f} |")
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'compare', 'live', 'verify', 'summarize'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.out)
    elif args.command in ('compare', 'live'):
        run_phase(args.out, args.command, args.workers)
    elif args.command == 'verify':
        print(json.dumps(verify(args.out) and json.loads((args.out/'verification.json').read_text()), indent=2))
    else:
        summarize(args.out)


if __name__ == '__main__':
    main()
