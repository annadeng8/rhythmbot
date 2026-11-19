"""Frozen CAP/BC experiment in the minimal-display dual-arm environment.

All demonstrations and evaluations run full Drake physics. Camera policies
receive RGB-derived tracked notes, never chart state. Oracle is a labelled
control upper bound. The display, judge and router are shared and fixed.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
import signal
from pathlib import Path
import time
import traceback
import numpy as np
from .chart import Chart, Note
from .experiments import make_chart, write_json, source_hash, digest

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {'oracle': ('oracle', 'screen'), 'screen': ('concert', 'screen'), 'camera': ('concert', 'camera')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out):
    path = out / 'protocol.json'
    if path.exists():
        return json.loads(path.read_text())
    entries = []
    for split, seeds in [('train', range(51000, 51016)), ('validation', range(61000, 61002)), ('test_iid', range(71000, 71002))]:
        for seed in seeds:
            for family in ('crossing', 'holds', 'chords'):
                chart = make_chart(family, seed, (.5, .6, .8)[seed % 3] if family == 'crossing' else None)
                tempo = float(np.random.default_rng(seed + 97531).uniform(.85, 1.15))
                chart = chart.scaled(tempo)
                rel = f'charts/{split}/{chart.name}.json'
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                chart.save(out / rel)
                entries.append(dict(split=split, family=family, seed=seed, tempo=tempo, path=rel, sha256=sha(out / rel)))
    for seed, speed in [(81000, .3), (81001, .4)]:
        chart = make_chart('crossing', seed, speed)
        rel = f'charts/test_speed/{chart.name}.json'
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        chart.save(out / rel)
        entries.append(dict(split='test_speed', family='crossing', seed=seed, path=rel, sha256=sha(out / rel)))
    # Timing/lane/duration metadata already committed in the original evaluation;
    # no outcomes, generated policy decisions or perception outputs are reused.
    for difficulty in ('easy', 'intermediate', 'normal'):
        src = ROOT / f'results/eval-v2/classical/oracle/song-cityyy-unknown-{difficulty}.json'
        data = json.loads(src.read_text())
        start = min(n['time'] for n in data['notes'])
        stop = start + 30.
        ns = [Note(n['kind'], n['lane'], n['time'] - start + 2.5, n['duration'])
              for n in data['notes'] if start <= n['time'] < stop and n['time'] + n['duration'] <= stop]
        assert all(n.kind in ('tap', 'hold') for n in ns)
        chart = Chart(ns, name=f'unknown-{difficulty}-first30s')
        rel = f'charts/test_song/{chart.name}.json'
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        chart.save(out / rel)
        entries.append(dict(split='test_song', family='song_excerpt', path=rel, sha256=sha(out / rel),
                            source=str(src.relative_to(ROOT)), source_sha256=sha(src),
                            original_window_seconds=[start, stop], boundary_rule='exclude notes ending past window'))
    semantic = [digest([n.to_dict() for n in Chart.load(out / e['path']).notes]) for e in entries]
    assert len(semantic) == len(set(semantic)), 'Duplicate chart contents across splits'
    protocol = dict(base_commit='6d5acc2', entries=entries,
        conditions=CONDITIONS, training_conditions=['oracle', 'camera'], bc_seeds=[0, 1, 2],
        bc_heads=['mlp', 'diffusion'], bc_epochs=60,
        cap_feedback_charts=[e['path'] for e in entries if e['split'] == 'train'][:6],
        cap_conditions=['oracle', 'camera'], cap_revisions=3,
        selection='BC minimum whole-chart validation loss; all CAP revisions reported; never select on test.',
        environment='minimal Sekai; 2 iiwa; 1 stylus each; current strict touch/slide judge; fixed shared handoff router',
        demo='Full Drake rollouts. Labels and policy observations use commanded hand state, not measured robot-state inputs.',
        limitations=['one CAP generation chain', 'RGB segmentation is not a trained neural detector',
                     'song results cover predeclared 30-second prefixes, not full songs',
                     'shared handoff routing is not learned arm assignment', 'no validated inter-arm collision avoidance'])
    write_json(path, protocol)
    return json.loads(path.read_text())


def cfg_and_input(condition):
    from .profiles import config
    return config({'preset': 'sekai-camera'}), *CONDITIONS[condition]


def job_identity(job):
    return digest(dict(source=source_hash(), chart=sha(job['chart']),
        condition=job['condition'], policy=sha(job['policy'].split(':', 1)[1]) if ':' in job.get('policy', '') else job.get('policy'),
        seed=job.get('seed', 0), mode=job['mode']))


def execute_job(job):
    os.environ['HEADLESS'] = '1'
    from pydrake.all import Simulator
    from .play import build
    from .policies import make_policy
    from .planner import ClassicalPlanner
    from .bc.train import _Recorder
    from .judge import Judge
    dest = Path(job['out'])
    dest.parent.mkdir(parents=True, exist_ok=True)
    identity = job_identity(job)
    if dest.exists():
        saved = json.loads(dest.read_text())
        if saved.get('identity') != identity:
            raise RuntimeError(f'Cannot reuse mismatched cached run: {dest}')
        if saved.get('execution_status') != 'ok':
            raise RuntimeError(f'Previous run failed; inspect before retrying: {dest}')
        if job['mode'] == 'collect' and sha(dest.with_suffix('.npz')) != saved['dataset_sha256']:
            raise RuntimeError('Demonstration checksum mismatch')
        return saved
    cfg, detector, view = cfg_and_input(job['condition'])
    chart = Chart.load(job['chart'])
    start = time.time()
    def timed_out(signum, frame):
        raise TimeoutError('Drake run exceeded 900 seconds')
    previous_handler = signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(900)
    try:
        policies = None
        if job['mode'] == 'collect':
            policies = [_Recorder(cfg, seed=k) for k in (0, 1)]
            for k, p in enumerate(policies):
                p.expert = ClassicalPlanner(cfg, k)
        elif job['policy'] != 'classical':
            if job['policy'].startswith('bc:'):
                import torch
                torch.set_num_threads(1)
                torch.manual_seed(0)
            policies = [make_policy(job['policy'], cfg, hand=k) for k in (0, 1)]
        diagram, robot, brain, judge = build(chart, cfg, detector, None, None, policies, view=view)
        sim = Simulator(diagram)
        context = sim.get_mutable_context()
        robot.initialize(context)
        end = chart.end_time + 1.
        sim.AdvanceTo(end)
        judge.finalize(end)
        report = judge.report(extra={'config': asdict(cfg), 'sim_seconds': end, 'detector': detector, 'view': view})
        if hasattr(brain.detector, 'stats'):
            report['vision'] = brain.detector.stats()
        if policies:
            report['policy_stats_by_hand'] = [p.stats() if hasattr(p, 'stats') else {} for p in policies]
        if job['mode'] == 'collect':
            obs = np.concatenate([np.asarray(p.obs, dtype=np.float32) for p in policies])
            act = np.concatenate([np.asarray(p.act, dtype=np.float32) for p in policies])
            assert len(obs) > 0 and np.isfinite(obs).all() and np.isfinite(act).all()
            np.savez_compressed(dest.with_suffix('.npz'), obs=obs, act=act)
            report.update(samples=len(obs), dataset_sha256=sha(dest.with_suffix('.npz')))
        report['execution_status'] = 'ok'
    except Exception:
        report = dict(execution_status='error', error=traceback.format_exc())
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
    report.update(identity=identity, job=job, source_sha256=source_hash(),
                  chart_sha256=sha(job['chart']), wall_seconds=time.time()-start)
    write_json(dest, report)
    return report


def run_jobs(jobs, workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(execute_job, j) for j in jobs]
        for index, f in enumerate(as_completed(futures), 1):
            result = f.result()
            print(index, len(jobs), result['job']['condition'], Path(result['job']['chart']).stem,
                  result.get('completion_rate'), result['execution_status'], flush=True)
            if result['execution_status'] != 'ok':
                raise RuntimeError(result['error'])


def collect(out, workers, warmup=False):
    protocol = prepare(out)
    entries = [e for e in protocol['entries'] if e['split'] in ('train', 'validation')]
    if warmup:
        entries = entries[:1]
    jobs = [dict(mode='collect', chart=str(out / e['path']), condition=c,
                 out=str(out / 'demos' / e['split'] / c / (Path(e['path']).stem + '.json')))
            for e in entries for c in protocol['training_conditions']]
    run_jobs(jobs, workers)
    if warmup:
        return
    for split in ('train', 'validation'):
        files = sorted((out / 'demos' / split).rglob('*.npz'))
        expected = 2 * sum(e['split'] == split for e in entries)
        assert len(files) == expected
        parts = [np.load(f) for f in files]
        arrays = {k: np.concatenate([p[k] for p in parts]) for k in ('obs', 'act')}
        np.savez_compressed(out / (split + '.npz'), **arrays)
        write_json(out / (split + '-data.json'), dict(rollouts=len(files), samples=len(arrays['obs']),
            sha256=sha(out / (split + '.npz')), conditions=protocol['training_conditions']))


def fit(out, head, seed):
    import torch
    from .bc.models import BCModel
    protocol = prepare(out)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.cuda.set_per_process_memory_fraction(.12)
    train, val = np.load(out / 'train.npz'), np.load(out / 'validation.npz')
    model = BCModel('state', head, 16, 256)
    flat = train['act'].reshape(len(train['act']), -1)
    model.act_mean.copy_(torch.from_numpy(flat.mean(0)))
    model.act_std.copy_(torch.from_numpy(flat.std(0) + 1e-3))
    model.encoder.obs_mean.copy_(torch.from_numpy(train['obs'].mean(0)))
    model.encoder.obs_std.copy_(torch.from_numpy(train['obs'].std(0) + 1e-3))
    model.cuda()
    O, A = torch.from_numpy(train['obs']).cuda(), torch.from_numpy(train['act']).cuda()
    V, VA = torch.from_numpy(val['obs']).cuda(), torch.from_numpy(val['act']).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    checkpoint = out / 'checkpoints' / f'bc-{head}-s{seed}.pt'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best, history = float('inf'), []
    latest = out / 'checkpoints' / f'training-state-{head}-s{seed}.pt'
    dataset_sha = sha(out / 'train.npz')
    if latest.exists():
        state = torch.load(latest, map_location='cuda', weights_only=False)
        assert state['training_sha256'] == dataset_sha
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        best, history = state['best'], state['history']
        torch.set_rng_state(state['cpu_rng'].cpu())
        torch.cuda.set_rng_state(state['cuda_rng'].cpu())
    for epoch in range(len(history), protocol['bc_epochs']):
        model.train()
        total = 0.
        for ids in torch.randperm(len(O), device='cuda').split(256):
            loss = model.loss({'obs': O[ids], 'act': A[ids]})
            assert torch.isfinite(loss)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step(); total += float(loss.detach()) * len(ids)
        model.eval()
        value = 0.
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(991)
            with torch.no_grad():
                for i in range(0, len(V), 256):
                    value += float(model.loss({'obs': V[i:i+256], 'act': VA[i:i+256]})) * len(V[i:i+256])
        value /= len(V)
        history.append(dict(epoch=epoch+1, train_loss=total/len(O), validation_loss=value))
        if value < best:
            best = value
            torch.save(dict(arch=model.arch, state_dict=model.state_dict(), tool_mode='bimanual_stylus',
                exec_steps=1 if head=='mlp' else 4, seed=seed, epoch=epoch+1, validation_loss=value), checkpoint)
        temporary = latest.with_suffix('.tmp')
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), best=best,
            history=history, cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
            training_sha256=dataset_sha), temporary)
        temporary.replace(latest)
        write_json(out / f'bc-{head}-s{seed}-training.json', dict(head=head, seed=seed, epochs=history,
            checkpoint_sha256=sha(checkpoint), training_sha256=sha(out / 'train.npz'),
            validation_sha256=sha(out / 'validation.npz'), best_validation_loss=best))
        print(head, seed, history[-1], flush=True)


def evaluate(out, split, specs, labels, conditions, workers):
    protocol = prepare(out)
    if split == 'test':
        frozen = json.loads((out / 'freeze.json').read_text())
        assert frozen['source_sha256'] == source_hash()
        assert frozen['protocol_sha256'] == sha(out / 'protocol.json')
        for spec in specs:
            if ':' in spec:
                assert sha(spec.split(':',1)[1]) in [a['sha256'] for a in frozen['artifacts'].values()]
    if split == 'feedback':
        entries = [e for e in protocol['entries'] if e['path'] in protocol['cap_feedback_charts']]
    else:
        entries = [e for e in protocol['entries'] if e['split'].startswith(split)]
    jobs = [dict(mode='eval', chart=str(out / e['path']), condition=c, policy=spec, label=label,
                 out=str(out / 'eval' / split / label / c / (Path(e['path']).stem + '.json')))
            for spec, label in zip(specs, labels) for c in conditions for e in entries]
    run_jobs(jobs, workers)


def feedback(out, revision):
    rows=[]
    for p in sorted((out / 'eval' / 'feedback' / revision).rglob('*.json')):
        r=json.loads(p.read_text())
        rows.append(dict(chart=r['chart'], condition=r['job']['condition'], completion=r['completion_rate'],
            failures=[{k:n.get(k) for k in ('id','kind','lane','time','duration','failure_reason','timing_error','maintenance','max_gap_seconds','tail_ticks','tail_contact_ticks')}
                      for n in r['notes'] if not n['completed']], policy_stats=r.get('policy_stats_by_hand')))
    assert len(rows)==12
    write_json(out / 'cap' / revision / 'feedback.json', rows)


def freeze(out):
    protocol=prepare(out)
    assert not (out/'freeze.json').exists()
    assert not list((out/'eval/test').rglob('*.json'))
    artifacts={}
    for p in sorted((out/'checkpoints').glob('bc-*.pt')):
        training = json.loads((out / (p.stem + '-training.json')).read_text())
        assert len(training['epochs']) == protocol['bc_epochs']
        assert training['checkpoint_sha256'] == sha(p)
        artifacts[p.stem]=dict(path=str(p),sha256=sha(p))
    assert len(artifacts)==6
    for i in range(3):
        p=out/f'cap/v{i}/policy.py'
        artifacts[f'cap-v{i}']=dict(path=str(p),sha256=sha(p))
        assert len(list((out/f'eval/feedback/v{i}').rglob('*.json')))==12
    for e in protocol['entries']:
        assert sha(out/e['path'])==e['sha256']
    write_json(out/'freeze.json',dict(frozen_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=source_hash(),protocol_sha256=sha(out/'protocol.json'),artifacts=artifacts,
        selection=protocol['selection'],expected_test_runs=10*11*3))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['prepare','warmup','collect','fit','eval','feedback','freeze'])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, choices=[1,2,3,4], default=2)
    ap.add_argument('--head', choices=['mlp','diffusion'], default='mlp')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--split', choices=['feedback','validation','test'], default='validation')
    ap.add_argument('--policy', nargs='+', default=['classical'])
    ap.add_argument('--label', nargs='+', default=['classical'])
    ap.add_argument('--condition', nargs='+', choices=list(CONDITIONS), default=list(CONDITIONS))
    ap.add_argument('--revision', default='v0')
    a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if len(a.policy)!=len(a.label): ap.error('policy and label lengths differ')
    if a.action=='prepare': prepare(out)
    elif a.action in ('warmup','collect'): collect(out,a.workers,a.action=='warmup')
    elif a.action=='fit': fit(out,a.head,a.seed)
    elif a.action=='eval': evaluate(out,a.split,a.policy,a.label,a.condition,a.workers)
    elif a.action=='feedback': feedback(out,a.revision)
    else: freeze(out)

if __name__=='__main__': main()
