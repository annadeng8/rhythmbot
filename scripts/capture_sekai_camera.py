"""Record rectified dual-arm camera frames with labels, for fine-tuning v7.

    python scripts/capture_sekai_camera.py --split train --out data/sekai-camera/train --workers 6
    python scripts/capture_sekai_camera.py --split validation --out data/sekai-camera/val --workers 6

Each chart of the given `yifan-final-v1` split is played by the classical
planner with the color/shape camera detector, at the study camera's
960x720 resolution. Every `--every`-th 30 Hz vision tick, the detector's
rectified network input (the exact pixels `concert-torch` feeds v7, robot
occlusion included) is saved with `FrameRenderer.label` ground truth, in the
`rhythmbot.dataset` layout that `rhythmbot.train` reads. `--dense N` adds N
`chart.dense_random_chart` charts (seeds 90000+, unlike any test chart) for
the song excerpts' density, which the synthetic split lacks. Test charts are
never touched.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPLITS = ROOT / 'results/yifan-final-v1/charts'


def capture(chart_path: str, out: str, every: int) -> tuple[str, int, float]:
    import numpy as np
    from PIL import Image
    from pydrake.all import Simulator
    from rhythmbot.chart import Chart
    from rhythmbot.game import FrameRenderer
    from rhythmbot.play import build
    from rhythmbot.policies import make_policy
    from rhythmbot.profiles import config

    chart = Chart.load(chart_path)
    stem = Path(chart_path).stem
    cfg = config({'preset': 'sekai-realtime'})
    cfg = dataclasses.replace(cfg, camera=dataclasses.replace(cfg.camera, width=960, height=720))
    policies = [make_policy('classical', cfg, hand=k) for k in range(cfg.arms.n)]
    diagram, robot, brain, judge = build(chart, cfg, 'concert', None, None, policies, view='camera')
    labeller = FrameRenderer(cfg)
    detector = brain.detector
    out_dir = Path(out)
    (out_dir / 'frames').mkdir(parents=True, exist_ok=True)
    rows, ticks = [], [0]
    detect = detector.detect

    def recording_detect(frame, t):
        found = detect(frame, t)
        if ticks[0] % every == 0:
            name = f'frames/{stem}_{ticks[0]:06d}.png'
            Image.fromarray(detector.last_rectified).save(out_dir / name)
            rows.append(dict(frame=name, chart=stem, t=float(t), notes=labeller.label(chart, t)))
        ticks[0] += 1
        return found

    detector.detect = recording_detect
    sim = Simulator(diagram)
    robot.initialize(sim.get_mutable_context())
    sim.Initialize()
    end = chart.end_time + 0.5
    sim.AdvanceTo(end)
    judge.finalize(end)
    report = judge.report()
    chart.save(out_dir / f'{stem}.json')
    (out_dir / f'{stem}.labels.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return stem, len(rows), report['score'] / max(report['max_score'], 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--split', choices=('train', 'validation'), required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--every', type=int, default=3, help='keep every Nth 30 Hz tick (3 -> 10 fps)')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--dense', type=int, default=0, help='extra dense random charts')
    a = p.parse_args()
    from rhythmbot.chart import dense_random_chart
    charts = sorted((SPLITS / a.split).glob('*.json'))
    a.out.mkdir(parents=True, exist_ok=True)
    for k in range(a.dense):
        path = a.out / 'generated' / f'dense-{90000 + k}.json'
        path.parent.mkdir(exist_ok=True)
        dense_random_chart(90000 + k, n_lanes=6, duration=20.0).save(path)
        charts.append(path)
    todo = [c for c in charts if not (a.out / f'{c.stem}.labels.jsonl').exists()]
    with ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context('spawn'),
                             max_tasks_per_child=2) as pool:
        futures = [pool.submit(capture, str(c), str(a.out), a.every) for c in todo]
        for n, f in enumerate(as_completed(futures), 1):
            name, frames, score = f.result()
            print(f'{n}/{len(todo)} {name}: {frames} frames, score {score:.1%}', flush=True)
    # One labels.jsonl in chart order, as rhythmbot.train expects.
    with open(a.out / 'labels.jsonl', 'w') as labels:
        for c in charts:
            labels.write((a.out / f'{c.stem}.labels.jsonl').read_text())
    print(f'{sum(1 for _ in open(a.out / "labels.jsonl"))} frames -> {a.out}')


if __name__ == '__main__':
    main()
