"""Render plain-screen training frames from a `yifan-final-v1` chart split.

    python scripts/render_study_frames.py --split train --mix data/train --out data/study-screen/train
    python scripts/render_study_frames.py --split validation --mix data/val --out data/study-screen/val

v7's own training charts (`rhythmbot.dataset`) have slides of at most three
segments; the study's charts also have zigzags of four (lanes 3-5-3-5-4).
This renders every chart of the given split with `FrameRenderer` at `--fps`,
in the `rhythmbot.dataset` layout, and with `--mix DIR` also lists DIR's
frames in `labels.jsonl` (paths relative to the output), so fine-tuning
sees the old charts too and does not forget them. Test charts are never
touched.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rhythmbot.chart import Chart
from rhythmbot.config import DEFAULT
from rhythmbot.dataset import render_chart
from rhythmbot.game import FrameRenderer

SPLITS = ROOT / 'results/yifan-final-v1/charts'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--split', required=True, choices=['train', 'validation'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--fps', type=float, default=30.0)
    p.add_argument('--mix', type=Path, help='existing dataset whose frames are listed too')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    renderer = FrameRenderer(DEFAULT)
    total = 0
    with open(a.out / 'labels.jsonl', 'w') as labels:
        for path in sorted((SPLITS / a.split).glob('*.json')):
            chart = Chart.load(path)
            chart.save(a.out / f'{chart.name}.json')
            total += render_chart(renderer, chart, a.fps, a.out, labels)
        print(f'{total} frames from {a.split}')
        if a.mix:
            rel = os.path.relpath(a.mix, a.out)
            mixed = 0
            for line in (a.mix / 'labels.jsonl').read_text().splitlines():
                rec = json.loads(line)
                rec['frame'] = f"{rel}/{rec['frame']}"
                labels.write(json.dumps(rec) + '\n')
                mixed += 1
            print(f'{mixed} frames listed from {a.mix}')


if __name__ == '__main__':
    main()
