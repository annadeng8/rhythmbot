"""Export the native Drake/Meshcat observer without changing camera control.

HEADLESS=1 python -m rhythmbot.meshcat_snapshot --time 4 --out results/meshcat
Open scene.html in a browser; the static export needs no running simulator.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
from pydrake.all import Meshcat, Simulator
from .chart import Chart, demo_chart
from .play import build
from .profiles import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chart', type=Path)
    parser.add_argument('--time', type=float, default=4)
    parser.add_argument('--out', type=Path, default=Path('results/meshcat'))
    args = parser.parse_args()
    if args.time < 0:
        parser.error('--time must be nonnegative')
    chart = Chart.load(args.chart) if args.chart else demo_chart()
    cfg = config({'preset': 'sekai-camera'})
    meshcat = Meshcat()
    meshcat.SetCamera(Meshcat.PerspectiveCamera(fov=50))
    center = cfg.layout.screen_center()
    meshcat.SetCameraPose(center + np.array([1.05, -1.4, 1.05]),
                          center + np.array([0, .18, .22]))
    diagram, robot, brain, judge = build(chart, cfg, 'concert', None, meshcat, view='camera')
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    robot.initialize(context)
    simulator.AdvanceTo(args.time)
    diagram.ForcedPublish(context)
    meshcat.Flush()
    args.out.mkdir(parents=True, exist_ok=True)
    html = meshcat.StaticHtml().replace('</head>',
        '<style>.dg { display: none !important; }</style></head>')
    (args.out / 'scene.html').write_text(html)
    # This is a presentation snapshot, not a completed-chart evaluation.
    (args.out / 'snapshot.json').write_text(json.dumps({
        'time': args.time, 'chart': chart.name, 'arms': cfg.arms.n,
        'controller': 'classical', 'control_input': 'sekai-camera RGB / concert detector',
        'observer': 'native Meshcat / default materials and lighting',
        'display': 'Meshcat note geometry; control camera retains concert artwork',
        'completed_notes_so_far': sum(r.completed is True for r in judge.results.values()),
        'benchmark': False,
    }, indent=2) + '\n')

if __name__ == '__main__':
    main()
