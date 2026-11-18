"""Render a well-lit full-arm observer view alongside unchanged camera control.

HEADLESS=1 python -m rhythmbot.observer_replay --out results/observer
The observer has its own renderer and never supplies policy observations.
"""
from __future__ import annotations
import argparse
from dataclasses import replace, asdict
from pathlib import Path
import numpy as np
from PIL import Image
from pydrake.all import Simulator, Rgba
from pydrake.geometry import LightParameter, RenderEngineVtkParams
from .camera import CameraScreen, camera_config
from .chart import Chart, demo_chart
from .config import Camera
from .play import build
from .profiles import config
from .sekai_screen import SekaiScreen
from .judge import Judge


def observer_config(cfg):
    return replace(cfg, camera=Camera(width=1440, height=1080, fov_y_deg=42,
        eye_offset=(1.25, -1.65, 1.15), target_offset=(0, .15, .22), noise_std=0))


def observer_camera(cfg):
    cam = camera_config(cfg)
    cam.name = 'observer_cam'
    cam.renderer_name = 'observer_renderer'
    cam.background = Rgba(.72, .72, .75, 1)
    cam.renderer_class = RenderEngineVtkParams(default_clear_color=[.72, .72, .75], lights=[
        LightParameter(direction=[-.5, -.7, 1], intensity=3.0),
        LightParameter(direction=[.8, .1, 1], intensity=2.0),
        LightParameter(direction=[0, .8, 1], intensity=1.0),
    ])
    return cam


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path('results/observer'))
    ap.add_argument('--chart', type=Path)
    ap.add_argument('--duration', type=float, default=8)
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--snapshot', type=float, help='render one frame at this simulation time')
    args = ap.parse_args()
    if args.fps <= 0 or args.duration < 0 or (args.snapshot is not None and args.snapshot < 0):
        ap.error('fps must be positive and times nonnegative')
    chart = Chart.load(args.chart) if args.chart else demo_chart()
    cfg = config({'preset': 'sekai-camera'})
    obs_cfg = observer_config(cfg)
    diagram, robot, brain, judge = build(chart, cfg, 'concert', None, None,
        view='camera', observer_camera=observer_camera(obs_cfg))
    sim = Simulator(diagram)
    context = sim.get_mutable_context()
    robot.initialize(context)
    compositor = CameraScreen(chart, obs_cfg, int(robot.plant.GetBodyByName('screen').index()))
    compositor.renderer = SekaiScreen(obs_cfg)
    compositor.renderer.results = judge.results
    station_context = robot.station.GetMyContextFromRoot(context)
    args.out.mkdir(parents=True, exist_ok=True)
    times = [args.snapshot] if args.snapshot is not None else np.arange(0, args.duration + 1e-8, 1 / args.fps)
    for i, t in enumerate(times):
        sim.AdvanceTo(float(t))
        rgb = robot.station.GetOutputPort('observer_cam.rgb_image').Eval(station_context).data
        labels = robot.station.GetOutputPort('observer_cam.label_image').Eval(station_context).data[..., 0]
        Image.fromarray(compositor.compose(rgb, labels, float(t))).save(args.out / f'{i:06d}.png')
    judge.finalize(float(times[-1]))
    report = judge.report(extra={'view': 'camera', 'detector': 'concert',
        'observer_only': True, 'control_config': asdict(cfg), 'observer_config': asdict(obs_cfg),
        'sim_seconds': float(times[-1])})
    Judge.save_report(report, args.out / 'report.json')
    Judge.print_report(report)

if __name__ == '__main__':
    main()
