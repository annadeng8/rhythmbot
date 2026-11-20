"""Profile one headless camera rollout, separating startup and runtime.

This measures simulation throughput; use rhythmbot.realtime for live Meshcat
and wall-clock pacing. --instrument attributes camera work to RGB rendering,
label rendering, and composition. --profile adds Python profiling overhead.
"""
import argparse
import cProfile
import dataclasses
import json
import math
import pstats
import time
from pathlib import Path

from pydrake.all import Simulator
from rhythmbot.chart import Chart
from rhythmbot.profiles import config
from rhythmbot.play import build
from rhythmbot.experiments import source_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--chart', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=8)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--preset', choices=('sekai-camera', 'sekai-realtime'), default='sekai-camera')
    p.add_argument('--instrument', action='store_true')
    p.add_argument('--width', type=int)
    a = p.parse_args()
    if a.seconds <= 0:
        p.error('--seconds must be positive')
    a.out.mkdir(parents=True, exist_ok=True)
    components = {}
    if a.instrument:
        from rhythmbot.camera import CameraScreen

        def calc(self, context, output):
            start = time.perf_counter()
            rgb = self.color_port.Eval(context).data
            components['render_rgb'] = components.get('render_rgb', 0) + time.perf_counter() - start
            start = time.perf_counter()
            label = self.label_port.Eval(context).data[..., 0]
            components['render_label'] = components.get('render_label', 0) + time.perf_counter() - start
            start = time.perf_counter()
            out = self.compose(rgb, label, context.get_time())
            components['compose'] = components.get('compose', 0) + time.perf_counter() - start
            output.set_value(out)
        CameraScreen._calc = calc

    start = time.perf_counter()
    cfg = config({'preset': a.preset})
    if a.width:
        cfg = dataclasses.replace(cfg, camera=dataclasses.replace(cfg.camera,
            width=a.width, height=round(a.width * .75)))
    chart = Chart.load(a.chart)
    diagram, robot, brain, _ = build(chart, cfg, 'concert', None, None, view='camera')
    sim = Simulator(diagram)
    robot.initialize(sim.get_mutable_context())
    sim.Initialize()
    startup = time.perf_counter() - start
    profiler, ticks = cProfile.Profile(), []
    start = time.perf_counter()
    if a.profile:
        profiler.enable()
    steps = int(math.ceil(a.seconds * cfg.perception.tick_hz))
    for i in range(steps):
        tick = time.perf_counter()
        sim.AdvanceTo(min((i + 1) / cfg.perception.tick_hz, a.seconds))
        ticks.append(time.perf_counter() - tick)
    if a.profile:
        profiler.disable()
    wall = time.perf_counter() - start
    report = dict(startup_seconds=startup, sim_seconds=a.seconds, wall_seconds=wall,
        wall_per_sim=wall / a.seconds, tick_seconds=ticks, profiled=a.profile,
        components=components, vision=brain.detector.stats(), config=dataclasses.asdict(cfg),
        source_sha256=source_hash(), meshcat=False)
    (a.out / 'timing.json').write_text(json.dumps(report, indent=2) + '\n')
    if a.profile:
        profiler.dump_stats(str(a.out / 'cpu.prof'))
        with (a.out / 'profile.txt').open('w') as f:
            pstats.Stats(profiler, stream=f).sort_stats('cumtime').print_stats(55)
    print(json.dumps({k: v for k, v in report.items() if k not in ('tick_seconds', 'vision', 'config')}))


if __name__ == '__main__':
    main()
