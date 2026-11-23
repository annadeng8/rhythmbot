"""Run the dual-arm camera environment with measured wall-clock pacing.

    python -m rhythmbot.realtime --chart charts/stylus-hard-handoffs.json --out results/live.json

Uses actual camera RGB, unchanged 1 ms physics, 30 Hz vision, 200 Hz judging,
and native Meshcat robot rendering. --unpaced measures available throughput.
Startup is excluded; per-tick computation and accumulated lag are reported.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import gc
import importlib.metadata
import platform
import time
from pathlib import Path

import numpy as np
from pydrake.all import Meshcat, Simulator
from .chart import Chart
from .play import build
from .policies import make_policy
from .profiles import config
from .experiments import digest, source_hash


def run(chart, out, *, meshcat=True, paced=True, duration=None, policy='classical', hold=False,
        camera_width=None, wait_for_browser=0., detector='concert', model_path=None, seed=0):
    started = time.perf_counter()
    source = source_hash()
    cfg = config({'preset': 'sekai-realtime'})
    if camera_width is not None:
        if camera_width < 32 or camera_width % 4:
            raise ValueError('camera width must be at least 32 and divisible by 4')
        cfg = dataclasses.replace(cfg, camera=dataclasses.replace(cfg.camera,
            width=camera_width, height=camera_width * 3 // 4))
    viewer = Meshcat() if meshcat else None
    if viewer is not None:
        c = cfg.layout.screen_center()
        viewer.SetCameraPose(c + np.array([.9, -1.1, .8]), c + np.array([0., .15, 0.]))
        print(f'Meshcat: {viewer.web_url()}', flush=True)
    if detector not in ('concert', 'concert-torch'):
        raise ValueError('live mode requires a camera RGB detector')
    if policy.startswith('bc:') or detector == 'concert-torch':
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(seed)
    policies = [make_policy(policy, cfg, hand=k) for k in range(cfg.arms.n)]
    diagram, robot, brain, judge = build(chart, cfg, detector, model_path, viewer, policies, view='camera')
    sim = Simulator(diagram)
    robot.initialize(sim.get_mutable_context())
    sim.Initialize()
    diagram.ForcedPublish(sim.get_context())
    # Compile camera shaders and exercise the fixed image warp before starting
    # the wall clock. This evaluates t=0 only; no gameplay time is skipped.
    screen = diagram.GetSubsystemByName('screen')
    frame = screen.get_output_port().Eval(screen.GetMyContextFromRoot(sim.get_context()))
    brain.detector.rectifier.rectify(frame)
    if hasattr(brain.detector, 'warmup'):
        brain.detector.warmup(frame)
    sim.AdvanceTo(0.)
    if viewer is not None:
        deadline = time.perf_counter() + wait_for_browser
        while viewer.GetNumActiveConnections() == 0 and time.perf_counter() < deadline:
            time.sleep(.1)
    browser_start = viewer.GetNumActiveConnections() if viewer is not None else 0
    end = chart.end_time + 1 if duration is None else duration
    if end <= 0:
        raise ValueError('duration must be positive')
    dt = 1 / cfg.perception.tick_hz
    ticks, lag = [], []
    plant_context = robot.plant.GetMyContextFromRoot(sim.get_context())
    q_min = [robot.plant.GetPositions(plant_context, arm.iiwa).copy() for arm in robot.arms]
    q_max = [q.copy() for q in q_min]
    # Reference counting still frees frame buffers immediately. Defer cyclic
    # collection during the timed loop to avoid collector-induced frame stalls.
    gc.collect()
    startup = time.perf_counter() - started
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        start = time.perf_counter()
        for i in range(int(np.ceil(end / dt))):
            target = min((i + 1) * dt, end)
            before = time.perf_counter()
            sim.AdvanceTo(target)
            for k, arm in enumerate(robot.arms):
                q = robot.plant.GetPositions(plant_context, arm.iiwa)
                np.minimum(q_min[k], q, out=q_min[k])
                np.maximum(q_max[k], q, out=q_max[k])
            finished = time.perf_counter()
            ticks.append(finished - before)
            lag.append(max(0., finished - start - target))
            if paced:
                time.sleep(max(0., start + target - finished))
        wall = time.perf_counter() - start
    finally:
        if gc_enabled:
            gc.enable()
        if screen.noise_prefetch is not None:
            screen.noise_prefetch.close()
        if viewer is not None:
            diagram.GetSubsystemByName('highway').close()
    judge.finalize(end)
    report = judge.report(extra=dict(
        policy=policy, preset='sekai-realtime', view='camera', detector=detector,
        seed=seed, config=dataclasses.asdict(cfg),
        vision=brain.detector.stats(), startup_seconds=startup, sim_seconds=end,
        wall_seconds=wall, compute_seconds=sum(ticks), wall_per_sim=wall / end,
        compute_per_sim=sum(ticks) / end, paced=paced, meshcat=meshcat,
        browser_connections_start=browser_start,
        browser_connections_end=viewer.GetNumActiveConnections() if viewer is not None else 0,
        machine=platform.machine(), platform=platform.platform(),
        source_sha256=source, chart_sha256=digest(chart.to_dict()),
        versions={name: importlib.metadata.version(name) for name in ('drake', 'numpy', 'scipy', 'pillow')},
        tick_seconds=ticks, tick_p50_seconds=float(np.quantile(ticks, .5)),
        tick_p95_seconds=float(np.quantile(ticks, .95)), tick_max_seconds=max(ticks),
        ticks_over_budget=sum(t > dt for t in ticks), tick_budget_seconds=dt,
        max_lag_seconds=max(lag), end_lag_seconds=lag[-1],
        policy_stats=[p.stats() if hasattr(p, 'stats') else {} for p in policies],
        executed_arms=[dict(model=arm.name, joint_range_rad=(q_max[k] - q_min[k]).tolist())
                       for k, arm in enumerate(robot.arms)],
        # Soft real time: bounded lag within one 30 Hz frame, not a hard
        # deadline guarantee or a measurement of browser display latency.
        realtime_pass=bool(paced and max(lag) <= dt),
    ))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('sim_seconds', 'wall_seconds', 'compute_per_sim',
        'tick_p95_seconds', 'ticks_over_budget', 'max_lag_seconds', 'realtime_pass', 'score', 'max_score')}, indent=2), flush=True)
    if hold and viewer is not None:
        print('Meshcat stays open; Ctrl-C to exit.', flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--chart', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--duration', type=float)
    p.add_argument('--policy', default='classical')
    p.add_argument('--detector', choices=('concert', 'concert-torch'), default='concert')
    p.add_argument('--model', help='Anna neural detector checkpoint for concert-torch')
    p.add_argument('--seed', type=int, default=0, help='inference RNG seed, independent of training seed')
    p.add_argument('--camera-width', type=int, help='control camera width; default 640, use 960 for the study resolution')
    p.add_argument('--wait-for-browser', type=float, default=0., metavar='SECONDS')
    p.add_argument('--no-meshcat', action='store_true')
    p.add_argument('--unpaced', action='store_true')
    p.add_argument('--hold', action='store_true')
    a = p.parse_args()
    run(Chart.load(a.chart), a.out, meshcat=not a.no_meshcat, paced=not a.unpaced,
        duration=a.duration, policy=a.policy, hold=a.hold, camera_width=a.camera_width,
        wait_for_browser=a.wait_for_browser, detector=a.detector, model_path=a.model, seed=a.seed)


if __name__ == '__main__':
    main()
