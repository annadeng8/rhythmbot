"""Play a chart: build the whole diagram, simulate, score.

    cd project
    python -m rhythmbot.play --demo                       # watch in Meshcat
    python -m rhythmbot.play --demo --audio --hold        # with sound, server kept up
    python -m rhythmbot.play --chart charts/song.json --song song.mp3 --tempo 0.6
    HEADLESS=1 python -m rhythmbot.play --demo            # no browser, fast
    python -m rhythmbot.play --chart charts/random-1.json --detector color
    python -m rhythmbot.play --random 7 --noise 0.03 --latency 0.1

Diagram:

    GameScreen.frame ----> RhythmBrain.frame
    RhythmBrain.X_WG_desired ----> DiffIK.X_AE_desired
    DiffIK.joint_positions ----> station.iiwa.position
    RhythmBrain.wsg_position ----> station.wsg.position
    station.body_poses ----> TouchSensor.body_poses
    TouchSensor.touch ----> Judge.touch
    HighwayVisualizer (draws the chart in Meshcat; no ports)

The judge's report is printed and written to results/<chart>-<detector>.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
from pydrake.all import DiagramBuilder, RigidTransform, Simulator

from . import PROJECT_ROOT
from . import audio
from .brain import GameScreen, RhythmBrain, initial_tip_pose
from .chart import Chart, demo_chart, random_chart
from .config import DEFAULT, Config, Perception
from .game import HighwayVisualizer
from .judge import Judge
from .perception import make_detector
from .planner import ClassicalPlanner
from .robot import Robot, TouchSensor
from .viz import HEADLESS, get_meshcat, keep_meshcat_open, publish_recording, start_recording


def build(chart: Chart, cfg: Config, detector_name: str, model_path: str | None, meshcat):
    builder = DiagramBuilder()
    robot = Robot(builder, cfg, meshcat)

    # Start hovering over the middle of the judgment line.
    X_WT0 = initial_tip_pose(cfg)
    X_GT = np.array([0.0, cfg.gripper.tip_offset, 0.0])
    X_WG0 = X_WT0 @ RigidTransform(-X_GT)
    q0 = robot.solve_start_configuration(X_WG0)
    X_WG0_actual = robot.gripper_pose_at(q0)

    detector = make_detector(detector_name, chart, cfg, model_path)
    policy = ClassicalPlanner(cfg)
    screen = builder.AddSystem(GameScreen(chart, cfg))
    brain = builder.AddSystem(RhythmBrain(detector, policy, cfg, X_WG0_actual))
    brain.set_name("brain")
    touch = builder.AddSystem(TouchSensor(robot.plant, cfg, meshcat))
    touch.set_name("touch_sensor")
    judge = builder.AddSystem(Judge(chart, cfg))
    judge.set_name("judge")
    builder.AddSystem(HighwayVisualizer(chart, cfg, meshcat)).set_name("highway")

    builder.Connect(screen.get_output_port(), brain.GetInputPort("frame"))
    builder.Connect(brain.GetOutputPort("X_WG_desired"), robot.diffik.GetInputPort("X_AE_desired"))
    builder.Connect(brain.GetOutputPort("wsg_position"), robot.station.GetInputPort("wsg.position"))
    builder.Connect(robot.station.GetOutputPort("body_poses"), touch.GetInputPort("body_poses"))
    builder.Connect(touch.get_output_port(), judge.GetInputPort("touch"))

    diagram = builder.Build()
    diagram.set_name("rhythmbot")
    return diagram, robot, brain, judge


def run(
    chart: Chart,
    cfg: Config = DEFAULT,
    detector: str = "oracle",
    model_path: str | None = None,
    duration: float | None = None,
    realtime: bool = True,
    out_path: str | Path | None = None,
    quiet: bool = False,
    with_audio: bool = False,
    song: str | Path | None = None,
    song_offset: float = 0.0,
    song_rate: float = 1.0,
) -> dict:
    meshcat = get_meshcat()
    if meshcat is not None:
        # Look at the screen from the robot's front-left, slightly above.
        c = cfg.layout.screen_center()
        meshcat.SetCameraPose(c + np.array([0.9, -1.1, 0.8]), c + np.array([0.0, 0.15, 0.0]))
    diagram, robot, brain, judge = build(chart, cfg, detector, model_path, meshcat)
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    robot.initialize(context)
    if realtime and not HEADLESS:
        simulator.set_target_realtime_rate(1.0)
    t_end = duration if duration is not None else chart.end_time + 1.0

    diagram.ForcedPublish(context)
    start_recording()
    player = None
    wall = time.time()
    try:
        if realtime and not HEADLESS:
            if song is not None:
                # The song starts `song_offset` seconds into the chart: run
                # the simulation up to there silently, then start playback.
                if song_offset > 0:
                    simulator.AdvanceTo(min(song_offset, t_end))
                player = audio.play(song, rate=song_rate)
            elif with_audio:
                wav = audio.write_soundtrack(
                    chart, cfg.layout.n_lanes, PROJECT_ROOT / "results" / f"{chart.name}.wav"
                )
                player = audio.play(wav)
        simulator.AdvanceTo(t_end)
    finally:
        audio.stop(player)
    wall = time.time() - wall
    publish_recording()

    judge.finalize(t_end)
    rep = judge.report(
        extra={
            "detector": detector,
            "skipped_by_planner": sorted(brain.last_state.skipped_note_ids),
            "n_detections": brain.last_state.n_detections,
            "sim_seconds": t_end,
            "wall_seconds": wall,
            "config": dataclasses.asdict(cfg),
        }
    )
    if not quiet:
        Judge.print_report(rep)
        print(f"(simulated {t_end:.1f} s in {wall:.1f} s wall)")
    if out_path is None:
        out_path = PROJECT_ROOT / "results" / f"{chart.name}-{detector}.json"
    Judge.save_report(rep, out_path)
    if not quiet:
        print(f"report -> {out_path}")
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--chart", help="chart JSON file")
    src.add_argument("--random", type=int, metavar="SEED", help="generate a random chart")
    src.add_argument("--demo", action="store_true", help="the built-in demo chart (default)")
    ap.add_argument("--detector", default="oracle", choices=["oracle", "color", "torch"])
    ap.add_argument("--model", help="TorchScript model for --detector torch")
    ap.add_argument("--noise", type=float, default=0.0, help="oracle hit-time noise std (s)")
    ap.add_argument("--latency", type=float, default=0.0, help="oracle detection latency (s)")
    ap.add_argument("--duration", type=float, help="simulate this long instead of the whole chart")
    ap.add_argument("--no-realtime", action="store_true", help="run as fast as possible")
    ap.add_argument("--out", help="where to write the JSON report")
    ap.add_argument("--audio", action="store_true", help="play a soundtrack synthesized from the chart")
    ap.add_argument("--song", help="audio file to play in sync (default: the chart's own, if it names one)")
    ap.add_argument("--song-offset", type=float, help="chart time at which the song starts (default: chart's)")
    ap.add_argument("--tempo", type=float, default=1.0, help="play the chart (and song) at this fraction of speed")
    ap.add_argument(
        "--hold", action="store_true",
        help="keep the Meshcat server up after the run (until Ctrl-C) so the recording can be scrubbed",
    )
    args = ap.parse_args(argv)

    if args.chart:
        chart = Chart.load(args.chart)
    elif args.random is not None:
        chart = random_chart(args.random, n_lanes=DEFAULT.layout.n_lanes)
    else:
        chart = demo_chart()

    if args.tempo != 1.0:
        chart = chart.scaled(args.tempo)

    song = args.song or chart.audio
    if song is not None and not Path(song).is_absolute() and args.chart:
        song = Path(args.chart).parent / song
    if song is not None and not Path(song).exists():
        print(f"song file not found: {song}; playing without it")
        song = None
    song_offset = args.song_offset if args.song_offset is not None else chart.audio_offset

    cfg = dataclasses.replace(
        DEFAULT,
        perception=dataclasses.replace(
            DEFAULT.perception, timing_noise_std=args.noise, latency=args.latency
        ),
    )
    print(f"chart {chart.name}: {chart.counts()}  ends at {chart.end_time:.1f} s")
    run(
        chart,
        cfg,
        detector=args.detector,
        model_path=args.model,
        duration=args.duration,
        realtime=not args.no_realtime,
        out_path=args.out,
        with_audio=args.audio,
        song=song,
        song_offset=song_offset,
        song_rate=args.tempo,
    )
    if args.hold and not HEADLESS:
        print("Holding Meshcat open; Ctrl-C to quit.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
    elif sys.stdin.isatty():
        keep_meshcat_open()


if __name__ == "__main__":
    main()
