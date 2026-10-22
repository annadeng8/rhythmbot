"""Render labelled game frames for training the vision model.

    python -m rhythmbot.dataset --out data/train --charts 40 --fps 12
    python -m rhythmbot.dataset --out data/val --charts 8 --seed 1000 --fps 12

Writes `frames/<chart>_<idx>.png` and one `labels.jsonl` line per frame:

    {"frame": "frames/random-1_000017.png", "chart": "random-1", "t": 1.42,
     "notes": [{"id": 3, "kind": "tap", "lane": 2, "time_to_hit": 0.81,
                "duration": 0.0, "head_x": 106.7, "head_row": 143.2}, ...]}

`time_to_hit` is in seconds from the frame's time; the planner only needs
kind, lane, time_to_hit and (for holds/slides) duration and the path, which
is exactly what `perception.DetectedNote` carries, so a model trained on
these labels drops into `TorchDetector.decode`.

Also handy: `--preview` writes a contact sheet of a few frames so the
rendering can be eyeballed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .chart import random_chart, varied_random_chart
from .config import DEFAULT
from .game import FrameRenderer


def render_chart(renderer: FrameRenderer, chart, fps: float, out_dir: Path, labels, start=0.0):
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    n = int((chart.end_time + 0.5 - start) * fps)
    for i in range(n):
        t = start + i / fps
        frame = renderer.render(chart, t)
        name = f"frames/{chart.name}_{i:06d}.png"
        Image.fromarray(frame).save(out_dir / name)
        labels.write(
            json.dumps(
                {"frame": name, "chart": chart.name, "t": t, "notes": renderer.label(chart, t)}
            )
            + "\n"
        )
    return n


def contact_sheet(renderer: FrameRenderer, chart, times, path: Path):
    frames = [renderer.render(chart, t) for t in times]
    sheet = np.concatenate(frames, axis=1)
    Image.fromarray(sheet).save(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--charts", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--duration", type=float, default=20.0, help="seconds per chart")
    ap.add_argument("--preview", action="store_true", help="only write a contact sheet")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    renderer = FrameRenderer(DEFAULT)
    if args.preview:
        chart = random_chart(args.seed, n_lanes=DEFAULT.layout.n_lanes, duration=args.duration)
        chart.save(out / f"{chart.name}.json")
        contact_sheet(renderer, chart, np.arange(2.0, 8.0, 0.75), out / "preview.png")
        print(f"preview -> {out / 'preview.png'}")
        return
    total = 0
    with open(out / "labels.jsonl", "w") as labels:
        for k in range(args.charts):
            chart = varied_random_chart(
                args.seed + k, n_lanes=DEFAULT.layout.n_lanes, duration=args.duration
            )
            chart.save(out / f"{chart.name}.json")
            total += render_chart(renderer, chart, args.fps, out, labels)
    print(f"{total} frames from {args.charts} charts -> {out}")


if __name__ == "__main__":
    main()
