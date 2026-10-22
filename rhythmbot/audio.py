"""A soundtrack synthesized from the chart, so a live run has music.

Nothing in the robot's pipeline hears this; it is for the person watching.
The chart *is* the song here, so the soundtrack is generated from it:

  * a metronome click on every beat when the chart has a BPM;
  * a short pluck for each tap, pitched by lane;
  * a sustained tone for a hold, for exactly its duration;
  * a gliding tone for a slide, whose pitch follows the lane along the path.

`write_soundtrack` renders a 16-bit WAV with the standard library, and
`play` starts it with the platform's command-line player (`afplay` on
macOS, else `aplay`/`ffplay`) right before the simulation starts so the
sound lines up with the Meshcat animation when the simulator runs at
real-time rate.
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

from .chart import HOLD, SLIDE, TAP, Chart

SAMPLE_RATE = 44100


def lane_frequency(lane: float, n_lanes: int) -> float:
    """A pentatonic-ish spread: lane 0 low, last lane high."""
    semitones = np.interp(lane, [0, max(n_lanes - 1, 1)], [0, 12])
    return 330.0 * 2 ** (semitones / 12.0)


def _env(n: int, attack: float, release: float, sr: int) -> np.ndarray:
    env = np.ones(n)
    a = min(int(attack * sr), n)
    r = min(int(release * sr), n)
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a)
    if r > 0:
        env[n - r :] *= np.linspace(1.0, 0.0, r)
    return env


def synthesize(chart: Chart, n_lanes: int, sr: int = SAMPLE_RATE, tail: float = 1.0) -> np.ndarray:
    """Float32 mono samples in [-1, 1] for the whole chart."""
    total = chart.end_time + tail
    out = np.zeros(int(total * sr) + 1, dtype=np.float64)

    def add(start: float, samples: np.ndarray) -> None:
        i0 = int(start * sr)
        i1 = min(i0 + len(samples), len(out))
        if i1 > i0:
            out[i0:i1] += samples[: i1 - i0]

    if chart.bpm:
        beat = 60.0 / chart.bpm
        n = int(0.03 * sr)
        tt = np.arange(n) / sr
        click = 0.5 * np.sin(2 * np.pi * 1500 * tt) * np.exp(-tt * 120)
        for b in np.arange(0.0, total, beat):
            add(b, click)

    for note in chart.notes:
        if note.kind == TAP:
            n = int(0.18 * sr)
            tt = np.arange(n) / sr
            f = lane_frequency(note.lane, n_lanes)
            tone = np.sin(2 * np.pi * f * tt) + 0.4 * np.sin(2 * np.pi * 2 * f * tt)
            add(note.time, 0.45 * tone * np.exp(-tt * 14))
        elif note.kind == HOLD:
            n = int(note.duration * sr)
            tt = np.arange(n) / sr
            f = lane_frequency(note.lane, n_lanes)
            tone = np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * 1.5 * f * tt)
            add(note.time, 0.35 * tone * _env(n, 0.01, 0.08, sr))
        elif note.kind == SLIDE:
            n = int(note.duration * sr)
            tt = np.arange(n) / sr
            offs = np.array([w[0] for w in note.waypoints])
            lanes = np.array([w[1] for w in note.waypoints])
            lane_t = np.interp(tt, offs, lanes)
            freq = lane_frequency(lane_t, n_lanes)
            phase = 2 * np.pi * np.cumsum(freq) / sr
            tone = np.sin(phase) + 0.3 * np.sin(2 * phase)
            add(note.time, 0.35 * tone * _env(n, 0.01, 0.08, sr))

    peak = np.abs(out).max()
    if peak > 0:
        out = 0.9 * out / peak
    return out.astype(np.float32)


def write_soundtrack(chart: Chart, n_lanes: int, path: str | Path, sr: int = SAMPLE_RATE) -> Path:
    samples = synthesize(chart, n_lanes, sr)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def play(path: str | Path, rate: float = 1.0) -> subprocess.Popen | None:
    """Start playing `path` in the background; None if no player is found.

    `rate` slows or speeds playback (afplay -r; ffplay via atempo), used
    when a chart is played at a different tempo than its song.
    """
    if shutil.which("afplay"):
        cmd = ["afplay"] + (["-r", f"{rate:g}"] if rate != 1.0 else []) + [str(path)]
        return subprocess.Popen(cmd)
    if shutil.which("ffplay"):
        cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
        if rate != 1.0:
            cmd += ["-af", f"atempo={rate:g}"]
        return subprocess.Popen(cmd + [str(path)])
    if shutil.which("aplay") and rate == 1.0:
        return subprocess.Popen(["aplay", "-q", str(path)])
    print("no suitable audio player found (afplay/ffplay/aplay); running silent")
    return None


def stop(proc: subprocess.Popen | None) -> None:
    if proc is not None and proc.poll() is None:
        proc.terminate()
