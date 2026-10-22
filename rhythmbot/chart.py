"""Rhythm charts: the notes a song asks the player to play.

A `Note` is one of

    tap    -- touch lane `lane` at `time`;
    hold   -- touch lane `lane` at `time` and stay down for `duration`;
    slide  -- touch at `time` and, while staying down, follow `waypoints`,
              a list of (time_offset, lane) pairs ending at `duration`.
              Lanes may be fractional for slides.

Two taps with the same time form a chord; the planner plays those with the
two fingers of the WSG when the lanes are close enough (GuitarHeroBot's
trick), and gives up on the second one otherwise.

Charts load from and save to JSON, and `random_chart` makes new ones for
generalization tests.  GuitarHeroBot wrote its songs as (button, start,
duration) tuples; this is the same idea with lanes and slides.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

TAP, HOLD, SLIDE = "tap", "hold", "slide"
KINDS = (TAP, HOLD, SLIDE)


@dataclass
class Note:
    kind: str
    lane: float
    time: float
    duration: float = 0.0
    # Slide path: (offset from `time`, lane).  Always starts at (0, lane) and
    # ends at (duration, last lane).  Empty for taps and holds.
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    id: int = -1

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown note kind {self.kind!r}")
        if self.kind == TAP:
            self.duration = 0.0
            self.waypoints = []
        elif self.kind == HOLD:
            self.waypoints = []
        elif self.kind == SLIDE:
            if not self.waypoints:
                self.waypoints = [(0.0, self.lane), (self.duration, self.lane)]
            if self.waypoints[0][0] != 0.0:
                self.waypoints.insert(0, (0.0, self.lane))
            self.duration = float(self.waypoints[-1][0])
            self.lane = float(self.waypoints[0][1])

    @property
    def end_time(self) -> float:
        return self.time + self.duration

    def lane_at(self, t: float) -> float:
        """Target lane at absolute time t (holds and taps are constant)."""
        if self.kind != SLIDE:
            return self.lane
        offs = np.array([w[0] for w in self.waypoints])
        lanes = np.array([w[1] for w in self.waypoints])
        return float(np.interp(t - self.time, offs, lanes))

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "lane": self.lane, "time": self.time}
        if self.kind != TAP:
            d["duration"] = self.duration
        if self.kind == SLIDE:
            d["waypoints"] = [list(w) for w in self.waypoints]
        return d

    @staticmethod
    def from_dict(d: dict, id: int = -1) -> "Note":
        return Note(
            kind=d["kind"],
            lane=float(d["lane"]),
            time=float(d["time"]),
            duration=float(d.get("duration", 0.0)),
            waypoints=[tuple(w) for w in d.get("waypoints", [])],
            id=id,
        )


@dataclass
class Chart:
    notes: list[Note]
    name: str = "untitled"
    bpm: float | None = None
    # Optional soundtrack: an audio file (path relative to the chart file,
    # or absolute) and the chart time at which the audio starts playing.
    audio: str | None = None
    audio_offset: float = 0.0
    source: str | None = None  # where the chart was imported from

    def __post_init__(self):
        self.notes.sort(key=lambda n: (n.time, n.lane))
        for i, n in enumerate(self.notes):
            n.id = i

    def __len__(self) -> int:
        return len(self.notes)

    @property
    def end_time(self) -> float:
        return max((n.end_time for n in self.notes), default=0.0)

    def visible(self, t: float, approach_time: float) -> list[Note]:
        """Notes any part of which is on the highway at time t: those whose
        hit time is within `approach_time` and that have not finished."""
        return [
            n for n in self.notes if n.time - approach_time <= t and n.end_time >= t
        ]

    def counts(self) -> dict[str, int]:
        return {k: sum(1 for n in self.notes if n.kind == k) for k in KINDS}

    def scaled(self, tempo: float) -> "Chart":
        """The same chart played at `tempo` times the speed (0.5 = half)."""
        notes = []
        for n in self.notes:
            notes.append(
                Note(
                    n.kind,
                    n.lane,
                    n.time / tempo,
                    n.duration / tempo,
                    [(off / tempo, lane) for off, lane in n.waypoints],
                )
            )
        return Chart(
            notes,
            name=self.name if tempo == 1.0 else f"{self.name}-x{tempo:g}",
            bpm=self.bpm * tempo if self.bpm else None,
            audio=self.audio,
            audio_offset=self.audio_offset / tempo,
            source=self.source,
        )

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "bpm": self.bpm,
            "notes": [n.to_dict() for n in self.notes],
        }
        if self.audio:
            d["audio"] = self.audio
            d["audio_offset"] = self.audio_offset
        if self.source:
            d["source"] = self.source
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @staticmethod
    def load(path: str | Path) -> "Chart":
        d = json.loads(Path(path).read_text())
        return Chart(
            notes=[Note.from_dict(n) for n in d["notes"]],
            name=d.get("name", Path(path).stem),
            bpm=d.get("bpm"),
            audio=d.get("audio"),
            audio_offset=float(d.get("audio_offset", 0.0)),
            source=d.get("source"),
        )


def random_chart(
    seed: int,
    n_lanes: int = 6,
    duration: float = 20.0,
    lead_in: float = 2.5,
    density: float = 1.6,
    p_hold: float = 0.2,
    p_slide: float = 0.2,
    p_chord: float = 0.12,
    min_gap: float = 0.28,
    name: str | None = None,
    hold_range: tuple[float, float] = (0.4, 1.4),
    slide_range: tuple[float, float] = (0.5, 1.4),
    p_long_hold: float = 0.0,
    long_hold_range: tuple[float, float] = (2.0, 7.0),
    p_repeat_lane: float = 0.0,
) -> Chart:
    """A playable random chart: notes at least `min_gap` apart (one finger has
    to travel between them), holds and slides with durations drawn from the
    given ranges (plus, with `p_long_hold`, a long sustained hold), slides
    crossing one to three lanes, occasional two-note chords within two
    lanes, and with `p_repeat_lane` a tendency to stay in the same lane as
    the previous note (real charts repeat lanes a lot)."""
    rng = np.random.default_rng(seed)
    prev_lane = int(rng.integers(0, n_lanes))
    notes: list[Note] = []
    t = lead_in
    while t < duration:
        kind = rng.choice([TAP, HOLD, SLIDE], p=[1 - p_hold - p_slide, p_hold, p_slide])
        lane = prev_lane if rng.random() < p_repeat_lane else int(rng.integers(0, n_lanes))
        prev_lane = lane
        if kind == TAP:
            notes.append(Note(TAP, lane, round(t, 3)))
            if rng.random() < p_chord:
                gap = int(rng.choice([1, 2]))
                other = lane + gap if lane + gap < n_lanes else lane - gap
                notes.append(Note(TAP, other, round(t, 3)))
            dur = 0.0
        elif kind == HOLD:
            if rng.random() < p_long_hold:
                dur = float(rng.uniform(*long_hold_range))
            else:
                dur = float(rng.uniform(*hold_range))
            notes.append(Note(HOLD, lane, round(t, 3), round(dur, 3)))
        else:
            dur = float(rng.uniform(*slide_range))
            n_seg = int(rng.integers(1, 4))
            cur = lane
            wps = [(0.0, float(cur))]
            for k in range(1, n_seg + 1):
                step = int(rng.choice([-2, -1, 1, 2]))
                cur = int(np.clip(cur + step, 0, n_lanes - 1))
                wps.append((round(dur * k / n_seg, 3), float(cur)))
            notes.append(Note(SLIDE, lane, round(t, 3), waypoints=wps))
        t += dur + max(min_gap, float(rng.exponential(1.0 / density)))
    return Chart(notes, name=name or f"random-{seed}")


def varied_random_chart(seed: int, n_lanes: int = 6, duration: float = 20.0) -> Chart:
    """A random chart whose style parameters are themselves random, spanning
    what real charts look like: sparse to dense (0.2 s gaps), holds from
    0.15 s to several seconds, lanes that repeat.  Training data for the
    vision model comes from these."""
    rng = np.random.default_rng(10_000 + seed)
    return random_chart(
        seed,
        n_lanes=n_lanes,
        duration=duration,
        density=float(rng.uniform(1.0, 3.5)),
        p_hold=float(rng.uniform(0.1, 0.4)),
        p_slide=float(rng.uniform(0.05, 0.3)),
        p_chord=float(rng.uniform(0.0, 0.25)),
        min_gap=float(rng.uniform(0.18, 0.35)),
        hold_range=(0.15, float(rng.uniform(0.6, 2.0))),
        slide_range=(0.3, float(rng.uniform(0.8, 1.8))),
        p_long_hold=float(rng.uniform(0.0, 0.2)),
        p_repeat_lane=float(rng.uniform(0.0, 0.6)),
        name=f"varied-{seed}",
    )


def demo_chart() -> Chart:
    """A short hand-written chart that exercises every note type."""
    notes = [
        Note(TAP, 2, 3.0),
        Note(TAP, 3, 3.6),
        Note(TAP, 1, 4.2),
        Note(TAP, 4, 4.8),
        Note(HOLD, 2, 5.6, 0.9),
        Note(TAP, 5, 7.0),
        Note(TAP, 0, 7.6),
        Note(SLIDE, 1, 8.4, waypoints=[(0.0, 1), (0.5, 3), (1.0, 4)]),
        Note(TAP, 2, 10.0),
        Note(TAP, 3, 10.0),  # chord, adjacent lanes
        Note(TAP, 1, 10.8),
        Note(TAP, 3, 10.8),  # chord, two lanes apart
        Note(HOLD, 4, 11.6, 0.6),
        Note(SLIDE, 5, 12.8, waypoints=[(0.0, 5), (0.6, 3), (1.2, 1), (1.6, 2)]),
        Note(TAP, 0, 15.0),
        Note(TAP, 5, 15.5),
        Note(TAP, 0, 16.0),
        Note(TAP, 5, 16.5),
    ]
    return Chart(notes, name="demo", bpm=100)


def write_builtin_charts(directory: str | Path) -> list[Path]:
    """Write the demo chart and a few random ones to `directory`."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for chart in [demo_chart()] + [random_chart(s) for s in (1, 2, 3)]:
        p = directory / f"{chart.name}.json"
        chart.save(p)
        paths.append(p)
    return paths
