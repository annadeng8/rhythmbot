"""Import charts from real rhythm games.

    python -m rhythmbot.importers song.osu   --out charts/song.json
    python -m rhythmbot.importers song.sm    --out charts/song.json --difficulty Hard
    python -m rhythmbot.importers notes.chart --out charts/song.json --difficulty ExpertSingle
    python -m rhythmbot.importers --guitarherobot --out charts/      # Slow Ride, 3 difficulties

Supported formats (all plain text, all documented by their communities):

    .osu           osu!mania beatmaps.  Column from x, hold from type bit 7.
    .sm / .ssc     StepMania / ITG.  Beats -> seconds through #BPMS and
                   #STOPS, holds (2...3) and rolls (4...3) become holds.
    .chart         Clone Hero / Moonscraper (the format GuitarHeroBot's
                   songs came from).  Ticks -> seconds through [SyncTrack],
                   five frets, sustains become holds, open notes dropped.
    GuitarHeroBot  the (button, start, duration) tuples hand-transcribed
                   in the GuitarHeroBot notebook for Foghat's "Slow Ride":
                   buttons 1-5 -> lanes, "13" chords -> two taps.

Lane mapping: a game with `k` columns is spread over our `n_lanes` by
`lane = round(col * (n_lanes - 1) / (k - 1))` when k > n_lanes, else the
columns are centred.  None of these games has slides, so imported charts
hold only taps and holds; the audio file named in the chart header is
recorded in `Chart.audio` (relative to the chart file) for `play --song`.

Real charts are often denser than one fingertip can play (notes less than
~0.25 s apart in different lanes): `--tempo 0.5` slows the chart (and
`play` slows the audio to match with `afplay -r`), and `--min-gap` thins
notes that follow another too closely.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from .chart import HOLD, TAP, Chart, Note
from .config import DEFAULT


# ---------------------------------------------------------------------------
# Lane mapping and cleanup
# ---------------------------------------------------------------------------
def map_columns(k: int, n_lanes: int) -> list[int]:
    """Lane for each of k game columns."""
    if k <= n_lanes:
        start = (n_lanes - k) // 2
        return [start + c for c in range(k)]
    return [int(round(c * (n_lanes - 1) / (k - 1))) for c in range(k)]


def thin(notes: list[Note], min_gap: float) -> list[Note]:
    """Drop notes that start within `min_gap` of the previous kept note's
    end (chords at the same instant are kept)."""
    if min_gap <= 0:
        return notes
    kept: list[Note] = []
    last_end = -np.inf
    last_time = -np.inf
    for n in sorted(notes, key=lambda n: (n.time, n.lane)):
        if abs(n.time - last_time) < 1e-3 or n.time - last_end >= min_gap:
            kept.append(n)
            last_end = max(last_end, n.end_time)
            last_time = n.time
    return kept


def with_lead_in(chart: Chart, lead_in: float) -> Chart:
    """Shift the chart so its first note is at least `lead_in` seconds in.

    Game charts often start a note at 0 s, but here a note has to be on the
    highway for the approach time before its hit and the arm has to get
    there.  The song is started the same amount later (`audio_offset`), so
    music and notes stay in sync.
    """
    if not chart.notes:
        return chart
    shift = max(0.0, lead_in - min(n.time for n in chart.notes))
    if shift == 0.0:
        return chart
    notes = [Note(n.kind, n.lane, n.time + shift, n.duration, list(n.waypoints)) for n in chart.notes]
    return Chart(notes, chart.name, chart.bpm, chart.audio, chart.audio_offset + shift, chart.source)


def dedupe(notes: list[Note]) -> list[Note]:
    """Two notes mapped onto the same lane at the same time become one."""
    seen = {}
    for n in notes:
        key = (round(n.time, 3), n.lane)
        if key not in seen or n.duration > seen[key].duration:
            seen[key] = n
    return list(seen.values())


# ---------------------------------------------------------------------------
# osu!mania
# ---------------------------------------------------------------------------
def parse_osu(text: str, n_lanes: int) -> Chart:
    section = None
    general: dict[str, str] = {}
    difficulty: dict[str, str] = {}
    metadata: dict[str, str] = {}
    hit_objects: list[str] = []
    timing: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        m = re.match(r"^\[(\w+)\]$", line)
        if m:
            section = m.group(1)
            continue
        if section in ("General", "Difficulty", "Metadata"):
            if ":" in line:
                k, v = line.split(":", 1)
                {"General": general, "Difficulty": difficulty, "Metadata": metadata}[section][
                    k.strip()
                ] = v.strip()
        elif section == "HitObjects":
            hit_objects.append(line)
        elif section == "TimingPoints":
            timing.append(line)
    if general.get("Mode", "0") != "3":
        raise ValueError("not an osu!mania beatmap (Mode != 3)")
    keys = int(float(difficulty.get("CircleSize", "4")))
    lanes = map_columns(keys, n_lanes)
    notes = []
    for line in hit_objects:
        parts = line.split(",")
        x, _y, t_ms, kind = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
        col = min(int(x * keys // 512), keys - 1)
        t = t_ms / 1000.0
        if kind & 128:  # mania hold: objectParams = endTime:hitSample
            end_ms = int(parts[5].split(":")[0])
            notes.append(Note(HOLD, lanes[col], t, max((end_ms - t_ms) / 1000.0, 0.05)))
        elif kind & 1:
            notes.append(Note(TAP, lanes[col], t))
    bpm = None
    for line in timing:
        p = line.split(",")
        if len(p) > 6 and p[6] == "1" and float(p[1]) > 0:
            bpm = 60000.0 / float(p[1])
            break
    title = metadata.get("Title", "osu")
    version = metadata.get("Version", "")
    return Chart(
        dedupe(notes),
        name=_slug(f"{title}-{version}" if version else title),
        bpm=bpm,
        audio=general.get("AudioFilename"),
        source="osu!mania",
    )


# ---------------------------------------------------------------------------
# StepMania (.sm / .ssc)
# ---------------------------------------------------------------------------
_NOTE_CHARS = {"1": "tap", "2": "hold", "4": "hold", "3": "tail"}


def _sm_tags(text: str) -> list[tuple[str, str]]:
    text = re.sub(r"//[^\n]*", "", text)
    return [(m.group(1).upper(), m.group(2).strip()) for m in re.finditer(r"#([^:;]+):([^;]*);", text, re.S)]


class _BeatClock:
    """Beats -> seconds under #BPMS and #STOPS (StepMania semantics)."""

    def __init__(self, bpms: str, stops: str, offset: float):
        self.bpms = sorted(
            (float(b), float(v)) for b, v in (x.split("=") for x in bpms.split(",") if "=" in x)
        )
        self.stops = sorted(
            (float(b), float(v)) for b, v in (x.split("=") for x in stops.split(",") if "=" in x)
        )
        self.offset = offset

    def time(self, beat: float) -> float:
        t = -self.offset
        prev_beat, bpm = self.bpms[0][0], self.bpms[0][1]
        for b, v in self.bpms[1:]:
            if b >= beat:
                break
            t += (b - prev_beat) * 60.0 / bpm
            prev_beat, bpm = b, v
        t += (beat - prev_beat) * 60.0 / bpm
        for b, dur in self.stops:
            if b < beat:
                t += dur
        return t


def parse_sm(text: str, n_lanes: int, difficulty: str | None = None) -> Chart:
    tags = _sm_tags(text)
    head = {k: v for k, v in tags if k != "NOTES" and k != "NOTEDATA"}
    clock = _BeatClock(head.get("BPMS", "0=120"), head.get("STOPS", ""), float(head.get("OFFSET", "0") or 0))
    # Collect charts: .sm packs everything into #NOTES; .ssc has #NOTEDATA
    # blocks followed by their own tags.
    charts: list[tuple[str, str, str]] = []  # (steps type, difficulty, note data)
    cur: dict[str, str] = {}
    for k, v in tags:
        if k == "NOTES" and ":" in v:  # .sm
            f = [x.strip() for x in v.split(":")]
            charts.append((f[0], f[2], f[-1]))
        elif k == "NOTEDATA":
            cur = {}
        elif k in ("STEPSTYPE", "DIFFICULTY"):
            cur[k] = v
        elif k == "NOTES":  # .ssc
            charts.append((cur.get("STEPSTYPE", ""), cur.get("DIFFICULTY", ""), v))
    if not charts:
        raise ValueError("no #NOTES found")
    singles = [c for c in charts if "single" in c[0] or c[0] == ""] or charts
    pick = singles[-1]
    if difficulty:
        for c in singles:
            if c[1].lower() == difficulty.lower():
                pick = c
    steps_type, diff, data = pick
    panels = 4 if "single" in steps_type else (8 if "double" in steps_type else 4)
    lanes = map_columns(panels, n_lanes)
    notes: list[Note] = []
    open_holds: dict[int, float] = {}
    measures = [m.strip() for m in data.split(",")]
    for mi, measure in enumerate(measures):
        rows = [r.strip() for r in measure.splitlines() if r.strip()]
        for ri, row in enumerate(rows):
            beat = 4.0 * (mi + ri / len(rows))
            t = clock.time(beat)
            for col, ch in enumerate(row[:panels]):
                what = _NOTE_CHARS.get(ch)
                if what == "tap":
                    notes.append(Note(TAP, lanes[col], t))
                elif what == "hold":
                    open_holds[col] = t
                elif what == "tail" and col in open_holds:
                    t0 = open_holds.pop(col)
                    notes.append(Note(HOLD, lanes[col], t0, max(t - t0, 0.05)))
    bpm = clock.bpms[0][1] if clock.bpms else None
    title = head.get("TITLE", "stepmania")
    return Chart(
        dedupe(notes),
        name=_slug(f"{title}-{diff}" if diff else title),
        bpm=bpm,
        audio=head.get("MUSIC") or None,
        source=f"StepMania {steps_type} {diff}".strip(),
    )


# ---------------------------------------------------------------------------
# Clone Hero / Moonscraper .chart
# ---------------------------------------------------------------------------
def parse_clonehero(text: str, n_lanes: int, difficulty: str = "ExpertSingle") -> Chart:
    sections: dict[str, list[str]] = {}
    cur = None
    for raw in text.splitlines():
        line = raw.strip()
        m = re.match(r"^\[(.+)\]$", line)
        if m:
            cur = m.group(1)
            sections[cur] = []
        elif line and line not in "{}" and cur:
            sections[cur].append(line)
    song = dict(
        (k.strip(), v.strip().strip('"'))
        for k, v in (l.split("=", 1) for l in sections.get("Song", []) if "=" in l)
    )
    resolution = float(song.get("Resolution", 192))
    offset = float(song.get("Offset", 0))
    # Tempo map: tick -> bpm (B values are bpm * 1000).
    tempos = []
    for l in sections.get("SyncTrack", []):
        m = re.match(r"^(\d+)\s*=\s*B\s+(\d+)", l)
        if m:
            tempos.append((int(m.group(1)), int(m.group(2)) / 1000.0))
    tempos.sort()
    if not tempos:
        tempos = [(0, 120.0)]

    def tick_time(tick: int) -> float:
        t = offset
        prev_tick, bpm = tempos[0]
        for tk, b in tempos[1:]:
            if tk >= tick:
                break
            t += (tk - prev_tick) / resolution * 60.0 / bpm
            prev_tick, bpm = tk, b
        return t + (tick - prev_tick) / resolution * 60.0 / bpm

    if difficulty not in sections:
        avail = [k for k in sections if k.endswith("Single")]
        raise ValueError(f"no [{difficulty}] section; available: {avail}")
    lanes = map_columns(5, n_lanes)
    notes = []
    for l in sections[difficulty]:
        m = re.match(r"^(\d+)\s*=\s*N\s+(\d+)\s+(\d+)", l)
        if not m:
            continue
        tick, fret, length = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if fret > 4:  # 5/6 = forced/tap modifiers, 7 = open note
            continue
        t = tick_time(tick)
        if length > 0:
            notes.append(Note(HOLD, lanes[fret], t, max(tick_time(tick + length) - t, 0.05)))
        else:
            notes.append(Note(TAP, lanes[fret], t))
    return Chart(
        dedupe(notes),
        name=_slug(f"{song.get('Name', 'chart')}-{difficulty}"),
        bpm=tempos[0][1],
        audio=song.get("MusicStream") or None,
        source=f"Clone Hero {difficulty}",
    )


# ---------------------------------------------------------------------------
# GuitarHeroBot's hand-transcribed songs
# ---------------------------------------------------------------------------
# (button(s), start time, sustain) as typed in GuitarHeroBotFinal.ipynb.  A
# two-digit button means a chord of both; sustain 0 is a plain strum.
GUITARHEROBOT_SONGS = {
    "slow-ride-easy": [
        (1, 4.36, 0.25), (1, 5.09, 0.25), (1, 5.59, 0.00), (1, 6.16, 0.25), (1, 6.68, 0.00),
        (1, 7.21, 0.00), (2, 7.49, 0.00), (2, 7.96, 0.00), (1, 8.48, 0.25), (1, 9.23, 0.25),
        (1, 9.73, 0.00), (1, 10.2, 0.25), (2, 10.9, 1.00), (3, 13.0, 6.95),
    ],
    "slow-ride-medium": [
        (1, 4.44, 0.00), (1, 4.68, 0.00), (1, 5.24, 0.25), (1, 5.73, 0.00), (1, 6.23, 0.25),
        (1, 6.78, 0.00), (1, 7.29, 0.00), (2, 7.56, 0.00), (1, 7.81, 0.00), (2, 8.08, 0.25),
        (1, 8.59, 0.00), (1, 8.83, 0.00), (1, 9.31, 0.00), (1, 9.89, 0.00), (1, 10.4, 0.00),
        (1, 10.8, 0.00), (2, 11.1, 1.00), (4, 12.5, 6.95),
    ],
    "slow-ride-hard": [
        (13, 4.31, 0.00), (13, 4.51, 0.00), (13, 5.11, 0.25), (13, 5.59, 0.00), (13, 6.11, 0.25),
        (13, 6.63, 0.00), (13, 7.21, 0.00), (2, 7.48, 0.00), (1, 7.69, 0.00), (2, 7.99, 0.25),
        (13, 8.46, 0.00), (13, 8.71, 0.00), (13, 9.24, 0.00), (1, 9.51, 0.00), (13, 9.76, 0.00),
        (13, 10.2, 0.00), (1, 10.4, 0.00), (13, 10.7, 0.00), (2, 10.9, 1.00), (5, 12.4, 6.95),
    ],
}


def from_guitarherobot(name: str, n_lanes: int) -> Chart:
    lanes = map_columns(5, n_lanes)
    notes = []
    for buttons, start, sustain in GUITARHEROBOT_SONGS[name]:
        for digit in str(buttons):
            lane = lanes[int(digit) - 1]
            if sustain > 0 and len(str(buttons)) == 1:
                notes.append(Note(HOLD, lane, start, sustain))
            else:
                # Sustained chords become two taps: one finger per lane can
                # tap both, but cannot hold both.
                notes.append(Note(TAP, lane, start))
    return Chart(notes, name=name, bpm=None, source="GuitarHeroBot notebook (Foghat, Slow Ride)")


# ---------------------------------------------------------------------------
def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower() or "chart"


def import_file(path: str | Path, n_lanes: int = DEFAULT.layout.n_lanes, difficulty: str | None = None) -> Chart:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    ext = path.suffix.lower()
    if ext == ".osu":
        chart = parse_osu(text, n_lanes)
    elif ext in (".sm", ".ssc"):
        chart = parse_sm(text, n_lanes, difficulty)
    elif ext == ".chart":
        chart = parse_clonehero(text, n_lanes, difficulty or "ExpertSingle")
    else:
        raise ValueError(f"unknown chart format {ext!r}")
    return chart


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", help=".osu, .sm, .ssc or .chart file")
    ap.add_argument("--guitarherobot", action="store_true", help="write GuitarHeroBot's Slow Ride charts")
    ap.add_argument("--out", required=True, help="output .json file, or a directory")
    ap.add_argument("--difficulty", help="StepMania difficulty name or Clone Hero section (e.g. ExpertSingle)")
    ap.add_argument("--tempo", type=float, default=1.0, help="play the chart at this fraction of its speed")
    ap.add_argument("--min-gap", type=float, default=0.0, help="drop notes closer than this (s) to the previous one")
    ap.add_argument("--lead-in", type=float, default=2.5, help="delay chart and song so the first note is at least this late (s)")
    args = ap.parse_args(argv)

    n_lanes = DEFAULT.layout.n_lanes
    charts = []
    if args.guitarherobot:
        charts = [from_guitarherobot(n, n_lanes) for n in GUITARHEROBOT_SONGS]
    elif args.file:
        if not Path(args.file).exists():
            ap.error(
                f"{args.file!r} does not exist. Give the path to a real chart file: an "
                "osu!mania .osu (unzip an .osz beatmap you downloaded; use the difficulty "
                "whose [General] has Mode: 3), a StepMania .sm/.ssc, or a Clone Hero "
                "notes.chart. For a ready-made example run with --guitarherobot."
            )
        chart = import_file(args.file, n_lanes, args.difficulty)
        if chart.audio:
            # Keep the audio path resolvable from wherever the JSON lands.
            chart.audio = str((Path(args.file).parent / chart.audio).resolve())
        charts = [chart]
    else:
        ap.error("give a chart file or --guitarherobot")

    out = Path(args.out)
    for chart in charts:
        if args.min_gap > 0:
            chart = Chart(thin(chart.notes, args.min_gap), chart.name, chart.bpm, chart.audio, chart.audio_offset, chart.source)
        chart = with_lead_in(chart, args.lead_in)
        if args.tempo != 1.0:
            chart = chart.scaled(args.tempo)
        path = out / f"{chart.name}.json" if (out.is_dir() or out.suffix == "") else out
        path.parent.mkdir(parents=True, exist_ok=True)
        chart.save(path)
        print(f"{chart.name}: {chart.counts()}  {chart.end_time:.1f} s  -> {path}")


if __name__ == "__main__":
    main()
