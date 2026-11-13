"""Chart sets shared by the policy comparisons.

Policies are developed on one set of charts and judged on others they have
never seen.  The coding agent gets feedback only from `cap-train`; behavior
cloning collects demonstrations only on `bc-train`; neither ever sees a
held-out chart.  Held-out charts vary along the axes the proposal names:

    heldout-iid      more charts from the same generator as the training ones
    heldout-dense    denser charts: shorter gaps, more chords and holds
    heldout-real     hand-made and imported charts (demo, Slow Ride, osu!mania)
    hard             osu!mania-style dense charts (`dense_random_chart`): up to
                     ~13 notes/s and four-note chords, more than one arm can play

Procedural charts are generated from seeds, so a split is a list of names
that `load` turns back into charts.
"""

from __future__ import annotations

from . import PROJECT_ROOT
from .chart import Chart, demo_chart, dense_random_chart, random_chart, varied_random_chart

CHARTS = PROJECT_ROOT / "charts"
SONGS = PROJECT_ROOT / "songs"


def _dense(seed: int) -> Chart:
    return random_chart(
        seed, density=3.2, min_gap=0.18, p_hold=0.3, p_slide=0.2, p_chord=0.25,
        hold_range=(0.15, 1.0), slide_range=(0.3, 1.0), p_repeat_lane=0.3, name=f"dense-{seed}",
    )


SPLITS: dict[str, list[str]] = {
    "cap-train": ["file:random-1", "file:random-2", "varied:100", "varied:101", "varied:102", "varied:103"],
    "bc-train": [f"varied:{s}" for s in range(1000, 1200)],
    "heldout-iid": [f"varied:{s}" for s in range(5000, 5010)],
    "heldout-dense": [f"dense:{s}" for s in range(6000, 6005)],
    "hard": [f"hard:{s}" for s in range(7000, 7010)],
    "heldout-real": [
        "demo",
        "file:random-3",
        "file:slow-ride-easy",
        "file:slow-ride-medium",
        "file:slow-ride-hard",
        "song:nekoteto/easy-neko",
    ],
}
SPLITS["heldout"] = SPLITS["heldout-iid"] + SPLITS["heldout-dense"] + SPLITS["heldout-real"]


def load(name: str) -> Chart | None:
    """A chart by split entry name; None if it is a song chart that is not
    on disk (songs/ is gitignored)."""
    kind, _, arg = name.partition(":")
    if kind == "demo":
        return demo_chart()
    if kind == "file":
        return Chart.load(CHARTS / f"{arg}.json")
    if kind == "song":
        path = SONGS / f"{arg}.json"
        return Chart.load(path) if path.exists() else None
    if kind == "varied":
        return varied_random_chart(int(arg))
    if kind == "dense":
        return _dense(int(arg))
    if kind == "hard":
        return dense_random_chart(int(arg))
    raise ValueError(f"unknown chart {name!r}")


def expand(split: str) -> list[str]:
    """A comma-separated list of splits and chart names, with `heldout`
    spelled out into its three parts."""
    out = []
    for part in split.split(","):
        out += ["heldout-iid", "heldout-dense", "heldout-real"] if part == "heldout" else [part]
    return out


def charts(split: str) -> list[Chart]:
    """The charts of a split, or of a comma-separated list of splits and
    chart names."""
    out = []
    for part in expand(split):
        names = SPLITS.get(part, [part])
        out += [c for c in (load(n) for n in names) if c is not None]
    return out

