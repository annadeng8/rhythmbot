"""Scoring: did the fingertips do what the chart asked, and how precisely?

The judge watches the touch sensor at a fixed rate and compares against the
chart (ground truth, not the brain's belief), producing the proposal's
metrics:

  * note completion rate (hit, and for holds/slides contact maintained for
    at least `Timing.hold_fraction` of the note);
  * timing accuracy: touch-down time minus chart hit time, judged
    perfect/great/good/miss by `Timing` windows, like the game does;
  * hold contact maintenance: fraction of the hold during which a tip was
    down in the lane;
  * slide tracking error: lateral distance from the fingertip to the moving
    target along the path, RMS over the slide.

A tap counts only for a *fresh* touch: a tip that has been down since
before the window (e.g. still finishing a hold) does not hit the next note
in that lane.  Both fingertips are eligible, which is what makes two-finger
chords score.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from pydrake.all import LeafSystem

from .chart import HOLD, SLIDE, TAP, Chart, Note
from .config import Config
from .robot import touch_width, unpack_touch


@dataclass
class NoteResult:
    id: int
    kind: str
    lane: float
    time: float
    duration: float
    status: str = "pending"  # pending | hit | miss
    timing_error: float | None = None
    judgment: str | None = None  # perfect | great | good | miss
    head_judgment: str | None = None
    judged_at: float | None = None
    gap_seconds: float = 0.0
    max_gap_seconds: float = 0.0
    tail_ticks: int = 0
    tail_contact_ticks: int = 0
    failure_reason: str | None = None
    contact_ticks_by_hand: list[int] = field(default_factory=lambda: [0, 0])
    handoffs: int = 0
    last_contact_hand: int | None = None
    active_ticks: int = 0
    contact_ticks: int = 0
    lateral_errors: list[float] = field(default_factory=list)
    completed: bool | None = None

    @property
    def maintenance(self) -> float | None:
        if self.active_ticks == 0:
            return None
        return self.contact_ticks / self.active_ticks

    @property
    def slide_rms(self) -> float | None:
        if not self.lateral_errors:
            return None
        return float(np.sqrt(np.mean(np.square(self.lateral_errors))))


class Judge(LeafSystem):
    def __init__(self, chart: Chart, cfg: Config):
        super().__init__()
        self.chart = chart
        self.cfg = cfg
        self.L = cfg.layout
        self.T = cfg.timing
        self.touch_port = self.DeclareVectorInputPort("touch", touch_width(cfg))
        self.DeclarePeriodicPublishEvent(1.0 / cfg.judge_hz, 0.0, self._tick)
        self.results: dict[int, NoteResult] = {
            n.id: NoteResult(n.id, n.kind, n.lane, n.time, n.duration) for n in chart.notes
        }
        n_tips = cfg.gripper.contacts_per_arm * cfg.arms.n
        self.down_since: list[float | None] = [None] * n_tips
        # A fresh touch may be spent on one note only.
        self.touch_used: list[float | None] = [None] * n_tips
        self.t_last = 0.0

    # -- geometry helpers -----------------------------------------------------
    def _in_lane(self, x: float, lane: float, tol_lanes: float = 0.5) -> bool:
        return abs(x - self.L.lane_x(lane)) <= tol_lanes * self.L.lane_width + 0.004

    def _near_line(self, y: float) -> bool:
        return abs(y - self.L.hit_y) <= 0.06

    # -- the tick -------------------------------------------------------------
    def _tick(self, context):
        t = context.get_time()
        touches = unpack_touch(self.touch_port.Eval(context))
        for k, (down, p) in enumerate(touches):
            if down and self.down_since[k] is None:
                self.down_since[k] = t
                self.touch_used[k] = None
            elif not down:
                self.down_since[k] = None
        for note in self.chart.notes:
            r = self.results[note.id]
            if r.status == "pending":
                self._judge_start(note, r, t, touches)
            if r.status == "hit" and note.kind != TAP and r.completed is None:
                self._track_body(note, r, t, touches)
        self.t_last = t

    def _judge_start(self, note: Note, r: NoteResult, t: float, touches) -> None:
        if t < note.time - self.T.good:
            return
        if t > note.time + self.T.miss_after:
            r.status, r.judgment, r.completed = "miss", "miss", False
            r.judged_at = t
            return
        for k, (down, p) in enumerate(touches):
            since = self.down_since[k]
            if not down or since is None or self.touch_used[k] is not None:
                continue
            if since < note.time - self.T.good - 0.01:
                continue  # stale contact, not a tap at this note
            if self._in_lane(p[0], note.lane) and self._near_line(p[1]):
                err = since - note.time
                r.status, r.timing_error = "hit", err
                r.head_judgment = self._judgment(abs(err))
                r.judgment = r.head_judgment if note.kind == TAP else None
                r.judged_at = t if note.kind == TAP else None
                if note.kind == SLIDE and t > note.time:
                    missed = max(0, int(round((t-note.time)*self.cfg.judge_hz)))
                    r.active_ticks = missed
                    r.gap_seconds = missed/self.cfg.judge_hz
                    r.max_gap_seconds = r.gap_seconds
                self.touch_used[k] = note.time
                if note.kind == TAP:
                    r.completed = True
                return

    def _judgment(self, abs_err: float) -> str:
        if abs_err <= self.T.perfect:
            return "perfect"
        if abs_err <= self.T.great:
            return "great"
        return "good"

    def _finish_sustain(self, note, r, t):
        if note.kind == SLIDE:
            expected_tail = max(1, int(min(note.duration, self.T.slide_tail_window)*self.cfg.judge_hz))
            tail_ok = r.tail_ticks >= expected_tail and r.tail_contact_ticks == r.tail_ticks
            gap_ok = r.max_gap_seconds <= self.T.slide_max_gap + 1e-9
            fraction_ok = (r.maintenance or 0) >= self.T.slide_min_maintenance
            r.completed = bool(tail_ok and gap_ok and fraction_ok)
            r.failure_reason = ("tail_missing" if not tail_ok else "contact_break" if not gap_ok
                                else "insufficient_tracking" if not fraction_ok else None)
        else:
            r.completed = (r.maintenance or 0) >= self.T.hold_fraction
            r.failure_reason = None if r.completed else "insufficient_contact"
        r.judgment = r.head_judgment if r.completed else "miss"
        r.judged_at = t

    def _track_body(self, note: Note, r: NoteResult, t: float, touches) -> None:
        if t > note.end_time + 1e-9:
            self._finish_sustain(note, r, t)
            return
        if t < note.time:
            return
        r.active_ticks += 1
        target_x = self.L.lane_x(note.lane_at(t))
        tol = 0.5 if note.kind == HOLD else self.T.slide_tolerance_lanes
        down_xs = [p[0] for down, p in touches if down and self._near_line(p[1])]
        error = min((abs(x-target_x) for x in down_xs), default=float("inf"))
        tracked = error <= tol * self.L.lane_width + (0.004 if note.kind == HOLD else 0)
        if note.kind == SLIDE:
            if down_xs: r.lateral_errors.append(error)
            r.gap_seconds = 0.0 if tracked else r.gap_seconds + 1/self.cfg.judge_hz
            r.max_gap_seconds = max(r.max_gap_seconds, r.gap_seconds)
            if t >= note.end_time-self.T.slide_tail_window-1e-9:
                r.tail_ticks += 1
                # Tail requires both being down and reaching the final lane.
                tail_error = min((abs(x-self.L.lane_x(note.lane_at(note.end_time)))
                                  for x in down_xs), default=float("inf"))
                r.tail_contact_ticks += int(tracked and tail_error <= tol*self.L.lane_width)
        if tracked:
            r.contact_ticks += 1
            if note.kind == SLIDE:
                eligible = [(abs(p[0]-target_x), k//self.cfg.gripper.contacts_per_arm) for k,(down,p) in enumerate(touches)
                            if down and self._near_line(p[1])]
                hand = min(eligible)[1]
                r.contact_ticks_by_hand[hand] += 1
                if r.last_contact_hand is not None and hand != r.last_contact_hand:
                    r.handoffs += 1
                r.last_contact_hand = hand

    # -- report -----------------------------------------------------------------
    def finalize(self, t_end: float) -> None:
        for note in self.chart.notes:
            r = self.results[note.id]
            if r.status == "pending":
                r.status, r.judgment, r.completed = "miss", "miss", False
            if r.completed is None:
                if t_end >= note.end_time:
                    self._finish_sustain(note, r, t_end)
                else:
                    r.completed = False
                    r.judgment = "miss"
                    r.failure_reason = "truncated"
                    r.judged_at = t_end

    def report(self, extra: dict | None = None) -> dict:
        rs = list(self.results.values())
        by_kind = {}
        for kind in (TAP, HOLD, SLIDE):
            ks = [r for r in rs if r.kind == kind]
            if not ks:
                continue
            by_kind[kind] = {
                "n": len(ks),
                "hit": sum(r.status == "hit" for r in ks),
                "completed": sum(bool(r.completed) for r in ks),
            }
            if kind != TAP:
                ms = [r.maintenance for r in ks if r.maintenance is not None]
                by_kind[kind]["mean_maintenance"] = float(np.mean(ms)) if ms else None
            if kind == SLIDE:
                es = [r.slide_rms for r in ks if r.slide_rms is not None]
                by_kind[kind]["mean_slide_rms_m"] = float(np.mean(es)) if es else None
        errs = np.array([r.timing_error for r in rs if r.timing_error is not None])
        judg = {j: sum(r.judgment == j for r in rs) for j in ("perfect", "great", "good", "miss")}
        score = 3 * judg["perfect"] + 2 * judg["great"] + judg["good"]
        rep = {
            "chart": self.chart.name,
            "n_notes": len(rs),
            "completion_rate": float(np.mean([bool(r.completed) for r in rs])) if rs else None,
            "hit_rate": float(np.mean([r.status == "hit" for r in rs])) if rs else None,
            "judgments": judg,
            "score": score,
            "max_score": 3 * len(rs),
            "timing_error_mean_s": float(errs.mean()) if len(errs) else None,
            "timing_error_abs_mean_s": float(np.abs(errs).mean()) if len(errs) else None,
            "timing_error_std_s": float(errs.std()) if len(errs) else None,
            "by_kind": by_kind,
            "notes": [
                {
                    **{k: v for k, v in asdict(r).items() if k != "lateral_errors"},
                    "maintenance": r.maintenance,
                    "slide_rms_m": r.slide_rms,
                }
                for r in rs
            ],
        }
        if extra:
            rep.update(extra)
        return rep

    @staticmethod
    def print_report(rep: dict) -> None:
        print(f"\n=== {rep['chart']}: {rep['n_notes']} notes ===")
        print(
            f"completion {100 * rep['completion_rate']:.1f}%   hit {100 * rep['hit_rate']:.1f}%"
            f"   score {rep['score']}/{rep['max_score']}"
        )
        j = rep["judgments"]
        print(
            f"perfect {j['perfect']}  great {j['great']}  good {j['good']}  miss {j['miss']}"
        )
        if rep["timing_error_mean_s"] is not None:
            print(
                f"timing error: mean {1000 * rep['timing_error_mean_s']:+.0f} ms, "
                f"|mean| {1000 * rep['timing_error_abs_mean_s']:.0f} ms, "
                f"std {1000 * rep['timing_error_std_s']:.0f} ms"
            )
        for kind, d in rep["by_kind"].items():
            line = f"  {kind:5s}: {d['completed']}/{d['n']} completed, {d['hit']} hit"
            if d.get("mean_maintenance") is not None:
                line += f", contact maintained {100 * d['mean_maintenance']:.0f}%"
            if d.get("mean_slide_rms_m") is not None:
                line += f", slide RMS error {1000 * d['mean_slide_rms_m']:.1f} mm"
            print(line)
        if rep.get("skipped_by_planner"):
            print(f"  planner skipped notes: {sorted(rep['skipped_by_planner'])}")
        misses = [n for n in rep["notes"] if n["status"] == "miss"]
        if misses:
            print("  missed:", ", ".join(f"#{n['id']} {n['kind']}@{n['time']:.2f}" for n in misses))

    @staticmethod
    def save_report(rep: dict, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(rep, indent=1))
