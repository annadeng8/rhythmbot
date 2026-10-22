"""The brain: perception tick, note tracking, replanning, pose output.

`GameScreen` renders the game frame for the current simulation time on
demand (an abstract output port).  `RhythmBrain` runs a periodic update at
the perception rate: it pulls a frame if the detector wants one, asks the
detector for notes, merges them into its tracked set, drops notes that are
finished, and asks the policy for a fresh plan starting from the pose it is
currently commanding.  Between ticks it outputs the planned gripper pose and
width for the current time.

Everything the brain knows lives in one abstract state variable, updated only
in the periodic event, so the output ports are pure functions of state and
time as Drake expects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydrake.all import AbstractValue, LeafSystem, RigidTransform

from .chart import SLIDE, Chart
from .config import Config
from .game import FrameRenderer
from .perception import DetectedNote, NoteDetector
from .planner import Plan, Policy, R_DOWN, TrackedNote


class GameScreen(LeafSystem):
    """Outputs the rendered game frame (H, W, 3 uint8) for the current time."""

    def __init__(self, chart: Chart, cfg: Config):
        super().__init__()
        self.renderer = FrameRenderer(cfg)
        self.chart = chart
        self.DeclareAbstractOutputPort(
            "frame",
            lambda: AbstractValue.Make(np.zeros((1, 1, 3), dtype=np.uint8)),
            self._calc,
        )

    def _calc(self, context, output):
        output.set_value(self.renderer.render(self.chart, context.get_time()))


@dataclass
class BrainState:
    tracked: dict[int, TrackedNote] = field(default_factory=dict)
    plan: Plan | None = None
    X_WT_cmd: RigidTransform = field(default_factory=RigidTransform)
    width_cmd: float = 0.0
    next_key: int = 0
    n_finished: int = 0
    skipped_note_ids: set[int] = field(default_factory=set)
    n_detections: int = 0


class RhythmBrain(LeafSystem):
    def __init__(
        self,
        detector: NoteDetector,
        policy: Policy,
        cfg: Config,
        X_WG_initial: RigidTransform,
        width_initial: float = 0.0,
    ):
        super().__init__()
        self.detector = detector
        self.policy = policy
        self.cfg = cfg
        self.X_GT = RigidTransform([0.0, cfg.gripper.tip_offset, 0.0])
        self.X_TG = self.X_GT.inverse()

        X_WT0 = X_WG_initial @ self.X_GT
        state0 = BrainState(X_WT_cmd=X_WT0, width_cmd=width_initial)
        state0.plan = policy.plan([], 0.0, X_WT0, width_initial)
        self._state_index = self.DeclareAbstractState(AbstractValue.Make(state0))

        self.frame_port = self.DeclareAbstractInputPort(
            "frame", AbstractValue.Make(np.zeros((1, 1, 3), dtype=np.uint8))
        )
        self.DeclarePeriodicUnrestrictedUpdateEvent(
            1.0 / cfg.perception.tick_hz, 0.0, self._tick
        )
        self.DeclareAbstractOutputPort(
            "X_WG_desired", lambda: AbstractValue.Make(RigidTransform()), self._calc_pose
        )
        self.DeclareVectorOutputPort("wsg_position", 1, self._calc_width)
        # Mirror of the latest state for reporting after the run.
        self.last_state: BrainState = state0

    # -- state access --------------------------------------------------------
    def _get(self, context) -> BrainState:
        return context.get_abstract_state(self._state_index).get_value()

    # -- perception + planning tick -----------------------------------------
    def _tick(self, context, state):
        t = context.get_time()
        s: BrainState = state.get_mutable_abstract_state(self._state_index).get_mutable_value()

        frame = self.frame_port.Eval(context) if self.detector.needs_frames else None
        detections = self.detector.detect(frame, t)
        s.n_detections += len(detections)
        self._merge(s, detections, t)
        self._prune(s, t)

        # Where the previous plan has the hand right now: the new plan starts
        # there, so the commanded pose is continuous across replans.
        if s.plan is not None:
            s.X_WT_cmd = s.plan.tip_pose(t)
            s.width_cmd = s.plan.width_at(t)
        # Only notes seen in at least two frames are planned for: a single
        # spurious detection (vision models hiccup right as a head crosses
        # the line) must not send the arm chasing a phantom.
        confirmed = [n for n in s.tracked.values() if n.n_obs >= self.cfg.perception.min_observations]
        s.plan = self.policy.plan(confirmed, t, s.X_WT_cmd, s.width_cmd)
        self.last_state = s

    def _merge(self, s: BrainState, detections: list[DetectedNote], t: float) -> None:
        for d in detections:
            match = self._match(s, d)
            if match is None and d.active:
                # First seen only once it is already being played: its hit
                # is lost, and chasing it would only spoil the next note.
                continue
            if match is None:
                s.tracked[s.next_key] = TrackedNote(
                    key=s.next_key,
                    kind=d.kind,
                    lane=d.lane,
                    hit_time=d.hit_time,
                    duration=d.duration,
                    duration_final=d.duration_final,
                    waypoints=list(d.waypoints),
                    note_id=d.note_id,
                    last_seen=t,
                )
                s.next_key += 1
                continue
            if d.active:
                # Already playing: the detection only tells us where the
                # note ends; the hit time and path stay as committed.  While
                # the end is still off-screen the report is a lower bound,
                # so the duration only grows (this is what keeps a long hold
                # down).  Once the end is in view the report is a measurement:
                # it replaces whatever was guessed before, overestimates
                # included, and later frames are averaged in.
                old = match.duration
                new = d.end_time - match.hit_time
                if d.duration_final:
                    match.duration = 0.5 * (old + new) if match.active_final else new
                    match.active_final = True
                elif not match.active_final:
                    match.duration = max(new, old)
                match.duration_final = match.duration_final or d.duration_final
                self._rescale_slide(match, old)
                match.n_obs += 1
                match.last_seen = t
                continue
            # Refine: average the hit time (noisy detectors), take the newer
            # duration/path while the tail is still coming into view.
            if t < match.hit_time - 0.1:
                alpha = 1.0 / (match.n_obs + 1)
                match.hit_time += alpha * (d.hit_time - match.hit_time)
            if not match.duration_final or d.duration_final:
                if d.duration >= match.duration - 0.05 or d.duration_final:
                    match.duration = d.duration
                    match.waypoints = list(d.waypoints) or match.waypoints
                match.duration_final = d.duration_final
            if match.note_id < 0:
                match.note_id = d.note_id
            match.n_obs += 1
            match.last_seen = t

    @staticmethod
    def _rescale_slide(n: TrackedNote, old_duration: float) -> None:
        """A slide's path was predicted at fractions of its duration; when
        the duration is corrected, stretch the path to match."""
        if n.kind != SLIDE or not n.waypoints or old_duration <= 0:
            return
        ratio = n.duration / old_duration
        if abs(ratio - 1.0) > 0.05:
            n.waypoints = [(off * ratio, lane) for off, lane in n.waypoints]

    @staticmethod
    def _match(s: BrainState, d: DetectedNote) -> TrackedNote | None:
        if d.note_id >= 0:
            for n in s.tracked.values():
                if n.note_id == d.note_id:
                    return n
        if d.active:
            # Its head is on the line: match the note of this kind that is
            # being played now (slides have moved lanes, so lane is ignored).
            for n in s.tracked.values():
                if n.kind == d.kind and n.hit_time <= d.hit_time + 0.1 and n.end_time >= d.hit_time - 0.1:
                    if n.kind == SLIDE or abs(n.lane - d.lane) <= 0.6:
                        return n
            return None
        best, best_dt = None, 0.15
        for n in s.tracked.values():
            if n.kind != d.kind or abs(n.lane - d.lane) > 0.6:
                continue
            dt = abs(n.hit_time - d.hit_time)
            if dt < best_dt:
                best, best_dt = n, dt
        return best

    def _prune(self, s: BrainState, t: float) -> None:
        post = self.cfg.motion.post_time
        for key in list(s.tracked):
            n = s.tracked[key]
            if t > n.end_time + post + 0.05:
                # Report a skip only if it stood when the note finished.
                if n.skipped and n.note_id >= 0:
                    s.skipped_note_ids.add(n.note_id)
                del s.tracked[key]
                s.n_finished += 1
            elif n.n_obs < self.cfg.perception.min_observations and t - n.last_seen > 0.1:
                # Reported once, then never again: a false positive.
                del s.tracked[key]
            elif t < n.hit_time and t - n.last_seen > self.cfg.perception.stale_after:
                del s.tracked[key]

    # -- outputs --------------------------------------------------------------
    def _calc_pose(self, context, output):
        s = self._get(context)
        t = context.get_time() + self.cfg.motion.command_lead
        X_WT = s.plan.tip_pose(t) if s.plan is not None else s.X_WT_cmd
        output.set_value(X_WT @ self.X_TG)

    def _calc_width(self, context, output):
        s = self._get(context)
        t = context.get_time() + self.cfg.motion.command_lead
        w = s.plan.width_at(t) if s.plan is not None else s.width_cmd
        output.SetFromVector([w])


def initial_tip_pose(cfg: Config, x: float = 0.0) -> RigidTransform:
    """A sensible rest pose: hovering above the judgment line, pointing down."""
    return RigidTransform(
        R_DOWN, [x, cfg.layout.hit_y, cfg.layout.screen_z + cfg.motion.hover_height]
    )
