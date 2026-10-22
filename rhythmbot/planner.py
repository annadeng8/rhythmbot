"""From tracked notes to a fingertip trajectory: the classical policy.

This is approach 1 of the proposal, and the direct descendant of
GuitarHeroBot's `MakeGripperFrames` / `MakeGripperPoseTrajectory`: each note
becomes a few fingertip keyframes (hover above the lane, touch at the hit
time, stay down for a hold or follow the path for a slide, lift), the
keyframes are strung together in time order, and the hand is interpolated
linearly between them (`Plan`).  Four things are new:

  * the plan is rebuilt every perception tick from the *currently commanded*
    pose, so notes can be added and refined as the vision model sees more of
    them;
  * two taps at the same time in lanes up to three apart are played as a
    chord by opening the WSG so one fingertip lands in each lane;
  * notes that cannot be reached in time are skipped explicitly rather than
    smeared into their neighbors;
  * while one fingertip holds a note, the other can play a note in another
    lane: the WSG opens so the free fingertip is over that lane and the hand
    rolls about the holding fingertip, which stays put on the screen, so the
    free one dips to the screen and back.

`Policy` is the interface the brain calls; an LLM or behavior-cloned policy
would implement the same `plan` method.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydrake.all import RigidTransform, RotationMatrix

from .chart import HOLD, SLIDE, TAP
from .config import Config, Gripper

# Tip frame T: same orientation as the gripper frame G, origin at the
# fingertip contact point (midway between the two fingertips).  "Pointing
# down" = G's +y along world -z, with G's x (the finger sliding axis) along
# world x, i.e. across the lanes.
R_DOWN = RotationMatrix.MakeXRotation(-np.pi / 2)


def R_rolled(theta: float) -> RotationMatrix:
    """R_DOWN rolled about world y (along the lanes) so that T's x axis is
    (cos theta, 0, sin theta): the +x fingertip rises for theta > 0."""
    return RotationMatrix.MakeYRotation(-theta) @ R_DOWN


def tip_gap(G: Gripper, width: float) -> float:
    """Distance between the two fingertips for a commanded opening."""
    return 2.0 * G.tip_x_offsets(width)[1]


def roll_for_rise(G: Gripper, width: float, rise: float) -> float:
    """Roll angle that puts the +x fingertip `rise` above the -x one."""
    return float(np.arcsin(np.clip(rise / tip_gap(G, width), -1.0, 1.0)))


@dataclass
class TrackedNote:
    """A note the brain believes in, merged from detections over time."""

    key: int
    kind: str
    lane: float
    hit_time: float
    duration: float = 0.0
    duration_final: bool = True
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    note_id: int = -1
    n_obs: int = 1
    last_seen: float = 0.0
    skipped: bool = False
    # Set once a detection made while the note is being played has seen its
    # end on screen; from then on its end time is measured, not guessed.
    active_final: bool = False
    # For a hold: (fingertip side, x, opening) it is played with, as the
    # latest plan has it, and, fixed once the hold has started, as the hand
    # actually put it down.  Later plans keep the committed fingertip where
    # it is even after the notes that decided it are gone.
    planned_hold: tuple[int, float, float] | None = None
    held: tuple[int, float, float] | None = None

    @property
    def end_time(self) -> float:
        return self.hit_time + self.duration

    def lane_at_offset(self, off: float) -> float:
        if self.kind != SLIDE or not self.waypoints:
            return self.lane
        offs = np.array([w[0] for w in self.waypoints])
        lanes = np.array([w[1] for w in self.waypoints])
        return float(np.interp(off, offs, lanes))


@dataclass
class Keyframe:
    t: float
    x: float
    z: float
    width: float
    y: float
    down: bool  # fingertip meant to be in contact
    # Which point (x, y, z) places: 0 the midpoint between the fingertips,
    # -1 / +1 the fingertip on the -x / +x side.  Between two keyframes the
    # hand moves so that this point travels in a straight line, which is
    # what keeps a holding fingertip still while the hand rolls about it.
    ref: int = 0
    # Height of the +x fingertip above the -x one; nonzero means the hand
    # is rolled about world y.
    rise: float = 0.0


class Plan:
    """A fingertip pose trajectory plus a gripper-width trajectory.

    Each segment between two keyframes is interpolated about one reference
    point, the first keyframe's `ref` (or the second's, if the first is the
    midpoint): its position, the opening and the rise go linearly, and the
    midpoint pose follows from them.  Keyframes are converted exactly to the
    segment's reference, so the trajectory is continuous whichever point a
    neighbouring segment uses.
    """

    def __init__(self, keyframes: list[Keyframe], gripper: Gripper | None = None):
        if len(keyframes) < 2:
            k = keyframes[0]
            keyframes = [k, Keyframe(k.t + 1.0, k.x, k.z, k.width, k.y, k.down, k.ref, k.rise)]
        self.G = gripper or Gripper()
        self.keyframes = keyframes
        self.times = np.array([k.t for k in keyframes])
        self.start_time = float(self.times[0])
        self.end_time = float(self.times[-1])

    def _point(self, k: Keyframe, ref: int) -> np.ndarray:
        """Where keyframe k puts the point `ref`."""
        p = np.array([k.x, k.y, k.z])
        if ref == k.ref:
            return p
        th = roll_for_rise(self.G, k.width, k.rise)
        half_x = 0.5 * tip_gap(self.G, k.width) * np.array([np.cos(th), 0.0, np.sin(th)])
        return p + (ref - k.ref) * half_x

    def _state(self, t: float) -> tuple[int, np.ndarray, float, float]:
        t = float(np.clip(t, self.start_time, self.end_time))
        i = int(np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, len(self.times) - 2))
        k0, k1 = self.keyframes[i], self.keyframes[i + 1]
        ref = k0.ref or k1.ref
        dt = k1.t - k0.t
        s = 0.0 if dt <= 0 else (t - k0.t) / dt
        p = (1 - s) * self._point(k0, ref) + s * self._point(k1, ref)
        width = (1 - s) * k0.width + s * k1.width
        rise = (1 - s) * k0.rise + s * k1.rise
        return ref, p, width, rise

    def tip_pose(self, t: float) -> RigidTransform:
        ref, p, width, rise = self._state(t)
        th = roll_for_rise(self.G, width, rise)
        half_x = 0.5 * tip_gap(self.G, width) * np.array([np.cos(th), 0.0, np.sin(th)])
        return RigidTransform(R_rolled(th), p - ref * half_x)

    def width_at(self, t: float) -> float:
        return float(self._state(t)[2])


class Policy:
    def plan(
        self,
        notes: list[TrackedNote],
        t_now: float,
        X_WT_cmd: RigidTransform,
        width_cmd: float,
    ) -> Plan:
        raise NotImplementedError


@dataclass
class _Action:
    """One thing the hand does: a note, or a two-note chord."""

    t_hit: float
    t_end: float  # last moment the tip is meant to be down
    kind: str
    width: float
    # (offset from t_hit, x) for the down phase; first offset is 0.
    path: list[tuple[float, float]]
    notes: list[TrackedNote]
    # Which point the path places (`Keyframe.ref`): 0 the midpoint, for
    # single notes (both fingertips straddle the lane) and chords; -1 / +1
    # for a hold kept by that fingertip while the other plays around it.
    side: int = 0

    @property
    def x_start(self) -> float:
        return self.path[0][1]

    @property
    def x_end(self) -> float:
        return self.path[-1][1]


@dataclass
class _Hold:
    """A hold being kept down, by fingertip `side` at `x` (0: by both, and
    then nothing can be played around it)."""

    t_end: float
    x: float
    side: int
    width: float
    rise: float
    free_at: float  # when the other fingertip is raised and ready


@dataclass
class _Release:
    """Where the hand leaves the screen from after its last note."""

    t_end: float
    x_end: float  # of point `ref`
    width: float
    kind: str
    ref: int = 0


class ClassicalPlanner(Policy):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.L = cfg.layout
        self.M = cfg.motion
        self.G = cfg.gripper
        self.hover_z = self.L.screen_z + self.M.hover_height
        self.touch_z = self.L.screen_z - self.M.touch_depth

    # -- grouping ------------------------------------------------------------
    def _actions(self, notes: list[TrackedNote]) -> list[_Action]:
        notes = sorted((n for n in notes if not n.skipped), key=lambda n: n.hit_time)
        actions: list[_Action] = []
        i = 0
        while i < len(notes):
            n = notes[i]
            group = [n]
            j = i + 1
            while j < len(notes) and abs(notes[j].hit_time - n.hit_time) < self.M.chord_window:
                group.append(notes[j])
                j += 1
            i = j
            if len(group) >= 2 and all(g.kind == TAP for g in group[:2]):
                a, b = sorted(group[:2], key=lambda g: g.lane)
                gap = abs(self.L.lane_x(b.lane) - self.L.lane_x(a.lane))
                w = self.G.width_for_tip_gap(gap, 2 * self.M.lane_slack)
                if w is not None:
                    xc = 0.5 * (self.L.lane_x(a.lane) + self.L.lane_x(b.lane))
                    t0 = 0.5 * (a.hit_time + b.hit_time)
                    actions.append(
                        _Action(t0, t0 + self.M.tap_dwell, TAP, w, [(0.0, xc)], [a, b])
                    )
                    for extra in group[2:]:
                        extra.skipped = True
                    continue
            # Not a playable chord: play the first, skip the rest.
            for extra in group[1:]:
                extra.skipped = True
            actions.append(self._single(n))
        return actions

    def _single(self, n: TrackedNote) -> _Action:
        if n.kind == TAP:
            path = [(0.0, self.L.lane_x(n.lane))]
            return _Action(n.hit_time, n.hit_time + self.M.tap_dwell, TAP, 0.0, path, [n])
        if n.kind == HOLD:
            side, x, w = n.held or (0, self.L.lane_x(n.lane), 0.0)
            path = [(0.0, x), (max(n.duration, 0.05), x)]
            return _Action(n.hit_time, n.hit_time + path[-1][0], HOLD, w, path, [n], side)
        wps = n.waypoints or [(0.0, n.lane), (max(n.duration, 0.05), n.lane)]
        path = [(float(off), self.L.lane_x(lane)) for off, lane in wps]
        if path[0][0] > 0.0:
            path.insert(0, (0.0, self.L.lane_x(n.lane)))
        return _Action(n.hit_time, n.hit_time + path[-1][0], SLIDE, 0.0, path, [n])

    # -- rolling about a holding fingertip ---------------------------------
    @staticmethod
    def _rise(side: int, dz: float) -> float:
        """`Keyframe.rise` that puts the fingertip opposite `side` dz above
        the one on `side`."""
        return -side * dz

    def _mid_x(self, x: float, ref: int, width: float) -> float:
        return x - ref * 0.5 * tip_gap(self.G, width)

    def _free_fit(self, x_hold: float, b: _Action, can_shift: bool = False) -> tuple[int, float, float] | None:
        """(holding side, opening, shift) that puts the other fingertip over
        note b while one holds at x_hold + shift, or None if b cannot be
        played that way.  When b is farther than the WSG opens, each
        fingertip may land `lane_slack` off its lane centre; the holding one
        only if it is not down yet (`can_shift`)."""
        if len(b.notes) != 1 or b.kind not in (TAP, HOLD):
            return None
        dx = b.x_start - x_hold
        slack = (2 if can_shift else 1) * self.M.lane_slack
        w = self.G.width_for_tip_gap(abs(dx), slack)
        if w is None or abs(dx) < 1e-6:
            return None
        short = abs(dx) - tip_gap(self.G, w)
        shift = np.sign(dx) * 0.5 * short if can_shift and short > 0 else 0.0
        return (-1 if dx > 0 else 1), w, float(shift)

    def _next_fit(self, a: _Action, rest: list[_Action], t_ready: float) -> tuple[int, float, float] | None:
        """`_free_fit` for the first later note that starts while hold a is
        still down and that the other fingertip, ready at t_ready, can play."""
        M = self.M
        for b in rest:
            if b.t_hit >= a.t_end + M.post_time * 0.5:
                break
            fit = self._free_fit(a.x_start, b, can_shift=True)
            if fit is not None and b.t_hit - M.roll_dip_time >= t_ready + 0.01:
                return fit
        return None

    def _choose_pivot(self, a: _Action, rest: list[_Action]) -> None:
        """For a hold about to be played: if a later note starts while it is
        still down and the other fingertip can play it, keep the hold with
        the fingertip away from that note, opened so the free one is over
        it.  Otherwise both fingertips straddle the hold's lane."""
        fit = self._next_fit(a, rest, a.t_hit + self.M.press_time)
        if fit is not None:
            a.side, a.width, shift = fit
            a.path = [(off, x + shift) for off, x in a.path]

    def _roll_in(
        self, a: _Action, rest: list[_Action], hold: _Hold, kfs: list[Keyframe]
    ) -> _Hold | _Release | None:
        """Play action a with the free fingertip while `hold` stays down:
        roll about the holding fingertip so the free one dips onto a's lane.
        Returns the new hold state, or how the hand leaves the screen if the
        hold is over by the time a is, or None if a cannot be played."""
        M, y, z = self.M, self.L.hit_y, self.touch_z
        fit = self._free_fit(hold.x, a)
        if hold.side == 0 or fit is None or fit[0] != hold.side:
            return None
        s, w = hold.side, fit[1]
        t_press = a.t_hit + M.press_time
        if a.kind == HOLD and a.t_end > hold.t_end + 1e-3:
            # a's fingertip will be the one holding next: land it off-centre
            # toward the note the other one plays then, if that is farther
            # than the WSG opens.  That note is on the hold's side, so this
            # only narrows the opening.
            if a.notes[0].held is not None:
                w = a.notes[0].held[2]
            else:
                nxt = self._next_fit(a, rest, max(hold.t_end, t_press) + M.roll_lift_time)
                if nxt is not None and nxt[0] == -s and nxt[2] != 0.0:
                    w = self.G.width_for_tip_gap(abs(a.x_start + nxt[2] - hold.x), M.lane_slack)
        x_free = hold.x - s * tip_gap(self.G, w)
        # Reopening about a holding fingertip swings the whole gripper
        # sideways by half the change, so it is given time.
        t_dip = a.t_hit - M.roll_dip_time
        t_ready = max(hold.free_at, kfs[-1].t)
        if t_dip < t_ready + max(0.01, abs(w - hold.width) / M.pivot_width_speed):
            return None
        up = self._rise(s, M.roll_raise)

        def k(t, rise, ref=s, x=hold.x):
            return Keyframe(t, x, z, w, y, True, ref, rise)

        # Free fingertip: raised over a's lane, crossing the contact height
        # at the hit, pressed in level with the holding one.
        kfs.append(k(t_dip, up))
        kfs.append(k(a.t_hit, self._rise(s, M.contact_tolerance + M.touch_depth)))
        if a.kind == HOLD and a.t_end > hold.t_end + 1e-3:
            # a outlasts the hold: once both are down and the hold is over,
            # a's fingertip becomes the pivot and the other one lifts.
            t_switch = max(hold.t_end, t_press)
            if t_switch > t_press + 1e-3:
                kfs.append(k(t_press, 0.0))
            kfs.append(k(t_switch, 0.0, -s, x_free))
            t_up = t_switch + M.roll_lift_time
            new_up = self._rise(-s, M.roll_raise)
            kfs.append(k(t_up, new_up, -s, x_free))
            if a.notes[0].held is None:
                a.notes[0].planned_hold = (-s, x_free, w)
            return _Hold(a.t_end, x_free, -s, w, new_up, t_up)
        t_done = max(a.t_end if a.kind == HOLD else a.t_hit + M.tap_dwell, t_press)
        kfs.append(k(t_press, 0.0))
        if t_done > t_press + 1e-3:
            kfs.append(k(t_done, 0.0))
        if hold.t_end <= t_done + 1e-3:
            # The hold ended while a was down: lift both from here.
            return _Release(t_done, hold.x, w, a.kind, s)
        t_up = t_done + M.roll_lift_time
        kfs.append(k(t_up, up))
        return _Hold(hold.t_end, hold.x, s, w, up, t_up)

    def _release(self, hold: _Hold, kfs: list[Keyframe]) -> _Release:
        t_end = max(hold.t_end, kfs[-1].t)
        if hold.t_end > kfs[-1].t + 1e-3:
            kfs.append(
                Keyframe(hold.t_end, hold.x, self.touch_z, hold.width, self.L.hit_y, True, hold.side, hold.rise)
            )
        return _Release(t_end, hold.x, hold.width, HOLD, hold.side)

    # -- keyframes -----------------------------------------------------------
    def _keyframes(self, actions: list[_Action]) -> list[Keyframe]:
        M, L = self.M, self.L
        y = L.hit_y
        kfs: list[Keyframe] = []
        prev: _Release | None = None
        hold: _Hold | None = None
        for i, a in enumerate(actions):
            if hold is not None:
                if a.t_hit < hold.t_end + M.post_time * 0.5:
                    # The hold is still down when this one is due: play it
                    # with the other fingertip, or give up.
                    res = self._roll_in(a, actions[i + 1 :], hold, kfs)
                    if res is None:
                        for n in a.notes:
                            n.skipped = True
                    elif isinstance(res, _Hold):
                        hold = res
                    else:
                        hold, prev = None, res
                    continue
                prev, hold = self._release(hold, kfs), None
            if prev is not None:
                gap = a.t_hit - prev.t_end
                if prev.kind != TAP and gap < M.post_time * 0.5:
                    # Previous note still down when this one is due: give up.
                    for n in a.notes:
                        n.skipped = True
                    continue
                if prev.kind == TAP and gap < 0.0:
                    for n in a.notes:
                        n.skipped = True
                    continue
            if a.kind == HOLD and len(a.notes) == 1 and a.notes[0].held is None:
                self._choose_pivot(a, actions[i + 1 :])
                a.notes[0].planned_hold = (a.side, a.x_start, a.width)
            up = self._rise(a.side, M.roll_raise)
            if prev is not None:
                dist = abs(a.x_start - prev.x_end)
                travel = max(M.min_travel_time, dist / M.travel_speed)
                if gap < M.pre_time + M.post_time:
                    # Too close for a full lift and hover: a single apex
                    # between the two touches, lower when the gap is tight
                    # but always high enough to register as a release.
                    frac = max(gap, 0.02) / (M.pre_time + M.post_time)
                    z = L.screen_z + max(0.015, M.hover_height * frac)
                    t_apex = prev.t_end + 0.5 * max(gap, 0.02)
                    x_apex = 0.5 * (
                        self._mid_x(a.x_start, a.side, a.width) + self._mid_x(prev.x_end, prev.ref, prev.width)
                    )
                    kfs.append(Keyframe(t_apex, x_apex, z, a.width, y, False))
                else:
                    t_lift = prev.t_end + M.post_time
                    kfs.append(Keyframe(t_lift, prev.x_end, self.hover_z, prev.width, y, False, prev.ref))
                    t_hover = a.t_hit - M.pre_time
                    t_leave = t_hover - travel
                    if t_leave > t_lift + 0.01:
                        kfs.append(Keyframe(t_leave, prev.x_end, self.hover_z, a.width, y, False, prev.ref))
                    kfs.append(Keyframe(t_hover, a.x_start, self.hover_z, a.width, y, False, a.side, up))
            else:
                kfs.append(
                    Keyframe(a.t_hit - M.pre_time, a.x_start, self.hover_z, a.width, y, False, a.side, up)
                )
            # Down phase: the fingertip crosses the contact height exactly at
            # the hit time, then presses on to touch_z.
            kfs.append(Keyframe(a.t_hit, a.x_start, L.screen_z + M.contact_tolerance, a.width, y, True, a.side, up))
            if a.kind == HOLD and len(a.notes) == 1:
                t_press = a.t_hit + min(M.press_time, a.t_end - a.t_hit)
                kfs.append(Keyframe(t_press, a.x_start, self.touch_z, a.width, y, True, a.side, up))
                hold, prev = _Hold(a.t_end, a.x_start, a.side, a.width, up, t_press), None
                continue
            for off, x in a.path:
                if off < M.press_time:
                    continue
                kfs.append(Keyframe(a.t_hit + off, x, self.touch_z, a.width, y, True))
            if a.kind == TAP or a.path[-1][0] < M.press_time:
                kfs.append(Keyframe(a.t_end, a.x_end, self.touch_z, a.width, y, True))
            prev = _Release(a.t_end, a.x_end, a.width, a.kind)
        if hold is not None:
            prev = self._release(hold, kfs)
        if prev is not None:
            kfs.append(Keyframe(prev.t_end + M.post_time, prev.x_end, self.hover_z, prev.width, y, False, prev.ref))
        return kfs

    # -- Policy --------------------------------------------------------------
    def plan(self, notes, t_now, X_WT_cmd, width_cmd) -> Plan:
        # Skips are re-decided on every replan for notes the hand has not
        # started descending toward: an early, wrong belief (say a previous
        # hold's duration overestimated) must not drop a note for good once
        # perception corrects it.  Notes already under way keep their
        # decision, so the hand never dives late at a note it gave up on.
        for n in notes:
            if n.hit_time - self.M.pre_time > t_now:
                n.skipped = False
        actions = self._actions(notes)
        kfs = self._keyframes(actions)
        for n in notes:
            if n.held is None and n.planned_hold is not None and n.hit_time <= t_now and not n.skipped:
                n.held = n.planned_hold
        # Keep only the future, and start from where the hand is commanded
        # to be right now so the new plan is continuous with the old one.
        p = X_WT_cmd.translation()
        x_axis = X_WT_cmd.rotation().matrix()[:, 0]
        rise = tip_gap(self.G, width_cmd) * float(np.sin(np.arctan2(x_axis[2], x_axis[0])))
        start = Keyframe(t_now, p[0], p[2], width_cmd, p[1], False, 0, rise)
        future = [k for k in kfs if k.t > t_now + 1e-3]
        out = [start]
        last_t = t_now
        for k in future:
            if k.t <= last_t + 1e-3:
                k.t = last_t + 1e-3
            out.append(k)
            last_t = k.t
        return Plan(out, self.G)
