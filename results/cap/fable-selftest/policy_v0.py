"""Rhythm-game policy: one hand, two fingertips, greedy-optimal scheduling.

Strategy
--------
* Every tick, the visible notes are turned into candidate *events*: a single
  note, or a pair of notes close in time (<= PAIR_WINDOW) and close in lane
  (<= max tip gap) that can be played by one landing with the gripper opened
  so that each fingertip sits in its own lane.
* A small dynamic programme picks the sequence of events that maximises the
  number of notes played, subject to the hand being able to travel (lift,
  move, change opening, descend) between consecutive events in time.
* Events whose descent has begun are *committed* (kept across ticks), so the
  plan does not flip-flop while landing; long notes being pressed stay pressed
  until they are safely past the required contact fraction (or end).
* Keyframes: lift only as high as the available time allows (fast re-taps
  lift just enough to register a fresh touch), travel at hover, wait over the
  lane, descend vertically, press, then follow slides along their waypoints.
"""

import math
from dataclasses import dataclass
from typing import Optional

from rhythmbot.cap.api import Keyframe, Plan, TAP, HOLD, SLIDE


# ---------------------------------------------------------------- tunables --
PAIR_WINDOW = 0.16      # notes closer than this in time may share one landing
DWELL = 0.03            # how long a tap stays pressed after the touch command (s)
LEAD = -0.025           # command reaches touch_z this long before the hit time
#   (negative: contact registers ~4 mm above the surface, i.e. before the command
#    arrives, so the command itself must come a little after the hit time)
LATE_SLACK = 0.08       # accept landing this late (judge window is +-140 ms)
EARLY_SLACK = 0.08      # ... or this early, when the next note is tight
GAP_RATE = 1.0          # m/s, opening change while travelling
H_MIN = 0.012           # minimum lift between touches (m above screen)
MIN_RETAP = 0.19        # measured: touches closer than this are not "fresh"
TRAVEL_SCALE = 1.1      # multiplier on the comfortable lateral speed (1.4 hit IK limits)
HOLD_FRAC = 0.88        # fraction of a long note's duration we keep contact
SLIDE_LEAD = 0.04       # anticipate the moving slide target by this much (s)
SAMPLE_DT = 0.10        # resample long-note trajectories at this interval
HORIZON_HOLD = 0.6      # how far ahead to command a hold whose end is unknown


@dataclass
class Event:
    notes: list                      # TrackedNote(s) played by this landing (1 or 2)
    T: float                         # nominal hit time of the landing
    first: int                       # index (time order) of its first note
    last: int                        # index of its last note
    keys: frozenset
    x: float = 0.0                   # midpoint x at landing
    gap: float = 0.0                 # fingertip gap at landing
    width: float = 0.0
    long: bool = False
    sides: dict = None               # note.key -> -1 (the -x fingertip) / +1
    fixed_gap: float = None          # single note that keeps a chord's opening
    # filled by the scheduler
    t_touch: float = 0.0
    mandatory: bool = False
    in_progress: bool = False


class Policy:
    def __init__(self, api):
        self.api = api
        lim = api.limits
        self.v = float(lim["travel_speed"]) * TRAVEL_SCALE
        self.hover_h = float(lim["hover_height"])
        self.v_down = self.hover_h / float(lim["descend_time"])
        self.v_up = self.hover_h / float(lim["lift_time"])
        self.screen_z = api.screen_z
        self.touch_z = api.touch_z
        self.max_gap = api.tip_gap(api.max_width)
        self.min_gap = api.tip_gap(0.0)
        self.lane_w = api.lane_width
        # committed landings: frozenset(note keys) -> {"t": t_touch, "gap", "sides", "fixed"}
        self.commit: dict = {}
        self.skipped_keys: set = set()
        self.last_touch = -math.inf     # t_touch of the latest committed landing

    # ------------------------------------------------------------ helpers --
    def t_desc(self, h: float) -> float:
        return max(h, 0.0) / self.v_down

    def t_lift(self, dz: float) -> float:
        return max(dz, 0.0) / self.v_up

    def width_for_gap(self, gap: float) -> float:
        if gap <= self.min_gap + 1e-6:
            return 0.0
        w = self.api.width_for_tip_gap(min(gap, self.max_gap))
        if w is None:
            return self.api.max_width
        return float(max(0.0, min(w, self.api.max_width)))

    @staticmethod
    def is_long(n) -> bool:
        return n.kind != TAP

    def note_x_at(self, n, t: float) -> float:
        """x of the fingertip playing note n at absolute time t."""
        if self.is_long(n):
            off = t - n.hit_time
            if off < 0.0:
                off = 0.0
            if off > n.duration:
                off = n.duration
            try:
                lane = n.lane_at_offset(off)
            except Exception:
                lane = n.lane
            return self.api.lane_x(lane)
        return self.api.lane_x(n.lane)

    def event_pose(self, e: Event, t: float):
        """(x_mid, gap) of the hand playing event e at absolute time t."""
        if len(e.notes) == 1:
            n = e.notes[0]
            x = self.note_x_at(n, t)
            if e.fixed_gap is not None:
                # the surviving half of a chord: keep the opening, follow the note
                side = e.sides.get(n.key, -1) if e.sides else -1
                return x - side * 0.5 * e.fixed_gap, e.fixed_gap
            return x, self.min_gap
        a, b = e.notes
        xa = self.note_x_at(a, t)
        xb = self.note_x_at(b, t)
        if e.sides is None:
            if xa > xb:
                xa, xb = xb, xa
        else:
            if e.sides.get(a.key, -1) > 0:
                a, b = b, a
                xa, xb = xb, xa
            la, lb = self.is_long(a), self.is_long(b)
            if la != lb and t > e.T + DWELL and e.gap > 0.0:
                # the tap partner is spent: keep the opening, follow the long note
                if la:
                    xb = xa + e.gap
                else:
                    xa = xb - e.gap
        gap = xb - xa
        if gap < self.min_gap:
            gap = self.min_gap
        if gap > self.max_gap:
            gap = self.max_gap
        return 0.5 * (xa + xb), gap

    def travel_time(self, x0, gap0, x1, gap1) -> float:
        return max(abs(x1 - x0) / self.v, abs(gap1 - gap0) / GAP_RATE)

    def rel_bounds(self, e: Event):
        """(earliest release, latest/natural release) for an event."""
        rel_min = e.t_touch + DWELL
        t_end = e.t_touch + DWELL
        for n in e.notes:
            if not self.is_long(n):
                continue
            if getattr(n, "duration_final", True):
                # contact must cover 80% of the duration, counted from when we landed
                start = max(n.hit_time, e.t_touch)
                rel_min = max(rel_min, start + HOLD_FRAC * n.duration + 0.02)
                t_end = max(t_end, n.end_time)
            else:
                rel_min = max(rel_min, math.inf)
                t_end = math.inf
        return rel_min, t_end

    # ------------------------------------------------------------- events --
    def build_events(self, notes, t_now: float):
        """Candidate landings from the visible notes, in time order."""
        live = []
        for n in notes:
            # a note whose time has passed stays relevant only if we committed to it
            if n.hit_time < t_now - LATE_SLACK:
                committed = any(n.key in keys for keys in self.commit)
                if not committed:
                    if n.key not in self.skipped_keys:
                        self.skipped_keys.add(n.key)
                        try:
                            n.skipped = True
                        except Exception:
                            pass
                    continue
            live.append(n)
        live.sort(key=lambda n: (n.hit_time, n.lane))
        events: list[Event] = []
        for i, n in enumerate(live):
            events.append(Event([n], n.hit_time, i, i, frozenset([n.key])))
            for j in range(i + 1, len(live)):
                m = live[j]
                if m.hit_time - n.hit_time > PAIR_WINDOW:
                    break
                if abs(m.lane - n.lane) * self.lane_w > self.max_gap + 1e-9:
                    continue
                if abs(m.lane - n.lane) < 0.5:
                    continue  # same lane: one fingertip each is impossible
                T = 0.5 * (n.hit_time + m.hit_time)
                events.append(Event([n, m], T, i, j, frozenset([n.key, m.key])))
        for e in events:
            e.long = any(self.is_long(n) for n in e.notes)
            rec = self.commit.get(e.keys)
            if rec is not None and rec.get("fixed"):
                e.fixed_gap = rec["gap"]
                e.sides = rec["sides"]
            elif len(e.notes) == 2:
                a, b = e.notes
                xa, xb = self.note_x_at(a, e.T), self.note_x_at(b, e.T)
                e.sides = {a.key: -1, b.key: 1} if xa <= xb else {a.key: 1, b.key: -1}
            e.x, e.gap = self.event_pose(e, e.T)
            e.width = self.width_for_gap(e.gap)
        events.sort(key=lambda e: (e.T, e.first, -len(e.notes)))
        return events

    def prune_commits(self, notes):
        """Drop finished commitments; when half a chord has been judged and
        vanished, carry the commitment (and the hand's opening) over to the
        surviving note."""
        alive = {n.key for n in notes}
        for keys in list(self.commit):
            rec = self.commit[keys]
            left = frozenset(k for k in keys if k in alive)
            if not left:
                del self.commit[keys]
            elif left != keys:
                del self.commit[keys]
                if len(keys) == 2 and rec.get("gap", 0.0) > self.min_gap + 1e-6:
                    rec = dict(rec, fixed=True)
                self.commit[left] = rec

    # ---------------------------------------------------------- scheduler --
    def schedule(self, events, t_now: float, k0: Keyframe):
        """Choose the sequence of events maximising notes played."""
        committed_keys = set()
        for keys in self.commit:
            committed_keys |= set(keys)

        # mark mandatory / in-progress and drop events that conflict with a commitment
        cand = []
        for e in events:
            if e.keys in self.commit:
                e.mandatory = True
                e.t_touch = self.commit[e.keys]["t"]
                tt = e.t_touch
                if tt + DWELL < t_now - 0.02 and not e.long:
                    # a finished tap landing; its commitment is stale
                    del self.commit[e.keys]
                    continue
                if tt <= t_now:
                    e.in_progress = True
                cand.append(e)
            elif e.keys & committed_keys:
                continue
            else:
                cand.append(e)
        events = cand
        n = len(events)
        if n == 0:
            return []

        # start state from the current commanded pose
        x0, z0, gap0 = k0.x, k0.z, self.api.tip_gap(k0.width)
        NEG = -1
        # best[i] = (count, cost, t_free, x_free, gap_free, prev, pre_shift, slack_before)
        #   cost: accumulated timing deviation (s) of all notes on the path
        #   pre_shift: how much earlier the predecessor must land for this transition
        #   slack_before: how much earlier this event itself could still land
        best = [None] * n
        mand_idx = [i for i, e in enumerate(events) if e.mandatory]
        lift_after = self.t_lift(H_MIN - (self.touch_z - self.screen_z))

        def end_state(e: Event):
            rel_min, _ = self.rel_bounds(e)
            if math.isinf(rel_min):
                return rel_min, e.x, e.gap
            xr, gr = self.event_pose(e, rel_min)
            return rel_min, xr, gr

        def dev(e: Event, tt: float) -> float:
            return sum(abs(tt - nn.hit_time) for nn in e.notes)

        for i, e in enumerate(events):
            # which mandatory events must precede i (all with index < i)
            last_mand = NEG
            for m in mand_idx:
                if m < i:
                    last_mand = m
            # transitions: (count, cost, t_touch, prev, pre_shift, slack_before)
            options = []
            if e.in_progress:
                # forced: hand is already down on it
                options.append((len(e.notes), 0.0, e.t_touch, NEG, 0.0, 0.0))
            else:
                if last_mand == NEG:
                    h0 = z0 - self.screen_z
                    t_up = self.t_lift(H_MIN - h0)
                    tr = self.travel_time(x0, gap0, e.x, e.gap)
                    # descend during travel if we are high, then the final approach
                    earliest = (t_now + t_up + max(tr, self.t_desc(h0 - H_MIN))
                                + self.t_desc(H_MIN) + LEAD)
                    earliest = max(earliest, self.last_touch + MIN_RETAP)
                    if e.mandatory:
                        options.append((len(e.notes), 0.0, e.t_touch, NEG, 0.0, 0.0))
                    else:
                        tt = max(e.T, earliest)
                        if tt <= e.T + LATE_SLACK:
                            options.append((len(e.notes), dev(e, tt), tt, NEG, 0.0,
                                            min(EARLY_SLACK, e.T - earliest)))
                for j in range(max(last_mand, 0), i):
                    if best[j] is None:
                        continue
                    ej = events[j]
                    if ej.last >= e.first:
                        continue
                    cj, costj, tfj, xfj, gfj, _, _, slack_j = best[j]
                    if math.isinf(tfj):
                        continue
                    tr = self.travel_time(xfj, gfj, e.x, e.gap)
                    earliest = tfj + lift_after + tr + self.t_desc(H_MIN) + LEAD
                    earliest = max(earliest, ej.t_touch + MIN_RETAP, tfj + MIN_RETAP - DWELL)
                    if e.mandatory:
                        options.append((cj + len(e.notes), costj, e.t_touch, j, 0.0, 0.0))
                        continue
                    # land the predecessor early if that is what it takes
                    shift = min(max(0.0, earliest - e.T), max(0.0, slack_j))
                    earliest -= shift
                    tt = max(e.T, earliest)
                    if tt <= e.T + LATE_SLACK:
                        slack = min(EARLY_SLACK, e.T - earliest) if shift <= 0.0 else 0.0
                        cost = costj + dev(e, tt) + shift * len(ej.notes)
                        options.append((cj + len(e.notes), cost, tt, j, shift, slack))
            if not options:
                continue
            # prefer more notes, then the smallest timing deviation
            options.sort(key=lambda o: (-o[0], o[1], o[2]))
            cnt, cost, tt, j, shift, slack = options[0]
            e.t_touch = tt
            tf, xf, gf = end_state(e)
            best[i] = (cnt, cost, tf, xf, gf, j, shift, slack)

        # pick the best terminal: most notes, least deviation, then earliest free time
        last_mand = mand_idx[-1] if mand_idx else NEG
        term = None
        for i in range(n):
            if best[i] is None or i < last_mand:
                continue
            key = (-best[i][0], best[i][1], best[i][2])
            if term is None or key < term[0]:
                term = (key, i)
        if term is None:
            return []
        path = []
        i = term[1]
        shift = 0.0
        while i != NEG:
            e = events[i]
            if shift > 0.0 and not e.mandatory:
                e.t_touch -= shift
            path.append(e)
            shift = best[i][6]
            i = best[i][5]
        path.reverse()
        return path

    # ---------------------------------------------------------- keyframes --
    def long_phase(self, e: Event, t_from: float, t_rel: float, kfs: list):
        """Keyframes keeping contact on e's notes from t_from to t_rel."""
        times = set()
        for n in e.notes:
            if not self.is_long(n):
                continue
            try:
                for off, _lane in n.waypoints:
                    tw = n.hit_time + off
                    if t_from < tw < t_rel:
                        times.add(tw)
            except Exception:
                pass
        t = t_from + SAMPLE_DT
        while t < t_rel:
            times.add(t)
            t += SAMPLE_DT
        times.add(t_rel)
        for t in sorted(times):
            x, gap = self.event_pose(e, t + SLIDE_LEAD)
            kfs.append(self.api.keyframe(t, x, self.touch_z, self.width_for_gap(gap), True))

    def release_time(self, e: Event, nxt: Optional[Event], t_now: float) -> float:
        rel_min, t_end = self.rel_bounds(e)
        if math.isinf(t_end):
            # end still off-screen: keep pressing well past what we know
            known = max(n.end_time for n in e.notes if self.is_long(n))
            return max(known, t_now + HORIZON_HOLD)
        if nxt is None:
            return t_end
        xr, gr = self.event_pose(e, rel_min)
        need = (self.t_lift(H_MIN - (self.touch_z - self.screen_z)) + self.travel_time(xr, gr, nxt.x, nxt.gap)
                + self.t_desc(H_MIN) + LEAD)
        need = max(need, MIN_RETAP - DWELL)
        latest = nxt.t_touch - need
        return min(t_end, max(rel_min, latest))

    def commit_event(self, e: Event):
        rec = self.commit.get(e.keys)
        if rec is None:
            rec = {"t": e.t_touch, "gap": e.gap, "sides": e.sides, "fixed": e.fixed_gap is not None}
            self.commit[e.keys] = rec
        self.last_touch = max(self.last_touch, e.t_touch)

    def build_keyframes(self, path, t_now: float, k0: Keyframe):
        kfs = [k0]
        x_prev, z_prev, gap_prev, t_rel_prev = k0.x, k0.z, self.api.tip_gap(k0.width), t_now
        w_prev = k0.width
        for idx, e in enumerate(path):
            nxt = path[idx + 1] if idx + 1 < len(path) else None
            if e.in_progress:
                t_rel = self.release_time(e, nxt, t_now)
                if t_rel > t_now:
                    self.long_phase(e, t_now, t_rel, kfs)
                xr, gr = self.event_pose(e, t_rel)
                x_prev, z_prev, gap_prev, w_prev, t_rel_prev = xr, self.touch_z, gr, self.width_for_gap(gr), t_rel
                continue
            from_start = idx == 0
            t_cmd = e.t_touch - LEAD
            tr = self.travel_time(x_prev, gap_prev, e.x, e.gap)
            over_lane = abs(e.x - x_prev) < 0.003 and abs(e.gap - gap_prev) < 0.003
            if from_start and e.mandatory and over_lane:
                # final approach already under way: just finish the descent
                remaining = max(0.0, z_prev - self.touch_z) / self.v_down
                if t_cmd < t_now + 0.5 * remaining:
                    t_cmd = t_now + 0.5 * remaining
                t_desc_start = t_cmd - remaining
                if t_desc_start > t_now + 1e-3:
                    kfs.append(self.api.keyframe(t_desc_start, e.x, z_prev, e.width))
                kfs.append(self.api.keyframe(t_cmd, e.x, self.touch_z, e.width, True))
                self.commit_event(e)
            else:
                avail = t_cmd - t_rel_prev - tr
                # lift height that fits the available time (from z_prev up, then down)
                denom = 1.0 / self.v_up + 1.0 / self.v_down
                h = (avail + max(0.0, z_prev - self.screen_z) / self.v_up) / denom
                h = max(H_MIN, min(self.hover_h, h))
                if from_start and e.mandatory:
                    # descent begun: never command a lift again
                    h = min(h, max(z_prev - self.screen_z, 0.0))
                z_h = self.screen_z + h
                t_up = t_rel_prev + self.t_lift(z_h - z_prev)
                t_arr = t_up + tr
                t_desc_start = t_cmd - self.t_desc(h)
                # keep times monotone if the schedule is tighter than comfortable
                t_up = max(t_up, t_rel_prev + 1e-3)
                t_arr = max(t_arr, t_up + 1e-3)
                if t_cmd < t_arr + 1e-3:
                    t_cmd = t_arr + 1e-3
                if t_up < t_cmd - 1e-3 and z_h > z_prev + 1e-4:
                    kfs.append(self.api.keyframe(t_up, x_prev, z_h, w_prev))
                if t_arr < t_cmd - 1e-3 and (tr > 1e-6 or z_h < z_prev - 1e-4 or not from_start):
                    kfs.append(self.api.keyframe(t_arr, e.x, z_h, e.width))
                if t_desc_start > t_arr + 1e-3 and t_desc_start < t_cmd - 1e-3:
                    kfs.append(self.api.keyframe(t_desc_start, e.x, z_h, e.width))
                kfs.append(self.api.keyframe(t_cmd, e.x, self.touch_z, e.width, True))
                # commit once the descent has (nearly) begun
                if t_desc_start <= t_now + 0.05:
                    self.commit_event(e)
            t_rel = self.release_time(e, nxt, t_now)
            t_rel = max(t_rel, t_cmd + 1e-3)
            self.long_phase(e, t_cmd, t_rel, kfs)
            xr, gr = self.event_pose(e, t_rel)
            x_prev, z_prev, gap_prev, w_prev, t_rel_prev = xr, self.touch_z, gr, self.width_for_gap(gr), t_rel
        # finally lift to hover and rest
        t_up = t_rel_prev + self.t_lift(self.screen_z + self.hover_h - z_prev)
        kfs.append(self.api.keyframe(max(t_up, t_rel_prev + 1e-3), x_prev, self.screen_z + self.hover_h, w_prev))
        return kfs

    # --------------------------------------------------------------- plan --
    def plan(self, notes, t_now, X_WT_cmd, width_cmd) -> Plan:
        k0 = self.api.current_keyframe(t_now, X_WT_cmd, width_cmd)
        self.prune_commits(notes)
        events = self.build_events(notes, t_now)
        path = self.schedule(events, t_now, k0)
        if not path:
            hover = self.api.keyframe(t_now + 0.2, k0.x, self.screen_z + self.hover_h, k0.width)
            return self.api.make_plan([k0, hover], t_now)
        kfs = self.build_keyframes(path, t_now, k0)
        return self.api.make_plan(kfs, t_now)


def make_policy(api):
    return Policy(api)
