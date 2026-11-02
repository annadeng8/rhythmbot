"""Rhythm-game policy: one hand, two fingertips, greedy-optimal scheduling.

Strategy
--------
* Every tick, the visible notes are turned into candidate *events*: a single
  note, or a pair of notes close in time (<= PAIR_WINDOW) and close in lane
  (<= max tip gap) that can be played by one landing with the gripper opened
  so that each fingertip sits in its own lane.
* A small dynamic programme picks the sequence of events that maximises the
  number of notes played, subject to the hand being able to get there in time.
  It tracks an explicit two-fingertip hand state, so between events it may
  - lift the whole hand, travel and land again ("air" landing), or
  - keep one fingertip pressed (on a hold/slide, or on a spent tap that is
    simply dragged along) and roll the other fingertip up, move it and roll
    it down on the next note ("drum" landing), which avoids the measured
    ~0.19 s minimum between plain re-touches.
* "Rocking" air landings for two notes 0.07-0.5 s apart (same lane or up to
  2.6 lanes apart): the hand lands rolled so only the first fingertip touches,
  then rolls flat so the second fingertip lands on the later note.
* Events whose descent / roll-down has begun are *committed* (kept across
  ticks) so the plan does not flip-flop while landing; long notes stay pressed
  until safely past the required contact fraction (or their end).
* Keyframes: lift only as high as the available time allows, travel at hover,
  wait over the lane, descend vertically, press; while pressed, the two
  fingertip trajectories are sampled and turned into (ref, rise, opening)
  keyframes.
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
GAP_RATE_PRESSED = 0.4  # m/s, opening change while a fingertip holds a note
H_MIN = 0.012           # minimum lift between touches (m above screen)
MIN_RETAP = 0.19        # measured: whole-hand re-touches closer than this are not "fresh"
TRAVEL_SCALE = 1.1      # multiplier on the comfortable lateral speed (1.4 hit IK limits)
HOLD_FRAC = 0.88        # fraction of a long note's duration we keep contact
SLIDE_LEAD = 0.04       # anticipate the moving slide target by this much (s)
SAMPLE_DT = 0.10        # resample pressed-phase trajectories at this interval
HORIZON_HOLD = 0.6      # how far ahead to command a hold whose end is unknown
# rocking / drumming with one fingertip pressed
ROCK_MIN_DT = 0.07      # second note at least this much later than the first
ROCK_MAX_DT = 0.50      # ... and at most this much
ROCK_RISE = 0.010       # lifted fingertip height above the pressed one (m)
ROLL_T = 0.08           # time to roll a fingertip up or down (s)
SAME_LANE_GAP = 0.040   # fingertip gap when both notes are in one lane
ROCK_COST = 0.05        # scheduling cost (s) so plain landings win when possible
MIN_DRUM = 0.20         # min time between two touches of the same rolled fingertip
DRUM_COST = 0.05        # scheduling cost (s) so plain landings win when possible
DRUM_LEAD = -0.01       # roll-down ends this long before the hit time (measured lag)
DRUM_GAP_MIN = 0.040    # smallest opening used while drumming (roll angle)


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
    xoff: dict = None                # note.key -> x offset within its lane
    fixed_gap: float = None          # single note that keeps a chord's opening
    rock: bool = False               # second fingertip lands `delta` later
    delta: float = 0.0
    ref_side: int = 0                # side of the fingertip that lands first
    rise: float = 0.0                # signed rise while the second one waits
    second_key: object = None
    # filled by the scheduler
    t_touch: float = 0.0
    mandatory: bool = False
    in_progress: bool = False
    mode: str = "air"                # "air": land from above; "drum": roll one tip down
    tip: int = 0                     # drum: side of the landing fingertip
    anchor_fixed: bool = False       # drum: other fingertip stays on its note
    t_free_tip: float = 0.0          # drum: when the landing fingertip may leave its note
    x_from: float = 0.0              # drum: landing fingertip x before the move
    x_land: float = 0.0              # drum: landing fingertip x at touch
    xa0: float = 0.0                 # drum, dragged anchor: anchor x before / after
    xa1: float = 0.0
    t_move: float = 0.0              # drum: when the lateral move may start
    t_move_end: float = 0.0          # drum: when it is over (moving at full speed)
    state: object = None             # HandState after this event

    def t_last(self) -> float:
        """Time of this landing's last fresh touch."""
        return self.t_touch + (self.delta if self.rock else 0.0)


class Tip:
    """What one fingertip is doing after a landing."""
    __slots__ = ("note", "x", "xoff", "t_touch", "t_free", "end", "lift_frac", "follow", "x_now")

    def __init__(self, note, x, t_touch, t_free, end, xoff=0.0, lift_frac=0.0, follow=None):
        self.note, self.x, self.xoff = note, x, xoff
        self.t_touch, self.t_free, self.end = t_touch, t_free, end
        self.lift_frac = lift_frac       # how far it is already rolled up (0..1)
        self.follow = follow             # (long note, x offset): dragged along beside it
        self.x_now = None                # actual commanded x right now (in-progress states)


class HandState:
    __slots__ = ("tips",)

    def __init__(self, tips):
        self.tips = tips

    def t_free(self) -> float:
        return max(t.t_free for t in self.tips.values())

    def t_end(self) -> float:
        return max(t.end for t in self.tips.values())


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
        self.x_max = max(abs(api.lane_x(0)), abs(api.lane_x(api.n_lanes - 1))) + 0.5 * api.lane_width
        # committed landings: frozenset(note keys) -> record dict
        self.commit: dict = {}
        self.skipped_keys: set = set()
        # last fresh touch (time, x) of each fingertip, survives vanished notes
        self.tip_mem = {-1: (-math.inf, 0.0), 1: (-math.inf, 0.0)}

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
        return n is not None and n.kind != TAP

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

    def tip_x(self, tip: Tip, t: float) -> float:
        if self.is_long(tip.note):
            return self.note_x_at(tip.note, t) + tip.xoff
        if tip.follow is not None:
            return self.note_x_at(tip.follow[0], t) + tip.follow[1]
        return tip.x

    def tip_bounds(self, n, landed: float):
        """(earliest release, natural end) of a fingertip that landed on n."""
        if n is None or not self.is_long(n):
            return landed + DWELL, landed + DWELL
        if getattr(n, "duration_final", True):
            start = max(n.hit_time, landed)
            return start + HOLD_FRAC * n.duration + 0.02, max(n.end_time, landed + DWELL)
        return math.inf, math.inf

    def event_pose(self, e: Event, t: float):
        """(x_mid, gap) of the hand playing event e at absolute time t."""
        xoff = e.xoff or {}
        if len(e.notes) == 1:
            n = e.notes[0]
            x = self.note_x_at(n, t) + xoff.get(n.key, 0.0)
            if e.fixed_gap is not None:
                # the surviving half of a chord: keep the opening, follow the note
                side = e.sides.get(n.key, -1) if e.sides else -1
                return x - side * 0.5 * e.fixed_gap, e.fixed_gap
            return x, self.min_gap
        a, b = e.notes
        xa = self.note_x_at(a, t) + xoff.get(a.key, 0.0)
        xb = self.note_x_at(b, t) + xoff.get(b.key, 0.0)
        if e.sides is None:
            if xa > xb:
                xa, xb = xb, xa
        else:
            if e.sides.get(a.key, -1) > 0:
                a, b = b, a
                xa, xb = xb, xa
            la, lb = self.is_long(a), self.is_long(b)
            if la != lb and t > e.T + DWELL + (e.delta if e.rock else 0.0) and e.gap > 0.0:
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

    # --------------------------------------------------------- hand state --
    def state_after_air(self, e: Event) -> HandState:
        """Both fingertips after an air landing of e at e.t_touch."""
        tt = e.t_touch
        tips = {}
        if len(e.notes) == 1:
            n = e.notes[0]
            if e.fixed_gap is not None:
                side = e.sides.get(n.key, -1) if e.sides else -1
                xn = self.note_x_at(n, tt)
                tf, te = self.tip_bounds(n, tt)
                tips[side] = Tip(n, xn, tt, tf, te)
                tips[-side] = Tip(None, xn - side * e.fixed_gap, tt, tt + DWELL, tt + DWELL)
                if e.rock and e.second_key == n.key:
                    # rolling onto the surviving second note
                    tb = tt + e.delta
                    tf, te = self.tip_bounds(n, tb)
                    tips[side] = Tip(n, xn, tb, tf, te)
            else:
                xn = self.note_x_at(n, tt)
                tf, te = self.tip_bounds(n, tt)
                tips[-1] = Tip(n, xn, tt, tf, te)
                tips[1] = Tip(n, xn, tt, tf, te)
            return HandState(tips)
        xoff = e.xoff or {}
        for n in e.notes:
            side = e.sides[n.key]
            landed = tt + e.delta if (e.rock and n.key == e.second_key) else tt
            tf, te = self.tip_bounds(n, landed)
            tips[side] = Tip(n, self.note_x_at(n, landed) + xoff.get(n.key, 0.0), landed, tf, te,
                             xoff.get(n.key, 0.0))
        return HandState(tips)

    def state_after_drum(self, e: Event, prev: HandState) -> HandState:
        n = e.notes[0]
        s = e.tip
        tf, te = self.tip_bounds(n, e.t_touch)
        tips = dict(prev.tips)
        tips[s] = Tip(n, e.x_land, e.t_touch, tf, te)
        anc = prev.tips[-s]
        if not e.anchor_fixed:
            follow = (n, e.xa1 - e.x_land) if self.is_long(n) else None
            tips[-s] = Tip(None, e.xa1, anc.t_touch, max(anc.t_free, e.t_touch + DWELL),
                           max(anc.end, e.t_touch + DWELL), follow=follow)
        return HandState(tips)

    def free_pose(self, st: HandState):
        """(t, x_mid, gap) when the hand lifts off from state st."""
        t = st.t_free()
        if math.isinf(t):
            return t, 0.0, self.min_gap
        xm, xp = self.tip_x(st.tips[-1], t), self.tip_x(st.tips[1], t)
        gap = max(self.min_gap, min(self.max_gap, xp - xm))
        return t, 0.5 * (xm + xp), gap

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
        events: list = []
        for i, n in enumerate(live):
            events.append(Event([n], n.hit_time, i, i, frozenset([n.key])))
            for j in range(i + 1, len(live)):
                m = live[j]
                dt = m.hit_time - n.hit_time
                if dt > max(PAIR_WINDOW, ROCK_MAX_DT):
                    break
                dl = abs(m.lane - n.lane)
                if dl * self.lane_w > self.max_gap + 1e-9:
                    continue
                keys = frozenset([n.key, m.key])
                if dt <= PAIR_WINDOW and dl >= 0.5:
                    # both fingertips land together, one lane each
                    T = 0.5 * (n.hit_time + m.hit_time)
                    events.append(Event([n, m], T, i, j, keys))
                if ROCK_MIN_DT <= dt <= ROCK_MAX_DT:
                    # first fingertip lands on n, the hand rolls flat onto m
                    e = Event([n, m], n.hit_time, i, j, keys, rock=True, delta=dt, second_key=m.key)
                    if dl < 0.5:
                        e.xoff = {n.key: -0.5 * SAME_LANE_GAP, m.key: 0.5 * SAME_LANE_GAP}
                    events.append(e)
        for e in events:
            e.long = any(self.is_long(n) for n in e.notes)
            rec = self.commit.get(e.keys)
            if rec is not None and rec.get("fixed"):
                e.fixed_gap = rec["gap"]
                e.sides = rec["sides"]
                e.xoff = rec.get("xoff")
                if rec.get("rock") and len(e.notes) == 1 and e.notes[0].key == rec.get("second_key"):
                    # the first note of a rocking landing was judged and vanished
                    # before the second landed: keep rolling onto the second
                    e.rock, e.delta, e.second_key = True, rec["delta"], rec["second_key"]
                    e.ref_side, e.rise = rec["ref_side"], rec["rise"]
                    e.T = rec["t"]
            elif len(e.notes) == 2:
                a, b = e.notes
                xoff = e.xoff or {}
                xa = self.note_x_at(a, e.T) + xoff.get(a.key, 0.0)
                xb = self.note_x_at(b, e.T) + xoff.get(b.key, 0.0)
                e.sides = {a.key: -1, b.key: 1} if xa <= xb else {a.key: 1, b.key: -1}
                if e.rock:
                    e.ref_side = e.sides[a.key]
                    e.rise = ROCK_RISE * e.sides[b.key]
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
    def start_state(self, k0: Keyframe, t_now: float):
        """Hand state implied by the current command (used when no committed
        event describes what the fingertips are pressing)."""
        pm, pp = self.api.fingertips(k0)
        tips, pressed = {}, {}
        for s, p in ((-1, pm), (1, pp)):
            tm, _ = self.tip_mem[s]
            t_free = max(tm + DWELL, t_now)
            frac = min(1.0, max(0.0, (float(p[2]) - self.touch_z) / ROCK_RISE))
            tips[s] = Tip(None, float(p[0]), tm, t_free, t_free, lift_frac=frac)
            tips[s].x_now = float(p[0])
            pressed[s] = float(p[2]) <= self.touch_z + 0.004
        return HandState(tips), pressed

    def sync_state(self, st: HandState, k0: Keyframe):
        """Attach the actual commanded fingertip positions / lift to a state
        describing what the hand is doing right now."""
        pm, pp = self.api.fingertips(k0)
        for s, p in ((-1, pm), (1, pp)):
            tip = st.tips[s]
            tip.x_now = float(p[0])
            tip.lift_frac = min(1.0, max(0.0, (float(p[2]) - self.touch_z) / ROCK_RISE))
            if not self.is_long(tip.note) and tip.follow is None:
                tip.x = tip.x_now

    def drum_option(self, st: HandState, e: Event, s: int, t_min: float):
        """Land e's single note with fingertip s while -s stays pressed."""
        if len(e.notes) != 1 or e.rock or e.fixed_gap is not None:
            return None
        n = e.notes[0]
        tip, anc = st.tips[s], st.tips[-s]
        t_free = max(tip.t_free, t_min)
        if math.isinf(t_free):
            return None
        # from an in-progress state, continue from where the fingertips really are
        live = tip.x_now is not None and t_min > -math.inf
        x_from = tip.x_now if live else self.tip_x(tip, t_free)
        x_land = self.note_x_at(n, e.T)
        fixed = self.is_long(anc.note) and e.T < anc.t_free
        # the lateral move / opening runs concurrently with the roll-up (a spent
        # fingertip may be dragged a little), so a closed hand opens before it
        # has to roll far
        t_move = t_free
        t_rolled = t_free + ROLL_T * (1.0 - tip.lift_frac)
        if fixed:
            xa = self.tip_x(anc, e.T)
            g = (x_land - xa) * s
            if g < DRUM_GAP_MIN - 1e-9 or g > self.max_gap + 1e-9:
                return None
            g0 = abs(x_from - (anc.x_now if live else self.tip_x(anc, t_free)))
            travel = max(abs(x_land - x_from) / self.v, abs(g - g0) / GAP_RATE_PRESSED)
            xa0 = xa1 = xa
        else:
            if self.is_long(anc.note):
                t_move = max(t_move, anc.t_free)   # do not drag a hold before it may end
                if math.isinf(t_move):
                    return None
            if live and (anc.x_now is not None) and (t_move <= t_min + 1e-9 or not self.is_long(anc.note)):
                xa0 = anc.x_now
            else:
                xa0 = self.tip_x(anc, t_move)
            g0 = abs(x_from - xa0)
            g = min(max(g0, DRUM_GAP_MIN), self.max_gap, self.x_max + s * x_land)  # stay on the screen
            if g < DRUM_GAP_MIN - 1e-9:
                return None
            xa1 = x_land - s * g
            travel = max(abs(x_land - x_from) / self.v, abs(g - g0) / GAP_RATE)
        earliest = max(max(t_move + travel, t_rolled) + ROLL_T + DRUM_LEAD, tip.t_touch + MIN_DRUM, t_min)
        tt = max(e.T, earliest)
        if tt > e.T + LATE_SLACK:
            return None
        # prefer the fingertip that has rested longer (alternate when possible)
        extra = 0.01 if tip.t_touch > anc.t_touch else 0.0
        return dict(tt=tt, tip=s, afixed=fixed, t_free_tip=t_free, x_from=x_from,
                    x_land=self.note_x_at(n, tt), xa0=xa0, xa1=xa1, t_move=t_move,
                    t_move_end=t_move + travel, extra=extra)

    @staticmethod
    def apply_drum(e: Event, p: dict):
        e.mode = "drum"
        e.tip, e.anchor_fixed, e.t_free_tip = p["tip"], p["afixed"], p["t_free_tip"]
        e.x_from, e.x_land, e.xa0, e.xa1 = p["x_from"], p["x_land"], p["xa0"], p["xa1"]
        e.t_move, e.t_move_end = p["t_move"], p["t_move_end"]

    def apply_memory(self, st: HandState, e: Event):
        """A fingertip that touched a (now vanished) note after this landing
        is remembered as pressed there."""
        for s in (-1, 1):
            tm, xm = self.tip_mem[s]
            if tm > e.t_last() + 1e-6:
                st.tips[s] = Tip(None, xm, tm, tm + DWELL, tm + DWELL)

    def schedule(self, events, t_now: float, k0: Keyframe):
        """Choose the sequence of events maximising notes played."""
        committed_keys = set()
        for keys in self.commit:
            committed_keys |= set(keys)

        # mark mandatory / in-progress and drop events that conflict with a commitment
        cand = []
        for e in events:
            rec = self.commit.get(e.keys)
            if rec is not None:
                e.mandatory = True
                e.t_touch = rec["t"]
                if rec.get("mode") == "drum":
                    self.apply_drum(e, rec)
                if e.t_last() + DWELL < t_now - 0.02 and not e.long:
                    # a finished tap landing; its commitment is stale
                    del self.commit[e.keys]
                    continue
                if e.t_touch <= t_now:
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

        st0, pressed0 = self.start_state(k0, t_now)
        x0, z0, gap0 = k0.x, k0.z, self.api.tip_gap(k0.width)
        NEG = -1
        # best[i] = (count, cost, state, prev, pre_shift, slack_before)
        best = [None] * n
        mand_idx = [i for i, e in enumerate(events) if e.mandatory]
        lift_after = self.t_lift(H_MIN - (self.touch_z - self.screen_z))
        fresh_mem = min(self.tip_mem[-1][0], self.tip_mem[1][0]) + MIN_RETAP

        def dev(e: Event, tt: float) -> float:
            if e.rock:
                return ROCK_COST + sum(abs(tt + (e.delta if nn.key == e.second_key else 0.0) - nn.hit_time)
                                       for nn in e.notes)
            return sum(abs(tt - nn.hit_time) for nn in e.notes)

        def air_fresh(st: HandState, e: Event) -> float:
            ts = [t.t_touch for t in st.tips.values()]
            return (max(ts) if len(e.notes) == 2 else min(ts)) + MIN_RETAP

        for i, e in enumerate(events):
            last_mand = NEG
            for m in mand_idx:
                if m < i:
                    last_mand = m
            # options: (count, cost, t_touch, prev, pre_shift, slack_before, mode, params)
            options = []
            if e.mandatory:
                j = last_mand
                if j != NEG and best[j] is None:
                    continue
                cj, costj = (best[j][0], best[j][1]) if j != NEG else (0, 0.0)
                options.append((cj + len(e.notes), costj, e.t_touch, j, 0.0, 0.0, e.mode, None))
            else:
                if last_mand == NEG:
                    h0 = z0 - self.screen_z
                    t_up = self.t_lift(H_MIN - h0)
                    tr = self.travel_time(x0, gap0, e.x, e.gap)
                    # descend during travel if we are high, then the final approach
                    earliest = (t_now + t_up + max(tr, self.t_desc(h0 - H_MIN))
                                + self.t_desc(H_MIN) + LEAD)
                    earliest = max(earliest, fresh_mem)
                    tt = max(e.T, earliest)
                    if tt <= e.T + LATE_SLACK:
                        options.append((len(e.notes), dev(e, tt), tt, NEG, 0.0,
                                        min(EARLY_SLACK, e.T - earliest), "air", None))
                    for s in (-1, 1):
                        if pressed0[-s]:
                            p = self.drum_option(st0, e, s, t_now)
                            if p is not None:
                                options.append((len(e.notes), dev(e, p["tt"]) + DRUM_COST + p["extra"], p["tt"], NEG,
                                                0.0, 0.0, "drum", p))
                for j in range(max(last_mand, 0), i):
                    if best[j] is None:
                        continue
                    ej = events[j]
                    if ej.last >= e.first:
                        continue
                    cj, costj, stj, _, _, slack_j = best[j]
                    tfj, xfj, gfj = self.free_pose(stj)
                    if not math.isinf(tfj):
                        tr = self.travel_time(xfj, gfj, e.x, e.gap)
                        earliest = tfj + lift_after + tr + self.t_desc(H_MIN) + LEAD
                        earliest = max(earliest, air_fresh(stj, e), tfj + MIN_RETAP - DWELL)
                        # land the predecessor early if that is what it takes
                        shift = min(max(0.0, earliest - e.T), max(0.0, slack_j))
                        earliest -= shift
                        tt = max(e.T, earliest)
                        if tt <= e.T + LATE_SLACK:
                            slack = min(EARLY_SLACK, e.T - earliest) if shift <= 0.0 else 0.0
                            cost = costj + dev(e, tt) + shift * len(ej.notes)
                            options.append((cj + len(e.notes), cost, tt, j, shift, slack, "air", None))
                    for s in (-1, 1):
                        p = self.drum_option(stj, e, s, t_now if ej.in_progress else -math.inf)
                        if p is not None:
                            options.append((cj + len(e.notes), costj + dev(e, p["tt"]) + DRUM_COST + p["extra"],
                                            p["tt"], j, 0.0, 0.0, "drum", p))
            if not options:
                continue
            # prefer more notes, then the smallest timing deviation
            options.sort(key=lambda o: (-o[0], o[1], o[2]))
            cnt, cost, tt, j, shift, slack, mode, p = options[0]
            e.t_touch = tt
            if p is not None:
                self.apply_drum(e, p)
            if mode == "drum":
                prev_st = best[j][2] if j != NEG else st0
                st = self.state_after_drum(e, prev_st)
            else:
                e.mode = "air"
                st = self.state_after_air(e)
                if e.in_progress:
                    self.apply_memory(st, e)
            if e.in_progress:
                self.sync_state(st, k0)
            e.state = st
            best[i] = (cnt, cost, st, j, shift, slack)

        # pick the best terminal: most notes, least deviation, then earliest free time
        last_mand = mand_idx[-1] if mand_idx else NEG
        term = None
        for i in range(n):
            if best[i] is None or i < last_mand:
                continue
            key = (-best[i][0], best[i][1], best[i][2].t_free())
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
            shift = best[i][4]
            i = best[i][3]
        path.reverse()
        return path

    # ---------------------------------------------------------- keyframes --
    def rock_params(self, e: Event, t: float):
        """(ref, rise) of event e's approach keyframe at time t: a rocking
        landing arrives rolled so that only the first fingertip touches."""
        if not e.rock or e.ref_side == 0:
            return 0, 0.0
        t_b = e.t_touch + e.delta - LEAD
        if t >= t_b:
            return 0, 0.0
        if t <= t_b - ROLL_T:
            return e.ref_side, e.rise
        return e.ref_side, e.rise * (t_b - t) / ROLL_T

    def make_kf(self, e: Event, t: float, x_mid: float, gap: float, z: float, down: bool = False):
        ref, rise = self.rock_params(e, t)
        x = x_mid + ref * 0.5 * gap if ref else x_mid
        return self.api.keyframe(t, x, z, self.width_for_gap(gap), down, ref, rise)

    def release_time(self, st: HandState, nxt: Optional[Event], t_now: float) -> float:
        t_free, xr, gr = self.free_pose(st)
        t_end = st.t_end()
        if math.isinf(t_end):
            # end still off-screen: keep pressing well past what we know
            known = max((t.note.end_time for t in st.tips.values() if self.is_long(t.note)), default=t_now)
            return max(known, t_now + HORIZON_HOLD)
        if nxt is None:
            return t_end
        need = (self.t_lift(H_MIN - (self.touch_z - self.screen_z)) + self.travel_time(xr, gr, nxt.x, nxt.gap)
                + self.t_desc(H_MIN) + LEAD)
        need = max(need, MIN_RETAP - DWELL)
        return min(t_end, max(t_free, nxt.t_touch - need))

    def commit_event(self, e: Event):
        if e.keys in self.commit:
            return
        rec = {"t": e.t_touch, "gap": e.gap, "sides": e.sides, "xoff": e.xoff,
               "fixed": e.fixed_gap is not None, "rock": e.rock, "delta": e.delta,
               "second_key": e.second_key, "ref_side": e.ref_side, "rise": e.rise, "mode": e.mode}
        if e.mode == "drum":
            rec.update(tip=e.tip, afixed=e.anchor_fixed, t_free_tip=e.t_free_tip, x_from=e.x_from,
                       x_land=e.x_land, xa0=e.xa0, xa1=e.xa1, t_move=e.t_move, t_move_end=e.t_move_end)
            self.tip_mem[e.tip] = (e.t_touch, e.x_land)
            if not e.anchor_fixed:
                self.tip_mem[-e.tip] = (self.tip_mem[-e.tip][0], e.xa1)
        elif e.state is not None:
            for s, tip in e.state.tips.items():
                if tip.t_touch > self.tip_mem[s][0]:
                    self.tip_mem[s] = (tip.t_touch, tip.x)
        self.commit[e.keys] = rec

    def pressed_phase(self, chain, t_from: float, t_rel: float, k0: Keyframe, kfs: list):
        """Keyframes while at least one fingertip is pressed: the two
        fingertip trajectories (press / roll up, move, roll down / drag)
        sampled and expressed as reference-tip keyframes."""
        head = chain[0]
        phases = {-1: [], 1: []}
        lifted = {-1: 0.0, 1: 0.0}          # current roll-up fraction of each fingertip
        r = ROCK_RISE
        pm, pp = self.api.fingertips(k0)
        if head.mode == "drum":
            # chain continues from whatever the hand is doing now
            for s, p in ((-1, pm), (1, pp)):
                phases[s].append((-math.inf, "press", (None, float(p[0]), 0.0)))
                lifted[s] = min(1.0, max(0.0, (float(p[2]) - self.touch_z) / r))
            drums = chain
        else:
            t0 = head.t_touch - LEAD
            for s, tip in head.state.tips.items():
                if tip.follow is not None:
                    phases[s].append((t0, "follow", tip.follow))
                else:
                    phases[s].append((t0, "press", (tip.note, tip.x, tip.xoff)))
                if head.in_progress:
                    lifted[s] = tip.lift_frac
            if head.rock:
                sb = -head.ref_side
                tb = t0 + head.delta
                tipb = head.state.tips[sb]
                phases[sb] = [(t0, "lift", dict(pre=1.0, a=t0, b=tb, x0=tipb.x, x1=tipb.x, t_move=t0)),
                              (tb, "press", (tipb.note, tipb.x, tipb.xoff))]
            drums = chain[1:]
        for d in drums:
            s = d.tip
            a, b = d.t_free_tip, d.t_touch - DRUM_LEAD
            ma, mb = d.t_move, min(max(d.t_move_end, d.t_move + 1e-3), max(d.t_move + 1e-3, b - ROLL_T))
            # only the roll-up that is under way right now starts from the actual lift
            pre = lifted[s] if a <= t_from < b else 0.0
            phases[s].append((a, "lift", dict(pre=pre, a=a, b=b, x0=d.x_from, x1=d.x_land,
                                              t_move=ma, t_move_end=mb)))
            phases[s].append((b, "press", (d.notes[0], d.x_land, 0.0)))
            if not d.anchor_fixed:
                phases[-s].append((ma, "drag", dict(a=ma, b=mb, x0=d.xa0, x1=d.xa1)))
                if self.is_long(d.notes[0]):
                    phases[-s].append((mb, "follow", (d.notes[0], d.xa1 - d.x_land)))
                else:
                    phases[-s].append((mb, "press", (None, d.xa1, 0.0)))

        def tip_at(s: int, t: float):
            ph = None
            for cand in phases[s]:
                if cand[0] <= t + 1e-9:
                    ph = cand
            if ph is None:
                ph = phases[s][0]
            kind, data = ph[1], ph[2]
            if kind == "press":
                note, x, xoff = data
                if self.is_long(note):
                    x = self.note_x_at(note, t + SLIDE_LEAD) + xoff
                return x, self.touch_z
            if kind == "follow":
                return self.note_x_at(data[0], t + SLIDE_LEAD) + data[1], self.touch_z
            if kind == "drag":
                a, b = data["a"], data["b"]
                f = 1.0 if b <= a else min(1.0, max(0.0, (t - a) / (b - a)))
                return data["x0"] + f * (data["x1"] - data["x0"]), self.touch_z
            a, b = data["a"], data["b"]
            f0 = float(data["pre"])
            up_end, down_start = a + ROLL_T * (1.0 - f0), b - ROLL_T
            if down_start < up_end:
                up_end = down_start = 0.5 * (a + b)
            if t >= up_end:
                z = r if t < down_start else r * max(0.0, (b - t) / max(b - down_start, 1e-6))
            else:
                z = r * min(1.0, f0 + (1.0 - f0) * (t - a) / max(up_end - a, 1e-6))
            ma, mb = data["t_move"], data.get("t_move_end", down_start)
            f = 1.0 if mb <= ma else min(1.0, max(0.0, (t - ma) / (mb - ma)))
            return data["x0"] + f * (data["x1"] - data["x0"]), self.touch_z + z

        times = set()
        for s in (-1, 1):
            for ph in phases[s]:
                if ph[1] in ("press", "follow"):
                    times.add(ph[0])
                    note = ph[2][0]
                    if self.is_long(note):
                        try:
                            for off, _lane in note.waypoints:
                                times.add(note.hit_time + off)
                        except Exception:
                            pass
                else:
                    a, b = ph[2]["a"], ph[2]["b"]
                    times.update((a, a + ROLL_T, b - ROLL_T, b, 0.5 * (a + b)))
                    if "t_move" in ph[2]:
                        times.update((ph[2]["t_move"], ph[2]["t_move_end"]))
        t = t_from + SAMPLE_DT
        while t < t_rel:
            times.add(t)
            t += SAMPLE_DT
        times.add(t_rel)
        for t in sorted(tt for tt in times if t_from < tt <= t_rel):
            xm, zm = tip_at(-1, t)
            xp, zp = tip_at(1, t)
            gap = max(self.min_gap, min(self.max_gap, xp - xm))
            w = self.width_for_gap(gap)
            if abs(zp - zm) < 1e-6:
                kfs.append(self.api.keyframe(t, 0.5 * (xm + xp), self.touch_z, w, True, 0, 0.0))
            elif zm < zp:
                kfs.append(self.api.keyframe(t, xm, zm, w, True, -1, zp - zm))
            else:
                kfs.append(self.api.keyframe(t, xp, zp, w, True, 1, zp - zm))

    def build_keyframes(self, path, t_now: float, k0: Keyframe):
        kfs = [k0]
        x_prev, z_prev, gap_prev, t_rel_prev = k0.x, k0.z, self.api.tip_gap(k0.width), t_now
        w_prev = k0.width
        chains = []
        for e in path:
            if e.mode == "drum" and chains:
                chains[-1].append(e)
            else:
                chains.append([e])
        for ci, chain in enumerate(chains):
            head = chain[0]
            nxt = chains[ci + 1][0] if ci + 1 < len(chains) else None
            in_prog = head.in_progress or head.mode == "drum"
            if in_prog:
                t_from = t_now
            else:
                from_start = ci == 0
                e = head
                t_cmd = e.t_touch - LEAD
                tr = self.travel_time(x_prev, gap_prev, e.x, e.gap)
                over_lane = abs(e.x - x_prev) < 0.003 and abs(e.gap - gap_prev) < 0.003
                if from_start and e.mandatory and over_lane:
                    # final approach already under way: just finish the descent
                    z_ref = z_prev
                    if e.rock and e.ref_side:
                        z_ref = z_prev + e.ref_side * 0.5 * k0.rise
                    remaining = max(0.0, z_ref - self.touch_z) / self.v_down
                    if t_cmd < t_now + 0.5 * remaining:
                        t_cmd = t_now + 0.5 * remaining
                    t_desc_start = t_cmd - remaining
                    if t_desc_start > t_now + 1e-3:
                        kfs.append(self.make_kf(e, t_desc_start, e.x, e.gap, z_ref))
                    kfs.append(self.make_kf(e, t_cmd, e.x, e.gap, self.touch_z, True))
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
                    if t_arr < t_cmd - 1e-3 and (tr > 1e-6 or z_h < z_prev - 1e-4 or not from_start or e.rock):
                        kfs.append(self.make_kf(e, t_arr, e.x, e.gap, z_h))
                    if t_desc_start > t_arr + 1e-3 and t_desc_start < t_cmd - 1e-3:
                        kfs.append(self.make_kf(e, t_desc_start, e.x, e.gap, z_h))
                    kfs.append(self.make_kf(e, t_cmd, e.x, e.gap, self.touch_z, True))
                    # commit once the descent has (nearly) begun
                    if t_desc_start <= t_now + 0.05:
                        self.commit_event(e)
                t_from = t_cmd
            last = chain[-1]
            t_rel = self.release_time(last.state, nxt, t_now)
            if not in_prog:
                t_rel = max(t_rel, t_from + 1e-3)
            if t_rel > t_from:
                self.pressed_phase(chain, t_from, t_rel, k0, kfs)
            elif in_prog:
                # already released: the hand is lifting, continue from where it is
                x_prev, z_prev, gap_prev, w_prev, t_rel_prev = k0.x, k0.z, self.api.tip_gap(k0.width), k0.width, t_now
                continue
            for d in chain:
                if d.mode == "drum" and not d.mandatory and d.t_touch - DRUM_LEAD - ROLL_T <= t_now + 0.05:
                    self.commit_event(d)
            xm, xp = self.tip_x(last.state.tips[-1], t_rel), self.tip_x(last.state.tips[1], t_rel)
            gap_prev = max(self.min_gap, min(self.max_gap, xp - xm))
            x_prev, z_prev, w_prev, t_rel_prev = 0.5 * (xm + xp), self.touch_z, self.width_for_gap(gap_prev), t_rel
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
