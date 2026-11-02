"""Rhythm-game policy: pixel perception + note tracking + greedy/DP scheduling.

Screen conventions measured on the training frames (see probe.py):
  * 384 x 256 RGB, 6 lanes of 42.67 px; note heads are 38 px wide, 12 rows tall.
  * notes scroll down at 206.26 px/s; a head's bottom row is 335.18 - 206.26 * t_to_hit,
    its centre is on the judgment line (row 329.5) at the hit time.
  * tap = cyan bar; hold = green head + 24 px green column above it (its top reaches the
    line at the hold's end); slide = magenta head + 20 px magenta band tracing the lane
    path (row r reaches the line at t_now + (329.5 - r) / speed).
"""
import bisect
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from rhythmbot.cap.api import Keyframe, Plan, TAP, HOLD, SLIDE

# ----------------------------------------------------------------------------- screen
ROWS, COLS = 384, 256
N_LANES = 6
LANE_W_PX = COLS / N_LANES
PX_PER_S = 206.26
LINE_ROW = 329.5          # row of a head centre at its hit time
HEAD_BOT_ROW0 = 335.18    # row of a head's bottom at its hit time
HEAD_H = 12
LANE_C0 = [int(round(L * LANE_W_PX)) for L in range(N_LANES)]
LANE_C1 = [int(round((L + 1) * LANE_W_PX)) for L in range(N_LANES)]
HEAD_MIN = 28             # min pixels in a lane window for a head row (bar is 38 wide)
BODY_MIN, BODY_MAX = 12, 27   # hold body is 24-25 wide, slide body 20 wide


def _runs(idx: np.ndarray) -> List[Tuple[int, int]]:
    """Maximal runs of consecutive integers in a sorted index array."""
    if len(idx) == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    return list(zip(starts.tolist(), ends.tolist()))


class Detection:
    __slots__ = ("kind", "lane", "hit", "hit_ok", "end", "end_lb", "profile")

    def __init__(self, kind, lane, hit, hit_ok, end=0.0, end_lb=False, profile=None):
        self.kind = kind              # TAP / HOLD / SLIDE
        self.lane = lane              # head lane
        self.hit = hit                # absolute hit time estimate
        self.hit_ok = hit_ok          # head bottom fully visible and not stopped on the line
        self.end = end                # absolute end time (holds / slides)
        self.end_lb = end_lb          # end is only a lower bound (body clipped at top)
        self.profile = profile        # slide: list of (abs time, lane)


def perceive(frame: np.ndarray, t_now: float) -> List[Detection]:
    f = frame
    r, g, b = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    cyan = (r < 90) & (g > 140) & (b > 140)
    green = (r < 130) & (g > 140) & (b < 160)
    magenta = (r > 140) & (g < 130) & (b > 140)
    out: List[Detection] = []

    def head_time(bottom_row: int) -> float:
        return t_now + (HEAD_BOT_ROW0 - bottom_row) / PX_PER_S

    # --- taps and holds: per lane column counts
    for L in range(N_LANES):
        c0, c1 = LANE_C0[L], LANE_C1[L]
        cc = cyan[:, c0:c1].sum(axis=1)
        for a, bb in _runs(np.where(cc >= HEAD_MIN)[0]):
            if bb - a + 1 < 3 or bb >= ROWS - 1:
                continue
            out.append(Detection(TAP, L, head_time(bb), hit_ok=(bb < HEAD_BOT_ROW0 - 2)))
        gc = green[:, c0:c1].sum(axis=1)
        heads = _runs(np.where(gc >= HEAD_MIN)[0])
        if heads:
            body_rows = np.where((gc >= BODY_MIN) & (gc < HEAD_MIN))[0]
            body_runs = _runs(body_rows)
            for a, bb in heads:
                if bb - a + 1 < 3 or bb >= ROWS - 1:
                    continue
                top = None
                for ba, be in body_runs:
                    if be == a - 1 or be == a - 2:
                        top = ba
                        break
                hit = head_time(bb)
                if top is None:
                    end, lb = hit + 0.05, True
                else:
                    end, lb = t_now + (LINE_ROW - top) / PX_PER_S, (top <= 0)
                out.append(Detection(HOLD, L, hit, hit_ok=(bb < HEAD_BOT_ROW0 - 2), end=end, end_lb=lb))

    # --- slides: connected components of magenta
    if magenta.any():
        labels, n = ndimage.label(magenta, structure=np.ones((3, 3), dtype=int))
        objs = ndimage.find_objects(labels)
        for k, sl in enumerate(objs, start=1):
            if sl is None:
                continue
            rs, cs = sl
            comp = labels[rs, :] == k           # full width rows of this component
            r0 = rs.start
            heads = []
            for L in range(N_LANES):
                c0, c1 = LANE_C0[L], LANE_C1[L]
                win = comp[:, c0:c1]
                cnt = win.sum(axis=1)
                # clean head rows: bar fills the lane but leaves the lane margins blank
                edge = win[:, 0] | win[:, 1] | win[:, -1] | win[:, -2]
                ok = (cnt >= 34) & (~edge)
                for a, bb in _runs(np.where(ok)[0]):
                    if bb - a + 1 < 7:
                        continue
                    # a real head has nothing of this component right below it (a bend
                    # joint of the band can look like a bar, but the band continues below)
                    below = cnt[bb + 1] if bb + 1 < len(cnt) else 0
                    if below > 3 and bb + r0 < ROWS - 1:
                        continue
                    heads.append((L, a + r0, bb + r0))
            # usually one head per component; crossing slides share a component, so
            # trace a body upward from every head
            for L, a, bb in heads:
                if bb >= ROWS - 1:
                    continue
                hit = head_time(bb)
                hit_ok = (bb < HEAD_BOT_ROW0 - 2)
                prof: List[Tuple[float, float]] = []
                centre = 0.5 * (LANE_C0[L] + LANE_C1[L])
                top_row = bb - HEAD_H + 1
                row = top_row - 1
                last_row = row
                while row >= r0:
                    line = comp[row - r0]
                    cols = np.where(line)[0]
                    if len(cols) == 0:
                        break
                    best = None
                    for ca, cb in _runs(cols):
                        c = 0.5 * (ca + cb)
                        if best is None or abs(c - centre) < abs(best - centre):
                            best = c
                    if best is None or abs(best - centre) > 30:
                        break
                    centre = best
                    lane = (centre + 0.5) / LANE_W_PX - 0.5
                    prof.append((t_now + (LINE_ROW - row) / PX_PER_S, lane))
                    last_row = row
                    row -= 2
                if prof:
                    end = prof[-1][0]
                    lb = last_row <= 1
                else:
                    end, lb = hit + 0.05, True
                prof.insert(0, (hit, float(L)))
                out.append(Detection(SLIDE, L, hit, hit_ok, end=end, end_lb=lb, profile=prof))
    return out


# ----------------------------------------------------------------------------- tracking
class Track:
    __slots__ = ("id", "kind", "lane", "hit", "hits", "end", "end_lb", "profile", "first_seen", "last_seen", "n_seen")

    def __init__(self, tid, det, t_now):
        self.id = tid
        self.kind = det.kind
        self.lane = det.lane
        self.hit = det.hit
        self.hits = [det.hit] if det.hit_ok else []
        self.end = det.end if det.kind != TAP else det.hit
        self.end_lb = det.end_lb
        self.profile = list(det.profile) if det.profile else None
        self.first_seen = t_now
        self.last_seen = t_now
        self.n_seen = 1

    def update(self, det, t_now):
        self.last_seen = t_now
        self.n_seen += 1
        if det.hit_ok:
            self.hits.append(det.hit)
            if len(self.hits) > 12:
                self.hits = self.hits[-12:]
            self.hit = float(np.mean(self.hits))
        if self.kind != TAP:
            if not det.end_lb:
                self.end = det.end
                self.end_lb = False
            elif self.end_lb or det.end > self.end:
                self.end = max(self.end, det.end)
        if self.kind == SLIDE and det.profile:
            new = det.profile
            t_max = new[-1][0]
            keep = [p for p in (self.profile or []) if p[0] > t_max + 1e-6]
            if det.end_lb or not keep:
                self.profile = list(new) + keep
            else:
                self.profile = list(new)

    def lane_at(self, t):
        """Lane of the (moving) target at absolute time t."""
        if self.kind != SLIDE or not self.profile:
            return float(self.lane)
        p = self.profile
        if t <= p[0][0]:
            return p[0][1]
        if t >= p[-1][0]:
            return p[-1][1]
        ts = [q[0] for q in p]
        i = bisect.bisect_right(ts, t)
        (t0, l0), (t1, l1) = p[i - 1], p[i]
        if t1 - t0 < 1e-6:
            return l1
        return l0 + (l1 - l0) * (t - t0) / (t1 - t0)


class Tracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, dets, t_now):
        unmatched_tracks = set(self.tracks.keys())
        for d in dets:
            best, best_err = None, 1e9
            for tid in unmatched_tracks:
                tr = self.tracks[tid]
                if tr.kind != d.kind or tr.lane != d.lane:
                    continue
                if d.hit_ok:
                    err = abs(tr.hit - d.hit)
                    if err < 0.06 and err < best_err:
                        best, best_err = tid, err
                else:
                    # head clipped at top or parked on the line: match the track nearest in time
                    err = abs(tr.hit - d.hit)
                    if tr.hit <= d.hit + 0.08 and tr.hit >= d.hit - 0.08:
                        if err < best_err:
                            best, best_err = tid, err
                    elif d.hit - t_now < 0.05 and tr.hit < t_now + 0.05 and tr.end >= t_now - 0.1:
                        err = t_now - tr.hit
                        if err < best_err:
                            best, best_err = tid, err
            if best is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = Track(tid, d, t_now)
            else:
                self.tracks[best].update(d, t_now)
                unmatched_tracks.discard(best)
        # prune
        for tid in list(self.tracks.keys()):
            tr = self.tracks[tid]
            gone = t_now - tr.last_seen
            if tr.kind == TAP:
                if (gone > 0.2 and tr.hit < t_now) or tr.hit < t_now - 0.25 or (gone > 0.3):
                    del self.tracks[tid]
            else:
                if tr.end < t_now - 0.2 or gone > 0.4:
                    del self.tracks[tid]

    def live(self, t_now):
        return [tr for tr in self.tracks.values() if tr.n_seen >= 3]


# ----------------------------------------------------------------------------- scheduling
class Action:
    """One press of the hand: one note, or two notes played with one fingertip each."""
    __slots__ = ("ids", "notes", "kind", "t_hit", "t_end", "t_rel", "x0", "width", "value", "slide", "offset",
                 "x_end", "side", "subtaps", "rock")

    def __init__(self, notes, api, x0=None, width=None, side=0, subtaps=None):
        """x0 / width: the hand pose (midpoint, opening) for this press.  Default: closed
        gripper centred on a single note, or opened to span a pair.  `side` (+1/-1) says
        which fingertip is on a single note's lane (0: closed / pair).  `subtaps`: taps
        played by the other, rolled-up fingertip while a hold is pressed."""
        self.notes = notes
        self.side = side
        self.subtaps = list(subtaps) if subtaps else []
        self.rock = None          # rocking chain: [(touch time, finger side, finger x), ...]
        self.ids = tuple(sorted(n.id for n in notes))
        self.t_hit = min(n.hit for n in notes)
        self.kind = TAP
        for n in notes:
            if n.kind == SLIDE:
                self.kind = SLIDE
            elif n.kind == HOLD and self.kind != SLIDE:
                self.kind = HOLD
        self.slide = next((n for n in notes if n.kind == SLIDE), None)
        self.value = float(len(notes))
        xs = [api.lane_x(n.lane) for n in notes]
        if x0 is not None:
            self.x0, self.width = x0, width
        elif len(notes) == 1:
            self.x0 = xs[0]
            self.width = 0.0
        else:
            gap = abs(xs[0] - xs[1])
            self.width = api.width_for_tip_gap(gap)
            self.x0 = 0.5 * (xs[0] + xs[1])
        if self.slide is not None:
            self.offset = self.x0 - api.lane_x(self.slide.lane)
        else:
            self.offset = 0.0
        if self.kind == TAP:
            self.t_end = self.t_hit
        else:
            self.t_end = max(n.end for n in notes if n.kind != TAP)
        self.finish(api)

    @property
    def rolled(self):
        """True when one fingertip presses while the other is rolled up (hold with
        sub-taps, or a rocking chain of taps alternating fingertips)."""
        return bool(self.subtaps) or self.rock is not None

    def rise(self):
        """Keyframe `rise` that lifts the free fingertip: +x above -x when the hold finger is -x."""
        return -self.side * ROLL_RISE if self.rolled else 0.0

    def finish(self, api):
        """Derive release time and end position (call after editing timing fields)."""
        if self.rock is not None:
            self.t_rel = self.rock[-1][0] + T_DWELL
        elif self.kind == TAP:
            self.t_rel = self.t_hit + LEAD + T_DWELL
        else:
            self.t_rel = max(self.t_end - HOLD_RELEASE_EARLY, self.t_hit + LEAD + T_DWELL)
        self.x_end = self.x_at(self.t_rel + (SLIDE_LEAD if self.slide is not None else 0.0), api)

    def x_at(self, t, api):
        if self.slide is None:
            return self.x0
        return api.lane_x(self.slide.lane_at(t)) + self.offset


DEBUG = False                # raise once at DEBUG_T with a log of commits / skips (diagnostics only)
DEBUG_T = 20.4
EXPERIMENT = None            # calibration only: (delta, dx), ("roll", gap) or ("rock", gap, delta); None = off

# motion parameters
V_TRAVEL = 1.4          # m/s lateral
W_RATE = 0.4            # m/s opening change (the gripper itself is slow)
HOVER_MAX = 0.035       # m
HOVER_MIN = 0.020       # m
T_DWELL = 0.06          # s a tap stays pressed
SETTLE = 0.035          # s hovering above the target before the descent (lets lateral lag die out)
SETTLE_PAIR = 0.07      # same for two-note presses (both fingertips must land in lane)
STICKY = 0.3            # value bonus for keeping the previously planned next action
LEAD = 0.010            # s: command the touch this much after the hit time (contact starts ~15 ms before the keyframe)
SLIDE_LEAD = 0.06       # s: look-ahead when following a slide
HOLD_RELEASE_EARLY = 0.03
SLACK = 0.0             # s: transitions may be this much tighter than the nominal model
MOVE_PENALTY = 0.05     # value lost per second of transition time (tie-breaker)
ROLL_RISE = 0.012       # m: how high the free fingertip is rolled up while the other holds
ROLL_DOWN = 0.074       # s: rolling the free fingertip down onto the screen
ROLL_UP = 0.06          # s: rolling it back up
ROLL_CYCLE = 0.24       # s: min spacing of rolled taps
ROLL_FIRST = 0.22       # s: earliest rolled tap after the hold's own hit
ROLL_LAST = 0.08        # s: a rolled tap must end this long before the hold's release
SUBTAP_VALUE = 0.9      # value of a rolled tap (slightly below a normal tap)
ROCK_MIN = 0.16         # s: min spacing of taps in a rocking chain (alternating fingertips)
ROCK_LEAD = 0.03        # s: the rolling fingertip lags, so command its touch this much early
ROCK_MAX = 0.40         # s: beyond this a normal transition is used instead
ROCK_VALUE = 0.95       # value per tap in a chain (slightly below normal taps)
ROCK_GAPS = (0.05, 0.10)   # fingertip gaps usable for rocking between two lanes


def t_desc(h):
    return 0.074 + max(0.0, h - HOVER_MIN) / 0.35


def t_lift(h):
    return 0.06 + max(0.0, h - HOVER_MIN) / 0.4


def settle_for(a, w_from):
    """Two-note presses need both fingertips in lane, so give the (slow) gripper extra
    time to finish an opening change before the descent."""
    return SETTLE_PAIR if (len(a.notes) > 1 and not a.rolled and abs(a.width - w_from) > 0.002) else SETTLE


def transition_time(x_from, w_from, a, h=HOVER_MIN):
    """Release -> next touch: the lift overlaps the lateral move, then settle, then descend."""
    travel = max(abs(a.x0 - x_from) / V_TRAVEL, abs(a.width - w_from) / W_RATE)
    return max(t_lift(h), travel) + settle_for(a, w_from) + t_desc(h)


def choose_hover(avail, travel, settle):
    """Largest hover height whose lift/travel + settle + descend fits in `avail` seconds."""
    rest = avail - max(travel, 0.06) - settle - 0.074 - 0.0
    # extra height costs (h-HOVER_MIN)/0.35 on the descent and (h-HOVER_MIN)/0.4 on the lift,
    # but the lift only matters when it is longer than the travel; be conservative
    if rest <= 0:
        return HOVER_MIN
    return float(min(HOVER_MAX, max(HOVER_MIN, HOVER_MIN + rest / 5.357)))


def rock_chain(groups, gi, first, side, gap, api):
    """Taps in consecutive groups alternating between the two fingertips of a hand held
    still: fingertip `side` on `first`'s lane, the other one `gap` away.  Returns an
    Action for the chain, or None when the next note does not continue it."""
    k = int(round(gap / api.lane_width))
    lanes = {side: first.lane, -side: first.lane - side * k}
    if not (0 <= lanes[-side] < N_LANES):
        return None
    chain = [first]
    sides = [side]
    for g in groups[gi + 1:]:
        if len(g) != 1 or g[0].kind != TAP:
            break
        n = g[0]
        s = -sides[-1]
        dt = n.hit - chain[-1].hit
        if n.lane != lanes[s] or dt < ROCK_MIN or dt > ROCK_MAX:
            break
        chain.append(n)
        sides.append(s)
    if len(chain) < 2:
        return None
    w = api.width_for_tip_gap(gap)
    x0 = api.lane_x(first.lane) - side * 0.5 * gap
    act = Action(chain, api, x0, w, side)
    act.kind = TAP
    act.rock = [(n.hit + LEAD, s, x0 + s * 0.5 * gap) for n, s in zip(chain, sides)]
    act.t_hit = chain[0].hit
    act.t_end = chain[-1].hit
    act.value = ROCK_VALUE * len(chain)
    act.finish(api)
    return act


def build_actions(tracks, api, t_now, exclude_ids):
    notes = [n for n in tracks if n.id not in exclude_ids and n.hit > t_now + 0.03]
    notes.sort(key=lambda n: n.hit)
    groups = []
    for n in notes:
        if groups and n.hit - groups[-1][0].hit < 0.035:
            groups[-1].append(n)
        else:
            groups.append([n])
    taps_by_lane = {}
    for n in notes:
        if n.kind == TAP:
            taps_by_lane.setdefault(n.lane, []).append(n)
    actions = []
    for gi, g in enumerate(groups):
        cands = []
        for n in g:
            x = api.lane_x(n.lane)
            cands.append(Action([n], api, x, 0.0))                 # closed, both tips in the lane
            for gap, w in OPEN_POSES:
                for side in (-1, 1):                               # which fingertip is on the lane
                    cands.append(Action([n], api, x - side * 0.5 * gap, w, side))
                    if n.kind == TAP and gap in ROCK_GAPS and ROLL_RISE > 0:
                        chain = rock_chain(groups, gi, n, side, gap, api)
                        if chain is not None:
                            cands.append(chain)
                    if n.kind != HOLD or ROLL_RISE <= 0:
                        continue
                    # a hold pressed with one fingertip: the other, rolled up, can tap in its lane
                    k = gap / api.lane_width
                    if abs(k - round(k)) > 0.05:
                        continue
                    free_lane = n.lane - side * int(round(k))
                    if not (0 <= free_lane < N_LANES):
                        continue
                    subs = []
                    last = -1e9
                    for t in taps_by_lane.get(free_lane, []):
                        if t.hit < n.hit + ROLL_FIRST or t.hit + LEAD + T_DWELL > n.end - HOLD_RELEASE_EARLY - ROLL_LAST:
                            continue
                        if t.hit - last < ROLL_CYCLE:
                            continue
                        subs.append(t)
                        last = t.hit
                    if subs:
                        act = Action([n] + subs, api, x - side * 0.5 * gap, w, side, subs)
                        act.kind = HOLD
                        act.t_hit = n.hit
                        act.t_end = n.end
                        act.value = 1.0 + SUBTAP_VALUE * len(subs)
                        act.finish(api)
                        cands.append(act)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g[i], g[j]
                if a.lane == b.lane or (a.kind == SLIDE and b.kind == SLIDE):
                    continue
                act = Action([a, b], api)
                if act.width is None:
                    continue
                if act.slide is not None and any(n.kind == HOLD for n in (a, b)):
                    act.value = 1.5
                cands.append(act)
        actions.append(cands)
    return actions


OPEN_POSES = []   # (tip gap, opening) pairs for playing a single note with one fingertip


def init_poses(api):
    OPEN_POSES.clear()
    for gap in (0.05, 0.10, api.tip_gap(api.max_width) - 0.002):
        w = api.width_for_tip_gap(gap)
        if w is not None:
            OPEN_POSES.append((gap, w))


def schedule(groups, root, t_now, x_now, w_now, lifted, prefer=None):
    """DP over groups (each a list of alternative actions / poses): choose at most one
    action per group maximising total value subject to the hand reaching each in time.
    `root` is the last committed action (the hand starts from its release), else the
    hand starts from its current pose.  `prefer` = (ids, x0, width) of the action planned
    first last tick: it gets a bonus so the plan does not flip between equal options."""
    flat = []
    gstart = []          # index of the first candidate of each candidate's group
    for gi, cands in enumerate(groups):
        s = len(flat)
        for a in cands:
            flat.append((gi, a))
            gstart.append(s)
    n = len(flat)
    if n == 0:
        return []
    # vectorised transition model over all earlier candidates
    X0 = np.array([a.x0 for _, a in flat])
    W = np.array([a.width for _, a in flat])
    TH = np.array([a.t_hit + LEAD for _, a in flat])
    XE = np.array([a.x_end for _, a in flat])
    TR = np.array([a.t_rel for _, a in flat])
    VAL = np.array([a.value for _, a in flat])
    PAIR = np.array([len(a.notes) > 1 and not a.rolled for _, a in flat])
    best = np.full(n, -1.0)
    prev = np.full(n, -2, dtype=int)
    t_l, t_d = t_lift(HOVER_MIN), t_desc(HOVER_MIN)
    for i, (gi, a) in enumerate(flat):
        # from the start
        if root is not None:
            need = transition_time(root.x_end, root.width, a)
            start_ok = TH[i] - root.t_rel >= need - SLACK
            cost = need
        else:
            travel = max(abs(a.x0 - x_now) / V_TRAVEL, abs(a.width - w_now) / W_RATE)
            need = travel + settle_for(a, w_now) + t_d + (0.0 if lifted else t_l)
            start_ok = TH[i] - t_now >= need - SLACK
            cost = travel
        if start_ok:
            best[i] = a.value - MOVE_PENALTY * cost
            prev[i] = -1
            if prefer is not None and a.ids == prefer[0]:
                best[i] += STICKY if (abs(a.x0 - prefer[1]) < 1e-6 and abs(a.width - prefer[2]) < 1e-6) else 0.5 * STICKY
        m = gstart[i]
        if m == 0:
            continue
        travel = np.maximum(np.abs(X0[i] - XE[:m]) / V_TRAVEL, np.abs(W[i] - W[:m]) / W_RATE)
        settle = SETTLE
        if PAIR[i]:
            settle = np.where(np.abs(W[i] - W[:m]) > 0.002, SETTLE_PAIR, SETTLE)
        need = np.maximum(t_l, travel) + settle + t_d
        ok = (best[:m] >= 0) & (TH[i] - TR[:m] >= need - SLACK)
        if ok.any():
            v = np.where(ok, best[:m] + VAL[i] - MOVE_PENALTY * need, -np.inf)
            j = int(np.argmax(v))
            if v[j] > best[i] + 1e-9:
                best[i] = float(v[j])
                prev[i] = j
    i_best = max(range(n), key=lambda i: (best[i], -flat[i][1].t_hit))
    if best[i_best] < 0:
        return []
    seq = []
    i = i_best
    while i >= 0:
        seq.append(flat[i][1])
        i = prev[i]
    seq.reverse()
    return seq


# ----------------------------------------------------------------------------- policy
class Policy:
    def __init__(self, api):
        self.api = api
        init_poses(api)
        self.tracker = Tracker()
        self.active = None          # last committed Action (in progress or just finished)
        self.active_touch_t = None
        self.active_h = HOVER_MIN
        self.prefer = None
        self.hover_z = api.screen_z + HOVER_MAX
        self.err_count = 0
        self.log = []
        self.logged_ids = set()
        self.debug_done = False

    # -- helpers
    def kf(self, t, x, z, w, down=False):
        return self.api.keyframe(t, x, z, w, down, 0, 0.0)

    def refresh_active(self, t_now):
        """Rebuild the committed action from the live tracks (slides' paths and hold ends
        keep being refined while they are played).  A finished action is kept as the
        scheduling root until the next commit, so decisions stay consistent."""
        a = self.active
        if a is None:
            return None
        if t_now > a.t_rel + 1.5:
            self.active = None
            return None
        if t_now > a.t_rel + 0.05:
            return a
        notes = [self.tracker.tracks.get(i) for i in a.ids]
        notes = [n for n in notes if n is not None]
        if notes and a.kind != TAP and any(n.kind != TAP for n in notes):
            sub_ids = set(n.id for n in a.subtaps)
            subs = [n for n in notes if n.id in sub_ids]
            if a.rolled:
                # taps in the free fingertip's lane that came into view since the commit
                hold = next(n for n in notes if n.kind != TAP)
                k = int(round(self.api.tip_gap(a.width) / self.api.lane_width))
                free_lane = hold.lane - a.side * k
                have = sorted(n.hit for n in subs)
                for tr in self.tracker.live(t_now):
                    if tr.kind != TAP or tr.lane != free_lane or tr.id in sub_ids or tr.id in a.ids:
                        continue
                    if tr.hit < max(t_now + 0.15, a.t_hit + ROLL_FIRST):
                        continue
                    if tr.hit + LEAD + T_DWELL > hold.end - HOLD_RELEASE_EARLY - ROLL_LAST:
                        continue
                    if any(abs(tr.hit - h) < ROLL_CYCLE for h in have):
                        continue
                    subs.append(tr)
                    notes.append(tr)
                    have.append(tr.hit)
            b = Action(notes, self.api, a.x0, a.width, a.side, subs)
            b.ids = tuple(sorted(set(a.ids) | set(n.id for n in notes)))
            b.kind = a.kind
            b.offset = a.offset
            b.t_hit = a.t_hit
            b.value = a.value
            if a.rolled and not b.subtaps:
                b.subtaps = list(a.subtaps)        # keep the rolled posture after the taps are gone
            b.finish(self.api)
            self.active = b
        return self.active

    def kf2(self, t, x, z, w, down, ref, rise):
        return self.api.keyframe(t, x, z, w, down, ref, rise)

    def ref_x(self, a):
        """x of the fingertip that presses a rolled hold (the keyframe `ref` point)."""
        return a.x0 + a.side * 0.5 * self.api.tip_gap(a.width)

    def inject_fakes(self, seq, active, t_now):
        """Calibration experiment: before a well-isolated tap, insert an extra press
        EXPERIMENT[0] seconds earlier, EXPERIMENT[1] metres to the side."""
        api = self.api
        out = []
        prev_rel = active.t_rel if active is not None else t_now
        for a in seq:
            isolated = a.kind == TAP and len(a.notes) == 1 and a.t_hit - prev_rel > 0.8 and not a.ids[0] == "fake"
            if isolated and EXPERIMENT[0] == "roll":
                # play the tap with the rolled-up fingertip while the other presses a fake hold next to it
                tap = a.notes[0]
                gap = EXPERIMENT[1]
                k = int(round(gap / api.lane_width))
                hold_lane = tap.lane - k if tap.lane - k >= 0 else tap.lane + k
                side = -1 if hold_lane < tap.lane else 1     # free finger must be on tap.lane
                w = api.width_for_tip_gap(gap)
                f = Action([tap], api, api.lane_x(hold_lane) - side * 0.5 * gap, w, side, [tap])
                f.kind = HOLD
                f.t_hit = tap.hit - 0.4
                f.t_end = tap.hit + 0.35
                f.finish(api)
                out.append(f)
                prev_rel = f.t_rel
                continue
            if isolated and EXPERIMENT[0] == "rock":
                # fake tap by the other fingertip EXPERIMENT[2] s earlier, then rock onto the real tap
                tap = a.notes[0]
                gap, delta = EXPERIMENT[1], EXPERIMENT[2]
                k = int(round(gap / api.lane_width))
                other = tap.lane - k if tap.lane - k >= 0 else tap.lane + k
                side = 1 if other < tap.lane else -1          # real tap's fingertip
                w = api.width_for_tip_gap(gap)
                x0 = api.lane_x(tap.lane) - side * 0.5 * gap
                f = Action([tap], api, x0, w, -side)
                f.kind = TAP
                f.rock = [(tap.hit + LEAD - delta, -side, x0 - side * 0.5 * gap), (tap.hit + LEAD, side, x0 + side * 0.5 * gap)]
                f.t_hit = tap.hit - delta
                f.t_end = tap.hit
                f.finish(api)
                out.append(f)
                prev_rel = f.t_rel
                continue
            if isolated and EXPERIMENT[0] != "roll":
                delta, dx = EXPERIMENT
                f = Action(list(a.notes), api, a.x0 + dx, a.width)
                f.ids = ("fake", -a.ids[0])
                f.t_hit = a.t_hit - delta
                f.t_end = f.t_hit
                f.value = 0.0
                f.finish(api)
                out.append(f)
            out.append(a)
            prev_rel = a.t_rel
        return out

    def plan(self, frame, t_now, X_WT_cmd, width_cmd):
        api = self.api
        k0 = api.current_keyframe(t_now, X_WT_cmd, width_cmd)
        try:
            return self._plan(frame, t_now, k0)
        except Exception:
            self.err_count += 1
            if DEBUG and self.debug_done:
                raise
            # recover: drop the committed action and hover in place for this tick
            self.active = None
            self.prefer = None
            hover = self.kf(t_now + 0.2, k0.x, self.hover_z, k0.width)
            return api.make_plan([k0, hover], t_now)

    def _plan(self, frame, t_now, k0):
        api = self.api
        dets = perceive(frame, t_now)
        self.tracker.update(dets, t_now)
        tracks = self.tracker.live(t_now)
        if DEBUG:
            for tr in tracks:
                if tr.hit < t_now and tr.id not in self.logged_ids:
                    self.logged_ids.add(tr.id)
                    a = self.active
                    self.log.append("SKIP %s L%d %.2f n%d [root %s rel%.2f x%.0f w%.0f]" % (
                        tr.kind[0], tr.lane, tr.hit, tr.n_seen, a.kind[0] if a else "-",
                        a.t_rel if a else 0, 1000 * a.x_end if a else 0, 1000 * a.width if a else 0))
            if t_now > DEBUG_T and not self.debug_done:
                self.debug_done = True
                raise RuntimeError(" | ".join(self.log))

        active = self.refresh_active(t_now)
        in_progress = active is not None and t_now < active.t_rel + 0.03
        exclude = set(active.ids) if active else set()
        groups = build_actions(tracks, api, t_now, exclude)
        lifted = k0.z > api.contact_z + 0.004
        seq = schedule(groups, active, t_now, k0.x, k0.width, lifted, self.prefer)
        self.prefer = (seq[0].ids, seq[0].x0, seq[0].width) if seq else None
        if EXPERIMENT is not None:
            seq = self.inject_fakes(seq, active, t_now)

        kfs = [k0]
        if active is not None:
            # the hand starts from the committed action's release (possibly in the past)
            cur_t, cur_x, cur_w, cur_down = active.t_rel, active.x_end, active.width, True
            first_new = 0
            if in_progress:
                seq = [active] + seq
                first_new = 1
        else:
            cur_t, cur_x, cur_w, cur_down = t_now, k0.x, k0.width, not lifted
            first_new = 0

        def akf(a, t, z, down=False, rise=None):
            """Keyframe for action `a`: normally the midpoint at a.x0; a rolled hold places
            its pressing fingertip (ref) with the other one rolled up."""
            if a.rolled:
                return self.kf2(t, self.ref_x(a), z, a.width, down, a.side, a.rise() if rise is None else rise)
            return self.kf(t, a.x0, z, a.width, down)

        for idx, a in enumerate(seq):
            t_touch = a.t_hit + LEAD
            is_active = (idx == 0 and in_progress)
            if is_active and t_now >= self.active_touch_t - 0.005:
                pass                                   # already pressing: no approach
            elif is_active:
                # approaching the committed action: hover plateau at the target, then descend
                h = self.active_h
                hz = api.screen_z + h
                t_down = t_touch - t_desc(h)
                if t_down > t_now + 0.005:
                    kfs.append(akf(a, t_down, hz))
                kfs.append(akf(a, max(t_touch, t_now + 0.01), api.touch_z, True))
            else:
                settle = settle_for(a, cur_w)
                avail = t_touch - cur_t
                travel = max(abs(a.x0 - cur_x) / V_TRAVEL, abs(a.width - cur_w) / W_RATE)
                if idx == 0 and not cur_down:
                    h = float(min(HOVER_MAX, max(HOVER_MIN, k0.z - api.screen_z)))
                    tl = 0.0
                else:
                    h = choose_hover(avail, travel, settle)
                    tl = t_lift(h)
                hz = api.screen_z + h
                t_arrive = max(t_touch - t_desc(h) - settle, cur_t + max(tl, travel))
                if tl > 0 and cur_t + tl > t_now + 0.005:
                    # lift while already moving: x and width progress at the pace of the whole move
                    frac = min(1.0, tl / max(t_arrive - cur_t, 1e-6))
                    kfs.append(self.kf(cur_t + tl, cur_x + frac * (a.x0 - cur_x), hz,
                                       cur_w + frac * (a.width - cur_w)))
                if t_arrive > t_now + 0.01 and t_arrive < t_touch - 0.01:
                    kfs.append(akf(a, t_arrive, hz))
                t_down = max(t_arrive + settle, t_touch - t_desc(h))
                if t_down > t_arrive + 0.005 and t_down > t_now + 0.01 and t_down < t_touch - 0.01:
                    kfs.append(akf(a, t_down, hz))
                kfs.append(akf(a, max(t_touch, t_now + 0.01), api.touch_z, True))
                if idx == first_new and t_now >= t_touch - t_desc(h) - settle - 0.02:
                    # the hand should be arriving above the target: commit to this press
                    self.active = a
                    self.active_touch_t = t_touch
                    self.active_h = h
                    in_progress = True
                    if DEBUG:
                        for n in a.notes:
                            self.logged_ids.add(n.id)
                        self.log.append("DO %s %s %.2f w%.0f x%.0f dt%.2f" % (
                            a.kind[0], ",".join("L%d" % n.lane for n in a.notes), t_touch,
                            1000 * a.width, 1000 * a.x0, t_touch - t_now))
            t_release = a.t_rel
            if a.kind == SLIDE:
                ts = max(t_now, t_touch) + 1.0 / 30
                while ts < t_release:
                    kfs.append(self.kf(ts, a.x_at(ts + SLIDE_LEAD, api), api.touch_z, a.width, True))
                    ts += 1.0 / 30
            elif a.rock is not None:
                # rocking chain: after each tap's dwell, roll onto the other fingertip's tap
                for i, (tt, s, xf) in enumerate(a.rock):
                    if tt + T_DWELL < t_now:
                        continue
                    if i > 0:
                        kfs.append(self.kf2(tt - ROCK_LEAD, xf, api.touch_z, a.width, True, s, -s * ROLL_RISE))
                    kfs.append(self.kf2(tt + T_DWELL, xf, api.touch_z, a.width, True, s, -s * ROLL_RISE))
            elif a.rolled:
                # taps by the rolled-up fingertip: roll it down onto its lane and back up
                for tp in sorted(a.subtaps, key=lambda n: n.hit):
                    tt = tp.hit + LEAD
                    if tt + T_DWELL + ROLL_UP > t_release or tt - ROLL_DOWN < t_touch + 0.05:
                        continue
                    if tt + T_DWELL + ROLL_UP < t_now:
                        continue
                    kfs.append(akf(a, tt - ROLL_DOWN, api.touch_z, True))
                    kfs.append(akf(a, tt, api.touch_z, True, rise=0.0))
                    kfs.append(akf(a, tt + T_DWELL, api.touch_z, True, rise=0.0))
                    kfs.append(akf(a, tt + T_DWELL + ROLL_UP, api.touch_z, True))
            x_end = a.x_end
            if a.rock is not None:
                tt, s, xf = a.rock[-1]
                kfs.append(self.kf2(max(t_release, t_now + 0.02), xf, api.touch_z, a.width, True, s, -s * ROLL_RISE))
            else:
                kfs.append(akf(a, max(t_release, t_now + 0.02), api.touch_z, True))
            cur_t, cur_x, cur_w = max(t_release, t_now + 0.02), x_end, a.width
            cur_down = True
        # park: lift after the last action
        kfs.append(self.kf(max(cur_t + t_lift(HOVER_MAX), t_now + 0.05), cur_x, self.hover_z, cur_w))
        return api.make_plan(kfs, t_now)


def make_policy(api):
    return Policy(api)
