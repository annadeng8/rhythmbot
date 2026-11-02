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
                    if bb - a + 1 >= 3:
                        heads.append((L, a + r0, bb + r0))
            if not heads:
                continue
            # one head per component normally; take the lowest (closest to the line) as the note head
            heads.sort(key=lambda h: -h[2])
            L, a, bb = heads[0]
            if bb >= ROWS - 1:
                continue
            hit = head_time(bb)
            hit_ok = (bb < HEAD_BOT_ROW0 - 2)
            # trace the body upward from above the head
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
        return [tr for tr in self.tracks.values() if tr.n_seen >= 2]


# ----------------------------------------------------------------------------- scheduling
class Action:
    """One press of the hand: one note, or two notes played with one fingertip each."""
    __slots__ = ("ids", "notes", "kind", "t_hit", "t_end", "t_rel", "x0", "width", "value", "slide", "offset", "x_end")

    def __init__(self, notes, api, x0=None, width=None):
        """x0 / width: the hand pose (midpoint, opening) for this press.  Default: closed
        gripper centred on a single note, or opened to span a pair."""
        self.notes = notes
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

    def finish(self, api):
        """Derive release time and end position (call after editing timing fields)."""
        if self.kind == TAP:
            self.t_rel = self.t_hit + LEAD + T_DWELL
        else:
            self.t_rel = max(self.t_end - HOLD_RELEASE_EARLY, self.t_hit + LEAD + T_DWELL)
        self.x_end = self.x_at(self.t_rel + (SLIDE_LEAD if self.slide is not None else 0.0), api)

    def x_at(self, t, api):
        if self.slide is None:
            return self.x0
        return api.lane_x(self.slide.lane_at(t)) + self.offset


DEBUG = True
DEBUG_T = 19.8

# motion parameters
V_TRAVEL = 1.1          # m/s lateral
W_RATE = 1.0            # m/s opening change while lifted
HOVER_MAX = 0.030       # m
HOVER_MIN = 0.012       # m
T_DWELL = 0.06          # s a tap stays pressed
LEAD = 0.0              # s: command the touch this much before the hit time
SLIDE_LEAD = 0.06       # s: look-ahead when following a slide
HOLD_RELEASE_EARLY = 0.03
SLACK = 0.02            # s: transitions may be this much tighter than the nominal model
MOVE_PENALTY = 0.05     # value lost per second of transition time (tie-breaker)


def t_desc(h):
    return 0.05 + h / 0.3


def t_lift(h):
    return 0.025 + h / 0.45


def transition_time(x_from, w_from, x_to, w_to, h=HOVER_MIN):
    travel = max(abs(x_to - x_from) / V_TRAVEL, abs(w_to - w_from) / W_RATE)
    return t_lift(h) + travel + t_desc(h)


def choose_hover(avail, travel):
    """Largest hover height whose lift + travel + descend fits in `avail` seconds."""
    rest = avail - travel - 0.075   # constant parts of t_lift + t_desc
    if rest <= 0:
        return HOVER_MIN
    # h/0.3 + h/0.45 = rest  ->  h = rest / 5.556
    return float(min(HOVER_MAX, max(HOVER_MIN, rest / 5.556)))


def build_actions(tracks, api, t_now, exclude_ids):
    notes = [n for n in tracks if n.id not in exclude_ids and n.hit > t_now + 0.03]
    notes.sort(key=lambda n: n.hit)
    groups = []
    for n in notes:
        if groups and n.hit - groups[-1][0].hit < 0.035:
            groups[-1].append(n)
        else:
            groups.append([n])
    actions = []
    for g in groups:
        cands = []
        for n in g:
            x = api.lane_x(n.lane)
            cands.append(Action([n], api, x, 0.0))                 # closed, both tips in the lane
            for gap, w in OPEN_POSES:
                for side in (-1.0, 1.0):                           # which fingertip is on the lane
                    cands.append(Action([n], api, x - side * 0.5 * gap, w))
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
    for gap in (0.05, 0.10):
        w = api.width_for_tip_gap(gap)
        if w is not None:
            OPEN_POSES.append((gap, w))


def schedule(groups, root, t_now, x_now, w_now, lifted):
    """DP over groups (each a list of alternative actions / poses): choose at most one
    action per group maximising total value subject to the hand reaching each in time.
    `root` is the last committed action (the hand starts from its release), else the
    hand starts from its current pose."""
    flat = []
    for gi, cands in enumerate(groups):
        for a in cands:
            flat.append((gi, a))
    n = len(flat)
    best = [-1.0] * n
    prev = [-2] * n
    for i, (gi, a) in enumerate(flat):
        # from the start
        if root is not None:
            need = transition_time(root.x_end, root.width, a.x0, a.width)
            start_ok = a.t_hit + LEAD - root.t_rel >= need - SLACK
            cost = need
        else:
            travel = max(abs(a.x0 - x_now) / V_TRAVEL, abs(a.width - w_now) / W_RATE)
            need = travel + t_desc(HOVER_MIN) + (0.0 if lifted else t_lift(HOVER_MIN))
            start_ok = a.t_hit + LEAD - t_now >= need - SLACK
            cost = travel
        if start_ok:
            best[i] = a.value - MOVE_PENALTY * cost
            prev[i] = -1
        for j in range(i):
            gj, b = flat[j]
            if gj >= gi or best[j] < 0:
                continue
            need = transition_time(b.x_end, b.width, a.x0, a.width)
            if a.t_hit + LEAD - b.t_rel >= need - SLACK:
                v = best[j] + a.value - MOVE_PENALTY * need
                if v > best[i] + 1e-9:
                    best[i] = v
                    prev[i] = j
    if n == 0:
        return []
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
            b = Action(notes, self.api, a.x0, a.width)
            b.ids = a.ids
            b.kind = a.kind
            b.offset = a.offset
            b.t_hit = a.t_hit
            b.value = a.value
            b.finish(self.api)
            self.active = b
        return self.active

    def plan(self, frame, t_now, X_WT_cmd, width_cmd):
        api = self.api
        k0 = api.current_keyframe(t_now, X_WT_cmd, width_cmd)
        try:
            return self._plan(frame, t_now, k0)
        except Exception:
            self.err_count += 1
            if self.err_count > 3 or (DEBUG and self.debug_done):
                raise
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
                    self.log.append("SKIP %s L%d %.2f" % (tr.kind[0], tr.lane, tr.hit))
            if t_now > DEBUG_T and not self.debug_done:
                self.debug_done = True
                raise RuntimeError(" | ".join(self.log))

        active = self.refresh_active(t_now)
        in_progress = active is not None and t_now < active.t_rel + 0.03
        exclude = set(active.ids) if active else set()
        groups = build_actions(tracks, api, t_now, exclude)
        lifted = k0.z > api.contact_z + 0.004
        seq = schedule(groups, active, t_now, k0.x, k0.width, lifted)

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

        for idx, a in enumerate(seq):
            t_touch = a.t_hit + LEAD
            is_active = (idx == 0 and in_progress)
            if is_active and t_now >= self.active_touch_t - 0.005:
                pass                                   # already pressing: no approach
            elif is_active:
                # approaching the committed action: keep the hover we are at
                h = float(min(HOVER_MAX, max(HOVER_MIN, k0.z - api.screen_z)))
                kfs.append(self.kf(max(t_touch, t_now + 0.01), a.x0, api.touch_z, a.width, True))
            else:
                avail = t_touch - cur_t
                travel = max(abs(a.x0 - cur_x) / V_TRAVEL, abs(a.width - cur_w) / W_RATE)
                if idx == 0 and not cur_down:
                    h = float(min(HOVER_MAX, max(HOVER_MIN, k0.z - api.screen_z)))
                    tl = 0.0
                else:
                    h = choose_hover(avail, travel)
                    tl = t_lift(h)
                hz = api.screen_z + h
                if tl > 0 and cur_t + tl > t_now + 0.005:
                    kfs.append(self.kf(cur_t + tl, cur_x, hz, cur_w))
                t_arrive = max(t_touch - t_desc(h), cur_t + tl + travel)
                if t_arrive > t_now + 0.01 and t_arrive < t_touch - 0.01:
                    kfs.append(self.kf(t_arrive, a.x0, hz, a.width))
                kfs.append(self.kf(max(t_touch, t_now + 0.01), a.x0, api.touch_z, a.width, True))
                if idx == first_new and t_now >= t_touch - t_desc(h) - 0.02:
                    # descent has started: commit
                    self.active = a
                    self.active_touch_t = t_touch
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
            x_end = a.x_end
            kfs.append(self.kf(max(t_release, t_now + 0.02), x_end, api.touch_z, a.width, True))
            cur_t, cur_x, cur_w = max(t_release, t_now + 0.02), x_end, a.width
            cur_down = True
        # park: lift after the last action
        kfs.append(self.kf(max(cur_t + t_lift(HOVER_MAX), t_now + 0.05), cur_x, self.hover_z, cur_w))
        return api.make_plan(kfs, t_now)


def make_policy(api):
    return Policy(api)


# ----------------------------------------------------------------------------- probe
def main(examples):
    # split into charts (t restarts)
    charts = []
    cur = []
    last_t = -1
    for e in examples:
        if e["t"] < last_t:
            charts.append(cur)
            cur = []
        cur.append(e)
        last_t = e["t"]
    charts.append(cur)
    print("charts", len(charts), [len(c) for c in charts])
    for ci, ch in enumerate(charts):
        tracker = Tracker()
        ghosts = []
        for e in ch:
            dets = perceive(e["frame"], e["t"])
            tracker.update(dets, e["t"])
            for tr in tracker.tracks.values():
                if tr.n_seen == 2:
                    # verify against truth: any note of same kind/lane within 0.08 s?
                    ok = any(n["kind"] == tr.kind and n["lane"] == tr.lane and abs(n["time"] - tr.hit) < 0.08
                             for n in e["notes"])
                    if not ok:
                        ghosts.append((e["t"], tr.kind, tr.lane, round(tr.hit, 2), round(tr.end, 2)))
            if ci == 5 and 9.6 <= e["t"] <= 10.4:
                print("  t %.1f live:" % e["t"], [(tr.kind[0], tr.lane, round(tr.hit, 2), round(tr.end, 2), tr.end_lb, tr.n_seen)
                                                   for tr in tracker.live(e["t"])])
        print("chart", ci, "ghost tracks:", ghosts)
