"""Camera-driven rhythm-game policy.

Perception: the fixed camera image is rectified (via a homography calibrated
on the training frames) into a (lane, time-to-hit) grid; note pixels are
classified by colour (cyan = tap, green = hold, magenta = slide) and turned
into note detections; detections are tracked across frames so that each note
gets a stable hit-time estimate (median of many frames) even when the hand
later hides it.

Control: a scheduler assigns tracked notes to the two fingertips (closed
gripper by default, opened for chords), skips what cannot be reached in time,
and emits a keyframe trajectory each tick.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from rhythmbot.cap.api import Keyframe, Plan, TAP, HOLD, SLIDE

# --------------------------------------------------------------------------
# Perception
# --------------------------------------------------------------------------

# Homography from image pixel (col, row) -> (lane, time_to_hit), calibrated
# with ./probe.sh on the training frames (lane resid 0.01, time resid 2 ms).
H_PIX_TO_LT = np.array(
    [
        [7.828943e-03, 1.076226e-01, -9.577812e00],
        [-8.405248e-03, 3.765154e-03, 2.226354e00],
        [5.535644e-06, 3.534344e-03, 1.000000e00],
    ]
)

N_LANES = 6
ROWS_PER_LANE = 12          # rectified rows per lane
T_MIN, T_MAX = -0.20, 1.75  # rectified time window (s)
DT_COL = 1.0 / 160.0        # rectified column pitch (s) ~ 1 image pixel
IMG_H, IMG_W = 256, 384
NOTE_W_S = 0.068            # head/tap width in seconds (about 11 px)


@dataclass
class Detection:
    kind: str
    lane: int              # lane of the head (tap/hold/slide start lane)
    hit_tth: float         # time-to-hit of the head centre (s)
    complete: bool         # the head looked complete (not cut by occlusion)
    tail_tth: float = 0.0  # time-to-hit of the body end (hold/slide)
    tail_seen: bool = False  # the tail was inside the visible area
    path: Optional[np.ndarray] = None  # slides: (n,2) rows of (tth, lane)
    head_seen: bool = True


class Perception:
    def __init__(self):
        self.R = N_LANES * ROWS_PER_LANE
        self.C = int(round((T_MAX - T_MIN) / DT_COL))
        lanes = -0.5 + (np.arange(self.R) + 0.5) / ROWS_PER_LANE
        tths = T_MIN + (np.arange(self.C) + 0.5) * DT_COL
        self.col_tth = tths
        self.row_lane = lanes
        L, T = np.meshgrid(lanes, tths, indexing="ij")
        pts = np.stack([L.ravel(), T.ravel(), np.ones(L.size)], 0)
        Hinv = np.linalg.inv(H_PIX_TO_LT)
        p = Hinv @ pts
        u = p[0] / p[2]
        v = p[1] / p[2]
        ui = np.rint(u).astype(int)
        vi = np.rint(v).astype(int)
        self.valid = ((ui >= 0) & (ui < IMG_W) & (vi >= 0) & (vi < IMG_H)).reshape(self.R, self.C)
        self.ui = np.clip(ui, 0, IMG_W - 1).reshape(self.R, self.C)
        self.vi = np.clip(vi, 0, IMG_H - 1).reshape(self.R, self.C)
        # the visible time range per lane (columns that map inside the image)
        self.col_visible = self.valid[:, :].all(0)
        self.head_cols = max(3, int(round(NOTE_W_S / DT_COL)))

    # -- colour classification --------------------------------------------
    def masks(self, frame):
        fr = frame[self.vi, self.ui].astype(np.int16)  # (R, C, 3)
        r, g, b = fr[..., 0], fr[..., 1], fr[..., 2]
        cyan = (g > 140) & (b > 160) & (r < 110)
        green = (g > 150) & (r < 130) & (b < 130)
        mag = (r > 150) & (b > 150) & (g < 130)
        v = self.valid
        return cyan & v, green & v, mag & v

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def runs(flags):
        """Start/end (inclusive) indices of True runs in a 1-D bool array."""
        if not flags.any():
            return []
        f = np.concatenate([[False], flags, [False]]).astype(np.int8)
        d = np.diff(f)
        starts = np.nonzero(d == 1)[0]
        ends = np.nonzero(d == -1)[0] - 1
        return list(zip(starts.tolist(), ends.tolist()))

    def lane_rows(self, lane, half):
        rc = (lane + 0.5) * ROWS_PER_LANE
        r0 = int(round(rc - half))
        r1 = int(round(rc + half))
        return max(0, r0), min(self.R, r1)

    def tth_of_cols(self, c0, c1):
        return float(0.5 * (self.col_tth[c0] + self.col_tth[c1]))

    # -- taps ---------------------------------------------------------------
    def detect_taps(self, cyan):
        out = []
        for lane in range(N_LANES):
            r0, r1 = self.lane_rows(lane, 2.5)
            prof = cyan[r0:r1].mean(0)
            fr0, fr1 = self.lane_rows(lane, 5.0)
            full = cyan[fr0:fr1]
            for c0, c1 in self.runs(prof > 0.5):
                w = c1 - c0 + 1
                if w < 3:
                    continue
                if w > 1.7 * self.head_cols:
                    # merged taps: split into two heads at the ends
                    hw = self.head_cols
                    for a, b in ((c0, c0 + hw - 1), (c1 - hw + 1, c1)):
                        out.append(Detection(TAP, lane, self.tth_of_cols(a, b), False))
                    continue
                complete = (w >= 0.75 * self.head_cols) and bool(full[:, c0:c1 + 1].mean() > 0.6)
                out.append(Detection(TAP, lane, self.tth_of_cols(c0, c1), complete))
        return out

    # -- holds --------------------------------------------------------------
    def detect_holds(self, green):
        out = []
        vis = self.col_visible
        for lane in range(N_LANES):
            r0, r1 = self.lane_rows(lane, 2.5)
            prof = green[r0:r1].mean(0)
            fr0, fr1 = self.lane_rows(lane, 5.0)
            fullprof = green[fr0:fr1].mean(0)
            # columns: low index = small time-to-hit (near the line) = head end
            for c0, c1 in self.runs(prof > 0.5):
                if c1 - c0 + 1 < 3:
                    continue
                # head columns fill the lane (1.0); body columns only ~0.8 of it
                heads = [(a, b) for a, b in self.runs(fullprof[c0:c1 + 1] > 0.9) if b - a + 1 >= 3]
                tail_tth = float(self.col_tth[c1])
                tail_seen = tail_tth < T_VISIBLE - 0.02
                if heads:
                    a, b = heads[0]
                    a += c0
                    b += c0
                    w = b - a + 1
                    head_tth = self.tth_of_cols(a, b) if w <= 1.5 * self.head_cols else float(self.col_tth[a]) + 0.5 * NOTE_W_S
                    complete = w >= 0.75 * self.head_cols
                    out.append(Detection(HOLD, lane, head_tth, complete, tail_tth, tail_seen, None, True))
                else:
                    # body only (head hidden or already on the line)
                    head_tth = float(self.col_tth[c0]) + 0.5 * NOTE_W_S
                    out.append(Detection(HOLD, lane, head_tth, False, tail_tth, tail_seen, None, False))
        return out

    # -- slides -------------------------------------------------------------
    def detect_slides(self, mag):
        out = []
        vis = self.col_visible
        lab, n = ndimage.label(mag)
        if n == 0:
            return out
        objs = ndimage.find_objects(lab)
        for i, sl in enumerate(objs):
            comp = lab[sl] == (i + 1)
            if comp.sum() < 12:
                continue
            rs, cs = sl
            rows = np.arange(rs.start, rs.stop)
            cols = np.arange(cs.start, cs.stop)
            cnt = comp.sum(0).astype(float)           # pixels per column
            if (cnt > 0).sum() < 3:
                continue
            wsum = (comp * rows[:, None]).sum(0)
            mean_row = np.where(cnt > 0, wsum / np.maximum(cnt, 1), np.nan)
            lane_of_col = -0.5 + (mean_row + 0.5) / ROWS_PER_LANE
            # head columns: nearly a full lane tall
            # slide heads span 10-12 rows, bodies at most 7
            full = cnt >= 0.75 * ROWS_PER_LANE
            heads = [(a, b) for a, b in self.runs(full) if b - a + 1 >= 3]
            col_tth = self.col_tth[cols]
            # columns: low index = small time-to-hit (near the line) = head end
            nz = np.nonzero(cnt > 0)[0]
            c_lo, c_hi = int(nz.min()), int(nz.max())
            if not heads:
                # body only: head hidden / on the line
                path = np.c_[col_tth[c_lo:c_hi + 1], lane_of_col[c_lo:c_hi + 1]]
                path = path[~np.isnan(path[:, 1])]
                if len(path) < 2:
                    continue
                lane = int(round(float(path[0, 1])))
                tail_seen = float(col_tth[c_hi]) < T_VISIBLE - 0.02
                out.append(Detection(SLIDE, lane, float(col_tth[c_lo]) + 0.5 * NOTE_W_S, False,
                                     float(col_tth[c_hi]), tail_seen, path, False))
                continue
            # one slide per head; the body of a head extends towards larger
            # time-to-hit until the next head (heads sorted by column)
            for hi, (a, b) in enumerate(heads):
                w = b - a + 1
                head_tth = self.tth_of_cols(cols[a], cols[b]) if w <= 1.5 * self.head_cols else float(col_tth[a]) + 0.5 * NOTE_W_S
                lane = int(round(float(np.nanmean(lane_of_col[a:b + 1]))))
                right = c_hi if hi == len(heads) - 1 else heads[hi + 1][0] - 1
                seg = np.nonzero(cnt[a:right + 1] > 0)[0]
                if len(seg) == 0:
                    continue
                c_end = a + int(seg.max())
                path = np.c_[col_tth[a:c_end + 1], lane_of_col[a:c_end + 1]]
                path = path[~np.isnan(path[:, 1])]
                # the head block itself sits on the start lane
                path[path[:, 0] <= head_tth + 0.5 * NOTE_W_S, 1] = lane
                tail_seen = (hi == len(heads) - 1) and float(col_tth[c_end]) < T_VISIBLE - 0.02
                out.append(Detection(SLIDE, lane, head_tth, w >= 0.75 * self.head_cols,
                                     float(col_tth[c_end]), tail_seen, path, True))
        return out

    def detect(self, frame):
        cyan, green, mag = self.masks(frame)
        return self.detect_taps(cyan) + self.detect_holds(green) + self.detect_slides(mag)


# --------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------

T_VISIBLE = 1.60          # notes enter the screen this long before their hit
HEAD_LIVE_TTH = 0.06      # a head closer than this to the line may be stopped (being played)
MATCH_TOL = 0.05          # s, hit-time tolerance when matching detections to tracks
PATH_BIN = 0.01           # s, slide path sample pitch


def wmedian(vals, weights):
    idx = np.argsort(vals)
    v = np.asarray(vals)[idx]
    w = np.asarray(weights)[idx]
    c = np.cumsum(w)
    return float(v[int(np.searchsorted(c, 0.5 * c[-1]))])


class Track:
    _next_id = 0

    def __init__(self, kind, lane, t, det):
        self.id = Track._next_id
        Track._next_id += 1
        self.kind = kind
        self.lane = lane
        self.hit_vals = []
        self.hit_w = []
        self.hit = t + det.hit_tth
        self.dur_vals = []
        self.dur_lb = 0.0
        self.duration = 0.0
        self.path = {}          # bin(abs time) -> lane (slides)
        self.n_obs = 0
        self.n_complete = 0
        self.first_seen = t
        self.last_seen = t
        self.done = False
        self.skipped = False
        self.add(t, det)

    def add(self, t, det):
        self.n_obs += 1
        self.last_seen = t
        if det.head_seen and det.hit_tth > HEAD_LIVE_TTH:
            w = 1.0 if det.complete else 0.2
            self.hit_vals.append(t + det.hit_tth)
            self.hit_w.append(w)
            if det.complete:
                self.n_complete += 1
            if len(self.hit_vals) > 60:
                self.hit_vals = self.hit_vals[-60:]
                self.hit_w = self.hit_w[-60:]
            self.hit = wmedian(self.hit_vals, self.hit_w)
        if self.kind in (HOLD, SLIDE):
            tail_abs = t + det.tail_tth
            if det.tail_seen:
                self.dur_vals.append(tail_abs - self.hit)
                if len(self.dur_vals) > 40:
                    self.dur_vals = self.dur_vals[-40:]
                self.duration = float(np.median(self.dur_vals))
            else:
                self.dur_lb = max(self.dur_lb, tail_abs - self.hit)
                if not self.dur_vals:
                    self.duration = self.dur_lb
        if self.kind == SLIDE and det.path is not None:
            for tth, lane in det.path:
                b = int(round((t + tth) / PATH_BIN))
                old = self.path.get(b)
                self.path[b] = float(lane) if old is None else 0.5 * (old + float(lane))

    @property
    def end(self):
        return self.hit + self.duration

    @property
    def confirmed(self):
        return self.n_obs >= 2 and len(self.hit_vals) >= 1

    def lane_at(self, t_abs):
        """Lane of a slide at absolute time t_abs (start lane for others)."""
        if self.kind != SLIDE or not self.path:
            return float(self.lane)
        b = (t_abs) / PATH_BIN
        keys = self._keys()
        i = int(np.searchsorted(keys, b))
        if i <= 0:
            return self.path[int(keys[0])]
        if i >= len(keys):
            return self.path[int(keys[-1])]
        k0, k1 = keys[i - 1], keys[i]
        l0, l1 = self.path[int(k0)], self.path[int(k1)]
        if k1 - k0 > 20:  # gap: nearest
            return l0 if b - k0 < k1 - b else l1
        return l0 + (l1 - l0) * (b - k0) / (k1 - k0)

    def _keys(self):
        if getattr(self, "_keys_cache_n", -1) != len(self.path):
            self._keys_cache = np.array(sorted(self.path.keys()), dtype=float)
            self._keys_cache_n = len(self.path)
        return self._keys_cache


class Tracker:
    def __init__(self):
        self.tracks: List[Track] = []

    def update(self, dets, t):
        used = set()
        # 1) head detections: match by kind, lane, hit time
        for d in dets:
            if not (d.head_seen and d.hit_tth > HEAD_LIVE_TTH):
                continue
            hit = t + d.hit_tth
            best, bd = None, MATCH_TOL
            for tr in self.tracks:
                if tr.kind != d.kind or tr.lane != d.lane or tr.id in used:
                    continue
                dd = abs(tr.hit - hit)
                if dd < bd:
                    best, bd = tr, dd
            if best is None:
                # no note can start inside a hold/slide already occupying the lane
                blocked = any(tr.kind != TAP and tr.confirmed and tr.lane == d.lane
                              and tr.hit - 0.05 < hit < tr.end + 0.05 and tr.kind == HOLD
                              for tr in self.tracks)
                if d.hit_tth < T_VISIBLE + 0.1 and not blocked:
                    best = Track(d.kind, d.lane, t, d)
                    self.tracks.append(best)
            else:
                best.add(t, d)
            used.add(best.id)
        # 2) body-only / stopped-head detections: update live hold/slide tracks
        for d in dets:
            if d.head_seen and d.hit_tth > HEAD_LIVE_TTH:
                continue
            if d.kind == TAP:
                continue
            best, bd = None, 1e9
            for tr in self.tracks:
                if tr.kind != d.kind or tr.id in used:
                    continue
                if not (tr.hit - 0.15 < t < tr.end + 0.3 or (tr.hit - 0.15 < t and not tr.dur_vals)):
                    continue
                if d.kind == HOLD and tr.lane != d.lane:
                    continue
                if d.kind == SLIDE and abs(tr.lane_at(t) - d.lane) > 1.2:
                    continue
                dd = abs(tr.hit - t)
                if dd < bd:
                    best, bd = tr, dd
            if best is not None:
                best.add(t, d)
                used.add(best.id)
        # 3) expire
        keep = []
        for tr in self.tracks:
            if t > tr.end + 0.6:
                continue
            if not tr.confirmed and t - tr.first_seen > 0.5:
                continue
            keep.append(tr)
        self.tracks = keep

    def notes(self, t):
        return sorted((tr for tr in self.tracks if tr.confirmed and not tr.done), key=lambda tr: tr.hit)




# --------------------------------------------------------------------------
# Control
# --------------------------------------------------------------------------

HOLD_FRACTION_SAFE = 0.82   # a sustain may be released after this fraction (+margin) of its length


class Event:
    """One primary action of the hand: a single note or a two-note chord."""

    def __init__(self, notes, api):
        self.api = api
        self.notes = notes
        self.key = tuple(sorted(n.id for n in notes))
        self.hit = float(np.mean([n.hit for n in notes]))
        self.end = max(n.end for n in notes)
        self.sustain = any(n.kind != TAP for n in notes)
        self.n_sustain = sum(1 for n in notes if n.kind != TAP)
        if len(notes) == 1:
            self.anchor = notes[0]
            self.gap = 0.0
            self.width = 0.0
            self.hold_tip = 0           # 0: both tips (closed hand) play the anchor
        else:
            lo = min(notes, key=lambda n: n.lane)
            hi = max(notes, key=lambda n: n.lane)
            self.gap = (hi.lane - lo.lane) * api.lane_width
            w = api.width_for_tip_gap(self.gap)
            self.width = float(w) if w is not None else 0.0
            slides = [n for n in notes if n.kind == SLIDE]
            sus = [n for n in notes if n.kind != TAP]
            self.anchor = slides[0] if slides else (sus[0] if sus else lo)
            self.hold_tip = -1 if self.anchor is lo else 1
        self.secondary = []     # [(note, T, x_tip)] taps for the free fingertip
        self.free_side = 0
        self.gap0 = 0.05        # fingertip gap at the touch when a free tip is raised
        if self.sustain:
            dur = self.end - self.hit
            known = all(bool(n.dur_vals) for n in notes if n.kind != TAP)
            # contact starts a little after the hit; keep a margin over the 80% rule
            self.end_eff = min(self.end, self.hit + HOLD_FRACTION_SAFE * dur + 0.07) if known else self.end
        else:
            self.end_eff = self.hit

    def anchor_x(self, t_abs):
        return self.api.lane_x(self.anchor.lane_at(t_abs))

    @property
    def side(self):
        """Fingertip that plays the anchor: -1 / +1, or 0 for the closed hand."""
        if self.hold_tip != 0:
            return self.hold_tip
        return -self.free_side if self.free_side else 0


class Controller:
    def __init__(self, api):
        self.api = api
        lim = api.limits
        self.v_travel = 1.8       # the arm follows this without visible lag
        self.t_desc = 0.20
        self.t_desc_min = 0.08
        self.t_lift = 0.10
        self.t_lift_min = 0.05
        self.hover = float(lim.get("hover_height", 0.035))
        self.hover_low = 0.018
        self.lead = -0.015        # touch keyframe offset (contact begins before it)
        self.late_ok = 0.06       # accept touching this late to fit a note in
        self.dwell = 0.05
        self.slide_lead = 0.04
        # free fingertip during a sustain
        self.rise_up = 0.016
        self.press_extra = 0.003  # extra press depth of the held tip while rolling
        self.sec_pre = 0.12
        self.sec_post = 0.12
        self.sec_dwell = 0.04
        self.sec_spacing = self.sec_pre + self.sec_dwell + self.sec_post + 0.02
        self.touch_z = api.touch_z
        self.contact_z = api.contact_z
        self.screen_z = api.screen_z
        self.half_lane = 0.5 * api.lane_width
        self.commit_key = None
        self.commit_T = 0.0

    # -- keyframe helpers ---------------------------------------------------
    def kf_mid(self, t, x, z, width=0.0, rise=0.0):
        return self.api.keyframe(t, x, z, width, False, 0, rise)

    def kf_tip(self, t, side, x_tip, z_tip, width, rise):
        """Midpoint keyframe that puts the `side` (-1/+1) fingertip at (x_tip, z_tip)."""
        gap = self.api.tip_gap(width)
        s = max(-0.9, min(0.9, rise / gap))
        c = math.sqrt(1.0 - s * s)
        half = 0.5 * gap
        return self.kf_mid(t, x_tip - side * half * c, z_tip - side * half * s, width, rise)

    def width_for(self, gap):
        w = self.api.width_for_tip_gap(gap)
        if w is None:
            w = self.api.max_width
        return float(max(0.0, min(self.api.max_width, w)))

    def touch_pose(self, ev, T):
        """(x_mid, width, rise) of the hand at the touch of `ev` at time T."""
        if ev.hold_tip != 0:
            x_mid = ev.anchor_x(T) - ev.hold_tip * 0.5 * ev.gap
            return x_mid, ev.width, 0.0
        if ev.sustain and ev.free_side:
            w = self.width_for(ev.gap0)
            r = ev.free_side * self.rise_up
            k = self.kf_tip(T, -ev.free_side, ev.anchor_x(T), self.touch_z - self.press_extra, w, r)
            return k.x, w, r
        return ev.anchor_x(T), 0.0, 0.0

    # -- grouping -----------------------------------------------------------
    def group(self, notes):
        events = []
        i = 0
        while i < len(notes):
            grp = [notes[i]]
            j = i + 1
            while j < len(notes) and abs(notes[j].hit - notes[i].hit) < 0.03:
                grp.append(notes[j])
                j += 1
            i = j
            if len(grp) == 1:
                events.append(Event(grp, self.api))
                continue
            best = None
            for a in range(len(grp)):
                for b in range(a + 1, len(grp)):
                    d = abs(grp[a].lane - grp[b].lane)
                    if 1 <= d <= 2:
                        score = (sum(1 for n in (grp[a], grp[b]) if n.kind != TAP), -d)
                        if best is None or score > best[0]:
                            best = (score, [grp[a], grp[b]])
            if best is not None:
                events.append(Event(best[1], self.api))
            else:
                grp.sort(key=lambda n: (n.kind == TAP, n.lane))
                events.append(Event([grp[0]], self.api))
        return events

    # -- secondary taps during a sustain -----------------------------------
    def assign_secondary(self, ev, T, later):
        """Pick taps the free fingertip can play while `ev` is held."""
        if not ev.sustain or ev.n_sustain > 1:
            return
        api = self.api
        cands = []   # (Tk, side, note, end_k)
        for e2 in later:
            for n in e2.notes:
                Tk = n.hit - self.lead
                end_k = Tk + self.sec_dwell if n.kind == TAP else n.end
                if Tk < T + 0.12 or end_k > ev.end_eff - 0.02:
                    continue
                if n.kind != TAP and not n.dur_vals:
                    continue  # duration still unknown
                # the free tip must stay 1-2.5 lanes to one side of the held tip
                ok = True
                sd = 0
                for tt in np.linspace(Tk, end_k, 6 if n.kind == SLIDE else 2):
                    dx = api.lane_x(n.lane_at(tt)) - ev.anchor_x(tt)
                    s = 1 if dx > 0 else -1
                    if not (0.03 <= abs(dx) <= 0.125) or (sd and s != sd):
                        ok = False
                        break
                    sd = s
                if ok:
                    cands.append((Tk, sd, n, end_k))
        if not cands and TEST_PHANTOM and ev.n_sustain == 1 and ev.end_eff - T > 0.7:
            lane = ev.anchor.lane + (1 if ev.anchor.lane < 5 else -1)
            for k, dt in enumerate((0.3, 0.55)):
                ph = _Phantom(1000 + 10 * ev.anchor.id + k, lane, T + dt)
                dx = api.lane_x(lane) - ev.anchor_x(ph.hit)
                cands.append((ph.hit, 1 if dx > 0 else -1, ph, ph.hit + self.sec_dwell))
        if not cands:
            if TEST_STATIC_ROLL and ev.n_sustain == 1 and ev.hold_tip == 0:
                ev.free_side = 1 if ev.anchor.lane < 5 else -1
                ev.gap0 = 0.05
            return
        if ev.hold_tip != 0:
            side = -ev.hold_tip
        else:
            n_pos = sum(1 for c in cands if c[1] > 0)
            n_neg = len(cands) - n_pos
            side = 1 if n_pos >= n_neg else -1
        chosen = []
        last_end = -1e9
        last_gap = None
        for Tk, sd, n, end_k in sorted(cands, key=lambda c: c[0]):
            if sd != side:
                continue
            gap = abs(api.lane_x(n.lane_at(Tk)) - ev.anchor_x(Tk))
            need = self.sec_pre + self.sec_post + 0.02 + (abs(gap - last_gap) / 0.5 if last_gap is not None else 0.0)
            if Tk - last_end < need:
                continue
            chosen.append((n, Tk, end_k))
            last_end = end_k
            last_gap = abs(api.lane_x(n.lane_at(end_k)) - ev.anchor_x(end_k))
        ev.free_side = side
        ev.secondary = chosen
        if DEBUG_SEC:
            for n, Tk, end_k in chosen:
                if (ev.key, n.id) not in _SEEN_SEC:
                    _SEEN_SEC.add((ev.key, n.id))
                    print(f"SEC: {ev.anchor.kind} L{ev.anchor.lane} {ev.hit:.2f}-{ev.end:.2f} side {side} <- {n.kind} L{n.lane} {n.hit:.2f}-{n.end:.2f}")
        if chosen:
            n0 = chosen[0][0]
            ev.gap0 = abs(api.lane_x(n0.lane_at(chosen[0][1])) - ev.anchor_x(T))
        elif TEST_STATIC_ROLL and ev.hold_tip == 0:
            ev.gap0 = 0.05
        else:
            ev.free_side = 0

    # -- keyframes of a sustain ------------------------------------------
    def sustain_kfs(self, ev, T, t_now, release):
        """Keyframes from the touch at T until `release` for a hold/slide event.
        Returns (keyframes, x_mid_last, width_last, rise_last)."""
        kfs = []
        side = ev.side
        if side == 0:
            # closed hand, both tips on the anchor
            if ev.anchor.kind == SLIDE:
                tt = max(T, t_now) + 1.0 / 30.0
                while tt < release:
                    kfs.append(self.kf_mid(tt, ev.anchor_x(tt + self.slide_lead), self.touch_z, 0.0))
                    tt += 1.0 / 30.0
            x_last = ev.anchor_x(release)
            kfs.append(self.kf_mid(release, x_last, self.touch_z, 0.0))
            return kfs, x_last, 0.0, 0.0
        free = -side
        rise_up = free * self.rise_up
        # (time, gap, rise) profile of the free fingertip
        if ev.hold_tip != 0:
            prof = [(T, ev.gap, 0.0)]
            if ev.secondary:
                prof.append((T + self.dwell, ev.gap, 0.0))
                prof.append((T + self.dwell + self.sec_post, ev.gap, rise_up))
        else:
            prof = [(T, ev.gap0, rise_up)]
        z_tip = self.touch_z - self.press_extra
        for n, Tk, end_k in ev.secondary:
            pts = [(Tk - self.sec_pre, rise_up), (Tk, 0.0)]
            if n.kind == SLIDE:
                tt = Tk + 1.0 / 30.0
                while tt < end_k:
                    pts.append((tt, 0.0))
                    tt += 1.0 / 30.0
            pts += [(end_k, 0.0), (end_k + self.sec_post, rise_up)]
            for tt, rz in pts:
                if tt <= T + 1e-3 or tt >= release:
                    continue
                lead_k = self.slide_lead if n.kind == SLIDE else 0.0
                prof.append((tt, abs(self.api.lane_x(n.lane_at(tt + lead_k)) - ev.anchor_x(tt)), rz))
            if t_now >= end_k:
                n.done = True
        prof.sort(key=lambda p: p[0])
        g_last, r_last = prof[-1][1], prof[-1][2]
        prof.append((release, g_last, r_last))
        times = [p[0] for p in prof]
        if ev.anchor.kind == SLIDE:
            tt = T + 1.0 / 30.0
            while tt < release:
                times.append(tt)
                tt += 1.0 / 30.0
        times = sorted(set(round(x, 4) for x in times))
        pt = np.array([p[0] for p in prof])
        pg = np.array([p[1] for p in prof])
        pr = np.array([p[2] for p in prof])
        lead = self.slide_lead if ev.anchor.kind == SLIDE else 0.0
        x_last, w_last, rise_last = None, 0.0, 0.0
        for tt in times:
            if tt <= t_now - 0.05 and tt < release:
                continue
            g = float(np.interp(tt, pt, pg))
            rz = float(np.interp(tt, pt, pr))
            w = self.width_for(g)
            k = self.kf_tip(tt, side, ev.anchor_x(tt + lead), z_tip, w, rz)
            kfs.append(k)
            x_last, w_last, rise_last = k.x, w, rz
        if x_last is None:
            w = self.width_for(g_last)
            k = self.kf_tip(release, side, ev.anchor_x(release), z_tip, w, r_last)
            kfs.append(k)
            x_last, w_last, rise_last = k.x, w, r_last
        return kfs, x_last, w_last, rise_last

    # -- main planner ---------------------------------------------------------
    def plan(self, t, k0, notes):
        api = self.api
        kfs = [k0]
        x_prev = k0.x
        w_prev = k0.width
        down_prev = k0.z < self.contact_z + 0.004
        t_free = t
        events = self.group(notes)
        committed = None
        if self.commit_key is not None:
            for ev in events:
                if ev.key == self.commit_key:
                    committed = ev
                    break
            if committed is None:
                self.commit_key = None
        assigned = set()
        planned = 0
        pending = None   # (ev, T) sustain whose release waits for the next event
        for idx, ev in enumerate(events):
            if planned >= 6:
                break
            if any(n.id in assigned for n in ev.notes):
                continue
            T = ev.hit - self.lead
            is_committed = ev is committed
            if is_committed:
                T = self.commit_T
                if not ev.sustain and t >= T + self.dwell - 1e-3:
                    for n in ev.notes:
                        n.done = True
                    self.commit_key = None
                    continue
            elif T < t - self.late_ok:
                for n in ev.notes:
                    n.done = True
                continue
            if ev.sustain:
                self.assign_secondary(ev, T, events[idx + 1:])
                for n, _, _ in ev.secondary:
                    assigned.add(n.id)
            xs, w_touch, r_touch = self.touch_pose(ev, T)
            if is_committed:
                if T > t + 1e-3:
                    if r_touch == 0.0:
                        kfs.append(self.kf_mid(T, xs, self.touch_z, w_touch, 0.0))
                    else:
                        kfs.append(self.kf_tip(T, ev.side, ev.anchor_x(T), self.touch_z - self.press_extra, w_touch, r_touch))
            else:
                need_lift = (self.t_lift if down_prev else 0.0)
                dist = abs(xs - x_prev)
                travel = dist / self.v_travel
                t_free_eff = t_free
                if pending is not None:
                    t_free_eff = max(pending[0].end_eff, t)
                avail = T - t_free_eff
                need_lift_min = (self.t_lift_min if down_prev else 0.0)
                need_full = need_lift + travel + self.t_desc
                need_min = need_lift_min + travel + self.t_desc_min
                if avail >= need_full:
                    frac = 1.0
                else:
                    avail_late = ev.hit + self.late_ok - t_free_eff
                    if avail_late < need_min:
                        continue  # cannot make it: skip this event
                    T = max(T, t_free_eff + need_min)
                    avail = T - t_free_eff
                    frac = max(0.0, min(1.0, (avail - need_min) / max(1e-6, need_full - need_min)))
                    xs, w_touch, r_touch = self.touch_pose(ev, T)
                # scale the hop between the rushed and the comfortable motion
                hover_z = self.hover_low + frac * (self.hover - self.hover_low)
                t_lift = need_lift_min + frac * (need_lift - need_lift_min)
                t_desc = self.t_desc_min + frac * (self.t_desc - self.t_desc_min)
                if pending is not None:
                    pev, pT = pending
                    release = min(pev.end, max(pev.end_eff, T - t_desc - travel - t_lift))
                    release = max(release, t + 1e-3)
                    skfs, x_prev, w_prev, _ = self.sustain_kfs(pev, pT, t, release)
                    kfs.extend(skfs)
                    if pev is committed and t >= release:
                        for n in pev.notes:
                            n.done = True
                        self.commit_key = None
                    pending = None
                    t_free = max(release, t)
                    dist = abs(xs - x_prev)
                    travel = dist / self.v_travel
                if down_prev:
                    if planned == 0:
                        frac = (self.screen_z + hover_z - k0.z) / max(1e-6, hover_z + (self.screen_z - self.touch_z))
                        t_lift = max(0.02, t_lift * min(1.0, max(0.0, frac)))
                    kfs.append(self.kf_mid(t_free + t_lift, x_prev, self.screen_z + hover_z, w_prev, 0.0))
                z_hover = self.screen_z + hover_z
                if r_touch == 0.0:
                    kfs.append(self.kf_mid(T - t_desc, xs, z_hover, w_touch, 0.0))
                    kfs.append(self.kf_mid(T, xs, self.touch_z, w_touch, 0.0))
                else:
                    kfs.append(self.kf_tip(T - t_desc, ev.side, ev.anchor_x(T), z_hover, w_touch, r_touch))
                    kfs.append(self.kf_tip(T, ev.side, ev.anchor_x(T), self.touch_z - self.press_extra, w_touch, r_touch))
                if planned == 0 and committed is None and T - t_desc <= t + 1.0 / 30.0 + 1e-3:
                    self.commit_key = ev.key
                    self.commit_T = T
                    committed = ev
            if ev.sustain:
                pending = (ev, T)
                t_free = ev.end
                x_prev = xs
                w_prev = w_touch
            else:
                kfs.append(self.kf_mid(T + self.dwell, xs, self.touch_z, w_touch, 0.0))
                t_free = T + self.dwell
                x_prev = xs
                w_prev = w_touch
            down_prev = True
            t_free = max(t_free, t)
            planned += 1
        if pending is not None:
            pev, pT = pending
            release = max(pev.end, t + 1e-3)
            skfs, x_prev, w_prev, _ = self.sustain_kfs(pev, pT, t, release)
            kfs.extend(skfs)
            t_free = max(release, t)
            if pev is committed and t >= pev.end:
                for n in pev.notes:
                    n.done = True
                self.commit_key = None
        if planned == 0:
            if down_prev:
                kfs.append(self.kf_mid(t + 0.1, k0.x, self.screen_z + self.hover, k0.width, 0.0))
            x_rest = api.lane_x(2.5)
            kfs.append(self.kf_mid(t + 0.3 + abs(x_rest - k0.x) / 0.5, x_rest, self.screen_z + self.hover, 0.0, 0.0))
        else:
            kfs.append(self.kf_mid(t_free + self.t_lift, x_prev, self.screen_z + self.hover, w_prev, 0.0))
        return api.make_plan(kfs, t)


class CameraPolicy:
    def __init__(self, api):
        self.api = api
        self.perc = Perception()
        self.tracker = Tracker()
        self.ctrl = Controller(api)

    def plan(self, frame, t_now, X_WT_cmd, width_cmd):
        k0 = self.api.current_keyframe(t_now, X_WT_cmd, width_cmd)
        try:
            dets = self.perc.detect(frame)
            self.tracker.update(dets, t_now)
        except Exception:
            dets = []
        notes = [n for n in self.tracker.notes(t_now)
                 if (n.kind == TAP and n.hit > t_now - 0.12) or (n.kind != TAP and n.end > t_now)]
        try:
            p = self.ctrl.plan(t_now, k0, notes)
        except Exception:
            if not SAFE:
                raise
            # never leave the hand stranded: hold the pose, drift up a little
            self.ctrl.commit_key = None
            z = max(k0.z, self.api.screen_z + 0.02)
            p = self.api.make_plan([k0, self.ctrl.kf_mid(t_now + 0.2, k0.x, z, k0.width, 0.0)], t_now)
        if DEBUG and int(round(t_now * 30)) % 3 == 0:
            print(f"DBG t={t_now:.3f} k0 lane={self.api.x_to_lane(k0.x):.2f} z={(k0.z - self.api.screen_z)*1000:.1f} w={k0.width:.3f} rise={k0.rise*1000:.1f}")
            print("   notes: " + "; ".join(f"{n.kind} L{n.lane} {n.hit:.2f}" for n in notes[:5]))
            def tips(k):
                a, b = self.api.fingertips(k)
                return f"[{self.api.x_to_lane(a[0]):.2f}@{(a[2]-self.api.screen_z)*1000:.0f} {self.api.x_to_lane(b[0]):.2f}@{(b[2]-self.api.screen_z)*1000:.0f}]"
            print("   kfs: " + " | ".join(f"{k.t:.2f}:w{k.width:.3f},r{k.rise*1000:.0f}{tips(k)}" for k in p.keyframes[:7]))
        return p


SAFE = True
DEBUG = False
DEBUG_SEC = False
_SEEN_SEC = set()
TEST_PHANTOM = False
TEST_STATIC_ROLL = False


class _Phantom:
    def __init__(self, id, lane, hit):
        self.id = id
        self.lane = lane
        self.hit = hit
        self.end = hit
        self.kind = TAP
        self.done = False
        self.dur_vals = []

    def lane_at(self, t):
        return float(self.lane)


def make_policy(api):
    return CameraPolicy(api)
