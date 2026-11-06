"""Two-fingertip scheduler for the rhythm-game arm.

Strategy
--------
The hand is treated as two independent fingertips, L (-x) and R (+x), that
share a gripper: their horizontal distance must stay within the tip-gap range
and their height difference is limited by how far the hand may roll.

* Every note that becomes visible is greedily assigned to one fingertip (or
  skipped) as a *touch segment* [start, end] with a lane-centre function.
  A candidate assignment is accepted only if that fingertip has time to lift,
  travel and descend between its neighbouring touches, and if the other
  fingertip's overlapping touches are geometrically compatible (right side,
  gap within range, enough gap to roll one tip clear of the screen while the
  other presses).  Chords are tried as an L/R pair first.  Small lateral
  offsets inside the judge's half-lane tolerance are used to make tight
  pairs feasible (e.g. a tap next to a held lane).
* Each tick the per-finger timelines are turned into a trajectory: a touch
  segment pins the tip at touch_z on the (moving) lane; between touches the
  tip follows a lift / travel / descend "hop" whose height adapts to the
  available time.  A free fingertip rides along inside the band allowed by
  the busy one, and is kept clearly above the contact height whenever it
  has a tap coming soon (so that touch is fresh).
* The two per-finger poses are converted into one keyframe (ref = -x tip,
  width from the 3-D tip distance, rise = height difference).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from rhythmbot.cap.api import Keyframe, Plan, TAP, HOLD, SLIDE

INF = float("inf")
L, R = 0, 1
SIGN = (-1.0, 1.0)
FREE, DESC, TOUCH = 0, 1, 2


class Seg:
    """A touch of one fingertip: pressed from s to e following a lane."""

    __slots__ = ("pol", "finger", "s", "e", "kind", "xoff", "xconst", "note", "hit")

    def __init__(self, pol, finger, s, e, kind, note=None, xconst=0.0):
        self.pol = pol
        self.finger = finger
        self.s = float(s)
        self.e = float(e)
        self.kind = kind
        self.xoff = 0.0
        self.xconst = float(xconst)
        self.note = note
        self.hit = float(note.hit_time) if note is not None else float(s)

    # lane centre (world x) the tip should be on at time tau, without offset
    def center(self, tau: float) -> float:
        n = self.note
        if n is None or self.kind != SLIDE:
            return self.xconst
        off = tau - n.hit_time + self.pol.slide_lead
        dur = max(float(n.duration), 0.0)
        if off < 0.0:
            off = 0.0
        elif off > dur:
            off = dur
        try:
            return self.pol.api.lane_x(float(n.lane_at_offset(off)))
        except Exception:
            return self.xconst

    def x_at(self, tau: float) -> float:
        return self.center(tau) + self.xoff

    def e_eff(self, t_now: float) -> float:
        return self.e if self.e < INF else t_now + 10.0

    def refresh(self):
        """Re-read the end time (long notes grow while their end is off-screen)."""
        n = self.note
        if n is None:
            return
        if self.kind == TAP:
            self.e = max(n.hit_time + self.pol.dwell, self.s + 0.04)
        else:
            fin = getattr(n, "duration_final", True)
            self.e = float(n.end_time) if fin else INF

    def freeze(self, t_now: float):
        """The note is gone: keep a constant-x record of where the tip was."""
        e = min(self.e, t_now)
        self.xconst = self.x_at(e)
        self.xoff = 0.0
        self.e = e
        self.note = None
        self.kind = TAP


class State:
    """A partial assignment in the beam search."""

    __slots__ = ("segs", "offs", "played", "cost", "asg")

    def __init__(self, segs, offs, played, cost, asg):
        self.segs = segs
        self.offs = offs
        self.played = played
        self.cost = cost
        self.asg = asg

    def child(self):
        return State([list(self.segs[0]), list(self.segs[1])], dict(self.offs),
                     self.played, self.cost, dict(self.asg))


class Policy:
    def __init__(self, api):
        self.api = api
        lim = api.limits
        self.v = float(lim["travel_speed"])   # speed assumed when judging feasibility
        self.v_cmd = 1.4                      # speed the x command is allowed to move at
        self.drag_after = 0.015               # a pressed tap tip may start sliding away this long after the hit
        self.dt = float(lim["tick_dt"])
        self.x_mid = 0.5 * (api.lane_x(0.0) + api.lane_x(float(api.n_lanes - 1)))
        self.lock_ahead = 0.25    # assignments starting sooner than this are frozen
        self.late_max = 0.11      # a touch may start this late (judge window is 140 ms)
        self.beam_width = 4
        self.max_groups = 12
        self.hover = api.screen_z + float(lim["hover_height"])
        self.touch = float(api.touch_z)
        self.contact = float(api.contact_z)
        self.lr = (self.hover - self.touch) / float(lim["lift_time"])      # lift rate m/s
        self.dr = (self.hover - self.touch) / float(lim["descend_time"])   # descend rate m/s
        self.gap_min = float(api.tip_gap(0.0))
        self.gap_max = float(api.tip_gap(api.max_width))
        self.z_fresh = api.screen_z + 0.010          # a lifted tip stays at least here
        self.sin_max = math.sin(math.radians(20.0))  # roll limit
        self.gap_fresh = (self.z_fresh - self.touch) / self.sin_max
        self.tol = 0.019          # lateral offset we allow inside the half-lane tolerance
        self.lead = 0.0           # command reaches touch_z at hit - lead
        self.dwell = 0.04         # a tap stays pressed this long after the hit
        self.t_hop = 0.11         # minimum lift+descend time between two touches
        self.fresh_window = 0.30  # keep a tip lifted this long before its tap
        self.slide_lead = 0.04    # look-ahead on slide lane tracking
        self.horizon = 0.5
        self.v_lim = 1.5          # rate limit for a free tip (m/s)
        self.assign: Dict[object, Optional[Seg]] = {}
        self.recent: List[Seg] = []
        self.segs: List[List[Seg]] = [[], []]
        self.cur: List[Tuple[float, float]] = [(0.0, self.hover), (self.gap_min, self.hover)]
        self.t_now = 0.0

    # ------------------------------------------------------------------ tick
    def plan(self, notes, t_now, X_WT_cmd, width_cmd):
        api = self.api
        t_now = float(t_now)
        self.t_now = t_now
        k0 = api.current_keyframe(t_now, X_WT_cmd, width_cmd)
        tl, tr = api.fingertips(k0)
        self.cur = [(float(tl[0]), float(tl[2])), (float(tr[0]), float(tr[2]))]

        tracked = {n.key: n for n in notes}
        for key in list(self.assign):
            if key not in tracked:
                seg = self.assign.pop(key)
                if seg is not None:
                    seg.freeze(t_now)
                    self.recent.append(seg)
        self.recent = [s for s in self.recent if s.e > t_now - 0.6]
        for seg in self.assign.values():
            if seg is not None:
                seg.refresh()
        self._rebuild_segs()
        self._assign_new(notes)
        kfs = self._build(k0)
        return api.make_plan(kfs, t_now)

    def _rebuild_segs(self):
        segs: List[List[Seg]] = [[], []]
        for seg in self.assign.values():
            if seg is not None:
                segs[seg.finger].append(seg)
        for seg in self.recent:
            segs[seg.finger].append(seg)
        for f in (L, R):
            segs[f].sort(key=lambda s: s.s)
        self.segs = segs

    # ------------------------------------------------------------ assignment
    def _make_seg(self, n, f) -> Seg:
        api = self.api
        s = float(n.hit_time) - self.lead
        kind = n.kind
        if kind == TAP:
            e = float(n.hit_time) + self.dwell
            xc = api.lane_x(float(n.lane))
        else:
            fin = getattr(n, "duration_final", True)
            e = float(n.end_time) if fin else INF
            if kind == SLIDE:
                try:
                    xc = api.lane_x(float(n.lane_at_offset(0.0)))
                except Exception:
                    xc = api.lane_x(float(n.lane))
            else:
                xc = api.lane_x(float(n.lane))
        return Seg(self, f, s, e, kind, n, xc)

    # Each tick, notes that start later than `lock_ahead` are re-assigned by
    # a beam search over (L, R, skip) choices in hit order; notes about to be
    # played keep their assignment so the trajectory stays stable.
    def _assign_new(self, notes):
        t_now = self.t_now
        tracked = {n.key: n for n in notes}
        self._visible = sorted(notes, key=lambda n: float(n.hit_time))
        lock_t = t_now + self.lock_ahead
        locked: List[List[Seg]] = [[], []]
        locked_keys = set()
        prev_finger: Dict[object, int] = {}
        for key, seg in self.assign.items():
            if seg is not None:
                prev_finger[key] = seg.finger
                if seg.s <= lock_t:
                    locked[seg.finger].append(seg)
                    locked_keys.add(key)
            else:
                n = tracked.get(key)
                if n is None or float(n.hit_time) - self.lead <= lock_t:
                    locked_keys.add(key)
        for seg in self.recent:
            locked[seg.finger].append(seg)
        for f in (L, R):
            locked[f].sort(key=lambda s: s.s)

        cands = [n for n in notes if n.key not in locked_keys and float(n.hit_time) - self.lead > lock_t]
        cands.sort(key=lambda n: (float(n.hit_time), float(n.lane)))
        groups = []
        i = 0
        while i < len(cands) and len(groups) < self.max_groups:
            n = cands[i]
            if i + 1 < len(cands) and abs(float(cands[i + 1].hit_time) - float(n.hit_time)) < 0.02:
                groups.append((n, cands[i + 1]))
                i += 2
            else:
                groups.append((n,))
                i += 1

        beam = [State(locked, {}, 0, 0.0, {})]
        for group in groups:
            new: List[State] = []
            for st in beam:
                self._expand(st, group, prev_finger, new)
            new.sort(key=lambda s: (-s.played, s.cost))
            beam = new[: self.beam_width]
        best = beam[0]

        # finalize
        self.assign = {k: v for k, v in self.assign.items() if k in locked_keys}
        for g in groups:
            for n in g:
                seg = best.asg.get(n.key)
                self.assign[n.key] = seg
                try:
                    n.skipped = seg is None
                except Exception:
                    pass
        for seg, off in best.offs.items():
            seg.xoff = off
        self._rebuild_segs()

    def _expand(self, st, group, prev_finger, out):
        if len(group) == 1:
            n = group[0]
            for f in (L, R):
                c = self._try(st, n, f, prev_finger)
                if c is not None:
                    out.append(c)
            out.append(st.child())
            return
        a, b = group
        if self.api.lane_x(float(a.lane)) > self.api.lane_x(float(b.lane)):
            a, b = b, a
        c = self._try(st, a, L, prev_finger)
        if c is not None:
            c2 = self._try(c, b, R, prev_finger)
            if c2 is not None:
                out.append(c2)
        for n in (a, b):
            for f in (L, R):
                c = self._try(st, n, f, prev_finger)
                if c is not None:
                    out.append(c)
        out.append(st.child())

    def _try(self, st, n, f, prev_finger):
        seg = self._make_seg(n, f)
        offs = self._fits(seg, st.segs[f], st.segs[1 - f], st.offs)
        if offs is None:
            return None
        c = st.child()
        own = c.segs[f]
        own.append(seg)
        own.sort(key=lambda s: s.s)
        c.offs.update(offs)
        c.played += 1
        c.cost += self._cost(seg, own, prev_finger.get(n.key))
        c.asg[n.key] = seg
        return c

    def _t_settle(self, d: float) -> float:
        """How long before touchdown the x command must have arrived after a
        move of `d` metres: the lagging arm's error scales with the move."""
        return 0.10 + 0.4 * d

    def _travel_start(self, o: Seg, oe: float) -> float:
        """When a fingertip finishing touch `o` can start moving sideways: a
        tap tip may drag away shortly after its hit, long notes only at the end."""
        if o.kind == TAP and o.note is not None:
            return min(oe, max(o.s, o.hit) + self.drag_after)
        return oe

    # Return the dict of lateral offsets needed ({seg: xoff}) or None if the
    # touch cannot be played by its fingertip.  May delay seg.s (a late but
    # still judged touch) to fit after the finger's previous touch.
    def _fits(self, seg: Seg, own, other, offs):
        f = seg.finger
        sg = SIGN[f]
        t_now = self.t_now
        hit = seg.hit
        n = seg.note
        if seg.kind == TAP:
            late_max = self.late_max
        else:
            dur = float(n.duration) if n is not None else 0.0
            late_max = min(self.late_max, max(0.12 * dur, 0.0))
        s0 = hit - self.lead
        xs = seg.center(s0)

        # -- timing against this fingertip's other touches
        s_min = s0
        prev_recent = False
        later = []
        for o in own:
            oe = o.e_eff(t_now)
            if o.s >= hit:
                later.append(o)
                continue
            if oe > hit + late_max:
                return None
            d = abs(xs - o.center(oe))
            t_tr = self._travel_start(o, oe)
            s_min = max(s_min, oe + self.t_hop, t_tr + d / self.v + self._t_settle(d))
            if oe >= t_now:
                prev_recent = True
        if not prev_recent:
            xc, zc = self.cur[f]
            lift = 0.0
            if zc < self.z_fresh:
                lift = (self.z_fresh - zc) / self.lr + 0.02
            d = abs(xs - xc)
            need = t_now + max(lift + 0.07, d / self.v + self._t_settle(d))
            s_min = max(s_min, need)
        if s_min > hit + late_max:
            return None
        s = s_min
        seg.s = s
        if seg.kind == TAP:
            seg.e = max(hit + self.dwell, s + 0.04)
        elif seg.e < INF and seg.e < s + 0.02:
            seg.e = s + 0.02
        e = seg.e_eff(t_now)
        t_tr = self._travel_start(seg, e)
        for o in later:
            d = abs(o.center(o.s) - seg.center(e))
            if o.s < max(e + self.t_hop, t_tr + d / self.v + self._t_settle(d)):
                return None

        # -- geometry against the other fingertip's touches
        # `outs` holds each segment's lateral offset measured *outward* from
        # the pair (positive widens the L-R distance); xoff = SIGN * outward.
        outs: Dict[Seg, float] = {seg: 0.0}
        tol = self.tol

        def out_of(sg_: Seg) -> float:
            if sg_ in outs:
                return outs[sg_]
            return SIGN[sg_.finger] * offs.get(sg_, sg_.xoff)

        def require(o: Seg, d_raw: float, lo: float, hi: float, o_free: bool) -> bool:
            oF, oG = outs[seg], out_of(o)
            d = d_raw + oF + oG
            if d < lo:
                need = lo - d + 0.002
                addF = min(need, max(tol - oF, 0.0))
                rest = need - addF
                if rest > 1e-9:
                    if not o_free or rest > max(tol - oG, 0.0):
                        return False
                    oG += rest
                oF += addF
            elif d > hi:
                need = d - hi + 0.002
                subF = min(need, max(oF + tol, 0.0))
                rest = need - subF
                if rest > 1e-9:
                    if not o_free or rest > max(oG + tol, 0.0):
                        return False
                    oG -= rest
                oF -= subF
            outs[seg] = oF
            if o_free and abs(oG - out_of(o)) > 1e-12:
                outs[o] = oG
            return True

        for o in other:
            oe = o.e_eff(t_now)
            if not (o.s - 0.3 < e + 0.15 and s - 0.3 < oe + 0.15):
                continue
            sim = abs(s - o.s) < 0.03
            overlap = (s - 0.12 < oe) and (o.s - 0.12 < e)
            o_free = o.s > t_now + 0.05 and o.note is not None
            if sim or overlap:
                lo = max(s - 0.12, o.s - 0.12)
                hi = min(e, oe)
                if hi - lo > 2.0:
                    hi = lo + 2.0
                taus = [lo + (hi - lo) * k / 4.0 for k in range(5)]
                need_lo = self.gap_min if sim else self.gap_fresh
                for tau in taus:
                    d_raw = sg * (seg.center(max(tau, s)) - o.center(min(max(tau, o.s), oe)))
                    if not require(o, d_raw, need_lo, self.gap_max, o_free):
                        return None
            else:
                if oe <= s:
                    tau_f, tau_o = s, oe
                    slack = max(0.0, s - 0.10 - oe)
                else:
                    tau_f, tau_o = e, o.s
                    slack = max(0.0, o.s - 0.10 - e)
                d_raw = sg * (seg.center(tau_f) - o.center(tau_o))
                travel = self.v * slack
                if not require(o, d_raw, self.gap_min - travel, self.gap_max + travel, o_free):
                    return None
        return {sg_: SIGN[sg_.finger] * val for sg_, val in outs.items()}

    def _cost(self, seg: Seg, own, prev_f) -> float:
        """Tie-break cost of playing `seg` with its finger (lower is better)."""
        f = seg.finger
        sg = SIGN[f]
        t_now = self.t_now
        s, e = seg.s, seg.e_eff(t_now)
        xs = seg.center(s)
        prev = None
        for o in own:
            if o is seg:
                continue
            if o.e_eff(t_now) <= s and (prev is None or o.e > prev.e):
                prev = o
        if prev is not None:
            cost = abs(xs - prev.center(prev.e_eff(t_now)))
            if s - prev.e < 0.25:
                cost += 0.02
        else:
            cost = abs(xs - self.cur[f][0])
        # lateness
        cost += 0.3 * max(0.0, s - (seg.hit - self.lead))
        # natural orientation: L for the left half of the screen, R for the right
        side = sg * (xs - self.x_mid)
        if side < -0.03:
            cost += 0.04 if seg.kind != TAP else 0.015
        # hysteresis: keep last tick's choice when nothing else distinguishes
        if prev_f == f:
            cost -= 0.012
        if seg.kind != TAP:
            api = self.api
            for m in self._visible:
                hm = float(m.hit_time)
                if m is seg.note or hm <= s or hm >= min(e, s + 3.0):
                    continue
                xm = api.lane_x(float(m.lane))
                rel = sg * (seg.center(hm) - xm)   # +: note on the partner's side
                if rel >= self.gap_fresh - self.tol:
                    cost -= 0.02
                else:
                    cost += 0.06
        return cost

    # ------------------------------------------------------------ trajectories
    def _hop(self, t0, x0, z0, s, x1):
        """Lift / travel / descend profile from (x0, z0) at t0 to touch at
        (x1, touch) at time s (s None: just lift to hover and wait)."""
        if s is None:
            zp = max(self.hover, z0)
            tu = max(0.0, (zp - z0) / self.lr)
            return dict(t0=t0, x0=x0, z0=z0, s=None, x1=x0, zp=zp, tu=tu, td=0.0, t_arr=t0)
        T = max(s - t0, 1e-3)
        zp = (T + z0 / self.lr + self.touch / self.dr) / (1.0 / self.lr + 1.0 / self.dr)
        zp = min(zp, self.hover)
        if zp <= z0:
            zp = z0
            tu = 0.0
            td = min(T, max(zp - self.touch, 0.0) / self.dr)
        else:
            tu = (zp - z0) / self.lr
            td = max(zp - self.touch, 0.0) / self.dr
            if tu + td > T:
                td = max(T - tu, 0.0)
        # x travels early: it should be commanded in place well before the
        # touchdown so the (lagging) arm has settled in the lane.
        d = abs(x1 - x0)
        t_arr = max(t0 + d / self.v_cmd, s - self._t_settle(d))
        if t_arr > s:
            t_arr = max(s, t0 + 1e-3)
        return dict(t0=t0, x0=x0, z0=z0, s=s, x1=x1, zp=zp, tu=tu, td=td, t_arr=t_arr)

    def _hop_at(self, h, tau):
        t0 = h["t0"]
        if h["s"] is None:
            z = h["z0"] + self.lr * (tau - t0)
            if z > h["zp"]:
                z = h["zp"]
            return h["x0"], z, FREE
        s, td = h["s"], h["td"]
        if tau < t0 + h["tu"]:
            z = h["z0"] + self.lr * (tau - t0)
        elif tau < s - td:
            z = h["zp"]
        else:
            z = h["zp"] - (h["zp"] - self.touch) * (tau - (s - td)) / max(td, 1e-6)
            z = max(z, self.touch)
        if h["t_arr"] > t0 + 1e-9 and tau < h["t_arr"]:
            x = h["x0"] + (h["x1"] - h["x0"]) * (tau - t0) / (h["t_arr"] - t0)
        else:
            x = h["x1"]
        state = DESC if tau >= s - td else FREE
        return x, z, state

    def _profile(self, f):
        """Pieces covering [t_now, t_now + horizon] for fingertip f."""
        t_now = self.t_now
        pieces = []
        t = t_now
        x, z = self.cur[f]
        end = t_now + self.horizon
        segs = [s for s in self.segs[f] if s.e_eff(t_now) > t_now]
        for i, seg in enumerate(segs):
            ee = seg.e_eff(t_now)
            if seg.s > t:
                h = self._hop(t, x, z, seg.s, seg.x_at(seg.s))
                pieces.append(("hop", h, t, seg.s, None))
            # a tap tip may drag toward the next touch while still pressed
            drag = None
            if seg.kind == TAP and seg.note is not None and i + 1 < len(segs):
                nxt = segs[i + 1]
                t_d0 = max(max(seg.s, seg.hit) + self.drag_after, t)
                if t_d0 < ee:
                    drag = (t_d0, nxt.x_at(nxt.s))
            pieces.append(("touch", seg, max(seg.s, t), ee, drag))
            t = ee
            x, z = self._touch_x(seg, drag, ee), self.touch
            if t > end:
                break
        if t <= end:
            pieces.append(("hop", self._hop(t, x, z, None, x), t, INF, None))
        return pieces

    def _touch_x(self, seg, drag, tau):
        x = seg.x_at(tau)
        if drag is not None and tau > drag[0]:
            dx = drag[1] - x
            step = min(abs(dx), self.v_cmd * (tau - drag[0]))
            x += math.copysign(step, dx) if dx != 0.0 else 0.0
        return x

    def _state_at(self, pieces, tau):
        for kind, obj, a, b, drag in pieces:
            if tau < b or b == INF:
                if kind == "touch":
                    return self._touch_x(obj, drag, tau), self.touch, TOUCH, None
                x, z, st = self._hop_at(obj, tau)
                return x, z, st, obj["s"]
        kind, obj, a, b, drag = pieces[-1]
        if kind == "touch":
            return self._touch_x(obj, drag, tau), self.touch, TOUCH, None
        x, z, st = self._hop_at(obj, tau)
        return x, z, st, obj["s"]

    def _sample_times(self, profiles):
        t_now = self.t_now
        end = t_now + self.horizon
        ts = set()
        k = 1
        while t_now + 0.05 * k <= end + 1e-9:
            ts.add(t_now + 0.05 * k)
            k += 1
        ts.add(t_now + self.dt)
        for pieces in profiles:
            for kind, obj, a, b, drag in pieces:
                for tb in (a, b):
                    if t_now < tb <= end:
                        ts.add(tb)
                if drag is not None:
                    tb = drag[0]
                    if t_now < tb <= end:
                        ts.add(tb)
                    tb = drag[0] + abs(drag[1] - obj.x_at(drag[0])) / self.v_cmd
                    if t_now < tb <= min(b, end):
                        ts.add(tb)
                if kind == "hop":
                    for key in ("tu", "td", "t_arr"):
                        if key == "tu":
                            tb = obj["t0"] + obj["tu"]
                        elif key == "td":
                            tb = (obj["s"] - obj["td"]) if obj["s"] is not None else -1.0
                        else:
                            tb = obj["t_arr"]
                        if t_now < tb <= end:
                            ts.add(tb)
                elif obj.kind == SLIDE:
                    tb = max(a, t_now) + 0.04
                    while tb < min(b, end):
                        ts.add(tb)
                        tb += 0.04
        out = []
        for tb in sorted(ts):
            if tb <= t_now + 0.004:
                continue
            if out and tb - out[-1] < 0.004:
                continue
            out.append(tb)
        return out

    def _combine(self, tau, SL, SR):
        xl, zl, stl, nl = SL
        xr, zr, str_, nr = SR
        gmin, gmax, smax = self.gap_min, self.gap_max, self.sin_max
        cl = stl != FREE
        cr = str_ != FREE
        if cl and cr:
            dx = xr - xl
            if dx < gmin:
                sh = 0.5 * (gmin - dx)
                xl -= sh
                xr += sh
            elif dx > gmax:
                sh = 0.5 * (dx - gmax)
                xl += sh
                xr -= sh
        elif cl or cr:
            if cl:
                xB, zB, xF, zF, sg, nF = xl, zl, xr, zr, 1.0, nr
            else:
                xB, zB, xF, zF, sg, nF = xr, zr, xl, zl, -1.0, nl
            rel = sg * (xF - xB)
            rel = min(max(rel, gmin), gmax)
            fresh = nF is not None and nF - tau < self.fresh_window
            if fresh:
                if zF < self.z_fresh:
                    zF = self.z_fresh
                need = abs(zF - zB) / smax
                if need > rel:
                    rel = min(need, gmax)
            lim = rel * smax
            dz = zF - zB
            if dz > lim:
                zF = zB + lim
            elif dz < -lim:
                zF = zB - lim
            xF = xB + sg * rel
            if cl:
                xr, zr = xF, zF
            else:
                xl, zl = xF, zF
        else:
            uL = nl if nl is not None else INF
            uR = nr if nr is not None else INF
            dx = xr - xl
            if dx < gmin or dx > gmax:
                target = min(max(dx, gmin), gmax)
                if uL <= uR:
                    xr = xl + target
                else:
                    xl = xr - target
                dx = target
            dz = zr - zl
            if abs(dz) > dx * smax:
                need = min(abs(dz) / smax, gmax)
                if need > dx:
                    if uL <= uR:
                        xr = xl + need
                    else:
                        xl = xr - need
                    dx = need
                lim = dx * smax
                if abs(dz) > lim:
                    if zr < zl:
                        zr = zl - lim
                    else:
                        zl = zr - lim
        return xl, zl, xr, zr

    def _kf(self, t, xl, zl, xr, zr) -> Keyframe:
        api = self.api
        dx = xr - xl
        if dx < 0.5 * self.gap_min:
            dx = 0.5 * self.gap_min
        dz = zr - zl
        gap = math.hypot(dx, dz)
        if gap < self.gap_min:
            gap = self.gap_min
        elif gap > self.gap_max:
            gap = self.gap_max
        if abs(dz) > 0.9 * gap:
            dz = math.copysign(0.9 * gap, dz)
        w = api.width_for_tip_gap(gap)
        if w is None:
            w = 0.0 if gap <= self.gap_min + 1e-6 else api.max_width
        w = min(max(float(w), 0.0), float(api.max_width))
        down = zl <= self.contact or zr <= self.contact
        return api.keyframe(t, xl, zl, w, down, -1, dz)

    def _build(self, k0) -> List[Keyframe]:
        profiles = [self._profile(L), self._profile(R)]
        times = self._sample_times(profiles)
        kfs = [k0]
        px = [self.cur[L][0], self.cur[R][0]]
        pt = self.t_now
        for tau in times:
            SL = self._state_at(profiles[L], tau)
            SR = self._state_at(profiles[R], tau)
            xl, zl, xr, zr = self._combine(tau, SL, SR)
            # rate-limit free tips so the command never jumps
            dtk = max(tau - pt, 1e-3)
            if SL[2] == FREE:
                lim = self.v_lim * dtk
                xl = min(max(xl, px[L] - lim), px[L] + lim)
            if SR[2] == FREE:
                lim = self.v_lim * dtk
                xr = min(max(xr, px[R] - lim), px[R] + lim)
            if xr - xl < self.gap_min:
                if SL[2] == FREE and SR[2] != FREE:
                    xl = xr - self.gap_min
                else:
                    xr = xl + self.gap_min
            kfs.append(self._kf(tau, xl, zl, xr, zr))
            px = [xl, xr]
            pt = tau
        return kfs


def make_policy(api):
    return Policy(api)
