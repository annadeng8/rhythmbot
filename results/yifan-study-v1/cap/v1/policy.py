import math
import numpy as np

DWELL = 0.07
CHORD_DWELL = 0.045
CHORD_SHIFT = 0.10
LAG = 0.015
D_DESC = 0.15
V_UP = 0.40
V_LAT = 0.9
K_MAX = 1.6
STEP = 0.04
LEAD = 0.02
TAP_LATE = 0.125
SUS_LATE = 0.09


class LocalPolicy:
    def __init__(self, api):
        self.api = api
        self.hand = int(api.hand)
        self.hover = float(api.screen_z) + float(getattr(api, "hover_height", 0.035))
        self.press = float(api.touch_z)
        self.contact = float(api.contact_z)
        self.press_t = {}

    def _clip(self, x):
        return min(0.0, max(-0.36, x)) if self.hand == 0 else min(0.36, max(0.0, x))

    def _lane_x(self, lane):
        return (float(lane) + 0.5) * 0.12 - 0.36

    def _kind(self, n):
        return str(getattr(n, "kind", "tap"))

    def _dur(self, n):
        d = getattr(n, "duration", None)
        if d is None:
            d = float(getattr(n, "end_time", n.hit_time)) - float(n.hit_time)
        return max(0.0, float(d))

    def _end(self, n):
        return float(n.hit_time) + self._dur(n)

    def _nid(self, n):
        return (getattr(n, "key", None), round(float(n.hit_time), 2))

    def _lane_at(self, n, off):
        lane = n.lane
        if self._kind(n) == "slide":
            try:
                lane = n.lane_at_offset(min(self._dur(n), max(0.0, off)))
            except Exception:
                lane = n.lane
        return float(lane)

    def _px(self, n, off):
        return self._clip(self._lane_x(self._lane_at(n, off)))

    def _desc_time(self, zu):
        return D_DESC * max(0.0, zu - self.press) / max(1e-6, self.hover - self.press)

    def plan(self, notes, t_now, X_WT_cmd, width_cmd):
        api = self.api
        t_now = float(t_now)
        cur = api.current_keyframe(t_now, X_WT_cmd, width_cmd)
        st = [float(cur.t), float(cur.x), float(cur.z)]
        kfs = [cur]

        def add(tt, xx, zz):
            tt = max(float(tt), st[0] + 1e-3)
            kfs.append(api.keyframe(tt, float(xx), float(zz)))
            st[0], st[1], st[2] = tt, float(xx), float(zz)

        def follow(n, tp):
            hit, dur = float(n.hit_time), self._dur(n)
            o0 = max(0.0, st[0] - hit, tp - hit)
            offs = set(np.arange(o0, dur, STEP).tolist())
            for w in (getattr(n, "waypoints", None) or []):
                try:
                    wo = float(w[0]) - LEAD
                    if o0 < wo < dur:
                        offs.add(wo)
                except Exception:
                    pass
            offs.add(dur)
            for o in sorted(offs):
                add(hit + o, self._px(n, o + LEAD), self.press)
            add(hit + dur + 0.04, self._px(n, dur), self.press)

        def trans(th, xb, late):
            t, x, z = st
            dx = abs(xb - x)
            down = z <= self.contact + 0.001
            if not down and dx <= 0.01:
                opts = [max(z, self.contact + 0.004)]
                xb_use = x if dx <= 0.004 else xb
            else:
                opts = [self.hover, 0.126, 0.118, 0.1145]
                xb_use = xb
            best = None
            for zu in opts:
                tl = max(0.0, zu - z) / V_UP
                tm = dx / V_LAT if dx > 0.004 else 0.0
                tdc = self._desc_time(zu)
                tot = tl + tm + tdc
                if tot <= th - t:
                    best = (zu, tl, tm, tdc, 1.0)
                    break
                if best is None or tot < best[1] + best[2] + best[3]:
                    best = (zu, tl, tm, tdc, None)
            zu, tl, tm, tdc, k = best
            if k is None:
                tot = tl + tm + tdc
                gap = th - t
                k = K_MAX if gap <= 1e-3 else min(K_MAX, max(1.0, tot / gap))
            tp = max(th, t + (tl + tm + tdc) / k)
            if tp > th + late:
                return None
            t1 = t + tl / k
            t2 = t1 + tm / k
            t3 = max(t2, tp - tdc / k)
            return [(t1, x, zu), (t2, xb_use, zu), (t3, xb_use, zu), (tp, xb_use, self.press)], tp

        notes = list(notes or [])
        by_id = {}
        for n in notes:
            by_id[self._nid(n)] = n
        committed = {}
        for nid, n in by_id.items():
            tp = self.press_t.get(nid)
            if tp is not None and tp <= t_now:
                committed[nid] = tp

        new_press = dict(committed)
        down = st[2] <= self.contact + 0.001
        if down:
            act = None
            if committed:
                akey = max(committed, key=lambda k: committed[k])
                act = by_id[akey]
            handled = False
            if act is not None and self._kind(act) == "tap":
                add(max(committed[akey] + DWELL, t_now + 0.01), st[1], st[2])
                handled = True
            elif act is not None and self._end(act) > t_now:
                follow(act, committed[akey])
                handled = True
            if not handled:
                # Continue any active sustain under the stylus (e.g. regenerated
                # or re-split slide segments) instead of lifting mid-path.
                best, bd = None, 0.05
                for n in notes:
                    if self._kind(n) == "tap":
                        continue
                    h = float(n.hit_time)
                    if h > t_now + 0.05 or self._end(n) <= t_now + 0.03:
                        continue
                    d = abs(self._px(n, t_now - h) - st[1])
                    if d < bd:
                        best, bd = n, d
                if best is not None:
                    new_press[self._nid(best)] = min(float(best.hit_time), t_now)
                    follow(best, t_now)

        pend = []
        for nid, n in by_id.items():
            if nid in new_press:
                continue
            h = float(n.hit_time)
            if h < t_now - 0.14 or h > t_now + 2.5:
                continue
            if self._kind(n) != "tap" and self._end(n) <= t_now:
                continue
            pend.append(n)
        pend.sort(key=lambda n: float(n.hit_time))

        i = 0
        while i < len(pend):
            n = pend[i]
            h = float(n.hit_time)
            kind = self._kind(n)
            dwell = DWELL
            target = h + LAG
            if kind == "tap":
                late = TAP_LATE
                partner = None
                for j in range(i + 1, len(pend)):
                    m = pend[j]
                    if float(m.hit_time) - h >= 0.12:
                        break
                    if self._kind(m) == "tap" and abs(self._px(m, 0.0) - self._px(n, 0.0)) > 0.03:
                        partner = j
                        break
                if partner is not None:
                    m = pend[partner]
                    if abs(self._px(m, 0.0) - st[1]) < abs(self._px(n, 0.0) - st[1]):
                        pend[i], pend[partner] = m, n
                        n = m
                        h = float(n.hit_time)
                    target = h - CHORD_SHIFT
                    dwell = CHORD_DWELL
            else:
                late = SUS_LATE
                target = h - 0.005
                if abs(self._lane_at(n, 0.0) - 2.5) < 0.05:
                    target = h - 0.04
            i += 1
            if h + late < st[0]:
                continue
            xb = self._px(n, 0.0)
            r = trans(target, xb, late + (h - target) if kind == "tap" else late + (h - target))
            if r is None:
                continue
            seq, tp = r
            if tp > h + late:
                continue
            if kind != "tap" and tp >= self._end(n) - 0.05:
                continue
            for (tt, xx, zz) in seq:
                add(tt, xx, zz)
            new_press[self._nid(n)] = st[0]
            if kind == "tap":
                add(st[0] + dwell, st[1], self.press)
            else:
                follow(n, st[0])

        if st[2] < self.hover:
            add(st[0] + (self.hover - st[2]) / V_UP, st[1], self.hover)
        add(st[0] + 0.5, st[1], self.hover)
        self.press_t = new_press
        return api.make_plan(kfs, t_now=t_now)


def make_policy(api):
    return LocalPolicy(api)
