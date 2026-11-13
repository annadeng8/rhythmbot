import math
import numpy as np

DWELL = 0.07
LAG = 0.015
D_DESC = 0.15
V_UP = 0.40
V_LAT = 0.9
K_MAX = 1.6
STEP = 0.04


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

    def _px(self, n, off):
        lane = n.lane
        if self._kind(n) == "slide":
            try:
                lane = n.lane_at_offset(min(self._dur(n), max(0.0, off)))
            except Exception:
                lane = n.lane
        return self._clip(self._lane_x(lane))

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
                    if o0 < float(w[0]) < dur:
                        offs.add(float(w[0]))
                except Exception:
                    pass
            offs.add(dur)
            for o in sorted(offs):
                add(hit + o, self._px(n, o), self.press)
            add(hit + dur + 0.025, self._px(n, dur), self.press)

        def trans(th, xb):
            t, x, z = st
            dx = abs(xb - x)
            down = z <= self.contact + 0.001
            if not down and dx <= 0.01:
                opts = [max(z, self.contact + 0.004)]
                xb_use = x if dx <= 0.004 else xb
            else:
                opts = [self.hover, 0.126, 0.118]
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
            if tp > th + 0.09:
                return None
            t1 = t + tl / k
            t2 = t1 + tm / k
            t3 = max(t2, tp - tdc / k)
            return [(t1, x, zu), (t2, xb_use, zu), (t3, xb_use, zu), (tp, xb_use, self.press)], tp

        notes = list(notes or [])
        committed = {}
        for n in notes:
            tp = self.press_t.get(n.key)
            if tp is not None and tp <= t_now:
                committed[n.key] = tp
        down = st[2] <= self.contact + 0.001
        if down and committed:
            akey = max(committed, key=lambda k: committed[k])
            act = [n for n in notes if n.key == akey][0]
            if self._kind(act) == "tap":
                add(max(committed[akey] + DWELL, t_now + 0.01), st[1], st[2])
            elif self._end(act) > t_now:
                follow(act, committed[akey])

        new_press = dict(committed)
        pend = []
        for n in notes:
            if n.key in committed:
                continue
            h = float(n.hit_time)
            if h < t_now - 0.1 or h > t_now + 2.5:
                continue
            if self._kind(n) != "tap" and self._end(n) <= t_now:
                continue
            pend.append(n)
        pend.sort(key=lambda n: float(n.hit_time))

        for n in pend:
            h = float(n.hit_time)
            if h + 0.09 < st[0]:
                continue
            xb = self._px(n, 0.0)
            r = trans(h + LAG, xb)
            if r is None:
                continue
            seq, tp = r
            if self._kind(n) != "tap" and tp >= self._end(n) - 0.05:
                continue
            for (tt, xx, zz) in seq:
                add(tt, xx, zz)
            new_press[n.key] = st[0]
            if self._kind(n) == "tap":
                add(st[0] + DWELL, st[1], self.press)
            else:
                follow(n, st[0])

        if st[2] < self.hover:
            add(st[0] + (self.hover - st[2]) / V_UP, st[1], self.hover)
        add(st[0] + 0.5, st[1], self.hover)
        self.press_t = new_press
        return api.make_plan(kfs, t_now=t_now)


def make_policy(api):
    return LocalPolicy(api)
