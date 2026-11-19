import math
import numpy as np


def make_policy(api):
    class Policy:
        def __init__(self):
            self.active = None
            self.done = set()
            self.hover = float(api.screen_z) + 0.035
            self.touch = float(api.touch_z)
            self.contact = float(api.contact_z)
            self.lo, self.hi = (-0.36, 0.0) if api.hand == 0 else (0.0, 0.36)

        def identity(self, note):
            return str(note.key)

        def x(self, lane):
            return float(np.clip((float(lane) + 0.5) * 0.12 - 0.36,
                                 self.lo, self.hi))

        def end(self, note):
            return max(float(note.hit_time), float(note.end_time))

        def lane(self, note, offset):
            offset = max(0.0, float(offset))
            try:
                return float(note.lane_at_offset(offset))
            except Exception:
                points = sorted(
                    [(float(t), float(l)) for t, l in note.waypoints],
                    key=lambda p: p[0]
                )
                if not points:
                    return float(note.lane)
                if points[0][0] > 0.0:
                    points.insert(0, (0.0, float(note.lane)))
                if offset <= points[0][0]:
                    return points[0][1]
                for (ta, la), (tb, lb) in zip(points, points[1:]):
                    if offset <= tb:
                        u = (offset - ta) / max(tb - ta, 1e-9)
                        return la + u * (lb - la)
                return points[-1][1]

        def head_x(self, note):
            if note.kind == "slide":
                return self.x(self.lane(note, 0.0))
            return self.x(note.lane)

        def finish(self, note, landing):
            if note.kind == "tap":
                return landing + 0.065
            return max(landing + 0.065, self.end(note) + 0.045)

        def appointment(self, note, now, x, z):
            head = float(note.hit_time)
            target = self.head_x(note)
            distance = abs(target - x)

            # Lift before moving into another lane or making another head.
            if z < self.contact + 0.004:
                lift = max(0.045, 0.105 * (self.hover - z) /
                           max(self.hover - self.touch, 1e-6))
            elif distance > 0.025:
                lift = max(0.0, 0.085 * (self.hover - z) /
                           max(self.hover - self.touch, 1e-6))
            else:
                lift = 0.0

            travel = distance / 0.9
            preferred = lift + travel + 0.205
            minimum_descent = 0.105
            minimum = lift + distance / 1.25 + minimum_descent
            landing = max(head - 0.005, now + minimum)

            if landing > head + 0.105:
                return None

            available = landing - now
            if available >= preferred:
                descent = 0.205
                movement = travel
                start = landing - descent - movement - lift
            else:
                start = now
                movement = distance / 1.25
                descent = min(
                    0.205, max(minimum_descent, available - lift - movement)
                )

            lift_end = start + lift
            align = landing - descent
            return {
                "note": note,
                "key": self.identity(note),
                "landing": landing,
                "start": start,
                "lift_end": lift_end,
                "align": align,
                "origin_x": x,
                "origin_z": z,
                "finish": self.finish(note, landing),
            }

        def plan(self, notes, t_now, X_WT_cmd, width_cmd):
            now = float(t_now)
            current = api.current_keyframe(now, X_WT_cmd, width_cmd)
            current.width = 0.0
            current.ref = 0.0
            current.rise = 0.0
            position = np.asarray(X_WT_cmd.translation(), dtype=float)
            cx = float(np.clip(position[0], self.lo, self.hi))
            cz = float(position[2])

            observed = {}
            for note in notes:
                observed[self.identity(note)] = note

            if self.active is not None:
                key = self.active["key"]
                if key in observed:
                    self.active["note"] = observed[key]
                    self.active["finish"] = self.finish(
                        observed[key], self.active["landing"]
                    )
                if now >= self.active["finish"]:
                    self.done.add(key)
                    self.active = None

            candidates = sorted(
                (n for k, n in observed.items() if k not in self.done),
                key=lambda n: (float(n.hit_time), self.identity(n))
            )

            if self.active is None:
                for note in candidates:
                    if float(note.hit_time) + 0.14 < now:
                        self.done.add(self.identity(note))
                        continue
                    appointment = self.appointment(note, now, cx, cz)
                    if appointment is not None:
                        self.active = appointment
                        break
                    self.done.add(self.identity(note))

            frames = [current]

            def add(t, x, z, down=False):
                t = float(t)
                if t <= now + 1e-7:
                    return
                frame = api.keyframe(
                    t, float(x), float(z),
                    width=0.0, down=bool(down), ref=0.0, rise=0.0
                )
                if t > float(frames[-1].t) + 1e-7:
                    frames.append(frame)

            def render(a):
                note = a["note"]
                landing = a["landing"]
                target = self.head_x(note)
                origin_x = a["origin_x"]
                origin_z = a["origin_z"]

                add(a["start"], origin_x, origin_z,
                    origin_z <= self.contact)
                if a["lift_end"] > a["start"] + 1e-7:
                    add(a["lift_end"], origin_x, self.hover)
                if a["align"] > a["lift_end"] + 1e-7:
                    add(a["align"], target, self.hover)
                elif a["align"] > now:
                    add(a["align"], target,
                        max(self.hover, origin_z))

                add(landing, target, self.touch, True)

                if note.kind == "slide":
                    head = float(note.hit_time)
                    end = self.end(note)
                    begin = max(now, landing)
                    sample_times = set()
                    count = max(0, int(math.ceil((end - begin) / 0.02)))
                    for i in range(1, count + 1):
                        sample_times.add(min(end, begin + i * 0.02))
                    for offset, _ in note.waypoints:
                        turn = head + float(offset)
                        sample_times.add(turn - 0.020)
                        sample_times.add(turn)
                    sample_times.add(end - 0.040)
                    sample_times.add(end)

                    for t in sorted(sample_times):
                        if t > max(now, landing) + 1e-7 and t <= end:
                            # Small positional lead compensates for IK lag.
                            offset = min(end - head, t - head + 0.020)
                            add(t, self.x(self.lane(note, offset)),
                                self.touch, True)
                    tail_x = self.x(self.lane(note, end - head))
                else:
                    tail_x = target

                add(a["finish"], tail_x, self.touch, True)
                return a["finish"], tail_x

            if self.active is None:
                lift_time = max(
                    0.04,
                    0.105 * max(0.0, self.hover - cz) /
                    max(self.hover - self.touch, 1e-6)
                )
                add(now + lift_time, cx, self.hover)
                add(now + lift_time + 0.4, cx, self.hover)
                return api.make_plan(frames, t_now=now)

            end_t, end_x = render(self.active)
            reserved = {self.active["key"]}

            # Forecast transitions so releases start at their absolute times,
            # including between perception updates.
            for note in candidates:
                key = self.identity(note)
                if key in reserved or key in self.done:
                    continue
                if float(note.hit_time) > end_t + 2.0:
                    break
                next_a = self.appointment(note, end_t, end_x, self.touch)
                if next_a is None:
                    continue
                end_t, end_x = render(next_a)
                reserved.add(key)
                if len(reserved) >= 4:
                    break

            add(end_t + 0.105, end_x, self.hover)
            add(end_t + 0.405, end_x, self.hover)
            return api.make_plan(frames, t_now=now)

    return Policy()
