import math


def make_policy(api):
    class Policy:
        def __init__(self):
            self.active = None
            self.pending = []
            self.done = set()
            self.notes = {}
            self.touch = float(api.touch_z)
            self.contact = float(api.contact_z)
            self.hover = float(api.screen_z) + 0.035
            self.clearance = self.contact + 0.014
            self.lo, self.hi = (
                (-0.36, 0.0) if api.hand == 0 else (0.0, 0.36)
            )

        def key(self, note):
            return str(note.key)

        def x(self, lane):
            return min(
                self.hi,
                max(self.lo, (float(lane) + 0.5) * 0.12 - 0.36),
            )

        def end(self, note):
            return max(float(note.hit_time), float(note.end_time))

        def lane(self, note, offset):
            offset = max(0.0, float(offset))
            try:
                return float(note.lane_at_offset(offset))
            except Exception:
                points = sorted(
                    (float(t), float(l))
                    for t, l in getattr(note, "waypoints", ())
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

        def tail_x(self, note):
            if note.kind == "slide":
                return self.x(
                    self.lane(note, self.end(note) - float(note.hit_time))
                )
            return self.head_x(note)

        def finish(self, note, landing):
            if note.kind == "tap":
                return landing + 0.060
            return max(landing + 0.045, self.end(note) + 0.020)

        def refresh(self, appointment):
            note = self.notes.get(appointment["key"], appointment["note"])
            appointment["note"] = note
            appointment["finish"] = self.finish(
                note, appointment["landing"]
            )

        def schedule(self, note, now, x, z):
            head = float(note.hit_time)
            target = self.head_x(note)
            distance = abs(target - x)
            low = z <= self.contact + 0.006

            # A short, explicit release permits fresh contacts in dense runs.
            if low:
                cruise = self.clearance
                lift = 0.065
                descent = 0.115
                travel = distance / 0.95
            else:
                cruise = min(self.hover, max(self.clearance, z))
                lift = 0.0
                descent = 0.205 * (
                    (cruise - self.touch) /
                    max(self.hover - self.touch, 1e-9)
                )
                descent = max(0.105, descent)
                travel = distance / 0.95

            preferred = lift + travel + descent
            nominal = head - 0.020
            landing = max(nominal, now + preferred)

            if landing > head + 0.030:
                lift = 0.055 if low else 0.0
                travel = distance / 1.20
                descent = min(descent, 0.105 if low else 0.155)
                landing = max(nominal, now + lift + travel + descent)

            latest = head + (0.100 if note.kind == "tap" else 0.035)
            if landing > latest:
                return None

            start = landing - lift - travel - descent
            return {
                "key": self.key(note),
                "note": note,
                "start": max(now, start),
                "lift_end": landing - travel - descent,
                "align": landing - descent,
                "landing": landing,
                "finish": self.finish(note, landing),
                "origin_x": x,
                "origin_z": z,
                "cruise": cruise,
                "lift": lift,
            }

        def plan(self, notes, t_now, X_WT_cmd, width_cmd):
            now = float(t_now)
            current = api.current_keyframe(now, X_WT_cmd, width_cmd)
            current.width = 0.0
            current.ref = 0.0
            current.rise = 0.0
            pos = X_WT_cmd.translation()
            cx = min(self.hi, max(self.lo, float(pos[0])))
            cz = float(pos[2])

            for note in notes:
                self.notes[self.key(note)] = note

            if self.active is not None:
                self.refresh(self.active)

            # Promote the actual forecast already being commanded. Rebuilding
            # its descent from the current pose would postpone every contact.
            while self.active is not None and now >= self.active["finish"]:
                self.done.add(self.active["key"])
                self.active = None
                while self.pending:
                    candidate = self.pending.pop(0)
                    if candidate["key"] not in self.done:
                        self.refresh(candidate)
                        self.active = candidate
                        break

            candidates = sorted(
                (
                    n for k, n in self.notes.items()
                    if k not in self.done
                ),
                key=lambda n: (float(n.hit_time), self.key(n)),
            )

            if self.active is None:
                self.pending = []
                for note in candidates:
                    if float(note.hit_time) + 0.14 < now:
                        self.done.add(self.key(note))
                        continue
                    a = self.schedule(note, now, cx, cz)
                    if a is not None:
                        self.active = a
                        break

            frames = [current]

            def add(t, x, z, down=False):
                t = float(t)
                if t <= now + 1e-8:
                    return
                if t <= float(frames[-1].t) + 1e-8:
                    return
                frames.append(
                    api.keyframe(
                        t, float(x), float(z),
                        width=0.0, down=bool(down), ref=0.0, rise=0.0,
                    )
                )

            def render(a):
                note = a["note"]
                head = float(note.hit_time)
                landing = a["landing"]
                target = self.head_x(note)

                add(
                    a["start"], a["origin_x"], a["origin_z"],
                    a["origin_z"] <= self.contact,
                )
                if a["lift"] > 0.0:
                    add(a["lift_end"], a["origin_x"], a["cruise"])
                add(a["align"], target, a["cruise"])
                add(landing, target, self.touch, True)

                if note.kind == "slide":
                    end = self.end(note)
                    begin = max(now, landing)
                    times = set()
                    # A near-term anchor keeps updated paths responsive even
                    # when the original head and waypoints are already past.
                    if begin < end:
                        times.add(min(end, begin + 0.010))
                        count = int(math.ceil((end - begin) / 0.020))
                        for i in range(1, count + 1):
                            times.add(min(end, begin + i * 0.020))
                    for offset, _ in getattr(note, "waypoints", ()):
                        turn = head + float(offset)
                        times.add(turn - 0.015)
                        times.add(turn)
                    times.add(end - 0.040)
                    times.add(end)
                    for t in sorted(times):
                        if t <= max(now, landing) + 1e-8 or t > end:
                            continue
                        offset = min(end - head, max(0.0, t - head + 0.015))
                        add(t, self.x(self.lane(note, offset)), self.touch, True)

                tail = self.tail_x(note)
                add(a["finish"], tail, self.touch, True)
                return a["finish"], tail

            if self.active is None:
                add(now + 0.110, cx, self.hover)
                add(now + 0.450, cx, self.hover)
                return api.make_plan(frames, t_now=now)

            end_t, end_x = render(self.active)
            reserved = {self.active["key"]}
            new_pending = []

            # Reuse any transition that has already started, including its
            # absolute release, alignment, and landing times.
            for a in self.pending:
                if a["key"] in self.done or a["key"] in reserved:
                    continue
                if a["start"] > now + 1e-8:
                    break
                self.refresh(a)
                if a["landing"] <= end_t:
                    continue
                end_t, end_x = render(a)
                new_pending.append(a)
                reserved.add(a["key"])

            for note in candidates:
                key = self.key(note)
                if key in reserved or key in self.done:
                    continue
                if float(note.hit_time) + 0.14 < end_t:
                    continue
                if float(note.hit_time) > end_t + 2.5:
                    break
                a = self.schedule(note, end_t, end_x, self.touch)
                if a is None:
                    continue
                end_t, end_x = render(a)
                new_pending.append(a)
                reserved.add(key)
                if len(new_pending) >= 5:
                    break

            self.pending = new_pending
            add(end_t + 0.110, end_x, self.hover)
            add(end_t + 0.450, end_x, self.hover)
            return api.make_plan(frames, t_now=now)

    return Policy()
