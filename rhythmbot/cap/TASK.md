# Task: write the control program for a rhythm-game-playing robot arm

A simulated KUKA iiwa arm with a two-finger Schunk WSG gripper plays a
Project SEKAI-style rhythm game on a flat touch screen in front of it. You
write the Python program that decides, every 1/30 s, where the fingertips go.
Your goal is to **complete as many notes as possible on charts you have not
seen**: the charts you get feedback on are only a sample; the program will
be scored afterwards on other charts, including denser ones and real songs.
Write general strategy, not something tuned to particular charts.

## The game

The screen has 6 lanes side by side along world x, each 50 mm wide (lane 0
at the most negative x; `api.lane_x(lane)` gives a lane centre). Notes
scroll toward the robot and must be played when they reach the judgment line
at y = `api.hit_y`. The screen surface is at z = `api.screen_z`. Note kinds:

* **tap**: touch the lane on the line at the note's hit time. Only a *fresh*
  touch counts: a fingertip that is already down does not hit a new tap.
* **hold**: touch at the hit time and stay down in the lane until the note
  ends.
* **slide**: touch at the hit time, then keep a fingertip on the moving
  target lane (`note.lane_at_offset(seconds since hit)`, piecewise linear
  between `note.waypoints`) until it ends.

The judge (it watches the actual fingertips in the physics simulation):

* a touch counts for a note if it starts within ±140 ms of the hit time
  (±45 ms is "perfect"), with the fingertip within half a lane of the lane
  centre and on the judgment line (|y − hit_y| ≤ 60 mm);
* a fingertip "is down" when it is within 4 mm above the screen surface
  (`api.contact_z`); command `api.touch_z` (2 mm below the surface: the
  screen is not physical and does not push back) to press;
* a hold or slide is **completed** only if it was hit and contact in the
  (moving) lane was kept for at least 80% of its duration; a tap is completed
  when hit. Each fresh touch can be spent on one note only, but either
  fingertip can play any note, so two notes at once (a chord) can be played
  with one fingertip each.

Several notes can be due at once, and notes may come faster than one hand can
move between them. Skipping a note you cannot make is often better than
ruining the next one too.

## What your program sees

`plan(notes, t_now, X_WT_cmd, width_cmd)` is called every perception tick
(30 Hz, `t_now` in seconds of simulated time) with:

* `notes`: list of `TrackedNote`, the notes perception currently believes
  in: every note that is on screen (up to 1.6 s before its hit) and not yet
  finished, including holds/slides being played right now. Fields: `kind`
  (`"tap" | "hold" | "slide"`, also `api.TAP` etc.), `lane` (float),
  `hit_time`, `duration`, `end_time`, `waypoints` [(offset s, lane)],
  `lane_at_offset(off)`, `duration_final` (False while a long note's end is
  still off-screen: `duration` is then a lower bound that will grow), `key`
  (stable while the note is tracked). The same objects come back on later
  ticks, so you may keep your own state in your policy object keyed by
  `note.key`. You may set `note.skipped = True` to say you are giving a
  note up (only used in reporting).
* `X_WT_cmd`, `width_cmd`: the fingertip pose and gripper opening your
  previous plan commands at `t_now` (a pydrake `RigidTransform`; use
  `api.current_keyframe(t_now, X_WT_cmd, width_cmd)`).

You return a `Plan` (build it with `api.make_plan(keyframes, t_now)`): a
piecewise-linear trajectory through `Keyframe`s. **Start it with
`api.current_keyframe(...)`** so the command is continuous. The arm follows
the plan until the next tick, when you are called again with newer
information and return a new plan. The notes' timing from perception is
exact here, but plan as if information can be refined tick to tick.

## The hand

* A keyframe places a point at (x, hit_y, z) at time t: by default the
  midpoint between the two fingertips (`ref=0`), or the −x / +x fingertip
  (`ref=-1` / `ref=+1`).
* `width` is the gripper opening (0 … `api.max_width` = 0.107 m). The
  fingertips are `api.tip_gap(width)` apart: 23 mm when closed (both
  fingertips land in one lane), up to 130 mm wide open, i.e. up to about
  2.6 lanes apart. `api.width_for_tip_gap(gap)` inverts that.
* `rise` (m) rolls the hand about world y so the +x fingertip is `rise`
  above the −x one (negative: below). With it, one fingertip can stay
  pressed on the screen while the other is lifted. Between two keyframes the
  `ref` point moves in a straight line (the segment uses the first
  keyframe's `ref`, or the second's if the first is 0), and opening, rise
  and position all interpolate linearly.
* `api.fingertips(keyframe)` gives both fingertip positions for a keyframe.
* The arm is position-controlled through differential inverse kinematics; it
  lags commands that move fast. `api.limits` lists the speeds and descent /
  lift times a hand-written controller uses comfortably (about 0.9 m/s
  sideways, 0.22 s to descend from 35 mm hover, 0.12 s to lift). Rolling
  the hand or changing the opening while a fingertip is held down swings
  the whole gripper, so do that slowly (the hand-written controller keeps
  the opening change under 1 m/s and rolls under ~15°).

## Rules

* Your program is the single file `policy.py` in this directory. It must
  define `make_policy(api)` returning an object with the `plan` method
  above. `api` is a `RobotAPI`; `api.py` here is a copy of its source, and
  of the types it re-exports you only need `Keyframe`, `Plan`, `TAP`,
  `HOLD`, `SLIDE`.
* It may import only `numpy`, `math`, `dataclasses`, `typing`,
  `functools`, `itertools`, `collections`, `heapq`, `bisect`, `enum`, `copy`
  and `rhythmbot.cap.api`. No file or network access, no `eval`/`exec`.
  Anything else is rejected before it runs.
* If `plan` raises or returns something malformed, the hand is lifted in
  place for that tick and the error is reported to you.
* `plan` should take well under 33 ms.
* Work only inside this directory.

## Minimal example (plays nothing)

```python
from rhythmbot.cap.api import Keyframe, Plan, TAP, HOLD, SLIDE

class Idle:
    def __init__(self, api):
        self.api = api

    def plan(self, notes, t_now, X_WT_cmd, width_cmd):
        k0 = self.api.current_keyframe(t_now, X_WT_cmd, width_cmd)
        hover = self.api.keyframe(t_now + 0.2, k0.x, self.api.screen_z + 0.035)
        return self.api.make_plan([k0, hover], t_now)

def make_policy(api):
    return Idle(api)
```

## Feedback

After each revision your program is run on a set of training charts and
`feedback.md` is written here: the score on each chart, and every note that
was not completed, what the judge saw (missed, early/late, contact lost),
and the notes around it, plus any errors your program raised.
