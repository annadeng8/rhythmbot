"""The robot-control API a generated policy is written against.

A generated policy is a Python file that defines

    def make_policy(api: RobotAPI):
        return <an object with a .plan(notes, t_now, X_WT_cmd, width_cmd) method>

and may import only numpy, math, a few standard-library modules and this
module.  The brain calls `plan` every perception tick (30 Hz) with the notes
it currently believes in; `plan` returns a `Plan`, a trajectory of the
fingertip pose and gripper opening that the arm follows until the next tick
replaces it.  Whatever the plan says for times before the next tick is what
the robot actually does.

Everything here is geometry and bookkeeping; deciding which notes to play
and how to move between them is the policy's job.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..chart import HOLD, SLIDE, TAP
from ..config import Config
from ..planner import Keyframe, Plan, TrackedNote, roll_for_rise, tip_gap

__all__ = ["RobotAPI", "Keyframe", "Plan", "TrackedNote", "TAP", "HOLD", "SLIDE"]


@dataclass
class RobotAPI:
    """Geometry of the screen and the hand, and helpers to build plans.

    Coordinates are world metres.  Lanes run along x (lane 0 at the most
    negative x); notes are hit on the judgment line y = `hit_y`; the screen
    surface is at z = `screen_z`.  A `Keyframe(t, x, y, z, width, ...)`
    places the point midway between the two fingertips (or, with ref=-1/+1,
    one fingertip) at (x, y, z) at time t with the gripper opened `width`;
    the hand points straight down unless `rise` is nonzero.
    """

    cfg: Config
    hand: int | None = None

    @property
    def tilt(self) -> float:
        return math.radians(self.cfg.gripper.tilt_degrees)*(-1 if self.hand==0 else 1) if self.cfg.gripper.single_tip else 0.0

    # -- the screen ------------------------------------------------------------
    @property
    def n_lanes(self) -> int:
        return self.cfg.layout.n_lanes

    @property
    def lane_width(self) -> float:
        return self.cfg.layout.lane_width

    @property
    def hit_y(self) -> float:
        return self.cfg.layout.hit_y + ((-1 if self.hand==0 else 1)*self.cfg.gripper.hand_y_offset if self.cfg.gripper.single_tip else 0)

    @property
    def screen_z(self) -> float:
        return self.cfg.layout.screen_z + (self.cfg.gripper.tip_radius if self.cfg.gripper.single_tip else 0)

    @property
    def contact_z(self) -> float:
        """A fingertip at or below this height counts as touching."""
        return self.screen_z + self.cfg.motion.contact_tolerance

    @property
    def touch_z(self) -> float:
        """Where to command a fingertip that should stay pressed (a little
        below the surface; the screen does not push back)."""
        return self.screen_z - self.cfg.motion.touch_depth

    def lane_x(self, lane: float) -> float:
        """World x of a lane centre; fractional lanes interpolate."""
        return self.cfg.layout.lane_x(lane)

    def x_to_lane(self, x: float) -> float:
        return self.cfg.layout.x_to_lane(x)

    # -- the judge -------------------------------------------------------------
    @property
    def windows(self) -> dict[str, float]:
        """Judgment windows, seconds either side of a note's hit time.  A
        touch counts for a note if it *starts* within `good` of the hit
        time, in the note's lane (within half a lane), near the judgment
        line.  A tap needs a fresh touch: a fingertip already down does not
        count.  Holds and slides also need contact in the (moving) lane for
        `hold_fraction` of their duration."""
        T = self.cfg.timing
        return {"perfect": T.perfect, "great": T.great, "good": T.good, "hold_fraction": T.hold_fraction,
                "slide_fraction": T.slide_min_maintenance, "slide_max_gap": T.slide_max_gap,
                "slide_tail_window": T.slide_tail_window, "slide_tolerance_lanes": T.slide_tolerance_lanes}

    # -- the hand --------------------------------------------------------------
    @property
    def max_width(self) -> float:
        """Widest gripper opening (m)."""
        return 0.0 if self.cfg.gripper.single_tip else self.cfg.gripper.max_width

    def tip_gap(self, width: float) -> float:
        """Distance between the two fingertips at opening `width`.  At width
        0 the fingertips are 23 mm apart, so a closed hand covers one lane
        with both fingertips."""
        return tip_gap(self.cfg.gripper, width)

    def width_for_tip_gap(self, gap: float) -> float | None:
        """Opening that puts the fingertips `gap` apart (e.g. two lanes of a
        chord), or None if the hand cannot open that wide."""
        return self.cfg.gripper.width_for_tip_gap(gap)

    def fingertips(self, k: Keyframe) -> tuple[np.ndarray, np.ndarray]:
        """World positions of the -x and +x fingertips for a keyframe."""
        th = roll_for_rise(self.cfg.gripper, k.width, k.rise)
        half = 0.5 * self.tip_gap(k.width) * np.array([math.cos(th), 0.0, math.sin(th)])
        p = np.array([k.x, k.y, k.z]) - k.ref * half
        return p - half, p + half

    @property
    def limits(self) -> dict[str, float]:
        """What the classical controller found the arm can follow
        comfortably: lateral speed (m/s), and the descent and lift times it
        uses around a touch.  Faster is possible but the arm lags (it is
        position-controlled through differential IK)."""
        M = self.cfg.motion
        return {
            "travel_speed": M.travel_speed,
            "descend_time": M.pre_time,
            "lift_time": M.post_time,
            "hover_height": M.hover_height,
            "tick_dt": 1.0 / self.cfg.perception.tick_hz,
        }

    # -- building plans --------------------------------------------------------
    def current_keyframe(self, t_now: float, X_WT_cmd, width_cmd: float) -> Keyframe:
        """The keyframe for where the hand is commanded right now.  A plan
        should start here, or the commanded pose jumps."""
        p = X_WT_cmd.translation()
        x_axis = X_WT_cmd.rotation().matrix()[:, 0]
        rise = self.tip_gap(width_cmd) * float(np.sin(np.arctan2(x_axis[2], x_axis[0])))
        if self.cfg.gripper.single_tip: rise = 0.0
        return Keyframe(t_now, float(p[0]), float(p[2]), float(width_cmd), float(p[1]), False, 0, rise)

    def keyframe(
        self, t: float, x: float, z: float, width: float = 0.0, down: bool = False, ref: int = 0, rise: float = 0.0
    ) -> Keyframe:
        """A keyframe on the judgment line.  `ref` says which point (x, z)
        places: 0 the midpoint between the fingertips, -1 / +1 the -x / +x
        fingertip.  `rise` is how far the +x fingertip is above the -x one
        (the hand rolled about world y): with it one fingertip can stay on
        the screen while the other is lifted, and between two keyframes the
        `ref` point moves in a straight line."""
        return Keyframe(float(t), float(x), float(z), float(width), self.hit_y, bool(down), int(ref), float(rise))

    def make_plan(self, keyframes: list[Keyframe], t_now: float | None = None) -> Plan:
        """A Plan through the keyframes.  They are sorted by time; with
        `t_now`, keyframes at or before it are dropped except the first
        (which should be `current_keyframe`), and times are nudged apart so
        they strictly increase."""
        kfs = sorted(keyframes, key=lambda k: k.t)
        if t_now is not None and kfs:
            kfs = [kfs[0]] + [k for k in kfs[1:] if k.t > t_now + 1e-3]
        out: list[Keyframe] = []
        for k in kfs:
            if out and k.t <= out[-1].t + 1e-3:
                k = Keyframe(out[-1].t + 1e-3, k.x, k.z, k.width, k.y, k.down, k.ref, k.rise)
            out.append(k)
        if self.cfg.gripper.single_tip:
            lo,hi = ((-self.cfg.layout.half_width,0.0) if self.hand==0 else (0.0,self.cfg.layout.half_width))
            out = [Keyframe(k.t,float(np.clip(k.x,lo,hi)),float(np.clip(k.z,self.touch_z,self.screen_z+.10)),0.0,self.hit_y,k.down,0,0.0) for k in out]
        return Plan(out, self.cfg.gripper, self.tilt)

