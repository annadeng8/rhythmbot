"""All tunable numbers in one place.

Everything downstream (chart, game, planner, judge, robot scene) reads from
these dataclasses, so the lane geometry the vision model is trained on is the
same lane geometry the arm plays and the judge scores.

Frames and axes
---------------
World frame W: the iiwa base sits at the origin, z up.  With the course's
default joint angles the arm points along -y, so the playing surface is laid
out in front of it along -y:

    * lanes run across the robot's view, along world x (lane 0 at -x);
    * notes travel along +y, toward the robot, and are "hit" when they reach
      the judgment line at y = HIT_Y;
    * the highway (the part of the screen where approaching notes are
      visible) extends from the judgment line away from the robot, i.e.
      toward more negative y.

Gripper frame G is the WSG body frame: the fingers slide along G's x axis and
reach out along +y, so "pointing down" is G's +y aligned with world -z.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Layout:
    """Where the playing surface is, in metres, and how it is divided."""

    n_lanes: int = 6
    lane_width: float = 0.05
    # Judgment line: notes are hit here.
    hit_y: float = -0.50
    # Top surface of the screen.  Fingertips touch at this height.
    screen_z: float = 0.10
    # How long a note is visible before its hit time, and how fast it moves.
    # Their product is the length of the visible highway.
    approach_time: float = 1.6
    scroll_speed: float = 0.45
    # Extra screen around the active area, for looks.
    margin: float = 0.03

    @property
    def half_width(self) -> float:
        return 0.5 * self.n_lanes * self.lane_width

    @property
    def highway_length(self) -> float:
        return self.approach_time * self.scroll_speed

    def lane_x(self, lane: float) -> float:
        """World x of a lane centre; fractional lanes interpolate."""
        return (lane + 0.5) * self.lane_width - self.half_width

    def x_to_lane(self, x: float) -> float:
        return (x + self.half_width) / self.lane_width - 0.5

    def note_y(self, time_to_hit: float) -> float:
        """World y of a note that will be hit in `time_to_hit` seconds."""
        return self.hit_y - time_to_hit * self.scroll_speed

    def far_y(self) -> float:
        """y where notes first appear."""
        return self.hit_y - self.highway_length

    def screen_center(self) -> np.ndarray:
        return np.array([0.0, 0.5 * (self.hit_y + self.far_y()) + 0.02, self.screen_z])

    def screen_size(self) -> np.ndarray:
        """(x, y) extent of the screen slab, with margins and a strip past the
        judgment line so the fingertip never hangs off the edge."""
        return np.array(
            [
                2 * self.half_width + 2 * self.margin,
                self.highway_length + 2 * self.margin + 0.04,
            ]
        )


@dataclass(frozen=True)
class Timing:
    """Judgment windows, seconds, each side of the note's hit time."""

    perfect: float = 0.045
    great: float = 0.090
    good: float = 0.140
    # Fraction of a hold/slide during which contact must be maintained for
    # the note to count as completed.
    hold_fraction: float = 0.80
    # A slide is tracked when the fingertip is within this many lanes of the
    # moving target.
    slide_tolerance_lanes: float = 0.6

    @property
    def miss_after(self) -> float:
        return self.good


@dataclass(frozen=True)
class Gripper:
    """WSG numbers measured off the Drake model (see probes in README)."""

    # Distance from G's origin to the fingertip contact point, along G's +y.
    # Finger body frames are at 0.077 and the finger mesh reaches 0.041
    # beyond that.
    tip_offset: float = 0.118
    # Distance from each finger body frame to its tip, along G's +y.
    finger_tip_offset: float = 0.041
    # Half the gap between the two finger frames when fully closed: the
    # fingertips sit at x = -/+ this when the command is 0.
    half_closed: float = 0.0115
    max_width: float = 0.107

    def tip_x_offsets(self, width: float) -> tuple[float, float]:
        """Fingertip x positions in G for a commanded opening `width`."""
        d = self.half_closed + 0.5 * width
        return (-d, d)

    def width_for_tip_gap(self, gap: float, slack: float = 0.0) -> float | None:
        """Opening that puts the fingertips `gap` apart, or None if too wide.
        Up to `slack` too wide is allowed: then the gripper opens fully and
        the fingertips land that much short of `gap` in total."""
        w = gap - 2 * self.half_closed
        if w < 0 or w > self.max_width + slack:
            return None
        return min(w, self.max_width)


@dataclass(frozen=True)
class Motion:
    """How the classical planner shapes a note."""

    hover_height: float = 0.035  # above screen_z between notes
    touch_depth: float = 0.002  # commanded below screen_z (screen is visual)
    pre_time: float = 0.22  # hover-to-touch descent duration
    post_time: float = 0.12  # touch-to-hover lift duration
    tap_dwell: float = 0.06  # how long a tap stays down
    press_time: float = 0.03  # from reaching the plane to touch_depth
    # A fingertip this close above the plane is sensed as touching; the
    # planner aims to cross this height at the hit time.
    contact_tolerance: float = 0.004
    # Two taps closer in time than this are one chord (noisy detectors
    # jitter the two hit times apart).
    chord_window: float = 0.06
    # Lateral travel: how fast the tip may move between notes, used to decide
    # when to leave the previous note's hover pose.
    travel_speed: float = 0.9
    min_travel_time: float = 0.12
    # Trajectory lookahead applied to the commanded pose to compensate for the
    # DiffIK integrator's tracking lag.
    command_lead: float = 0.0
    # Playing around a hold: the free fingertip waits this far above the
    # holding one (the hand rolled about the holding fingertip), takes this
    # long to dip to the screen, and this long to rise again after.
    roll_raise: float = 0.010
    roll_dip_time: float = 0.08
    roll_lift_time: float = 0.08
    # How fast the opening may change while a fingertip is held down: the
    # arm has to swing the gripper sideways by half the change, and faster
    # than this the holding fingertip slides in its lane as the arm lags.
    pivot_width_speed: float = 1.0
    # Two fingertips may each land this far off their lane centre (lanes are
    # 50 mm wide) when the lanes are farther apart than the WSG opens.
    lane_slack: float = 0.012


@dataclass(frozen=True)
class Perception:
    tick_hz: float = 30.0
    # Rendered frame size for the vision model (W, H).
    frame_size: tuple[int, int] = (256, 384)
    # Simulated detector imperfections for the oracle.
    timing_noise_std: float = 0.0
    latency: float = 0.0
    # A detected note is planned for only after this many sightings.
    min_observations: int = 2
    # An upcoming note is on screen, and reported, for the whole approach
    # time; one that has gone unreported this long before its hit is a
    # phantom (or a duplicate from a detector flipping its kind) and is
    # dropped.
    stale_after: float = 0.2


@dataclass(frozen=True)
class Config:
    layout: Layout = field(default_factory=Layout)
    timing: Timing = field(default_factory=Timing)
    gripper: Gripper = field(default_factory=Gripper)
    motion: Motion = field(default_factory=Motion)
    perception: Perception = field(default_factory=Perception)
    # Simulation
    plant_time_step: float = 0.001
    diffik_time_step: float = 0.005
    judge_hz: float = 200.0
    visualizer_hz: float = 30.0


DEFAULT = Config()
