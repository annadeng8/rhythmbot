"""What the behavior-cloned policy sees and what it predicts.

Observation, at a brain tick:

  * the hand as commanded now: midpoint x and height above the screen, gripper
    opening, rise (how far the +x fingertip is above the -x one);
  * the next `N_NOTES` tracked notes that have not ended, soonest first, each
    as kind (one-hot), time to its hit and to its end, whether it is being
    played, whether its end is known, and its lane at the hit and at 25/50/
    75/100 % of its duration *relative to the hand's x*.  Missing slots are
    zero with a presence flag of 0.

Relative positions and times make the same motion look the same wherever and
whenever it happens, which is what lets the policy carry over to charts it
was not trained on.

Action: the hand state (x, height, opening, rise) at the next `HORIZON`
ticks, x and height as offsets from now.  The policy plays the first ticks of
that chunk and replans at the next tick, as the classical planner does.
"""

from __future__ import annotations

import numpy as np

from ..chart import HOLD, SLIDE, TAP
from ..config import Config
from ..planner import Plan, TrackedNote, tip_gap

N_NOTES = 8
NOTE_DIM = 13
HAND_DIM = 4
OBS_DIM = HAND_DIM + N_NOTES * NOTE_DIM
HORIZON = 10
ACT_DIM = 4
# Times are clipped to this many seconds ahead.
T_CLIP = 2.0
# Lane offsets and hand positions are measured in lanes (50 mm), heights in
# centimetres, opening and rise in centimetres, so every feature is O(1).
CM = 0.01


def hand_state(cfg: Config, x_axis_z: float, x_axis_x: float, p: np.ndarray, width: float) -> np.ndarray:
    rise = tip_gap(cfg.gripper, width) * float(np.sin(np.arctan2(x_axis_z, x_axis_x)))
    return np.array([p[0], p[2], width, rise])


def hand_from_pose(cfg: Config, X_WT, width: float) -> np.ndarray:
    """(midpoint x, z, opening, rise) of a commanded fingertip pose."""
    x_axis = X_WT.rotation().matrix()[:, 0]
    return hand_state(cfg, x_axis[2], x_axis[0], X_WT.translation(), width)


def hand_from_plan(cfg: Config, plan: Plan, t: float) -> np.ndarray:
    return hand_from_pose(cfg, plan.tip_pose(t), plan.width_at(t))


def encode_obs(cfg: Config, notes: list[TrackedNote], t_now: float, hand: np.ndarray) -> np.ndarray:
    L = cfg.layout
    x, z, width, rise = hand
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[:HAND_DIM] = [L.x_to_lane(x) - 0.5 * (L.n_lanes - 1), (z - L.screen_z) / CM, width / CM, rise / CM]
    lane_hand = L.x_to_lane(x)
    upcoming = sorted((n for n in notes if n.end_time >= t_now), key=lambda n: (n.hit_time, n.lane))
    for i, n in enumerate(upcoming[:N_NOTES]):
        f = obs[HAND_DIM + i * NOTE_DIM : HAND_DIM + (i + 1) * NOTE_DIM]
        f[0] = 1.0
        f[1:4] = [n.kind == TAP, n.kind == HOLD, n.kind == SLIDE]
        f[4] = np.clip(n.hit_time - t_now, -T_CLIP, T_CLIP)
        f[5] = np.clip(n.end_time - t_now, -T_CLIP, T_CLIP)
        f[6] = float(n.hit_time <= t_now)
        f[7] = float(n.duration_final)
        f[8] = n.lane - lane_hand
        d = max(n.duration, 1e-3)
        for k, frac in enumerate((0.25, 0.5, 0.75, 1.0)):
            f[9 + k] = n.lane_at_offset(frac * d) - lane_hand
    return obs


def encode_action(cfg: Config, hands: np.ndarray, hand_now: np.ndarray) -> np.ndarray:
    """(HORIZON, 4) future hand states -> normalized action chunk."""
    L = cfg.layout
    a = np.empty((HORIZON, ACT_DIM), dtype=np.float32)
    a[:, 0] = (hands[:, 0] - hand_now[0]) / L.lane_width
    a[:, 1] = (hands[:, 1] - L.screen_z) / CM
    a[:, 2] = hands[:, 2] / CM
    a[:, 3] = hands[:, 3] / CM
    return a


def decode_action(cfg: Config, a: np.ndarray, hand_now: np.ndarray) -> np.ndarray:
    L = cfg.layout
    hands = np.empty((HORIZON, 4))
    hands[:, 0] = hand_now[0] + a[:, 0] * L.lane_width
    hands[:, 1] = L.screen_z + a[:, 1] * CM
    hands[:, 2] = np.clip(a[:, 2] * CM, 0.0, cfg.gripper.max_width)
    hands[:, 3] = a[:, 3] * CM
    return hands
