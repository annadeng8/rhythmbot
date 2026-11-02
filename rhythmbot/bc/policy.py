"""The behavior-cloned policy: a `BCModel` from the observation to the next
`horizon` hand states, played as a short plan.

The MLP head is replanned every tick, like the classical planner.  The
diffusion head samples a chunk and plays its first `exec_steps` ticks before
sampling again (Chi et al.'s receding horizon): resampling every tick would
switch between modes of the expert distribution from one tick to the next.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..config import Config
from ..planner import Keyframe, Plan, Policy
from .features import decode_action, encode_obs, frame_input, hand_from_pose, hand_obs
from .models import BCModel


class BCPolicy(Policy):
    def __init__(self, model: BCModel, cfg: Config, exec_steps: int | None = None, device: str = "cpu"):
        self.model = model.eval().to(device)
        self.device = device
        self.cfg = cfg
        self.dt = 1.0 / cfg.perception.tick_hz
        self.pixels = model.arch["encoder"] == "pixels"
        self.wants_frames = self.pixels
        self.frame: np.ndarray | None = None
        if exec_steps is None:
            exec_steps = 1 if model.arch["head"] == "mlp" else 4
        self.exec_steps = exec_steps
        self._plan: Plan | None = None
        self._planned_at = -1e9
        self.generator = torch.Generator(device=device).manual_seed(0)

    @staticmethod
    def load(path: str | Path, cfg: Config, device: str = "cpu") -> "BCPolicy":
        torch.set_num_threads(1)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = BCModel(**ckpt["arch"])
        model.load_state_dict(ckpt["state_dict"])
        return BCPolicy(model, cfg, ckpt.get("exec_steps"), device)

    def observe(self, notes, t_now, hand) -> dict:
        if self.pixels:
            frame = torch.from_numpy(frame_input(self.frame))[None]
            return {"frame": frame.to(self.device), "hand": torch.from_numpy(hand_obs(self.cfg, hand))[None].to(self.device)}
        return {"obs": torch.from_numpy(encode_obs(self.cfg, notes, t_now, hand))[None].to(self.device)}

    def plan(self, notes, t_now, X_WT_cmd, width_cmd) -> Plan:
        # Keep playing the last chunk unless it is used up, or someone else
        # (the expert, in DAgger) has moved the hand since.
        if (
            self._plan is not None
            and t_now < self._planned_at + (self.exec_steps - 0.5) * self.dt
            and np.allclose(self._plan.tip_pose(t_now).translation(), X_WT_cmd.translation(), atol=1e-4)
        ):
            return self._plan
        L, M = self.cfg.layout, self.cfg.motion
        hand = hand_from_pose(self.cfg, X_WT_cmd, width_cmd)
        if self.pixels and self.frame is None:
            # The brain's very first plan is made before any frame exists.
            p = X_WT_cmd.translation()
            k = Keyframe(t_now, p[0], p[2], width_cmd, p[1], False, 0, hand[3])
            return Plan([k], self.cfg.gripper)
        a = self.model.act(self.observe(notes, t_now, hand))[0].cpu().numpy()
        self.last_action = a
        hands = decode_action(self.cfg, a, hand)
        p = X_WT_cmd.translation()
        kfs = [Keyframe(t_now, p[0], p[2], width_cmd, p[1], False, 0, hand[3])]
        lo, hi = L.lane_x(-0.5), L.lane_x(L.n_lanes - 0.5)
        for k, (x, z, w, rise) in enumerate(hands, start=1):
            x = float(np.clip(x, lo, hi))
            z = float(max(z, L.screen_z - M.touch_depth))
            kfs.append(Keyframe(t_now + k * self.dt, x, z, float(w), L.hit_y, False, 0, float(rise)))
        self._plan, self._planned_at = Plan(kfs, self.cfg.gripper), t_now
        return self._plan
