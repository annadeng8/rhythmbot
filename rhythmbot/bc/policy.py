"""The behavior-cloned policy: an MLP from the encoded observation to the
next `HORIZON` hand states, played as a short plan and replaced every tick."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..config import Config
from ..planner import Keyframe, Plan, Policy
from .features import ACT_DIM, HORIZON, OBS_DIM, decode_action, encode_obs, hand_from_pose


class BCNet(nn.Module):
    def __init__(self, hidden: int = 512, layers: int = 3):
        super().__init__()
        dims = [OBS_DIM] + [hidden] * layers
        mods: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            mods += [nn.Linear(a, b), nn.LayerNorm(b), nn.GELU()]
        mods.append(nn.Linear(dims[-1], HORIZON * ACT_DIM))
        self.net = nn.Sequential(*mods)
        for name in ("obs_mean", "obs_std", "act_mean", "act_std"):
            n = OBS_DIM if name.startswith("obs") else HORIZON * ACT_DIM
            self.register_buffer(name, torch.zeros(n) if name.endswith("mean") else torch.ones(n))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalized prediction (what the loss is computed on)."""
        return self.net((obs - self.obs_mean) / self.obs_std)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Action chunk in `features` units, (B, HORIZON, ACT_DIM)."""
        return (self.forward(obs) * self.act_std + self.act_mean).view(-1, HORIZON, ACT_DIM)


class BCPolicy(Policy):
    def __init__(self, net: BCNet, cfg: Config):
        self.net = net.eval()
        self.cfg = cfg
        self.dt = 1.0 / cfg.perception.tick_hz

    @staticmethod
    def load(path: str | Path, cfg: Config) -> "BCPolicy":
        torch.set_num_threads(1)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        net = BCNet(**ckpt["arch"])
        net.load_state_dict(ckpt["state_dict"])
        return BCPolicy(net, cfg)

    def plan(self, notes, t_now, X_WT_cmd, width_cmd) -> Plan:
        L, M = self.cfg.layout, self.cfg.motion
        hand = hand_from_pose(self.cfg, X_WT_cmd, width_cmd)
        obs = encode_obs(self.cfg, notes, t_now, hand)
        with torch.no_grad():
            a = self.net.act(torch.from_numpy(obs)[None])[0].numpy()
        hands = decode_action(self.cfg, a, hand)
        p = X_WT_cmd.translation()
        kfs = [Keyframe(t_now, p[0], p[2], width_cmd, p[1], False, 0, hand[3])]
        lo, hi = L.lane_x(-0.5), L.lane_x(L.n_lanes - 0.5)
        for k, (x, z, w, rise) in enumerate(hands, start=1):
            x = float(np.clip(x, lo, hi))
            z = float(max(z, L.screen_z - M.touch_depth))
            kfs.append(Keyframe(t_now + k * self.dt, x, z, float(w), L.hit_y, False, 0, float(rise)))
        return Plan(kfs, self.cfg.gripper)
