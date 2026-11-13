"""Behavior-cloning networks: an observation encoder and an action head.

Encoders
  * `state`: MLP over the encoded tracked notes and hand state
    (`features.encode_obs`), i.e. on top of the perception stage;
  * `pixels`: CNN over the rendered game frame (downsampled to 192 x 128),
    plus the hand state, i.e. perception learned inside the policy.

Heads
  * `mlp`: regresses the action chunk (MSE, near-term steps weighted up);
  * `diffusion`: a diffusion policy (Chi et al., 2023): DDPM over the
    normalized action chunk with 100 noise levels, a residual MLP denoiser
    conditioned on the observation embedding and the noise level (FiLM),
    sampled with 10 deterministic DDIM steps.  It models the distribution of
    expert chunks instead of their mean, which matters where the expert's
    choice is discrete (play this note or skip it; which fingertip).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .features import ACT_DIM, HAND_DIM, OBS_DIM


def _mlp(dims: list[int]) -> nn.Sequential:
    mods: list[nn.Module] = []
    for a, b in zip(dims[:-1], dims[1:]):
        mods += [nn.Linear(a, b), nn.LayerNorm(b), nn.GELU()]
    return nn.Sequential(*mods)


class StateEncoder(nn.Module):
    def __init__(self, emb: int = 512, layers: int = 3):
        super().__init__()
        self.net = _mlp([OBS_DIM] + [emb] * layers)
        self.register_buffer("obs_mean", torch.zeros(OBS_DIM))
        self.register_buffer("obs_std", torch.ones(OBS_DIM))

    def forward(self, batch: dict) -> torch.Tensor:
        return self.net((batch["obs"] - self.obs_mean) / self.obs_std)


class PixelEncoder(nn.Module):
    """Four stride-2 conv stages keep a 12 x 8 map that is flattened, not
    pooled: where a note head is on screen is its lane and time to hit."""

    def __init__(self, emb: int = 512):
        super().__init__()
        chans = [3, 32, 64, 128, 128]
        convs: list[nn.Module] = []
        for a, b in zip(chans[:-1], chans[1:]):
            convs += [nn.Conv2d(a, b, 3, stride=2, padding=1), nn.GroupNorm(8, b), nn.GELU(),
                      nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(8, b), nn.GELU()]
        self.cnn = nn.Sequential(*convs)
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(128 * 12 * 8, emb), nn.LayerNorm(emb), nn.GELU())
        self.hand = _mlp([HAND_DIM, 64])
        self.fuse = _mlp([emb + 64, emb, emb])
        self.register_buffer("hand_mean", torch.zeros(HAND_DIM))
        self.register_buffer("hand_std", torch.ones(HAND_DIM))

    def forward(self, batch: dict) -> torch.Tensor:
        img = batch["frame"].float() / 255.0
        h = self.hand((batch["hand"] - self.hand_mean) / self.hand_std)
        return self.fuse(torch.cat([self.proj(self.cnn(img)), h], -1))


class MLPHead(nn.Module):
    def __init__(self, emb: int, horizon: int):
        super().__init__()
        self.horizon = horizon
        self.out = nn.Linear(emb, horizon * ACT_DIM)
        self.register_buffer("w", torch.linspace(2.0, 1.0, horizon).repeat_interleave(ACT_DIM))

    def loss(self, cond: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return ((self.out(cond) - act) ** 2 * self.w).mean()

    def sample(self, cond: torch.Tensor) -> torch.Tensor:
        return self.out(cond)


def _timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / half)
    x = t.float()[:, None] * freqs[None]
    return torch.cat([x.sin(), x.cos()], -1)


class _ResBlock(nn.Module):
    def __init__(self, width: int, cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.film = nn.Linear(cond, 2 * width)
        self.ff = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, x, c):
        scale, shift = self.film(c).chunk(2, -1)
        return x + self.ff(self.norm(x) * (1 + scale) + shift)


class DiffusionHead(nn.Module):
    def __init__(self, emb: int, horizon: int, width: int = 1024, blocks: int = 4,
                 train_steps: int = 100, sample_steps: int = 10):
        super().__init__()
        self.horizon = horizon
        self.dim = horizon * ACT_DIM
        self.train_steps = train_steps
        self.sample_steps = sample_steps
        self.t_dim = 128
        self.cond = nn.Sequential(nn.Linear(emb + self.t_dim, width), nn.GELU(), nn.Linear(width, width))
        self.inp = nn.Linear(self.dim, width)
        self.blocks = nn.ModuleList(_ResBlock(width, width) for _ in range(blocks))
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, self.dim))
        # Squared-cosine noise schedule.
        s = 0.008
        steps = torch.arange(train_steps + 1) / train_steps
        abar = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        abar = (abar / abar[0]).clamp(1e-5, 1.0)
        self.register_buffer("abar", abar[1:])

    def _eps(self, x, t, cond):
        c = self.cond(torch.cat([cond, _timestep_embedding(t, self.t_dim)], -1))
        h = self.inp(x)
        for b in self.blocks:
            h = b(h, c)
        return self.out(h)

    def loss(self, cond: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        t = torch.randint(0, self.train_steps, (len(act),), device=act.device)
        noise = torch.randn_like(act)
        ab = self.abar[t][:, None]
        x = ab.sqrt() * act + (1 - ab).sqrt() * noise
        return ((self._eps(x, t, cond) - noise) ** 2).mean()

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        x = torch.randn(len(cond), self.dim, device=cond.device, generator=generator)
        ts = torch.linspace(self.train_steps - 1, 0, self.sample_steps).round().long().tolist()
        for i, t in enumerate(ts):
            ab = self.abar[t]
            eps = self._eps(x, torch.full((len(x),), t, device=x.device), cond)
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-6, 6)
            ab_prev = self.abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=x.device)
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return x


class BCModel(nn.Module):
    def __init__(self, encoder: str = "state", head: str = "mlp", horizon: int = 16, emb: int = 512):
        super().__init__()
        self.arch = {"encoder": encoder, "head": head, "horizon": horizon, "emb": emb}
        self.encoder = StateEncoder(emb) if encoder == "state" else PixelEncoder(emb)
        self.head = MLPHead(emb, horizon) if head == "mlp" else DiffusionHead(emb, horizon)
        self.register_buffer("act_mean", torch.zeros(horizon * ACT_DIM))
        self.register_buffer("act_std", torch.ones(horizon * ACT_DIM))

    def loss(self, batch: dict) -> torch.Tensor:
        act = (batch["act"].flatten(1) - self.act_mean) / self.act_std
        return self.head.loss(self.encoder(batch), act)

    @torch.no_grad()
    def act(self, batch: dict) -> torch.Tensor:
        """Action chunk in `features` units, (B, horizon, ACT_DIM)."""
        a = self.head.sample(self.encoder(batch)) * self.act_std + self.act_mean
        return a.view(-1, self.arch["horizon"], ACT_DIM)
