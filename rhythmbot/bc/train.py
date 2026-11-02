"""Behavior cloning from the classical planner, with DAgger.

    python -m rhythmbot.bc.train --out results/bc_policy.pt                 # demos + 2 DAgger rounds
    python -m rhythmbot.bc.train --out results/bc_plain.pt --dagger-rounds 0  # plain BC

Demonstrations are collected without the arm: the brain alone runs the
tracker and a policy against the oracle detector, as fast as it can, and the
policy's own commanded pose is what it observes next (the brain never reads
the robot's measured pose), so nothing is lost by leaving out the physics.
At every tick the classical planner is asked for its plan from the current
state, and the hand states it would command over the next `HORIZON` ticks
are the label.

Plain BC only ever sees states the expert visits; a cloned policy that
drifts a few millimetres off them meets states it has never seen.  DAgger
fixes that: in later rounds the learned policy drives (with probability
1 - beta per tick), the expert still labels every state, and the network is
retrained on everything collected so far.

Collection and evaluation charts are disjoint: `bc-train` only.  Oracle
timing noise and detection latency are randomized per chart, so the policy
also sees imperfect perception.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .. import PROJECT_ROOT, splits
from ..config import DEFAULT, Config
from .features import ACT_DIM, HORIZON, OBS_DIM, encode_action, encode_obs, hand_from_plan, hand_from_pose


class _Recorder:
    """Policy wrapper: labels every tick with the expert's plan, and plays
    the expert's or the learner's plan."""

    def __init__(self, cfg: Config, learner=None, beta: float = 1.0, seed: int = 0):
        from ..planner import ClassicalPlanner

        self.cfg = cfg
        self.expert = ClassicalPlanner(cfg)
        self.learner = learner
        self.beta = beta
        self.rng = np.random.default_rng(seed)
        self.dt = 1.0 / cfg.perception.tick_hz
        self.obs: list[np.ndarray] = []
        self.act: list[np.ndarray] = []

    def plan(self, notes, t_now, X_WT_cmd, width_cmd):
        hand = hand_from_pose(self.cfg, X_WT_cmd, width_cmd)
        expert_plan = self.expert.plan(notes, t_now, X_WT_cmd, width_cmd)
        if t_now > 0.0:
            hands = np.array([hand_from_plan(self.cfg, expert_plan, t_now + k * self.dt) for k in range(1, HORIZON + 1)])
            self.obs.append(encode_obs(self.cfg, notes, t_now, hand))
            self.act.append(encode_action(self.cfg, hands, hand))
        if self.learner is not None and self.rng.random() >= self.beta:
            return self.learner.plan(notes, t_now, X_WT_cmd, width_cmd)
        return expert_plan


def rollout(job: dict) -> tuple[np.ndarray, np.ndarray]:
    """One chart through brain + recorder; returns (obs, act)."""
    from pydrake.all import DiagramBuilder, RigidTransform, Simulator

    from ..brain import RhythmBrain, initial_tip_pose
    from ..perception import OracleDetector

    rng = np.random.default_rng(job["seed"])
    cfg = dataclasses.replace(
        DEFAULT,
        perception=dataclasses.replace(
            DEFAULT.perception,
            timing_noise_std=float(rng.choice([0.0, 0.0, 0.01, 0.02])),
            latency=float(rng.choice([0.0, 0.0, 0.05, 0.1])),
        ),
    )
    chart = splits.load(job["chart"])
    learner = None
    if job.get("learner"):
        from .policy import BCPolicy

        learner = BCPolicy.load(job["learner"], cfg)
    rec = _Recorder(cfg, learner, job.get("beta", 1.0), job["seed"])
    X_WG0 = initial_tip_pose(cfg) @ RigidTransform([0.0, -cfg.gripper.tip_offset, 0.0])
    builder = DiagramBuilder()
    builder.AddSystem(RhythmBrain(OracleDetector(chart, cfg, seed=job["seed"]), rec, cfg, X_WG0))
    sim = Simulator(builder.Build())
    sim.AdvanceTo(chart.end_time + 1.0)
    return np.array(rec.obs, dtype=np.float32), np.array(rec.act, dtype=np.float32)


def collect(charts: list[str], learner: str | None, beta: float, seed: int, workers: int):
    jobs = [{"chart": c, "learner": learner, "beta": beta, "seed": seed * 100_000 + i} for i, c in enumerate(charts)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(rollout, jobs))
    return np.concatenate([o for o, _ in res]), np.concatenate([a for _, a in res])


def fit(obs: np.ndarray, act: np.ndarray, out: Path, epochs: int, hidden: int, layers: int, device: str) -> dict:
    import torch

    from .policy import BCNet

    torch.manual_seed(0)
    act = act.reshape(len(act), -1)
    net = BCNet(hidden, layers)
    net.obs_mean.copy_(torch.from_numpy(obs.mean(0)))
    net.obs_std.copy_(torch.from_numpy(obs.std(0) + 1e-3))
    net.act_mean.copy_(torch.from_numpy(act.mean(0)))
    net.act_std.copy_(torch.from_numpy(act.std(0) + 1e-3))
    net.to(device)
    # Hold out 5 % of samples to watch for overfitting.
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(obs))
    n_val = max(1, len(obs) // 20)
    val, tr = perm[:n_val], perm[n_val:]
    O = torch.from_numpy(obs).to(device)
    A = ((torch.from_numpy(act).to(device)) - net.act_mean) / net.act_std
    # Near-term steps are what the robot actually plays: weight them up.
    w = torch.linspace(2.0, 1.0, HORIZON, device=device).repeat_interleave(ACT_DIM)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = epochs * (len(tr) // 1024 + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
    hist = []
    for ep in range(epochs):
        net.train()
        order = torch.from_numpy(rng.permutation(tr)).to(device)
        tot = 0.0
        for i in range(0, len(order), 1024):
            idx = order[i : i + 1024]
            loss = ((net(O[idx]) - A[idx]) ** 2 * w).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            vidx = torch.from_numpy(val).to(device)
            vl = (((net(O[vidx]) - A[vidx]) ** 2) * w).mean().item()
            # Error of the next-tick fingertip height in millimetres, the
            # quantity that decides whether a touch registers.
            pred = net.act(O[vidx])
            true = A[vidx] * net.act_std + net.act_mean
            z_mm = 10 * (pred[:, 0, 1] - true.view(-1, HORIZON, ACT_DIM)[:, 0, 1]).abs().mean().item()
        hist.append({"epoch": ep, "train": tot / len(tr), "val": vl, "val_z_next_mm": z_mm})
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:3d}  train {tot / len(tr):.4f}  val {vl:.4f}  next-tick z error {z_mm:.2f} mm", flush=True)
    net.cpu()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": {"hidden": hidden, "layers": layers}, "state_dict": net.state_dict()}, out)
    return {"samples": len(obs), "history": hist}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "bc_policy.pt"))
    ap.add_argument("--split", default="bc-train")
    ap.add_argument("--dagger-rounds", type=int, default=2)
    ap.add_argument("--dagger-charts", type=int, default=100, help="charts collected per DAgger round")
    ap.add_argument("--beta", type=float, default=0.2, help="probability the expert drives a tick in DAgger rounds")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    out = Path(args.out)
    names = splits.SPLITS[args.split]
    log = {"args": vars(args), "rounds": []}
    t0 = time.time()
    print(f"[bc] collecting expert demos on {len(names)} charts", flush=True)
    O, A = collect(names, None, 1.0, 0, args.workers)
    print(f"[bc] {len(O)} samples ({time.time() - t0:.0f} s)", flush=True)
    for r in range(args.dagger_rounds + 1):
        path = out if r == args.dagger_rounds else out.with_name(f"{out.stem}_r{r}{out.suffix}")
        print(f"[bc] round {r}: training on {len(O)} samples -> {path}", flush=True)
        info = fit(O, A, path, args.epochs, args.hidden, args.layers, args.device)
        log["rounds"].append({"round": r, "model": str(path), "samples": info["samples"], "final": info["history"][-1]})
        if r == args.dagger_rounds:
            break
        rng = np.random.default_rng(r + 1)
        sub = list(rng.choice(names, size=min(args.dagger_charts, len(names)), replace=False))
        print(f"[bc] DAgger round {r + 1}: learner drives, expert labels, {len(sub)} charts", flush=True)
        o, a = collect(sub, str(path), args.beta, r + 1, args.workers)
        O, A = np.concatenate([O, o]), np.concatenate([A, a])
    log["wall_s"] = time.time() - t0
    out.with_suffix(".json").write_text(json.dumps(log, indent=1))
    print(f"[bc] done in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
