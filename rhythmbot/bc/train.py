"""Behavior cloning from the classical planner, with DAgger.

    # state policy on the vision model's tracked notes, MLP or diffusion head
    python -m rhythmbot.bc.train --encoder state --head mlp --detector torch --out results/bc/state_mlp.pt
    python -m rhythmbot.bc.train --encoder state --head diffusion --detector torch --out results/bc/state_diffusion.pt
    # pixel policy: the game frame in, perception learned end to end
    python -m rhythmbot.bc.train --encoder pixels --head diffusion --out results/bc/pixels_diffusion.pt
    # teacher-student: a state policy on oracle notes, distilled into a pixel policy
    python -m rhythmbot.bc.train --encoder state --head diffusion --out results/bc/teacher.pt
    python -m rhythmbot.bc.train --encoder pixels --head diffusion --teacher results/bc/teacher.pt \
        --out results/bc/student.pt
    python -m rhythmbot.bc.train ... --dagger-rounds 0           # plain BC

Demonstrations are collected without the arm: the brain alone runs the
detector, the tracker and a policy, as fast as it can, and the policy's own
commanded pose is what it observes next (the brain never reads the robot's
measured pose), so nothing is lost by leaving out the physics.  At every
tick the classical planner is asked for its plan from the current state and
the hand states it would command over the next `horizon` ticks are the
label.  With `--detector torch` the notes come from the trained vision model,
so the state policy is trained on the observations it will get when it is
deployed with it, and the expert labels are what the classical planner does
with that perception.  Pixel samples store only (chart, time): frames are
re-rendered when training (0.04 ms each).

Plain BC only ever sees states the expert visits; DAgger lets the learned
policy drive (with probability 1 - beta per tick), the expert still labels
every state, and the network is retrained on everything collected so far.

**Distillation (`--teacher`).** Instead of the classical planner, a trained
state policy labels the states: it sees the exact (oracle) tracked notes,
privileged information the pixel student does not get, and its action chunk
for the current state is the label.  The first round lets the teacher drive;
later rounds let the student drive (DAgger), so the student learns to
recover from its own mistakes with the teacher's help.

Collection uses `bc-train` charts only; oracle timing noise and latency are
randomized per chart.
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
from .features import HAND_DIM, HORIZON, encode_action, encode_obs, hand_from_plan, hand_from_pose, hand_obs

VISION_MODEL = PROJECT_ROOT / "results" / "note_detector.pt"


class _Recorder:
    """Policy wrapper: labels every tick with the expert's plan, and plays
    the expert's or the learner's plan."""

    def __init__(self, cfg: Config, learner=None, beta: float = 1.0, seed: int = 0, teacher=None):
        from ..planner import ClassicalPlanner

        self.cfg = cfg
        self.expert = teacher if teacher is not None else ClassicalPlanner(cfg)
        self.teacher = teacher
        self.learner = learner
        self.wants_frames = bool(learner is not None and learner.wants_frames)
        self.frame = None
        self.beta = beta
        self.rng = np.random.default_rng(seed)
        self.dt = 1.0 / cfg.perception.tick_hz
        self.obs: list[np.ndarray] = []
        self.hand: list[np.ndarray] = []
        self.act: list[np.ndarray] = []
        self.t: list[float] = []

    def plan(self, notes, t_now, X_WT_cmd, width_cmd):
        hand = hand_from_pose(self.cfg, X_WT_cmd, width_cmd)
        obs = encode_obs(self.cfg, notes, t_now, hand)
        if self.teacher is not None:
            # The teacher's freshly sampled chunk for this state is the label
            # (not its cached plan, which may be a few ticks old).
            self.teacher._plan = None
            expert_plan = self.teacher.plan(notes, t_now, X_WT_cmd, width_cmd)
            label = self.teacher.last_action
        else:
            expert_plan = self.expert.plan(notes, t_now, X_WT_cmd, width_cmd)
            hands = np.array([hand_from_plan(self.cfg, expert_plan, t_now + k * self.dt) for k in range(1, HORIZON + 1)])
            label = encode_action(self.cfg, hands, hand)
        if t_now > 0.0:
            self.obs.append(obs)
            self.hand.append(hand_obs(self.cfg, hand))
            self.act.append(label)
            self.t.append(t_now)
        if self.learner is not None and self.rng.random() >= self.beta:
            self.learner.frame = self.frame
            return self.learner.plan(notes, t_now, X_WT_cmd, width_cmd)
        return expert_plan


def rollout(job: dict) -> dict:
    """One chart through brain + recorder."""
    from pydrake.all import DiagramBuilder, RigidTransform, Simulator

    from ..brain import GameScreen, RhythmBrain, initial_tip_pose
    from ..perception import make_detector

    rng = np.random.default_rng(job["seed"])
    cfg = DEFAULT
    if job["detector"] == "oracle":
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
    teacher = None
    if job.get("teacher"):
        from .policy import BCPolicy

        teacher = BCPolicy.load(job["teacher"], cfg)
    rec = _Recorder(cfg, learner, job.get("beta", 1.0), job["seed"], teacher)
    detector = make_detector(job["detector"], chart, cfg, str(VISION_MODEL))
    if job["detector"] == "torch":
        import torch

        torch.set_num_threads(1)
    X_WG0 = initial_tip_pose(cfg) @ RigidTransform([0.0, -cfg.gripper.tip_offset, 0.0])
    builder = DiagramBuilder()
    screen = builder.AddSystem(GameScreen(chart, cfg))
    brain = builder.AddSystem(RhythmBrain(detector, [rec], cfg, [X_WG0]))
    builder.Connect(screen.get_output_port(), brain.GetInputPort("frame"))
    sim = Simulator(builder.Build())
    sim.AdvanceTo(chart.end_time + 1.0)
    n = len(rec.t)
    return {
        "obs": np.array(rec.obs, dtype=np.float32).reshape(n, -1),
        "hand": np.array(rec.hand, dtype=np.float32).reshape(n, HAND_DIM),
        "act": np.array(rec.act, dtype=np.float32).reshape(n, HORIZON, -1),
        "t": np.array(rec.t, dtype=np.float64),
        "chart": np.full(n, job["chart_index"], dtype=np.int32),
    }


def collect(names: list[str], indices: list[int], detector: str, learner: str | None, beta: float,
            seed: int, workers: int, teacher: str | None = None) -> dict:
    jobs = [
        {"chart": names[i], "chart_index": i, "detector": detector, "learner": learner, "beta": beta,
         "seed": seed * 100_000 + i, "teacher": teacher}
        for i in indices
    ]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(rollout, jobs))
    return {k: np.concatenate([r[k] for r in res]) for k in res[0]}


def _cat(a: dict | None, b: dict) -> dict:
    return b if a is None else {k: np.concatenate([a[k], b[k]]) for k in a}


class _FrameData:
    """Re-renders the frame of each (chart, time) sample, for a DataLoader."""

    def __init__(self, names: list[str], data: dict):
        self.names = names
        self.data = data
        self._charts: dict[int, object] = {}
        self._renderer = None

    def __len__(self):
        return len(self.data["t"])

    def __getitem__(self, i):
        from ..game import FrameRenderer
        from .features import frame_input

        if self._renderer is None:
            self._renderer = FrameRenderer(DEFAULT)
        c = int(self.data["chart"][i])
        if c not in self._charts:
            self._charts[c] = splits.load(self.names[c])
        frame = frame_input(self._renderer.render(self._charts[c], float(self.data["t"][i])))
        return frame, self.data["hand"][i], self.data["act"][i]


def fit(data: dict, names: list[str], out: Path, args) -> dict:
    import torch

    from .models import BCModel

    torch.manual_seed(0)
    dev = args.device
    model = BCModel(args.encoder, args.head, HORIZON)
    act = data["act"].reshape(len(data["act"]), -1)
    model.act_mean.copy_(torch.from_numpy(act.mean(0)))
    model.act_std.copy_(torch.from_numpy(act.std(0) + 1e-3))
    if args.encoder == "state":
        model.encoder.obs_mean.copy_(torch.from_numpy(data["obs"].mean(0)))
        model.encoder.obs_std.copy_(torch.from_numpy(data["obs"].std(0) + 1e-3))
    else:
        model.encoder.hand_mean.copy_(torch.from_numpy(data["hand"].mean(0)))
        model.encoder.hand_std.copy_(torch.from_numpy(data["hand"].std(0) + 1e-3))
    model.to(dev)

    n = len(act)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    val, tr = perm[: max(1, n // 20)], perm[max(1, n // 20) :]
    bs = args.batch
    steps_per_epoch = len(tr) // bs + 1
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch)

    if args.encoder == "state":
        O = torch.from_numpy(data["obs"]).to(dev)
        A = torch.from_numpy(data["act"]).to(dev)

        def batches(idx, shuffle):
            idx = rng.permutation(idx) if shuffle else idx
            for i in range(0, len(idx), bs):
                j = torch.from_numpy(idx[i : i + bs]).to(dev)
                yield {"obs": O[j], "act": A[j]}
    else:
        ds = _FrameData(names, data)

        def batches(idx, shuffle):
            sub = torch.utils.data.Subset(ds, idx)
            dl = torch.utils.data.DataLoader(
                sub, batch_size=bs, shuffle=shuffle, num_workers=args.loader_workers,
                persistent_workers=False, pin_memory=True, drop_last=False,
            )
            for frame, hand, a in dl:
                yield {"frame": frame.to(dev, non_blocking=True), "hand": hand.to(dev), "act": a.to(dev)}

    hist = []
    for ep in range(args.epochs):
        model.train()
        tot, cnt, t0 = 0.0, 0, time.time()
        for b in batches(tr, True):
            loss = model.loss(b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(b["act"])
            cnt += len(b["act"])
        rec = {"epoch": ep, "train_loss": tot / cnt, "epoch_s": time.time() - t0}
        if ep % 5 == 0 or ep == args.epochs - 1:
            # Validation: loss, and the error of the sampled next-tick height
            # in millimetres, which decides whether a touch registers.
            model.eval()
            vl, zerr, vn = 0.0, 0.0, 0
            with torch.no_grad():
                for b in batches(val[:20_000], False):
                    vl += model.loss(b).item() * len(b["act"])
                    pred = model.act(b)
                    zerr += 10 * (pred[:, 0, 1] - b["act"][:, 0, 1]).abs().sum().item()
                    vn += len(b["act"])
            rec.update(val_loss=vl / vn, val_z_next_mm=zerr / vn)
            print(f"  epoch {ep:3d}  train {rec['train_loss']:.4f}  val {rec['val_loss']:.4f}  "
                  f"next-tick z error {rec['val_z_next_mm']:.2f} mm  ({rec['epoch_s']:.0f} s/epoch)", flush=True)
        hist.append(rec)
    model.cpu()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": model.arch, "state_dict": model.state_dict(), "exec_steps": args.exec_steps}, out)
    return {"samples": n, "history": hist}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="state", choices=["state", "pixels"])
    ap.add_argument("--head", default="mlp", choices=["mlp", "diffusion"])
    ap.add_argument("--detector", default="oracle", choices=["oracle", "torch"],
                    help="perception the demonstrations are tracked with (state encoder)")
    ap.add_argument("--teacher", help="state BC policy to distill from (oracle notes) instead of the classical planner")
    ap.add_argument("--split", default="bc-train")
    ap.add_argument("--max-charts", type=int, help="use only the first N charts of the split (smoke tests)")
    ap.add_argument("--dagger-rounds", type=int, default=2)
    ap.add_argument("--dagger-charts", type=int, default=100, help="charts collected per DAgger round")
    ap.add_argument("--beta", type=float, default=0.2, help="probability the expert drives a tick in DAgger rounds")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--exec-steps", type=int, default=None, help="ticks played per sampled chunk (default 1 mlp, 4 diffusion)")
    ap.add_argument("--workers", type=int, default=8, help="rollout processes")
    ap.add_argument("--loader-workers", type=int, default=8, help="frame-rendering DataLoader workers (pixels)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    out = Path(args.out)
    names = splits.SPLITS[args.split]
    if args.max_charts:
        names = names[: args.max_charts]
    log = {"args": vars(args), "rounds": []}
    t0 = time.time()
    print(f"[bc] collecting expert demos on {len(names)} charts ({args.detector})", flush=True)
    if args.teacher and args.detector != "oracle":
        raise SystemExit("--teacher sees oracle notes; use --detector oracle")
    data = collect(names, list(range(len(names))), args.detector, None, 1.0, 0, args.workers, args.teacher)
    print(f"[bc] {len(data['t'])} samples ({time.time() - t0:.0f} s)", flush=True)
    for r in range(args.dagger_rounds + 1):
        path = out if r == args.dagger_rounds else out.with_name(f"{out.stem}_r{r}{out.suffix}")
        print(f"[bc] round {r}: training {args.encoder}/{args.head} on {len(data['t'])} samples -> {path}", flush=True)
        info = fit(data, names, path, args)
        log["rounds"].append({"round": r, "model": str(path), "samples": info["samples"], "history": info["history"]})
        out.with_suffix(".json").write_text(json.dumps(log, indent=1))
        if r == args.dagger_rounds:
            break
        rng = np.random.default_rng(r + 1)
        sub = sorted(rng.choice(len(names), size=min(args.dagger_charts, len(names)), replace=False).tolist())
        print(f"[bc] DAgger round {r + 1}: learner drives, expert labels, {len(sub)} charts", flush=True)
        data = _cat(data, collect(names, sub, args.detector, str(path), args.beta, r + 1, args.workers, args.teacher))
    log["wall_s"] = time.time() - t0
    out.with_suffix(".json").write_text(json.dumps(log, indent=1))
    print(f"[bc] done in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
