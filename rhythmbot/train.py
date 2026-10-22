"""Train the note detector: game frame in, upcoming notes per lane out.

    python -m rhythmbot.train --train data/train --val data/val --epochs 12
    python -m rhythmbot.train --eval-only --model results/note_detector.pt --val data/val

Model
-----
A small CNN whose feature map is resampled into one column per lane,
followed by an MLP shared across lanes.  For every lane it predicts `K` slots, ordered by time to hit, each
holding the nearest not-yet-active note heads in that lane:

    [presence_logit, kind_logits(3), time_to_hit, duration, lane@25%, lane@50%, lane@75%, lane@100%]

The four lane values are the note's lane at those fractions of its
duration (constant for taps and holds; the path for slides).  Regression
targets are normalised (time by the highway's approach time, durations
log1p-scaled up to MAX_DURATION, lanes by n_lanes-1).

One more slot per lane (index K) reports the hold or slide being played
*now* in that lane, i.e. whose head is pinned on the judgment line:

    [active_logit, kind_logits(hold, slide), time_to_end, unused...]

This is what lets the brain keep a long hold down: before the hit the end
of a hold longer than the approach time is off-screen and its duration is a
guess, and only while it is being played does the end come into view.

The exported TorchScript module takes the full-resolution frame as a
(B, 3, H, W) float tensor in [0, 1] and downsamples inside `forward`, so
`perception.TorchDetector` needs no preprocessing code of its own.  A JSON
sidecar next to the .pt file records the normalisation constants.

Training runs on Apple MPS when available, else CPU; the whole thing takes
minutes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .chart import HOLD, SLIDE, TAP
from .config import DEFAULT
from .game import PATH_FRACTIONS

KINDS = [TAP, HOLD, SLIDE]
K_SLOTS = 3
MAX_DURATION = 8.0  # durations are log1p-scaled: d_n = log1p(d) / log1p(MAX)
DOWNSAMPLE = 2
N_PATH = len(PATH_FRACTIONS)
N_REG = 2 + N_PATH  # time_to_hit, duration, lane at each path fraction
N_OUT = 1 + 3 + N_REG  # presence, kind logits, regressions
T_COLS = 2 + N_REG  # target columns: presence, kind, regressions
ACTIVE_KINDS = [HOLD, SLIDE]  # kinds of the active slot (index K_SLOTS)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def norm_duration(d: float) -> float:
    return float(np.log1p(min(max(d, 0.0), MAX_DURATION)) / np.log1p(MAX_DURATION))


def denorm_duration(x) -> float:
    return float(np.expm1(np.clip(x, 0.0, 1.0) * np.log1p(MAX_DURATION)))


def build_targets(notes: list[dict], n_lanes: int, approach: float) -> np.ndarray:
    """(n_lanes, K_SLOTS + 1, T_COLS).

    Slots 0..K-1 (upcoming): presence, kind_idx, tth_n, dur_n, path lanes_n.
    Slot K (active): presence, active kind_idx, time_to_end_n, zeros.
    """
    tgt = np.zeros((n_lanes, K_SLOTS + 1, T_COLS), dtype=np.float32)
    per_lane: dict[int, list[dict]] = {}
    for n in notes:
        if n.get("active") or n["time_to_hit"] <= 0.0:
            if n["kind"] in ACTIVE_KINDS and n.get("time_to_end", 0.0) > 0.0:
                lane_now = int(round(n.get("lane_now", n["lane"])))
                if 0 <= lane_now < n_lanes:
                    tgt[lane_now, K_SLOTS, :3] = [
                        1.0,
                        ACTIVE_KINDS.index(n["kind"]),
                        norm_duration(n["time_to_end"]),
                    ]
            continue
        lane = int(round(n["lane"]))
        if 0 <= lane < n_lanes:
            per_lane.setdefault(lane, []).append(n)
    lane_scale = max(n_lanes - 1, 1)
    for lane, ns in per_lane.items():
        ns.sort(key=lambda n: n["time_to_hit"])
        for k, n in enumerate(ns[:K_SLOTS]):
            path = n.get("path_lanes") or [n.get("end_lane", n["lane"])] * N_PATH
            tgt[lane, k, :4] = [
                1.0,
                KINDS.index(n["kind"]),
                n["time_to_hit"] / approach,
                norm_duration(n["duration"]),
            ]
            tgt[lane, k, 4:] = np.asarray(path, dtype=np.float32) / lane_scale
    return tgt


def load_split(root: Path, n_lanes: int, approach: float, limit: int | None = None):
    """Frames as uint8 (N, 3, H/DS, W/DS) and targets (N, n_lanes, K+1, T_COLS)."""
    lines = (root / "labels.jsonl").read_text().splitlines()
    if limit:
        lines = lines[:limit]
    frames, targets = [], []
    for line in lines:
        rec = json.loads(line)
        img = np.asarray(Image.open(root / rec["frame"]).convert("RGB"))
        x = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        x = F.avg_pool2d(x, DOWNSAMPLE)
        frames.append(x.round().to(torch.uint8).squeeze(0))
        targets.append(torch.from_numpy(build_targets(rec["notes"], n_lanes, approach)))
    return torch.stack(frames), torch.stack(targets)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=2, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class NoteNet(nn.Module):
    def __init__(self, n_lanes: int, rows: int = 12, k_slots: int = K_SLOTS):
        super().__init__()
        self.n_lanes = n_lanes
        self.k_slots = k_slots
        self.downsample = DOWNSAMPLE
        self.n_out = N_OUT
        self.backbone = nn.Sequential(
            conv_block(3, 24), conv_block(24, 48), conv_block(48, 96), conv_block(96, 128)
        )
        self.rows = rows
        self.head = nn.Sequential(
            nn.Linear(128 * rows, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, (k_slots + 1) * N_OUT),
        )

    @torch.jit.export
    def head_forward(self, x_small: torch.Tensor) -> torch.Tensor:
        """x_small: (B, 3, h, w) float in [0, 1], already downsampled."""
        # Resample the feature map to one column per lane.  (Adaptive
        # average pooling would be the obvious choice, but MPS only supports
        # it for divisible sizes.)
        f = F.interpolate(
            self.backbone(x_small), size=(self.rows, self.n_lanes), mode="bilinear",
            align_corners=False,
        )  # (B, C, rows, n_lanes)
        f = f.permute(0, 3, 1, 2).flatten(2)  # (B, n_lanes, C*rows)
        out = self.head(f)  # (B, n_lanes, (K+1)*N_OUT)
        return out.view(out.shape[0], self.n_lanes, self.k_slots + 1, self.n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) float in [0, 1], full resolution."""
        return self.head_forward(F.avg_pool2d(x, self.downsample))


def loss_fn(pred: torch.Tensor, tgt: torch.Tensor) -> tuple[torch.Tensor, dict]:
    zero = pred.sum() * 0.0
    # Upcoming slots.
    up_p, up_t = pred[:, :, :K_SLOTS], tgt[:, :, :K_SLOTS]
    presence = up_t[..., 0]
    bce = F.binary_cross_entropy_with_logits(up_p[..., 0], presence)
    mask = presence > 0.5
    if mask.any():
        ce = F.cross_entropy(up_p[..., 1:4][mask], up_t[..., 1][mask].long())
        reg = F.smooth_l1_loss(up_p[..., 4:][mask], up_t[..., 2:][mask], beta=0.02)
    else:
        ce = reg = zero
    # Active slot.
    ac_p, ac_t = pred[:, :, K_SLOTS], tgt[:, :, K_SLOTS]
    a_bce = F.binary_cross_entropy_with_logits(ac_p[..., 0], ac_t[..., 0])
    a_mask = ac_t[..., 0] > 0.5
    if a_mask.any():
        a_ce = F.cross_entropy(ac_p[..., 1:3][a_mask], ac_t[..., 1][a_mask].long())
        a_reg = F.smooth_l1_loss(ac_p[..., 3][a_mask], ac_t[..., 2][a_mask], beta=0.02)
    else:
        a_ce = a_reg = zero
    total = bce + 0.5 * ce + 5.0 * reg + a_bce + 0.5 * a_ce + 5.0 * a_reg
    return total, {"bce": bce.item(), "ce": ce.item(), "reg": reg.item(), "a_reg": a_reg.item()}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, frames, targets, device, approach: float, n_lanes: int, batch=256):
    model.eval()
    tp = fp = fn = 0
    kind_ok = kind_n = 0
    tth_err, dur_err, path_err = [], [], []
    a_tp = a_fp = a_fn = 0
    tte_err, offscreen_ok = [], []
    for i in range(0, len(frames), batch):
        x = frames[i : i + batch].to(device).float() / 255.0
        t_all = targets[i : i + batch].to(device)
        p_all = model.head_forward(x)
        # Active slot.
        ap_, at_ = p_all[:, :, K_SLOTS], t_all[:, :, K_SLOTS]
        a_pres, a_true = torch.sigmoid(ap_[..., 0]) > 0.5, at_[..., 0] > 0.5
        a_tp += (a_pres & a_true).sum().item()
        a_fp += (a_pres & ~a_true).sum().item()
        a_fn += (~a_pres & a_true).sum().item()
        am = a_pres & a_true
        if am.any():
            scale = float(np.log1p(MAX_DURATION))
            tte_p = torch.expm1(ap_[..., 3][am].clamp(0, 1) * scale)
            tte_t = torch.expm1(at_[..., 2][am] * scale)
            visible = tte_t <= approach
            tte_err.append((tte_p - tte_t).abs()[visible].cpu())
            # Off-screen ends only need to be reported as off-screen.
            offscreen_ok.append((tte_p[~visible] >= 0.9 * approach).float().cpu())
        # Upcoming slots.
        p, t = p_all[:, :, :K_SLOTS], t_all[:, :, :K_SLOTS]
        present = torch.sigmoid(p[..., 0]) > 0.5
        truth = t[..., 0] > 0.5
        tp += (present & truth).sum().item()
        fp += (present & ~truth).sum().item()
        fn += (~present & truth).sum().item()
        m = present & truth
        if m.any():
            kind_ok += (p[..., 1:4][m].argmax(-1) == t[..., 1][m].long()).sum().item()
            kind_n += m.sum().item()
            tth_err.append(((p[..., 4][m] - t[..., 2][m]).abs() * approach).cpu())
            is_long = t[..., 1][m] > 0.5
            scale = float(np.log1p(MAX_DURATION))
            dur_err.append(
                (torch.expm1(p[..., 5][m].clamp(0, 1) * scale) - torch.expm1(t[..., 3][m] * scale))
                .abs()[is_long]
                .cpu()
            )
            path_err.append(
                ((p[..., 6:][m] - t[..., 4:][m]).abs().mean(-1) * (n_lanes - 1))[is_long].cpu()
            )
    cat = lambda xs: torch.cat(xs) if xs else torch.zeros(0)
    tth, dur, end = cat(tth_err), cat(dur_err), cat(path_err)
    tte, off = cat(tte_err), cat(offscreen_ok)
    return {
        "active_precision": a_tp / max(a_tp + a_fp, 1),
        "active_recall": a_tp / max(a_tp + a_fn, 1),
        "active_end_mae_ms": 1000 * tte.mean().item() if len(tte) else None,
        "active_offscreen_ok": off.mean().item() if len(off) else None,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "kind_accuracy": kind_ok / max(kind_n, 1),
        "time_to_hit_mae_ms": 1000 * tth.mean().item() if len(tth) else None,
        "time_to_hit_p95_ms": 1000 * tth.quantile(0.95).item() if len(tth) else None,
        "duration_mae_ms": 1000 * dur.mean().item() if len(dur) else None,
        "path_lane_mae": end.mean().item() if len(end) else None,
        "n_true_notes": int(tp + fn),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def export(model: NoteNet, path: Path, meta: dict) -> None:
    model.eval().cpu()
    scripted = torch.jit.script(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(path))
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="data/train")
    ap.add_argument("--val", default="data/val")
    ap.add_argument("--model", default="results/note_detector.pt")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, help="use only this many frames (debug)")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args(argv)

    cfg = DEFAULT
    n_lanes, approach = cfg.layout.n_lanes, cfg.layout.approach_time
    device = pick_device()
    print(f"device {device}")

    t0 = time.time()
    val_x, val_y = load_split(Path(args.val), n_lanes, approach, args.limit)
    print(f"val   {tuple(val_x.shape)} in {time.time() - t0:.0f}s")
    meta = {
        "n_lanes": n_lanes,
        "k_slots": K_SLOTS,
        "approach_time": approach,
        "max_duration": MAX_DURATION,
        "duration_norm": "log1p",
        "downsample": DOWNSAMPLE,
        "path_fractions": list(PATH_FRACTIONS),
        "kinds": KINDS,
        "active_slot": True,
        "active_kinds": ACTIVE_KINDS,
        "frame_size": list(cfg.perception.frame_size),
    }

    if args.eval_only:
        model = torch.jit.load(args.model, map_location="cpu").to(device)
        print(json.dumps(evaluate(model, val_x, val_y, device, approach, n_lanes), indent=1))
        return

    t0 = time.time()
    tr_x, tr_y = load_split(Path(args.train), n_lanes, approach, args.limit)
    print(f"train {tuple(tr_x.shape)} in {time.time() - t0:.0f}s")

    model = NoteNet(n_lanes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(tr_x) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(tr_x))
        tot, n = 0.0, 0
        t0 = time.time()
        for i in range(0, len(perm), args.batch):
            idx = perm[i : i + args.batch]
            x = tr_x[idx].to(device).float() / 255.0
            y = tr_y[idx].to(device)
            loss, parts = loss_fn(model.head_forward(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        ev = evaluate(model, val_x, val_y, device, approach, n_lanes)
        ms = lambda v: "   -" if v is None else f"{v:4.0f}"
        print(
            f"epoch {epoch:2d}  loss {tot / n:.4f}  "
            f"P {ev['precision']:.3f} R {ev['recall']:.3f} kind {ev['kind_accuracy']:.3f}  "
            f"tth MAE {ms(ev['time_to_hit_mae_ms'])} ms  dur MAE {ms(ev['duration_mae_ms'])} ms  "
            f"path-lane MAE {ev['path_lane_mae'] if ev['path_lane_mae'] is None else round(ev['path_lane_mae'], 2)}  "
            f"| active P {ev['active_precision']:.3f} R {ev['active_recall']:.3f} "
            f"end MAE {ms(ev['active_end_mae_ms'])} ms"
            f"  ({time.time() - t0:.0f}s)"
        )

    ev = evaluate(model, val_x, val_y, device, approach, n_lanes)
    meta["val"] = ev
    export(model, Path(args.model), meta)
    print(f"model -> {args.model}")
    print(json.dumps(ev, indent=1))


if __name__ == "__main__":
    main()
