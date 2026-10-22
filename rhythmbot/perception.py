"""Perception: from what is on the game screen to a list of upcoming notes.

The contract every detector satisfies:

    detector.detect(frame, t) -> list[DetectedNote]

`frame` is the (H, W, 3) uint8 game screen rendered at time `t` (detectors
that do not look at pixels may ignore it), and each `DetectedNote` says what
kind of note it is, which lane, when it should be hit (absolute time), how
long it lasts if known, and for slides the path.  `note_id` is optional: the
oracle fills it from the chart; image detectors leave it at -1 and the brain
matches detections across frames by lane and time.

Three implementations:

    OracleDetector            -- reads the chart; perfect perception with
                                 optional timing noise and latency.  The
                                 upper bound any vision model is measured
                                 against.
    ColorSegmentationDetector -- GuitarHeroBot's approach brought over:
                                 threshold the known note colors, label the
                                 connected blobs, read lane from blob x and
                                 time from blob y.  Uses numpy + scipy only.
    TorchDetector             -- the trained model from `train.py`.  `torch`
                                 is imported lazily, so the rest of the
                                 pipeline runs without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .chart import HOLD, SLIDE, TAP, Chart
from .config import Config
from .game import COLORS, HEAD_TIME, FrameRenderer


@dataclass
class DetectedNote:
    kind: str
    lane: float
    hit_time: float
    duration: float = 0.0
    # True when the whole hold/slide was visible, so `duration` is final.
    duration_final: bool = True
    # Slide path as (offset from hit_time, lane); empty otherwise.
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    note_id: int = -1
    confidence: float = 1.0
    # True when the note is already being played (its head sits on the
    # judgment line): `hit_time` is then just "now" and only `end_time` is
    # informative.
    active: bool = False

    @property
    def end_time(self) -> float:
        return self.hit_time + self.duration


class NoteDetector:
    """Interface.  Subclasses override `detect`; `needs_frames` tells the
    brain whether rendering a frame each tick is worth the cost."""

    needs_frames: bool = True

    def detect(self, frame: np.ndarray | None, t: float) -> list[DetectedNote]:
        raise NotImplementedError


class OracleDetector(NoteDetector):
    """Reads the chart: every note currently on the highway, exactly.

    `timing_noise_std` jitters each note's reported hit time (fresh noise
    every frame, so the brain's tracker has something to average), and
    `latency` delays when a note is first reported after it appears.
    """

    needs_frames = False

    def __init__(self, chart: Chart, cfg: Config, seed: int = 0):
        self.chart = chart
        self.approach = cfg.layout.approach_time
        self.noise = cfg.perception.timing_noise_std
        self.latency = cfg.perception.latency
        self.rng = np.random.default_rng(seed)

    def detect(self, frame, t):
        out = []
        for n in self.chart.visible(t - self.latency, self.approach):
            if n.end_time < t:
                continue
            jitter = self.rng.normal(0.0, self.noise) if self.noise > 0 else 0.0
            out.append(
                DetectedNote(
                    kind=n.kind,
                    lane=n.lane,
                    hit_time=n.time + jitter,
                    duration=n.duration,
                    duration_final=True,
                    waypoints=list(n.waypoints),
                    note_id=n.id,
                    active=n.time <= t,
                )
            )
        return out


class ColorSegmentationDetector(NoteDetector):
    """Finds notes in a rendered frame by their colors.

    For each note kind, pixels within `tolerance` of its color are masked and
    labelled into connected components (scipy.ndimage.label stands in for
    OpenCV's findContours).  A component's bottom edge is the note's hit row,
    its horizontal centre its lane, and for holds and slides its top edge
    the end of the note.  A slide's path is read off by sampling the
    component's centre column every few rows from bottom to top.

    Blobs touching the top edge of the frame are still arriving, so their
    duration is reported as not final.
    """

    needs_frames = True

    def __init__(self, cfg: Config, tolerance: int = 60, min_pixels: int = 12):
        self.renderer = FrameRenderer(cfg)
        self.tol = tolerance
        self.min_pixels = min_pixels

    def _mask(self, frame: np.ndarray, color) -> np.ndarray:
        diff = np.abs(frame.astype(np.int16) - np.array(color, dtype=np.int16))
        return diff.max(axis=2) < self.tol

    def detect(self, frame, t):
        from scipy import ndimage

        R = self.renderer
        out: list[DetectedNote] = []
        half_head_rows = 0.5 * HEAD_TIME * R.px_per_sec
        for kind, color in COLORS.items():
            mask = self._mask(frame, color)
            labels, n = ndimage.label(mask)
            if n == 0:
                continue
            for sl in ndimage.find_objects(labels):
                comp = labels[sl] > 0
                if comp.sum() < self.min_pixels:
                    continue
                rows = slice(sl[0].start, sl[0].stop)
                cols = slice(sl[1].start, sl[1].stop)
                bottom = rows.stop - 1
                top = rows.start
                # The head is centred on the hit row and half a head thick.
                hit_row = bottom - half_head_rows + 0.5
                active = hit_row >= R.hit_row - 1.0
                hit_time = t if active else t + R.row_to_dt(hit_row)
                if kind == TAP:
                    # Head centre column -> lane.
                    cx = 0.5 * (cols.start + cols.stop - 1)
                    out.append(DetectedNote(TAP, round(R.px_to_lane(cx)), hit_time, active=active))
                    continue
                clipped = top <= 0
                end_row = top + half_head_rows  # body top ~ end time
                duration = max(R.row_to_dt(end_row) - R.row_to_dt(hit_row), 0.0)
                # Lane from the bottom rows (the head), which are the widest.
                head_rows = comp[-int(max(2, 2 * half_head_rows)) :]
                xs = np.nonzero(head_rows.any(axis=0))[0]
                lane0 = R.px_to_lane(cols.start + xs.mean())
                if kind == HOLD:
                    out.append(
                        DetectedNote(
                            HOLD, round(lane0), hit_time, duration,
                            duration_final=not clipped, active=active,
                        )
                    )
                else:
                    wps = self._slide_path(comp, rows, cols, hit_row, lane0)
                    out.append(
                        DetectedNote(
                            SLIDE,
                            round(lane0),
                            hit_time,
                            duration,
                            duration_final=not clipped,
                            waypoints=wps,
                            active=active,
                        )
                    )
        return out

    def _slide_path(self, comp, rows, cols, hit_row, lane0, step_rows: int = 6):
        """Sample the body's centre column up the component -> (offset, lane)."""
        R = self.renderer
        wps = [(0.0, float(round(lane0)))]
        h = comp.shape[0]
        head_h = int(HEAD_TIME * R.px_per_sec) + 1
        for r in range(h - head_h, -1, -step_rows):
            xs = np.nonzero(comp[r])[0]
            if len(xs) == 0:
                continue
            lane = R.px_to_lane(cols.start + xs.mean())
            off = R.row_to_dt(rows.start + r) - R.row_to_dt(hit_row)
            if off > wps[-1][0] + 1e-3:
                wps.append((float(off), float(lane)))
        return wps


class TorchDetector(NoteDetector):
    """The trained vision model (see `train.py`).

    Loads a TorchScript `NoteNet` and the JSON sidecar written next to it
    with the normalisation constants.  The model takes the full-resolution
    frame as a (1, 3, H, W) float tensor in [0, 1] and returns, per lane and
    per slot, `[presence_logit, kind_logits(3), time_to_hit, duration,
    lane at 25/50/75/100% of the duration]` (regressions normalised; see
    `train.py`).  `decode` turns that into `DetectedNote`s, with a slide's
    waypoints taken from the four predicted lanes.

    Models trained with an active slot (sidecar `active_slot: true`) also
    report, per lane, the hold or slide being played there and how long it
    has left; those come out as `active=True` detections, which the brain
    uses to keep a long hold down until its end is actually in view.
    """

    needs_frames = True

    def __init__(self, model_path: str, cfg: Config, device: str = "cpu", threshold=0.5):
        import json
        from pathlib import Path

        import torch  # lazy: optional dependency

        self.torch = torch
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        self.device = device
        self.threshold = threshold
        self.cfg = cfg
        sidecar = Path(model_path).with_suffix(".json")
        meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        self.approach = meta.get("approach_time", cfg.layout.approach_time)
        self.max_duration = meta.get("max_duration", 2.0)
        self.duration_norm = meta.get("duration_norm", "linear")
        self.kinds = meta.get("kinds", [TAP, HOLD, SLIDE])
        self.n_lanes = meta.get("n_lanes", cfg.layout.n_lanes)
        self.path_fractions = meta.get("path_fractions", [0.25, 0.5, 0.75, 1.0])
        self.k_slots = meta.get("k_slots", 2)
        self.active_slot = bool(meta.get("active_slot", False))
        self.active_kinds = meta.get("active_kinds", [HOLD, SLIDE])

    def detect(self, frame, t):
        x = self.torch.from_numpy(np.array(frame, copy=True)).permute(2, 0, 1).float().div(255.0)[None]
        with self.torch.no_grad():
            pred = self.model(x.to(self.device))[0].cpu().numpy()  # (n_lanes, K, 7)
        return self.decode(pred, t)

    def _duration(self, x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        if self.duration_norm == "log1p":
            return float(np.expm1(x * np.log1p(self.max_duration)))
        return x * self.max_duration

    def decode(self, pred: np.ndarray, t: float) -> list[DetectedNote]:
        out = []
        n_upcoming = self.k_slots if self.active_slot else pred.shape[1]
        if self.active_slot:
            out.extend(self._decode_active(pred[:, self.k_slots], t))
        for lane in range(pred.shape[0]):
            for slot in range(n_upcoming):
                row = pred[lane, slot]
                conf = float(1.0 / (1.0 + np.exp(-row[0])))
                if conf < self.threshold:
                    continue
                kind = self.kinds[int(np.argmax(row[1:4]))]
                tth = float(np.clip(row[4], 0.0, 1.0)) * self.approach
                duration = 0.0 if kind == TAP else self._duration(row[5])
                waypoints = []
                if kind == SLIDE:
                    duration = max(duration, 0.1)
                    path = np.clip(row[6 : 6 + len(self.path_fractions)], 0.0, 1.0) * (self.n_lanes - 1)
                    waypoints = [(0.0, float(lane))] + [
                        (f * duration, float(l)) for f, l in zip(self.path_fractions, path)
                    ]
                out.append(
                    DetectedNote(
                        kind=kind,
                        lane=float(lane),
                        hit_time=t + tth,
                        duration=duration,
                        # The whole note is on screen once its end is within
                        # the highway; until then the duration is a guess.
                        duration_final=(tth + duration) <= self.approach,
                        waypoints=waypoints,
                        confidence=conf,
                    )
                )
        return out


    def _decode_active(self, rows: np.ndarray, t: float) -> list[DetectedNote]:
        """rows: (n_lanes, N_OUT) active slot -> active DetectedNotes."""
        out = []
        for lane, row in enumerate(rows):
            conf = float(1.0 / (1.0 + np.exp(-row[0])))
            if conf < self.threshold:
                continue
            kind = self.active_kinds[int(np.argmax(row[1:3]))]
            tte = self._duration(row[3])
            # The end is in view only if it is below the top of the frame;
            # otherwise all the frame shows is that it is at least that far.
            end_visible = tte < 0.9 * self.approach
            out.append(
                DetectedNote(
                    kind=kind,
                    lane=float(lane),
                    hit_time=t,
                    duration=tte if end_visible else max(tte, self.approach),
                    duration_final=end_visible,
                    active=True,
                    confidence=conf,
                )
            )
        return out


def make_detector(name: str, chart: Chart, cfg: Config, model_path: str | None = None):
    if name == "oracle":
        return OracleDetector(chart, cfg)
    if name == "color":
        return ColorSegmentationDetector(cfg)
    if name == "torch":
        if model_path is None:
            raise ValueError("--model is required for the torch detector")
        return TorchDetector(model_path, cfg)
    raise ValueError(f"unknown detector {name!r}")
