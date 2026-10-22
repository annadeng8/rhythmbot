"""The rhythm game itself: where the notes are at time t, drawn two ways.

`FrameRenderer` draws the game screen as a 2D image, the way a player (or a
vision model) sees it: lanes as columns, notes falling toward a judgment line
near the bottom.  This is the input to `perception.NoteDetector`s and the
training data for the vision model (`dataset.py`).

`HighwayVisualizer` draws the same notes as 3D geometry in Meshcat, moving
across the touch screen in the Drake scene toward the judgment line, so the
arm can be watched playing them.

Both read the same `Chart` and `Layout`, so the two views agree.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from pydrake.all import Box, LeafSystem, Meshcat, Rgba, RigidTransform

from .chart import HOLD, SLIDE, TAP, Chart, Note
from .config import Config, Layout

# Note colors as (R, G, B).  The color detector keys on these, and the dataset
# labels name them, so keep them well apart in hue.
COLORS = {
    TAP: (0, 220, 255),
    HOLD: (70, 235, 95),
    SLIDE: (245, 85, 225),
}
BACKGROUND = (14, 14, 24)
LANE_LINE = (60, 60, 80)
HIT_LINE = (255, 240, 200)

# Drawn geometry, in lanes and seconds, shared by both renderers.
HEAD_LANE_FRACTION = 0.86  # note head width as a fraction of the lane
# Where along a hold/slide the vision model predicts the lane.
PATH_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
HEAD_TIME = 0.055  # note head thickness, in seconds of scroll
HOLD_BODY_LANE_FRACTION = 0.55
SLIDE_BODY_LANE_FRACTION = 0.45


class FrameRenderer:
    """Renders the game screen at time t as an (H, W, 3) uint8 array.

    Pixel geometry: lanes are `lane_px` wide starting at x = 0; the judgment
    line is `hit_row` pixels from the top; a note `dt` seconds from being hit
    is drawn `dt * px_per_sec` pixels above it.  `approach_time` seconds of
    highway fit above the judgment line, so a note appears at the top edge.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.layout = cfg.layout
        self.W, self.H = cfg.perception.frame_size
        self.lane_px = self.W / self.layout.n_lanes
        self.hit_row = int(self.H * 0.86)
        self.px_per_sec = self.hit_row / self.layout.approach_time

    # -- coordinate helpers (also used by the color detector) --------------
    def lane_to_px(self, lane: float) -> float:
        return (lane + 0.5) * self.lane_px

    def px_to_lane(self, x: float) -> float:
        return x / self.lane_px - 0.5

    def dt_to_row(self, time_to_hit: float) -> float:
        return self.hit_row - time_to_hit * self.px_per_sec

    def row_to_dt(self, row: float) -> float:
        return (self.hit_row - row) / self.px_per_sec

    # -- drawing -------------------------------------------------------------
    def render(self, chart: Chart, t: float) -> np.ndarray:
        img = Image.new("RGB", (self.W, self.H), BACKGROUND)
        draw = ImageDraw.Draw(img)
        for k in range(1, self.layout.n_lanes):
            x = k * self.lane_px
            draw.line([(x, 0), (x, self.H)], fill=LANE_LINE, width=1)
        draw.line([(0, self.hit_row), (self.W, self.hit_row)], fill=HIT_LINE, width=2)
        for note in chart.visible(t, self.layout.approach_time):
            self._draw_note(draw, note, t)
        return np.asarray(img)

    def _draw_note(self, draw: ImageDraw.ImageDraw, note: Note, t: float) -> None:
        color = COLORS[note.kind]
        half_head = 0.5 * HEAD_TIME * self.px_per_sec
        # Once a note is active (hold/slide in progress) its head sits on the
        # judgment line and the body shrinks toward it, as in the game.
        dt_head = max(note.time - t, 0.0)
        if note.kind == HOLD:
            x0 = self.lane_to_px(note.lane - 0.5 * HOLD_BODY_LANE_FRACTION)
            x1 = self.lane_to_px(note.lane + 0.5 * HOLD_BODY_LANE_FRACTION)
            y_top = self.dt_to_row(note.end_time - t)
            y_bot = self.dt_to_row(dt_head)
            draw.rectangle([x0, y_top, x1, y_bot], fill=color)
        elif note.kind == SLIDE:
            w = SLIDE_BODY_LANE_FRACTION * self.lane_px
            pts = [
                (self.lane_to_px(lane), self.dt_to_row(note.time + off - t))
                for off, lane in note.waypoints
            ]
            for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
                if yb > self.hit_row:
                    continue  # segment already played
                if ya > self.hit_row:
                    # Clip the segment at the judgment line.
                    f = (ya - self.hit_row) / (ya - yb)
                    xa, ya = xa + f * (xb - xa), self.hit_row
                draw.polygon(
                    [(xa - w / 2, ya), (xa + w / 2, ya), (xb + w / 2, yb), (xb - w / 2, yb)],
                    fill=color,
                )
        # Head (all kinds): a bar at the hit time, at the current lane for an
        # active slide.
        lane_head = note.lane_at(t) if note.kind == SLIDE else note.lane
        hx0 = self.lane_to_px(lane_head - 0.5 * HEAD_LANE_FRACTION)
        hx1 = self.lane_to_px(lane_head + 0.5 * HEAD_LANE_FRACTION)
        hy = self.dt_to_row(dt_head)
        draw.rectangle([hx0, hy - half_head, hx1, hy + half_head], fill=color)

    def label(self, chart: Chart, t: float) -> list[dict]:
        """Ground truth for a frame: every visible note with its pixel
        geometry, for training/evaluating a vision model."""
        out = []
        for note in chart.visible(t, self.layout.approach_time):
            d = {
                "id": note.id,
                "kind": note.kind,
                "lane": note.lane,
                "time_to_hit": note.time - t,
                "active": note.time <= t,
                "time_to_end": note.end_time - t,
                # Lane the head is in now: differs from `lane` only for a
                # slide that is being played.
                "lane_now": note.lane_at(t) if note.time <= t else note.lane,
                "duration": note.duration,
                "end_lane": note.lane_at(note.end_time),
                # Lane at 25/50/75/100% of the duration: the slide path as
                # the vision model predicts it.
                "path_lanes": [
                    note.lane_at(note.time + f * note.duration) for f in PATH_FRACTIONS
                ],
                "head_x": self.lane_to_px(note.lane),
                "head_row": self.dt_to_row(note.time - t),
            }
            if note.kind == SLIDE:
                d["waypoints_px"] = [
                    [self.lane_to_px(lane), self.dt_to_row(note.time + off - t)]
                    for off, lane in note.waypoints
                ]
            out.append(d)
        return out


class HighwayVisualizer(LeafSystem):
    """Moves note geometry across the Meshcat touch screen as time passes.

    Each note gets its geometry once at construction (a box for a tap head, a
    long box for a hold body, a line for a slide path) under
    `/highway/notes/<id>`, and a periodic publish sets its transform from the
    simulation time.  Notes that have not yet appeared or have passed are
    hidden.  This is what makes the Drake scene look like the game.
    """

    def __init__(self, chart: Chart, cfg: Config, meshcat: Meshcat | None):
        super().__init__()
        self.chart = chart
        self.layout = cfg.layout
        self.meshcat = meshcat
        self.DeclarePeriodicPublishEvent(1.0 / cfg.visualizer_hz, 0.0, self._publish)
        self._visible: dict[int, bool] = {}
        if meshcat is not None:
            self._build()

    def _path(self, note: Note, part: str) -> str:
        return f"/highway/notes/{note.id}/{part}"

    def _build(self) -> None:
        L = self.layout
        m = self.meshcat
        m.Delete("/highway")
        # Judgment line.
        m.SetObject(
            "/highway/hit_line",
            Box(2 * L.half_width + 0.02, 0.006, 0.002),
            Rgba(1.0, 0.95, 0.8, 0.9),
        )
        m.SetTransform("/highway/hit_line", RigidTransform([0, L.hit_y, L.screen_z + 0.002]))
        for note in self.chart.notes:
            c = COLORS[note.kind]
            rgba = Rgba(c[0] / 255, c[1] / 255, c[2] / 255, 1.0)
            head = Box(HEAD_LANE_FRACTION * L.lane_width, HEAD_TIME * L.scroll_speed, 0.008)
            m.SetObject(self._path(note, "head"), head, rgba)
            if note.kind == HOLD:
                body = Box(
                    HOLD_BODY_LANE_FRACTION * L.lane_width, note.duration * L.scroll_speed, 0.004
                )
                m.SetObject(self._path(note, "body"), body, rgba)
            elif note.kind == SLIDE:
                pts = np.array(
                    [
                        [L.lane_x(lane), -off * L.scroll_speed, 0.004]
                        for off, lane in note.waypoints
                    ]
                ).T
                m.SetLine(self._path(note, "body"), pts, 6.0, rgba)
            self._set_visible(note, False)

    def _set_visible(self, note: Note, visible: bool, t: float | None = None) -> None:
        if self._visible.get(note.id) == visible:
            return
        self._visible[note.id] = visible
        # Passing the time puts the change into the Meshcat recording too, so
        # the scrubber replays the notes along with the arm.
        self.meshcat.SetProperty(f"/highway/notes/{note.id}", "visible", visible, t)

    def _publish(self, context) -> None:
        if self.meshcat is None:
            return
        t = context.get_time()
        L = self.layout
        for note in self.chart.notes:
            on_screen = note.time - L.approach_time <= t <= note.end_time + 0.05
            self._set_visible(note, on_screen, t)
            if not on_screen:
                continue
            # The head travels toward the robot (+y) and stops at the judgment
            # line once the note is active (hold/slide in progress).
            dt = max(note.time - t, 0.0)
            head_y = L.note_y(dt)
            self.meshcat.SetTransform(
                self._path(note, "head"),
                RigidTransform([L.lane_x(note.lane), head_y, L.screen_z + 0.004]),
                t,
            )
            if note.kind == HOLD:
                # Body spans from the head to the note's end, further out.
                # While active, the part already played is behind the line;
                # we keep drawing the whole bar anchored at its true times.
                y_start = L.note_y(note.time - t)
                y_end = L.note_y(note.end_time - t)
                self.meshcat.SetTransform(
                    self._path(note, "body"),
                    RigidTransform(
                        [L.lane_x(note.lane), 0.5 * (y_start + y_end), L.screen_z + 0.002]
                    ),
                    t,
                )
            elif note.kind == SLIDE:
                # The line was built with its start at the origin; translate
                # the whole path so its start sits at the (moving) hit time.
                self.meshcat.SetTransform(
                    self._path(note, "body"),
                    RigidTransform([0.0, L.note_y(note.time - t), L.screen_z]),
                    t,
                )
