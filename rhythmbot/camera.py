"""An external camera on the touch screen (`--view camera`).

By default the perception stage reads the game's own frame (`GameScreen`),
as if the robot were plugged into the screen's video output.  With the
camera view it instead gets what a fixed RGB camera beside the screen sees:
the screen in perspective, the arm and gripper in front of it, and the notes
hidden wherever the hand covers them.

The notes are not Drake geometry (they are drawn in Meshcat only), so the
camera image is composited: Drake renders the scene (arm, gripper, floor,
screen slab) with an `RgbdSensor`, and every pixel whose label is the screen
body is replaced by the game frame at the point where that pixel's ray meets
the screen plane.  The ray-to-frame lookup is computed once, since the
camera does not move; occlusion comes for free from the rendered labels.

    Station ports:  screen_cam.rgb_image, screen_cam.label_image
    CameraScreen:   (rgb_image, label_image) -> frame (H, W, 3) uint8
"""

from __future__ import annotations

import numpy as np
from pydrake.all import AbstractValue, ImageLabel16I, ImageRgba8U, LeafSystem, RigidTransform, RotationMatrix
from pydrake.systems.sensors import CameraConfig

from .chart import Chart
from .config import Config
from .game import FrameRenderer

CAMERA_NAME = "screen_cam"


def look_at(eye: np.ndarray, target: np.ndarray) -> RigidTransform:
    """Camera pose (Drake convention: +z forward, +x right, +y down)."""
    z = target - eye
    z = z / np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return RigidTransform(RotationMatrix(np.column_stack([x, y, z])), eye)


def camera_pose(cfg: Config) -> RigidTransform:
    C = cfg.camera
    c = cfg.layout.screen_center()
    return look_at(c + np.array(C.eye_offset), c + np.array(C.target_offset))


def camera_config(cfg: Config) -> CameraConfig:
    from pydrake.common.schema import Transform

    C = cfg.camera
    cam = CameraConfig()
    cam.name = CAMERA_NAME
    cam.width, cam.height = C.width, C.height
    cam.focal = CameraConfig.FovDegrees(y=C.fov_y_deg)
    cam.fps = cfg.perception.tick_hz
    cam.rgb = True
    cam.depth = False
    cam.label = True
    cam.background = cam.background  # floor fills most of the view anyway
    cam.X_PB = Transform(camera_pose(cfg))
    return cam


def screen_lookup(cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every camera pixel, the game-frame (row, col) its ray hits on the
    screen plane, and whether that lands inside the game frame."""
    C, L = cfg.camera, cfg.layout
    color_cam, _ = camera_config(cfg).MakeCameras()
    K = color_cam.core().intrinsics()
    X_WC = camera_pose(cfg)
    u, v = np.meshgrid(np.arange(C.width), np.arange(C.height))
    d_C = np.stack([(u - K.center_x()) / K.focal_x(), (v - K.center_y()) / K.focal_y(), np.ones_like(u, float)], -1)
    d_W = d_C @ X_WC.rotation().matrix().T
    o = X_WC.translation()
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (L.screen_z - o[2]) / d_W[..., 2]
    p = o + s[..., None] * d_W
    W, H = cfg.perception.frame_size
    renderer = FrameRenderer(cfg)
    col = (p[..., 0] + L.half_width) / (L.n_lanes * L.lane_width) * W
    time_to_hit = (L.hit_y - p[..., 1]) / L.scroll_speed
    row = renderer.hit_row - time_to_hit * renderer.px_per_sec
    ok = (s > 0) & (col >= 0) & (col < W) & (row >= 0) & (row < H)
    rows = np.clip(np.floor(row), 0, H - 1).astype(np.intp)
    cols = np.clip(np.floor(col), 0, W - 1).astype(np.intp)
    return rows, cols, ok


class CameraScreen(LeafSystem):
    """The camera image with the game frame drawn onto the visible screen."""

    def __init__(self, chart: Chart, cfg: Config, screen_label: int):
        super().__init__()
        self.cfg = cfg
        self.chart = chart
        self.renderer = FrameRenderer(cfg)
        self.screen_label = screen_label
        self.rows, self.cols, self.ok = screen_lookup(cfg)
        self.noise = cfg.camera.noise_std
        self.color_port = self.DeclareAbstractInputPort("rgb_image", AbstractValue.Make(ImageRgba8U()))
        self.label_port = self.DeclareAbstractInputPort("label_image", AbstractValue.Make(ImageLabel16I()))
        self.DeclareAbstractOutputPort(
            "frame", lambda: AbstractValue.Make(np.zeros((1, 1, 3), dtype=np.uint8)), self._calc
        )

    def compose(self, rgb: np.ndarray, label: np.ndarray, t: float) -> np.ndarray:
        game = self.renderer.render(self.chart, t)
        out = rgb[..., :3].copy()
        m = self.ok & (label == self.screen_label)
        out[m] = game[self.rows[m], self.cols[m]]
        if self.noise > 0:
            rng = np.random.default_rng(int(round(t * 1000)))
            out = np.clip(out + rng.normal(0.0, self.noise, out.shape), 0, 255).astype(np.uint8)
        return out

    def _calc(self, context, output):
        rgb = self.color_port.Eval(context).data
        label = self.label_port.Eval(context).data[..., 0]
        output.set_value(self.compose(rgb, label, context.get_time()))
