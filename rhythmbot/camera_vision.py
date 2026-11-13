"""RGB-only calibrated vision for the concert display.

The detector accepts pixels and a clock, never a Chart, render labels, depth,
robot poses or a clean backing texture. Calibration is fixed screen/camera
geometry. Occluded pixels stay occluded through rectification.
"""
from __future__ import annotations
import hashlib
import numpy as np
from scipy import ndimage
from .camera import camera_config, camera_pose
from .game import FrameRenderer
from .perception import NoteDetector, DetectedNote
from .chart import TAP, HOLD, SLIDE


class DisplayRectifier:
    def __init__(self, cfg, camera=True, concert=True):
        self.camera = camera
        self.cfg = cfg
        R = FrameRenderer(cfg)
        W, H = cfg.perception.frame_size
        row, col = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        # Undo the display's artistic lane contraction before RGB detection.
        depth = np.clip((R.hit_row - row) / R.hit_row, 0, 1)
        if concert:
            col = W * .5 + (col - W * .5) * (1 - .76 * depth)
            row = R.hit_row - (R.hit_row - row) * .79
        if camera:
            L = cfg.layout
            x = (col + .5) / W * (2 * L.half_width) - L.half_width
            y = L.hit_y - (R.hit_row - (row + .5)) / R.px_per_sec * L.scroll_speed
            world = np.stack([x, y, np.full_like(x, L.screen_z)], -1)
            X = camera_pose(cfg)
            pc = (world - X.translation()) @ X.rotation().matrix()
            cam, _ = camera_config(cfg).MakeCameras()
            K = cam.core().intrinsics()
            self.u = K.focal_x() * pc[..., 0] / pc[..., 2] + K.center_x()
            self.v = K.focal_y() * pc[..., 1] / pc[..., 2] + K.center_y()
            self.valid = pc[..., 2] > 0
        else:
            self.u, self.v = col, row
            self.valid = np.ones((H, W), bool)

    def rectify(self, rgb):
        expected = ((self.cfg.camera.height, self.cfg.camera.width) if self.camera
                    else self.cfg.perception.frame_size[::-1])
        if rgb is None or rgb.shape != (*expected, 3):
            raise ValueError(f'Expected RGB {(*expected, 3)}, got {None if rgb is None else rgb.shape}')
        out = np.stack([ndimage.map_coordinates(rgb[..., k], [self.v, self.u],
                         order=1, mode='constant', cval=0) for k in range(3)], -1)
        out[~self.valid] = 0
        return out.astype(np.uint8)


class ConcertVisionDetector(NoteDetector):
    """Color/shape vision; green sustain shape distinguishes holds from slides."""
    needs_frames = True

    def __init__(self, cfg, camera=True):
        self.rectifier = DisplayRectifier(cfg, camera=camera)
        self.renderer = FrameRenderer(cfg)  # coordinate helpers only; never render()
        self.last_rgb = None
        self.last_rectified = None
        self.last_detections = []
        self.last_time = None
        self.input_hash = hashlib.sha256()
        self.frames = 0
        self.blank_frames = 0

    def detect(self, frame, t):
        rgb = self.rectifier.rectify(frame)
        self.last_time = float(t)
        self.input_hash.update(frame.tobytes())
        self.last_rgb = frame.copy()
        self.last_rectified = rgb
        self.frames += 1
        R = self.renderer
        r, g, b = np.moveaxis(rgb.astype(np.int16), -1, 0)
        masks = [(TAP, (g > 155) & (b > 170) & (r < 180) & (b > r + 35) & (b > g - 20)),
                 (HOLD, (g > 105) & (g > r + 35) & (g > b + 25))]
        out = []
        for kind, mask in masks:
            mask[R.hit_row + 12:] = False
            mask = ndimage.binary_closing(np.pad(mask, ((6, 6), (1, 1)), mode="edge"),
                                            structure=np.ones((13, 3)))[6:-6, 1:-1]
            labels, _ = ndimage.label(mask)
            for ident, sl in enumerate(ndimage.find_objects(labels), 1):
                if sl is None: continue
                comp = labels[sl] == ident
                if comp.sum() < 16 or comp.shape[1] < 3: continue
                bottom = sl[0].stop - 1
                depth = np.clip((R.hit_row - bottom) / R.hit_row, 0, 1)
                head_half = 8 * (1 - .65 * depth) / .79
                hit_row = bottom - head_half
                active = bottom >= R.hit_row - 6
                if active: hit_row = float(R.hit_row)
                hit_time = t if active else t + R.row_to_dt(hit_row)
                head = comp[-max(2, int(head_half)):]
                xs = np.nonzero(head.any(axis=0))[0]
                if not len(xs): continue
                lane = float(R.px_to_lane(sl[1].start + xs.mean()))
                if kind == TAP:
                    out.append(DetectedNote(TAP, round(lane), hit_time, active=active))
                    continue
                top = sl[0].start
                end_row = top + 1 / .79
                duration = max(0., R.row_to_dt(end_row) - R.row_to_dt(hit_row))
                wps = [(0., lane)]
                for row in range(comp.shape[0] - int(2 * head_half), -1, -3):
                    xx = np.nonzero(comp[row])[0]
                    if not len(xx): continue
                    off = R.row_to_dt(sl[0].start + row) - R.row_to_dt(hit_row)
                    if off > wps[-1][0] + .001:
                        wps.append((float(off), float(R.px_to_lane(sl[1].start + xx.mean()))))
                spread = np.ptp([x for _, x in wps])
                sustain_kind = SLIDE if spread > .4 else HOLD
                out.append(DetectedNote(sustain_kind, round(lane), hit_time, duration,
                    duration_final=top > 3, active=active,
                    waypoints=wps if sustain_kind == SLIDE else []))
        self.last_detections = out
        self.blank_frames += not bool(out)
        return out

    def stats(self):
        return dict(frames=self.frames, blank_detection_frames=self.blank_frames,
                    input_rgb_sha256=self.input_hash.hexdigest(),
                    input='Drake composed RGB' if self.rectifier.camera else 'direct concert RGB',
                    chart_access=False, label_input=False, depth_input=False,
                    method='fixed geometric calibration + RGB color/shape segmentation')
