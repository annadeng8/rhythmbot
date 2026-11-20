import unittest
from copy import deepcopy
from dataclasses import replace
import numpy as np
from scipy import ndimage
from rhythmbot.camera import CameraScreen
from rhythmbot.camera_vision import close_note_mask, DisplayRectifier
from rhythmbot.planner import Keyframe, Plan, TrackedNote
from rhythmbot.profiles import config, resolve
from rhythmbot.chart import Chart

class RealtimeTests(unittest.TestCase):
    def test_tracked_path_copy_is_independent(self):
        original = TrackedNote(1, 'slide', 0, 1., waypoints=[(0., 0.), (1., 5.)])
        original.alias = original.waypoints
        copied = deepcopy(original)
        self.assertIs(copied.alias, copied.waypoints)
        copied.waypoints.append((2., 1.))
        self.assertEqual(len(original.waypoints), 2)

    def test_keyframe_copy_preserves_independence_and_aliases(self):
        k = Keyframe(1., .2, .3, .04, .1, True)
        k.extension = [np.array([1., 2.])]
        copied = deepcopy([k, k])
        self.assertIs(copied[0], copied[1])
        self.assertIsNot(copied[0], k)
        copied[0].x = 10
        copied[0].extension[0][0] = 5
        self.assertEqual(k.x, .2)
        self.assertEqual(k.extension[0][0], 1.)

    def test_fixed_warp_preserves_scipy_pixels(self):
        rng = np.random.default_rng(11)
        base = config({'preset': 'sekai-camera'})
        for camera in (False, True):
            for width in (32, 480, 960):
                cfg = replace(base, camera=replace(base.camera, width=width, height=width * 3 // 4))
                rect = DisplayRectifier(cfg, camera=camera)
                shape = ((cfg.camera.height, cfg.camera.width) if camera else cfg.perception.frame_size[::-1])
                for index in range(8):
                    rgb = rng.integers(0, 256, (*shape, 3), dtype=np.uint8)
                    if index < 2:
                        rgb[:] = 255 * index
                    expected = np.stack([ndimage.map_coordinates(rgb[..., k], [rect.v, rect.u],
                        order=1, mode='constant', cval=0) for k in range(3)], -1)
                    expected[~rect.valid] = 0
                    np.testing.assert_array_equal(rect.rectify(rgb), expected)

    def test_separable_closing_is_exact(self):
        rng=np.random.default_rng(7)
        for shape in ((1,1),(9,4),(80,65),(384,256)):
            for density in (0.,.01,.3,.9,1.):
                mask=rng.random(shape)<density
                old=ndimage.binary_closing(np.pad(mask,((6,6),(1,1)),mode='edge'),structure=np.ones((13,3)))[6:-6,1:-1]
                np.testing.assert_array_equal(close_note_mask(mask),old)

    def test_prefetch_preserves_original_camera_pixels(self):
        cfg=config({'preset':'sekai-camera'})
        cfg=replace(cfg,camera=replace(cfg.camera,width=32,height=24,prefetch_noise=True))
        screen=CameraScreen(Chart([]),cfg,5)
        rng=np.random.default_rng(1);rgb=rng.integers(0,256,(24,32,4),dtype=np.uint8)
        labels=rng.integers(4,7,(24,32))
        try:
            for t in (0.,1/30,2/30,.151,.2,0.,1.):
                game=screen.renderer.render(screen.chart,t);expected=rgb[...,:3].copy()
                mask=screen.ok & (labels==5)
                expected[mask]=game[screen.rows[mask],screen.cols[mask]]
                noise=np.random.default_rng(int(round(t*1000))).normal(0.,3.,expected.shape)
                expected=np.clip(expected+noise,0,255).astype(np.uint8)
                np.testing.assert_array_equal(screen.compose(rgb,labels,t),expected)
                self.assertLessEqual(len(screen.noise_prefetch.pending),4)
        finally:screen.noise_prefetch.close()

    def test_width_fast_path_matches_geometry_interpolation(self):
        frames=[Keyframe(t,x,.1,w,0,False,ref,rise) for t,x,w,ref,rise in [(1,-.2,.03,-1,.01),(2,.1,.06,1,-.01),(2.5,.2,.04,0,0)]]
        p=Plan(frames)
        for t in np.linspace(0,4,201):self.assertEqual(p.width_at(t),float(p._state(t)[2]))

    def test_realtime_preserves_physical_camera_configuration(self):
        base=config({'preset':'sekai-camera'});fast=config({'preset':'sekai-realtime'})
        self.assertEqual((fast.camera.width, fast.camera.height), (640, 480))
        self.assertEqual(replace(fast,camera=replace(fast.camera,width=960,height=720,prefetch_noise=False)),base)
        self.assertEqual(resolve('sekai-realtime'),(2,'camera',('concert',),True))
