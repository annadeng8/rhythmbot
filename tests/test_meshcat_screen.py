import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
import numpy as np
from rhythmbot.config import stylus_config
from rhythmbot.chart import Chart, Note, TAP, HOLD, SLIDE
from rhythmbot.sekai_screen import SekaiScreen
from rhythmbot.meshcat_screen import screen_quad, write_screen_gltf, MeshcatScreen
from rhythmbot.profiles import config
from unittest.mock import Mock
from rhythmbot.game import FrameRenderer
from rhythmbot.camera_vision import ConcertVisionDetector


class MinimalScreenTests(unittest.TestCase):
    def test_prefetched_texture_matches_current_chart_frame_and_is_bounded(self):
        cfg = config({'preset': 'sekai-realtime'})
        chart = Chart([Note(TAP, 1, 3.)])
        screen = MeshcatScreen(chart, cfg, Mock())
        try:
            for tick in (0, 1, 70, 0):
                screen._fill(tick)
                path = screen._pending[tick].result()
                data = json.loads(path.read_text())
                png = base64.b64decode(data['images'][0]['uri'].split(',', 1)[1])
                expected = SekaiScreen(cfg).render(chart, tick / cfg.visualizer_hz, scale=4)
                np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(png))), expected)
                self.assertEqual(len(screen._pending), 4)
        finally:
            screen.close()

    def test_empty_screen_has_no_animation_or_feedback(self):
        renderer = SekaiScreen(stylus_config())
        a = renderer.render(Chart([]), 0.)
        renderer.results = {123: object()}
        b = renderer.render(Chart([]), 10.)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(np.unique(a.reshape(-1, 3), axis=0)), 3)

    def test_tap_hold_slide_are_detectable_from_minimal_pixels(self):
        cfg = stylus_config()
        for kind in (TAP, HOLD, SLIDE):
            note = Note(kind, 1, 3., duration=0 if kind == TAP else .5,
                        waypoints=[(0, 1), (.5, 4)] if kind == SLIDE else [])
            rgb = SekaiScreen(cfg).render(Chart([note]), 2.3)
            found = ConcertVisionDetector(cfg, camera=False).detect(rgb, 2.3)
            self.assertTrue(any(n.kind == kind for n in found), kind)

    def test_high_resolution_texture_preserves_note_locations(self):
        r = SekaiScreen(stylus_config())
        chart = Chart([Note(TAP, 1, 3.)])
        a, b = r.render(chart, 2.3), r.render(chart, 2.3, scale=4)
        self.assertEqual(b.shape, (a.shape[0] * 4, a.shape[1] * 4, 3))
        x, y = r.project(1, .7)
        np.testing.assert_array_equal(a[int(y), int(x)], b[int(4*y), int(4*x)])
        np.testing.assert_array_equal(a[r.hit_row, 0], b[4*r.hit_row, 0])

    def test_texture_is_embedded_and_judgment_line_aligns_with_touch(self):
        cfg = stylus_config()
        rgb = SekaiScreen(cfg).render(Chart([]), 0.)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'screen.gltf'
            write_screen_gltf(path, rgb, cfg)
            data = json.loads(path.read_text())
        encoded = data['images'][0]['uri'].split(',', 1)[1]
        np.testing.assert_array_equal(np.array(Image.open(io.BytesIO(base64.b64decode(encoded)))), rgb)
        p, uv = screen_quad(cfg)
        binary = base64.b64decode(data['buffers'][0]['uri'].split(',', 1)[1])
        y_up = np.frombuffer(binary[:48], dtype='<f4').reshape(4, 3)
        recovered = np.column_stack((y_up[:, 0], -y_up[:, 2], y_up[:, 1]))
        np.testing.assert_allclose(recovered, p)
        r = FrameRenderer(cfg)
        fraction = (r.hit_row / r.H) / uv[2, 1]
        world_y = p[0, 1] + fraction * (p[2, 1] - p[0, 1])
        self.assertAlmostEqual(world_y, cfg.layout.hit_y, places=6)
        self.assertLessEqual(p[2, 1], cfg.layout.screen_center()[1] + cfg.layout.screen_size()[1] / 2 + 1e-6)

if __name__ == '__main__':
    unittest.main()
