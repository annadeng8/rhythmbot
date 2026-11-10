import unittest
from unittest.mock import patch
import numpy as np
from rhythmbot.config import stylus_config
from rhythmbot.camera_vision import ConcertVisionDetector, DisplayRectifier
from rhythmbot.chart import Chart, Note, TAP, SLIDE
from rhythmbot.sekai_screen import SekaiScreen
from rhythmbot.play import build

class CameraVisionTests(unittest.TestCase):
    def test_camera_rejects_oracle_before_building_robot(self):
        with self.assertRaisesRegex(ValueError, 'image detector'):
            build(Chart([]),stylus_config(),'oracle',None,None,view='camera')

    def test_blank_rgb_cannot_reveal_notes(self):
        cfg=stylus_config();d=ConcertVisionDetector(cfg)
        frame=np.zeros((cfg.camera.height,cfg.camera.width,3),np.uint8)
        with patch('rhythmbot.game.FrameRenderer.render',side_effect=AssertionError('No backing texture access')):
            self.assertEqual(d.detect(frame,3.),[])
        self.assertFalse(hasattr(d,'chart'))
        self.assertTrue(np.all(d.last_rectified==0))

    def test_detections_change_with_pixels_at_fixed_time(self):
        cfg=stylus_config();d=ConcertVisionDetector(cfg,camera=False)
        chart=Chart([Note(TAP,1,3.)]);rgb=SekaiScreen(cfg).render(chart,2.3)
        notes=d.detect(rgb,2.3)
        self.assertTrue(any(n.kind==TAP and n.lane==1 and abs(n.hit_time-3)<.1 for n in notes))
        self.assertTrue(all(n.note_id==-1 for n in notes))
        self.assertEqual(d.detect(np.zeros_like(rgb),2.3),[])

    def test_partial_blackout_is_not_filled_from_texture(self):
        cfg=stylus_config();rect=DisplayRectifier(cfg,camera=False)
        W,H=cfg.perception.frame_size
        rgb=np.full((H,W,3),220,np.uint8);rgb[:,:W//2]=0
        out=rect.rectify(rgb)
        self.assertTrue(np.all(out[:,:W//2-8]==0))
        self.assertGreater(out[:,W//2+8:].mean(),200)

    def test_wrong_rgb_dimensions_fail_loudly(self):
        with self.assertRaises(ValueError):
            ConcertVisionDetector(stylus_config()).detect(np.zeros((10,10,3),np.uint8),0.)

if __name__=='__main__':unittest.main()
