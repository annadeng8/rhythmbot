import hashlib
import unittest
from unittest.mock import patch
import numpy as np
from rhythmbot.camera_vision import NeuralCameraDetector, DisplayRectifier
from rhythmbot.perception import TorchDetector, DetectedNote
from rhythmbot.profiles import config, resolve


class NeuralCameraTests(unittest.TestCase):
    def test_network_receives_only_rectified_pixels_and_clock(self):
        cfg = config({'preset': 'sekai-realtime'})
        detector = object.__new__(NeuralCameraDetector)
        detector.rectifier = DisplayRectifier(cfg)
        detector.input_hash = hashlib.sha256()
        detector.rectified_hash = hashlib.sha256()
        detector.frames = detector.blank_frames = 0
        detector.inference_seconds = []
        frame = np.zeros((480, 640, 3), np.uint8)
        frame[120:300, 130:410] = (100, 220, 80)
        expected = detector.rectifier.rectify(frame)
        found = [DetectedNote('tap', 1, 2.)]
        with patch.object(TorchDetector, 'detect', return_value=found) as network, \
             patch('rhythmbot.game.FrameRenderer.render', side_effect=AssertionError('No chart renderer')):
            self.assertEqual(detector.detect(frame, 1.2), found)
        np.testing.assert_array_equal(network.call_args.args[0], expected)
        self.assertEqual(network.call_args.args[1], 1.2)
        self.assertFalse(hasattr(detector, 'chart'))
        self.assertEqual(detector.input_hash.hexdigest(), hashlib.sha256(frame.tobytes()).hexdigest())
        # An occlusion cannot be filled from underlying game state.
        black = detector.rectifier.rectify(np.zeros_like(frame))
        self.assertFalse(black.any())

    def test_neural_preset_keeps_two_arms_and_rejects_oracle(self):
        self.assertEqual(resolve('sekai-camera', detectors=('concert-torch',)),
                         (2, 'camera', ('concert-torch',), True))
        with self.assertRaises(ValueError):
            resolve('sekai-camera', detectors=('oracle',))
