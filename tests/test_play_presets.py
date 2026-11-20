"""The normal play entry point must not silently select a legacy robot."""
import contextlib
import io
import unittest
from unittest.mock import patch
from rhythmbot.play import main


class PlayPresetTests(unittest.TestCase):
    def test_live_routes_to_dual_arm_camera_runner(self):
        with patch('rhythmbot.realtime.run', return_value={}) as run:
            main(['--preset', 'sekai-realtime', '--demo', '--out', 'unused.json',
                  '--policy', 'cap:unused.py', '--duration', '8', '--no-realtime'])
        self.assertEqual(run.call_args.kwargs['policy'], 'cap:unused.py')
        self.assertEqual(run.call_args.kwargs['duration'], 8)
        self.assertFalse(run.call_args.kwargs['paced'])

    def test_live_rejects_single_arm_or_oracle_overrides(self):
        for override in (['--arms', '1'], ['--detector', 'oracle'], ['--view', 'screen']):
            with self.subTest(override=override), contextlib.redirect_stderr(io.StringIO()), \
                    patch('rhythmbot.realtime.run') as run, self.assertRaises(SystemExit):
                main(['--preset', 'sekai-realtime', *override])
            run.assert_not_called()

    def test_study_preset_remains_full_resolution_and_bimanual(self):
        with patch('rhythmbot.play.run') as run, patch('sys.stdin.isatty', return_value=False):
            main(['--preset', 'sekai-camera', '--demo'])
        cfg = run.call_args.args[1]
        self.assertEqual(cfg.arms.n, 2)
        self.assertEqual(cfg.gripper.contacts_per_arm, 1)
        self.assertEqual((cfg.camera.width, cfg.camera.height), (960, 720))
        self.assertEqual(run.call_args.kwargs['view'], 'camera')
        self.assertEqual(run.call_args.kwargs['detector'], 'concert')
