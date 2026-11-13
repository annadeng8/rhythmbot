"""Named evaluation environments; no controller or scoring overrides."""
from dataclasses import replace
from .config import DEFAULT, Camera, stylus_config, two_arm_config

PRESETS = ('sekai-camera', 'sekai-screen')

def resolve(preset=None, arms=None, view=None, detectors=None, stylus=False):
    if preset:
        expected_view = 'camera' if preset == 'sekai-camera' else 'screen'
        if preset not in PRESETS:
            raise ValueError(f'Unknown preset: {preset}')
        if arms not in (None, 2) or view not in (None, expected_view) or detectors not in (None, ('concert',)):
            raise ValueError(f'{preset} requires two single-tip arms, {expected_view} input and concert RGB detection')
        return 2, expected_view, ('concert',), True
    arms, view, detectors = arms or (2 if stylus else 1), view or 'screen', detectors or ('oracle',)
    if stylus and arms != 2:
        raise ValueError('Single-stylus mode requires two arms')
    if view == 'camera' and 'oracle' in detectors:
        raise ValueError('Oracle is not camera vision; use --preset sekai-camera')
    return arms, view, detectors, stylus

def config(job):
    cfg = stylus_config() if job.get('stylus') or job.get('preset') in PRESETS else (two_arm_config() if job.get('arms', 1) == 2 else DEFAULT)
    if job.get('preset') in PRESETS:
        cfg = replace(cfg, camera=Camera(width=960, height=720, fov_y_deg=48,
            eye_offset=(.35, -.50, 1.15), target_offset=(0., .05, 0.), noise_std=3.))
    return cfg
