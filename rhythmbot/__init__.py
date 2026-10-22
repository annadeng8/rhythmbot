"""RhythmBot: a simulated KUKA iiwa playing a Project SEKAI-style rhythm game.

Pipeline (one Drake diagram):

    chart --> game (highway + 2D screen frames) --> perception (NoteDetector)
          --> brain (tracking + policy/planner, replans every tick)
          --> DiffIK --> HardwareStation (iiwa + wsg) --> TouchSensor --> Judge

The perception stage is an interface: `OracleDetector` reads the chart (an
upper bound on any vision model), `ColorSegmentationDetector` segments the
rendered game frames the way GuitarHeroBot did with HSV masks, and
`TorchDetector` is the slot for the trained vision model.

Meshcat helpers live in `viz.py`, so the package has no dependency on the
course repository it was first developed in.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
