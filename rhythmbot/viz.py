"""Meshcat helpers, so the package runs on its own outside the course repo.

One Meshcat server per process, started on first use.  `HEADLESS=1` in the
environment turns visualization off entirely: no server, no waiting for a
browser, no real-time pacing.
"""

from __future__ import annotations

import os
import time

from pydrake.all import Meshcat, StartMeshcat

HEADLESS = bool(os.environ.get("HEADLESS"))

# How long the first `get_meshcat()` waits for a browser tab to connect, so
# the start of a run is not missed.
BROWSER_WAIT_S = 15.0

_meshcat: Meshcat | None = None


def get_meshcat() -> Meshcat | None:
    """The process's Meshcat, started on first call; None when HEADLESS."""
    global _meshcat
    if HEADLESS:
        return None
    if _meshcat is None:
        _meshcat = StartMeshcat()
        print(f"Check your Meshcat window ({_meshcat.web_url()})")
        deadline = time.time() + BROWSER_WAIT_S
        while time.time() < deadline and _meshcat.GetNumActiveConnections() == 0:
            time.sleep(0.1)
        if _meshcat.GetNumActiveConnections() == 0:
            print("   ...no browser connected; running anyway.")
    return _meshcat


def start_recording(**kwargs) -> None:
    """Start recording an animation if a Meshcat is running."""
    if _meshcat is not None:
        _meshcat.StartRecording(**kwargs)


def publish_recording() -> None:
    """Stop recording and publish it, giving the page a playback scrubber."""
    if _meshcat is not None:
        _meshcat.StopRecording()
        _meshcat.PublishRecording()


def keep_meshcat_open() -> None:
    """Block until Enter so the page stays alive after the run ends."""
    if _meshcat is not None:
        input("Press Enter to quit.")
