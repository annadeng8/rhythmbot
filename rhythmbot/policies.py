"""Every policy the brain can run, by name.

    classical            ClassicalPlanner (approach 1)
    cap:<policy.py>      a program written by a coding agent (approach 2, `rhythmbot.cap`)
    cappx:<policy.py>    the same, but the program gets only the game frame and does its own perception
    bc:<model.pt>        a behavior-cloned network (approach 3, `rhythmbot.bc`)

All three implement `planner.Policy` and see the same tracked notes, so they
can be compared on the same detector and the same judge.
"""

from __future__ import annotations

from .config import Config
from .planner import ClassicalPlanner, Policy


def make_policy(spec: str, cfg: Config) -> Policy:
    kind, _, arg = spec.partition(":")
    if kind == "classical":
        return ClassicalPlanner(cfg)
    if kind in ("cap", "cappx"):
        from .cap.sandbox import load_policy

        return load_policy(arg, cfg, pixels=kind == "cappx")
    if kind == "bc":
        from .bc.policy import BCPolicy

        return BCPolicy.load(arg, cfg)
    raise ValueError(f"unknown policy {spec!r}: use classical, cap:<policy.py>, cappx:<policy.py> or bc:<model.pt>")
