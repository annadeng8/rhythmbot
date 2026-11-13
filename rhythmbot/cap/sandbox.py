"""Loading a generated policy file and running it safely inside the brain.

The file may import only a short list of modules (checked on its syntax
tree, before it runs), so a generated program cannot call the classical
planner or read the chart: it sees the same tracked notes as every other
policy.  At run time `GuardedPolicy` catches exceptions and malformed plans,
falls back to lifting the hand where it is, and keeps counts and the first
traceback of each distinct error for the feedback report.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

from ..config import Config
from ..planner import Plan, Policy
from .api import RobotAPI

ALLOWED_IMPORTS = {
    "numpy", "scipy", "math", "dataclasses", "typing", "functools", "itertools", "collections",
    "heapq", "bisect", "enum", "copy", "__future__", "rhythmbot.cap.api",
}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "breakpoint", "input"}


class PolicyRejected(Exception):
    pass


def check_source(src: str) -> None:
    """Raise PolicyRejected if the program imports anything outside the
    allowed list or calls builtins that reach outside the simulation."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [("." * node.level) + (node.module or "")]
        else:
            if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
                raise PolicyRejected(f"line {node.lineno}: `{node.id}` is not allowed")
            if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr != "__init__":
                raise PolicyRejected(f"line {node.lineno}: dunder attribute `{node.attr}` is not allowed")
            continue
        for m in mods:
            root = m.split(".")[0]
            if m not in ALLOWED_IMPORTS and root not in ALLOWED_IMPORTS - {"rhythmbot"}:
                raise PolicyRejected(
                    f"line {node.lineno}: import of `{m}` is not allowed "
                    f"(allowed: {', '.join(sorted(ALLOWED_IMPORTS - {'__future__'}))})"
                )


class GuardedPolicy(Policy):
    """Runs a generated policy; on any error, lifts the hand in place.

    A pixels-only program (`pixels=True`) is called with the rendered game
    frame in place of the tracked notes: it does its own perception."""

    def __init__(self, inner, api: RobotAPI, path: str, pixels: bool = False):
        self.inner = inner
        self.api = api
        self.path = path
        self.pixels = pixels
        self.wants_frames = pixels
        self.frame = None
        self.n_calls = 0
        self.n_errors = 0
        self.errors: Counter[str] = Counter()
        self.first_traceback: dict[str, str] = {}
        self.plan_ms: list[float] = []

    def _fallback(self, t_now, X_WT_cmd, width_cmd) -> Plan:
        k0 = self.api.current_keyframe(t_now, X_WT_cmd, width_cmd)
        z = max(k0.z, self.api.screen_z + self.api.limits["hover_height"])
        return self.api.make_plan([k0, self.api.keyframe(t_now + 0.15, k0.x, z, width_cmd)])

    def _record(self, key: str, tb: str) -> None:
        self.n_errors += 1
        self.errors[key] += 1
        self.first_traceback.setdefault(key, tb)

    def plan(self, notes, t_now, X_WT_cmd, width_cmd) -> Plan:
        if self.pixels and self.frame is None:
            # The brain's very first plan is made before any frame exists.
            return self._fallback(t_now, X_WT_cmd, width_cmd)
        self.n_calls += 1
        t0 = time.perf_counter()
        try:
            seen = self.frame.copy() if self.pixels else notes
            plan = self.inner.plan(seen, t_now, X_WT_cmd, width_cmd)
            if not isinstance(plan, Plan):
                raise TypeError(f"plan() returned {type(plan).__name__}, expected api.Plan")
            if self.api.cfg.gripper.single_tip:
                plan = self.api.make_plan(plan.keyframes)
            for dt in (0.0, 1.0 / 30, 0.1):
                p = plan.tip_pose(t_now + dt).translation()
                w = plan.width_at(t_now + dt)
                if not (np.all(np.isfinite(p)) and math.isfinite(w)):
                    raise ValueError(f"plan has a non-finite pose or width at t_now+{dt:.3f}")
        except Exception as e:  # noqa: BLE001 - anything the program does wrong
            tb = traceback.format_exc(limit=-4)
            self._record(f"{type(e).__name__}: {e}"[:200], tb)
            plan = self._fallback(t_now, X_WT_cmd, width_cmd)
        self.plan_ms.append(1000 * (time.perf_counter() - t0))
        return plan

    def stats(self) -> dict:
        ms = np.array(self.plan_ms) if self.plan_ms else np.zeros(1)
        return {
            "plan_calls": self.n_calls,
            "plan_errors": self.n_errors,
            "errors": [
                {"error": k, "count": c, "traceback": self.first_traceback[k]} for k, c in self.errors.most_common(5)
            ],
            "plan_ms_mean": float(ms.mean()),
            "plan_ms_max": float(ms.max()),
        }


def load_module(path: str | Path):
    """Check and import a generated file."""
    path = Path(path)
    check_source(path.read_text())
    spec = importlib.util.spec_from_file_location(f"generated_{abs(hash(str(path.resolve())))}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_policy(path: str | Path, cfg: Config, pixels: bool = False, hand: int | None = None) -> GuardedPolicy:
    module = load_module(path)
    if not hasattr(module, "make_policy"):
        raise PolicyRejected("the file must define make_policy(api)")
    api = RobotAPI(cfg, hand)
    inner = module.make_policy(api)
    if not callable(getattr(inner, "plan", None)):
        raise PolicyRejected("make_policy(api) must return an object with a plan(...) method")
    return GuardedPolicy(inner, api, str(path), pixels)
