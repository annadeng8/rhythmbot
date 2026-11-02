"""Agentic code-as-policy: a coding agent writes the control program, the
simulator scores it, the agent reads the feedback and revises.

    python -m rhythmbot.cap.loop run --name fable-1 --iterations 6
    python -m rhythmbot.cap.loop run --name fable-selftest --self-test     # agent may also run the sim itself
    python -m rhythmbot.cap.loop run --name fable-pixels --self-test --pixels  # program gets only the frame
    python -m rhythmbot.cap.loop heldout /Users/Shared/rhythmbot_cap/fable-1                  # every revision on the held-out charts
    python -m rhythmbot.cap.loop selftest /Users/Shared/rhythmbot_cap/fable-1/policy.py   # what --self-test lets the agent run

`run` makes a workspace `<RHYTHMBOT_CAP_RUNS>/<name>/` (default
`/Users/Shared/rhythmbot_cap`, outside the repository and home directory) holding only `TASK.md` (the
problem statement), `api.py` (the control API's source) and, once the agent
has written it, `policy.py`.  Each iteration is one non-interactive Claude
Code call in that directory, continuing the same conversation: the first
asks for a program, later ones point at `feedback.md`.  After each call the
program is copied to `history/policy_v<i>.py`, run headless on the
`cap-train` charts with the oracle detector, and the feedback for the next
iteration is written.  `history.json` keeps the per-revision score, the
agent's cost and turns.

With `--pixels` the program gets only the rendered game frame, no tracked
notes, and must do its own perception.  The workspace then also has example
frames from the training charts and `./probe.sh`, which runs the agent's
`probe.py` on every training frame with the notes visible in it (ground
truth for calibrating its perception code).

The agent works only in the workspace: it is allowed Read/Write/Edit/Glob/Grep
there (and, with --self-test, `./evaluate.sh`), nothing else; the
classical planner, the charts and the held-out set are not in it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .. import PROJECT_ROOT
from .feedback import load_reports, revision_feedback

HERE = Path(__file__).resolve().parent
# Workspaces live outside the repository and, where /Users/Shared exists,
# outside the home directory, so the agent can be shut out of both.
_SHARED = Path("/Users/Shared")
RUNS = Path(
    os.environ.get(
        "RHYTHMBOT_CAP_RUNS",
        _SHARED / "rhythmbot_cap" if _SHARED.is_dir() else PROJECT_ROOT.parent / "rhythmbot_cap_runs",
    )
)
TRAIN_SPLIT = "cap-train"

FIRST_PROMPT = (
    "Read TASK.md and api.py in this directory, then write policy.py: the control program for the "
    "rhythm-game robot described there. Aim for a program that completes as many notes as possible "
    "on charts you have not seen. If the program is long, write it in several smaller edits rather "
    "than one huge write. When policy.py is written and you are done, reply with a short "
    "summary of its strategy."
)
NEXT_PROMPT = (
    "Your revision {prev} was run on the training charts: {score}. feedback.md has the details "
    "(per-chart scores, every note not completed and why, and any errors your program raised). "
    "Revise policy.py to complete more notes; remember it will be judged on unseen charts. "
    "Reply with a short summary of what you changed and why."
)
PROBE_NOTE = (
    " `./probe.sh` runs your probe.py on every training frame with its ground-truth notes, for "
    "calibrating and testing your perception code."
)
SELFTEST_NOTE = (
    " You can test a revision yourself at any time with `./evaluate.sh` (runs policy.py on the "
    "training charts, about a minute, and prints the same kind of feedback)."
)


def _claude(prompt: str, cwd: Path, model: str, session: str | None, self_test: bool, log: Path) -> dict:
    tools = ["Read", "Write", "Edit", "Glob", "Grep"]
    denied = ["WebFetch", "WebSearch", "Task", "Agent", "NotebookEdit"]
    # Reads are allowed anywhere by default: shut out the project (the
    # classical planner, the charts), the home directory (Claude's own
    # transcripts of other runs) unless the workspace is in it, and every
    # other workspace (other runs, held-out reports).  Create the
    # workspaces of runs started together before starting any of them.
    shut = [PROJECT_ROOT] + [p for p in cwd.parent.iterdir() if p.is_dir() and p != cwd]
    if not cwd.resolve().is_relative_to(Path.home().resolve()):
        shut.append(Path.home())
    for d in shut:
        for tool in ("Read", "Glob", "Grep"):
            denied.append(f"{tool}(/{d.resolve()}/**)")
    if self_test:
        tools.append("Bash(./evaluate.sh)")
        tools.append("Bash(./probe.sh)")
    else:
        denied.append("Bash")
    cmd = [
        "claude", "-p", prompt,
        "--model", model,
        "--output-format", "json",
        "--permission-mode", "acceptEdits",
        "--allowedTools", *tools,
        "--disallowedTools", *denied,
    ]
    if session:
        cmd += ["--resume", session]
    t0 = time.time()
    # The default output cap cuts off a long policy.py written in one go.
    env = {**os.environ, "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000"}
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=3 * 3600)
    log.write_text(proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else ""))
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        out = {"is_error": True, "result": proc.stdout[-2000:] + proc.stderr[-2000:]}
    out["wall_s"] = time.time() - t0
    return out


def _evaluate(policy: Path, out: Path, workers: int, split: str = TRAIN_SPLIT, pixels: bool = False) -> list[dict]:
    from ..evaluate import evaluate

    if out.exists():
        shutil.rmtree(out)
    evaluate([f"{'cappx' if pixels else 'cap'}:{policy}"], split, out, ("oracle",), workers=workers)
    return load_reports(next(out.glob("*/oracle")))


def _score(reports: list[dict]) -> tuple[int, int]:
    n = sum(r["n_notes"] for r in reports)
    done = sum(sum(bool(x["completed"]) for x in r.get("notes", [])) for r in reports)
    return done, n


def _examples(step: float) -> list[dict]:
    """Every `step` seconds of every training chart: the frame and the notes
    visible in it."""
    from ..config import DEFAULT
    from ..game import FrameRenderer
    from ..splits import charts

    renderer = FrameRenderer(DEFAULT)
    out = []
    for chart in charts(TRAIN_SPLIT):
        for t in np.arange(0.0, chart.end_time + 0.5, step):
            notes = [
                {"kind": n.kind, "lane": n.lane, "time": n.time, "time_to_hit": round(float(n.time - t), 4),
                 "duration": n.duration, "waypoints": [list(w) for w in n.waypoints]}
                for n in chart.visible(t, DEFAULT.layout.approach_time)
            ]
            out.append({"chart": chart.name, "t": round(float(t), 4), "frame": renderer.render(chart, float(t)), "notes": notes})
    return out


def _write_examples(ws: Path) -> None:
    """A few dozen PNG frames with many notes on screen, and their index."""
    from PIL import Image

    d = ws / "examples"
    d.mkdir(exist_ok=True)
    ex = _examples(0.5)
    rng = np.random.default_rng(0)
    busy = [e for e in ex if len(e["notes"]) >= 3]
    pick = sorted(rng.choice(len(busy), size=min(30, len(busy)), replace=False))
    lines = ["# Example frames (training charts)", ""]
    for i in pick:
        e = busy[i]
        name = f"{e['chart']}_t{e['t']:.1f}.png"
        Image.fromarray(e["frame"]).save(d / name)
        lines += [f"## {name}", "", f"chart {e['chart']}, t = {e['t']:.2f} s; visible notes:", ""]
        for n in e["notes"]:
            s = f"- {n['kind']} lane {n['lane']:g}, hit at {n['time']:.3f} s (in {n['time_to_hit']:+.3f} s)"
            if n["kind"] != "tap":
                s += f", duration {n['duration']:.3f} s"
            if n["kind"] == "slide":
                s += f", waypoints {n['waypoints']}"
            lines.append(s)
        lines.append("")
    (d / "index.md").write_text("\n".join(lines))


def setup_workspace(ws: Path, self_test: bool, pixels: bool = False) -> None:
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "history").mkdir(exist_ok=True)
    (ws / "agent_log").mkdir(exist_ok=True)
    task = (HERE / "TASK.md").read_text()
    if pixels:
        # Swap the tracked-notes interface for the frame-only one.
        start, end = task.index("## What your program sees"), task.index("## The hand")
        task = task[:start] + (HERE / "TASK_PIXELS.md").read_text() + "\n" + task[end:]
        task = task.replace("def plan(self, notes, t_now", "def plan(self, frame, t_now")
        _write_examples(ws)
    if self_test:
        task += "\n## Testing it yourself\n\n`./evaluate.sh` runs `policy.py` on the training charts and prints that feedback.\n"
    (ws / "TASK.md").write_text(task)
    shutil.copy(HERE / "api.py", ws / "api.py")
    py = f'cd "{PROJECT_ROOT}" && HEADLESS=1 "{sys.executable}" -m rhythmbot.cap.loop'
    if self_test:
        sh = ws / "evaluate.sh"
        sh.write_text(f'#!/bin/sh\n{py} selftest "{ws / "policy.py"}"{" --pixels" if pixels else ""}\n')
        sh.chmod(0o755)
    if pixels:
        sh = ws / "probe.sh"
        sh.write_text(f'#!/bin/sh\n{py} probe "{ws / "probe.py"}"\n')
        sh.chmod(0o755)


def run(name: str, iterations: int, model: str, self_test: bool, workers: int, pixels: bool = False) -> Path:
    ws = RUNS / name
    hist_path = ws / "history.json"
    history = (
        json.loads(hist_path.read_text())
        if hist_path.exists()
        else {"model": model, "self_test": self_test, "pixels": pixels, "revisions": []}
    )
    pixels = history.get("pixels", False)
    if not history["revisions"]:
        setup_workspace(ws, self_test, pixels)
    session = history.get("session")
    for i in range(len(history["revisions"]), iterations):
        if i == 0:
            prompt = FIRST_PROMPT
        else:
            prev = history["revisions"][-1]
            prompt = NEXT_PROMPT.format(prev=i - 1, score=prev["train_summary"])
        if self_test:
            prompt += SELFTEST_NOTE
        if self_test and pixels:
            prompt += PROBE_NOTE
        print(f"[cap] {name}: iteration {i}, calling {model} …", flush=True)
        res = _claude(prompt, ws, model, session, self_test, ws / "agent_log" / f"v{i}.json")
        session = res.get("session_id", session)
        history["session"] = session
        rev = {
            "revision": i,
            "agent_error": bool(res.get("is_error")),
            "agent_reply": (res.get("result") or "")[:4000],
            "cost_usd": res.get("total_cost_usd"),
            "num_turns": res.get("num_turns"),
            "agent_wall_s": res.get("wall_s"),
        }
        policy = ws / "policy.py"
        if not policy.exists():
            rev["train_summary"] = "no policy.py was written"
            (ws / "feedback.md").write_text("# Feedback\n\nThere is no policy.py in this directory.\n")
        else:
            snap = ws / "history" / f"policy_v{i}.py"
            shutil.copy(policy, snap)
            reports = _evaluate(snap, ws / "history" / f"train_v{i}", workers, pixels=pixels)
            done, n = _score(reports)
            fb = revision_feedback(reports, i)
            (ws / "feedback.md").write_text(fb)
            (ws / "history" / f"feedback_v{i}.md").write_text(fb)
            rev.update(train_completed=done, train_notes=n, train_completion=done / max(n, 1))
            rev["train_summary"] = f"{done}/{n} notes completed ({100 * done / max(n, 1):.1f}%)"
        history["revisions"].append(rev)
        hist_path.write_text(json.dumps(history, indent=1))
        print(f"[cap] {name}: revision {i}: {rev['train_summary']}  (cost ${rev['cost_usd'] or 0:.2f})", flush=True)
    return ws


def heldout(ws: Path, workers: int, detector: str = "oracle") -> None:
    """Score every revision in a workspace on the held-out split."""
    from ..evaluate import evaluate, summarize

    hist_path = ws / "history.json"
    history = json.loads(hist_path.read_text())
    # Outside the workspace, so a resumed agent never sees held-out charts.
    out = ws.parent / f"{ws.name}_heldout"
    kind = "cappx" if history.get("pixels") else "cap"
    specs = [f"{kind}:{p}" for p in sorted((ws / "history").glob("policy_v*.py"), key=lambda p: int(p.stem[8:]))]
    evaluate(specs, "heldout", out, (detector,), workers=workers)
    summary = summarize(out)
    for rev in history["revisions"]:
        key = f"{kind}-policy_v{rev['revision']}|{detector}|heldout (all)"
        if key in summary:
            rev[f"heldout_completion_{detector}"] = summary[key]["completion"]
            for split in ("heldout-iid", "heldout-dense", "heldout-real"):
                s = summary.get(f"{kind}-policy_v{rev['revision']}|{detector}|{split}")
                if s:
                    rev[f"{split}_completion_{detector}"] = s["completion"]
    hist_path.write_text(json.dumps(history, indent=1))
    for rev in history["revisions"]:
        print(
            f"v{rev['revision']}: train {100 * rev.get('train_completion', 0):.1f}%  "
            f"held-out {100 * rev.get(f'heldout_completion_{detector}', 0):.1f}%"
        )


def selftest(policy: Path, workers: int, pixels: bool = False) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        reports = _evaluate(policy, Path(tmp) / "eval", workers, pixels=pixels)
    print(revision_feedback(reports))


def probe(script: Path, limit: int = 20_000) -> None:
    """Run the agent's probe.py `main(examples)` on the training frames, under
    the same import rules as a policy, and print what it prints."""
    import contextlib
    import io
    import traceback

    from .sandbox import load_module

    buf = io.StringIO()
    try:
        module = load_module(script)
        ex = [{k: e[k] for k in ("t", "frame", "notes")} for e in _examples(0.1)]
        with contextlib.redirect_stdout(buf):
            module.main(ex)
    except Exception:  # noqa: BLE001 - report the agent's own errors to it
        buf.write("\n" + traceback.format_exc(limit=-5))
    out = buf.getvalue()
    print(out[:limit] + (f"\n… ({len(out) - limit} more characters cut)" if len(out) > limit else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--iterations", type=int, default=6)
    r.add_argument("--model", default="claude-fable-5-1")
    r.add_argument("--self-test", action="store_true", help="let the agent run ./evaluate.sh itself")
    r.add_argument("--pixels", action="store_true", help="the program gets only the game frame")
    r.add_argument("--workers", type=int, default=6)
    h = sub.add_parser("heldout")
    h.add_argument("workspace")
    h.add_argument("--detector", default="oracle", choices=["oracle", "color", "torch"])
    h.add_argument("--workers", type=int, default=6)
    s = sub.add_parser("selftest")
    s.add_argument("policy")
    s.add_argument("--workers", type=int, default=6)
    s.add_argument("--pixels", action="store_true")
    pr = sub.add_parser("probe")
    pr.add_argument("script")
    args = ap.parse_args(argv)
    os.environ.setdefault("HEADLESS", "1")
    if args.cmd == "run":
        run(args.name, args.iterations, args.model, args.self_test, args.workers, args.pixels)
    elif args.cmd == "heldout":
        heldout(Path(args.workspace), args.workers, args.detector)
    elif args.cmd == "probe":
        probe(Path(args.script))
    else:
        selftest(Path(args.policy), args.workers, args.pixels)


if __name__ == "__main__":
    main()
