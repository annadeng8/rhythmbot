"""Agentic code-as-policy: a coding agent writes the control program, the
simulator scores it, the agent reads the feedback and revises.

    python -m rhythmbot.cap.loop run --name fable-1 --iterations 6
    python -m rhythmbot.cap.loop run --name fable-selftest --self-test     # agent may also run the sim itself
    python -m rhythmbot.cap.loop heldout ../rhythmbot_cap_runs/fable-1                  # every revision on the held-out charts
    python -m rhythmbot.cap.loop selftest ../rhythmbot_cap_runs/fable-1/policy.py   # what --self-test lets the agent run

`run` makes a workspace `<RHYTHMBOT_CAP_RUNS>/<name>/` (default
`../rhythmbot_cap_runs`, outside the repository) holding only `TASK.md` (the
problem statement), `api.py` (the control API's source) and, once the agent
has written it, `policy.py`.  Each iteration is one non-interactive Claude
Code call in that directory, continuing the same conversation: the first
asks for a program, later ones point at `feedback.md`.  After each call the
program is copied to `history/policy_v<i>.py`, run headless on the
`cap-train` charts with the oracle detector, and the feedback for the next
iteration is written.  `history.json` keeps the per-revision score, the
agent's cost and turns.

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

from .. import PROJECT_ROOT
from .feedback import load_reports, revision_feedback

HERE = Path(__file__).resolve().parent
# Workspaces live outside the repository by default, so nothing above the
# agent's working directory is the project it could read.
RUNS = Path(os.environ.get("RHYTHMBOT_CAP_RUNS", PROJECT_ROOT.parent / "rhythmbot_cap_runs"))
TRAIN_SPLIT = "cap-train"

FIRST_PROMPT = (
    "Read TASK.md and api.py in this directory, then write policy.py: the control program for the "
    "rhythm-game robot described there. Aim for a program that completes as many notes as possible "
    "on charts you have not seen. When policy.py is written and you are done, reply with a short "
    "summary of its strategy."
)
NEXT_PROMPT = (
    "Your revision {prev} was run on the training charts: {score}. feedback.md has the details "
    "(per-chart scores, every note not completed and why, and any errors your program raised). "
    "Revise policy.py to complete more notes; remember it will be judged on unseen charts. "
    "Reply with a short summary of what you changed and why."
)
SELFTEST_NOTE = (
    " You can test a revision yourself at any time with `./evaluate.sh` (runs policy.py on the "
    "training charts, about a minute, and prints the same kind of feedback)."
)


def _claude(prompt: str, cwd: Path, model: str, session: str | None, self_test: bool, log: Path) -> dict:
    tools = ["Read", "Write", "Edit", "Glob", "Grep"]
    denied = ["WebFetch", "WebSearch", "Task", "Agent", "NotebookEdit"]
    # Reads are allowed anywhere by default: shut out the project (the
    # classical planner, the charts) and every other workspace (other runs,
    # held-out reports).
    for d in [PROJECT_ROOT] + [p for p in cwd.parent.iterdir() if p.is_dir() and p != cwd]:
        for tool in ("Read", "Glob", "Grep"):
            denied.append(f"{tool}(/{d.resolve()}/**)")
    if self_test:
        tools.append("Bash(./evaluate.sh)")
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
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=3 * 3600)
    log.write_text(proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else ""))
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        out = {"is_error": True, "result": proc.stdout[-2000:] + proc.stderr[-2000:]}
    out["wall_s"] = time.time() - t0
    return out


def _evaluate(policy: Path, out: Path, workers: int, split: str = TRAIN_SPLIT) -> list[dict]:
    from ..evaluate import evaluate

    if out.exists():
        shutil.rmtree(out)
    evaluate([f"cap:{policy}"], split, out, ("oracle",), workers=workers)
    return load_reports(next(out.glob("*/oracle")))


def _score(reports: list[dict]) -> tuple[int, int]:
    n = sum(r["n_notes"] for r in reports)
    done = sum(sum(bool(x["completed"]) for x in r.get("notes", [])) for r in reports)
    return done, n


def setup_workspace(ws: Path, self_test: bool) -> None:
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "history").mkdir(exist_ok=True)
    (ws / "agent_log").mkdir(exist_ok=True)
    task = (HERE / "TASK.md").read_text()
    if self_test:
        task += "\n## Testing it yourself\n\n`./evaluate.sh` runs `policy.py` on the training charts and prints that feedback.\n"
    (ws / "TASK.md").write_text(task)
    shutil.copy(HERE / "api.py", ws / "api.py")
    if self_test:
        sh = ws / "evaluate.sh"
        sh.write_text(
            "#!/bin/sh\n"
            f'cd "{PROJECT_ROOT}" && HEADLESS=1 "{sys.executable}" -m rhythmbot.cap.loop selftest "{ws / "policy.py"}"\n'
        )
        sh.chmod(0o755)


def run(name: str, iterations: int, model: str, self_test: bool, workers: int) -> Path:
    ws = RUNS / name
    hist_path = ws / "history.json"
    history = json.loads(hist_path.read_text()) if hist_path.exists() else {"model": model, "self_test": self_test, "revisions": []}
    if not history["revisions"]:
        setup_workspace(ws, self_test)
    session = history.get("session")
    for i in range(len(history["revisions"]), iterations):
        if i == 0:
            prompt = FIRST_PROMPT
        else:
            prev = history["revisions"][-1]
            prompt = NEXT_PROMPT.format(prev=i - 1, score=prev["train_summary"])
        if self_test:
            prompt += SELFTEST_NOTE
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
            reports = _evaluate(snap, ws / "history" / f"train_v{i}", workers)
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
    specs = [f"cap:{p}" for p in sorted((ws / "history").glob("policy_v*.py"), key=lambda p: int(p.stem[8:]))]
    evaluate(specs, "heldout", out, (detector,), workers=workers)
    summary = summarize(out)
    for rev in history["revisions"]:
        key = f"cap-policy_v{rev['revision']}|{detector}|heldout (all)"
        if key in summary:
            rev[f"heldout_completion_{detector}"] = summary[key]["completion"]
            for split in ("heldout-iid", "heldout-dense", "heldout-real"):
                s = summary.get(f"cap-policy_v{rev['revision']}|{detector}|{split}")
                if s:
                    rev[f"{split}_completion_{detector}"] = s["completion"]
    hist_path.write_text(json.dumps(history, indent=1))
    for rev in history["revisions"]:
        print(
            f"v{rev['revision']}: train {100 * rev.get('train_completion', 0):.1f}%  "
            f"held-out {100 * rev.get(f'heldout_completion_{detector}', 0):.1f}%"
        )


def selftest(policy: Path, workers: int) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        reports = _evaluate(policy, Path(tmp) / "eval", workers)
    print(revision_feedback(reports))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--iterations", type=int, default=6)
    r.add_argument("--model", default="claude-fable-5-1")
    r.add_argument("--self-test", action="store_true", help="let the agent run ./evaluate.sh itself")
    r.add_argument("--workers", type=int, default=6)
    h = sub.add_parser("heldout")
    h.add_argument("workspace")
    h.add_argument("--detector", default="oracle", choices=["oracle", "color", "torch"])
    h.add_argument("--workers", type=int, default=6)
    s = sub.add_parser("selftest")
    s.add_argument("policy")
    s.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    os.environ.setdefault("HEADLESS", "1")
    if args.cmd == "run":
        run(args.name, args.iterations, args.model, args.self_test, args.workers)
    elif args.cmd == "heldout":
        heldout(Path(args.workspace), args.workers, args.detector)
    else:
        selftest(Path(args.policy), args.workers)


if __name__ == "__main__":
    main()
