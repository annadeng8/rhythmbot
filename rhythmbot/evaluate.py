"""Run one or more policies on a chart split, headless and in parallel, and
summarize: the shared evaluation for the three approaches.

    python -m rhythmbot.evaluate --policy classical --split heldout
    python -m rhythmbot.evaluate --policy classical bc:results/bc_policy.pt cap:results/cap/policy.py \\
        --split heldout --detector oracle torch --out results/compare
    python -m rhythmbot.evaluate --summarize results/compare      # table from existing reports

Each (policy, detector, chart) run writes `<out>/<policy tag>/<detector>/<chart>.json`
(the judge's report, plus the split name and, for generated programs, their
error counts and planning time).  `summary.md` / `summary.json` in `<out>`
aggregate completion, timing, hold contact and slide error per split.

Every report also carries a `provenance` block (code commit and whether the
code had uncommitted edits, the judge's timing and slide rules, the arm and
gripper setup, and fingerprints of the vision model and the policy file), and
`summarize` warns when one summary mixes reports that differ in any of them:
numbers from different scoring rules are not comparable.  A run that exceeds
`--time-limit` (a generated program stuck in a loop) or raises is recorded as
a failed run with its reason instead of hanging or killing the evaluation.
Song charts that are not on disk (`songs/` is not in the repository) are
listed in `missing_charts.json` and in `summary.md` rather than skipped
silently, so two machines' totals can be told apart.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import signal
import subprocess
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("HEADLESS", "1")

import numpy as np  # noqa: E402

from . import PROJECT_ROOT, splits  # noqa: E402

DEFAULT_MODEL = PROJECT_ROOT / "results" / "note_detector.pt"
DEFAULT_TIME_LIMIT_S = 900


class RunTimeout(BaseException):
    """Raised by the alarm when a run exceeds its time limit.  A BaseException,
    so a generated policy's `except Exception` (or GuardedPolicy's) cannot
    swallow it."""


def _sha256(path: str | Path | None) -> str | None:
    if not path or not Path(path).is_file():
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def code_version() -> dict:
    """The commit the code was run from, and whether rhythmbot/ had
    uncommitted edits (in which case the commit alone does not pin it)."""

    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    return {"commit": git("rev-parse", "--short", "HEAD") or None,
            "uncommitted_code": bool(git("status", "--porcelain", "--", "rhythmbot"))}


def provenance(cfg, job: dict) -> dict:
    """What a report's numbers depend on besides the policy and the chart."""
    _, _, policy_file = job["policy"].partition(":")
    return {
        **job.get("code", {}),
        "judge": dataclasses.asdict(cfg.timing),
        "arms": cfg.arms.n,
        "contacts_per_arm": cfg.gripper.contacts_per_arm,
        "view": job.get("view", "screen"),
        "model_sha256": _sha256(job.get("model")),
        "policy_sha256": _sha256(policy_file) if policy_file else None,
    }


def _failed_report(chart, job: dict, extra: dict, reason: str) -> dict:
    return {"chart": chart.name, "n_notes": len(chart), "completion_rate": 0.0, "notes": [],
            "run_error": reason, **extra}


def _on_alarm(signum, frame):
    raise RunTimeout()


def policy_tag(spec: str) -> str:
    """A directory name for a policy spec: classical, bc-<file stem>, cap-<file stem>."""
    kind, _, arg = spec.partition(":")
    if not arg:
        return kind
    stem = Path(arg).stem
    if kind in ("cap", "cappx") and stem == "policy":
        stem = Path(arg).parent.name
    return f"{kind}-{re.sub(r'[^A-Za-z0-9_.-]', '_', stem)}"


def _run_one(job: dict) -> dict:
    from .config import DEFAULT, two_arm_config
    from .play import run
    from .policies import make_policy

    chart = splits.load(job["chart"])
    out = Path(job["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = two_arm_config() if job.get("arms", 1) == 2 else DEFAULT
    extra: dict = {"split": job["split"], "chart_entry": job["chart"], "provenance": provenance(cfg, job)}
    try:
        # Classical: `play` gives each arm its own planner.
        policy = None if job["policy"] == "classical" else make_policy(job["policy"], cfg)
    except Exception as e:  # a generated program that does not load
        rep = {"chart": chart.name, "n_notes": len(chart), "completion_rate": 0.0, "load_error": repr(e), **extra}
        out.write_text(json.dumps(rep, indent=1))
        return rep
    limit = int(job.get("time_limit") or 0)
    previous = signal.signal(signal.SIGALRM, _on_alarm) if limit > 0 else None
    if limit > 0:
        signal.alarm(limit)
    try:
        rep = run(
            chart,
            cfg,
            detector=job["detector"],
            model_path=job.get("model"),
            realtime=False,
            out_path=out,
            quiet=True,
            policy=policy,
            policy_name=job["policy"],
            view=job.get("view", "screen"),
        )
    except RunTimeout:
        rep = _failed_report(chart, job, extra, f"timeout: exceeded {limit} s")
    except Exception as e:  # noqa: BLE001 - record the crash, keep evaluating the rest
        rep = _failed_report(chart, job, extra, f"{type(e).__name__}: {e}"[:300])
        rep["traceback"] = traceback.format_exc(limit=-6)
    finally:
        if limit > 0:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
    if hasattr(policy, "stats"):
        extra["policy_stats"] = policy.stats()
    rep.update(extra)
    rep.pop("config", None)
    out.write_text(json.dumps(rep, indent=1))
    return rep


def evaluate(
    policies: list[str],
    split: str,
    out_dir: str | Path,
    detectors: tuple[str, ...] = ("oracle",),
    model: str | None = None,
    workers: int = 6,
    view: str = "screen",
    arms: int = 1,
    time_limit: int = DEFAULT_TIME_LIMIT_S,
) -> list[dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    code = code_version()
    if code["uncommitted_code"]:
        print("[evaluate] warning: rhythmbot/ has uncommitted edits; the commit in each report "
              "does not fully pin the code that produced it")
    jobs, missing = [], []
    for part in splits.expand(split):
        for name in splits.SPLITS.get(part, [part]):
            if name.startswith("song:") and splits.load(name) is None:
                missing.append({"split": part, "chart": name})
                continue
            for pol in policies:
                for det in detectors:
                    stem = name.replace(":", "-").replace("/", "-")
                    jobs.append(
                        {
                            "chart": name,
                            "split": part,
                            "policy": pol,
                            "detector": det,
                            "model": str(model or DEFAULT_MODEL) if det == "torch" else None,
                            "out": str(out_dir / policy_tag(pol) / det / f"{stem}.json"),
                            "view": view,
                            "arms": arms,
                            "time_limit": time_limit,
                            "code": code,
                        }
                    )
    _record_missing(out_dir, missing)
    if workers <= 1:
        return [_run_one(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_run_one, jobs))


def _record_missing(out_dir: Path, missing: list[dict]) -> None:
    """Merge this call's missing song charts into `<out>/missing_charts.json`."""
    path = out_dir / "missing_charts.json"
    known = json.loads(path.read_text()) if path.exists() else []
    for m in missing:
        if m not in known:
            known.append(m)
    path.write_text(json.dumps(known, indent=1))
    if missing:
        names = ", ".join(sorted({m["chart"] for m in missing}))
        print(f"[evaluate] warning: {len(missing)} song chart(s) not on disk and not run: {names} "
              "(songs/ is gitignored; see README to import them)")


# -- summaries ------------------------------------------------------------------
def _agg(reps: list[dict]) -> dict:
    notes = [n for r in reps for n in r.get("notes", [])]
    n = sum(r["n_notes"] for r in reps)
    done = sum(bool(x.get("completed")) for x in notes)
    errs = np.array([x["timing_error"] for x in notes if x.get("timing_error") is not None])
    maint = [x["maintenance"] for x in notes if x["kind"] != "tap" and x.get("maintenance") is not None]
    # Contact over every hold/slide, counting a missed one as zero contact.
    maint_all = [x.get("maintenance") or 0.0 for x in notes if x["kind"] != "tap"]
    rms = [x["slide_rms_m"] for x in notes if x.get("slide_rms_m") is not None]
    out = {
        "charts": len(reps),
        "notes": n,
        "completed": done,
        "completion": done / n if n else None,
        "chart_completion_mean": float(np.mean([r.get("completion_rate") or 0.0 for r in reps])) if reps else None,
        "hit_rate": sum(x.get("status") == "hit" for x in notes) / n if n else None,
        "perfect": sum(x.get("judgment") == "perfect" for x in notes),
        "timing_abs_ms": 1000 * float(np.abs(errs).mean()) if len(errs) else None,
        "timing_std_ms": 1000 * float(errs.std()) if len(errs) else None,
        "contact_when_hit": float(np.mean(maint)) if maint else None,
        "contact_all": float(np.mean(maint_all)) if maint_all else None,
        "slide_rms_mm": 1000 * float(np.mean(rms)) if rms else None,
        "load_errors": sum("load_error" in r for r in reps),
        "run_errors": sum("run_error" in r for r in reps),
        "plan_errors": sum(r.get("policy_stats", {}).get("plan_errors", 0) for r in reps),
    }
    for kind in ("tap", "hold", "slide"):
        ks = [x for x in notes if x["kind"] == kind]
        out[f"{kind}_completion"] = sum(bool(x.get("completed")) for x in ks) / len(ks) if ks else None
    return out


def summarize(out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for path in sorted(out_dir.glob("*/*/*.json")):
        rep = json.loads(path.read_text())
        pol, det = path.parts[-3], path.parts[-2]
        split = rep.get("split", "other")
        groups.setdefault((pol, det, split), []).append(rep)
        if split.startswith("heldout-"):
            groups.setdefault((pol, det, "heldout (all)"), []).append(rep)
    summary = {f"{p}|{d}|{s}": _agg(reps) for (p, d, s), reps in sorted(groups.items())}
    notes = consistency_notes([json.loads(p.read_text()) for p in sorted(out_dir.glob("*/*/*.json"))])
    missing_path = out_dir / "missing_charts.json"
    missing = json.loads(missing_path.read_text()) if missing_path.exists() else []
    if missing:
        notes.append("Not run, because the song chart is not on disk: "
                     + ", ".join(sorted({m["chart"] for m in missing})) + ".")
    for n in notes:
        print(f"[summarize] warning: {n}")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (out_dir / "summary_notes.json").write_text(json.dumps(notes, indent=1))
    table = format_table(summary)
    if notes:
        table += "\n" + "\n".join(f"* {n}" for n in notes) + "\n"
    (out_dir / "summary.md").write_text(table)
    return summary


def consistency_notes(reps: list[dict]) -> list[str]:
    """Warnings for reports that should not be added up into one table."""
    out = []
    provs = [r.get("provenance") for r in reps]
    if any(p is None for p in provs):
        out.append(f"{sum(p is None for p in provs)} of {len(provs)} reports have no provenance "
                   "(written before it was recorded): their scoring rules are unknown.")
    provs = [p for p in provs if p]
    judges = {json.dumps(p["judge"], sort_keys=True) for p in provs}
    if len(judges) > 1:
        out.append(f"Reports were scored under {len(judges)} different judge settings; "
                   "their completion rates are not comparable.")
    commits = {p.get("commit") for p in provs}
    if len(commits) > 1:
        out.append("Reports come from different code commits: " + ", ".join(sorted(str(c) for c in commits)) + ".")
    if any(p.get("uncommitted_code") for p in provs):
        out.append("Some reports were produced with uncommitted code edits.")
    models = {p.get("model_sha256") for p in provs if p.get("model_sha256")}
    if len(models) > 1:
        out.append(f"Vision-model runs used {len(models)} different models.")
    return out


def _pct(v):
    return "–" if v is None else f"{100 * v:.1f}%"


def _num(v, fmt="{:.0f}"):
    return "–" if v is None else fmt.format(v)


def format_table(summary: dict) -> str:
    lines = [
        "| policy | detector | split | charts | notes | completion | tap / hold / slide | "
        "\\|timing err\\| ms | contact (hit holds+slides) | slide RMS mm | errors |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, a in summary.items():
        pol, det, split = key.split("|")
        kinds = " / ".join(_pct(a[f"{k}_completion"]) for k in ("tap", "hold", "slide"))
        errs = a["load_errors"] + a["plan_errors"] + a.get("run_errors", 0)
        lines.append(
            f"| {pol} | {det} | {split} | {a['charts']} | {a['notes']} | {_pct(a['completion'])} | {kinds} | "
            f"{_num(a['timing_abs_ms'])} | {_pct(a['contact_when_hit'])} | {_num(a['slide_rms_mm'], '{:.1f}')} | "
            f"{errs or ''} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", nargs="+", default=["classical"])
    ap.add_argument("--split", default="heldout", help=f"one of {', '.join(splits.SPLITS)} (comma-separated ok)")
    ap.add_argument("--detector", nargs="+", default=["oracle"], choices=["oracle", "color", "torch"])
    ap.add_argument("--model", help="vision model for --detector torch (default results/note_detector.pt)")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "compare"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--view", default="screen", choices=["screen", "camera"],
                    help="perception input: the game's own frame, or an external camera on the screen")
    ap.add_argument("--arms", type=int, default=1, choices=[1, 2], help="one arm, or two side by side (classical only)")
    ap.add_argument("--time-limit", type=int, default=DEFAULT_TIME_LIMIT_S,
                    help="seconds per run before it is recorded as a timeout (0: no limit)")
    ap.add_argument("--summarize", metavar="DIR", help="only rebuild the summary of an existing output dir")
    args = ap.parse_args(argv)
    if args.summarize:
        print(format_table(summarize(args.summarize)))
        return
    evaluate(args.policy, args.split, args.out, tuple(args.detector), args.model, args.workers, args.view, args.arms,
             args.time_limit)
    print(format_table(summarize(args.out)))


if __name__ == "__main__":
    main()
