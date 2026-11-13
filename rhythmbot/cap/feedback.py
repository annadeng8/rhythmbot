"""Turn the judge's reports for one program revision into the feedback the
coding agent reads: per chart scores, then every note that was not
completed with what the chart asked and what the fingertips did, then the
program's own errors and planning time."""

from __future__ import annotations

import json
from pathlib import Path

MAX_NOTES_PER_CHART = 25


def _why(n: dict) -> str:
    if n["status"] == "miss":
        return "no fresh touch in its lane within ±140 ms of the hit time"
    m = n.get("maintenance")
    parts = [f"hit {1000 * n['timing_error']:+.0f} ms ({n['judgment']})"]
    if m is not None:
        parts.append(f"contact {100 * m:.1f}%; failure={n.get('failure_reason')}; max gap={1000*n.get('max_gap_seconds',0):.0f} ms; tail={n.get('tail_contact_ticks',0)}/{n.get('tail_ticks',0)}")
    if n.get("slide_rms_m") is not None:
        parts.append(f"slide lateral RMS error {1000 * n['slide_rms_m']:.0f} mm")
    return ", ".join(parts)


def _note(n: dict) -> str:
    s = f"#{n['id']} {n['kind']} lane {n['lane']:g} at {n['time']:.2f} s"
    if n["kind"] != "tap":
        s += f" for {n['duration']:.2f} s"
    return s


def chart_feedback(rep: dict, chart_notes: list[dict] | None = None) -> str:
    if "load_error" in rep:
        return f"### {rep['chart']}\n\nThe program did not load: `{rep['load_error']}`\n"
    j = rep["judgments"]
    lines = [
        f"### {rep['chart']}: {100 * rep['completion_rate']:.1f}% completed "
        f"({sum(bool(n['completed']) for n in rep['notes'])}/{rep['n_notes']})",
        "",
        f"judgments perfect {j['perfect']} / great {j['great']} / good {j['good']} / miss {j['miss']}",
    ]
    if rep.get("timing_error_mean_s") is not None:
        lines.append(
            f"timing error of hits: mean {1000 * rep['timing_error_mean_s']:+.0f} ms, "
            f"std {1000 * rep['timing_error_std_s']:.0f} ms"
        )
    for kind, d in rep["by_kind"].items():
        s = f"- {kind}: {d['completed']}/{d['n']} completed, {d['hit']} hit"
        if d.get("mean_maintenance") is not None:
            s += f", contact {100 * d['mean_maintenance']:.0f}%"
        if d.get("mean_slide_rms_m") is not None:
            s += f", slide RMS {1000 * d['mean_slide_rms_m']:.0f} mm"
        lines.append(s)
    bad = [n for n in rep["notes"] if not n["completed"]]
    if bad:
        notes = sorted(rep["notes"], key=lambda n: n["time"])
        idx = {n["id"]: i for i, n in enumerate(notes)}
        skipped = set(rep.get("skipped_by_planner", []))
        lines += ["", "Notes not completed (with the notes just before and after, for context):"]
        for n in bad[:MAX_NOTES_PER_CHART]:
            i = idx[n["id"]]
            ctx = []
            if i > 0:
                p = notes[i - 1]
                ctx.append(f"prev {p['kind']} lane {p['lane']:g} at {p['time']:.2f}" + (
                    f"–{p['time'] + p['duration']:.2f}" if p["kind"] != "tap" else ""))
            if i + 1 < len(notes):
                q = notes[i + 1]
                ctx.append(f"next {q['kind']} lane {q['lane']:g} at {q['time']:.2f}")
            tag = " [you marked it skipped]" if n["id"] in skipped else ""
            lines.append(f"- {_note(n)}{tag}: {_why(n)}. ({'; '.join(ctx)})")
        if len(bad) > MAX_NOTES_PER_CHART:
            lines.append(f"- … and {len(bad) - MAX_NOTES_PER_CHART} more")
    st = rep.get("policy_stats")
    if st:
        lines += ["", f"plan(): {st['plan_calls']} calls, mean {st['plan_ms_mean']:.1f} ms, max {st['plan_ms_max']:.1f} ms"]
        if st["plan_errors"]:
            lines.append(f"**{st['plan_errors']} calls raised or returned a bad plan** (the hand was lifted in place):")
            for e in st["errors"]:
                lines += [f"- {e['count']}× `{e['error']}`", "```", e["traceback"].rstrip(), "```"]
    return "\n".join(lines) + "\n"


def revision_feedback(reports: list[dict], revision: int | None = None) -> str:
    n = sum(r["n_notes"] for r in reports)
    done = sum(sum(bool(x["completed"]) for x in r.get("notes", [])) for r in reports)
    head = f"# Feedback{'' if revision is None else f' on revision {revision}'}\n\n"
    head += f"Overall: {done}/{n} notes completed ({100 * done / max(n, 1):.1f}%) on {len(reports)} charts.\n\n"
    return head + "\n".join(chart_feedback(r) for r in sorted(reports, key=lambda r: r["chart"]))


def load_reports(directory: str | Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(Path(directory).glob("*.json"))]
