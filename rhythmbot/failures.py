"""Why notes fail: perception, the robot's limits, or the policy's control.

    python -m rhythmbot.failures results/eval-v2                 # table for every policy and detector
    python -m rhythmbot.failures results/eval-v2 --examples 5    # plus a few example notes per cause

Reads the per-run reports that `rhythmbot.evaluate` writes
(`<out>/<policy>/<detector>/<chart>.json`) and puts every note that was not
completed into exactly one cause, checked in this order:

    perception        a vision-model (or color) run failed a note that the
                      same policy completed on the same chart with the oracle
                      detector: the control could have played it, the
                      perception lost it
    run failed        the whole run failed (generated program would not load,
                      timed out or crashed): every note of the chart
    lost contact      the head was hit but the hold or slide was not kept
                      (the judge's `failure_reason`: too little contact, a
                      contact break, a missing or wrong slide tail)
    beyond the robot  the head was missed and the chart asks for more than
                      the robot has at that moment:
                        * more lanes at once than it has fingertips
                          (simultaneous heads plus holds and slides in
                          progress in other lanes),
                        * (one arm) a chord wider than the gripper opens,
                        * a change to another lane less than
                          `FAST_LANE_CHANGE_S` after the previous note
    skipped           (classical planner only) the head was missed and the
                      planner reported skipping it, without the chart being
                      beyond the robot by the checks above
    missed head       the head was missed for no structural reason: the
                      policy did not touch the lane inside the timing window

"Beyond the robot" is a geometric screen on the chart, not a proof: a better
policy might still play some of those notes, and some notes outside it may
be impossible for reasons the screen does not see (joint speed limits, a
hand still lifting).  It is what separates "this embodiment cannot" from
"this policy did not" in a way that is identical for every approach.

Writes `failures.md` and `failures.json` next to the reports.
"""

from __future__ import annotations

import argparse
import bisect
import json
from collections import Counter, defaultdict
from pathlib import Path

from . import splits
from .chart import HOLD, SLIDE, Chart
from .config import DEFAULT, Config, two_arm_config

CAUSES = ["perception", "run failed", "lost contact", "beyond the robot", "skipped", "missed head"]
FAST_LANE_CHANGE_S = 0.15
SIMULTANEOUS_S = 0.02


# ---------------------------------------------------------------------------
# What the chart asks of the robot at each note
# ---------------------------------------------------------------------------
class ChartDemands:
    """Per-note structural demands of a chart, for a given robot setup."""

    def __init__(self, chart: Chart, cfg: Config, contacts: int):
        self.chart = chart
        self.cfg = cfg
        self.contacts = contacts
        self.notes = sorted(chart.notes, key=lambda n: n.time)
        self.times = [n.time for n in self.notes]
        self.long_notes = [n for n in self.notes if n.kind in (HOLD, SLIDE)]
        g = cfg.gripper
        self.reach = 2 * g.half_closed + g.max_width  # widest fingertip gap of one gripper

    def _lanes_needed(self, note) -> set[float]:
        lanes = {note.lane}
        i = bisect.bisect_left(self.times, note.time - SIMULTANEOUS_S)
        while i < len(self.notes) and self.notes[i].time <= note.time + SIMULTANEOUS_S:
            lanes.add(self.notes[i].lane)
            i += 1
        for m in self.long_notes:
            if m.id != note.id and m.time < note.time - SIMULTANEOUS_S and m.end_time > note.time:
                lanes.add(m.lane_at(note.time))
        return lanes

    def reason(self, note) -> str | None:
        """Why this note is beyond the robot, or None."""
        lanes = self._lanes_needed(note)
        if len(lanes) > self.contacts:
            return f"needs {len(lanes)} lanes at once ({self.contacts} fingertips)"
        if self.cfg.arms.n == 1 and self.cfg.gripper.contacts_per_arm > 1 and len(lanes) > 1:
            L = self.cfg.layout
            span = max(L.lane_x(a) for a in lanes) - min(L.lane_x(a) for a in lanes)
            if span > self.reach + 1e-9:
                return "chord wider than the gripper opens"
        i = bisect.bisect_left(self.times, note.time - SIMULTANEOUS_S) - 1
        if i >= 0:
            prev = self.notes[i]
            if prev.lane != note.lane and note.time - prev.time < FAST_LANE_CHANGE_S:
                return f"lane change in under {FAST_LANE_CHANGE_S:.2f} s"
        return None


def _config_for(prov: dict | None) -> tuple[Config, int]:
    prov = prov or {}
    cfg = two_arm_config() if prov.get("arms", 1) == 2 else DEFAULT
    contacts = prov.get("contacts_per_arm", cfg.gripper.contacts_per_arm) * cfg.arms.n
    return cfg, contacts


# ---------------------------------------------------------------------------
# Classifying a directory of reports
# ---------------------------------------------------------------------------
def load_reports(out_dir: Path) -> dict[tuple[str, str, str], dict]:
    """(policy, detector, chart entry) -> report."""
    reps = {}
    for path in sorted(out_dir.glob("*/*/*.json")):
        rep = json.loads(path.read_text())
        pol, det = path.parts[-3], path.parts[-2]
        reps[(pol, det, rep.get("chart_entry", path.stem))] = rep
    return reps


def classify(out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    reps = load_reports(out_dir)
    demands: dict[tuple[str, int, int], ChartDemands] = {}
    rows: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    totals: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    examples: dict[tuple[str, str, str], list] = defaultdict(list)

    for (pol, det, entry), rep in reps.items():
        groups = [rep.get("split", "other")]
        if groups[0].startswith("heldout-"):
            groups.append("heldout (all)")
        n_notes = rep.get("n_notes", 0)
        for g in groups:
            totals[(pol, det, g)]["notes"] += n_notes
        if "run_error" in rep or "load_error" in rep or not rep.get("notes"):
            for g in groups:
                rows[(pol, det, g)]["run failed"] += n_notes
            continue
        chart = splits.load(entry)
        if chart is None:
            continue
        prov = rep.get("provenance")
        cfg, contacts = _config_for(prov)
        key = (entry, cfg.arms.n, contacts)
        if key not in demands:
            demands[key] = ChartDemands(chart, cfg, contacts)
        dem = demands[key]
        by_id = {n.id: n for n in chart.notes}
        oracle = reps.get((pol, "oracle", entry)) if det != "oracle" else None
        oracle_done = {x["id"] for x in oracle.get("notes", []) if x.get("completed")} if oracle else set()
        skipped = set(rep.get("skipped_by_planner") or [])
        for x in rep["notes"]:
            if x.get("completed"):
                continue
            note = by_id.get(x["id"])
            if x["id"] in oracle_done:
                cause, detail = "perception", "completed with the oracle"
            elif x.get("status") == "hit":
                cause, detail = "lost contact", x.get("failure_reason") or "too little contact"
            else:
                why = dem.reason(note) if note is not None else None
                if why:
                    cause, detail = "beyond the robot", why
                elif x["id"] in skipped:
                    cause, detail = "skipped", "planner skipped it"
                else:
                    cause, detail = "missed head", "no touch in the lane in time"
            for g in groups:
                rows[(pol, det, g)][cause] += 1
                rows[(pol, det, g)][f"{cause}: {detail}"] += 1
                if len(examples[(pol, det, g)]) < 50:
                    examples[(pol, det, g)].append(
                        {"chart": entry, "id": x["id"], "kind": x["kind"], "lane": x["lane"],
                         "time": x["time"], "cause": cause, "detail": detail}
                    )

    result = {}
    for key in sorted(set(totals) | set(rows)):
        pol, det, g = key
        c = rows[key]
        n = totals[key]["notes"]
        failed = sum(c[k] for k in CAUSES)
        result[f"{pol}|{det}|{g}"] = {
            "notes": n,
            "failed": failed,
            "causes": {k: c[k] for k in CAUSES},
            "details": {k: v for k, v in sorted(c.items()) if ":" in k},
            "examples": examples[key],
        }
    (out_dir / "failures.json").write_text(json.dumps(result, indent=1))
    (out_dir / "failures.md").write_text(format_table(result))
    return result


def format_table(result: dict, examples: int = 0) -> str:
    head = "| policy | detector | split | notes | failed | " + " | ".join(CAUSES) + " |"
    lines = [head, "|" + "---|" * (5 + len(CAUSES))]
    for key, r in result.items():
        pol, det, g = key.split("|")
        cells = [str(r["causes"][k]) if r["causes"][k] else "" for k in CAUSES]
        lines.append(f"| {pol} | {det} | {g} | {r['notes']} | {r['failed']} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    if examples:
        for key, r in result.items():
            if not r["examples"]:
                continue
            text += f"\n**{key.replace('|', ' / ')}**\n\n"
            for d, v in sorted(r["details"].items(), key=lambda kv: -kv[1]):
                text += f"* {d}: {v}\n"
            for e in r["examples"][:examples]:
                text += f"  * {e['chart']} #{e['id']} {e['kind']} lane {e['lane']:g} at {e['time']:.2f} s: {e['detail']}\n"
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="an output directory of rhythmbot.evaluate")
    ap.add_argument("--examples", type=int, default=0, help="also list causes in detail and N example notes each")
    args = ap.parse_args(argv)
    result = classify(args.out_dir)
    print(format_table(result, args.examples))


if __name__ == "__main__":
    main()
