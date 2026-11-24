# Single-arm CAP with the v7 detector

The single-arm CAP programs were last scored with the older detector
(`note_detector.pt`, in [`eval-v2`](../eval-v2/summary.md)). This reruns them,
and classical, with v7 on the same 24 held-out charts (1,810 notes). Run on
2026-11-24:

```bash
HEADLESS=1 ../.venv/bin/python -m rhythmbot.evaluate --policy classical \
    cap:results/cap/fable-selftest/policy_v1.py cap:results/cap/fable-feedback/policy_v7.py \
    --split heldout,heldout-song --detector torch --model results/note_detector_v7.pt \
    --workers 4 --out results/eval-v3-cap
```

| Controller | Older detector (eval-v2) | v7 | Slides, older → v7 | Oracle slides |
|---|---:|---:|---|---:|
| Classical | 97.3% | **98.3%** | 52.6% → 84.2% | 91.2% |
| CAP `policy_v1` (self-tested) | 80.9% | 80.1% | 49.1% → **78.9%** | 91.2% |
| CAP `policy_v7` (feedback) | 85.3% | 85.3% | 36.8% → 33.3% | 80.7% |

- **Classical reproduces eval-v3 exactly** (98.3%, slides 84.2%), so the
  current code gives the same results as the frozen run.
- **v7 fixes slides for the first CAP program** (49% → 79%), as it did for
  classical, but its overall completion is unchanged (taps 81.5% → 79.4%).
- **The feedback-revised CAP program doesn't benefit.** Its slides stay at
  about a third even though it reaches 80.7% with the oracle. A likely cause,
  not tested: it aims 40 ms ahead along the slide (`slide_lead = 0.04`),
  which is where v7's read-off path is least accurate near corners (see the
  diagnosis in [`sekai-finetune-v1`](../sekai-finetune-v1/README.md)).
- **CAP's gap to classical is still mostly notes it chooses not to play**, on
  the songs: 71% and 81% vs 98%.

Per split: [`summary.md`](summary.md).
