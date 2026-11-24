# Fixing v7's slide misses on zigzags

Follow-up to the diagnosis in [`sekai-finetune-v1`](../sekai-finetune-v1/README.md#why-v7-misses-slides-on-two-arms-gripper-setup-b):
v7 missed 10 of 16 slides on the two gripper arms, 8 of them fast zigzags
(lanes 3→5→3→5→4 in 1.4 s). Run on Anna's Apple M5 Mac on 2026-11-24.

## What was wrong

Logging v7's raw rest-of-slide readings on `holds-s71000` showed that it is
not only the straight lines between its 5 points that cut corners: the
middle points themselves are off by 0.13–0.31 lanes on average (90th
percentile up to 0.96). v7's training charts (`rhythmbot.dataset`) only
have slides of 1–3 segments; the study's zigzags have 4, and 48 of them
are in Yifan's training charts.

## Two changes

1. **Tracker: merge readings instead of replacing them**
   (`RhythmBrain._replace_rest`). Each frame's 5 points sit at fractions of
   the time left, so they land at different times every frame. Points from
   earlier readings that are still ahead are now kept between the new ones
   (a new point replaces any within 10 ms), so the path ahead fills in as
   the slide is played. This only helps when the points are right.
2. **Model: `note_detector_v7_zigzag.pt`**, v7 fine-tuned on plain-screen
   frames of Yifan's 48 training charts (14,090 frames,
   [`scripts/render_study_frames.py`](../../scripts/render_study_frames.py))
   mixed with v7's own 33,067 training frames so it keeps what it knew.
   12 epochs, lr 3e-4, last epoch kept ([`finetune.log`](finetune.log)).
   Validation is Yifan's 6 validation charts plus v7's own validation set;
   no test chart is used. Rest-of-slide lane error 0.09 → 0.03, slide end
   lane error 0.78 → 0.40.

```bash
python scripts/render_study_frames.py --split train --mix data/train --out data/study-screen/train
python scripts/render_study_frames.py --split validation --mix data/val --out data/study-screen/val
python -m rhythmbot.train --init results/note_detector_v7.pt --train data/study-screen/train \
    --val data/study-screen/val --model results/note_detector_v7_zigzag.pt --epochs 12 --lr 3e-4
```

Path error on `holds-s71000` (two gripper arms,
[`scripts/diagnose_slide_path.py`](../../scripts/diagnose_slide_path.py)),
mean / max lanes; the judge allows 0.35:

| Look-ahead | v7, replace (before) | v7, merge | Zigzag v7, merge |
|---|---|---|---|
| Now | 0.03 / 0.13 | 0.03 / 0.13 | 0.01 / 0.03 |
| 33 ms | 0.07 / 0.35 | 0.13 / 0.50 | 0.02 / 0.14 |
| 100 ms | 0.14 / 0.58 | 0.15 / 0.50 | 0.03 / 0.20 |
| 200 ms | 0.15 / 0.49 | 0.18 / 0.69 | 0.03 / 0.22 |

## Two gripper arms, plain screen (11 test charts, 323 notes)

Classical planner, same command as
[`gripper-2arm-final-charts`](../gripper-2arm-final-charts/summary.md) with
`--arms 2 --detector torch`.

| Model | Tracker | All notes | Slides (of 16) | Reports |
|---|---|---:|---:|---|
| Oracle input | – | 99.4% | 14 | [gripper-2arm-final-charts](../gripper-2arm-final-charts/summary.md) |
| v7 | replace (before) | 96.9% | 6 | [gripper-2arm-v7-replace](gripper-2arm-v7-replace/summary.md) |
| v7 | merge | 96.6% | 5 | [gripper-2arm-v7-merge](gripper-2arm-v7-merge/summary.md) |
| Zigzag v7 | replace | 97.5% | 8 | [gripper-2arm-zigzag-replace](gripper-2arm-zigzag-replace/summary.md) |
| **Zigzag v7** | **merge** | **98.8%** | **12** | [gripper-2arm-zigzag-merge](gripper-2arm-zigzag-merge/summary.md) |

The "before" run reproduces the frozen result exactly. Taps (291) and holds
(16) are complete in every row. With both changes every zigzag slide is
completed; the 4 misses left are one slide on each `crossing` chart, two of
them on the speed charts that the oracle also misses.

## Single arm (24 held-out charts, 1,810 notes)

Zigzag v7 with the merge against [`eval-v3-cap`](../eval-v3-cap/README.md)
(v7, replace):

| Controller | All notes, before → after | Slides, before → after | Unseen song |
|---|---|---|---|
| Classical | 98.3% → 98.2% | 84.2% → **91.2%** (oracle 91.2%) | 98.4% → 97.9% |
| CAP self-test (`policy_v1`) | 80.1% → 80.5% | 78.9% → 87.7% | 71.2% → 71.5% |
| CAP feedback (`policy_v7`) | 85.3% → 85.6% | 33.3% → 29.8% | 81.2% → 82.0% |

- Classical slides now match oracle input. The song excerpts lose 5 taps
  of 1,106 (98.3% → 97.9%), the one regression.
- The merge alone with the original v7 changes nothing on one arm
  ([single-arm-v7-merge](single-arm-v7-merge/summary.md): 98.3%, slides
  84.2%), so the single-arm change is the model's.
- The feedback CAP program's slide problem is its own (it aims 40 ms ahead
  along the slide, `slide_lead = 0.04`), not the detector's.

## Yifan's camera setup (11 test charts, `concert-torch`)

The camera-fine-tuned model ([`sekai-finetune-v1`](../sekai-finetune-v1/README.md))
already saw zigzags in its training charts, so only the tracker change
applies. Rerun with [`eval_sekai_finetune.py`](../../scripts/eval_sekai_finetune.py)
`--detectors v7-fine-tuned`, [`sekai-camera`](sekai-camera/summary.md):

| Controller | Color | Fine-tuned v7, replace | Fine-tuned v7, merge |
|---|---|---|---|
| Classical | **89.2%** (10 slides) | 87.0% (7) | 87.6% (9) |
| CAP initial | 78.0% (6) | 78.6% (8) | **78.9%** (9) |
| CAP feedback 1 | 78.6% (2) | 79.3% (5) | **80.2%** (8) |
| CAP feedback 2 | **78.0%** (0) | 76.5% (4) | 76.8% (5) |

The merge adds 1–3 slides for every controller. Classical still trails
color by 1 slide and 4 taps. These runs shared the machine with other jobs,
so their compute times are not comparable with `sekai-finetune-v1`; the
reports' `job.out` paths name the folder they were first written to
(`results/sekai-slide-merge`).

## Limits

- 16 slides per setup, so slide counts move in small steps.
- One fine-tuning run each, last epoch, no seeds.
- The merge changes every run that uses a model with `active_path`, so
  earlier v7 results were produced by the replace tracker.
