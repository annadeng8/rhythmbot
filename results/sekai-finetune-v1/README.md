# v7 fine-tuned on the dual-arm Sekai camera

Follow-up to [`remaining-study-v1`](../remaining-study-v1/README.md), where
v7 was tried on Yifan's camera with no retraining (zero-shot) and lost to the
color/shape detector. Here v7 is fine-tuned on camera frames from training
charts only, then rerun on the same 11 held-out charts. Run on Anna's Apple
M5 Mac on 2026-11-24.

## What was done

1. **Training frames.** [`scripts/capture_sekai_camera.py`](../../scripts/capture_sekai_camera.py)
   plays each chart with two stylus arms, the classical planner and the color
   detector, using the study camera (960×720). Every 30 Hz vision tick it
   saves the rectified image `concert-torch` feeds the network, robot
   occlusion included, with labels from the chart.
   - Training: Yifan's 48 `train` charts plus 24 dense generated charts
     (`dense_random_chart`, seeds 90000–90023) for density like the song
     excerpts: 29,112 frames.
   - Validation: Yifan's 6 `validation` charts, 1,899 frames.
   - No test chart is used. Frames are in `data/sekai-camera/` (gitignored, 5 GB).
2. **Fine-tuning.** `python -m rhythmbot.train --init results/note_detector_v7.pt
   --train data/sekai-camera/train --val data/sekai-camera/val --model
   results/note_detector_v7_sekai.pt --epochs 12 --lr 3e-4`. Same network and
   loss as v7, starting from v7's weights. The last epoch is kept; there was
   no model selection on test charts. Log: [`finetune.log`](finetune.log).
3. **Evaluation.** [`scripts/eval_sekai_finetune.py`](../../scripts/eval_sekai_finetune.py)
   runs `realtime.run` with the settings of the study's `compare` phase
   (960×720, unpaced, no Meshcat, inference seed 0) for classical and the
   three frozen CAP programs. Each is run with three detectors: color,
   zero-shot v7 and fine-tuned v7. 132 runs, 0 errors.

The BC checkpoints are only on gpu156, which this Mac can't reach, so BC
isn't included.

## Perception on camera validation frames

| | Zero-shot v7 | Fine-tuned v7 |
|---|---:|---:|
| Precision (upcoming notes) | 0.43 | 1.00 |
| Recall (upcoming notes) | 1.00 | 1.00 |
| Recall (note being played) | 0.05 | 0.995 |
| Timing error, mean | 115 ms | 7 ms |
| Slide end lane error | 1.63 lanes | 0.18 lanes |
| Rest-of-slide lane error | 0.41 lanes | 0.02 lanes |

For reference, the original v7 on its own plain screen had a timing error of
16 ms and a slide end-lane error of 0.53 lanes.

## Held-out charts (11 charts, 323 notes)

| Controller | Color | v7 zero-shot | v7 fine-tuned |
|---|---:|---:|---:|
| Classical | **89.2%** | 84.5% | 87.0% |
| CAP initial | 78.0% | 46.1% | **78.6%** |
| CAP feedback 1 | 78.6% | 48.6% | **79.3%** |
| CAP feedback 2 | **78.0%** | 56.0% | 76.5% |

Slides (of 16) and holds (of 16):

| Controller | Color | v7 zero-shot | v7 fine-tuned |
|---|---|---|---|
| Classical | 10 slides, 16 holds | 0, 11 | 7, 16 |
| CAP initial | 6, 16 | 0, 5 | 8, 16 |
| CAP feedback 1 | 2, 16 | 0, 1 | 5, 16 |
| CAP feedback 2 | 0, 16 | 0, 5 | 4, 15 |

Full table, with taps and compute time: [`summary.md`](summary.md).

The color and zero-shot columns are reruns on this Mac. They match the frozen
gpu156 study exactly except CAP feedback 2 with zero-shot v7, which differs by
one note (181 vs 180 of 323).

## Findings

- **Fine-tuning fixes the transfer problem.** Classical recovers from 84.5% to
  87.0%, and CAP from 46–56% back to 77–79%, level with the color detector.
- **It does not beat the color detector overall.** Classical is 2 points
  below color, entirely from slides (7/16 vs 10/16) and 4 taps.
- **CAP slides improve with fine-tuned v7** (8, 5, 4 vs color's 6, 2, 0), but
  CAP's feedback revisions still lose slides.
- **The remaining slide gap is v7's output format, not perception accuracy.**
  The failing slides are zigzags (for example lanes 3→5→3→5→4 in 1.4 s), which
  fail by losing contact for 0.2–0.4 s with no hand handoff. v7 describes the
  rest of a slide with 5 evenly spaced points, which the tracker re-reads every
  frame. Between those points the path is a straight line, so it cuts each
  corner. The color detector samples the visible slide body every 3 pixel
  rows and keeps the corners. The same failure explains v7's 6/16 slides on
  the two gripper arms (see the diagnosis below).

## Why v7 misses slides on two arms (gripper setup B)

[`scripts/diagnose_slide_path.py`](../../scripts/diagnose_slide_path.py)
logs the path the tracker holds after each frame of v7 on
`holds-s71000` (two gripper arms, plain game screen) and compares it with the
true path:

| Look-ahead | Mean lane error | 90th percentile | Max |
|---|---:|---:|---:|
| Now | 0.03 | 0.06 | 0.13 |
| 33 ms | 0.07 | 0.19 | 0.35 |
| 100 ms | 0.14 | 0.39 | 0.58 |
| 200 ms | 0.15 | 0.35 | 0.49 |

The judge allows 0.35 lanes. The stylus is placed correctly at the current
moment, but the path just ahead, which the arm is moving toward, is off by
up to half a lane near corners, and contact breaks there. Of the 10 v7 slide
misses in [`gripper-2arm-final-charts`](../gripper-2arm-final-charts/summary.md),
8 are these zigzags. The other 2 are short straight slides, right after a
zigzag in the speed charts, that were never touched.
- The handoff between hands is not the cause: the `holds` slides stay in one
  hand and fail the same way.
- Turning the rest-of-path reading off entirely is worse: 0/16 slides,
  because the forecast made from further up the screen is less accurate.

Single-arm charts mostly have smoother slides, which is why v7 reached 84%
of slides there.

**Possible fix (not tried):** predict more path points, or keep the
forecast's corners and only correct lanes from the rest-of-path reading.
Either changes v7 or the tracker, and so every vision result.
