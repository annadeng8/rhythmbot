# Commands

Every command runs from the repository root with the course venv one
folder up (`../.venv/bin/python`). Any command takes `--help` for its full
option list. Per-result reproduction commands are also in each
`results/*/README.md`.

## Before you start

`HEADLESS=1` turns Meshcat off: no browser, faster runs. Use it for
evaluations, and make sure it is **unset** for demos, or no Meshcat URL is
printed.

```bash
export HEADLESS=1   # evaluations
unset HEADLESS      # demos
```

Plug the Mac in for anything live or with music: on battery the camera
setup runs slower than real time and falls behind the song.

## Watch the robot play

Open the Meshcat URL the command prints. `--hold` keeps the viewer up after
the run; **Open Controls → Animations** replays it at true speed (silently).
Ctrl-C quits.

**Single arm, v7 vision, the NekoTeto song** (setup A):

```bash
../.venv/bin/python -m rhythmbot.play --chart songs/nekoteto/normal-neko.json --arms 1 \
    --detector torch --model results/note_detector_v7_zigzag.pt --hold
```

**Two gripper arms, v7 vision, with the song** (setup B):

```bash
../.venv/bin/python -m rhythmbot.play --chart songs/nekoteto/normal-neko.json --arms 2 \
    --detector torch --model results/note_detector_v7_zigzag.pt --hold
```

**Yifan's two stylus arms with the camera, live** (setup C, 640×480 camera).
This one waits for the browser before it starts:

```bash
# with the chart's song
../.venv/bin/python -m rhythmbot.play --preset sekai-realtime --chart songs/nekoteto/easy-neko.json \
    --wait-for-browser 30 --hold
# silent
../.venv/bin/python -m rhythmbot.play --preset sekai-realtime --chart songs/nekoteto/easy-neko.json \
    --no-song --wait-for-browser 30 --hold
# a chart without audio, with a soundtrack synthesized from its notes
../.venv/bin/python -m rhythmbot.play --preset sekai-realtime --chart charts/stylus-hard-handoffs.json \
    --audio --wait-for-browser 30 --hold
# the v7 detector fine-tuned on camera frames instead of the color detector
../.venv/bin/python -m rhythmbot.play --preset sekai-realtime --chart charts/stylus-hard-handoffs.json \
    --detector concert-torch --model results/note_detector_v7_sekai.pt --wait-for-browser 30 --hold
```

The live run prints `realtime_pass` and `compute_per_sim` at the end. The
music only stays in sync if `realtime_pass` is true (`compute_per_sim`
below 1.0). For a recording that is always in sync, run with `--no-song`,
screen-record the Meshcat replay, and add `songs/nekoteto/audio.mp3`
starting at 2.42 s in a video editor.

**Yifan's setup exactly as in the frozen study** (960×720 camera, slower
than real time):

```bash
../.venv/bin/python -m rhythmbot.play --preset sekai-camera --hold \
    --chart results/yifan-final-v1/charts/test_iid/crossing-s71000-0.8s-x1.1287.json
```

**Other demos:**

```bash
../.venv/bin/python -m rhythmbot.play --demo --audio --hold              # built-in demo chart
../.venv/bin/python -m rhythmbot.play --random 7 --hold                  # a random chart
../.venv/bin/python -m rhythmbot.play --chart charts/slow-ride-easy.json --tempo 0.6 --hold
```

### `play` options worth knowing

| Option | What it does |
|---|---|
| `--chart FILE` / `--random SEED` / `--demo` | what to play |
| `--arms 1` or `2` | one arm, or two gripper arms (default 2) |
| `--preset sekai-realtime` / `sekai-camera` / `sekai-screen` | Yifan's stylus setup: live, as in the study, or reading the screen directly |
| `--detector` | `oracle` (perfect input), `torch` (v7 on the game image), `color`, `concert` / `concert-torch` (Yifan's setup) |
| `--model FILE` | the network for `torch` / `concert-torch` |
| `--policy` | `classical` (default), `cap:<program.py>`, `bc:<model.pt>` |
| `--audio` / `--song FILE` / `--no-song` | synthesized soundtrack / a given audio file / no audio |
| `--tempo 0.6` | slow the chart and its song together |
| `--duration 40` | only the first 40 s |
| `--no-realtime` | as fast as possible, no pacing |
| `--hold` | keep Meshcat up after the run |
| `--out FILE` | where the JSON report goes |

## Models, controllers and charts

| What | Where |
|---|---|
| v7 detector (plain game image) | `results/note_detector_v7.pt` |
| v7 fine-tuned on zigzag slides (plain game image) | `results/note_detector_v7_zigzag.pt` |
| v7 fine-tuned on camera frames (Yifan's setup) | `results/note_detector_v7_sekai.pt` |
| Single-arm CAP programs | `results/cap/fable-selftest/policy_v1.py`, `results/cap/fable-feedback/policy_v7.py` |
| Two-arm CAP programs | `results/yifan-final-v1/cap/v0/policy.py`, `v1/`, `v2/` |
| Two-arm BC checkpoints | gpu156 only (`results/yifan-final-v1/checkpoints/` there) |
| Demo and random charts | `charts/` |
| Yifan's 11 test charts | `results/yifan-final-v1/charts/test_*/` |
| Songs (not in Git) | `songs/nekoteto/` (with audio), `songs/cityyy-unknown/` (no audio) |

Use a controller in any `play` or `evaluate` command, for example
`--policy cap:results/cap/fable-selftest/policy_v1.py`.

## Evaluations

All headless and parallel; each writes per-run reports and a `summary.md`
in `--out`. `--workers` is how many runs at once (6 is fine on the M5 when
nothing else is running).

**Single arm, held-out charts** (24 charts, 1,810 notes; `eval-v3`,
`eval-v3-cap`):

```bash
HEADLESS=1 ../.venv/bin/python -m rhythmbot.evaluate --policy classical \
    cap:results/cap/fable-selftest/policy_v1.py cap:results/cap/fable-feedback/policy_v7.py \
    --split heldout,heldout-song --detector oracle torch \
    --model results/note_detector_v7_zigzag.pt --workers 6 --out results/my-single-arm
```

Splits: `heldout-iid`, `heldout-dense`, `heldout-real`, `heldout-song`,
`heldout` (the first three), `hard`, `cap-train`, `bc-train`,
`bimanual-test`, `bimanual-smoke`; comma-separate to combine.

**Two gripper arms on Yifan's 11 test charts** (setup B,
`gripper-2arm-final-charts`, `slide-path-v1`):

```bash
HEADLESS=1 ../.venv/bin/python -m rhythmbot.evaluate --policy classical --arms 2 \
    --detector oracle torch --model results/note_detector_v7_zigzag.pt --workers 6 \
    --out results/my-gripper \
    --split "$(ls results/yifan-final-v1/charts/test_*/*.json | sed 's#^results/#study:../#' | paste -sd, -)"
```

**Yifan's camera setup on the 11 test charts, locally** (setup C, classical
and the three CAP programs; color, zero-shot v7 and a fine-tuned model;
`sekai-finetune-v1`). All 132 runs take a few hours; filter with
`--detectors` and `--policies`:

```bash
HEADLESS=1 ../.venv/bin/python scripts/eval_sekai_finetune.py \
    --model results/note_detector_v7_sekai.pt --out results/my-sekai --workers 6 \
    --detectors v7-fine-tuned --policies classical
```

**After an evaluation:**

```bash
../.venv/bin/python -m rhythmbot.evaluate --summarize results/my-single-arm   # rebuild summary.md
../.venv/bin/python -m rhythmbot.failures results/my-single-arm --examples 5  # why notes failed
```

**Yifan's frozen studies** (330-run study, follow-up study, BC training,
CAP generation) run on gpu156. Their commands:
[yifan-final-v1](../results/yifan-final-v1/README.md#reproduce),
[remaining-study-v1](../results/remaining-study-v1/README.md),
[HANDOFF](HANDOFF.md).

## Vision model

**Render plain-screen training frames and train from scratch:**

```bash
../.venv/bin/python -m rhythmbot.dataset --out data/train --charts 80 --dense 80 --fps 10
../.venv/bin/python -m rhythmbot.dataset --out data/val --charts 12 --dense 12 --seed 1000 --fps 10
../.venv/bin/python -m rhythmbot.train --train data/train --val data/val --epochs 30 \
    --model results/my_detector.pt
```

**The zigzag fine-tune** (`note_detector_v7_zigzag.pt`, `slide-path-v1`):

```bash
../.venv/bin/python scripts/render_study_frames.py --split train --mix data/train --out data/study-screen/train
../.venv/bin/python scripts/render_study_frames.py --split validation --mix data/val --out data/study-screen/val
../.venv/bin/python -m rhythmbot.train --init results/note_detector_v7.pt \
    --train data/study-screen/train --val data/study-screen/val \
    --model results/note_detector_v7_zigzag.pt --epochs 12 --lr 3e-4
```

**The camera fine-tune** (`note_detector_v7_sekai.pt`, `sekai-finetune-v1`;
capturing frames plays every chart in the simulator, about an hour):

```bash
../.venv/bin/python scripts/capture_sekai_camera.py --split train --dense 24 --out data/sekai-camera/train --workers 6
../.venv/bin/python scripts/capture_sekai_camera.py --split validation --out data/sekai-camera/val --workers 6
../.venv/bin/python -m rhythmbot.train --init results/note_detector_v7.pt \
    --train data/sekai-camera/train --val data/sekai-camera/val \
    --model results/note_detector_v7_sekai.pt --epochs 12 --lr 3e-4
```

**Score a model on a validation set without training; export plain weights:**

```bash
../.venv/bin/python -m rhythmbot.train --eval-only --model results/note_detector_v7.pt --val data/val
../.venv/bin/python -m rhythmbot.train --export-state --model results/note_detector_v7_zigzag.pt
```

## Diagnostics and timing

```bash
# How far the tracked slide path is from the true path (two gripper arms)
HEADLESS=1 ../.venv/bin/python scripts/diagnose_slide_path.py \
    results/yifan-final-v1/charts/test_iid/holds-s71000-x1.1287.json results/note_detector_v7_zigzag.pt
# Can the live camera mode keep up? (no Meshcat, no sleeps)
../.venv/bin/python -m rhythmbot.realtime --chart charts/stylus-hard-handoffs.json \
    --no-meshcat --unpaced --out results/live-throughput.json
# The same at the study's 960x720 camera
../.venv/bin/python -m rhythmbot.realtime --chart charts/stylus-hard-handoffs.json \
    --camera-width 960 --no-meshcat --unpaced --out results/live-960.json
```

## Songs

Real charts come from osu!mania beatmaps you download yourself (`songs/` is
not in Git). The audio file named in the beatmap is recorded in the chart,
so `play` finds it.

```bash
../.venv/bin/python -m rhythmbot.importers path/to/song.osu --out songs/<name>/
../.venv/bin/python -m rhythmbot.importers path/to/song.osu --out songs/<name>/ --tempo 0.8 --min-gap 0.2
```

## Single-arm CAP and BC

```bash
# Generate a CAP program with a coding agent (writes to /Users/Shared/rhythmbot_cap/<name>)
../.venv/bin/python -m rhythmbot.cap.loop run --name my-cap --self-test --iterations 6
../.venv/bin/python -m rhythmbot.cap.loop heldout /Users/Shared/rhythmbot_cap/my-cap
# Train a single-arm BC policy
../.venv/bin/python -m rhythmbot.bc.train --encoder state --head mlp --detector torch --out results/bc/state_mlp.pt
```

## Pictures

```bash
HEADLESS=1 ../.venv/bin/python -m rhythmbot.meshcat_snapshot --time 4 --out results/meshcat   # open scene.html
HEADLESS=1 ../.venv/bin/python -m rhythmbot.drake_replay --out results/drake-stage
```

## Tests

```bash
../.venv/bin/python -m unittest discover -s tests
```
