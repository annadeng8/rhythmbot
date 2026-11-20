# RhythmBot

Simulated KUKA iiwa arms playing a Project SEKAI–style rhythm game in
[Drake](https://drake.mit.edu): **taps**, **holds** and **slides** on a flat
touch screen, read by a perception stage and played by a controller. We
compare a classical planner, **Code as Policy (CAP)**, where a coding agent
writes the control program and revises it from simulation feedback, and
**behavior cloning (BC)** of the classical planner. 6.4210 final project,
Anna Deng and Yifan Kang.

```
chart.json ──► game screen ──image──► detector ──notes──► tracker + policy ──poses──► DiffIK ──► iiwa + gripper
                                                                                                  │
                                                                         judge ◄── touch sensor ◄─┘ ──► report.json
```

The detector only ever sees pixels and a clock, except in **oracle** runs,
which read the notes straight from the chart as a perfect-perception
reference. The judge scores every note with the game's timing windows and a
strict contact rule: holds and slides need sustained contact, and slides
must follow the path and finish on the right lane.

## The two current setups

| | **Single arm, trained vision** (Anna) | **Two arms, styluses, camera** (Yifan) |
|---|---|---|
| Robot | 1 iiwa + WSG gripper; its two fingertips are two contacts | 2 iiwa; each closed WSG holds one stylus |
| Screen | plain game screen | minimal Sekai screen: lanes, notes, judgment line |
| Perception | trained neural detector (`note_detector_v7.pt`) on the game image | hand-coded color/shape detector on a simulated 960×720 camera image (or on the screen image directly, as an ablation) |
| Controllers | classical, CAP (Claude-written) | classical, CAP (Codex-written), BC (MLP and diffusion, 3 seeds each) |
| Results | [`results/eval-v3`](results/eval-v3/summary.md) (v7 detector), [`results/eval-v2`](results/eval-v2/summary.md) (all controllers, earlier detector) | [`results/yifan-final-v1`](results/yifan-final-v1/summary.md) |
| Speed | 2–5× faster than real time | live mode (640×480 camera): about real time on an M5 Mac; study camera (960×720): about 1.8× slower on gpu156 |

Everything else in `results/` (pilots, older screens, older scoring rules)
is historical; [docs/HISTORY.md](docs/HISTORY.md) has the full account. Do not
mix those numbers with the ones below.

## Results

### Single arm (24 charts, 1,810 notes)

The held-out charts are random, dense and real-song charts the policies
were not built on; the "unseen song" (Cityyy, *Unknown*: easy, intermediate,
normal) was imported after every controller and model was made.

| Controller / input | All notes | Unseen song | Source |
|---|---|---|---|
| Classical / oracle | 99.0% | 99.1% | eval-v3 |
| **Classical / v7 vision** | **98.3%** | **98.4%** | eval-v3 |
| Classical / earlier vision | 97.3% | 98.5% | eval-v2 |
| CAP, feedback only / earlier vision | 85.3% | 81.3% | eval-v2 |
| CAP, self-test / earlier vision | 80.9% | 74.1% | eval-v2 |
| CAP, pixels only (writes its own perception) | 77.7% | 73.2% | eval-v2 |

- **v7 fixed vision slides: 52.6% → 84.2%** (oracle: 91.2%). The earlier
  detector timed slide ends correctly but got the final lane wrong. v7 reads
  the rest of the slide while it is being played, and the tracker replaces
  the remaining path with that reading every tick.
- **CAP's failures are notes it chose not to play**, not notes the robot
  could not reach. The unseen song separates CAP from classical most.
- CAP has not been re-run with v7. Failure breakdowns:
  [eval-v3 failures](results/eval-v3/failures.md),
  [eval-v2 failures](results/eval-v2/failures.md).

### Two arms with styluses (11 held-out charts, 323 notes, 330 runs)

Frozen before testing: 48 training and 6 validation charts, then 11 test
charts (simultaneous holds, crossing slides that pass between hands, chords
with up to three notes for two styluses, faster slides, three 30-second song
excerpts).

| Controller | Oracle | Screen image | **Camera** |
|---|---|---|---|
| Classical | 91.3% | 89.5% | **89.2%** |
| CAP initial | 79.6% | 78.3% | 78.0% |
| CAP after feedback 1 | 82.4% | 78.0% | 78.6% |
| CAP after feedback 2 | 82.4% | 78.0% | 78.0% |
| BC diffusion (mean ± SD, 3 seeds) | 63.4 ± 15.0% | 55.9 ± 15.5% | 56.0 ± 15.9% |
| BC MLP (mean ± SD, 3 seeds) | 62.0 ± 2.3% | 35.9 ± 1.3% | 37.3 ± 1.8% |

- **Camera vision costs classical only ~2 points overall, but slides drop
  from 14/16 to 10/16**, the same slide weakness v7 fixed for one arm.
- **CAP feedback trades slides for taps:** camera slides 6/16 → 2/16 → 0/16
  over two revisions while taps improve, so the total stays flat.
- **BC is weak and seed-sensitive**, especially from images.
- All 330 runs finished with no execution or policy errors. Limits: one CAP
  generation chain, three BC seeds, song excerpts rather than full songs,
  fixed hand assignment, no robot–robot collision checking.

### Same charts, gripper fingertips vs styluses

The 11 test charts above, played by the older two-arm configuration (ordinary
grippers, two fingertips per hand, plain game screen) with the classical
planner, against the stylus study's classical runs. Reports:
[`results/gripper-2arm-final-charts`](results/gripper-2arm-final-charts/summary.md).

| Setup / input | All 323 notes | Taps (291) | Holds (16) | Slides (16) |
|---|---|---|---|---|
| Fingertips / oracle | **99.4%** | 291 | 16 | 14 |
| Fingertips / v7 vision on the game image | 96.9% | 291 | 16 | **6** |
| Styluses / oracle | 91.3% | 265 | 16 | 14 |
| Styluses / color vision on the screen image | 89.5% | 263 | 16 | 10 |
| Styluses / color vision through the camera | 89.2% | 262 | 16 | 10 |

- **Four contacts beat two on taps:** the styluses lose 26 taps even with
  oracle input, on three-note chords and dense song passages that need
  more simultaneous contacts than two. Slides tie with oracle input.
- **v7 does not carry over to two arms:** slides fall to 6/16, below the
  color detector's 10/16, although v7 fixed slides for one arm. These
  slides cross between hands; why v7 fails on them is not yet diagnosed.
- Only 16 slides, so slide counts are small. The fingertip runs use the
  plain game screen, not a camera.

```bash
HEADLESS=1 ../.venv/bin/python -m rhythmbot.evaluate --arms 2 --view screen \
    --detector oracle torch --model results/note_detector_v7.pt --workers 4 \
    --out results/gripper-2arm-final-charts --split "$(python3 -c "import json; print(','.join('study:../yifan-final-v1/' + e['path'] for e in json.load(open('results/yifan-final-v1/protocol.json'))['entries'] if e['split'].startswith('test')))")"
```

## Running

Everything runs from the repository root. The course venv one folder up
has every dependency; in a fresh clone use Python 3.13:

```bash
uv venv --python 3.13 && uv pip install -r requirements.txt
```

Watch a run live: open the Meshcat URL it prints. `--hold` keeps the server
up afterwards, so the recording can be replayed at true speed under
**Open Controls → Animations**. `HEADLESS=1` skips the viewer.

```bash
# Single arm, v7 vision, a real song with its audio (runs in real time)
../.venv/bin/python -m rhythmbot.play --chart songs/nekoteto/normal-neko.json --arms 1 \
    --detector torch --model results/note_detector_v7.pt --hold

# Yifan's two-arm camera setup, live at 1x speed (640x480 control camera)
../.venv/bin/python -m rhythmbot.realtime --chart charts/stylus-hard-handoffs.json \
    --out results/live.json --wait-for-browser 30 --hold

# The same setup exactly as in the frozen study (960x720 camera, slower than real time)
../.venv/bin/python -m rhythmbot.play --preset sekai-camera --hold \
    --chart results/yifan-final-v1/charts/test_iid/crossing-s71000-0.8s-x1.1287.json
```

`rhythmbot.realtime` uses the `sekai-realtime` preset: the study's camera
pose, noise and controller, at 640×480 (`--camera-width 960` for the
study's resolution, `--unpaced` to measure throughput). Yifan measured it at
1× on an Apple M5 Mac plugged in; it is soft real time, with brief stalls,
and on battery in Low Power Mode it runs about 1.5× slower. Its results are
not part of the frozen study, because the camera pixels differ. Timings and
checks: [results/realtime-v1](results/realtime-v1/README.md).

`play` with no setup options runs a third, older configuration: two arms
with ordinary grippers (two fingertips each) and oracle input. Other
options: `--policy cap:<program.py>` or `bc:<model.pt>`,
`--detector oracle|color|torch|concert`, `--audio` for a soundtrack
synthesized from the chart, `--tempo 0.6` to slow a chart and its song.
Song audio only stays in sync when the run keeps up with real time.

**Reproduce the evaluations** (headless, parallel; each writes per-run
reports plus `summary.md`):

```bash
# Single arm, classical, oracle and v7 vision (eval-v3)
HEADLESS=1 ../.venv/bin/python -m rhythmbot.evaluate --policy classical \
    --split heldout,heldout-song --detector oracle torch \
    --model results/note_detector_v7.pt --out results/eval-v3
python -m rhythmbot.failures results/eval-v3

# Two-arm camera setup on Yifan's earlier 8-chart split
HEADLESS=1 ../.venv/bin/python -m rhythmbot.evaluate --preset sekai-camera \
    --split bimanual-test --policy classical --out results/sekai-camera-check
```

Yifan's full 330-run study, its BC checkpoints and CAP generation run on
gpu156; the commands are in
[results/yifan-final-v1/README.md](results/yifan-final-v1/README.md) and
[docs/HANDOFF.md](docs/HANDOFF.md).

**Songs.** Real charts come from osu!mania beatmaps you download yourself;
`songs/` is not in Git because they are copyrighted. NekoTeto has audio;
the Cityyy charts do not.

```bash
python -m rhythmbot.importers path/to/song.osu --out songs/<name>/
```

**Training the vision model** (only needed to retrain):

```bash
python -m rhythmbot.dataset --out data/train --charts 80 --dense 80 --fps 10
python -m rhythmbot.train --epochs 30
```

**Tests:** `../.venv/bin/python -m unittest discover -s tests`

## Code map

| File | Role |
|---|---|
| `rhythmbot/play.py` | builds the Drake diagram, runs one chart, prints and saves the report |
| `rhythmbot/evaluate.py` | runs policies × detectors × charts in parallel; `summary.md` |
| `rhythmbot/final_study.py` | Yifan's frozen two-arm CAP/BC study (prepare, collect, fit, eval) |
| `rhythmbot/profiles.py` | the `sekai-camera` / `sekai-screen` presets |
| `rhythmbot/config.py` | every tunable: lanes, judgment windows, gripper, planner, camera |
| `rhythmbot/chart.py`, `splits.py`, `importers.py` | charts, train/test splits, osu!mania/StepMania/Clone Hero import |
| `rhythmbot/game.py`, `sekai_screen.py` | the plain game screen and the minimal Sekai screen |
| `rhythmbot/perception.py` | detector interface; oracle, old color, and trained (`TorchDetector`) detectors |
| `rhythmbot/camera.py`, `camera_vision.py` | the simulated camera and the color/shape detector for the Sekai screen |
| `rhythmbot/train.py`, `dataset.py` | the vision model, its training data and training |
| `rhythmbot/brain.py`, `planner.py` | note tracking, replanning, and the classical planner |
| `rhythmbot/robot.py`, `judge.py` | arms, grippers, styluses, touch sensing; scoring |
| `rhythmbot/cap/`, `bc/`, `policies.py` | code-as-policy loop, behavior cloning, policy loading |
| `rhythmbot/meshcat_snapshot.py`, `meshcat_screen.py` | presentation renders of the two-arm scene |
| `rhythmbot/realtime.py` | live 1× runner for the two-arm camera setup, with per-tick timing |

## More

- [docs/HISTORY.md](docs/HISTORY.md): everything else, in the order it happened
- [docs/HANDOFF.md](docs/HANDOFF.md): Yifan's two-arm handoff, gpu156 workflow, BC checkpoints
- [output/pdf/rhythmbot-progress.pdf](output/pdf/rhythmbot-progress.pdf): the progress report
