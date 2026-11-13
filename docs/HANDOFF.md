# Handoff: Yifan's dual-arm Sekai camera environment

## What to run

The intended configuration is **two iiwa arms + closed WSG single-tip styluses +
Sekai-inspired concert screen + camera RGB vision**. Use `rhythmbot.evaluate`,
not the historical study scripts, for new evaluations. This handoff builds on
Anna's `5c644a7` and retains her stricter touch-landing judge, timeout handling,
failure reports and provenance.

| Entry | Arms / contacts | Display | Perception input | Purpose |
|---|---|---|---|---|
| `--preset sekai-camera` | 2 / 1 per arm | concert | Drake scene RGB, calibrated color/shape vision | Target experiment |
| `--preset sekai-screen` | 2 / 1 per arm | identical concert | direct display RGB, same detector | Camera ablation |
| no preset, legacy defaults | 1 / 2 | simple calibrated display | oracle, unless overridden | Older baseline; **not** target |
| Anna's `results/eval-v2` | 1 / 2 | simple calibrated display | oracle or trained `torch` detector | Separate single-arm comparison |

Preset options are enforced: explicitly requesting one arm, oracle input or the
wrong view raises an error instead of silently changing the experiment. Custom
CAP/BC policies are instantiated independently for each arm. The camera preset
uses the prior camera study's exact camera geometry, 960×720 resolution and noise
standard deviation 3/255. No retraining or score-window relaxation is included.

The dynamic game display is composited into Drake's scene RGB using render labels
inside the renderer, preserving robot occlusion. The detector sees neither labels,
depth nor chart state. It receives RGB and time, rectifies with known calibration,
and extracts note estimates. This is not a native animated Drake texture or a
physical camera. The current vision is classical color/shape processing, not a
learned visual encoder. The judge and game renderer retain ground truth normally.

## Smoke check: no external weights required

From the repository root with the Drake dependencies installed:

```sh
HEADLESS=1 python -m rhythmbot.evaluate --preset sekai-camera \
  --split bimanual-smoke \
  --policy classical cap:results/yifan-study-v1/cap/v1/policy.py \
  --workers 2 --out results/sekai-camera-smoke
```

This runs **four simulations**: two methods × two committed charts (a repeated
cross-hand slide chart and a simultaneous-hold chart). No `songs/` download is
needed. `bimanual-test` selects the original eight charts, also committed. The
chart files are unchanged from `results/yifan-study-v1/protocol.json`.

For the direct-RGB comparison, rerun the command with `--preset sekai-screen` and
`--out results/sekai-screen-smoke`. Never reuse the camera output directory.
Preset evaluations reject incompatible environment/source manifests and refuse
to overwrite existing run reports. Choose a fresh output directory for a rerun.

## BC weights: not in Git

Use the exact trained single-stylus checkpoints, not the old single-gripper BC
models. Source of truth: `results/yifan-study-v1/freeze.json`, entries `bc-mlp` and
`bc-diffusion`. They live on gpu156 at:

```
/data/YifanKang/rhythmbot_yifan_study_v1/results/yifan-study-v1/checkpoints/bc-mlp.pt
/data/YifanKang/rhythmbot_yifan_study_v1/results/yifan-study-v1/checkpoints/bc-diffusion.pt
```

To run from another machine with authorized SSH access:

```sh
mkdir -p results/yifan-study-v1/checkpoints
scp gpu156:/data/YifanKang/rhythmbot_yifan_study_v1/results/yifan-study-v1/checkpoints/bc-mlp.pt results/yifan-study-v1/checkpoints/
scp gpu156:/data/YifanKang/rhythmbot_yifan_study_v1/results/yifan-study-v1/checkpoints/bc-diffusion.pt results/yifan-study-v1/checkpoints/
python - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('results/yifan-study-v1')
manifest = json.loads((root / 'freeze.json').read_text())
for name in ('bc-mlp', 'bc-diffusion'):
    p = root / 'checkpoints' / (name + '.pt')
    assert hashlib.sha256(p.read_bytes()).hexdigest() == manifest['artifacts'][name]['sha256'], name
print('Both checkpoint hashes match the original training artifacts')
PY
```

If gpu156 is inaccessible, the classical/CAP command still works. The BC experiment
is not complete until the checkpoints are available; do not silently omit BC or
report missing-weight runs as successful.

## Full paired benchmark under the current judge

After retrieving/checking the weights, these are **two separate 32-run suites**:

```sh
for preset in sekai-screen sekai-camera; do
  HEADLESS=1 python -m rhythmbot.evaluate --preset "$preset" \
    --split bimanual-test --workers 2 \
    --policy classical cap:results/yifan-study-v1/cap/v1/policy.py \
      bc:results/yifan-study-v1/checkpoints/bc-mlp.pt \
      bc:results/yifan-study-v1/checkpoints/bc-diffusion.pt \
    --out "results/current-$preset"
done
```

Expected report metadata: `arms=2`, `contacts_per_arm=1`, `single_tip=true`,
`display=concert`, the correct `view` and preset, camera parameters, source and
judge hashes, chart hash, and policy hash. CAP reports carry two per-hand policy
statistics. `vision.chart_access`, `vision.label_input` and `vision.depth_input`
are false; frame counts must be nonzero. A clean report must have no `run_error`
or `load_error`; a low note-completion score is a different, legitimate outcome.
Summaries count per-hand policy errors and flag mixed environments/judge source.

## gpu156 launch discipline

All workspaces, weights, temporary files, caches and outputs belong under
`/data/YifanKang/`; never use `/mnt/data` or the system disk. Before a long suite,
inspect mounts/free disk, RAM/swap, load and GPU availability. Example environment
from a workspace beneath `/data/YifanKang/`:

```sh
mkdir -p tmp cache
export TMPDIR="$PWD/tmp" PYTHONPYCACHEPREFIX="$PWD/cache/pycache"
export XDG_CACHE_HOME=/data/YifanKang/rhythmbot_sekai_20261109/cache
export HEADLESS=1 CUDA_VISIBLE_DEVICES=''
export VTK_EGL_DEVICE_INDEX=0 VTK_DEFAULT_EGL_DEVICE_INDEX=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LP_NUM_THREADS=1 VTK_SMP_MAX_THREADS=1
```

CUDA visibility does not disable EGL rendering: the example explicitly selects
GPU 0 for graphics. Check that it has headroom. The validated two-worker budget
is two CPU cores, 6 GB host RAM, 1 GB graphics VRAM and 1 GB disk. Inference is CPU;
there are no custom Triton kernels. Use the existing Drake environment at
`/data/YifanKang/rhythmbot/.venv/bin/python` or install requirements in a venv
under `/data/YifanKang/`. Run the smoke check before the full suite.

## Results and remaining work

- `results/camera-vision-v1/`: **historical** 64-run camera/direct comparison and
  scored-run GIFs, measured before `5c644a7`. The former 84.6% camera score is not
  a current-judge claim.
- `results/eval-v2/`: Anna's single-arm evaluation; its percentages are not a
  measurement of this bimanual camera environment.
- `results/handoff-20261113/`: new unified-entry-point validation against Anna's
  latest judge. Read the [validation note](../results/handoff-20261113/validation.md) and
  [test log](../results/handoff-20261113/tests.log) for exactly what was run.

The new 64-run benchmark above remains to be run before publishing a complete
current-judge comparison. The handoff smoke results must not be substituted for
it. Historical scripts `camera_vision_study.py` and `yifan_study.py` retain their
old frozen artifacts; running them against changed code is not a reproduction of
the original experiments. Use the original commits to reproduce historical data.

Existing camera GIFs are clearly labeled historical; this handoff does not claim
that they show the new judge. 
