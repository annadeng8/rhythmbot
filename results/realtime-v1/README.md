# Live camera environment — 2026-11-20

The new `python -m rhythmbot.realtime` runner sustains **1× overall playback on
the local Apple M5 Mac** with the classical controller. It remains soft real
time: individual frame stalls occur. None of the recorded paced runs passed
the stricter requirement that accumulated lag always stay below one 33.3 ms
vision frame. The JSON field `realtime_pass` therefore remains **false**;
finishing a chart on time is not a hard deadline guarantee.

## Configuration

- Two Drake iiwa arms with closed WSGs and one stylus each; unchanged robot model.
- 1 ms plant step, 5 ms differential IK, 30 Hz camera/vision, 200 Hz judging.
- Original slide continuity, cross-hand contact, and tail requirements retained.
- The opt-in `sekai-realtime` control camera is **640×480**, with the same pose,
  field of view, and Gaussian noise standard deviation. `sekai-camera` remains
  **960×720** for the frozen study. Different resolution means different input
  pixels; these results must not be substituted into the frozen experiment table.
- Native Meshcat robot materials and the original 4×-resolution minimal Sekai
  texture remain unchanged. The display texture is prepared a few frames ahead;
  the controller independently receives current Drake-composited camera RGB,
  including robot occlusion. It never receives the chart or the observer cache.

## Measurements

All rows below include Meshcat publishing. Startup is excluded, including a
camera/warp warmup at simulation time zero; no gameplay time is skipped.
`Compute/sim` excludes deliberate pacing sleeps. Raw ticks and note judgments
are in `validation/*.json`. The current package source hash is recorded in each.

| Controller / chart | Camera | Sim s | Wall s | Compute/sim | Tick p95 | Max lag | Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical / hard | 960×720, unpaced | 22.750 | 23.918 | 1.051 | 40.3 ms | 1168 ms | 66/72 |
| Classical / hard | 640×480, paced | 22.750 | 22.751 | 0.739 | 32.3 ms | 75 ms | 66/72 |
| Classical / song | 960×720, unpaced | 33.184 | 33.240 | 1.002 | 36.3 ms | 69 ms | 246/306 |
| Classical / song | 640×480, paced | 33.184 | 33.189 | 0.820 | 32.2 ms | 131 ms | 246/306 |
| Codex CAP initial / hard | 640×480, paced | 22.750 | 22.755 | 0.941 | 37.6 ms | 839 ms | 57/72 |

The two classical resolution comparisons have identical aggregate scores and
judgment counts. On the hard chart both complete 5/7 strict slides; slide RMS
changes from 17.6 to 19.3 mm. This is a two-chart regression check, not evidence
that resolution has no effect on every chart or policy. CAP has zero guarded
policy errors, but its larger stalls mean it is not consistently deadline-safe.
BC and diffusion real-time performance have not been validated here.

A separate browser-connected run (`live-browser.json`) had a browser connected
at both start and end: **22.750 simulated seconds / 22.752 wall seconds**, 66/72,
p95 33.3 ms, maximum lag 111 ms. Both native arms and the moving minimal highway
were visually inspected. This measures server-side simulation/publishing,
not browser frame presentation latency. The validation hard runs also had a
browser connection (see their start/end counters); the song runs had none.

On **gpu156**, a single full-resolution, headless 8-second rollout improved
from **26.178 to 14.754 wall seconds (3.27× → 1.84×)**. Both RGB-stream SHA-256
hashes are exactly `2a886d99190a39db66251be749693375573b2f53fea6873d8a8cb2208ab48401`.
The original source baseline is `1a19c10`; the optimized report records its
package source hash. This comparison uses the same 960×720 observations.
The final 640×480 run with Meshcat still takes **12.597 s for 8 simulated s
(1.57×)**: gpu156 is not real time. These short runs are not the old eight-worker
campaign's throughput measurement. Mac Mini SSH could not initialize an OpenGL
context, so no Mac Mini real-time result is claimed.

## Implementation and verification

Fixed geometric rectification reuses sparse bilinear weights, with the original
SciPy interpolation retained at floating-point rounding boundaries. Boolean
shift operations replace repeated rectangular morphology. Bounded workers
prepare identical seeded sensor noise and observer textures. State copies and
width interpolation avoid redundant work. The live runner warms camera shaders,
defers cyclic garbage collection during the measured loop, and records every
vision tick's cost and accumulated lag; it skips no physics or vision ticks.

**56 tests passed** (`tests.txt`), covering RGB/warp/morphology equivalence,
independent state copies, bounded observer preparation, and the existing
camera, tracker, handoff, slide, stylus, evaluation, and study checks. The final
full-resolution RGB hash matches the original rollout. Frozen results and
checkpoints were not modified.

## Reproduce

From the repository root, using the pinned Drake environment:

```sh
python -m rhythmbot.realtime --chart charts/stylus-hard-handoffs.json \
  --out results/live.json --wait-for-browser 30 --hold

# Same camera resolution as the frozen study; measure throughput without sleeps.
python -m rhythmbot.realtime --chart charts/stylus-hard-handoffs.json \
  --camera-width 960 --unpaced --out results/live-960.json

# Matched resolution checks plus the original Codex-generated CAP program.
python scripts/validate_realtime.py --out results/realtime-check

# Optional headless profiling; this does not include Meshcat publishing.
python scripts/profile_realtime.py --preset sekai-realtime --width 960 \
  --chart charts/stylus-hard-handoffs.json --seconds 8 --profile \
  --out results/realtime-profile
```

The measurements used one simulator process, `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, and `VECLIB_MAXIMUM_THREADS=1` on the Mac. gpu156 used
one worker, two noise threads and one texture thread, GPU 3 for EGL,
`VTK_EGL_DEVICE_INDEX=3`, `VTK_DEFAULT_EGL_DEVICE_INDEX=3`, `LP_NUM_THREADS=1`,
and `VTK_SMP_MAX_THREADS=1`. Remote outputs and caches stayed on the required
external/data volumes. Verify current resource availability before reusing any
GPU assignment. Use the original study preset and baseline checkout when
reproducing the frozen experiment, rather than mixing these live-mode results.
