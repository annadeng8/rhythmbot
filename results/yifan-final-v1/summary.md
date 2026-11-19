# Dual-arm CAP / BC study

All runs use the fixed minimal Sekai display, two Drake iiwa arms, one stylus per arm, and the strict touchdown/body/tail judge. CAP means **Code as Policy**: Codex writes a trajectory program from the public API and training-only simulation feedback. Initial, feedback 1, and feedback 2 denote successive programs in one generation chain. BC means behavior cloning of full-physics classical demonstrations.

Oracle receives chart-derived observations; screen RGB uses the direct game image; camera RGB uses the Drake camera image with RGB-only color/shape detection. The latter is not a trained neural detector. The router and physical environment are shared.

Training: 48 charts × oracle/camera; validation: 6 separate charts × oracle/camera. MLP and diffusion each train for 60 epochs with three seeds. Each seed is selected by validation loss. All policies are frozen before 11 held-out charts × 3 observation conditions × 10 controllers = 330 test runs. The test includes 6 synthetic IID charts, 2 faster crossing charts, and 3 predeclared 30-second song excerpts. Chord charts include deliberate three-note capacity stress for two styli.

**Main result:** camera classical completion is 89.2%. CAP initial / feedback 1 / feedback 2 achieve 78.0% / 78.6% / 78.0%. The final feedback revision does not improve overall camera completion over the initial program; slide completion falls from 6/16 to 2/16 to 0/16, while all three CAP programs complete every held-out hold. This aggregate score hides a tap/slide tradeoff.

| Controller | Input | Complete notes | Equal-weight chart mean | Policy errors |
|---|---|---:|---:|---:|
| bc-diffusion-s0 | camera | 189/323 (58.5%) | 40.3% | 0 |
| bc-diffusion-s0 | oracle | 229/323 (70.9%) | 46.7% | 0 |
| bc-diffusion-s0 | screen | 192/323 (59.4%) | 40.6% | 0 |
| bc-diffusion-s1 | camera | 228/323 (70.6%) | 51.9% | 0 |
| bc-diffusion-s1 | oracle | 236/323 (73.1%) | 57.1% | 0 |
| bc-diffusion-s1 | screen | 224/323 (69.3%) | 51.0% | 0 |
| bc-diffusion-s2 | camera | 126/323 (39.0%) | 26.4% | 0 |
| bc-diffusion-s2 | oracle | 149/323 (46.1%) | 28.9% | 0 |
| bc-diffusion-s2 | screen | 126/323 (39.0%) | 26.0% | 0 |
| bc-mlp-s0 | camera | 122/323 (37.8%) | 48.2% | 0 |
| bc-mlp-s0 | oracle | 209/323 (64.7%) | 61.8% | 0 |
| bc-mlp-s0 | screen | 119/323 (36.8%) | 45.0% | 0 |
| bc-mlp-s1 | camera | 114/323 (35.3%) | 51.3% | 0 |
| bc-mlp-s1 | oracle | 195/323 (60.4%) | 75.8% | 0 |
| bc-mlp-s1 | screen | 111/323 (34.4%) | 47.5% | 0 |
| bc-mlp-s2 | camera | 125/323 (38.7%) | 48.1% | 0 |
| bc-mlp-s2 | oracle | 197/323 (61.0%) | 60.8% | 0 |
| bc-mlp-s2 | screen | 118/323 (36.5%) | 47.1% | 0 |
| CAP initial | camera | 252/323 (78.0%) | 63.8% | 0 |
| CAP initial | oracle | 257/323 (79.6%) | 76.1% | 0 |
| CAP initial | screen | 253/323 (78.3%) | 64.1% | 0 |
| CAP feedback 1 | camera | 254/323 (78.6%) | 55.8% | 0 |
| CAP feedback 1 | oracle | 266/323 (82.4%) | 86.1% | 0 |
| CAP feedback 1 | screen | 252/323 (78.0%) | 49.8% | 0 |
| CAP feedback 2 | camera | 252/323 (78.0%) | 49.8% | 0 |
| CAP feedback 2 | oracle | 266/323 (82.4%) | 86.1% | 0 |
| CAP feedback 2 | screen | 252/323 (78.0%) | 49.8% | 0 |
| classical | camera | 288/323 (89.2%) | 80.2% | 0 |
| classical | oracle | 295/323 (91.3%) | 90.0% | 0 |
| classical | screen | 289/323 (89.5%) | 80.5% | 0 |

BC variation across three training seeds (sample SD, not a confidence interval):

| Head | Input | Mean ± SD | Min–max |
|---|---|---:|---:|
| mlp | oracle | 62.0% ± 2.3% | 60.4%–64.7% |
| mlp | screen | 35.9% ± 1.3% | 34.4%–36.8% |
| mlp | camera | 37.3% ± 1.8% | 35.3%–38.7% |
| diffusion | oracle | 63.4% ± 15.0% | 46.1%–73.1% |
| diffusion | screen | 55.9% ± 15.5% | 39.0%–69.3% |
| diffusion | camera | 56.0% ± 15.9% | 39.0%–70.6% |

Camera outcomes by note type (BC pools three training seeds):

| Controller | Tap | Hold | Slide |
|---|---:|---:|---:|
| Classical | 262/291 | 16/16 | 10/16 |
| CAP initial | 230/291 | 16/16 | 6/16 |
| CAP feedback 1 | 236/291 | 16/16 | 2/16 |
| CAP feedback 2 | 236/291 | 16/16 | 0/16 |
| BC MLP (3 seeds) | 306/873 | 32/48 | 23/48 |
| BC diffusion (3 seeds) | 506/873 | 34/48 | 3/48 |

Per-family, per-note-type, failure, timing and tracking metrics are in `summary.json`; raw per-note reports are under `eval/`. Timing is conditional on a detected head contact; slide RMS is conditional on available down-contact samples and does not penalize missing contact, so interpret it alongside completion/contact failures.

Limitations: one CAP generation chain; only three BC seeds; short song excerpts rather than full-song evaluation; commanded hand state as the policy input; fixed shared handoff routing rather than learned arm assignment; no validated robot–robot collision avoidance. New results should not be pooled with earlier pilots that used different scoring and charts.
