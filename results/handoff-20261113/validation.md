# Unified handoff validation

2 charts x 4 methods x 2 presets: entry-point validation, not a complete benchmark

16 real Drake runs verified against source `2be0043046f7c195f799d03bc60406e33e8fe073a6635ee9f51d94dc63a40fa5` and judge `447e724fdb8120bc`. No execution/load/policy errors.

| Preset | Policy | Chart | Complete |
|---|---|---|---:|
| sekai-camera | classical | crossing-s31000-0.6s | 2/3 |
| sekai-camera | classical | holds-s31000 | 5/6 |
| sekai-camera | cap-v1 | crossing-s31000-0.6s | 2/3 |
| sekai-camera | cap-v1 | holds-s31000 | 5/6 |
| sekai-camera | bc-mlp | crossing-s31000-0.6s | 0/3 |
| sekai-camera | bc-mlp | holds-s31000 | 2/6 |
| sekai-camera | bc-diffusion | crossing-s31000-0.6s | 0/3 |
| sekai-camera | bc-diffusion | holds-s31000 | 0/6 |
| sekai-screen | classical | crossing-s31000-0.6s | 2/3 |
| sekai-screen | classical | holds-s31000 | 5/6 |
| sekai-screen | cap-v1 | crossing-s31000-0.6s | 2/3 |
| sekai-screen | cap-v1 | holds-s31000 | 5/6 |
| sekai-screen | bc-mlp | crossing-s31000-0.6s | 0/3 |
| sekai-screen | bc-mlp | holds-s31000 | 2/6 |
| sekai-screen | bc-diffusion | crossing-s31000-0.6s | 0/3 |
| sekai-screen | bc-diffusion | holds-s31000 | 0/6 |

These runs use Anna's stricter touch-landing judge. They do not replace the historical 64-run table or establish a new full-suite completion rate. Reports from the remote source snapshot may have no git commit field; the full source hash and base revision above identify what was tested.
