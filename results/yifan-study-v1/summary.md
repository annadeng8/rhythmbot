# Yifan: bimanual CAP and BC pilot

Pilot: one BC seed per head and one 3-revision CAP chain; shared classical handoff routing and tool constraints. Color sees calibrated frames, not camera/concert art. Test accessed only after freezing models and programs.

| Phase | Policy | Input | Split | Charts | Complete | Execution / policy errors |
|---|---|---|---|---:|---:|---:|
| feedback | cap-v0 | oracle | train | 6 | 60/72 (83.3%) | 0 / 0 |
| feedback | cap-v1 | oracle | train | 6 | 71/72 (98.6%) | 0 / 0 |
| feedback | cap-v2 | oracle | train | 6 | 66/72 (91.7%) | 0 / 0 |
| test | bc-diffusion | color | test_iid | 6 | 36/72 (50.0%) | 0 / 0 |
| test | bc-diffusion | color | test_speed | 2 | 0/6 (0.0%) | 0 / 0 |
| test | bc-diffusion | oracle | test_iid | 6 | 44/72 (61.1%) | 0 / 0 |
| test | bc-diffusion | oracle | test_speed | 2 | 0/6 (0.0%) | 0 / 0 |
| test | bc-mlp | color | test_iid | 6 | 41/72 (56.9%) | 0 / 0 |
| test | bc-mlp | color | test_speed | 2 | 0/6 (0.0%) | 0 / 0 |
| test | bc-mlp | oracle | test_iid | 6 | 47/72 (65.3%) | 0 / 0 |
| test | bc-mlp | oracle | test_speed | 2 | 2/6 (33.3%) | 0 / 0 |
| test | cap-v0 | color | test_iid | 6 | 63/72 (87.5%) | 0 / 0 |
| test | cap-v0 | color | test_speed | 2 | 0/6 (0.0%) | 0 / 0 |
| test | cap-v0 | oracle | test_iid | 6 | 61/72 (84.7%) | 0 / 0 |
| test | cap-v0 | oracle | test_speed | 2 | 2/6 (33.3%) | 0 / 0 |
| test | cap-v1 | color | test_iid | 6 | 70/72 (97.2%) | 0 / 0 |
| test | cap-v1 | color | test_speed | 2 | 5/6 (83.3%) | 0 / 0 |
| test | cap-v1 | oracle | test_iid | 6 | 72/72 (100.0%) | 0 / 0 |
| test | cap-v1 | oracle | test_speed | 2 | 5/6 (83.3%) | 0 / 0 |
| test | cap-v2 | color | test_iid | 6 | 65/72 (90.3%) | 0 / 0 |
| test | cap-v2 | color | test_speed | 2 | 5/6 (83.3%) | 0 / 0 |
| test | cap-v2 | oracle | test_iid | 6 | 67/72 (93.1%) | 0 / 0 |
| test | cap-v2 | oracle | test_speed | 2 | 5/6 (83.3%) | 0 / 0 |
| test | classical | color | test_iid | 6 | 64/72 (88.9%) | 0 / 0 |
| test | classical | color | test_speed | 2 | 5/6 (83.3%) | 0 / 0 |
| test | classical | oracle | test_iid | 6 | 66/72 (91.7%) | 0 / 0 |
| test | classical | oracle | test_speed | 2 | 4/6 (66.7%) | 0 / 0 |
| validation | bc-diffusion | oracle | validation | 6 | 43/72 (59.7%) | 0 / 0 |
| validation | bc-mlp | oracle | validation | 6 | 47/72 (65.3%) | 0 / 0 |
| validation | classical | oracle | validation | 6 | 66/72 (91.7%) | 0 / 0 |

Completion above is note-weighted; chord charts contain more notes than crossing charts. The table below exposes sustain performance and gives each chart equal weight.

| Policy | Input | Tap | Hold | Slide | Mean chart completion |
|---|---|---:|---:|---:|---:|
| classical | oracle | 48/54 | 8/8 | 14/16 | 88.9% |
| classical | color | 48/54 | 8/8 | 13/16 | 84.7% |
| bc-mlp | oracle | 36/54 | 8/8 | 5/16 | 50.0% |
| bc-mlp | color | 36/54 | 5/8 | 0/16 | 27.1% |
| bc-diffusion | oracle | 36/54 | 8/8 | 0/16 | 33.3% |
| bc-diffusion | color | 36/54 | 0/8 | 0/16 | 16.7% |
| cap-v0 | oracle | 47/54 | 8/8 | 8/16 | 63.4% |
| cap-v0 | color | 47/54 | 8/8 | 8/16 | 63.4% |
| cap-v1 | oracle | 54/54 | 8/8 | 15/16 | 95.8% |
| cap-v1 | color | 54/54 | 8/8 | 13/16 | 87.5% |
| cap-v2 | oracle | 49/54 | 8/8 | 15/16 | 93.5% |
| cap-v2 | color | 49/54 | 8/8 | 13/16 | 85.2% |

Invalid runs are reported separately; no complete-rate claim is made for cells with execution errors. CAP runtime exceptions are counted even when guarded fallback permits the physics run to finish. Full per-note metrics remain in eval/.
