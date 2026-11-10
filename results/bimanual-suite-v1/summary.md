# Bimanual experiment results

Finished simulations: 61/61; errors: 0; missing: 0.

Fixed strict judge; paired identical charts; seeds vary charts, not stochastic repeats. Color uses calibrated FrameRenderer, NOT the concert art or external camera.

| Family | Cross time | Tool | Detector | Handoff | Charts | Complete | Slides |
|---|---:|---|---|---|---:|---:|---:|
| crossing | 0.3 | stylus | oracle | True | 3 | 6/9 (66.7%) | 6/9 |
| crossing | 0.3 | two_finger | oracle | True | 3 | 6/9 (66.7%) | 6/9 |
| crossing | 0.4 | stylus | oracle | True | 3 | 6/9 (66.7%) | 6/9 |
| crossing | 0.4 | two_finger | oracle | True | 3 | 6/9 (66.7%) | 6/9 |
| crossing | 0.4 | stylus | oracle | False | 3 | 0/9 (0.0%) | 0/9 |
| crossing | 0.5 | stylus | oracle | True | 3 | 9/9 (100.0%) | 9/9 |
| crossing | 0.5 | two_finger | oracle | True | 3 | 9/9 (100.0%) | 9/9 |
| crossing | 0.6 | stylus | oracle | True | 3 | 9/9 (100.0%) | 9/9 |
| crossing | 0.6 | two_finger | oracle | True | 3 | 9/9 (100.0%) | 9/9 |
| crossing | 0.6 | stylus | color | True | 3 | 6/9 (66.7%) | 6/9 |
| crossing | 0.6 | stylus | oracle | False | 3 | 0/9 (0.0%) | 0/9 |
| crossing | 0.8 | stylus | oracle | True | 3 | 9/9 (100.0%) | 9/9 |
| crossing | 0.8 | two_finger | oracle | True | 3 | 9/9 (100.0%) | 9/9 |
| holds | — | stylus | oracle | True | 3 | 18/18 (100.0%) | 6/6 |
| holds | — | two_finger | oracle | True | 3 | 18/18 (100.0%) | 6/6 |
| holds | — | stylus | color | True | 3 | 18/18 (100.0%) | 6/6 |
| chords | — | stylus | oracle | True | 3 | 72/81 (88.9%) | 0/0 |
| chords | — | two_finger | oracle | True | 3 | 81/81 (100.0%) | 0/0 |
| chords | — | stylus | color | True | 3 | 72/81 (88.9%) | 0/0 |
| anchor-demo | — | stylus | oracle | True | 1 | 18/18 (100.0%) | 2/2 |
| anchor-demo | — | two_finger | oracle | True | 1 | 18/18 (100.0%) | 2/2 |
| anchor-stylus-hard-handoffs | — | stylus | oracle | True | 1 | 23/24 (95.8%) | 6/7 |
| anchor-stylus-hard-handoffs | — | two_finger | oracle | True | 1 | 23/24 (95.8%) | 6/7 |

Errors and missing runs are never counted as successful simulations. Three chart seeds are paired procedural variants, not independent repeats or a confidence interval. Full per-note failures and configs are in runs/.
