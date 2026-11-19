# Fixed-environment dual-arm study

This study completes the current Code as Policy (CAP) feedback and behavior-cloning (BC) comparison in the minimal Sekai / dual-iiwa / one-stylus-per-arm environment. The generated programs use **Codex CLI**. The default underlying model identifier is not exposed by the CLI JSON interface and is not claimed here. A failed Claude quota attempt is retained separately; none of its output is used as a candidate.

The protocol was declared before test evaluation. `protocol.json` specifies chart hashes and separate training, validation, IID test, faster-slide test and song-excerpt test sets. `freeze.json` records all policy/checkpoint identities before any held-out run. `summary.md`, `summary.json`, and `overview.png` report results; `verification.json` checks coverage and provenance.

## Reproduce

The study code, demonstrations, checkpoints and intermediate replay frames live on **gpu156** under:

```
/data/YifanKang/rhythmbot_final_20261118/
```

All simulator code is under the repo's `rhythmbot/`; use the fixed source hash in `freeze.json`, not an evolving later checkout. All outputs and caches on gpu156 must stay under `/data/YifanKang/`. Resource budget and preflight/warmup logs are included. The exact campaign uses GPU 1 for eager PyTorch training and GPU 3 for EGL graphics, at most six simulator workers during collection, then eight evaluation workers plus at most two replay workers, and one BLAS thread per worker. Check the GPUs are free before reusing those indices. No custom Triton kernel or `torch.compile` is used.

From an isolated `/data/YifanKang/` checkout:

```bash
source scripts/run_final_study_env.sh
"$STUDY_PYTHON" -m unittest discover -s tests
"$STUDY_PYTHON" -m rhythmbot.final_study prepare --out "$STUDY_OUT"
"$STUDY_PYTHON" -m rhythmbot.final_study warmup --out "$STUDY_OUT" --workers 2
"$STUDY_PYTHON" -m rhythmbot.final_study collect --out "$STUDY_OUT" --workers 4
for head in mlp diffusion; do
  for seed in 0 1 2; do
    "$STUDY_PYTHON" -m rhythmbot.final_study fit --out "$STUDY_OUT" --head "$head" --seed "$seed"
  done
done
```

CAP generation runs in a fresh isolated directory with no project context; shell, app, multi-agent and web tools are disabled, and any tool activity observed in the event log causes rejection. Only the public API contract, prior program and training-only feedback are supplied. Full prompts, responses and audit logs are under `cap/`. Generate an initial candidate with:

```bash
python3 scripts/generate_stylus_cap.py --out results/yifan-final-v1/cap/v0
```

Copy that candidate to the study checkout and run training feedback:

```bash
"$STUDY_PYTHON" -m rhythmbot.final_study eval --out "$STUDY_OUT" --split feedback \
  --condition oracle camera --policy "cap:$STUDY_OUT/cap/v0/policy.py" --label v0 --workers 2
"$STUDY_PYTHON" -m rhythmbot.final_study feedback --out "$STUDY_OUT" --revision v0
```

For each of two revisions, provide only the previous candidate and its feedback to `generate_stylus_cap.py --previous ... --feedback ...`, then repeat feedback evaluation. Reuse the committed candidates for exact replay; regenerating with a coding model is not deterministic. After all six BC fits, both learned-policy deployment warmups (`scripts/warmup_final_policies.py`), and all three CAP feedback rounds:

```bash
bash scripts/run_final_study_test.sh
"$STUDY_PYTHON" scripts/verify_final_study.py "$STUDY_OUT"
"$STUDY_PYTHON" scripts/plot_final_study.py "$STUDY_OUT"
```

BC checkpoints and raw arrays are excluded from Git; fetch the files under `results/yifan-final-v1/checkpoints/` from the remote directory above and verify SHA-256 against `freeze.json`. Per-epoch optimizer/RNG snapshots enable interrupted training to resume, while completed per-chart runs are reused only after identity/checksum checks.

## Interpretation boundaries

- **Oracle:** chart-derived notes, a labelled control reference.
- **Screen RGB:** direct minimal Sekai image through the RGB detector.
- **Camera RGB:** the calibrated Drake camera image, including robot occlusion, through the same color/shape detector. Control does not receive note state, depth, masks or labels. The camera compositor itself uses geometry internally to draw the display.
- Camera perception here is color/shape segmentation, not Anna's trained detector. Her unmerged `anna/detector-v7` branch was not incorporated mid-study.
- All conditions share the physical robot, strict head/body/tail judge and handoff router. Arm assignment is not learned. Physical robot–robot collision avoidance is not validated.
- The song experiment uses predeclared 30-second excerpts, not full songs. One CAP feedback chain and three BC training seeds do not establish broad statistical significance.
- CAP initial / feedback 1 / feedback 2 mean successive Codex programs. MLP and diffusion are two BC action heads. The report keeps these names explicit rather than unexplained version numbers.

## Audit notes

An initial warmup used unjittered synthetic charts. Before collection, deterministic tempo jitter was added because seed separation alone can repeat hold layouts. The final protocol verifies unique chart contents across splits. The two earlier warmup runs and superseded chart draft are retained under `preflight/`, and are excluded from training, evaluation and summary counts.

Training controls include paired taps, simultaneous holds and crossing slides; chord charts also contain deliberate three-note capacity stress for two styli. `summary.json` separates tap performance by simultaneous-note count as well as note type and chart family.

## Verified delivery

All 330 held-out runs and 36 CAP training-feedback runs completed with zero execution errors and zero CAP runtime exceptions. All 45 unit tests passed. The remote verification checked all six checkpoint files and all three unedited Codex programs against their hashes; it also checked exact chart/condition coverage, identical physical configurations, and tool-free generation logs. These checks establish run validity, not successful note completion.

The three native Meshcat GIFs under `replays/` cover the same complete crossing chart with Drake camera RGB control: classical 2/3, final CAP 0/3, diffusion seed 0 0/3. Contact counts, tail counts, final judgments and timing match their frozen reports (`replay-verification.json`). The CAP/BC choices were recorded before testing; the classical reference was added on the same chart after viewing the failures. A BC slide with 99.3% maintenance still fails because only 7/8 tail samples are covered. No failed clip was replaced with an easier one.

The reusable resource budget is **32 GiB host RAM**: late-run persistent BC workers retained about 21.4 GiB together, exceeding the initial 16 GiB estimate. They exited normally and freed their memory. GPU use and disk use stayed small; `resource-budget.txt` records the correction. Future larger studies should recycle workers in bounded batches. The GIFs and summary are committed to Git; the larger arrays, weights and raw frame exports remain on gpu156.
