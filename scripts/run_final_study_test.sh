#!/usr/bin/env bash
set -euo pipefail
source scripts/run_final_study_env.sh
# Freeze only after every training and feedback run completes.
if [[ ! -f "$STUDY_OUT/freeze.json" ]]; then
  "$STUDY_PYTHON" -m rhythmbot.final_study freeze --out "$STUDY_OUT"
fi
policies=(classical)
labels=(classical)
for revision in 0 1 2; do
  policies+=("cap:$STUDY_OUT/cap/v$revision/policy.py")
  labels+=("cap-v$revision")
done
for head in mlp diffusion; do
  for seed in 0 1 2; do
    policies+=("bc:$STUDY_OUT/checkpoints/bc-$head-s$seed.pt")
    labels+=("bc-$head-s$seed")
  done
done
# Two disjoint shards, four workers each: eight simulation workers total.
"$STUDY_PYTHON" -m rhythmbot.final_study eval --out "$STUDY_OUT" --split test \
  --policy "${policies[@]:0:5}" --label "${labels[@]:0:5}" --workers 4 \
  > "$STUDY_OUT/test-shard-a.log" 2>&1 &
first=$!
"$STUDY_PYTHON" -m rhythmbot.final_study eval --out "$STUDY_OUT" --split test \
  --policy "${policies[@]:5}" --label "${labels[@]:5}" --workers 4 \
  > "$STUDY_OUT/test-shard-b.log" 2>&1 &
second=$!
status=0
wait "$first" || status=$?
wait "$second" || status=$?
if [[ "$status" != 0 ]]; then exit "$status"; fi
"$STUDY_PYTHON" scripts/summarize_final_study.py "$STUDY_OUT"
