#!/usr/bin/env bash
set -euo pipefail
case "$PWD" in /data/YifanKang/*) ;; *) exit 1;; esac
export TMPDIR="$PWD/tmp" PYTHONPATH="$PWD" PYTHONPYCACHEPREFIX="$PWD/cache/pycache"
export XDG_CACHE_HOME=/data/YifanKang/rhythmbot_sekai_20261109/cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 VTK_SMP_MAX_THREADS=1
export CUDA_VISIBLE_DEVICES='' HEADLESS=1
study_python=/data/YifanKang/rhythmbot/.venv/bin/python
study_out="$PWD/results/yifan-study-v1"
# Fixed first held-out crossing chart, chosen before seeing its test scores.
study_chart="$study_out/charts/test_iid/crossing-s31000-0.6s.json"
for study_label in cap-v2 bc-mlp; do
  if [ "$study_label" = cap-v2 ]; then
    study_policy="cap:$study_out/cap/v2/policy.py"
  else
    study_policy="bc:$study_out/checkpoints/bc-mlp.pt"
  fi
  "$study_python" scripts/replay_study_policy.py --policy "$study_policy" --chart "$study_chart" --out "$study_out/replays/$study_label" --fps 10
  ffmpeg -y -loglevel error -threads 1 -framerate 10 -i "$study_out/replays/$study_label/%06d.png" \
    -filter_complex_threads 1 -filter_complex '[0:v]scale=960:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse' \
    "$study_out/replays/$study_label.gif"
done
