#!/usr/bin/env bash
# Run on gpu156 from /data/YifanKang/rhythmbot_yifan_study_v1.
set -euo pipefail
case "$PWD" in /data/YifanKang/*) ;; *) echo 'Run inside /data/YifanKang' >&2; exit 1;; esac
export TMPDIR="$PWD/tmp"
export XDG_CACHE_HOME=/data/YifanKang/rhythmbot_sekai_20261109/cache
export PYTHONPYCACHEPREFIX="$PWD/cache/pycache"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' HEADLESS=1
study_python=/data/YifanKang/rhythmbot/.venv/bin/python
study_out="$PWD/results/yifan-study-v1"
"$study_python" scripts/freeze_yifan_study.py "$study_out"
"$study_python" -u -m rhythmbot.yifan_study eval --out "$study_out" --split test --detector oracle color \
 --policy classical "bc:$study_out/checkpoints/bc-mlp.pt" "bc:$study_out/checkpoints/bc-diffusion.pt" \
 "cap:$study_out/cap/v0/policy.py" "cap:$study_out/cap/v1/policy.py" "cap:$study_out/cap/v2/policy.py" \
 --label classical bc-mlp bc-diffusion cap-v0 cap-v1 cap-v2 --workers 2
"$study_python" scripts/verify_yifan_study.py "$study_out"
"$study_python" scripts/summarize_yifan_study.py "$study_out"
