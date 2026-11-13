#!/usr/bin/env bash
set -euo pipefail
case "$PWD" in /data/YifanKang/*) ;; *) exit 1;; esac
export TMPDIR="$PWD/tmp" PYTHONPATH="$PWD" PYTHONPYCACHEPREFIX="$PWD/cache/pycache"
export XDG_CACHE_HOME=/data/YifanKang/rhythmbot_sekai_20261109/cache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 VTK_SMP_MAX_THREADS=1
# CUDA visibility does not control EGL graphics. Pin the observed renderer GPU.
export VTK_EGL_DEVICE_INDEX=0 VTK_DEFAULT_EGL_DEVICE_INDEX=0
export CUDA_VISIBLE_DEVICES='' HEADLESS=1
exec /data/YifanKang/rhythmbot/.venv/bin/python -u scripts/camera_vision_study.py "$@"
