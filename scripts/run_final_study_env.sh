#!/usr/bin/env bash
# Source from the isolated study checkout on gpu156.
case "$PWD" in /data/YifanKang/*) ;; *) echo 'Use a /data/YifanKang checkout' >&2; return 1 2>/dev/null || exit 1;; esac
export TMPDIR="$PWD/tmp" PYTHONPYCACHEPREFIX="$PWD/cache/pycache" PYTHONPATH="$PWD"
export XDG_CACHE_HOME=/data/YifanKang/rhythmbot_sekai_20261109/cache
export HF_HOME="$PWD/cache/hf" TORCH_HOME="$PWD/cache/torch" PIP_CACHE_DIR="$PWD/cache/pip"
export UV_CACHE_DIR="$PWD/cache/uv" MPLCONFIGDIR="$PWD/cache/matplotlib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 VTK_SMP_MAX_THREADS=1
export HEADLESS=1 CUDA_VISIBLE_DEVICES=1 VTK_EGL_DEVICE_INDEX=3 VTK_DEFAULT_EGL_DEVICE_INDEX=3
export STUDY_PYTHON=/data/YifanKang/rhythmbot/.venv/bin/python
export STUDY_OUT="$PWD/results/yifan-final-v1"
mkdir -p "$TMPDIR" "$PWD/cache" "$STUDY_OUT"
