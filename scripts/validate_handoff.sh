#!/usr/bin/env bash
# Run from /data/YifanKang/rhythmbot_handoff_20261113 on gpu156.
set -euo pipefail
case "$PWD" in /data/YifanKang/*) ;; *) exit 1;; esac
export TMPDIR="$PWD/tmp" PYTHONPYCACHEPREFIX="$PWD/cache/pycache" PYTHONPATH="$PWD"
export XDG_CACHE_HOME=/data/YifanKang/rhythmbot_sekai_20261109/cache
export HEADLESS=1 CUDA_VISIBLE_DEVICES='' VTK_EGL_DEVICE_INDEX=0 VTK_DEFAULT_EGL_DEVICE_INDEX=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 VTK_SMP_MAX_THREADS=1
study_python=/data/YifanKang/rhythmbot/.venv/bin/python
"$study_python" - <<'PY'
import hashlib,json
from pathlib import Path
r=Path('results/yifan-study-v1');m=json.loads((r/'freeze.json').read_text())
for name in ('bc-mlp','bc-diffusion'):
    assert hashlib.sha256((r/'checkpoints'/(name+'.pt')).read_bytes()).hexdigest()==m['artifacts'][name]['sha256']
PY
# Two complete camera simulations validate the exact rendering/control setup first.
"$study_python" -m rhythmbot.evaluate --preset sekai-camera --split bimanual-smoke \
  --policy classical --workers 2 --out results/handoff-20261113/sekai-camera
"$study_python" - <<'PY'
import json
from pathlib import Path
rs=[json.loads(p.read_text()) for p in Path('results/handoff-20261113/sekai-camera/classical/concert').glob('*.json')]
assert len(rs)==2
assert all('run_error' not in r and 'load_error' not in r and r['vision']['frames']>0 for r in rs)
PY
"$study_python" -m rhythmbot.evaluate --preset sekai-camera --split bimanual-smoke \
  --policy cap:results/yifan-study-v1/cap/v1/policy.py \
    bc:results/yifan-study-v1/checkpoints/bc-mlp.pt bc:results/yifan-study-v1/checkpoints/bc-diffusion.pt \
  --workers 2 --out results/handoff-20261113/sekai-camera
"$study_python" -m rhythmbot.evaluate --preset sekai-screen --split bimanual-smoke \
  --policy classical cap:results/yifan-study-v1/cap/v1/policy.py \
    bc:results/yifan-study-v1/checkpoints/bc-mlp.pt bc:results/yifan-study-v1/checkpoints/bc-diffusion.pt \
  --workers 2 --out results/handoff-20261113/sekai-screen
