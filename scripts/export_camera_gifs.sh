#!/usr/bin/env bash
set -euo pipefail
case "$PWD" in /data/YifanKang/*) ;; *) exit 1;; esac
export TMPDIR="$PWD/tmp"
study_out="$PWD/results/camera-vision-v1"
for study_label in cap-v1 bc-mlp; do
  study_frames="$study_out/frames/$study_label/crossing-s31000-0.6s"
  ffmpeg -y -loglevel error -threads 1 -framerate 10 -i "$study_frames/%06d.png" \
    -filter_complex_threads 1 -filter_complex '[0:v]fps=8,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=none' \
    "$study_out/$study_label-camera.gif"
  cp "$study_frames/000039.png" "$study_out/$study_label-camera-preview.png"
done
PYTHONPYCACHEPREFIX="$PWD/cache/pycache" /data/YifanKang/rhythmbot/.venv/bin/python - <<'PY'
import hashlib,json
from pathlib import Path
out=Path('results/camera-vision-v1');items=[]
for label in ['cap-v1','bc-mlp']:
    gif=out/f'{label}-camera.gif'
    report=out/'eval'/label/'camera/crossing-s31000-0.6s.json'
    r=json.loads(report.read_text())
    items.append(dict(policy=label,gif=str(gif),sha256=hashlib.sha256(gif.read_bytes()).hexdigest(),
        bytes=gif.stat().st_size,report=str(report),evaluation_fingerprint=r['fingerprint'],
        completed=sum(n['completed'] is True for n in r['notes']),notes=r['n_notes'],fps=8,
        caption='Recorded during the evaluated camera run; last-observed RGB and rectified pixels. Resized to 720x390 and quantized to 64 colors for GIF delivery; original PNG frames remain on gpu156.'))
(out/'media.json').write_text(json.dumps(items,indent=2)+'\n')
print([(x['policy'],x['bytes']) for x in items])
PY
