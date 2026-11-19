"""Rasterize exported Meshcat HTML with headless Chrome and assemble a GIF."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import tempfile
from PIL import Image

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('frames',type=Path);ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--chrome',default='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
a=ap.parse_args();files=sorted(a.frames.glob('*.html'));assert files

def render(path):
    png=path.with_suffix('.png').resolve()
    if png.exists():return png
    with tempfile.TemporaryDirectory(prefix='rhythmbot-chrome-') as profile:
        proc=subprocess.run([a.chrome,'--headless','--no-first-run','--hide-scrollbars',
            '--window-size=1280,900','--force-device-scale-factor=1.5','--virtual-time-budget=1500',
            '--user-data-dir='+profile,'--screenshot='+str(png),path.resolve().as_uri()],
            capture_output=True,text=True,timeout=60)
        if proc.returncode or not png.exists():raise RuntimeError(proc.stderr[-3000:])
    return png

with ThreadPoolExecutor(max_workers=2) as pool:
    images=[]
    for i,path in enumerate(pool.map(render,files)):
        with Image.open(path) as im:images.append(im.convert('RGB').resize((960,675),Image.Resampling.LANCZOS))
        if i%16==0:print(i,len(files),flush=True)
report=json.loads((a.frames/'replay.json').read_text())
times=report['times']
durations=[max(10,10*(round(b*100)-round(t*100))) for t,b in zip(times,times[1:])]
durations.append(round(100/report['fps'])*10)
a.out.parent.mkdir(parents=True,exist_ok=True)
images[0].save(a.out,save_all=True,append_images=images[1:],duration=durations,loop=0,optimize=False)
images[len(images)//2].save(a.out.with_suffix('.png'))
a.out.with_suffix('.json').write_text(json.dumps(dict(source_report=report,frame_durations_ms=durations,frames=len(images)),indent=2)+'\n')
print(a.out)
