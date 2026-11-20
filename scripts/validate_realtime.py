"""Run matched camera resolutions and a Codex CAP smoke test sequentially.

Run from the repository root with the same Python/environment as the live app.
All measurements include native Meshcat publishing; browser latency is separate.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cases = {
        'hard': 'charts/stylus-hard-handoffs.json',
        'song': 'results/yifan-final-v1/charts/test_song/unknown-normal-first30s.json',
    }
    for name, chart in cases.items():
        for width in (960, 640):
            tag = f'{name}-{width}'
            cmd = [sys.executable, '-m', 'rhythmbot.realtime', '--chart', chart,
                   '--out', str(a.out / f'{tag}.json'), '--camera-width', str(width)]
            if width == 960:
                cmd += ['--unpaced']
            with (a.out / f'{tag}.log').open('w') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f'Completed {tag}', flush=True)
    with (a.out / 'cap-hard.log').open('w') as log:
        subprocess.run([sys.executable, '-m', 'rhythmbot.realtime', '--chart', cases['hard'],
            '--out', str(a.out / 'cap-hard.json'), '--policy',
            'cap:results/yifan-final-v1/cap/v0/policy.py'],
            stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Completed Codex CAP', flush=True)


if __name__ == '__main__':
    main()
