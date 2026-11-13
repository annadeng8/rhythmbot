"""Render a frozen policy with the existing actual-Drake stage renderer.

Example: python scripts/replay_study_policy.py --policy cap:path/policy.py
  --chart chart.json --out frames --fps 10
Remaining arguments are passed to rhythmbot.drake_replay.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from rhythmbot import drake_replay
from rhythmbot.policies import make_policy

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--policy', required=True)
args, rest = parser.parse_known_args()
original_build = drake_replay.build

def build(chart, cfg, *positional, **kwargs):
    if args.policy.startswith('bc:'):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(0)
    kwargs['policy'] = [make_policy(args.policy, cfg, hand=k) for k in (0, 1)]
    return original_build(chart, cfg, *positional, **kwargs)

drake_replay.build = build
sys.argv = [sys.argv[0]] + rest
drake_replay.main()
# Preserve the identity of the rendered controller beside the frame report.
out = Path(rest[rest.index('--out') + 1]) if '--out' in rest else Path('results/drake-stage')
artifact = Path(args.policy.split(':', 1)[1])
(out / 'policy.json').write_text(json.dumps({
    'policy': args.policy,
    'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
    'input': 'oracle',
    'note': 'Camera replay; quantitative benchmark uses the frozen evaluation reports.'
}, indent=2) + '\n')
