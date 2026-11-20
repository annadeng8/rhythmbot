# Dual-arm live launcher verification

The normal `rhythmbot.play` entry point now routes `--preset sekai-realtime`
to the measured live runner. Study presets retain their own camera resolution
and runner. Conflicting single-arm, oracle, and direct-screen overrides fail.
The shared preset interface incorporates the `play.py` changes from Anna's
`anna/readme-preset` branch; its other unmerged changes are not included here.

Verified on the local Apple M5 with native Meshcat publishing, using:

```sh
python -m rhythmbot.play --preset sekai-realtime \
  --chart charts/stylus-hard-handoffs.json --out results/live-2arm.json
```

`hard.json` records actual plant joint ranges: `iiwa_left` moved up to 0.607 rad
and `iiwa_right` up to 0.796 rad. Six of the seven slides received contact ticks
from both arms. The strict judge completed 5/7 slides; aggregate score was 66/72.
The run took 22.752 wall seconds for 22.750 simulated seconds, with a 30.1 ms
p95 tick. Maximum accumulated lag was 364 ms, so the strict one-frame-lag
`realtime_pass` remains false. This establishes 1× overall playback on this Mac,
not hard real time or real-time throughput on gpu156.

Ten focused launcher/optimization tests passed. They cover both live routing
and the unchanged full-resolution study configuration, plus rejection of
single-arm/oracle overrides. The report's package source hash was verified
against the implementation used for this run.
