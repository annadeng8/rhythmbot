# Minimal Sekai display validation

The screen now contains only a dark background, six lane divisions, the
judgment line, and tap/hold/slide notes. Both the camera compositor and the
Meshcat observer use `SekaiScreen`'s minimal renderer. Meshcat retains its
native robot meshes, materials, lighting and grid. Its screen is an embedded,
unlit glTF texture, updated during simulation, with correct Y-up/Z-up
conversion and UV coordinates calibrated to the physical judgment line.
The observer draws the same layout at 4x resolution for cleaner presentation;
the control camera's resolution, geometry, noise and input contract are unchanged.

## Verification

- 43 tests passed, including blank-screen stability (no decorations or feedback),
  RGB detection of all three note types, texture embedding, coordinate conversion,
  judgment-line alignment, and high-resolution coordinate consistency.
- Two full dual-arm, single-stylus classical runs using actual Drake camera RGB:
  crossing 2/3, holds 5/6, total **7/9**. Neither report has a load/run error.
  The detector processed 288 and 318 camera frames, respectively. Per-run RGB
  hashes and source fingerprints are in `minimal-camera/`.
- An actual camera-controlled Meshcat snapshot of `crossing-s31000-0.6s` at
  4 seconds was exported and inspected in Chrome. The screen is flat on the
  slab, both robot arms occlude it, and the old floating geometric notes are absent.
  `snapshot.json` describes this presentation frame, not a completed-chart score.
- The control smoke runs preceded the observer-only glTF axis correction and
  4x presentation rendering. The camera still uses the same scale=1 minimal
  rendering path; the final observer was separately rerun and visually inspected.

This is an integration check, not the pending full policy comparison. The
single-arm results in the progress report are unchanged historical measurements.

## Reproduce

```sh
HEADLESS=1 python -m rhythmbot.evaluate --preset sekai-camera \
  --split bimanual-smoke --policy classical --workers 1 --out results/minimal-camera
HEADLESS=1 python -m rhythmbot.meshcat_snapshot \
  --chart results/yifan-study-v1/charts/test_iid/crossing-s31000-0.6s.json \
  --time 4 --out results/minimal-meshcat
```

Open the self-contained `scene.html` in a browser. The report screenshot was
captured at a 1440x1080 viewport and device scale factor 2. Raw HTML remains on
gpu156 under `/data/YifanKang/rhythmbot_observer_20261117/results/`.
The runs used one process, one CPU thread and GPU 0 for EGL, with a budget of
6 GB host RAM, 2 GB GPU memory and 1 GB output. Disk/RAM/VRAM headroom was checked;
all remote data, cache and temporary paths were under `/data/YifanKang/`.
