## What your program sees

**Only the screen.** There is no note list and no chart: your program must
find the notes in the image itself, keep track of them across frames, and
decide what to play. `plan(frame, t_now, X_WT_cmd, width_cmd)` is called
every perception tick (30 Hz, `t_now` in seconds of simulated time) with:

* `frame`: the game screen as the player sees it at `t_now`, a numpy
  `uint8` array of shape (384, 256, 3) (rows, columns, RGB);
* `X_WT_cmd`, `width_cmd`: the fingertip pose and gripper opening your
  previous plan commands at `t_now` (a pydrake `RigidTransform`; use
  `api.current_keyframe(t_now, X_WT_cmd, width_cmd)`).

What a player knows about the screen: notes come into view 1.6 s before
their hit time and scroll at constant speed toward the judgment line; the
lanes are equal-width columns side by side, in the same order as world x;
each note kind (tap, hold, slide) has its own look; a hold or slide that is
being played has its head stopped on the judgment line while its body keeps
scrolling in. Everything else (where the line is in pixels, how pixels map
to lanes and to seconds, what each kind looks like) you have to work out.
Your policy object lives for the whole song, so it can keep its own state
(e.g. notes it is tracking) between calls.

## Calibration data

* `examples/` holds frames from the training charts as PNG files (you can
  look at them with the Read tool) and `examples/index.md`, which lists,
  for each frame, its time and the notes visible in it (kind, lane, hit
  time, duration, slide waypoints).
* `./probe.sh` runs `probe.py` from this directory: it must define
  `main(examples)`, where `examples` is a list of dicts
  `{"frame": array, "t": float, "notes": [...]}` with every frame of the
  training charts every 0.1 s and the notes visible in it, each a dict with
  `kind`, `lane`, `time` (hit time), `time_to_hit`, `duration`, `waypoints`.
  Whatever it prints (up to 20 000 characters) is shown to you. The same
  import rules apply as for `policy.py`. Use it to measure the screen
  geometry and test your perception code against ground truth.

Only training charts are in the calibration data; the program will be scored
on other charts.
