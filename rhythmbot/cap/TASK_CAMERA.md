## What your program sees

**Only a camera image.** There is no note list and no chart: your program must
find the notes in the image itself, keep track of them across frames, and
decide what to play. `plan(frame, t_now, X_WT_cmd, width_cmd)` is called
every perception tick (30 Hz, `t_now` in seconds of simulated time) with:

* `frame`: what a fixed RGB camera beside the touch screen sees at `t_now`,
  a numpy `uint8` array of shape (256, 384, 3) (rows, columns, RGB). The
  screen is seen at an angle, in perspective, and the robot arm and gripper
  are in the picture too: when the hand is over the screen it hides
  whatever is under it from the camera. The image has some sensor noise.
* `X_WT_cmd`, `width_cmd`: the fingertip pose and gripper opening your
  previous plan commands at `t_now` (a pydrake `RigidTransform`; use
  `api.current_keyframe(t_now, X_WT_cmd, width_cmd)`).

What a player knows about the screen: notes come into view 1.6 s before
their hit time and scroll at constant speed (on the screen, in metres)
toward the judgment line; the lanes are equal-width strips side by side, in
the same order as world x; each note kind (tap, hold, slide) has its own
look; a hold or slide that is being played has its head stopped on the
judgment line while its body keeps scrolling in. The camera never moves.
Everything else (where the screen, the line and the lanes are in the image,
how image positions map to lanes and to seconds, what each kind looks like,
what the arm looks like) you have to work out.
Your policy object lives for the whole song, so it can keep its own state
(e.g. notes it is tracking) between calls.

## Calibration data

* `examples/` holds camera frames from the training charts as PNG files (you
  can look at them with the Read tool) and `examples/index.md`, which lists,
  for each frame, its time and the notes visible on the screen (kind, lane,
  hit time, duration, slide waypoints). In these frames a hand-written
  controller is playing the chart, so the arm moves as a competent player's
  would.
* `./probe.sh` runs `probe.py` from this directory: it must define
  `main(examples)`, where `examples` is a list of dicts
  `{"frame": array, "t": float, "notes": [...]}` with every camera frame of
  the training charts every 0.1 s and the notes on the screen at that time
  (whether or not the arm hides them), each a dict with `kind`, `lane`,
  `time` (hit time), `time_to_hit`, `duration`, `waypoints`. Whatever it
  prints (up to 20 000 characters) is shown to you. The same import rules
  apply as for `policy.py`. Use it to measure the screen geometry and test
  your perception code against ground truth.

Only training charts are in the calibration data; the program will be scored
on other charts.
