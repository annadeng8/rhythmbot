# Feedback on revision 0

Overall: 60/72 notes completed (83.3%) on 6 charts.

### chords-s11000: 88.9% completed (24/27)

judgments perfect 22 / great 2 / good 0 / miss 3
timing error of hits: mean +16 ms, std 17 ms
- tap: 24/27 completed, 24 hit

Notes not completed (with the notes just before and after, for context):
- #20 tap lane 5 at 7.80 s: no fresh touch in its lane within ±140 ms of the hit time. (prev tap lane 4 at 7.80; next tap lane 1 at 8.50)
- #23 tap lane 5 at 8.50 s: no fresh touch in its lane within ±140 ms of the hit time. (prev tap lane 4 at 8.50; next tap lane 1 at 9.20)
- #26 tap lane 5 at 9.20 s: no fresh touch in its lane within ±140 ms of the hit time. (prev tap lane 4 at 9.20)

### chords-s11001: 85.2% completed (23/27)

judgments perfect 23 / great 0 / good 0 / miss 4
timing error of hits: mean +12 ms, std 8 ms
- tap: 23/27 completed, 23 hit

Notes not completed (with the notes just before and after, for context):
- #15 tap lane 3 at 6.80 s: no fresh touch in its lane within ±140 ms of the hit time. (prev tap lane 2 at 6.80; next tap lane 2 at 7.05)
- #20 tap lane 5 at 7.80 s: no fresh touch in its lane within ±140 ms of the hit time. (prev tap lane 4 at 7.80; next tap lane 1 at 8.50)
- #23 tap lane 5 at 8.50 s: no fresh touch in its lane within ±140 ms of the hit time. (prev tap lane 4 at 8.50; next tap lane 0 at 9.20)
- #25 tap lane 1 at 9.20 s: no fresh touch in its lane within ±140 ms of the hit time. (prev tap lane 0 at 9.20; next tap lane 4 at 9.20)

### crossing-s11000-0.8s: 33.3% completed (1/3)

judgments perfect 1 / great 0 / good 0 / miss 2
timing error of hits: mean +5 ms, std 0 ms
- slide: 1/3 completed, 3 hit, contact 75%, slide RMS 6 mm

Notes not completed (with the notes just before and after, for context):
- #0 slide lane 4.13245 at 3.00 s for 3.20 s: hit +5 ms (miss), contact 66.8%; failure=tail_missing; max gap=725 ms; tail=0/9, slide lateral RMS error 10 mm. (next slide lane 4.13245 at 6.80)
- #1 slide lane 4.13245 at 6.80 s for 1.60 s: hit +5 ms (miss), contact 58.3%; failure=tail_missing; max gap=665 ms; tail=0/9, slide lateral RMS error 6 mm. (prev slide lane 4.13245 at 3.00–6.20; next slide lane 1.04079 at 8.60)

### crossing-s11001-0.5s: 0.0% completed (0/3)

judgments perfect 0 / great 0 / good 0 / miss 3
timing error of hits: mean +7 ms, std 2 ms
- slide: 0/3 completed, 3 hit, contact 71%, slide RMS 14 mm

Notes not completed (with the notes just before and after, for context):
- #0 slide lane 1.12606 at 3.00 s for 2.00 s: hit +5 ms (miss), contact 96.3%; failure=contact_break; max gap=70 ms; tail=9/9, slide lateral RMS error 20 mm. (next slide lane 1.12606 at 5.60)
- #1 slide lane 1.12606 at 5.60 s for 1.00 s: hit +10 ms (miss), contact 55.7%; failure=tail_missing; max gap=435 ms; tail=0/9, slide lateral RMS error 11 mm. (prev slide lane 1.12606 at 3.00–5.00; next slide lane 3.9849 at 6.80)
- #2 slide lane 3.9849 at 6.80 s for 1.00 s: hit +5 ms (miss), contact 61.2%; failure=tail_missing; max gap=385 ms; tail=0/9, slide lateral RMS error 11 mm. (prev slide lane 1.12606 at 5.60–6.60)

### holds-s11000: 100.0% completed (6/6)

judgments perfect 6 / great 0 / good 0 / miss 0
timing error of hits: mean +8 ms, std 3 ms
- hold: 4/4 completed, 4 hit, contact 100%
- slide: 2/2 completed, 2 hit, contact 97%, slide RMS 13 mm

### holds-s11001: 100.0% completed (6/6)

judgments perfect 6 / great 0 / good 0 / miss 0
timing error of hits: mean +7 ms, std 2 ms
- hold: 4/4 completed, 4 hit, contact 100%
- slide: 2/2 completed, 2 hit, contact 97%, slide RMS 13 mm

Per-hand diagnostics:
[
  [
    {
      "plan_calls": 337,
      "plan_errors": 0,
      "errors": [],
      "plan_ms_mean": 0.6893946642735945,
      "plan_ms_max": 1.4157360419631004
    },
    {
      "plan_calls": 337,
      "plan_errors": 0,
      "errors": [],
      "plan_ms_mean": 0.6883294150255377,
      "plan_ms_max": 1.3589398004114628
    }
  ],
  [
    {
      "plan_calls": 319,
      "plan_errors": 0,
      "errors": [],
      "plan_ms_mean": 0.7924961311549976,
      "plan_ms_max": 1.4597470872104168
    },
    {
      "plan_calls": 319,
      "plan_errors": 0,
      "errors": [],
      "plan_ms_mean": 0.7552676148082021,
      "plan_ms_max": 1.4812638983130455
    }
  ],
  [
    {
      "plan_calls": 319,
      "plan_errors": 0,
      "errors": [],
      "plan_ms_mean": 0.8012518245831926,
      "plan_ms_max": 1.4591468498110771
    },
    {
      "plan_calls": 319,
      "plan_errors": 0,
      "errors": [],
      "plan_ms_mean": 0.7603887941536485,
      "plan_ms_max": 1.5030973590910435
    }
  ],
  [
    {
      "plan_calls": 307,
      "plan_errors": 0,
      "errors": [],
      "plan_ms_mean": 0.3924515274283552,
      "plan_ms_max": 0.5776593461632729
    },
    {
      "plan_calls": 307,
      "plan_errors": 0,
      "errors": [],
      "plan_ms_mean": 0.3563471014098741,
      "plan_ms_max": 0.5409508012235165
    }
  ],
  [
    {
      "plan_calls": 265,
      "plan_errors": 0,
      "errors": [],
      "plan_ms_mean": 0.652771244563584,
      "plan_ms_max": 1.1880560778081417
    },
    {
      "plan_calls": 265,
      "plan_errors": 0,
      "errors": [],
      "plan_ms_mean": 0.6116205896690207,
      "plan_ms_max": 1.0873680002987385
    }
  ],
  [
    {
      "plan_calls": 307,
      "plan_errors": 0,
      "errors": [],
      "plan_ms_mean": 0.3927087559932024,
      "plan_ms_max": 0.5768341943621635
    },
    {
      "plan_calls": 307,
      "plan_errors": 0,
      "errors": [],
      "plan_ms_mean": 0.3570228608792302,
      "plan_ms_max": 0.5404469557106495
    }
  ]
]