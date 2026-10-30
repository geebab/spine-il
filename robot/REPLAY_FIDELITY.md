# Replay fidelity budget — what the arm replay does to a recorded hand motion

`robot/endo_follow.py --mode replay` does not command the recorded motion as it is. This is the budget of every
change it makes **before** the arm's own tracking error, measured offline on the 2026-09-24 recording
`s_20260924T000500_altician_lumbar1_trk_0551` (rec-2.3.0-t0, 98.8 s, endo 6,451 / drill 6,450 frames, all 4-ball,
raw poses on `tExp`). Numbers: tool-tip distance to the recorded raw motion, **mm, RMS / p99 / max**.

```bash
python robot/replay_budget.py <recorder .jsonl> [--now-registration <today's T_scene←cam>] [--from-s S] [--json out.json]
```

No hardware, ~30 s; it runs `endo_follow`'s own `TargetShaper` / `clean_replay` and `recording.sample_pose`, with
a 90 Hz replay loop at a random phase and the recording's own tip model. Holes over the gap limit (none in 0551) are
left out of the statistics — there is no ground truth inside one — together with 1 s after each for the shaper and
clean rows (the catch-up after the jump); the output says how much was left out.

## What gets replayed

- `--replay-run <follow run dir>`: its own tap rows (`load_replay`, unchanged).
- `--replay-run <recorder v2 .jsonl>`: `lerobot_robot_spinearm/recording.py` — rows in `r` order (the header found
  even when its chunk came late), a re-sent `r` dropped, `lost`/`hold` rows out, raw pose (else the filtered one,
  flagged) on `tExp` (else `tSrc`, flagged; an out-of-order stamp drops only itself), H0 = the first valid sample of
  the window (`--replay-from-s`), the tool model's sha256_12 checked against `hdr.toolCalib`. Absolute mode on a
  dynamic registration drops rows without their own `T` (T0 would be wrong by the film's motion).
  `--replay-tool drill` is read but not sent: the right arm's IK / drill tip are not wired.
- Between recorded frames: `--replay-interp linear` (default; position linear, rotation slerp, ends held) or `zoh`.
  Frames more than `--replay-max-gap-s` apart (0.25 s of recording — an occlusion) are not bridged: the last one is
  held until the next one comes, as the old ZOH did.
- Both read before the arm is touched; what was read (source, pose, time, coordinates, tool model, holes, warnings)
  is printed and logged to the run's `events.jsonl` as `{"k": "replay_src", …}` with the replay flags.

## Budget (0551)

| Element | Setting | endo | drill |
|---|---|---|---|
| Sampling between frames, 1× | zoh (before 2026-09-27) | 0.30 / 1.35 / 4.16 | 0.36 / 1.44 / 6.32 |
| | linear | 0.00 / 0.00 / 0.02 | 0.00 / 0.01 / 0.05 |
| Same, 0.2× | zoh → linear | 0.28 / 1.23 / 4.19 → 0.00 / 0.00 / 0.02 | 0.34 / 1.34 / 6.83 → 0.00 / 0.01 / 0.05 |
| Hold-out (every other frame predicted from the rest) | zoh → linear | 0.43 / 1.80 / 4.44 → 0.20 / 0.73 / 1.63 | 0.51 / 1.81 / 7.07 → 0.26 / 0.76 / 1.74 |
| `--replay-clean-hz` | 2 Hz | 0.36 / 1.25 / 4.89 | 0.45 / 1.28 / 6.63 |
| | 4 Hz | 0.24 / 0.75 / 2.43 | 0.27 / 0.72 / 3.02 |
| Shaper `follow_left_smooth` (3 Hz) on the frames | | 1.15 / 4.13 / 4.73 (40 ms lag → 0.80) | 1.31 / 4.17 / 5.97 |
| Shaper `follow_left_best` (8 Hz) on the frames | | 1.17 / 4.31 / 8.30 (−15 ms lag → 1.12) | 1.39 / 4.96 / 16.68 |
| Replay loop 0.2× + smooth preset | zoh → linear | 0.99 / 2.65 / 5.50 → 0.88 / 1.72 / 2.11 | 1.15 / 2.83 / 7.78 → 1.01 / 1.77 / 2.44 |
| Replay loop 1× + smooth preset | zoh → linear | 1.35 / 5.18 / 7.19 → 1.22 / 4.32 / 5.83 | 1.59 / 5.64 / 9.35 → 1.44 / 4.69 / 7.91 |
| Replay loop 0.2× + `follow_left_best` (8 Hz) | zoh → linear | 0.88 / 1.98 / 5.34 → 0.78 / 1.00 / 1.00 | 1.00 / 2.13 / 7.54 → 0.87 / 0.99 / 1.00 |
| Replay loop 1× + `follow_left_best` (8 Hz) | zoh → linear | 0.84 / 1.91 / 6.52 → 0.84 / 2.13 / 5.20 | 1.26 / 2.33 / 21.95 → 1.09 / 2.69 / 16.22 |
| Holes over `--replay-max-gap-s` 0.25 s | held, then the next frame | none (longest 156 ms) | none (longest 156 ms) |
| Relative mapping, camera moved 11.85° / 92.5 mm (09-23 10:24 rig → 0551) | relative (default) | 23.9 / 25.7 / 26.2, rotation ≤ 3.3° | 31.9 / 35.0 / 35.2, rotation ≤ 6.7° |
| | `--replay-registration` | ≈ 0 (the two registrations' own error remains) | ≈ 0 |
| Pose choice | raw vs filtered pos/quat | 0.40 / 1.04 / 2.21 | 1.00 / 3.46 / 13.31 |
| | raw vs the filtered `tip` field (what the render bridge draws) | 0.32 / 0.66 / 1.04 | 1.10 / 3.99 / 13.94 |
| Radius clamp 160 mm (`--radius-mm`) | reach from H0 | max 127.7 → never | max 180.5 → 5,402 of 6,450 frames; from 11.4 s: 126, reach up to 204.3 (p99 194.6) → cut by up to 44 mm |

Reading it:

- **Sampling.** The "sampling" rows compare against the recorded frames interpolated linearly — the audit's
  measure (ZOH 0.31 / 4.1) — so linear reads ~0 by construction there. The hold-out row is the honest comparison
  (twice the frame spacing, ~0.2 mm of raw jitter included). A slower replay does not shrink ZOH steps (same height,
  5× longer), and a fast shaper passes them: at 0.2× the 8 Hz preset gave 5.3 / 7.5 mm max with ZOH, 1.0 mm with
  linear — that 1.0 is the preset's 1 mm deadband.
- **Holes.** 0551 has none over 0.25 s, other sessions do: 09-23 1170 endo 300 ms with a 382 mm tip jump,
  232e endo 2.58 s / 181 mm, 64f7 drill 1.14 s / 272 mm, 59b8 42 holes up to 0.69 s. Linear without the limit would
  draw a straight line through them that the hand never drew (21 mm off the last fix mid-way through a synthetic
  0.68 s occlusion); with it the target holds, then jumps, and the shaper's jump gate and speed caps take the step.
  The loader counts and warns (longest hole, jump).
- **Relative mapping.** The default replays camera-frame displacements from the engage point with today's hand-eye
  `R_bc`, so a camera turned by θ since the recording turns the path by θ (worst case 2·sin(θ/2)·reach: 11.1 /
  15.8 mm at 5° for endo / drill). `--replay-registration` (today's `T_scene←cam`, e.g. a recorder file of today's
  rig) moves the recording through the scene into today's camera (`p_now = T_now⁻¹·T_rec·p_rec`). It fixes the
  direction of the path, **not its start**: the replay still starts where the arm is at engage (`FK0`).
- **Pose.** The replay plays the raw optical fixes; the drill's filtered pose differs by up to 13 mm (gyro fusion,
  still lock), the endoscope's by ~1–2 mm.
- **Drill reach.** The drill left its start by ~160–180 mm for most of 0551: a drill replay needs a window
  (`--replay-from-s`, `--replay-max-s`) or a larger radius.
- The 8 Hz preset's max on the frames is numerically touchy: 8.30 with numpy 2.x, 7.91 with numpy 1.26 (the 09-26
  audit) for the same code — jump-hold / deadband thresholds flip; RMS moves 0.02. The 3 Hz preset is identical.
- Not in this table: the arm's own tracking (camera tip RMS 2.4 mm at 0.2×, 2026-09-26 bench), the track leash,
  joint-rate limit, z-floor / clearance skips, and the `--vis-ki` outer loop.
