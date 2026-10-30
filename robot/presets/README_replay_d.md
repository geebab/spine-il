# replay_{left,right}_d.args — E11 successor "d" (2026-09-29)

Pair-start search round 2 (8 rev.4 recordings × ≥2 runs, real bone IoU): d = vis-ki 1.0 + deadband 0.5 mm,
mean IoU 0.609 (a 0.547, e 0.558, f 0.574; d − f 95% CI +0.003..+0.068), robot/hand volume 0.97.
Includes refined robot tip models and the 09-28 hand tip/shaft. Append the recording's paired `--start=...`
on the command line (argparse: last wins). argparse @files take one argument per line, no comments.

# replay_{left,right}_j.args — final setting for robot-origin sessions (2026-09-29)

d with vis-ki 0.5 and deadband 1.5 mm, at 0.33×. Judged on robot-origin sessions (the FE's live recorded cut vs the hand's
recorded cut, 8 rev.4 recordings): IoU 0.70, 1-voxel tolerance overlap 0.976, ASSD 0.32 mm, volume 1.25× the hand
(d: 0.59 / 0.930 / 0.51 mm / 1.59×). Deadband 1.0 / 2.0 and 0.5× were within run noise; 1× removed the overcut (0.94×)
but lost IoU (0.61). The sim-based ranking (d first) did not hold on live cuts — the FE drill gate depends on its tick.
Final numbers over 8 recordings × 2 runs: IoU 0.683, volume 1.28×, 1-voxel tolerance 0.968, ASSD 0.34 mm (d: 0.587 / 1.59×).
j is the default of `eval/robot_origin_run.sh` (`RO_PRESET`). With j the drill tip sits a median 1.5 mm off
the planned path laterally (measured 2026-10-02; the deadband is 1.5 mm): harmless on deep demo cuts (1.07× volume), dominant
on shallow policy cuts (4.4×).
