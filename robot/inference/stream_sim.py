"""The continuous loop in miniature (2026-10-12): the timetable and the hand-over of stream_plan.py, a policy that answers
"the pose I was given + a displacement" as our relative-action models do, and arms that answer a command as the real ones
did. No sockets, no clock, no model — frames. A run of 250 windows takes a second.

    python robot/inference/stream_sim.py            the table of CONTINUOUS.md 8: both anchors, three leads

Why it exists: the first continuous robot run (R_G24a_cont_C1006_r0, --stream-anchor measured) swung both tools for its
whole length and no dry run had shown it — the dry runs' stand-in policy answered a fixed trajectory. Here the policy is
relative and the arm is the measured one, and the swing is there: same period (L+P frames), the per-frame limit reached.
What the loop needs before the robot is asked again is judged here first (tests/test_stream_plan.py).

The arm (Arm): measured tip = the command `delay` ago + a resonant share + slow wander + white noise. From the run's own
record (runs/groot_prep/stream/arm_response.py, commands → measured tips): gain 1.20–1.26 at 0.19–0.21 Hz, 0.98 at 0.4 Hz,
below 0.7 from 0.6 Hz, delay 0.27–0.33 s; standing arms rest 0.9–1.2 mm from their command (run 1, median) and that offset
wanders. What produces the 0.2 Hz gain in the arms' visual servo is not known — the model only repeats it.

The policy (Policy): every answer = the state it was given, moved along a fixed direction by `u` mm a frame, plus
    sample   each answer's own error, growing along the rows (the model samples its answers),
    goal     a correction toward a line it "sees" (the measured tip's distance from it), made in its first rows —
             the part of a plan that only a hand-over toward the plan's POSITIONS executes (stitch's gamma).
"""
from __future__ import annotations

import numpy as np

import stream_plan as SP

DT = 0.2      # s a policy frame at 0.33×
NEUTRAL = np.array([40.0, -20.0, 1030.0, 0.6, -0.6, -0.45, 0.28, 30.0, -25.0, 1020.0, 0.62, 0.64, 0.36, 0.3, 0.0])


class Arm:
    def __init__(self, delay_s: float = 0.27, bump: float = 0.3, f0: float = 0.2, q: float = 1.2, wander_mm: float = 1.0,
                 wander_tau_s: float = 8.0, white_mm: float = 0.15, seed: int = 0):
        self.d = delay_s / DT; self.bump = bump; self.rng = np.random.default_rng(seed)
        w0 = 2 * np.pi * f0 * DT; al = np.sin(w0) / (2 * q)          # a band-pass around f0 (peak gain 1), at the frame rate
        self.b = np.array([al, 0.0, -al]) / (1 + al); self.a = np.array([1.0, -2 * np.cos(w0), 1 - al]) / (1 + al)
        self.x = np.zeros((3, 6)); self.y = np.zeros((3, 6)); self.w = np.zeros(6)
        self.aw = float(np.exp(-DT / wander_tau_s)); self.wander = wander_mm; self.white = white_mm

    def measured(self, cmds: list) -> np.ndarray:
        """The pose measured now, given every command up to now (the last one is this frame's)."""
        n = len(cmds) - 1; k = max(n - self.d, 0.0); i = int(np.floor(k)); f = k - i
        late = SP.mix(cmds[i], cmds[min(i + 1, n)], f)                                   # the command `delay` ago
        tips = np.concatenate([late[0:3], late[7:10]])
        if not self.x.any():                                                             # the filter starts at rest on the pose
            self.x[:] = tips
        self.x = np.roll(self.x, 1, axis=0); self.x[0] = tips
        self.y = np.roll(self.y, 1, axis=0)
        self.y[0] = self.b[0] * self.x[0] + self.b[1] * self.x[1] + self.b[2] * self.x[2] - self.a[1] * self.y[1] - self.a[2] * self.y[2]
        self.w = self.aw * self.w + np.sqrt(1 - self.aw ** 2) * self.wander / np.sqrt(3) * self.rng.normal(size=6)
        tips = tips + self.bump * self.y[0] + self.w + self.white / np.sqrt(3) * self.rng.normal(size=6)
        out = late.copy(); out[0:3] = tips[:3]; out[7:10] = tips[3:]
        return out


class Policy:
    def __init__(self, u_mm: float = 0.12, sample_mm: float = 0.0, goal_gain: float = 0.0, goal_rows: int = 5, horizon: int = 40, seed: int = 1,
                 line_moves: tuple[int, float] | None = None):
        self.u, self.sample, self.g, self.gr, self.H = u_mm, sample_mm, goal_gain, goal_rows, horizon
        self.rng = np.random.default_rng(seed)
        self.line_y = {0: NEUTRAL[1], 7: NEUTRAL[8]}          # the line each tip should move along (its y)
        self.line_moves, self.n = line_moves, 0               # (after this many answers, by this many mm): something new in the image

    def answer(self, state: np.ndarray) -> np.ndarray:
        self.n += 1
        if self.line_moves is not None and self.n == self.line_moves[0] + 1:
            self.line_y = {o: y + self.line_moves[1] for o, y in self.line_y.items()}
        rows = np.repeat(np.asarray(state, float)[None, :], self.H, axis=0)
        for o in SP.TOOL_OFFSETS:
            e = self.rng.normal(size=3); e = self.sample * e / np.linalg.norm(e)
            k = np.arange(1, self.H + 1)
            rows[:, o] += self.u * k                                                       # the motion: along x
            rows[:, o + 1] += self.g * (self.line_y[o] - state[o + 1]) * np.minimum(1.0, k / self.gr)   # back to the line, early rows
            rows[:, o:o + 3] += e[None, :] * (k / self.H)[:, None]
        return rows


def run(anchor: str = "command", gamma: float = 0.5, lead: int = 16, period: int = 8, blend: int = 4, windows: int = 250,
        arm: Arm | None = None, policy: Policy | None = None, max_mm: float = 2.0, max_deg: float = 3.0, start_off_mm: float = 0.0) -> dict:
    """One run of `windows` looks. anchor: "measured" (the run of 10-12: rows as answered, faded in over `blend` frames) or
    "command" (re-anchored, stitched with `gamma`, the first answer from its first row). Returns what the two robot runs
    were compared by: the command's path against the policy's, the frames at the per-frame limit, the swing around the 5 s
    mean (command and measured), the correlation of the command's steps L+P frames apart."""
    arm = arm or Arm(); pol = policy or Policy()
    tm = SP.Timing(period, lead, blend, pol.H); P, L = tm.period, tm.lead
    N = windows * P + L + P
    c0 = NEUTRAL.copy(); c0[8] += start_off_mm                    # the drill starts this far off its line
    cmds = [c0.copy()]; meas = [arm.measured(cmds)]
    starts: dict = {}; sent: dict = {}; prev_rows = None; clamped = 0
    # the look at frame 0, arms standing
    rows = pol.answer(meas[0])
    if anchor == "measured":
        starts[L] = ("rows", SP.desired(rows, None, tm)); prev_rows = rows
    else:
        a, b = SP.first_rows(tm)
        starts[1] = ("rows", [r for r in SP.reanchor(rows, meas[0], SP.anchor_pose(meas[0], cmds[0]))[a:b]])
    for n in range(1, N):
        if n in starts:                                           # a window (or the first chunk) begins: from the last sent target
            kind, *what = starts.pop(n)
            want = what[0] if kind == "rows" else SP.stitch(what[0], what[1], cmds[n - 1], tm, gamma)
            tg, hit = SP.follow(cmds[n - 1], want, max_mm, max_deg); clamped += hit
            for k, t in enumerate(tg):
                sent[n + k] = t
        cmds.append(sent.pop(n) if n in sent else cmds[-1].copy())      # nothing sent: the arms hold
        meas.append(arm.measured(cmds))
        if n % P == 0 and n // P < windows:                       # a look while moving
            rows = pol.answer(meas[n])
            if anchor == "measured":
                starts[n + L] = ("rows", SP.desired(rows, prev_rows, tm)); prev_rows = rows
            else:
                anc = SP.anchor_pose(meas[n], cmds[n])
                starts[n + L] = ("stitch", SP.reanchor(rows, meas[n], anc), anc)
    C = np.asarray(cmds)[L + P:windows * P]; M = np.asarray(meas)[L + P:windows * P]
    out = {"anchor": anchor, "gamma": gamma if anchor == "command" else None, "lead": L, "frames": len(C), "clamped_frames": int(clamped)}
    for tool, o in (("endo", 0), ("drill", 7)):
        c, z = C[:, o:o + 3], M[:, o:o + 3]
        st = np.linalg.norm(np.diff(c, axis=0), axis=1)

        def swing(x, w=25):
            k = np.ones(w) / w
            sm = np.stack([np.convolve(np.pad(x[:, j], (w // 2, w - 1 - w // 2), mode="edge"), k, mode="valid") for j in range(3)], 1)
            return float(np.sqrt((np.linalg.norm(x - sm, axis=1) ** 2).mean()))
        s = np.diff(c, axis=0); s = s - s.mean(axis=0); lag = L + P      # the steps without the steady motion
        ac = float((s[:-lag] * s[lag:]).sum() / max(np.sqrt((s[:-lag] ** 2).sum() * (s[lag:] ** 2).sum()), 1e-12))
        out[tool] = {"command_path_mm": round(float(st.sum()), 1), "policy_path_mm": round(pol.u * len(c), 1), "net_mm": round(float(np.linalg.norm(c[-1] - c[0])), 1),
                     "step_mm_median": round(float(np.median(st)), 3), "step_mm_p90": round(float(np.percentile(st, 90)), 3),
                     "swing_command_mm": round(swing(c), 2), "swing_measured_mm": round(swing(z), 2), "step_corr_L+P": round(ac, 2),
                     "off_line_mm_end": round(float(np.mean(M[-50:, o + 1]) - pol.line_y[o]), 2), "off_line_rms": round(float(np.sqrt(np.mean((M[len(M) // 2:, o + 1] - pol.line_y[o]) ** 2))), 2)}
    return out


def main() -> None:
    import json
    print("anchor     gamma lead | command path / policy path (mm) | frames at the limit | swing command / measured (mm rms) | step correlation L+P frames apart   [drill]")
    for anchor, gamma, lead in (("measured", None, 16), ("measured", None, 10), ("command", 1.0, 16), ("command", 0.5, 16), ("command", 0.5, 10), ("command", 0.5, 14), ("command", 0.0, 16)):
        r = run(anchor, gamma if gamma is not None else 0.5, lead, policy=Policy(sample_mm=0.5)); d = r["drill"]
        print(f"{anchor:9s} {str(gamma):>5s} {lead:4d} | {d['command_path_mm']:8.0f} / {d['policy_path_mm']:.0f} | {r['clamped_frames']:5d} of {r['frames']} | "
              f"{d['swing_command_mm']:.2f} / {d['swing_measured_mm']:.2f} | {d['step_corr_L+P']:+.2f}")
    print(json.dumps(run("command", 0.5, 16, policy=Policy(sample_mm=0.5, goal_gain=1.0), start_off_mm=3.0)["drill"]))


if __name__ == "__main__":
    main()
