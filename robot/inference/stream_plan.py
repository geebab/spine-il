"""The timetable of the continuous closed loop — the arms keep moving while the next chunk is asked for.

User, 10-12: "최종 목표는, 실제 팔을 움직이는 인퍼런스 작동시, 끊킴 없이 구동되도록 하는 것" — and every chunk must still come from a
fresh look at the scene ("매 청크마다 현재의 환경(내시경 렌더 화면 등)을 받아서 다시 이를 인퍼런스로"). Design and measurements:
inference/CONTINUOUS.md. This file is the part with no robot in it: which answer row becomes the target of which frame.

Time is counted in policy frames. Frame n is due at wall time t0 + n·dt, dt = 1/(fps·speed) (0.2 s at 0.33×).

    observation i   at frame  n0 + i·P                 the page's image and the measured tool state, taken while moving
    its answer      rows r[j] = the tools at frame (n0 + i·P) + 1 + j,  j = 0 … H−1,  absolute
    its window      frames [n0 + i·P + L,  n0 + (i+1)·P + L)            = rows L−1 … L+P−2

P (period) = frames between two looks; L (lead) = frames the answer may take to arrive, be checked by the safety gate and
reach the arms. Windows follow each other without a gap, so the arms never wait — unless an answer is late: then the
whole timetable waits for it (late() = by how many frames; the loop pushes its clock back by that much). The arms hold at
their last target for exactly that long, trigger off, and no frame of the plan is lost. (The first version dropped the
late frames instead: the tools then skipped part of the planned path.)

A target that was sent is never changed: the followers append what they get (endo_follow --replay-stream-udp). Only the
frames of a window that has not been sent yet are open.

At a window's start the new plan is faded in over B frames from the previous plan's own continuation (its rows for the
same frames), and the result is followed from the last sent target with the loop's per-frame limit (safety.limit_step).
Why: every plan is anchored at the tool pose measured at its observation, and that pose wanders about 1 mm around the
command (the arms' visual servo follows the tracker's slow noise) — as much as the tools travel in a whole window at
0.33× (drill median 0.6 mm/s). Without the fade every window would begin with a step of the size of its own motion.

LeRobot's real-time chunking is not used: for a GR00T policy trained with relative actions (ours) lerobot 0.6.1 switches
its own RTC prefix off (modeling_groot._prepare_n1_7_rtc_inputs: "Disabling native GR00T RTC prefix for relative-action
policy").

THE ANCHOR (10-12, after the first continuous robot run R_G24a_cont_C1006_r0). The paragraph above is that run's way
(live_loop.py --stream-anchor measured): the answer's rows stay anchored at the MEASURED pose of their look and are
executed L frames later. On the robot both tools swung for the whole run — commands 4 mm rms around their 5 s mean,
0.21 Hz, 605 of 2018 frames at the per-frame limit, a command path of 2.0 m against 0.2–0.4 m stop-and-go — because that
is an echo: what the arms were measured to be off the command at a look comes back as a command L … L+P frames later,
and the real arms answer a 0.2 Hz command with a gain of 1.2–1.25 (runs/groot_prep/stream/arm_response.py). Stop-and-go
has no such loop (it measures standing arms and executes at once), and the dry runs could not show it: their stand-in
policy answered a fixed trajectory, not "the measured pose + a displacement" as the model does.

The way since (--stream-anchor command, the default):
    belief   the answer's rows with their TIPS re-anchored from the measured pose they were predicted from to the
             COMMAND of that moment (anchor_pose, reanchor): where the plan puts the command timeline. The model still
             gets the measured pose and the image — the loop stays closed through what the policy sees, not through an
             offset added to its targets. The tools' rotations stay anchored at the measured pose (anchor_pose says why).
    stitch   a window continues from the last sent target with the plan's own motion (row L−2 ↔ the last sent target),
             and is brought toward the plan's positions by `gamma` over the window (0.5): plans made L frames apart
             disagree a little (the model samples its answer), and a hand-over that takes all of the new plan at once
             (gamma 1) keeps every such disagreement circulating — the command at a look is what older plans made it.
    start    the first answer is executed from its first row (rows 0 … L+P−2, as a stop-and-go chunk): the arms stood
             while it was awaited, so its rows are as fresh as stop-and-go's. Looks while moving begin after it.
stream_sim.py is this loop in miniature with the arms' measured response; tests/test_stream_plan.py holds both ways to it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from safety import _slerp, limit_step

TOOL_OFFSETS = (0, 7)


@dataclass(frozen=True)
class Timing:
    period: int = 8      # P
    lead: int = 10       # L
    blend: int = 4       # B
    horizon: int = 40    # H: rows of an answer

    def __post_init__(self):
        if self.period < 1 or self.lead < 1 or self.blend < 0:
            raise ValueError(f"period {self.period}, lead {self.lead} must be ≥ 1 and blend {self.blend} ≥ 0")
        if self.lead + self.period - 1 > self.horizon:
            raise ValueError(f"a window needs rows up to {self.lead + self.period - 2}, an answer has {self.horizon}")
        if self.blend > self.period:
            raise ValueError("the fade cannot be longer than a window")

    def obs_frame(self, i: int, n0: int = 0) -> int:
        return n0 + i * self.period

    def window(self, i: int, n0: int = 0) -> tuple[int, int]:
        """[first frame, one past the last) of observation i's window."""
        return n0 + i * self.period + self.lead, n0 + (i + 1) * self.period + self.lead

    def rows(self) -> tuple[int, int]:
        """The slice of an answer its own window takes."""
        return self.lead - 1, self.lead - 1 + self.period

    def fade_rows(self) -> tuple[int, int] | None:
        """The slice of the PREVIOUS answer that covers the first `blend` frames of this window (its continuation), or None
        when that answer is too short for it — then the new plan is taken unfaded and only the step limit smooths."""
        a = self.lead - 1 + self.period
        return (a, a + self.blend) if self.blend and a + self.blend <= self.horizon else None


def mix(a: np.ndarray, b: np.ndarray, w: float, w_rot: float | None = None) -> np.ndarray:
    """a → b by w ∈ [0, 1]: tips linear, tool rotations along the shorter arc (by w_rot when given), everything else
    (trigger, progress) b's."""
    out = np.array(b, float); wr = w if w_rot is None else w_rot
    for o in TOOL_OFFSETS:
        out[o:o + 3] = (1.0 - w) * np.asarray(a[o:o + 3], float) + w * np.asarray(b[o:o + 3], float)
        q = _slerp(np.asarray(a[o + 3:o + 7], float), np.asarray(b[o + 3:o + 7], float), wr)
        out[o + 3:o + 7] = q / np.linalg.norm(q)
    return out


def desired(rows_new: np.ndarray, rows_prev: np.ndarray | None, tm: Timing) -> list[np.ndarray]:
    """The targets a window asks for: the new answer's rows for it, the first `blend` of them faded in from the previous
    answer's rows for the same frames (weight (k+1)/(B+1) on the new plan at frame k)."""
    a, b = tm.rows()
    new = [np.array(r, float) for r in np.asarray(rows_new, float)[a:b]]
    fr = tm.fade_rows()
    if rows_prev is None or fr is None or len(rows_prev) < fr[1]:
        return new
    old = np.asarray(rows_prev, float)[fr[0]:fr[1]]
    for k in range(min(tm.blend, len(new))):
        new[k] = mix(old[k], new[k], (k + 1) / (tm.blend + 1))
    return new


def follow(last_sent: np.ndarray, want: list[np.ndarray], max_mm: float, max_deg: float) -> tuple[list[np.ndarray], int]:
    """The targets actually proposed for the window: each frame at most one step limit away from the one before, starting
    from the last target the arms were sent. Returns (targets, how many frames the limit shortened)."""
    out, hit, v = [], 0, np.array(last_sent, float)
    for w in want:
        v, h = limit_step(v, np.asarray(w, float), max_mm, max_deg)   # the tools' poses clamped, trigger and progress as asked
        out.append(v.copy()); hit += int(h)
    return out, hit


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def shift(v: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """v moved as a would have to move to become b: each tip by (b − a), each tool's rotation by R_b·R_aᵀ (scene frame);
    trigger and progress stay v's. shift(row, measured, command) is a row re-anchored — the same pose the loop's
    chunk-relative form gives (dataset/actions.py: from_delta(command, to_delta(measured, row)))."""
    out = np.array(v, float)
    for o in TOOL_OFFSETS:
        out[o:o + 3] = np.asarray(v[o:o + 3], float) + np.asarray(b[o:o + 3], float) - np.asarray(a[o:o + 3], float)
        qa = np.asarray(a[o + 3:o + 7], float); qb = np.asarray(b[o + 3:o + 7], float); qv = np.asarray(v[o + 3:o + 7], float)
        qa, qb, qv = qa / np.linalg.norm(qa), qb / np.linalg.norm(qb), qv / np.linalg.norm(qv)
        q = _qmul(_qmul(qb, qa * np.array([1.0, -1.0, -1.0, -1.0])), qv)
        out[o + 3:o + 7] = (-q if float(np.dot(q, qv)) < 0 else q) / np.linalg.norm(q)
    return out


def anchor_pose(measured: np.ndarray, command: np.ndarray) -> np.ndarray:
    """What a plan is anchored at (--stream-anchor command): the tips of the COMMAND, the tools' rotations as MEASURED.
    The tips follow their command (gain 1 and more: that is what echoes); a turn the arms make about a tenth of
    (their IK holds the tool's direction), so a rotation carried on from command to command only grows away from the
    tool — in the dry run with the real model the endoscope's commanded turn reached the safety gate's 10° while the tool
    had turned 1.3°, and the gate held 262 frames — and, for the same reason, a rotation cannot echo."""
    out = np.array(measured, float)
    for o in TOOL_OFFSETS:
        out[o:o + 3] = np.asarray(command[o:o + 3], float)
    return out


def reanchor(rows: np.ndarray, fed: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """An answer's rows (absolute, predicted from the state `fed`) as the plan's belief on the command timeline: every row
    moved by what takes `fed` to `anchor` (the command at the time `fed` was measured)."""
    return np.stack([shift(r, fed, anchor) for r in np.asarray(rows, float)])


def first_rows(tm: Timing) -> tuple[int, int]:
    """The slice of the FIRST answer that is executed (--stream-anchor command): from its first row to where the second
    answer's window begins — frames 1 … L+P−1 of the clock."""
    return 0, tm.lead - 1 + tm.period


def stitch(belief: np.ndarray, anchor: np.ndarray, last_sent: np.ndarray, tm: Timing, gamma: float, gamma_rot: float = 1.0) -> list[np.ndarray]:
    """The targets a window asks for (--stream-anchor command): the plan's own motion continued from the last sent target —
    its row for that target's frame (L−2; the anchor itself when L = 1) is put on the target — and brought toward the plan's
    positions by gamma·(j+1)/P at the window's frame j. gamma 0 = the plan's motion only (its positions never count: a
    correction the plan makes in its first rows would never be executed), gamma 1 = at the plan's positions by the
    window's end. The rotations go by gamma_rot: 1 = a window ends at its plan's rotations, as a stop-and-go chunk does
    (they are anchored at the measured pose: anchor_pose). Trigger and progress are the plan's."""
    a, b = tm.rows()
    ref = np.asarray(belief[a - 1], float) if a >= 1 else np.asarray(anchor, float)
    out = []
    for j, k in enumerate(range(a, min(b, len(belief)))):
        own = shift(belief[k], ref, last_sent); f = (j + 1) / tm.period
        out.append(mix(own, np.asarray(belief[k], float), float(np.clip(gamma * f, 0.0, 1.0)), float(np.clip(gamma_rot * f, 0.0, 1.0))))
    return out


@dataclass(frozen=True)
class Clock:
    t0_ms: float          # wall time of frame 0
    dt_ms: float          # one policy frame on the wall clock

    def t(self, n: int) -> float:
        return self.t0_ms + n * self.dt_ms

    def frame(self, t_ms: float) -> int:
        """The last frame whose time has come."""
        return int(math.floor((t_ms - self.t0_ms) / self.dt_ms + 1e-9))


def late(clock: Clock, first_frame: int, n_frames: int, now_ms: float, send_lead_ms: float) -> int:
    """By how many frames a window is late: its first frame's time is less than `send_lead_ms` away, or past (at most
    n_frames). The loop pushes the clock back by this many frames and sends the window whole."""
    k = math.ceil((now_ms + send_lead_ms - clock.t(first_frame)) / clock.dt_ms - 1e-9)
    return int(min(max(k, 0), n_frames))


def simulate(tm: Timing, dt_s: float, delays_s: list[float], send_lead_s: float = 0.15) -> dict:
    """What a list of look → targets-ready delays (look + answer + gate, one per window) does to the timetable: a window
    that is ready too late makes everything after it wait. Returns the frames waited, the longest wait and the wall time
    a frame takes on average (dt when nothing is late)."""
    clock = Clock(0.0, dt_s * 1000.0)
    waited = 0; worst = 0; frames = 0
    for i, d in enumerate(delays_s):
        a, b = tm.window(i)
        ready_ms = clock.t(tm.obs_frame(i)) + d * 1000.0            # the look is taken on the clock as it is then
        k = late(clock, a, 10 ** 6, ready_ms, send_lead_s * 1000.0)
        if k:
            clock = Clock(clock.t0_ms + k * clock.dt_ms, clock.dt_ms)
        waited += k; worst = max(worst, k); frames += b - a
    return {"windows": len(delays_s), "frames": frames, "waited_frames": waited, "longest_wait_s": worst * dt_s,
            "s_per_frame": (frames + waited) * dt_s / frames if frames else dt_s, "budget_s": tm.lead * dt_s - send_lead_s}
