#!/usr/bin/env python3
"""Hand endoscope → arm follower that logs every timestamp on the way (latency & accuracy measurement).

Both endoscope constellations (hand-held, arm-carried) are seen by the same tracker camera; the tracker's
`--tap` stream gives each one's pose with its camera frame number and mid-exposure time. The arm server
stamps goal receipt / position read / bus write. Everything lands in `<run>/events.jsonl` for
`robot/latency_report.py`.

Mapping (relative, 1:1): on engage the hand tip/orientation H0 and the arm's FK tip pose FK0 are latched;
then the tip target in the base frame = FK0 moved by the hand's tip displacement and rotation since H0,
rotated into the base frame with the hand-eye R_base←cam. Only the hand-eye rotation matters (no jump on
engage). IK runs on a frame at the endoscope tip (hand-eye X), position first.

  follow  hold the hand endoscope still ~1 s → the arm follows; hide the hand marker → the arm stops
          (re-engage = hold still again; it re-latches, like a clutch). Ends after --duration or Ctrl+C.
  steps   no hand: the tip target steps ±A mm along camera x, y, z, then sine 0.5 / 1 Hz along x.

  python robot/endo_follow.py --mode steps
  python robot/endo_follow.py --mode follow --duration 60 --step-deg 6
"""

from __future__ import annotations

import argparse
import json
import signal
import socket
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from lerobot_robot_spinearm import handeye as he
from lerobot_robot_spinearm import recording as rec
from lerobot_robot_spinearm.client import ArmRpc
from lerobot_robot_spinearm.ee import URDF, Z_FLOOR_M, Clearance, MarkerKinematics, TipIK
from lerobot_robot_spinearm.protocol import DEFAULT_PORTS, JOINTS
from lerobot_robot_spinearm.tap import (ENDO_TOOLS, FrameAssembler, StateLog, TapReceiver, mono_offset_ms,
                                        pick_robot, pose_of, wall_ms)

TOOL_CALIB = Path.home() / "il-robot/tracking/tool_calib.json"
# which tracker constellations can ride each arm: the left carries an endoscope block (named endo / endo2 by screen
# side), the right the drill block (the tracker's --model2); the right arm's default tip is the hand drill's
ROBOT_TOOLS = {"left": ENDO_TOOLS, "right": ("drill",)}
TOOL_CALIB_ARM = {"left": TOOL_CALIB,
                  "right": Path.home() / "il-robot/tracking/tool_calib_drill.json"}


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 fromfile_prefix_chars="@")  # @robot/presets/follow_left_best.args
    ap.add_argument("--mode", choices=["follow", "steps", "replay"], default="follow")
    ap.add_argument("--replay-run", default=None, help="replay: a follow run dir whose hand motion (raw) is replayed")
    rec.add_replay_args(ap)  # --replay-tool / -interp / -max-gap-s / -registration / -from-s (recorder files)
    ap.add_argument("--replay-max-s", type=float, default=None, help="replay only the first N seconds")
    ap.add_argument("--ik-axis", choices=["none", "drill", "endo"], default="none",
                    help="tip IK: instead of the soft full orientation, align only this tool's shaft axis (the rig's "
                         "shaftLocal) — tip exact + axis exact on a 5-DOF arm; the roll is free")
    ap.add_argument("--axis-weight", type=float, default=1.0, help="--ik-axis: soft weight of the axis task")
    ap.add_argument("--ik-lock", default="", help="tip IK: hold these joints near the engage posture harder, e.g. "
                                                  "'gripper=10' (the jaw carries the marker block; its weak servo wobbles)")
    ap.add_argument("--align-ori-s", type=float, default=0.0,
                    help="replay: turn the arm's marker onto the recorded hand's absolute orientation over this many "
                         "seconds after engage (0 = relative only: the engage mismatch, ~11 deg, stays for the whole run)")
    ap.add_argument("--replay-advance-s", type=float, default=0.0,
                    help="replay: sample the recording this much ahead (arm-time s) to cancel the arm's own lag; pair "
                         "with --lead-ms 0")
    ap.add_argument("--replay-speed", type=float, default=0.2,
                    help="replay: play the recorded hand motion at this speed (default 0.2 = five times slower, same "
                         "path — the project default for any motion the robot reproduces, 2026-09-26: 1x 6.6 mm, "
                         "0.5x 3.9 mm, 0.2x 2.4 mm tip RMS; pass 1 for real time). --replay-max-s stays in recorded s")
    ap.add_argument("--replay-clean-hz", type=float, default=None,
                    help="replay: clean the recorded hand motion offline first — drop 3-ball fixes, then a zero-phase "
                         "(forward-backward) 2nd-order low-pass at this cutoff on position and orientation. What a "
                         "recording→dataset converter can do with no added delay")
    ap.add_argument("--ff-model", default=None,
                    help="learned feed-forward (robot/ff_fit.py): tip correction = W·[v, a, x_abs, 1] from the shaper's "
                         "state — the general part of the ILC corrections, usable live")
    ap.add_argument("--ff-max-mm", type=float, default=15.0)
    ap.add_argument("--ilc-file", default=None,
                    help="replay: iterative-learning feed-forward (npz: t s, corr m camera) added to the shaped target; "
                         "robot/ilc_update.py builds the next one from a run")
    ap.add_argument("--kf-q", type=float, default=2000.0, help="kalman: white-acceleration density (mm²/s³)")
    ap.add_argument("--kf-r", type=float, default=0.15, help="kalman: measurement noise (mm)")
    ap.add_argument("--filter", choices=["none", "clean", "shape", "kalman"], default="none",
                    help="hand target: clean = hold the rotation of 3-ball poses + drop tip jumps; "
                         "shape = clean + critically damped 2nd-order filter with speed/acceleration limits")
    ap.add_argument("--shape-hz", type=float, default=6.0, help="shape: filter natural frequency")
    ap.add_argument("--shape-vmax", type=float, default=400.0, help="shape: max tip speed mm/s")
    ap.add_argument("--shape-amax", type=float, default=4000.0, help="shape: max tip acceleration mm/s²")
    ap.add_argument("--frame-jump-mm", type=float, default=8.0, help="clean: a tip step larger than this in one frame is held")
    ap.add_argument("--deadband-mm", type=float, default=0.0,
                    help="shape: the target only moves once the hand leaves this radius (tremor ignored), then trails it")
    ap.add_argument("--deadband-deg", type=float, default=0.0, help="shape: the same for the rotation")
    ap.add_argument("--lead-ms", type=float, default=0.0,
                    help="shape: command the planned target this far ahead (x + v·lead, v = the shaper's own smooth "
                         "velocity): feed-forward for the arm's lag without amplifying encoder noise like overdrive")
    ap.add_argument("--smith-model", default=None,
                    help="joint FOPDT model JSON (a, b, c, delay_ticks per joint at the server tick): overdrive feedback "
                         "on a Smith-predictor estimate of the undelayed joint angle instead of the late encoder")
    ap.add_argument("--smith-k", type=float, default=2.0, help="Smith loop gain (sent = r + (K−1)(r − ŷ))")
    ap.add_argument("--smith-k-joint", default="", help="per-joint overrides, e.g. 'elbow_flex=3'")
    ap.add_argument("--smith-e-hz", type=float, default=0.0,
                    help="filtered Smith predictor (Normey-Rico & Camacho): low-pass the measured-minus-model correction "
                         "so encoder noise / mismatch reach the feedback only at low frequency (0 = unfiltered)")
    ap.add_argument("--mpc-lambda", type=float, default=0.0,
                    help="per-joint MPC / preview control on the Smith model (replaces the Smith push): minimise "
                         "Σ(y − r)² + λ·Σ(Δu)² over --mpc-horizon ticks, r extrapolated with the goal's velocity (0 = off)")
    ap.add_argument("--mpc-horizon", type=int, default=10)
    ap.add_argument("--dob-hz", type=float, default=0.0,
                    help="disturbance observer (needs --smith-model's a, b, d): input disturbance per joint estimated from "
                         "every tick and low-passed at this cutoff, then cancelled — acts while moving, unlike the gated "
                         "sag integrator (0 = off)")
    ap.add_argument("--dob-max-deg", type=float, default=8.0)
    ap.add_argument("--vis-ki", type=float, default=0.0,
                    help="camera outer loop: integrate (target − observed arm tip) (1/s) while the target is slow; "
                         "removes the steady bias left by sag / FK / hand-eye (0 = off)")
    ap.add_argument("--vis-gate-mms", type=float, default=30.0, help="integrate only while the target moves slower")
    ap.add_argument("--vis-max-mm", type=float, default=15.0, help="clamp of that correction")
    ap.add_argument("--vis-delay-ms", type=float, default=0.0,
                    help="compare the arm tip with the target this long ago (the arm's lag), not the current one")
    ap.add_argument("--vis-dead-mm", type=float, default=0.0,
                    help="camera outer loop: leave tip errors below this alone (the arm moves in stick-slip steps of about "
                         "a millimetre; chasing less than that keeps it hunting while the target is still)")
    ap.add_argument("--vis-settle-deg", type=float, default=0.0,
                    help="camera outer loop: integrate only while every joint is within this of its goal — a joint that is "
                         "still short of its goal (stuck) is the sag integrator's job, and integrating the same error here "
                         "as well winds the correction up, so the tip overshoots when the joint lets go (0 = always)")
    ap.add_argument("--z-hz", type=float, default=None, help="shape: natural frequency on the base vertical (gravity) axis")
    ap.add_argument("--z-deadband-mm", type=float, default=None, help="shape: deadband on the base vertical axis")
    ap.add_argument("--sag-joints", default="shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper",
                    help="joints the sag integrator acts on (gravity loads only lift, elbow, wrist_flex)")
    ap.add_argument("--sag-dead-deg", type=float, default=0.0,
                    help="ignore goal − encoder errors below this (gear backlash / encoder steps: stops hunting)")
    ap.add_argument("--arm", default="left", choices=["left", "right"])
    ap.add_argument("--transport", choices=("zmq", "ros2"), default="zmq",
                    help="arm commands over ZMQ (direct) or ROS 2 (/<arm>/joint_command via spinearm_bridge; needs the ROS env)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--cmd-port", type=int, default=None, help="default: the arm's standard port")
    ap.add_argument("--pub-port", type=int, default=None)
    ap.add_argument("--tap-port", type=int, default=9880)
    ap.add_argument("--tool-calib", default=None, help="tip model (default: the endoscope for the left arm, the drill "
                                                       "for the right)")
    ap.add_argument("--robot-tool-calib", default=None,
                    help="tip model of the arm's own marker block (the tracker's --model/--model2 file, e.g. "
                         "~/il-robot/calib/tracker_models/tool_calib_drill_robotR_tip.json): the IK tip, the arm "
                         "marker's tip. The recorded hand keeps --tool-calib. Default: --tool-calib for both")
    ap.add_argument("--handeye", default=None, help="default ~/il-robot/calib/handeye_<arm>.json")
    ap.add_argument("--duration", type=float, default=60.0, help="s (follow); steps runs its schedule")
    ap.add_argument("--step-deg", type=float, default=10.0,
                    help="server per-tick cap = how far past the measured angle the servo is pushed (gravity sag)")
    ap.add_argument("--max-joint-dps", type=float, default=90.0,
                    help="client goal rate limit, all joints scaled together (the tip keeps its path)")
    ap.add_argument("--min-clearance-mm", type=float, default=85.0,
                    help="jaw / jaw tip / gripper motor ≥ this from motor 2 (folded rest pose ≈ 74 mm, where it hit)")
    ap.add_argument("--limit-margin-deg", type=float, default=3.0, help="IK keeps this far inside the joint limits")
    ap.add_argument("--start", default=None,
                    help="move here first (arm angles), e.g. 'shoulder_lift=-60,elbow_flex=50,wrist_flex=70,gripper=-40'")
    ap.add_argument("--ik", choices=["tip", "frame"], default="tip",
                    help="tip: position hard + orientation soft + posture (TipIK); frame: MarkerKinematics soft 6-D")
    ap.add_argument("--ori-weight", type=float, default=0.1, help="orientation weight (tip: vs posture; frame: 1/rad vs 1/m)")
    ap.add_argument("--post-weight", type=float, default=0.01, help="tip IK: pull toward the engage posture")
    ap.add_argument("--pos-only", action="store_true", help="ignore the hand's rotation (tip position only)")
    ap.add_argument("--ik-iters", type=int, default=3)
    ap.add_argument("--vmax-dps", type=float, default=240.0, help="IK per-solve joint speed cap (solves at 90 Hz)")
    ap.add_argument("--source", choices=["filt", "raw"], default="filt", help="hand pose: tracker output or raw")
    ap.add_argument("--radius-mm", type=float, default=160.0, help="target stays within this of the engage point")
    ap.add_argument("--jump-mm", type=float, default=20.0, help="ignore a hand frame that jumps more than this")
    ap.add_argument("--lost-ms", type=float, default=100.0, help="stop sending after the hand is lost this long")
    ap.add_argument("--engage-still-s", type=float, default=1.0)
    ap.add_argument("--engage-still-mm", type=float, default=3.0)
    ap.add_argument("--gate-mm", type=float, default=60.0, help="arm marker within this of its reference (see pick_robot)")
    ap.add_argument("--margin-mm", type=float, default=50.0, help="… and the other marker at least this much farther")
    ap.add_argument("--ambiguous-ms", type=float, default=150.0, help="disengage only if identity is unclear this long")
    ap.add_argument("--stale-ms", type=float, default=150.0, help="ignore frames older than this (capture → now)")
    ap.add_argument("--seed", choices=["measured", "previous"], default="measured",
                    help="IK start: the arm's measured angles (closed loop, as EETeleop: each goal is a straight "
                         "Cartesian step from where the arm is, so lagging heavy joints are waited for) or the last solution")
    ap.add_argument("--track-leash-mm", type=float, default=20.0,
                    help="keep the tip target within this of the measured tip (EETeleop track leash; 0 = off)")
    ap.add_argument("--accel", type=int, default=None,
                    help="servo Acceleration register for this run (server default 32; restored on exit)")
    ap.add_argument("--sag-ki", type=float, default=0.0,
                    help="joint integral gain (1/s) on goal − encoder: cancels the servos' steady gravity sag (0 = off)")
    ap.add_argument("--sag-max-deg", type=float, default=8.0, help="clamp of that correction per joint")
    ap.add_argument("--overdrive", type=float, default=1.0,
                    help="send encoder + K·(goal − encoder): K > 1 pushes lagging joints harder (the servos are soft P)")
    ap.add_argument("--overdrive-max-deg", type=float, default=10.0, help="clamp of the overdrive extra per joint")
    ap.add_argument("--overdrive-lp-hz", type=float, default=0.0,
                    help="low-pass the overdrive extra (first order): boost the slow motion, not the 2–4 Hz ringing band")
    ap.add_argument("--overdrive-joint", default="",
                    help="per-joint K overrides, e.g. 'shoulder_lift=1.35,elbow_flex=1.35,wrist_flex=1.35' — the gravity "
                         "joints ring at 2–4 Hz under a strong overdrive (2026-09-23 replay spectra)")
    ap.add_argument("--overdrive-lead-s", type=float, default=0.0,
                    help="push against the encoder predicted this far ahead (+ its age) from its velocity: damping")
    ap.add_argument("--sag-gate-dps", type=float, default=30.0, help="integrate only while the goal moves slower")
    ap.add_argument("--max-lead-deg", type=float, default=35.0, help="disengage if a goal leads the encoders by more")
    ap.add_argument("--amp-mm", type=float, default=15.0, help="steps mode amplitude")
    ap.add_argument("--out", default=None, help="run dir (default ~/il-robot/runs/latency/<ts>_<mode>)")
    ap.add_argument("--trig-udp", default="",
                    help="replay (drill): send the drill trigger the arm is commanded, stamped on the robot's clock, as "
                         "UDP JSON {t (epoch ms), tool:'drill', btn_trig, src:'robot', run} to HOST:PORT (the robot-origin "
                         "relay's --trig-udp, default 127.0.0.1:9872; ROBOT_ORIGIN.md). Sent on every change and every "
                         "100 ms (relay stale 300 ms); the last one is 0 with end:1. Off when empty")
    ap.add_argument("--trig-delay-s", type=float, default=0.0,
                    help="robot time: switch the trigger this much after the recording's own press/release instant, "
                         "so it lands when the arm (which lags its target) reaches where the hand was")
    ap.add_argument("--run-id", default=None, help="id sent with the trigger (default: the run dir name)")
    ap.add_argument("--replay-stream-udp", type=int, default=None,
                    help="replay: more poses arrive on this UDP port while the arm runs (a closed-loop policy, "
                         "robot/inference/live_loop.py) — JSON {tool, t_wall (ms, when the arm should be there), quat "
                         "(wxyz, hand tool → camera), pos (m, camera — the marker origin, as a recorder row)} appended to the recording, or {end: 1}. "
                         "--replay-run is then the seed (the start pose); the arm holds the last pose while none "
                         "comes; --replay-max-s = the longest run in wall seconds (default 900)")
    ap.add_argument("--go-file", default=None,
                    help="steps / replay: at the start pose, wait until this file exists before engaging — two arms "
                         "(two processes) start the same recording on the same camera frame")
    a = ap.parse_args()
    a.tool_calib = a.tool_calib or str(TOOL_CALIB_ARM[a.arm])
    return a


def steps_schedule(A: float):
    """[(t_start_s, fn(t_rel) → dp_mm (3,))] for the steps mode, camera axes; returns (segments, total_s)."""
    segs, t = [], 0.0

    def hold(d, v):
        nonlocal t
        segs.append((t, t + d, lambda _t, v=np.array(v, float): v))
        t += d

    hold(1.0, [0, 0, 0])
    for ax in range(3):
        for s in (+1, 0, -1, 0):
            v = [0.0, 0.0, 0.0]
            v[ax] = s * A
            hold(1.5, v)
    for f in (0.5, 1.0):
        t0 = t
        segs.append((t0, t0 + 6.0, lambda tt, f=f, t0=t0: np.array([A * np.sin(2 * np.pi * f * (tt - t0)), 0, 0])))
        t += 6.0
    hold(1.0, [0, 0, 0])
    return segs, t


class Log:
    def __init__(self, path: Path):
        self.f = open(path, "w")
        self.lock = threading.Lock()

    def write(self, d: dict):
        s = json.dumps(d, separators=(",", ":"))
        with self.lock:
            self.f.write(s + "\n")

    def close(self):
        with self.lock:
            self.f.close()


def load_replay(run_dir: str):
    """Hand motion of a recorded follow run (raw tracker poses between its engage and disengage): t (s from the
    first sample), pos (m, camera), R (tool → camera), vis, and the recording's engage latch (H0 tip m, H0 R) —
    so filters can be compared on the same real motion."""
    import collections

    ev = collections.defaultdict(list)
    for line in Path(run_dir, "events.jsonl").read_text().splitlines():
        d = json.loads(line)
        ev[d["k"]].append(d)
    e = ev["engage"][0]
    t_end = ev["disengage"][0]["t"] if ev["disengage"] else float("inf")
    role = {f["fn"]: f["hand"] for f in ev["frame"]}
    rows = sorted((m["ts"], np.asarray(m["raw_pos"]) / 1000.0, he.quat_to_R(m["raw_quat"]), m.get("vis", 4))
                  for m in ev["tap"] if role.get(m.get("fn")) == m["tool"] and not m.get("lost")
                  and e["t"] - 20 <= m["ts"] <= t_end)
    t0 = rows[0][0]
    return {"t": np.array([(r[0] - t0) / 1000.0 for r in rows]), "pos": [r[1] for r in rows],
            "R": [r[2] for r in rows], "vis": [r[3] for r in rows],
            "H0_tip": np.asarray(e["H0_tip_mm"]) / 1000.0, "H0_R": np.asarray(e["H0_R"])}


def zero_phase_lp(x: np.ndarray, fc: float, fs: float) -> np.ndarray:
    """Forward-backward 2nd-order Butterworth low-pass (RBJ biquad, Q = 1/√2) along axis 0 — no phase lag, so it
    only suits recorded data. Edges are padded with the end values to limit start/stop transients."""
    w = 2 * np.pi * fc / fs
    al, c = np.sin(w) / np.sqrt(2.0), np.cos(w)
    b = np.array([(1 - c) / 2, 1 - c, (1 - c) / 2]) / (1 + al)
    a1, a2 = -2 * c / (1 + al), (1 - al) / (1 + al)
    pad = int(fs / fc) * 2

    def one(y):
        z = np.empty_like(y)
        z[0] = z[1] = y[0]
        for n in range(2, len(y)):
            z[n] = b[0] * y[n] + b[1] * y[n - 1] + b[2] * y[n - 2] - a1 * z[n - 1] - a2 * z[n - 2]
        return z

    x = np.asarray(x, float)
    y = np.concatenate([np.repeat(x[:1], pad, 0), x, np.repeat(x[-1:], pad, 0)])
    return one(one(y)[::-1])[::-1][pad:-pad]


def _R_to_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → unit quaternion (w, x, y, z), the tracker's order (inverse of he.quat_to_R)."""
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def clean_replay(rp: dict, fc: float, fs: float = 90.0) -> dict:
    """Offline-cleaned hand motion: 3-ball fixes dropped, positions and quaternions resampled to fs, zero-phase
    low-passed at fc, sampled back at the kept frame times."""
    keep = [i for i, v in enumerate(rp["vis"]) if v >= 4]
    t = rp["t"][keep]
    P = np.array([rp["pos"][i] for i in keep])
    Q = np.array([_R_to_wxyz(rp["R"][i]) for i in keep])
    Q[(Q @ Q[0]) < 0] *= -1
    g = np.arange(t[0], t[-1], 1.0 / fs)
    Pg = np.array([np.interp(g, t, P[:, k]) for k in range(3)]).T
    Qg = np.array([np.interp(g, t, Q[:, k]) for k in range(4)]).T
    Pf, Qf = zero_phase_lp(Pg, fc, fs), zero_phase_lp(Qg, fc, fs)
    P2 = np.array([np.interp(t, g, Pf[:, k]) for k in range(3)]).T
    Q2 = np.array([np.interp(t, g, Qf[:, k]) for k in range(4)]).T
    Q2 /= np.linalg.norm(Q2, axis=1, keepdims=True)
    R2 = [he.quat_to_R(q) for q in Q2]
    # the engage latch (H0: tip, not marker origin) stays the recorded one — cleaning moves it by < 1 mm
    return {**rp, "t": t, "pos": list(P2), "R": R2, "vis": [4] * len(t)}


class TargetShaper:
    """Turns the human tip target into one the arm can follow (it tracks ≈ 1–2 Hz with 60–120 ms lag; a 2 cm jump
    in one frame made it overshoot and ring, and 3-ball poses tilted the 166 mm lever tip by 15–20 mm).
      clean: a 3-ball pose keeps the last 4-ball rotation (its tip is re-derived from the position); a tip step
             larger than frame_jump_mm in one frame is held.
      shape: clean + a critically damped 2nd-order filter (natural frequency hz) with speed and acceleration caps.
    Works on the tip displacement dp (m) and the rotation dR; returns the shaped (dp, dR)."""

    def __init__(self, mode: str, hz: float, vmax_mm: float, amax_mm: float, jump_mm: float,
                 dead_mm: float = 0.0, dead_deg: float = 0.0, R_bc=None, z_hz=None, z_dead_mm=None, lead_s: float = 0.0,
                 kf_q: float = 2000.0, kf_r: float = 0.15):
        self.lead = lead_s
        self.kf_q, self.kf_r = kf_q, kf_r  # constant-velocity Kalman (mm units): predicted the hand 60 ms ahead
        # with 1.65 mm RMS on the 23:06 recording vs 2.75 for the 2nd-order shaper + lead (offline, 2026-09-24)
        # per-axis in the robot base frame: the vertical (gravity) axis can be filtered harder than the horizontal
        self.R = np.eye(3) if R_bc is None else np.asarray(R_bc, float)
        w = 2 * np.pi * hz
        self.wv = np.array([w, w, 2 * np.pi * z_hz if z_hz else w])
        self.deadv = np.array([dead_mm, dead_mm, dead_mm if z_dead_mm is None else z_dead_mm]) / 1000.0
        self.mode, self.w = mode, w
        self.vmax, self.amax, self.jump = vmax_mm / 1000.0, amax_mm / 1000.0, jump_mm / 1000.0
        self.dead, self.dead_rad = dead_mm / 1000.0, np.deg2rad(dead_deg)
        self.reset()

    def reset(self):
        self.x = self.v = self.last_dp = self.held = None
        self.n_held = 0
        self.kx = self.kP = None
        self.acc, self.v_prev = np.zeros(3), None
        self.last_R, self.t, self.held_R = np.eye(3), None, None

    def _deadband(self, dp, dR):
        """The held point trails the hand at the deadband radius: moves smaller than the radius (tremor) are ignored,
        larger ones pass minus that offset — continuous, no step when the hand leaves the band."""
        if self.held is None:
            self.held, self.held_R = dp.copy(), dR.copy()
        gap = self.R @ (dp - self.held)  # base frame: an ellipsoidal band, its vertical radius set separately
        dv = np.maximum(self.deadv, 1e-9)
        g = float(np.linalg.norm(gap / dv))
        if g > 1.0:
            self.held = self.held + self.R.T @ (gap * (1.0 - 1.0 / g))
        rel = self.held_R.T @ dR
        ang = float(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))
        if ang > self.dead_rad and ang > 1e-9:
            axis = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]]) / (2 * np.sin(ang))
            self.held_R = self.held_R @ he.rotvec_to_R(axis * (ang - self.dead_rad))
        return self.held.copy(), self.held_R.copy()

    def __call__(self, dp, dR, vis: int, t_s: float, pos_rel=None):
        if self.mode == "none":
            return dp, dR
        if vis is not None and vis < 4:
            if pos_rel is not None:  # re-derive the tip with the held rotation
                dp = pos_rel(self.last_R)
            dR = self.last_R
        else:
            self.last_R = dR
        # a single-frame jump is held for at most 3 frames (outlier); if it persists it is real motion and passes
        # (the speed/acceleration caps below then take it smoothly) — an unbounded hold froze the target for good
        if self.last_dp is not None and float(np.linalg.norm(dp - self.last_dp)) > self.jump and self.n_held < 3:
            dp = self.last_dp
            self.n_held += 1
        else:
            self.n_held = 0
        self.last_dp = dp
        if self.mode == "clean":
            return dp, dR
        if self.deadv.max() > 0 or self.dead_rad > 0:
            dp, dR = self._deadband(dp, dR)
        if self.mode == "kalman":
            z = dp * 1000.0
            if self.kx is None:
                self.kx = np.stack([z, np.zeros(3)], axis=1)  # per axis (pos mm, vel mm/s)
                self.kP = np.repeat((np.eye(2) * 10.0)[None], 3, axis=0)
                self.t = t_s
                return dp, dR
            dt = min(max(t_s - self.t, 1e-4), 0.05)
            self.t = t_s
            F = np.array([[1.0, dt], [0.0, 1.0]])
            Q = np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]]) * self.kf_q
            for i in range(3):
                x = F @ self.kx[i]
                P = F @ self.kP[i] @ F.T + Q
                K = P[:, 0] / (P[0, 0] + self.kf_r ** 2)
                self.kx[i] = x + K * (z[i] - x[0])
                self.kP[i] = P - np.outer(K, P[0])
            return (self.kx[:, 0] + self.kx[:, 1] * self.lead) / 1000.0, dR
        if self.x is None:
            self.x, self.v, self.t = dp.copy(), np.zeros(3), t_s
            return self.x.copy(), dR
        dt = min(max(t_s - self.t, 0.0), 0.05)
        self.t = t_s
        acc = self.R.T @ (self.wv ** 2 * (self.R @ (dp - self.x)) - 2 * self.wv * (self.R @ self.v))
        an = float(np.linalg.norm(acc))
        if an > self.amax:
            acc *= self.amax / an
        self.v = self.v + acc * dt
        vn = float(np.linalg.norm(self.v))
        if vn > self.vmax:
            self.v *= self.vmax / vn
        self.x = self.x + self.v * dt
        if dt > 0:  # causal acceleration estimate (EMA ~10 Hz) — a learned feed-forward uses it
            a_raw = (self.v - self.v_prev) / dt if self.v_prev is not None else np.zeros(3)
            self.acc = self.acc + (1.0 - np.exp(-2 * np.pi * 10.0 * dt)) * (a_raw - self.acc)
        self.v_prev = self.v.copy()
        return self.x + self.v * self.lead, dR


class SmithPredictor:
    """Per-joint Smith predictor on the arm's server ticks (arm angles, degrees).

    Plant model (fitted from steps logs, robot/…/joint_model_left.json): pos[k+1] = a·pos[k] + b·cmd[k−d] + c. Each new
    state tick k (encoder pos[k] read before the tick's write, cmd[k] the command the server wrote) updates
      y0  = the model without the extra delay d (what the joint will do from commands already written),
      yd  = the model with it (what the encoder should read), and the mismatch e = pos − yd.
    ŷ = y0 + e is the joint angle the feedback should see: the encoder's delay is taken out, disturbances (sag, model
    error) are kept. Feedback on ŷ can use a higher gain than on the raw encoder, which rang at 2–4 Hz.
    """

    def __init__(self, model: dict, e_hz: float = 0.0, dt: float = 1 / 30):
        self.e_alpha = 1.0 - np.exp(-2 * np.pi * e_hz * dt) if e_hz > 0 else 1.0
        J = model["joints"]
        self.a = np.array([J[j]["a"] for j in JOINTS])
        self.b = np.array([J[j]["b"] for j in JOINTS])
        self.c = np.array([J[j]["c"] for j in JOINTS])
        self.d = np.array([int(J[j]["delay_ticks"]) for j in JOINTS])
        self.reset()

    def reset(self):
        self.y0 = self.yd = None
        self.hist: deque = deque(maxlen=16)
        self.tick = None
        self.e = np.zeros(6)

    def update(self, s: dict) -> None:
        if not s.get("cmd") or s.get("state") != "MOVING":
            self.reset()
            return
        tick = s.get("tick")
        pos = np.array([s["pos"][j] for j in JOINTS])
        cmd = np.array([s["cmd"][j] for j in JOINTS])
        if self.y0 is None or tick != (self.tick or 0) + 1:
            self.y0, self.yd = pos.copy(), pos.copy()
            self.hist.clear()
            self.e = np.zeros(6)
        self.e = self.e + self.e_alpha * ((pos - self.yd) - self.e)
        self.hist.append(cmd)
        u_d = np.array([self.hist[max(0, len(self.hist) - 1 - self.d[i])][i] for i in range(6)])
        self.y0 = self.a * self.y0 + self.b * cmd + self.c
        self.yd = self.a * self.yd + self.b * u_d + self.c
        self.tick = tick

    def estimate(self):
        return None if self.y0 is None else self.y0 + self.e


class JointMPC:
    """Unconstrained per-joint MPC (preview control) on pos[k+1] = a·pos[k] + b·u + c.

    The Smith model's undelayed state y0 (+ the filtered mismatch e) is the joint angle the next command starts to
    act on (the extra delay d is already accounted for); free inputs u0…u(N−1) give Y = F·y + G·U + H. Minimising
    ||Y − R||² + λ||ΔU||² (ΔU includes u0 − u_prev) is linear: u0 = k·(R − F·y − H) + m·u_prev, precomputed per joint.
    """

    def __init__(self, model: dict, lam: float, N: int):
        J = model["joints"]
        self.N = N
        self.gains = []
        for j in JOINTS:
            a, b, c = J[j]["a"], J[j]["b"], J[j]["c"]
            F = np.array([a ** (i + 1) for i in range(N)])
            G = np.zeros((N, N))
            for i in range(N):
                for m in range(i + 1):
                    G[i, m] = a ** (i - m) * b
            H = np.array([c * sum(a ** q for q in range(i + 1)) for i in range(N)])
            D = np.eye(N) - np.eye(N, k=-1)
            e1 = np.zeros(N)
            e1[0] = 1.0
            A = np.linalg.inv(G.T @ G + lam * D.T @ D)
            k = (A @ G.T)[0]
            m = (A @ (lam * D.T @ e1))[0]
            self.gains.append((F, H, k, m, int(J[j]["delay_ticks"])))

    def command(self, y, r_now, r_vel, u_prev, dt=1 / 30):
        out = np.empty(6)
        steps = np.arange(1, self.N + 1) * dt
        for i, (F, H, k, m, d) in enumerate(self.gains):
            R = r_now[i] + r_vel[i] * (steps + d * dt)  # y0 is the output d ticks ahead
            out[i] = k @ (R - F * y[i] - H) + m * u_prev[i]
        return out


class JointDOB:
    """Input disturbance observer on the server ticks (arm angles). Nominal plant without the bias term:
    pos[k+1] = a·pos[k] + b·(cmd[k−d] + w). From each tick w_raw = (pos[k+1] − a·pos[k]) / b − cmd[k−d], low-passed
    (Q filter) → ŵ, and the command gets −ŵ. Gravity sag and friction show up as w; the Q cutoff must stay below the
    loop's delay (1+d ticks) to stay stable."""

    def __init__(self, model: dict, hz: float, dt: float = 1 / 30, clamp: float = 8.0):
        J = model["joints"]
        self.a = np.array([J[j]["a"] for j in JOINTS])
        self.b = np.array([J[j]["b"] for j in JOINTS])
        self.d = np.array([int(J[j]["delay_ticks"]) for j in JOINTS])
        self.alpha = 1.0 - np.exp(-2 * np.pi * hz * dt)
        self.clamp = clamp
        self.reset()

    def reset(self):
        self.w = np.zeros(6)
        self.prev_pos, self.tick = None, None
        self.hist: deque = deque(maxlen=16)

    def update(self, s: dict) -> None:
        if not s.get("cmd") or s.get("state") != "MOVING":
            self.prev_pos, self.tick = None, None
            self.hist.clear()
            return
        tick = s.get("tick")
        pos = np.array([s["pos"][j] for j in JOINTS])
        if self.prev_pos is not None and tick == self.tick + 1 and len(self.hist) > int(self.d.max()):
            u_d = np.array([self.hist[len(self.hist) - 1 - self.d[i]][i] for i in range(6)])
            w_raw = (pos - self.a * self.prev_pos) / self.b - u_d
            self.w = np.clip(self.w + self.alpha * (w_raw - self.w), -self.clamp, self.clamp)
        elif self.prev_pos is None:
            self.hist.clear()
        self.hist.append(np.array([s["cmd"][j] for j in JOINTS]))
        self.prev_pos, self.tick = pos, tick


def move_to_start(rpc, tok, spec: str, clear, off, min_clear_mm: float, speed_dps: float = 30.0) -> None:
    """Ramp the goal (arm angles) to the start pose at speed_dps, then wait until settled (≤ 6 s)."""
    q0 = np.array([rpc.state()["pos"][j] for j in JOINTS])
    q1 = q0.copy()
    for kv in spec.split(","):
        j, v = kv.split("=")
        q1[JOINTS.index(j.strip())] = float(v)
    if clear.mm(q1 + off) < min_clear_mm:
        sys.exit(f"--start 자세의 집게–2번 모터 거리 {clear.mm(q1 + off):.0f} mm < {min_clear_mm:.0f}")
    n = max(1, int(np.ceil(np.abs(q1 - q0).max() / (speed_dps / 30.0))))
    print(f"[follow] 시작 자세로 이동 {np.round(q1, 1).tolist()} ({n / 30:.1f} s)", flush=True)
    for i in range(1, n + 1):
        rpc.must(cmd="goal", token=tok, pos={j: float(v) for j, v in zip(JOINTS, q0 + (q1 - q0) * i / n)})
        time.sleep(1 / 30)
    hist: deque = deque(maxlen=10)
    t0 = time.time()
    while time.time() - t0 < 6.0:
        rpc.must(cmd="goal", token=tok, pos={j: float(v) for j, v in zip(JOINTS, q1)})
        hist.append([rpc.state()["pos"][j] for j in JOINTS])
        if len(hist) == hist.maxlen and float(np.ptp(np.array(hist), axis=0).max()) < 0.3:
            break
        time.sleep(1 / 30)
    time.sleep(0.3)


def trigger_timeline(rp: dict, speed: float) -> tuple[np.ndarray, np.ndarray]:
    """The recording's drill trigger (row `btn`, pressed = 1) on the replay clock: (t_s since the replay's first sample,
    divided by the replay speed like rp["t"], value). Empty when the recording carries none."""
    rec_path = (rp.get("recording") or {}).get("path")
    tf, t0 = rp.get("time_source") or "tExp", rp.get("t0_ms")
    if not rec_path or t0 is None:
        return np.zeros(0), np.zeros(0, bool)
    ts, vs = [], []
    for line in open(rec_path):
        if '"k":"f"' not in line[:12] or '"btn"' not in line:
            continue
        r = json.loads(line)
        if r.get("tool") == "drill" and "btn" in r and r.get(tf) is not None:
            ts.append((float(r[tf]) - float(t0)) / 1000.0 / speed)
            vs.append(bool(r["btn"]))
    o = np.argsort(ts)
    return np.asarray(ts, float)[o], np.asarray(vs, bool)[o]


def rotexp(v: np.ndarray) -> np.ndarray:
    """Rotation matrix of the rotation vector v (Rodrigues)."""
    ang = float(np.linalg.norm(v))
    if ang < 1e-12:
        return np.eye(3)
    k = v / ang
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def rotvec(R: np.ndarray) -> np.ndarray:
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if ang < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))
    return v * ang


def main():
    a = parse()
    rp_in, stop_msg = rec.replay_input(a, load_replay) if a.mode == "replay" else (None, None)
    if stop_msg:
        sys.exit(stop_msg)  # read (and described) before the arm is touched
    X, Z, he_meta = he.load(a.handeye or Path.home() / f"il-robot/calib/handeye_{a.arm}.json")
    off = he.joint_offsets(he_meta)  # model angle = arm angle + off (FK/IK work in model angles)
    pts_m, tip_m = he.tool_calib(a.tool_calib)       # the hand's tool (recording / followed hand)
    tip_r = he.tool_calib(a.robot_tool_calib)[1] if a.robot_tool_calib else tip_m   # the arm's own block
    R_bc = Z[:3, :3]
    Zi = he.inv(Z)
    run = Path(a.out or Path.home() / "il-robot/runs/latency" / f"{time.strftime('%Y%m%dT%H%M%S')}_{a.mode}")
    run.mkdir(parents=True, exist_ok=True)
    urdf = he.urdf_with_tip(URDF, X, tip_r, run / "tip.urdf")
    T_link_from_tip = he.inv(X @ he.compose(np.eye(3), tip_r))

    cmd_port = a.cmd_port or DEFAULT_PORTS[a.arm][0]
    pub_port = a.pub_port or DEFAULT_PORTS[a.arm][1]
    if a.transport == "ros2":   # commands through ROS 2 (spinearm_bridge) instead of ZMQ — ROS2-1-design.md stage 2
        from lerobot_robot_spinearm.ros_client import RosArmRpc
        rpc = RosArmRpc(a.arm)
    else:
        rpc = ArmRpc(a.host, cmd_port)
    st0 = rpc.status()
    bounds = {j: tuple(v) for j, v in st0["bounds"].items()}
    kin = MarkerKinematics(urdf=urdf, frame=he.TIP_FRAME)
    lim_model = {j: (lo + off[i] + a.limit_margin_deg, hi + off[i] - a.limit_margin_deg)
                 for i, (j, (lo, hi)) in enumerate(bounds.items())}
    axis_local = None
    if a.ik_axis != "none":  # align only the tool's shaft (5-DOF: tip + axis), roll left to the posture task
        rig = json.loads((Path(__file__).resolve().parent.parent / "bridge/src/spine_render/data/rig_default.json")
                         .read_text())["tools"]
        axis_local = rig[a.ik_axis]["shaftLocal"]
    lock = {kv.split("=")[0].strip(): float(kv.split("=")[1]) for kv in a.ik_lock.split(",") if "=" in kv}
    tik_solver = TipIK(urdf, he.TIP_FRAME, lim_model, a.ori_weight, a.post_weight, a.vmax_dps, 90.0,
                       axis_local=axis_local, w_axis=a.axis_weight, lock=lock) if a.ik == "tip" else None
    # IK works in model angles (arm + off): shift the arm's limits, and stay a margin inside them — placo shoves a
    # joint that crosses its limit back by several degrees, which made the solution flip between branches (the
    # 2026-09-23 run: wrist_flex riding its limit → lift/elbow jumping ±10° → tip spikes of 30–50 mm).
    kin.set_limits({j: (lo + off[i] + a.limit_margin_deg, hi + off[i] - a.limit_margin_deg)
                    for i, (j, (lo, hi)) in enumerate(bounds.items())})
    kin.set_speed(a.vmax_dps, 90.0)
    clear = Clearance()

    log = Log(run / "events.jsonl")
    if rp_in is not None:   # what this run replays — logged before the arm is leased or moved
        log.write({"k": "replay_src", **rec.provenance(rp_in, a)})
    states = StateLog(a.host, pub_port, sink=lambda s: log.write({"k": "state", **s}))
    tap = TapReceiver(a.tap_port)
    asm = FrameAssembler(ROBOT_TOOLS[a.arm])
    (run / "config.json").write_text(json.dumps({
        "args": vars(a), "handeye": str(a.handeye or ""), "handeye_fit": he_meta.get("fit"),
        "handeye_val": he_meta.get("val"), "T_base_cam": Z.tolist(), "T_link_const": X.tolist(),
        "tip_local_m": tip_m.tolist(), "robot_tip_local_m": tip_r.tolist(),
        "robot_tool_calib": a.robot_tool_calib, "joint_offset_deg": off.tolist(), "urdf": str(urdf), "frame": he.TIP_FRAME, "bounds": bounds,
        "clock_wall_minus_mono_ms": mono_offset_ms()}, indent=1))

    stop = {"flag": False, "why": ""}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True, why="sigint"))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True, why="sigterm"))

    tok = rpc.lease(f"endo_follow:{a.mode}")
    n_goal = n_frames = 0
    try:
        if rpc.state()["state"] == "OFF":
            rpc.must(cmd="enable", token=tok)
        rpc.must(cmd="set", token=tok, step_deg=a.step_deg)
        if a.accel is not None:
            rpc.must(cmd="set", token=tok, accel=a.accel)
        t_wait = time.time()
        while states.latest is None and time.time() - t_wait < 2:
            time.sleep(0.02)
        if states.latest is None:
            sys.exit("팔 서버 PUB 상태가 안 온다")
        if a.start:
            move_to_start(rpc, tok, a.start, clear, off, a.min_clearance_mm)
        # the tap queued up while we were busy: start from live frames only (a stale robot fix taken as the engage
        # reference shifted every steps-mode target by 5 cm, 2026-09-23 23:14)
        print(f"[follow] 밀린 탭 메시지 {tap.flush()}개 버림", flush=True)

        engaged = False
        go_wait = False  # --go-file: announced that the arm is at its start pose and waits
        eng: dict = {}
        seq = 0
        q_seed = None
        q_cmd, t_cmd = None, 0.0  # last sent goal (model angles) and its time, for the rate limit
        last_robot_tip, last_robot_wall, amb_since = None, 0.0, 0.0
        vis_I, vis_t, prev_tgt = np.zeros(3), 0.0, None  # camera-frame tip correction (m)
        tgt_hist: deque = deque(maxlen=200)  # (monotonic s, intended target) for the delayed comparison
        sag_I = np.zeros(6)       # joint correction added to the goal (arm angles), integrates goal − encoder
        sag_t, sag_tick = 0.0, -1  # last integration time / state tick used
        hand_hist: deque = deque(maxlen=400)  # (ts, tip_m)
        last_hand_wall = 0.0
        last_hand_tip = None
        jump_prev = None   # last rejected 4-ball hand tip (jump-gate re-anchoring)
        t_clock = t_renew = t_goal = 0.0   # t_goal: when the arm server last took a goal (a goal renews the lease)
        segs, total_s = steps_schedule(a.amp_mm) if a.mode == "steps" else (None, a.duration)
        rp = None
        stream_sock, n_stream = None, 0
        if a.mode == "replay":
            rp = rp_in  # a follow run dir (load_replay) or an FE recorder file, read before the arm moved
            if a.replay_clean_hz:
                n_raw = len(rp["t"])
                rp = clean_replay(rp, a.replay_clean_hz)
                print(f"[replay] 오프라인 정제 {a.replay_clean_hz:g} Hz — 3볼 {n_raw - len(rp['t'])}프레임 제외", flush=True)
            if a.replay_speed != 1.0:   # same path, stretched in time
                rp = {**rp, "t": rp["t"] / a.replay_speed}
                print(f"[replay] 배속 {a.replay_speed:g}×", flush=True)
            total_s = float(rp["t"][-1]) if a.replay_max_s is None else \
                min(float(rp["t"][-1]), a.replay_max_s / a.replay_speed)
            if a.replay_stream_udp:   # a growing recording: lists, appended as the poses arrive (sample_pose holds the last)
                rp = {**rp, "t": list(np.asarray(rp["t"], float)), "R": list(rp["R"]), "pos": list(rp["pos"]),
                      "vis": list(rp["vis"])}
                stream_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                stream_sock.bind(("127.0.0.1", a.replay_stream_udp)); stream_sock.setblocking(False)
                total_s = a.replay_max_s or 900.0
                print(f"[replay] 스트림 입력 udp :{a.replay_stream_udp} — 최대 {total_s:.0f} s", flush=True)
            print(f"[replay] {a.replay_run}: {len(rp['t'])} 손 포즈, {total_s:.1f} s", flush=True)
        trig_t, trig_v = (np.zeros(0), np.zeros(0, bool))
        trig_sock, trig_addr = None, None
        trig_state = {"last": None, "sent_t": 0.0}   # last value sent, when (monotonic s)
        if a.trig_udp and rp is not None and a.replay_tool == "drill":
            trig_t, trig_v = trigger_timeline(rp, a.replay_speed)
            host, port = a.trig_udp.rsplit(":", 1)
            trig_sock, trig_addr = socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (host, int(port))
            print(f"[replay] 방아쇠 → udp {a.trig_udp}: 녹화 전환 {int(np.count_nonzero(np.diff(trig_v.astype(int))))}회, "
                  f"지연 {a.trig_delay_s:g} s (로봇 시각)", flush=True)

        def send_trig(on: bool, t_rel_s: float | None, why: str = "") -> None:
            trig_state["last"] = on
            msg = {"t": wall_ms(), "tool": "drill", "btn_trig": int(on), "src": "robot",
                   "run": a.run_id or run.name}
            if why == "end":   # the relay stops holding the channel (else it logs stale rows until it stops)
                msg["end"] = 1
            try:
                trig_sock.sendto(json.dumps(msg).encode(), trig_addr)
            except OSError:
                pass
            if why:
                log.write({"k": "trig", **msg, "t_rel": None if t_rel_s is None else round(t_rel_s, 4), "why": why})
        ilc = dict(np.load(a.ilc_file)) if a.ilc_file else None
        ff_W = np.asarray(json.loads(Path(a.ff_model).read_text())["W"]) if a.ff_model else None
        shaper = TargetShaper(a.filter, a.shape_hz, a.shape_vmax, a.shape_amax, a.frame_jump_mm,
                              a.deadband_mm, a.deadband_deg, R_bc, a.z_hz, a.z_deadband_mm, a.lead_ms / 1000.0,
                              a.kf_q, a.kf_r)
        sag_mask = np.array([j in a.sag_joints.split(",") for j in JOINTS], float)
        od_k = np.full(6, a.overdrive)
        smith = SmithPredictor(json.loads(Path(a.smith_model).read_text()), a.smith_e_hz) if a.smith_model else None
        mpc = JointMPC(json.loads(Path(a.smith_model).read_text()), a.mpc_lambda, a.mpc_horizon) \
            if (a.mpc_lambda > 0 and a.smith_model) else None
        mpc_u, mpc_rprev, mpc_rvel, mpc_t = None, None, np.zeros(6), 0.0
        sm_k = np.full(6, a.smith_k)
        for kv in filter(None, a.smith_k_joint.split(",")):
            j, v = kv.split("=")
            sm_k[JOINTS.index(j.strip())] = float(v)
        sm_last_tick = None
        dob = JointDOB(json.loads(Path(a.smith_model).read_text()), a.dob_hz, clamp=a.dob_max_deg) \
            if (a.dob_hz > 0 and a.smith_model) else None
        od_lp, od_t = np.zeros(6), 0.0
        for kv in filter(None, a.overdrive_joint.split(",")):
            j, v = kv.split("=")
            od_k[JOINTS.index(j.strip())] = float(v)
        t_begin = time.time()
        print(f"[follow] 기록 → {run}", flush=True)
        if a.mode == "follow":
            print("[follow] 손 내시경을 1초 가만히 들고 있으면 추종 시작 · 마커를 가리면 정지", flush=True)

        def disengage(why):
            nonlocal engaged
            hand_hist.clear()
            if engaged:
                engaged = False
                log.write({"k": "disengage", "t": wall_ms(), "reason": why})
                print(f"[follow] 정지 ({why})", flush=True)

        while not stop["flag"] and time.time() - t_begin < total_s + (0 if a.mode == "follow" else 30):
            now = time.time()
            if now - t_clock > 1.0:
                t_clock = now
                log.write({"k": "clock", "wall": wall_ms(), "mono": time.monotonic() * 1000.0,
                           "off": mono_offset_ms()})
            if now - t_renew > 5.0 and (not engaged or now - t_goal > 5.0):
                # also while engaged but sending no goals: every frame skipped as stale (10-05: the tracker's frame stamps
                # drifted behind the host clock for minutes) — the server's 15 s lease would run out and the arm, still
                # holding its last goal, would refuse every goal for the rest of the run
                t_renew = now
                rpc.call(cmd="renew", token=tok)
            m = tap.recv(0.2)
            if m is None:
                if engaged and wall_ms() - last_hand_wall > a.lost_ms and a.mode == "follow":
                    disengage("no_tap")
                continue
            if m.get("tool") not in ROBOT_TOOLS[a.arm]:
                continue
            log.write({"k": "tap", **m})
            f = asm.add(m)
            if f is None:
                continue
            if wall_ms() - (m.get("ts") or 0) > a.stale_ms:  # never act on an old frame
                log.write({"k": "skip", "fn": m.get("fn"), "why": "stale"})
                continue
            n_frames += 1
            st = states.latest
            q_enc = np.array([st["pos"][j] for j in JOINTS], float) + off  # model angles
            pred_tip_cam = (Zi @ kin.fk(q_enc))[:3, 3]
            poses = {k: pose_of(v, tip_r, raw=a.source == "raw") for k, v in f.items()}   # arm tip model
            # reference: the arm marker's own last fix while fresh (≤ 60 ms, a few mm at 90 fps), else FK prediction
            ref = last_robot_tip if (last_robot_tip is not None and wall_ms() - last_robot_wall < 60) else pred_tip_cam
            robot, hand, status = pick_robot(poses, ref, a.gate_mm / 1000.0, a.margin_mm / 1000.0)
            if hand is not None and tip_r is not tip_m:   # the followed hand's tip with the hand's own model
                hand = pose_of(f[hand["tool"]], tip_m, raw=a.source == "raw")
            if robot is not None:
                last_robot_tip, last_robot_wall = robot["tip"], wall_ms()
            if status == "ambiguous":
                amb_since = amb_since or wall_ms()
            else:
                amb_since = 0.0
            fn = m.get("fn")
            ts_frame = m.get("ts")
            log.write({"k": "frame", "fn": fn, "ts": ts_frame, "status": status,
                       "robot": robot["tool"] if robot else None, "hand": hand["tool"] if hand else None,
                       "pred_tip_mm": np.round(pred_tip_cam * 1000, 2).tolist(), "tick": st.get("tick")})
            t_now = wall_ms()

            if a.mode == "follow":
                if status == "ambiguous":  # skip the frame; stop only if it lasts
                    if engaged and wall_ms() - amb_since > a.ambiguous_ms:
                        disengage("ambiguous")
                    continue
                if hand is not None:
                    if last_hand_tip is not None and engaged and \
                            np.linalg.norm(hand["tip"] - last_hand_tip) * 1000 > a.jump_mm:
                        # Re-anchor after a dropout: the hand moves while the marker is hidden, so comparing with the
                        # last accepted tip rejected every later frame and the run stopped as hand_lost
                        # (2026-09-26 21:05). Accept a full 4-ball fix that agrees with the previous rejected
                        # 4-ball fix; 3-ball fixes (their tip can be tens of mm off) never re-anchor.
                        vis_h = int(f[hand["tool"]].get("vis", 4))
                        if not (vis_h >= 4 and jump_prev is not None
                                and np.linalg.norm(hand["tip"] - jump_prev) * 1000 <= a.jump_mm):
                            jump_prev = hand["tip"] if vis_h >= 4 else None
                            last_hand_wall = t_now   # seen but not trusted: hold the arm, do not count it as lost
                            log.write({"k": "skip", "fn": fn, "why": "hand_jump"})
                            continue
                        log.write({"k": "reanchor", "fn": fn,
                                   "jump_mm": round(float(np.linalg.norm(hand["tip"] - last_hand_tip)) * 1000, 1)})
                    jump_prev = None
                    last_hand_tip = hand["tip"]
                    last_hand_wall = t_now
                    hand_hist.append((hand["ts"], hand["tip"]))
                elif engaged and t_now - last_hand_wall > a.lost_ms:
                    disengage("hand_lost")
                    continue
                if not engaged:
                    if hand is None or robot is None:
                        continue
                    win = [p for (ts, p) in hand_hist if ts is not None and ts >= hand["ts"] - a.engage_still_s * 1000]
                    span = (hand_hist[-1][0] - hand_hist[0][0]) if len(hand_hist) > 1 else 0
                    if span < a.engage_still_s * 1000 or len(win) < 20:
                        continue
                    P = np.array(win)
                    if float(np.linalg.norm(P.max(0) - P.min(0))) * 1000 > a.engage_still_mm:
                        continue
                    engaged, q_seed, q_cmd, t_cmd = True, q_enc.copy(), q_enc.copy(), 0.0
                    if tik_solver is not None:
                        tik_solver.set_posture(q_enc)
                    sag_I[:] = 0.0
                    vis_I[:] = 0.0
                    tgt_hist.clear()
                    eng = {"H0_tip": hand["tip"], "H0_R": hand["R"], "R0_tip": robot["tip"], "R0_R": robot["R"],
                           "FK0": kin.fk(q_enc)}
                    log.write({"k": "engage", "t": t_now, "fn": fn, "hand_tool": hand["tool"],
                               "robot_tool": robot["tool"], "H0_tip_mm": (hand["tip"] * 1000).tolist(),
                               "H0_R": hand["R"].tolist(), "R0_tip_mm": (robot["tip"] * 1000).tolist(),
                               "R0_R": robot["R"].tolist(), "FK0": eng["FK0"].tolist(), "q0": q_enc.tolist()})
                    shaper.reset()
                    print("[follow] ▶ 추종 시작", flush=True)
                    continue
                if hand is None:
                    continue
                dp = hand["tip"] - eng["H0_tip"]
                dR = np.eye(3) if a.pos_only else hand["R"] @ eng["H0_R"].T
                ts_ref = hand["ts"]
                vis = int(f[hand["tool"]].get("vis", 4))
                h_pos, H0_tip, H0_R = hand["pos"], eng["H0_tip"], eng["H0_R"]
                pos_rel = (lambda dRh, h_pos=h_pos, H0_tip=H0_tip, H0_R=H0_R: (dRh @ H0_R) @ tip_m + h_pos - H0_tip)
                t_shape = ts_ref / 1000.0
            else:  # steps / replay: no hand, the target is scripted or recorded
                if not engaged:
                    if robot is None:
                        continue
                    if a.go_file and not Path(a.go_file).exists():  # at the start pose, waiting for the other arm
                        if not go_wait:
                            go_wait = True
                            print(f"[{a.mode}] 준비됨 — {a.go_file} 을 기다린다", flush=True)
                        if trig_sock is not None and time.monotonic() - trig_state["sent_t"] >= 0.1:
                            send_trig(False, None)   # heartbeat before the start: the recording's first rows know btn
                            trig_state["sent_t"] = time.monotonic()
                        continue
                    engaged, q_seed, q_cmd, t_cmd = True, q_enc.copy(), q_enc.copy(), 0.0
                    if tik_solver is not None:
                        tik_solver.set_posture(q_enc)
                    sag_I[:] = 0.0
                    vis_I[:] = 0.0
                    tgt_hist.clear()
                    eng = {"R0_tip": robot["tip"], "R0_R": robot["R"], "FK0": kin.fk(q_enc), "t0": t_now}
                    log.write({"k": "engage", "t": t_now, "fn": fn, "robot_tool": robot["tool"],
                               "R0_tip_mm": (robot["tip"] * 1000).tolist(), "R0_R": robot["R"].tolist(),
                               "FK0": eng["FK0"].tolist(), "q0": q_enc.tolist()})
                    print(f"[{a.mode}] ▶ 시작", flush=True)
                    shaper.reset()
                t_rel = (t_now - eng["t0"]) / 1000.0
                if t_rel >= total_s:
                    break
                ts_ref, vis, pos_rel, t_shape = None, 4, None, t_rel
                if rp is None:
                    seg = next(s for s in segs if s[0] <= t_rel < s[1])
                    dp = seg[2](t_rel) / 1000.0
                    dR = np.eye(3)
                else:  # the recorded hand, relative to its own engage latch
                    if stream_sock is not None:   # the poses the closed-loop policy sent since the last frame
                        while True:
                            try:
                                sm = json.loads(stream_sock.recv(65536))
                            except BlockingIOError:
                                break
                            except ValueError:
                                continue
                            if sm.get("end"):
                                total_s = min(total_s, t_rel)
                                log.write({"k": "stream_end", "t_rel": round(t_rel, 4), "n": n_stream})
                                continue
                            if sm.get("tool") != a.replay_tool:
                                continue
                            ts_ = (float(sm["t_wall"]) - eng["t0"]) / 1000.0
                            if ts_ <= rp["t"][-1]:   # the time axis must rise (sample_pose)
                                ts_ = rp["t"][-1] + 1e-3
                            Rs = rec.quat_to_R(sm["quat"])
                            rp["t"].append(ts_); rp["R"].append(Rs); rp["vis"].append(4)
                            rp["pos"].append(np.asarray(sm["pos"], float))
                            n_stream += 1
                            log.write({"k": "stream", "t_rel": round(ts_, 4), "pos": sm["pos"], "quat": sm["quat"]})
                        if t_rel >= total_s:
                            break
                    # --replay-advance-s: a recording is known ahead — sample it that much later so the arm's lag
                    # is taken up by the input instead of by a velocity lead (which also amplifies noise)
                    Ri, pi, vis = rec.sample_pose(rp, t_rel + a.replay_advance_s, a.replay_interp,
                                                  a.replay_max_gap_s / a.replay_speed)
                    dp = Ri @ tip_m + pi - rp["H0_tip"]
                    dR = np.eye(3) if a.pos_only else Ri @ rp["H0_R"].T
                    pos_rel = (lambda dRh, pi=pi: (dRh @ rp["H0_R"]) @ tip_m + pi - rp["H0_tip"])
                    if trig_sock is not None:  # the trigger the arm is commanded now, on the robot's clock
                        tq = t_rel - a.trig_delay_s
                        k = int(np.searchsorted(trig_t, tq, side="right")) - 1
                        on = bool(trig_v[k]) if (k >= 0 and len(trig_t)) else False
                        now_s = time.monotonic()
                        if on != trig_state["last"]:
                            send_trig(on, t_rel, "change")
                            trig_state["sent_t"] = now_s
                        elif now_s - trig_state["sent_t"] >= 0.1:
                            send_trig(on, t_rel)
                            trig_state["sent_t"] = now_s
            dp_raw = dp.copy()
            dp, dR = shaper(dp, dR, vis, t_shape, pos_rel)
            if ff_W is not None and shaper.x is not None:  # general learned feed-forward (mm features → mm)
                f_ = np.concatenate([shaper.v * 1000, shaper.acc * 1000, (eng["R0_tip"] + shaper.x) * 1000, [1.0]])
                c_ = ff_W @ f_
                cn = float(np.linalg.norm(c_))
                if cn > a.ff_max_mm:
                    c_ *= a.ff_max_mm / cn
                dp = dp + c_ / 1000.0
            if ilc is not None and rp is not None:  # learned correction for this recorded trajectory
                dp = dp + np.array([np.interp(t_rel, ilc["t"], ilc["corr"][:, i]) for i in range(3)])

            # camera outer loop: the arm's own marker vs where it should be, integrated slowly while the target is
            # slow (the camera sees the tip ~18 ms late, so only the steady part is corrected)
            tgt_now = eng["R0_tip"] + dp
            if a.vis_ki > 0 and robot is not None:
                now_v = time.monotonic()
                dt_v = min(now_v - vis_t, 0.05) if vis_t else 0.0
                spd = (float(np.linalg.norm(tgt_now - prev_tgt)) / dt_v) if (prev_tgt is not None and dt_v > 0) else 1e9
                ref_t = now_v - a.vis_delay_ms / 1000.0  # the target the arm should be showing now (its lag)
                ref = next((x for (tt, x) in reversed(tgt_hist) if tt <= ref_t), None) if a.vis_delay_ms > 0 else tgt_now
                if dt_v > 0 and spd * 1000 < a.vis_gate_mms and ref is not None:
                    e_v = ref - robot["tip"]
                    if a.vis_dead_mm > 0:
                        e_n = float(np.linalg.norm(e_v))
                        e_v = e_v * max(0.0, 1.0 - a.vis_dead_mm / 1000.0 / e_n) if e_n > 0 else e_v
                    if a.vis_settle_deg > 0 and q_cmd is not None \
                            and float((np.abs(q_cmd - q_enc) * sag_mask).max()) > a.vis_settle_deg:
                        e_v = np.zeros(3)
                    vis_I = vis_I + a.vis_ki * e_v * dt_v
                    vn = float(np.linalg.norm(vis_I))
                    if vn * 1000 > a.vis_max_mm:
                        vis_I *= a.vis_max_mm / 1000.0 / vn
                vis_t = now_v
            prev_tgt = tgt_now
            tgt_hist.append((time.monotonic(), tgt_now))
            dp = dp + vis_I
            n = float(np.linalg.norm(dp))
            clamped = n * 1000 > a.radius_mm
            if clamped:
                dp = dp * (a.radius_mm / 1000.0 / n)
            FK0 = eng["FK0"]
            if a.align_ori_s > 0 and rp is not None and not a.pos_only:
                # target marker = dR·C·R0 = the hand's absolute orientation once C = H0·R0ᵀ is fully in (camera frame)
                C_v = rotvec(rp["H0_R"] @ eng["R0_R"].T)
                dR = dR @ rotexp(C_v * min(1.0, t_rel / a.align_ori_s))
            T = he.compose(R_bc @ dR @ R_bc.T @ FK0[:3, :3], FK0[:3, 3] + R_bc @ dp)
            T_goal = T.copy()  # the un-leashed target, for logging
            if a.track_leash_mm > 0:  # the command may lead the measured tip by at most the leash
                p_meas = kin.fk(q_enc)[:3, 3]
                gap = T[:3, 3] - p_meas
                gl = float(np.linalg.norm(gap))
                if gl * 1000 > a.track_leash_mm:
                    T[:3, 3] = p_meas + gap * (a.track_leash_mm / 1000.0 / gl)
            if a.seed == "measured":
                q_seed = q_enc.copy()
            tik0 = wall_ms()
            if tik_solver is not None:
                q_ik = tik_solver.ik(q_seed, T, iters=a.ik_iters)
            else:
                q_ik = kin.ik(q_seed, T, iters=a.ik_iters, orientation_weight=a.ori_weight)
            tik = wall_ms()
            # coordinated rate limit: every joint moves the same fraction toward the IK solution, so the tip keeps
            # its path instead of the fast joints finishing first
            if q_cmd is None:
                q_cmd = q_seed.copy()
            dt = min(max((tik - t_cmd) / 1000.0, 0.0), 0.05) if t_cmd else 1 / 90
            dq_ik = q_ik - q_cmd
            big = float(np.abs(dq_ik).max())
            q = q_cmd + dq_ik * min(1.0, a.max_joint_dps * dt / big) if big > 1e-9 else q_cmd.copy()
            Tq = kin.fk(q)
            if np.abs(q - q_enc).max() > a.max_lead_deg:
                disengage("ik_lead")
                continue
            if (Tq @ T_link_from_tip)[2, 3] < Z_FLOOR_M:
                log.write({"k": "skip", "fn": fn, "why": "z_floor"})
                continue
            if clear.mm(q) < a.min_clearance_mm:  # the jaw would come near motor 2 (self-collision)
                log.write({"k": "skip", "fn": fn, "why": "clearance"})
                continue
            # servo sag compensation: once per new encoder reading, integrate (last sent goal − encoder) while the
            # goal is nearly still; the correction rides on top of the goal (the servo is P-only and sags under load)
            if a.sag_ki > 0 and q_cmd is not None and st.get("tick") != sag_tick:
                now_s = time.monotonic()
                dt_s = min(now_s - sag_t, 0.1) if sag_t else 0.0
                goal_speed = float(np.abs(q - q_cmd).max()) / max(dt, 1e-3)
                if dt_s > 0 and goal_speed < a.sag_gate_dps:
                    err = (q_cmd - off) - (q_enc - off)  # desired arm angle (without correction) − measured
                    err = np.where(np.abs(err) > a.sag_dead_deg, err - np.sign(err) * a.sag_dead_deg, 0.0) * sag_mask
                    sag_I = np.clip(sag_I + a.sag_ki * err * dt_s, -a.sag_max_deg, a.sag_max_deg)
                sag_t, sag_tick = now_s, st.get("tick")
            seq += 1
            extra = np.zeros(6)
            if smith is not None or dob is not None:  # feed every new server tick (in order) into the models
                for s_ in list(states.hist):
                    if sm_last_tick is None or s_["tick"] > sm_last_tick:
                        if smith is not None:
                            smith.update(s_)
                        if dob is not None:
                            dob.update(s_)
                        sm_last_tick = s_["tick"]
            if mpc is not None and smith is not None and smith.estimate() is not None:
                r_arm = q - off + sag_I
                now_m = time.monotonic()
                if mpc_rprev is not None and now_m > mpc_t:  # goal velocity (EMA) for the preview
                    v_ = (r_arm - mpc_rprev) / (now_m - mpc_t)
                    mpc_rvel = mpc_rvel + 0.3 * (v_ - mpc_rvel)
                mpc_rprev, mpc_t = r_arm, now_m
                u_prev = mpc_u if mpc_u is not None else r_arm
                u = mpc.command(smith.estimate(), r_arm, mpc_rvel, u_prev)
                mpc_u = u
                extra = np.clip(u - r_arm, -a.overdrive_max_deg, a.overdrive_max_deg)
            elif smith is not None:
                y_hat = smith.estimate()
                if y_hat is not None:
                    r_arm = q - off + sag_I
                    extra = np.clip((sm_k - 1.0) * (r_arm - y_hat), -a.overdrive_max_deg, a.overdrive_max_deg)
            elif np.any(od_k != 1.0):  # q, q_enc are model angles; the difference is the same in arm angles
                q_ref = q_enc
                if a.overdrive_lead_s > 0 and len(states.hist) >= 3:
                    h0, h2 = states.hist[-3], states.hist[-1]
                    dt_h = h2["t"] - h0["t"]
                    if dt_h > 0:
                        v_enc = np.array([(h2["pos"][j] - h0["pos"][j]) / dt_h for j in JOINTS])
                        age = max(0.0, (wall_ms() - h2["trecv"]) / 1000.0)
                        q_ref = q_enc + v_enc * (a.overdrive_lead_s + age)
                extra = np.clip((od_k - 1.0) * (q - q_ref), -a.overdrive_max_deg, a.overdrive_max_deg)

            if a.overdrive_lp_hz > 0:
                now_o = time.monotonic()
                dt_o = min(now_o - od_t, 0.05) if od_t else 0.0
                od_t = now_o
                alpha = 1.0 - np.exp(-2 * np.pi * a.overdrive_lp_hz * dt_o)
                od_lp = od_lp + alpha * (extra - od_lp)
                extra = od_lp
            if dob is not None:
                extra = extra - dob.w
            r = rpc.call(cmd="goal", token=tok, pos={j: float(v) for j, v in zip(JOINTS, q - off + sag_I + extra)},
                         seq=seq)
            tack = wall_ms()
            if not r.get("ok"):
                print(f"[follow] goal 거부: {r}", flush=True)
                disengage(f"goal_{r.get('error')}")
                if a.mode == "steps":
                    break
                continue
            n_goal += 1
            t_goal = time.time()
            q_seed, q_cmd, t_cmd = q_ik, q, tik
            Tik = kin.fk(q_ik)
            target_tip_cam = tgt_now  # intended target (without the camera correction)
            log.write({"k": "cmd", "seq": seq, "fn": fn, "ts_ref": ts_ref, "trecv": m["trecv"], "tik0": tik0,
                       "tik": tik, "tack": tack, "rx": r.get("rx"), "clamped": int(clamped),
                       "dp_mm": np.round(dp * 1000, 3).tolist(), "drot": np.round(rotvec(dR), 6).tolist(),
                       "target_tip_cam_mm": np.round(target_tip_cam * 1000, 3).tolist(),
                       "target_raw_tip_cam_mm": np.round((eng["R0_tip"] + dp_raw) * 1000, 3).tolist(),
                       "t_rel": round(t_rel, 4) if rp is not None else None,
                       "target_base": np.round(T_goal, 6).tolist(), "q_goal": np.round(q, 3).tolist(),
                       "leashed_mm": round(float(np.linalg.norm(T_goal[:3, 3] - T[:3, 3])) * 1000, 2),
                       "q_ik": np.round(q_ik, 3).tolist(), "sag_I": np.round(sag_I, 3).tolist(),
                       "vis_I_mm": np.round(vis_I * 1000, 2).tolist(), "ik_fallbacks": tik_solver.fallbacks if tik_solver else 0,
                       "ik_err_mm": round(float(np.linalg.norm(Tik[:3, 3] - T[:3, 3])) * 1000, 3),
                       "ik_rot_deg": round(he.R_angle_deg(Tik[:3, :3].T @ T[:3, :3]), 3),
                       "rate_lag_mm": round(float(np.linalg.norm(Tq[:3, 3] - T[:3, 3])) * 1000, 3)})
    finally:
        if a.mode == "replay" and a.trig_udp and "send_trig" in locals() and trig_sock is not None:
            send_trig(False, None, "end")
        try:
            rpc.call(cmd="hold", token=tok)
            if a.accel is not None and st0.get("accel") is not None:
                rpc.call(cmd="set", token=tok, accel=int(st0["accel"]))
            rpc.call(cmd="release", token=tok)
        finally:
            time.sleep(0.2)
            states.stop()
            tap.close()
            log.close()
            rpc.close()
    print(f"[follow] 끝 ({stop['why'] or 'done'}) — 프레임 {n_frames}, goal {n_goal} → {run}", flush=True)


if __name__ == "__main__":
    main()
