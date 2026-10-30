#!/usr/bin/env python3
"""Two simulated arms for a dry run of live_loop.py (no robot, no tracker): they take the pose stream endo_follow would take
and publish the tracker tap the robot tools would give.

    python robot/inference/fake_arms.py --run FAKE_RUN --seed-traj seed.json --latency-dir DIR [--lag-s 0.25] [--bias-mm 0.8]
    python robot/inference/live_loop.py --live-run FAKE_RUN --latency-dir DIR --tap-port 9894 --policy-path … (as a real run)

Each arm does what endo_follow's relative replay does (endo_follow.py, mode replay): with H0 = the seed's hand pose in the
recording's camera frame and R0 = the arm's own marker at engage,
    tip target = R0_tip + (R_i·tip_hand + pos_i − H0_tip),   marker rotation target = (R_i·H0_Rᵀ)·R0_R
and follows it with a first-order lag, a constant tip bias and noise (the real arms: ~1.5 mm lateral, lag 0.2–0.3 s).
--ik (10-07): the target first goes through the follower's own IK (pairstart Kin: the tip a hard task, the tool's turn a
soft one against the engage posture, w_ori 0.1 / w_post 0.1) from the run's start joints (the seed's robot_start) — the
real arms make about a tenth of a turn they are asked for, and without this the dry run's arms make all of it: the run
of 10-07 that the safety gate held for 30 min (a 12° endoscope turn asked again every chunk) passed every dry run.
Written on its own (it does not import live_robot.py): the dry run then checks the link's maths against the arm's.
At start it writes <latency-dir>/<run>_<tool>/events.jsonl with the `engage` row and then sends tap messages (90 Hz) for
`endo` and `drill` to --tap-port. R0 = the engage rows of a real run (--r0-from, default the 10-04 run pr_0328i1).
"""
from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np

M = Path.home() / "il-robot/calib/tracker_models"
ROBOT_CALIB = {"endo": M / "tool_calib_endo_robotL_tip.json", "drill": M / "tool_calib_drill_robotR_tip.json"}
PORTS = {"endo": 9891, "drill": 9892}
OFF = {"endo": 0, "drill": 7}


def q2R(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R2q(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z]); return q / np.linalg.norm(q)


def rotvec(R: np.ndarray) -> np.ndarray:
    c = np.clip((np.trace(R) - 1) / 2, -1, 1); th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def rotexp(v: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


class Arm:
    def __init__(self, tool: str, seed: np.ndarray, hdr: dict, r0: dict, lag_s: float, bias_mm: float, rng, q_start=None, r0_model=False,
                 bump=None, wander=None):
        T0 = hdr["registration"]["T0"]
        R0s, t0s = np.asarray(T0["R"], float).reshape(3, 3), np.asarray(T0["t"], float)
        o = OFF[tool]
        self.tool = tool
        self.tl = np.asarray(hdr["toolCalib"][tool]["tipLocal"], float) / 1000.0     # the hand tool's tip (endo_follow --tool-calib)
        self.tip_r = np.asarray(json.loads(ROBOT_CALIB[tool].read_text())["tip_local"], float) / 1000.0
        self.H0_R = R0s.T @ q2R(seed[o + 3:o + 7]); self.H0_tip = R0s.T @ (seed[o:o + 3] - t0s) / 1000.0
        self.R0_tip, self.R0_R = np.asarray(r0["R0_tip_mm"], float) / 1000.0, np.asarray(r0["R0_R"], float)
        self.way = [(0.0, self.H0_tip.copy(), self.H0_R.copy())]      # (wall ms, hand tip, hand R) — the growing recording
        self.tip, self.R = self.R0_tip.copy(), self.R0_R.copy()         # where the arm's marker tip / rotation is now
        self.lag, self.rng = lag_s, rng
        # --bump / --wander-mm (10-12): what the real arms add to a plain lag — see step()
        self.bump, self.wander = bump, wander
        self.z1, self.z2, self.wn, self.u0 = np.zeros(3), np.zeros(3), np.zeros(3), None
        d = rng.normal(size=3); self.bias = d / np.linalg.norm(d) * bias_mm / 1000.0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", PORTS[tool])); self.sock.setblocking(False)
        self.ended = False
        self.kin, self.n_step, self.ik_tgt = None, 0, None
        if q_start is not None:   # --ik: the follower's IK from the run's start joints, posture = the engage posture
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval" / "pairstart"))
            import common as C
            self.kin = C.Kin("left" if tool == "endo" else "right", margin=3.0, w_ori=0.1, w_post=0.1)
            self.q = np.asarray(q_start, float) + self.kin.off
            self.kin.ik.set_posture(self.q); self.FK0 = self.kin.ik.fk(self.q)
            self.Rbc, self.compose = self.kin.Z[:3, :3], C.he.compose
            # the arm's own tool at these joints, in the camera frame: the engage pose of THIS run. (The engage rows of an old
            # run belong to its start pair: on another base the two tools would stand as far apart as that pair had them and the
            # link's start residual c would be the difference of two unrelated runs — 13.7 mm in the first dry run on 92c5.)
            if r0_model:
                Tc = self.kin.T_cam(np.asarray(q_start, float))
                self.R0_tip, self.R0_R = Tc[:3, 3].copy(), Tc[:3, :3].copy()
                self.tip, self.R = self.R0_tip.copy(), self.R0_R.copy()

    def through_ik(self, tgt_tip: np.ndarray, tgt_R: np.ndarray):
        """The marker target the arm's IK makes of (tgt_tip, tgt_R): the tip where asked, the turn as far as the IK turns."""
        dp, dR = tgt_tip - self.R0_tip, tgt_R @ self.R0_R.T                     # since engage, camera frame
        T = self.compose(self.Rbc @ dR @ self.Rbc.T @ self.FK0[:3, :3], self.FK0[:3, 3] + self.Rbc @ dp)
        self.q = self.kin.ik.ik(self.q, T, iters=3)
        Ta = self.kin.ik.fk(self.q)
        return (self.R0_tip + self.Rbc.T @ (Ta[:3, 3] - self.FK0[:3, 3]),
                (self.Rbc.T @ Ta[:3, :3] @ self.FK0[:3, :3].T @ self.Rbc) @ self.R0_R)

    def drain(self) -> None:
        while True:
            try:
                m = json.loads(self.sock.recv(65536))
            except BlockingIOError:
                return
            if m.get("end"):
                self.ended = True; continue
            if m.get("tool") != self.tool:
                continue
            R = q2R(m["quat"]); pos = np.asarray(m["pos"], float)
            self.way.append((float(m["t_wall"]), R @ self.tl + pos, R))

    def target(self, now_ms: float):
        """The streamed hand pose at `now` (linear / held at the ends), as the arm's marker target."""
        w = self.way
        i = max(0, max((k for k in range(len(w)) if w[k][0] <= now_ms), default=0))
        if i >= len(w) - 1 or w[i + 1][0] - w[i][0] > 760:     # endo_follow holds across gaps > 0.25 s of recording / speed
            tip, R = w[i][1], w[i][2]
        else:
            f = (now_ms - w[i][0]) / (w[i + 1][0] - w[i][0])
            tip = w[i][1] + f * (w[i + 1][1] - w[i][1])
            R = w[i][2] @ rotexp(f * rotvec(w[i][2].T @ w[i + 1][2]))
        return self.R0_tip + (tip - self.H0_tip), (R @ self.H0_R.T) @ self.R0_R

    def step(self, now_ms: float, dt: float, stamp_lag_ms: float = 0.0) -> dict:
        self.drain()
        if stamp_lag_ms <= 150.0:   # endo_follow skips frames whose stamp is older than --stale-ms 150: the arm holds
            tgt_tip, tgt_R = self.target(now_ms)
            if self.kin is not None:   # 30 Hz is enough for a target that moves 3 mm/s
                if self.ik_tgt is None or self.n_step % 3 == 0:
                    self.ik_tgt = self.through_ik(tgt_tip, tgt_R)
                tgt_tip, tgt_R = self.ik_tgt
                self.n_step += 1
            a = 1.0 - np.exp(-dt / self.lag)
            moving = len(self.way) > 1
            self.tip = self.tip + a * (tgt_tip + (self.bias if moving else 0.0) - self.tip)
            self.R = self.R @ rotexp(a * rotvec(self.R.T @ tgt_R))
            if self.bump is not None:      # a resonant share of the target's motion: a band-pass around f0 (peak gain 1) times k
                k, f0, q = self.bump; w0 = 2 * np.pi * f0
                self.u0 = tgt_tip.copy() if self.u0 is None else self.u0
                n_sub = max(1, int(np.ceil(dt / 0.012)))                # short steps: a late loop turn must not shake the filter
                for _ in range(n_sub):
                    h = dt / n_sub
                    self.z1 = self.z1 + h * (-w0 / q * self.z1 - w0 * w0 * self.z2 + (tgt_tip - self.u0))
                    self.z2 = self.z2 + h * self.z1
        extra = np.zeros(3)
        if self.bump is not None:
            extra = extra + self.bump[0] * (2 * np.pi * self.bump[1] / self.bump[2]) * self.z1
        if self.wander is not None:        # the standing arm's slow error around its target (an Ornstein-Uhlenbeck walk)
            sd, tau = self.wander
            self.wn = self.wn - self.wn * dt / tau + np.sqrt(2 * dt / tau) * sd / 1000.0 / np.sqrt(3) * self.rng.normal(size=3)
            extra = extra + self.wn
        tip = self.tip + extra + self.rng.normal(scale=0.05e-3, size=3)
        pos = tip - self.R @ self.tip_r
        return {"tool": self.tool, "ts": now_ms - stamp_lag_ms, "pos": (pos * 1000).round(3).tolist(), "quat": R2q(self.R).round(6).tolist(),
                "vis": 4, "tip": (tip * 1000).round(3).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed-traj", required=True, help="live_loop.py --prepare-out (frame 0 = the ready pose, ref_recording)")
    ap.add_argument("--latency-dir", required=True)
    ap.add_argument("--tap-port", type=int, default=9894)
    ap.add_argument("--r0-from", default=str(Path.home() / "il-robot/runs/latency/20261004T115923_replay_pr_0328i1"))
    ap.add_argument("--lag-s", type=float, default=0.25)
    ap.add_argument("--bias-mm", type=float, default=0.8)
    ap.add_argument("--bump", default=None, metavar="K:HZ:Q",
                    help="a resonant share of the target's motion on top of the lag, e.g. 0.3:0.2:1.2 — the real arms answered "
                         "a 0.2 Hz command with a gain of 1.2-1.26 in the first continuous robot run (10-12, "
                         "runs/groot_prep/stream/arm_response.py); a plain lag gives 0.95")
    ap.add_argument("--wander-mm", type=float, default=None, help="the tip's slow error around its target (std, mm): the real "
                    "arms rest 0.9-1.2 mm from their command and that offset moves (the visual servo follows the tracker's slow noise)")
    ap.add_argument("--wander-tau-s", type=float, default=8.0)
    ap.add_argument("--max-s", type=float, default=1800.0)
    ap.add_argument("--stamp-lag", default=None, metavar="T0:T1:MS",
                    help="between T0 and T1 seconds after the start the frame stamps are MS behind the clock and the arms hold "
                         "(10-05: the tracker's stamps drifted behind the host clock and the followers skipped every frame)")
    ap.add_argument("--lose-lease-s", type=float, default=None,
                    help="at this many seconds a `disengage` row (goal_no_lease) goes into both arms' events.jsonl and they stop")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ik", action="store_true", help="the arms' own IK between the stream and the pose (the tool's turn is "
                    "followed softly, as on the robot); start joints = the seed's meta.cycle.robot_start")
    ap.add_argument("--r0", choices=("run", "model"), default="run", help="the arms' marker pose at engage: the engage rows of the "
                    "real run --r0-from (its start pair: right for a seed on that pair), or — with --ik — what the arm model gives "
                    "at this seed's start joints (a seed on another base)")
    a = ap.parse_args()
    tj = json.loads(Path(a.seed_traj).read_text())
    seed = np.asarray(tj["frames"][0], float)
    with open(tj["ref_recording"]) as fh:
        hdr = next(d for d in map(json.loads, fh) if d.get("k") == "hdr")
    reg = (tj.get("meta") or {}).get("registration")   # rollout.py --gate-registration: the frame the link sends its rows in
    if reg and reg.get("T0"):
        hdr = {**hdr, "registration": {**hdr["registration"], "T0": reg["T0"]}}
    rng = np.random.default_rng(a.seed)
    arms = []
    for tool in ("endo", "drill"):
        r0 = next(json.loads(l) for l in open(f"{a.r0_from}_{tool}/events.jsonl") if '"engage"' in l[:40])
        qs = tj["meta"]["cycle"]["robot_start"]["left" if tool == "endo" else "right"] if a.ik else None
        arms.append(Arm(tool, seed, hdr, r0, a.lag_s, a.bias_mm, rng, q_start=qs, r0_model=a.ik and a.r0 == "model",
                        bump=tuple(float(x) for x in a.bump.split(":")) if a.bump else None,
                        wander=(a.wander_mm, a.wander_tau_s) if a.wander_mm else None))
        d = Path(a.latency_dir) / f"{a.run}_{tool}"; d.mkdir(parents=True, exist_ok=True)
        (d / "events.jsonl").write_text(json.dumps({"k": "engage", "t": time.time() * 1000.0, "robot_tool": tool,
                                                    "R0_tip_mm": (arms[-1].R0_tip * 1000.0).tolist(), "R0_R": arms[-1].R0_R.tolist()},
                                                   separators=(",", ":")) + "\n")
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[fake] arms engaged ({a.run}); tap → udp :{a.tap_port}" + (" — with the arms' IK" if a.ik else ""), flush=True)
    lag0, lag1, lag_ms = (float(v) for v in a.stamp_lag.split(":")) if a.stamp_lag else (0.0, 0.0, 0.0)
    t_start, lost = time.time(), False
    t_end, fn, prev = time.time() + a.max_s, 0, time.time()
    while time.time() < t_end and not all(x.ended for x in arms):
        now = time.time(); dt, prev = now - prev, now
        if a.lose_lease_s is not None and not lost and now - t_start >= a.lose_lease_s:
            lost = True
            for x in arms:
                with open(Path(a.latency_dir) / f"{a.run}_{x.tool}" / "events.jsonl", "a") as fh:
                    fh.write(json.dumps({"k": "disengage", "t": now * 1000.0, "reason": "goal_no_lease"}, separators=(",", ":")) + "\n")
            print("[fake] lease lost: the arms hold from here", flush=True)
        lag = 1e9 if lost else (lag_ms if lag0 <= now - t_start < lag1 else 0.0)
        for x in arms:
            m = x.step(now * 1000.0, max(dt, 1e-3), lag)
            if lost:
                m["ts"] = now * 1000.0
            m["fn"] = fn
            out.sendto(json.dumps(m).encode(), ("127.0.0.1", a.tap_port))
        fn += 1
        time.sleep(max(0.0, 1 / 90 - (time.time() - now)))
    print(f"[fake] ended: {[len(x.way) - 1 for x in arms]} streamed poses (endo, drill)", flush=True)


if __name__ == "__main__":
    main()
