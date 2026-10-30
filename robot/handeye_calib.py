#!/usr/bin/env python3
"""Automatic eye-to-hand calibration of one arm against the tracker camera.

The arm visits random joint-space poses around where it stands now; at each it settles, then averages the
arm-carried endoscope constellation (tracker `--tap` stream, raw pose) and the encoder angles. Solves
T_base←cam and T_link←const (lerobot_robot_spinearm.handeye), reports fit and hold-out residuals, saves JSON.

Keep the hand-held endoscope out of the camera's view while this runs — with two constellations visible the
pose is skipped (both are the same model, and nothing is calibrated yet to tell them apart).

  python robot/handeye_calib.py --arm left                      # ≈ 1–2 min, arm moves ±8–20° per joint
  python robot/handeye_calib.py --resolve ~/il-robot/runs/handeye/<ts>/samples.json   # re-solve, no motion
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from lerobot_robot_spinearm import handeye as he
from lerobot_robot_spinearm.client import ArmRpc
from lerobot_robot_spinearm.ee import MIN_CLEARANCE_MM, Z_FLOOR_M, Clearance, MarkerKinematics
from lerobot_robot_spinearm.protocol import DEFAULT_PORTS, JOINTS
from lerobot_robot_spinearm.tap import ENDO_TOOLS, FrameAssembler, TapReceiver

TRACKING = Path.home() / "il-robot/tracking"
TOOL_CALIB = TRACKING / "tool_calib.json"
# Marker per arm: the left arm carries the endoscope constellation (block A, tracked as endo/endo2), the right arm
# the drill constellation (block B, tracked as drill).
ARM_TOOLS = {"left": (("endo", "endo2"), TOOL_CALIB), "right": (("drill",), TRACKING / "tool_calib_drill_robotR.json")}  # the right arm's own block B, not the hand drill
AMP_DEG = {"shoulder_pan": 10, "shoulder_lift": 8, "elbow_flex": 8, "wrist_flex": 15, "wrist_roll": 20, "gripper": 15}


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="left", choices=["left", "right"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--cmd-port", type=int, default=None, help="default: the arm's standard port")
    ap.add_argument("--pub-port", type=int, default=None)
    ap.add_argument("--tap-port", type=int, default=9880)
    ap.add_argument("--tool-calib", default=None, help="default: the arm's marker (ARM_TOOLS)")
    ap.add_argument("--tools", default=None, help="tracker tool names of the arm's marker, comma-separated "
                                                   "(default: endo,endo2 for left, drill for right)")
    ap.add_argument("--n-fit", type=int, default=25)
    ap.add_argument("--n-val", type=int, default=8)
    ap.add_argument("--amp-scale", type=float, default=1.0, help="× per-joint amplitude (AMP_DEG)")
    ap.add_argument("--max-travel-mm", type=float, default=70.0, help="marker within this of the start pose")
    ap.add_argument("--step-deg", type=float, default=8.0,
                    help="server per-tick cap: how far ahead of the measured angle the servo is pushed (gravity sag)")
    ap.add_argument("--speed-dps", type=float, default=30.0, help="client-side goal ramp speed per joint")
    ap.add_argument("--start", default=None,
                    help="move here first, e.g. 'shoulder_lift=-55,elbow_flex=20,wrist_flex=90' (others unchanged)")
    ap.add_argument("--settle-s", type=float, default=0.6)
    ap.add_argument("--sample-s", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default ~/il-robot/calib/handeye_<arm>.json")
    ap.add_argument("--resolve", default=None, help="samples.json from an earlier run: solve only")
    ap.add_argument("--no-offsets", action="store_true", help="do not fit joint zero offsets")
    a = ap.parse_args()
    tools, calib = ARM_TOOLS[a.arm]
    a.tools = tuple(a.tools.split(",")) if a.tools else tools
    a.tool_calib = a.tool_calib or str(calib)
    return a


class TapThread:
    """Assembled endoscope frames (both constellations per camera frame) in a ring, with arrival order."""

    def __init__(self, port, tools=ENDO_TOOLS):
        self.rx = TapReceiver(port)
        self.asm = FrameAssembler(tools)
        self.frames: deque = deque(maxlen=4000)
        self._stop = threading.Event()
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _loop(self):
        while not self._stop.is_set():
            m = self.rx.recv(0.2)
            if m is None:
                continue
            f = self.asm.add(m)
            if f is not None:
                self.frames.append(f)

    def since(self, t_wall_ms):
        return [f for f in list(self.frames) if min(m.get("ts") or 0 for m in f.values()) >= t_wall_ms]

    def stop(self):
        self._stop.set()
        self.th.join(1.0)
        self.rx.close()


def mean_pose(ms: list[dict]) -> tuple[np.ndarray, np.ndarray, float]:
    """Average raw poses (mm, wxyz) → (4×4 metres, pos std mm)."""
    P = np.array([m["raw_pos"] for m in ms], float)
    Q = np.array([m["raw_quat"] for m in ms], float)
    Q[(Q @ Q[0]) < 0] *= -1
    q = Q.mean(0)
    T = he.compose(he.quat_to_R(q / np.linalg.norm(q)), P.mean(0) / 1000.0)
    return T, float(np.linalg.norm(P.std(0)))


def gen_poses(rng, q0, bounds, kin, n, amp_scale, max_travel_m, clear=None):
    p0 = kin.xyz(q0)
    out, tries = [], 0
    while len(out) < n and tries < 5000:
        tries += 1
        q = q0.copy()
        for i, j in enumerate(JOINTS):
            a = AMP_DEG[j] * amp_scale
            q[i] = np.clip(q0[i] + rng.uniform(-a, a), bounds[j][0] + 3, bounds[j][1] - 3)
        T = kin.fk(q)
        if T[2, 3] < Z_FLOOR_M or np.linalg.norm(T[:3, 3] - p0) > max_travel_m:
            continue
        if clear is not None and clear.mm(q) < MIN_CLEARANCE_MM:
            continue
        out.append(q)
    return out


def solve_and_report(samples, pts_m, tip_m, n_fit, offsets: bool = True):
    """Fit on the first n_fit samples, report the rest as hold-out. With offsets: joint zero offsets too (FK from
    the encoder angles + dq), which is what the arm's mid-range joint calibration needs (elbow ≈ −8°)."""
    kin = MarkerKinematics(bounds=None)
    q = [np.asarray(s["q_enc"], float) for s in samples]
    obs = [np.asarray(s["obs"]) for s in samples]
    fit = list(range(min(n_fit, len(samples))))
    val = list(range(len(fit), len(samples)))
    s = he.solve([kin.fk(q[i]) for i in fit], [obs[i] for i in fit], pts_m, tip_m,
                 q_deg=[q[i] for i in fit] if offsets else None, fk_fn=kin.fk if offsets else None)
    X, Z, dq = s["X"], s["Z"], s["dq"]
    fk = [kin.fk(qi + dq) for qi in q]
    kept = [fit[i] for i in s["keep"]]
    r_fit = he.residuals(X, Z, [fk[i] for i in kept], [obs[i] for i in kept], pts_m, tip_m)
    r_val = he.residuals(X, Z, [fk[i] for i in val], [obs[i] for i in val], pts_m, tip_m)
    rej = [fit[i] for i in range(len(fit)) if i not in s["keep"]]
    return X, Z, dq, he.summarize(r_fit), he.summarize(r_val), rej


def show(label, summ):
    if not summ.get("n"):
        print(f"  {label}: 없음")
        return
    print(f"  {label} (n={summ['n']}): 팁 RMS {summ['tip_mm']['rms']:.2f} mm (최대 {summ['tip_mm']['max']:.2f}) · "
          f"별자리 원점 {summ['org_mm']['rms']:.2f} mm · 회전 {summ['rot_deg']['rms']:.2f}° · "
          f"공 RMS {summ['ball_rms_mm']['rms']:.2f} mm")


def main():
    a = parse()
    pts_m, tip_m = he.tool_calib(a.tool_calib)
    out = Path(a.out or Path.home() / f"il-robot/calib/handeye_{a.arm}.json")
    if a.resolve:
        d = json.loads(Path(a.resolve).read_text())
        X, Z, dq, sf, sv, rej = solve_and_report(d["samples"], pts_m, tip_m, d.get("n_fit", a.n_fit), not a.no_offsets)
        show("맞춤", sf)
        show("검증", sv)
        print(f"  관절 영점 보정(°): {dict(zip(JOINTS, np.round(dq, 2).tolist()))}  제외 자세 {rej}")
        he.save(out, X, Z, {"arm": d.get("arm", a.arm), "frame": "marker_frame_link", "tool_calib": a.tool_calib,
                            "joint_offset_deg": dict(zip(JOINTS, np.round(dq, 4).tolist())),
                            "fit": sf, "val": sv, "rejected": rej, "samples": str(a.resolve)})
        print(f"저장: {out}")
        return

    cmd_port = a.cmd_port or DEFAULT_PORTS[a.arm][0]
    rpc = ArmRpc(a.host, cmd_port)
    st = rpc.status()
    bounds = {j: tuple(v) for j, v in st["bounds"].items()}
    kin = MarkerKinematics()
    kin.set_limits(bounds)
    tap = TapThread(a.tap_port, a.tools)
    time.sleep(0.5)
    if not tap.frames:
        sys.exit("트래커 탭(--tap 127.0.0.1:%d) 메시지가 없다 — ube-tracker 확인" % a.tap_port)

    tok = rpc.lease("handeye")
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    run_dir = Path.home() / "il-robot/runs/handeye" / time.strftime("%Y%m%dT%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    samples, skipped = [], []
    try:
        if rpc.state()["state"] == "OFF":
            rpc.must(cmd="enable", token=tok)
        rpc.must(cmd="set", token=tok, step_deg=a.step_deg)
        def goto(q, timeout_s=4.0, settle_s=0.0, max_sag_deg=15.0):
            """Ramp the goal to q at speed_dps, then wait until within 1° or settled (< max_sag_deg): under gravity
            the soft servos stop short (elbow 2.1° folded, shoulder_lift 8–13° stretched out, 2026-09-23). The fit
            uses the *encoder* angles, so a sagged pose is still a good sample. Jaw-on-motor-2 contact is excluded
            geometrically (Clearance) before a pose is tried."""
            q_from = np.array([rpc.state()["pos"][j] for j in JOINTS])
            n_ramp = max(1, int(np.ceil(np.abs(q - q_from).max() / (a.speed_dps / 30.0))))
            for i in range(1, n_ramp + 1):  # gentle: the goal walks at speed_dps, the server may push step_deg ahead
                if stop["flag"]:
                    return False
                qi = q_from + (q - q_from) * i / n_ramp
                rpc.must(cmd="goal", token=tok, pos={j: float(v) for j, v in zip(JOINTS, qi)})
                time.sleep(1 / 30)
            goal = {j: float(v) for j, v in zip(JOINTS, q)}
            t0, reached = time.time(), None
            hist: deque = deque(maxlen=10)
            while not stop["flag"]:
                rpc.must(cmd="goal", token=tok, pos=goal)
                pos = np.array([rpc.state()["pos"][j] for j in JOINTS])
                hist.append(pos)
                err = np.abs(pos - q).max()
                still = len(hist) == hist.maxlen and np.ptp(np.array(hist), axis=0).max() < 0.3
                if reached is None and (err < 1.0 or (still and err < max_sag_deg)):
                    reached = time.time()
                if reached is None and still and err >= max_sag_deg:
                    print(f"  ★ 멈춤(오차 {err:.1f}°, {JOINTS[int(np.argmax(np.abs(pos - q)))]}) — 건너뜀", flush=True)
                    return False
                if reached is not None and time.time() - reached >= settle_s:
                    return True
                if time.time() - t0 > timeout_s:
                    return False
                time.sleep(1 / 30)
            return False

        q0 = np.array([rpc.state()["pos"][j] for j in JOINTS])
        clear = Clearance()
        if a.start:
            q_start = q0.copy()
            for kv in a.start.split(","):
                j, v = kv.split("=")
                q_start[JOINTS.index(j.strip())] = float(v)
            if clear.mm(q_start) < MIN_CLEARANCE_MM:
                sys.exit(f"--start 자세의 집게–2번 모터 거리 {clear.mm(q_start):.0f} mm < {MIN_CLEARANCE_MM:.0f}")
            print(f"[handeye] 시작 자세로 이동 {np.round(q_start, 1).tolist()}", flush=True)
            goto(q_start, timeout_s=12.0, settle_s=0.5)
            q0 = np.array([rpc.state()["pos"][j] for j in JOINTS])
        c0 = clear.mm(q0)
        if c0 < MIN_CLEARANCE_MM:
            sys.exit(f"시작 자세가 접혀 있다 — 집게와 2번 모터 사이 {c0:.0f} mm (< {MIN_CLEARANCE_MM:.0f}). "
                     "팔을 펴서(예: lift −60°, elbow 40°) 마커가 카메라를 향하게 한 뒤 다시")
        poses = gen_poses(np.random.default_rng(a.seed), q0, bounds, kin, a.n_fit + a.n_val, a.amp_scale,
                          a.max_travel_mm / 1000.0, clear)
        print(f"[handeye] {len(poses)} 자세, 시작 q0={np.round(q0, 1).tolist()}", flush=True)

        for k, q in enumerate(poses):
            if stop["flag"]:
                break
            if not goto(q, settle_s=a.settle_s):
                skipped.append((k, "not_reached"))
                goto(q0, timeout_s=4.0)  # back to the start before the next pose
                continue
            for attempt in range(3):
                t_start = time.time() * 1000.0
                encs = []
                while time.time() * 1000.0 - t_start < a.sample_s * 1000.0:
                    rpc.must(cmd="goal", token=tok, pos={j: float(v) for j, v in zip(JOINTS, q)})
                    encs.append([rpc.state()["pos"][j] for j in JOINTS])
                    time.sleep(1 / 30)
                frames = tap.since(t_start)
                vis = [[m for m in f.values() if not m.get("lost")] for f in frames]
                two = sum(1 for v in vis if len(v) >= 2)
                one = [v[0] for v in vis if len(v) == 1]
                if two > len(frames) * 0.2:
                    print("  ★ 별자리 두 개가 보인다 — 손 내시경을 시야 밖으로 (재시도)", flush=True)
                    time.sleep(1.5)
                    continue
                break
            if len(one) < max(5, 0.6 * len(frames)):
                skipped.append((k, f"marker_lost {len(one)}/{len(frames)}"))
                print(f"  [{k + 1}/{len(poses)}] 마커 안 보임 — 건너뜀", flush=True)
                continue
            T_obs, sd = mean_pose(one)
            if sd > 0.5:
                skipped.append((k, f"unsteady {sd:.2f}mm"))
                continue
            q_enc = np.mean(np.array(encs), 0)
            samples.append({"k": k, "q_goal": q.tolist(), "q_enc": q_enc.tolist(), "fk": kin.fk(q_enc).tolist(),
                            "obs": T_obs.tolist(), "n_frames": len(one), "pos_sd_mm": round(sd, 3)})
            print(f"  [{k + 1}/{len(poses)}] 수집 {len(samples)} (프레임 {len(one)}, 흔들림 {sd:.2f} mm)", flush=True)
        goto(q0, timeout_s=6.0)
    finally:
        try:
            rpc.call(cmd="hold", token=tok)
            rpc.call(cmd="release", token=tok)
        finally:
            tap.stop()
    (run_dir / "samples.json").write_text(json.dumps({"arm": a.arm, "n_fit": a.n_fit, "samples": samples,
                                                      "skipped": skipped, "tool_calib": a.tool_calib}, indent=1))
    print(f"[handeye] 수집 {len(samples)} · 건너뜀 {len(skipped)} → {run_dir / 'samples.json'}", flush=True)
    if len(samples) < 10:
        sys.exit("자세가 10개 미만 — 시작 자세(마커가 카메라를 향하게)를 바꿔 다시")
    n_fit = min(a.n_fit, len(samples) - min(a.n_val, max(0, len(samples) - 10)))
    X, Z, dq, sf, sv, rej = solve_and_report(samples, pts_m, tip_m, n_fit, not a.no_offsets)
    show("맞춤", sf)
    show("검증", sv)
    print(f"  관절 영점 보정(°): {dict(zip(JOINTS, np.round(dq, 2).tolist()))}  제외 자세 {rej}")
    he.save(out, X, Z, {"arm": a.arm, "frame": "marker_frame_link", "tool_calib": a.tool_calib, "fit": sf,
                        "joint_offset_deg": dict(zip(JOINTS, np.round(dq, 4).tolist())),
                        "val": sv, "rejected": rej, "samples": str(run_dir / "samples.json")})
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
