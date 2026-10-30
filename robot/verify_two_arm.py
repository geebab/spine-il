#!/usr/bin/env python3
"""Move each arm through its safe zone and compare the viewer's model with the tracker (2026-09-26).

  verify_two_arm.py plan    [--n 14]     # per arm: targets in the optimal zone + checked moves → runs/verify/<ts>/plan.json
  verify_two_arm.py run     <run_dir>    # left, then right: torque on, slow moves, record at each stop, home, torque off
  verify_two_arm.py analyze <run_dir>    # model vs measured balls per stop

One arm moves at a time; the other rests at its start pose, torque off, and is an obstacle in every check (the arms'
workspaces overlap: no point of a move may bring them closer than `target_arms_mm` at a stop or than at rest on the
way). Zones (lerobot_robot_spinearm.safezone): safe = all 4 balls of the moving arm's marker seen by both IR imagers
and no part of either model (marker block, balls, rods) near the floor, the camera, the other arm or itself — every
move is a straight line in joint space checked point by point, so the way back is safe too; optimal = safe with all 4
balls 35–45 cm from the camera. Stops are optimal, moves stay safe. While moving, a marker lost for longer than
--lost-s sends the arm back along the same line; an arm lagging its goal by more than --lag-deg holds everything.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "lerobot_robot_spinearm"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros2_ws/src/spinearm_bridge"))
from lerobot_robot_spinearm import safezone as sz  # noqa: E402
from spinearm_bridge.visual_urdf import build_for_arm  # noqa: E402

CALIB = Path.home() / "il-robot/calib"
RUNS = Path.home() / "il-robot/runs/verify"
KIN = Path(__file__).resolve().parent / "lerobot_robot_spinearm/lerobot_robot_spinearm/assets/so101_marker.urdf"
PORTS = {"left": 5601, "right": 5611}
TOOLS = {"left": ("endo", "endo2"), "right": ("drill",)}
TAP_PORT = 9880
FRAME_URL = "http://127.0.0.1:8089/frame"      # the tracker's live IR view (detections drawn): kept at every stop
# where targets are drawn from (arm degrees); gripper = the range each arm used in its hand-eye calibration
RANGES = {"shoulder_pan": (-55, 55), "shoulder_lift": (-100, 25), "elbow_flex": (-45, 96), "wrist_flex": (-100, 100),
          "wrist_roll": (-90, 90)}
GRIPPER = {"left": (-62.0, -44.0), "right": (-55.0, -26.0)}


def calib_kind() -> str:
    """The viewer's calibration: the as-built fit when it exists (fit_asbuilt.py), else the hand-eye one."""
    return "asbuilt" if all((CALIB / f"asbuilt_{a}.json").exists() for a in PORTS) else "handeye"


def load_models(step_m: float = 0.006, kind: str | None = None) -> tuple[dict, sz.Camera, dict]:
    kind = kind or calib_kind()
    arms, info = {}, {}
    Z = {a: np.array(json.loads((CALIB / f"{kind}_{a}.json").read_text())["T_base_cam"]) for a in PORTS}
    for a in PORTS:
        hd = json.loads((CALIB / f"{kind}_{a}.json").read_text())
        urdf, rep = build_for_arm(a, KIN, sz.SHARE / "urdf/so101_new_calib.urdf", CALIB / f"{kind}_{a}.json",
                                  CALIB / "markers.json", sz.SHARE / "meshes/markers", link_prefix=f"{a}_")
        if rep.get("marker") is None:  # no marker-block mesh → the block's collision volume silently disappears
            raise SystemExit(f"[safety] {a}: marker block missing from the model ({rep.get('why')}). "
                             f"Model dir: {sz.SHARE}. Set SPINEARM_DESCRIPTION to a complete spinearm_description.")
        T_wb = np.eye(4) if a == "left" else Z["left"] @ np.linalg.inv(Z[a])
        arms[a] = sz.Arm(a, urdf, hd["joint_offset_deg"], T_wb, step_m)
        info[a] = {"model": rep, "offsets": hd["joint_offset_deg"], "T_world_base": T_wb.tolist(),
                   "T_base_cam": Z[a].tolist(), "calib": kind}
    return arms, sz.Camera(Z["left"]), info


def rpc(arm: str):
    from lerobot_robot_spinearm.client import ArmRpc
    return ArmRpc("127.0.0.1", PORTS[arm])


def read_q() -> dict[str, np.ndarray]:
    out = {}
    for a in PORTS:
        st = rpc(a).state()
        out[a] = np.array([st["pos"][j] for j in sz.JOINTS])
    return out


# ── plan ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def sample_targets(zone: sz.Zone, arm: str, bnd: dict, n_try: int, rng, keep: int) -> list[np.ndarray]:
    lim, cam, model = zone.lim, zone.cam, zone.arms[arm]
    out = []
    for _ in range(n_try):
        q = np.array([rng.uniform(max(RANGES[j][0], bnd[j][0] + 3), min(RANGES[j][1], bnd[j][1] - 3))
                      for j in sz.JOINTS[:5]] + [rng.uniform(*GRIPPER[arm])])
        B, n = model.balls(q)                                   # cheap screen before the full model
        z = cam.cam(B)[:, 2] * 1000
        if z.min() < lim.optimal_z_mm[0] or z.max() > lim.optimal_z_mm[1]:
            continue
        to_cam = cam.centres()[0] - B.mean(0)
        if math.degrees(math.acos(float(n @ to_cam / np.linalg.norm(to_cam)))) > lim.target_view_deg:
            continue
        px = cam.pixels(cam.cam(B))
        u0, u1, v0, v1 = lim.window_px
        if px[:, 0].min() < u0 or px[:, 0].max() > u1 or px[:, 2].min() < v0 or px[:, 2].max() > v1:
            continue
        if zone.single(arm, q, strict=True, optimal=True)[0]:
            out.append(q)
            if len(out) >= keep:
                break
    return out


def plan_arm(zone: sz.Zone, arm: str, start: dict, bnd: dict, a, rng) -> list[tuple[str, dict, dict]]:
    """Stops for one arm (the other resting at start): hub, targets (direct moves, or round by the hub), start."""
    other = next(x for x in start if x != arm)
    see = (arm,)
    t0 = time.time()
    cand = []
    for q in sample_targets(zone, arm, bnd, a.tries, rng, a.pool):
        qq = {arm: q, other: start[other]}
        ok, why, m = zone.check(qq, optimal=(arm,), strict=True, see=see)
        if ok:
            cand.append((qq, m))
    print(f"[plan] {arm}: {len(cand)} optimal targets with the {other} arm resting ({time.time() - t0:.0f} s)",
          flush=True)
    if not cand:
        sys.exit(f"{arm}: no target in the optimal zone — widen RANGES or the band")

    def central(m):
        return m[arm]["window_px"] / 50 - m[arm]["view_deg"] / 32 + min(m["arms_mm"], 120) / 120
    hub = next((k for k in sorted(range(len(cand)), key=lambda k: -central(cand[k][1]))[:10]
                if zone.path(start, cand[k][0], a.step_deg, see)[0]), None)
    if hub is None:
        sys.exit(f"{arm}: no target reachable from the start pose inside the safe zone")
    # grow a tour from the hub: each next target is the most different one (joint space) that a safe move reaches —
    # straight from here, or round by the hub — so every chosen target is connected
    feat = np.array([c[0][arm][:5] for c in cand])
    via = {}

    def hub_ok(k):
        if k not in via:
            via[k] = zone.path(cand[hub][0], cand[k][0], a.step_deg, see)[0]
        return via[k]
    seq, cur, chosen = [("hub", hub)], hub, [hub]
    while len(chosen) <= a.n:
        left_ = [k for k in range(len(cand)) if k not in chosen]
        if not left_:
            break
        d = np.min(np.linalg.norm(feat[left_][:, None, :] - feat[chosen][None, :, :], axis=2), axis=1)
        order = [left_[i] for i in np.argsort(-d)][:8]
        nxt = next((k for k in order if zone.path(cand[cur][0], cand[k][0], a.step_deg, see)[0]), None)
        if nxt is None and (cur == hub or hub_ok(cur)):
            nxt = next((k for k in order if hub_ok(k)), None)
            if nxt is not None and cur != hub:
                seq.append(("hub", hub))
        if nxt is None:
            print(f"[plan] {arm}: no further target reachable ({len(chosen) - 1} chosen)", flush=True)
            break
        seq.append(("target", nxt))
        chosen.append(nxt)
        cur = nxt
    if not zone.path(cand[cur][0], start, a.step_deg, see)[0]:
        seq.append(("hub", hub))
    print(f"[plan] {arm}: {sum(1 for s in seq if s[0] == 'target')} targets, {len(seq) + 1} stops "
          f"({time.time() - t0:.0f} s)", flush=True)
    return [(kind, cand[k][0], cand[k][1]) for kind, k in seq] + [("home", start, {})]


def plan(a) -> Path:
    rng = np.random.default_rng(a.seed)
    arms, cam, info = load_models()
    zone = sz.Zone(arms, cam)
    start = read_q() if a.start is None else {k: np.array(v) for k, v in json.loads(a.start).items()}
    zone.set_start(start)
    ok, why, _ = zone.check(start)
    if not ok:
        sys.exit(f"start pose is not safe: {why}")
    bnd = {x: rpc(x).status()["bounds"] for x in PORTS}
    phases = [{"arm": arm, "stops": [{"kind": kind, "q": {x: np.asarray(v).tolist() for x, v in q.items()},
                                      "check": m} for kind, q, m in plan_arm(zone, arm, start, bnd[arm], a, rng)]}
              for arm in a.arms.split(",")]
    run_dir = RUNS / time.strftime("%Y%m%dT%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    out = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "start": {k: v.tolist() for k, v in start.items()},
           "phases": phases, "limits": zone.lim.__dict__, "target_arms_mm": zone.target_arms_mm,
           "step_deg": a.step_deg, "relaxed": {"self": {x: {f"{p[0]}-{p[1]}": v for p, v in d.items()}
                                                         for x, d in zone.self_floor.items()},
                                               "floor": zone.floor_floor, "arms_mm": zone.arms_floor},
           "models": info, "camera": {"T_world_cam": cam.T_world_cam.tolist()}}
    (run_dir / "plan.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"[plan] → {run_dir / 'plan.json'}")
    return run_dir


# ── run ───────────────────────────────────────────────────────────────────────────────────────────────────────────
class Tap:
    """Tracker tap (UDP): every message logged with its arrival time; the last tracked time per tool."""

    def __init__(self, port: int, log):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.bind(("127.0.0.1", port))
        self.s.settimeout(0.2)
        self.log, self.seen, self.lock = log, {}, threading.Lock()
        self.stop = threading.Event()
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _loop(self):
        while not self.stop.is_set():
            try:
                m = json.loads(self.s.recv(65536))
            except (socket.timeout, ValueError):
                continue
            t = time.time()
            if not m.get("lost"):
                self.seen[m.get("tool")] = t
            with self.lock:
                self.log.write(json.dumps({"k": "tap", "trecv": round(t * 1000, 3), **m}) + "\n")

    def age(self, tools) -> float:
        return time.time() - max((self.seen.get(t, 0.0) for t in tools), default=0.0)

    def close(self):
        self.stop.set()
        self.th.join(1.0)
        self.s.close()


def run(a) -> None:
    run_dir = Path(a.run_dir)
    P = json.loads((run_dir / a.plan).read_text())
    start = {k: np.array(v) for k, v in P["start"].items()}
    now = read_q()
    for x in start:
        if np.abs(now[x] - start[x])[:5].max() > 3.0:
            sys.exit(f"{x} is not where the plan starts ({np.round(now[x], 1).tolist()} vs "
                     f"{np.round(start[x], 1).tolist()}) — plan again")
    log = open(run_dir / (f"events_{a.tag}.jsonl" if a.tag else "events.jsonl"), "a")
    tap = Tap(TAP_PORT, log)
    time.sleep(0.5)
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    R = {x: rpc(x) for x in PORTS}

    def event(**kw):
        with tap.lock:
            log.write(json.dumps({"k": "event", "t": round(time.time() * 1000, 3), **kw}, default=float) + "\n")
            log.flush()

    def state(x):
        st = R[x].state()
        return np.array([st["pos"][j] for j in sz.JOINTS])

    def enc_log(goal: dict):
        with tap.lock:
            for x in R:
                log.write(json.dumps({"k": "enc", "arm": x, "t": round(time.time() * 1000, 3),
                                      "pos": state(x).round(3).tolist(),
                                      "goal": None if x not in goal else np.round(goal[x], 3).tolist()}) + "\n")

    def move(arm, tok, q_from, q_to, check_marker=True) -> str:
        """Stream the straight line q_from → q_to at --speed; 'ok' | 'lost' | 'lag' | 'stop'."""
        n = max(1, int(math.ceil(float(np.abs(q_to - q_from).max()) / (a.speed / 30.0))))
        lag_since = None
        for i in range(1, n + 1):
            if stop["flag"]:
                return "stop"
            qi = q_from + (q_to - q_from) * i / n
            R[arm].must(cmd="goal", token=tok, pos={j: float(v) for j, v in zip(sz.JOINTS, qi)})
            enc_log({arm: qi})
            lag = float(np.abs(state(arm) - qi)[:5].max())
            lag_since = (lag_since or time.time()) if lag > a.lag_deg else None
            if lag_since and time.time() - lag_since > 0.5:
                event(what="lag", arm=arm, lag_deg=lag)
                return "lag"
            if check_marker and tap.age(TOOLS[arm]) > a.lost_s:
                event(what="lost", arm=arm, at=i / n)
                return "lost"
            time.sleep(1 / 30)
        return "ok"

    def dwell(arm, tok, q, record: bool, label: dict):
        """Hold q until the encoders settle, then record --rec-s of taps + encoders."""
        hist, t0 = [], time.time()
        while not stop["flag"] and time.time() - t0 < 4.0:
            R[arm].must(cmd="goal", token=tok, pos={j: float(v) for j, v in zip(sz.JOINTS, q)})
            enc_log({arm: q})
            hist.append(state(arm))
            if len(hist) >= 15 and np.ptp(np.array(hist[-15:]), axis=0).max() < 0.3:
                break
            time.sleep(1 / 30)
        t_settle = time.time()
        while not stop["flag"] and time.time() - t_settle < 0.5 + (a.rec_s if record else 0.0):
            R[arm].must(cmd="goal", token=tok, pos={j: float(v) for j, v in zip(sz.JOINTS, q)})
            enc_log({arm: q})
            time.sleep(1 / 30)
        if record:
            event(what="record", arm=arm, t0=round((t_settle + 0.5) * 1000, 3), t1=round(time.time() * 1000, 3),
                  **label)
            snap(f"{arm}_stop{label['stop']:02d}")

    def torque_off(arm, tok) -> dict:
        """Stop streaming → HOLD first (torque_off is refused while MOVING), then off."""
        R[arm].call(cmd="hold", token=tok)
        t_h = time.time()
        while time.time() - t_h < 2.0 and R[arm].state()["state"] not in ("HOLD", "OFF"):
            time.sleep(0.05)
        return R[arm].call(cmd="torque_off", token=tok)

    def snap(name):
        import urllib.request
        try:
            (run_dir / "ir").mkdir(exist_ok=True)
            (run_dir / "ir" / f"{a.tag + '_' if a.tag else ''}{name}.jpg").write_bytes(
                urllib.request.urlopen(FRAME_URL, timeout=2).read())
        except OSError:
            pass

    arms_m, cam, _ = load_models()                              # for re-checking a move after a skipped stop
    zone = sz.Zone(arms_m, cam)
    zone.set_start(start)

    def reach(arm, cur, q, hub):
        """Safe way from cur to q with the other arm resting: [q], [hub, q] or None."""
        other = next(x for x in start if x != arm)
        both = (lambda v: {arm: np.asarray(v), other: start[other]})
        if zone.path(both(cur), both(q), P["step_deg"], (arm,))[0]:
            return [q]
        if hub is not None and zone.path(both(cur), both(hub), P["step_deg"], (arm,))[0] and \
                zone.path(both(hub), both(q), P["step_deg"], (arm,))[0]:
            return [hub, q]
        return None

    result, active = "ok", None
    try:
        for ph in P["phases"]:
            arm, stops = ph["arm"], ph["stops"]
            active = arm
            hub = next((np.array(s["q"][arm]) for s in stops if s["kind"] == "hub"), None)
            tok = R[arm].lease("verify")
            if R[arm].state()["state"] == "OFF":
                R[arm].must(cmd="enable", token=tok)
            R[arm].must(cmd="set", token=tok, step_deg=a.guard_step)
            event(what="phase", arm=arm)
            cur = state(arm)
            print(f"[run] {arm}: {len(stops)} stops", flush=True)
            k, detour = 0, []
            while k < len(stops) and result == "ok":
                s = stops[k]
                q = detour.pop(0) if detour else np.array(s["q"][arm])
                t_move = time.time()
                r = move(arm, tok, cur, q, check_marker=s["kind"] != "home")
                if r == "lost":                       # back along the same line; skip this stop
                    print(f"[run] {arm} stop {k} ({s['kind']}): marker lost — going back", flush=True)
                    snap(f"{arm}_lost_before_stop{k:02d}")
                    r = move(arm, tok, state(arm), cur, check_marker=False)
                    if r != "ok":
                        result = r
                        break
                    k += 1
                    while k < len(stops):                 # the next moves were planned from the skipped stop:
                        way = reach(arm, cur, np.array(stops[k]["q"][arm]), hub)   # take the first one still safe
                        if way is not None:               # from here (direct or round by the hub); home is last
                            detour = way[:-1]
                            break
                        print(f"[run] {arm} stop {k}: no safe move from here — skipped", flush=True)
                        k += 1
                    else:
                        result = "no safe way home"
                    continue
                if r != "ok":
                    result = r
                    break
                if detour or not np.allclose(q, np.array(s["q"][arm])):
                    cur = q                                # a detour waypoint (the hub): no record
                    continue
                dwell(arm, tok, q, s["kind"] != "home", {"stop": k, "kind": s["kind"]})
                print(f"[run] {arm} stop {k:2d} {s['kind']:6s} moved in {time.time() - t_move:4.1f} s", flush=True)
                cur = q
                k += 1
            if result != "ok":
                break
            off = torque_off(arm, tok)
            if off.get("error") == "not_at_neutral":           # the rest pose is a few degrees off the server's home
                R[arm].must(cmd="home", token=tok)
                t_h = time.time()
                while time.time() - t_h < 30 and R[arm].state()["state"] == "HOMING":
                    R[arm].call(cmd="renew", token=tok)
                    time.sleep(0.3)
                off = torque_off(arm, tok)
            R[arm].call(cmd="release", token=tok)
            event(what="phase_done", arm=arm, torque_off=off.get("state"))
    finally:
        if result != "ok" and active is not None:
            try:                                      # the moving arm only (the resting one stays torque off)
                R[active].call(cmd="hold", token=tok)
            except Exception:
                pass
            print(f"[run] stopped: {result} — the {active} arm holds where it is (torque on)", flush=True)
        tap.close()
        log.close()
    print(f"[run] done ({result}) → {run_dir}")


# ── analyze ───────────────────────────────────────────────────────────────────────────────────────────────────────
def _quat_R(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _exp(xi) -> np.ndarray:
    """se(3) → 4×4 (rotation vector, translation)."""
    w, v = np.asarray(xi[:3], float), np.asarray(xi[3:], float)
    th = float(np.linalg.norm(w))
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    R = np.eye(3) + (np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * K @ K if th > 1e-12 else K)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, v
    return T


def _kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def _lm(f, x0, iters=60):
    x = np.asarray(x0, float).copy()
    r = f(x)
    c, lam = float(r @ r), 1e-3
    for _ in range(iters):
        J = np.empty((r.size, x.size))
        for k in range(x.size):
            e = np.zeros(x.size)
            e[k] = 1e-6
            J[:, k] = (f(x + e) - r) / 1e-6
        H, g = J.T @ J, J.T @ r
        while True:
            dx = -np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), g)
            rn = f(x + dx)
            cn = float(rn @ rn)
            if cn < c:
                x, r, c, lam = x + dx, rn, cn, lam / 3
                break
            lam *= 5
            if lam > 1e10:
                return x
        if np.abs(dx).max() < 1e-9:
            break
    return x


class BallModel:
    """The viewer's model of one arm's balls in the camera frame, with the calibration as parameters:
    p = [ξ of base←cam (6), joint-zero changes lift/elbow/wrist_flex/wrist_roll (4), jaw zero change (1)
         (, jaw hinge shift x/y/z in the gripper link, m (3) — `hinge=True`)]
    or, `mount=True`: [ξ (6), zeros (4), the marker's rigid deviation from its drawn seat on the jaw (6; contains the
    jaw zero)] — how far the real block sits off the drawing (pin clearance).
    The hinge shift moves the moving jaw — and the marker pinned to it — as a unit: where the real jaw sits on the
    gripper, which the drawings may not match (the flush mount needed ~10 mm, 2026-09-26)."""

    def __init__(self, arm: sz.Arm, Z: np.ndarray, hinge: bool = False, mount: bool = False):
        self.arm, self.Z, self.hinge, self.mount = arm, np.asarray(Z, float), hinge, mount
        self.n = 16 if mount else (14 if hinge else 11)
        self.jaw_parent = next(p for p, c, *_ in arm.joints if c == arm.ball_link)

    def balls(self, q, p=None) -> np.ndarray:
        p = np.zeros(self.n) if p is None else p
        d = np.r_[0.0, p[6:10], 0.0]
        T = self.arm.fk(np.asarray(q, float) + d)
        Tj = T[self.arm.ball_link]
        if self.hinge:                                   # shift the jaw joint origin in its parent (gripper) frame
            Tp = T[self.jaw_parent]
            Tj = Tp @ _exp(np.r_[0, 0, 0, p[11:14]]) @ np.linalg.inv(Tp) @ Tj
        c, s = math.cos(p[10]), math.sin(p[10])
        B = self.arm.balls_local @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]).T      # turn about joint 6
        if self.mount:                                   # the block off its drawn seat on the jaw (pin play)
            Dm = _exp(p[10:16])
            B = self.arm.balls_local @ Dm[:3, :3].T + Dm[:3, 3]
        Tc = np.linalg.inv(self.Z @ _exp(p[:6])) @ Tj
        return (B @ Tc[:3, :3].T + Tc[:3, 3]) * 1000.0


_TOOL_PTS: dict = {}


def _tool_points(arm: str) -> np.ndarray:
    """The tracker's constellation model for the arm's marker (the calibration's tool file), mm."""
    if arm not in _TOOL_PTS:
        hd = json.loads((CALIB / f"handeye_{arm}.json").read_text())
        _TOOL_PTS[arm] = np.array(json.loads(Path(hd["tool_calib"]).expanduser().read_text())["points_local"], float)
    return _TOOL_PTS[arm]


def stops_measured(run_dir: Path, P: dict | None = None, events: str = "events.jsonl") -> dict[str, list[dict]]:
    """Per recorded stop: median encoder angles and each ball's median raw centre (tracker index → mm, cam)."""
    ev = [json.loads(l) for l in open(run_dir / events)]
    recs = [e for e in ev if e["k"] == "event" and e.get("what") == "record"]
    out = {a: [] for a in PORTS}
    for r in recs:
        arm, t0, t1 = r["arm"], r["t0"], r["t1"]
        enc = np.array([e["pos"] for e in ev if e["k"] == "enc" and e["arm"] == arm and t0 <= e["t"] <= t1])
        taps = [e for e in ev if e["k"] == "tap" and e.get("tool") in TOOLS[arm] and not e.get("lost")
                and t0 <= e["trecv"] <= t1]
        balls: dict[int, list] = {}
        for m in taps:
            for b in m.get("balls") or []:
                balls.setdefault(int(b[0]), []).append(b[1:4])
        Rm = [_quat_R(m["raw_quat"]) for m in taps]
        # what the viewer draws (/tracker/balls): the tracker's pose × its constellation model — for the dim block-A
        # balls this differs from the raw centres by ~2 mm along the view ray (2026-09-27)
        pts = _tool_points(arm)
        pb = np.median(np.array([pts @ R_.T + np.array(m["raw_pos"]) for R_, m in zip(Rm, taps)]), 0) if taps else None
        out[arm].append({"stop": r["stop"], "kind": r["kind"], "q": np.median(enc, 0) if len(enc) else None,
                         "pose_balls": {k: pb[k] for k in range(len(pts))} if pb is not None else {},
                         "n_frames": len(taps), "balls": {k: np.median(np.array(v), 0) for k, v in balls.items()
                                                          if len(v) >= max(5, len(taps) // 3)},
                         "pose": (np.mean([m["raw_pos"] for m in taps], 0), Rm[len(Rm) // 2]) if taps else None})
    return out


def analyze(a) -> None:
    run_dir = Path(a.run_dir)
    P = json.loads((run_dir / "plan.json").read_text())
    arms, cam, info = load_models()
    meas = stops_measured(run_dir, P)
    rms = (lambda E: float(np.sqrt((np.asarray(E) ** 2).mean())))
    report = {}
    for arm in PORTS:
        perm = info[arm]["model"]["fit"]["mount"]["perm"]                  # model ball i ↔ tracker ball perm[i]
        Z = np.array(info[arm]["T_base_cam"])
        rows = [s for s in meas[arm] if s["q"] is not None and len(s["balls"]) == 4]
        if not rows:
            continue
        M = np.array([[s["balls"][perm[i]] for i in range(4)] for s in rows])        # measured, model order
        Q = [s["q"] for s in rows]
        # the arm's hand-eye poses (tracker pose × tool model = measured balls): fitted together with the stops — a
        # few stops alone cannot pin the camera (2026-09-27: 5 stops → it 'moved' 116 mm)
        hd = json.loads((CALIB / f"handeye_{arm}.json").read_text())
        he = json.loads(Path(hd["samples"]).read_text())["samples"]
        pts = np.array(json.loads(Path(hd["tool_calib"]).expanduser().read_text())["points_local"], float)
        he_q = [np.array(x["q_enc"], float) for x in he]
        he_M = []
        for x in he:
            To = np.array(json.loads(x["obs"]) if isinstance(x["obs"], str) else x["obs"])
            he_M.append((pts[perm] / 1000.0 @ To[:3, :3].T + To[:3, 3]) * 1000.0)
        B0 = BallModel(arms[arm], Z)
        E0 = np.array([np.linalg.norm(B0.balls(Q[k]) - M[k], axis=1) for k in range(len(rows))])
        # rigid part (the whole marker off) vs shape part (a ball off its seat), and each ball above/below its seat
        shape, trans, rot, off_n = [], [], [], []
        for k in range(len(rows)):
            B = B0.balls(Q[k])
            R, t = _kabsch(B, M[k])
            shape.append(np.linalg.norm(B @ R.T + t - M[k], axis=1))
            trans.append(float(np.linalg.norm(M[k].mean(0) - B.mean(0))))
            rot.append(math.degrees(math.acos(max(-1, min(1, (np.trace(R) - 1) / 2)))))
            R2, t2 = _kabsch(M[k], B)                                        # measured → model (marker pose removed)
            Tc = np.linalg.inv(Z) @ arms[arm].fk(np.asarray(Q[k], float))[arms[arm].ball_link]
            off_n.append(((M[k] @ R2.T + t2) - B) @ (Tc[:3, :3] @ arms[arm].normal_local))
        shape, off_n = np.array(shape), np.array(off_n)
        r = report[arm] = {
            "stops": len(rows), "frames_per_stop": int(np.median([s["n_frames"] for s in rows])),
            "handeye_poses": len(he_q), "tracker_ball_of_model_ball": perm,
            "model_as_drawn": {"rms_mm": round(rms(E0), 2), "max_mm": round(float(E0.max()), 2),
                               "per_ball_rms_mm": np.sqrt((E0 ** 2).mean(0)).round(2).tolist(),
                               "marker_shift_mm": round(float(np.median(trans)), 2),
                               "marker_turn_deg": round(float(np.median(rot)), 2),
                               "shape_per_ball_mm": np.sqrt((shape ** 2).mean(0)).round(2).tolist(),
                               "handeye_rms_mm": round(rms([np.linalg.norm(B0.balls(q) - m_, axis=1)
                                                            for q, m_ in zip(he_q, he_M)]), 2)},
            "ball_above_seat_mm": {"mean": off_n.mean(0).round(2).tolist(), "sd": off_n.std(0).round(2).tolist()},
            "per_stop": [{"stop": s["stop"], "kind": s["kind"], "q": np.round(s["q"], 1).tolist(),
                          "z_mm": np.round(M[k][:, 2], 0).tolist(), "err_mm": E0[k].round(2).tolist(),
                          "shape_mm": shape[k].round(2).tolist()} for k, s in enumerate(rows)]}
        print(f"== {arm}: {len(rows)} stops (~{r['frames_per_stop']} frames each) + {len(he_q)} hand-eye poses")
        d = r["model_as_drawn"]
        print(f"   model as drawn : stops rms {d['rms_mm']} mm (max {d['max_mm']}), per ball {d['per_ball_rms_mm']}; "
              f"hand-eye poses {d['handeye_rms_mm']} | whole marker off {d['marker_shift_mm']} mm / "
              f"{d['marker_turn_deg']}° | one-ball shape {d['shape_per_ball_mm']}")
        print(f"   each ball above(+)/below(−) its seat (model order; tracker {perm}): {r['ball_above_seat_mm']['mean']}"
              f" ± {r['ball_above_seat_mm']['sd']} mm")
        for name, hinge, mount in (("refit 11", False, False), ("refit 11 + jaw hinge 3", True, False),
                                   ("refit 10 + block seat 6", False, True)):
            BM = BallModel(arms[arm], Z, hinge=hinge, mount=mount)

            def res(p, idx=None):
                idx = range(len(rows)) if idx is None else idx
                return np.concatenate([(BM.balls(Q[k], p) - M[k]).ravel() for k in idx] +
                                      [(BM.balls(q, p) - m_).ravel() for q, m_ in zip(he_q, he_M)])
            pf = _lm(res, np.zeros(BM.n))
            E1 = np.array([np.linalg.norm(BM.balls(Q[k], pf) - M[k], axis=1) for k in range(len(rows))])
            H1 = np.array([np.linalg.norm(BM.balls(q, pf) - m_, axis=1) for q, m_ in zip(he_q, he_M)])
            loo = []
            for k in range(len(rows)):
                pk = _lm(lambda p: res(p, [i for i in range(len(rows)) if i != k]), pf, iters=25)
                loo.append(np.linalg.norm(BM.balls(Q[k], pk) - M[k], axis=1))
            f = {"stops_rms_mm": round(rms(E1), 2), "stops_max_mm": round(float(E1.max()), 2),
                 "leave_one_out_rms_mm": round(rms(loo), 2), "handeye_rms_mm": round(rms(H1), 2),
                 "per_ball_rms_mm": np.sqrt((E1 ** 2).mean(0)).round(2).tolist(),
                 "cam_move_mm": round(float(np.linalg.norm(pf[3:6])) * 1000, 2),
                 "cam_turn_deg": round(math.degrees(float(np.linalg.norm(pf[:3]))), 3),
                 "zero_change_deg": dict(zip(("shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "jaw"),
                                             np.round(np.r_[pf[6:10], math.degrees(pf[10])], 2).tolist())),
                 "p": pf.tolist()}
            if hinge:
                f["jaw_hinge_shift_mm"] = (pf[11:14] * 1000).round(2).tolist()
            if mount:                                  # block deviation, expressed at O along the hole axis / in plane
                Dm = _exp(pf[10:16])
                f["block_turn_deg"] = round(math.degrees(float(np.linalg.norm(pf[10:13]))), 2)
                f["block_turn_axis_jaw"] = (pf[10:13] / max(np.linalg.norm(pf[10:13]), 1e-12)).round(3).tolist()
                f["block_shift_at_balls_mm"] = round(float(np.linalg.norm(
                    (arms[arm].balls_local @ Dm[:3, :3].T + Dm[:3, 3]) - arms[arm].balls_local, axis=1).mean()) * 1000, 2)
            r[name] = f
            print(f"   {name:22s}: stops rms {f['stops_rms_mm']} mm (max {f['stops_max_mm']}), leave-one-out "
                  f"{f['leave_one_out_rms_mm']}, hand-eye {f['handeye_rms_mm']} | camera {f['cam_move_mm']} mm / "
                  f"{f['cam_turn_deg']}°, zeros {f['zero_change_deg']}"
                  + (f", hinge {f['jaw_hinge_shift_mm']} mm" if hinge else "")
                  + (f", block turned {f['block_turn_deg']}° (axis {f['block_turn_axis_jaw']} in the jaw link), balls moved "
                     f"{f['block_shift_at_balls_mm']} mm on average" if mount else ""))
    (run_dir / "analysis.json").write_text(json.dumps(report, indent=1, default=float))
    print(f"→ {run_dir / 'analysis.json'}")

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--arms", default="left,right", help="order of the phases")
    p.add_argument("--n", type=int, default=14, help="targets per arm")
    p.add_argument("--tries", type=int, default=15000, help="random joint draws per arm")
    p.add_argument("--pool", type=int, default=150, help="optimal single-arm targets kept per arm")
    p.add_argument("--step-deg", type=float, default=2.0, help="move check resolution")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start", help='JSON {"left": [...], "right": [...]} instead of the arms\' current angles')
    r = sub.add_parser("run")
    r.add_argument("run_dir")
    r.add_argument("--plan", default="plan.json", help="plan file inside run_dir (e.g. a part of the plan)")
    r.add_argument("--tag", default="", help="events_<tag>.jsonl and ir/<tag>_* (several runs in one directory)")
    r.add_argument("--speed", type=float, default=12.0, help="deg/s of the fastest joint (slow, cf. 0.2x)")
    r.add_argument("--guard-step", type=float, default=8.0,
                   help="server step_deg: how far its per-tick command may lead the joint. The move itself is slow "
                        "(--speed, streamed here); this is the lead a loaded joint needs to overcome gravity — at 2° the "
                        "right elbow stalled lifting its forearm (2026-09-27), hand-eye/follow use 8/10")
    r.add_argument("--rec-s", type=float, default=2.0)
    r.add_argument("--lost-s", type=float, default=0.5)
    r.add_argument("--lag-deg", type=float, default=20.0, help="goal − encoder beyond this for 0.5 s = blocked (sag alone reaches ~13° stretched out)")
    z = sub.add_parser("analyze")
    z.add_argument("run_dir")
    a = ap.parse_args()
    {"plan": plan, "run": run, "analyze": analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
