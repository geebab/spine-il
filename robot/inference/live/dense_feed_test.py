#!/usr/bin/env python3
"""How densely must the measured tool path be fed to the viewer for the policy to see the bone the recorder saw?

    python robot/inference/live/dense_feed_test.py RUN_DIR SEED.json --sub 1 2 4

A closed-loop run (live_loop.py) steps the viewer once per policy frame — one measured pose every 1/(fps·speed) s of wall
time (0.2 s). The FE recorder page meanwhile takes every tracker sample (~50 Hz) and cut 387 mm³ where the policy's
viewer cut 249 (run B2, 10-04). This replays the run's tracker tap (the arms' events.jsonl) through a step session with
`sub` poses per policy frame (the session clock split accordingly, trigger as the frame's) and prints the removed volume:
sub 1 must reproduce the run's own number, larger values show what a denser feed would have shown the policy.
"""
import argparse, asyncio, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import rollout as RO                      # noqa: E402
from live_robot import q2R, R2q, ROBOT_CALIB   # noqa: E402
from lerobot_robot_spinearm import handeye as he   # noqa: E402

TOOLS = (("endo", 0), ("drill", 7))


def load_tap(run: str, ldir: Path):
    """Per tool: engage row and the tap samples (ts ms, tip m, R) of the robot's marker block, as live_robot.py takes them."""
    out = {}
    for tool, _ in TOOLS:
        tip_r = he.tool_calib(ROBOT_CALIB[tool])[1]
        eng, ts, tips, Rs = None, [], [], []
        for l in open(ldir / f"{run}_{tool}" / "events.jsonl"):
            if l.startswith('{"k":"tap"'):
                m = json.loads(l)
                if m.get("lost") or m.get("pos") is None or int(m.get("vis", 4)) < 4:
                    continue
                if (tool == "drill") != (m.get("tool") == "drill"):
                    continue
                R = he.quat_to_R(m["quat"]); ts.append(float(m["ts"])); tips.append(R @ tip_r + np.asarray(m["pos"], float) / 1000.0); Rs.append(R)
            elif l.startswith('{"k":"engage"'):
                e = json.loads(l); eng = {"tip": np.asarray(e["R0_tip_mm"], float) / 1000.0, "R": np.asarray(e["R0_R"], float)}
        out[tool] = {"eng": eng, "ts": np.asarray(ts), "tip": np.asarray(tips), "R": Rs}
    return out


async def main(a):
    run_dir = Path(a.run_dir); live = json.loads((run_dir / "live.json").read_text()); m = live["meta"]
    ready = np.asarray(json.loads(Path(a.seed).read_text())["frames"][0], float)
    with open(live["ref_recording"]) as fh:
        hdr = next(x for x in map(json.loads, fh) if x.get("k") == "hdr")
    T0 = hdr["registration"]["T0"]; R0s, t0s = np.asarray(T0["R"], float).reshape(3, 3), np.asarray(T0["t"], float)
    tap = load_tap(run_dir.name, Path(a.latency_dir).expanduser())
    Rc0 = {t: R0s.T @ q2R(ready[o + 3:o + 7]) for t, o in TOOLS}; tc0 = {t: R0s.T @ (ready[o:o + 3] - t0s) / 1000.0 for t, o in TOOLS}
    c = {"endo": np.asarray(m["endo_start_residual_mm"], float) / 1000.0, "drill": np.zeros(3)}

    def state(t_ms: float, half: float) -> np.ndarray:
        v = ready.copy(); v[14] = 0.0
        for tool, o in TOOLS:
            d = tap[tool]; i0, i1 = np.searchsorted(d["ts"], [t_ms - half, t_ms + half])
            if i1 <= i0:
                i0 = int(np.clip(np.searchsorted(d["ts"], t_ms), 0, len(d["ts"]) - 1)); i1 = i0 + 1
            tip = np.median(d["tip"][i0:i1], axis=0); R = d["R"][(i0 + i1) // 2]
            tc = tc0[tool] + (tip - d["eng"]["tip"]) + c[tool]
            v[o:o + 3] = R0s @ tc * 1000.0 + t0s
            v[o + 3:o + 7] = R2q(R0s @ (R @ d["eng"]["R"].T @ Rc0[tool]), ready[o + 3:o + 7])
        return v

    F, kind = np.asarray(live["frames"], float), m["frame_kind"]
    dt_ms = 1000.0 / live["fps"]; dt_wall = 1000.0 / (live["fps"] * m["speed"])
    # target time of every non-settle frame, from the chunks' motion windows ([first − dt − 500, last + dt + 1500])
    times, k = [None] * len(F), 0
    for ch, w in zip(m["chunks"], m["motion_windows_ms"]):
        for j in range(ch["frames"]):
            times[k] = w[0] + dt_wall + 500.0 + j * dt_wall; k += 1
        k += 1                                   # the chunk's settle frame
    fe = hdr.get("fe") or {}
    flags = [f"{x}={fe[x]}" for x in ("bite", "meltk", "meltv", "touchoff", "meltmax", "scopecal") if fe.get(x) is not None]
    res = []
    for sub in a.sub:
        ws, r = await RO.session(a.api, hdr, ready, 256, flags, allow_outside=True)
        t0 = time.time(); vol = 0.0; n = 0; err = []
        try:
            await RO.step(ws, ready, dt_ms)
            await RO.step(ws, state(times[0] - dt_wall, 500.0), dt_ms, images=False)   # the measured start (as the run's first step)
            for i in range(len(F)):
                if kind[i] == "settle":
                    _, mm = await RO.step(ws, F[i], dt_ms, images=False); n += 1
                else:
                    for s in range(sub):
                        v = state(times[i] - dt_wall * (1 - (s + 1) / sub), 60.0 / sub); v[14] = F[i][14]
                        if s == sub - 1:
                            err.append(float(np.linalg.norm(v[7:10] - F[i][7:10])))
                        _, mm = await RO.step(ws, v, dt_ms / sub, images=False); n += 1
                vol = float(((mm.get("meta") or {}).get("bridge") or {}).get("removedMm3") or 0.0)
        finally:
            try:
                await ws.send(json.dumps({"op": "close"}))
            except Exception:
                pass
            await ws.close()
        res.append({"sub": sub, "steps": n, "seconds": round(time.time() - t0, 1), "removed_mm3": round(vol, 1),
                    "run_mm3": round(m["removed_mm3"][-1], 1), "rebuild_err_mm_median": round(float(np.median(err)), 3)})
        print(json.dumps(res[-1]), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run_dir"); ap.add_argument("seed"); ap.add_argument("--sub", type=int, nargs="+", default=[1, 2, 4])
ap.add_argument("--latency-dir", default="~/il-robot/runs/latency"); ap.add_argument("--api", default="ws://127.0.0.1:18780/v1/step")
ap.add_argument("-o", "--out", default=None)
asyncio.run(main(ap.parse_args()))
