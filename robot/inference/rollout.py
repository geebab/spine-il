#!/usr/bin/env python3
"""Closed-loop policy rollout against the FE viewer (render API /v1/step): state + rendered endoscope image → policy →
next tool poses → the viewer steps (and cuts) → next image … → a trajectory JSON for policy2rec.py.

    python robot/inference/rollout.py --policy-path RUN/checkpoints/last/pretrained_model \
        --dataset ~/il-robot/datasets/spine_rev4_20260929 --episode 6 --steps 300 -o traj.json [--frames-dir DIR]
    python robot/inference/rollout.py --check-image --dataset … --episode 6        # train/infer image consistency only

The step session reproduces the dataset renders' conditions: same case, registration and tool models (from the episode's
recording header), the recording's FE flags (bite/melt/touchoff/scopecal …), endoscope view at the dataset size, drill rule
r4, drill size from the header. Start = the episode's first state. Each step sends the policy's absolute next pose (tip +
quat wxyz, scene mm) and trigger with dt = 1/fps. The policy sees exactly what training saw: observation.state (15),
observation.images.endo (float 0–1, CHW), task = the phase instruction (phase 1 at the start; `--phase-rules` switches
on the viewer's own signals — skin entry, trigger/contact, back outside).
Known difference: the step session's engine ticks at the fake clock's 60 Hz, the live page at ~15 Hz (FE drill tick rate).
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
from pathlib import Path

import numpy as np

from safety import limit_step

sys_path_ds = str(Path(__file__).resolve().parents[1] / "dataset")
import sys as _sys  # noqa: E402
if sys_path_ds not in _sys.path:
    _sys.path.insert(0, sys_path_ds)
import actions as AC  # noqa: E402  (delta ↔ absolute, robot/dataset/actions.py)

PHASES = ["1 준비자세 / hold the ready pose", "2 피부진입점으로 이동 / move to the skin entry point",
          "3 피부에서 타겟으로 이동 / advance from the skin to the target", "4 L4-5 후궁 드릴링 / drill the L4-5 lamina",
          "5 피부 밖으로 이동 / withdraw out of the skin", "6 준비자세로 복귀 / return to the ready pose"]


def token() -> str | None:
    t = os.environ.get("RENDER_API_TOKEN")
    f = Path.home() / ".config/spinearm-render-api/token"
    return t or (f.read_text().strip() if f.exists() else None)


def episode(ds: Path, ep: int):
    import pyarrow.parquet as pq
    import glob
    meta = [json.loads(l) for l in (ds / "meta/spine_il/episodes.jsonl").read_text().splitlines() if l.strip()][ep]
    st, fi, tasks = [], [], []
    for f in sorted(glob.glob(str(ds / "data/*/*.parquet"))):
        t = pq.read_table(f)
        e = np.asarray(t.column("episode_index").to_pylist())
        sel = np.nonzero(e == ep)[0]
        if len(sel):
            s = t.column("observation.state").to_pylist(); k = t.column("frame_index").to_pylist()
            st += [s[i] for i in sel]; fi += [k[i] for i in sel]
    o = np.argsort(fi)
    return meta, np.asarray(st, float)[o][:, :15]   # tool state only (a --state-phase dataset appends the phase one-hot)


def dataset_frame(ds: Path, ep: int, k: int) -> np.ndarray:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    d = LeRobotDataset("local/x", root=str(ds), episodes=[ep], video_backend="pyav")
    x = d[k]["observation.images.endo"]            # CHW float 0–1
    return (x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def pose(v: np.ndarray, off: int) -> dict:
    q = np.asarray(v[off + 3:off + 7], float); q /= np.linalg.norm(q)
    return {"tip": np.asarray(v[off:off + 3], float).round(4).tolist(), "quat": q.round(6).tolist()}


async def session(api: str, hdr: dict, start: np.ndarray, size: int, flags: list[str], allow_outside: bool = False,
                  optics: str | None = None):
    """optics: None = the reference recording's scopecal flag (air in every rev.4 hand recording) · air | water = that
    endoscope medium instead. The hand recordings' header says air, but their &record page showed the operator the water
    medium (viewer session, 10-06) — images for a policy trained on water renders must be asked for here."""
    import websockets
    tok = token()
    ws = await websockets.connect(api, additional_headers={"Authorization": f"Bearer {tok}"} if tok else {},
                                  max_size=64 * 1024 * 1024, open_timeout=60,
                                  ping_interval=None)   # long steps while the renderer is shared: no keepalive timeout
    T0 = hdr["registration"]["T0"]
    reset = {"op": "reset", "case": hdr.get("meshId") or "altician_lumbar1",
             "registration": {"R": T0["R"], "t": T0["t"]},
             "tools": {t: {"tipLocal": hdr["toolCalib"][t]["tipLocal"]} for t in ("endo", "drill")},
             "options": {"views": ["endo"], "size": size, "drill_rule": "r4",
                         "drill_size": (hdr.get("drill") or {}).get("sizeKey", "medium"),
                         "extra_flags": [f for f in flags if not (optics and f.startswith("scopecal="))],
                         **({"optics": optics} if optics else {}),
                         # in-distribution with the replayed recordings the dataset images came from (render API flags.py):
                         # the rig's drive (not raw absolute poses) and the endoscope's auto-exposure
                         "ae": "on", "flags_override": {"rawabs": False}},
             "start": {"endo": pose(start, 0), "drill": pose(start, 7)}, "allow_outside": allow_outside}
    await ws.send(json.dumps(reset))
    r = json.loads(await ws.recv())
    if not r.get("ok"):
        raise SystemExit(f"reset refused: {r}")
    return ws, r


async def step(ws, v: np.ndarray, dt_ms: float, images: bool = True) -> tuple[np.ndarray | None, dict]:
    """One viewer step → (endoscope image, meta). images=False: the viewer steps and cuts but captures nothing (image None)."""
    from PIL import Image
    msg = {"op": "step", "dt_ms": dt_ms, "endo": pose(v, 0), "drill": {**pose(v, 7), "trig": bool(v[14] > 0.5)}}
    if not images:
        msg["images"] = False
    await ws.send(json.dumps(msg))
    meta = json.loads(await ws.recv())
    if not meta.get("ok"):
        raise SystemExit(f"step refused: {meta}")
    imgs = []
    for _ in meta.get("images", []):
        imgs.append(await ws.recv())
    if not imgs:
        return None, meta
    img = np.asarray(Image.open(io.BytesIO(imgs[0])).convert("RGB"))
    return img, meta


def load_policy(path: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    cfg = PreTrainedConfig.from_pretrained(path); cfg.device = "cpu"
    pol = get_policy_class(cfg.type).from_pretrained(path, config=cfg); pol.eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=path,
                                         preprocessor_overrides={"device_processor": {"device": "cpu"}})
    return pol, pre, post


def policy_optics(policy_path: str | None) -> str | None:
    """The endoscope medium of the images a policy was trained on (action_rep.json "optics", from the dataset's
    export.json "image_optics"), or None = the recordings' own flag (air)."""
    f = Path(policy_path) / "action_rep.json" if policy_path else None
    return (json.loads(f.read_text()).get("optics") or None) if f is not None and f.exists() else None


def load_step_norm(policy_path: str):
    """Per-chunk-step action statistics of a chunk-relative policy trained with --step-norm (train/lerobot-train-rel.py):
    {"mean": (T, 13), "std": (T, 13)} or None."""
    f = Path(policy_path) / "step_norm.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return {"mean": np.asarray(d["mean"], float), "std": np.asarray(d["std"], float)}


def policy_rep(policy_path: str | None) -> dict:
    """action_rep.json of a checkpoint (train/lerobot-train-rel.py), {} when there is none."""
    f = Path(policy_path) / "action_rep.json" if policy_path else None
    return json.loads(f.read_text()) if f is not None and f.exists() else {}


DONE_HOLD = 5   # steps (1/3 s at 15 fps) a predicted "done" has to stay up: a one-frame blip does not end the drilling
# The policy's end value comes in two forms, and each has the level above which it means "the drilling is over":
#   done      the 0/1 flag of the first such models (1 for the steps after the demonstration's last cut) — above one half
#   progress  user 10-09 "끝 출력을 0/1 대신 진행도로 바꾼다": bone removed / the demonstration's final removed volume.
#             The level is computed (user 10-09: "끝 수준은 네가 최적값 계산" — end_level.py): the level trades bone left
#             (low) against a prediction that falls short and never ends the run (high), and the best one depends on how
#             well the model predicts — 0.995 for a perfect predictor, 0.955 / 0.905 / 0.815 for a held-out error of
#             0.015 / 0.037 / 0.073. At 0.95 a predictor with an error of only 0.037 never ends 20 % of the
#             demonstrations. 0.91 is the level with the smallest worst-case loss over those predictors (at most 0.10
#             of the volume; a perfect predictor ends there a median 11.5 s before the last cut). To be computed again on
#             the trained model: end_level.py --policy-url … --episodes <held out>; --end-level / action_rep.json
#             "end_level" override the default.
END_LEVEL = {"done": 0.5, "progress": 0.91}
END_NAMES = {"drill.done": "done", "drill.progress": "progress"}


def end_output(rep: dict | None) -> tuple[str | None, float | None]:
    """(form, level) of a policy's end value from its action_rep.json: ("done", 0.5), ("progress", 0.91) or (None, None) for
    a policy that does not end the drilling itself."""
    rep = rep or {}
    kind = rep.get("end_output") or ("done" if rep.get("done_output") else None)
    if kind is None:
        return None, None
    return kind, float(rep.get("end_level") or END_LEVEL[kind])


def done_step(done, n_exec: int, hold: int = DONE_HOLD, level: float = END_LEVEL["done"]) -> int | None:
    """Where a predicted chunk says the drilling is over (the 14th action value of a --done-output / --progress-output
    policy): the first of the n_exec steps about to be executed from which the value stays above `level` for `hold` steps —
    looking into the predicted steps beyond n_exec; at the chunk's very end, for as many as are left. None: the drilling
    goes on. In the training data the value does not come down again (a step 0 → 1, or a fraction that only rises), so a
    prediction that means it stays up. The steps before the returned index are executed, then the policy's part is over:
    the harness does not decide the end any more, it only follows (user, 10-07: \"정책이 끝을 직접 내도록\")."""
    d = np.asarray(done, float).reshape(-1) > level
    for j in range(min(int(n_exec), len(d))):
        w = d[j:j + hold]
        if len(w) and bool(w.all()):
            return j
    return None


def predict_chunk(pol, pre, post, obs: dict, step_norm=None) -> np.ndarray:
    """The policy's action chunk in physical units, (T, action dim). `post` undoes the dataset-wide normalisation; a policy
    trained with --step-norm also has its own mean/std per chunk step."""
    import torch
    ch = pol.predict_action_chunk(pre(obs))
    ch = torch.stack([post(ch[:, j]) for j in range(ch.shape[1])], 1)[0].cpu().numpy()
    if step_norm is not None:
        ch = ch * step_norm["std"][:len(ch)] + step_norm["mean"][:len(ch)]
    return ch


def _rotvec_R(w: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def aim_endo(v: np.ndarray, meta: dict, gain: float, max_deg: float, dead_deg: float, pivot=None) -> tuple[np.ndarray, float]:
    """Keep the drill tip in the middle of the endoscope's view (the demos: 83–100 % visible while cutting, near the centre).
    From the viewer's last step: camera position + view direction (bridge.endoCamera), the drill burr centre
    (bridge.drill.B) and the skin entry (bridge.endoPose.entry). The rotation turning the view direction toward the drill
    tip (angle beyond `dead_deg`, times `gain`, capped at `max_deg` per step) is applied to the endoscope pose about the
    skin entry — the shaft keeps passing through the portal, like a surgeon pivoting the scope. Returns (frame, deg)."""
    b = ((meta or {}).get("meta") or {}).get("bridge") or {}
    cam, B, entry = b.get("endoCamera") or {}, (b.get("drill") or {}).get("B"), (b.get("endoPose") or {}).get("entry")
    if not (cam.get("position") and cam.get("focalPoint") and B and entry):
        return v, 0.0
    pos = np.asarray(cam["position"], float)
    d = np.asarray(cam["focalPoint"], float) - pos; d /= np.linalg.norm(d)
    t = np.asarray(B, float) - pos; n = np.linalg.norm(t)
    if n < 1e-6:
        return v, 0.0
    t /= n
    ang = float(np.degrees(np.arccos(np.clip(np.dot(d, t), -1, 1))))
    if ang <= dead_deg:
        return v, 0.0
    step = min(max_deg, gain * (ang - dead_deg))
    axis = np.cross(d, t); axis /= np.linalg.norm(axis)
    Rw = _rotvec_R(axis * np.radians(step))
    out = np.array(v, float); P = np.asarray(entry if pivot is None else pivot, float)
    out[0:3] = P + Rw @ (out[0:3] - P)
    q = out[3:7]; w_, x_, y_, z_ = q / np.linalg.norm(q)
    Rt = np.array([[1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - w_ * z_), 2 * (x_ * z_ + w_ * y_)],
                   [2 * (x_ * y_ + w_ * z_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - w_ * x_)],
                   [2 * (x_ * z_ - w_ * y_), 2 * (y_ * z_ + w_ * x_), 1 - 2 * (x_ * x_ + y_ * y_)]])
    R2 = Rw @ Rt
    tr = np.trace(R2)
    if tr > 0:
        S = np.sqrt(tr + 1) * 2; qn = [0.25 * S, (R2[2, 1] - R2[1, 2]) / S, (R2[0, 2] - R2[2, 0]) / S, (R2[1, 0] - R2[0, 1]) / S]
    elif R2[0, 0] > R2[1, 1] and R2[0, 0] > R2[2, 2]:
        S = np.sqrt(1 + R2[0, 0] - R2[1, 1] - R2[2, 2]) * 2; qn = [(R2[2, 1] - R2[1, 2]) / S, 0.25 * S, (R2[0, 1] + R2[1, 0]) / S, (R2[0, 2] + R2[2, 0]) / S]
    elif R2[1, 1] > R2[2, 2]:
        S = np.sqrt(1 + R2[1, 1] - R2[0, 0] - R2[2, 2]) * 2; qn = [(R2[0, 2] - R2[2, 0]) / S, (R2[0, 1] + R2[1, 0]) / S, 0.25 * S, (R2[1, 2] + R2[2, 1]) / S]
    else:
        S = np.sqrt(1 + R2[2, 2] - R2[0, 0] - R2[1, 1]) * 2; qn = [(R2[1, 0] - R2[0, 1]) / S, (R2[0, 2] + R2[2, 0]) / S, (R2[1, 2] + R2[2, 1]) / S, 0.25 * S]
    qn = np.asarray(qn); qn /= np.linalg.norm(qn)
    if np.dot(qn, q) < 0:
        qn = -qn
    out[3:7] = qn
    return out, step


def q2R(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R2q(R: np.ndarray, ref) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1) * 2; q = [0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; q = [(R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S]
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; q = [(R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S]
    else:
        S = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; q = [(R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S]
    q = np.asarray(q); q /= np.linalg.norm(q)
    return -q if np.dot(q, ref) < 0 else q


def shaft_axes(recording: str) -> dict:
    """Each tool's shaft direction in its own frame, from the recording's rows (scene.u = T0.R·R·shaft_local; constant,
    0.0° spread on 3868). The shaft points from the skin entry toward the tip."""
    rows = [json.loads(l) for l in open(recording)]
    R0 = np.asarray(rows[0]["registration"]["T0"]["R"], float).reshape(3, 3)
    out = {}
    for tool in ("endo", "drill"):
        S = [(R0 @ q2R(r["quat"])).T @ np.asarray(r["scene"]["u"], float) for r in rows
             if r.get("k") == "f" and (r.get("tool") or "endo") == tool and not r.get("lost") and (r.get("scene") or {}).get("u") and r.get("quat")]
        m = np.median(np.asarray(S), axis=0); out[tool] = m / np.linalg.norm(m)
    return out


def rcm(v: np.ndarray, off: int, portal: np.ndarray, shaft_local: np.ndarray, near_mm: float, max_deg: float) -> tuple[np.ndarray, float]:
    """Remote centre of motion: once the tip is within `near_mm` of the tool's portal, turn the tool about its tip (≤
    `max_deg` per step) so the shaft line passes through the portal — tools enter through the incision, as in surgery."""
    tip = np.asarray(v[off:off + 3], float)
    to = tip - portal; n = float(np.linalg.norm(to))
    if n > near_mm or n < 1e-6:
        return v, 0.0
    R = q2R(v[off + 3:off + 7]); u = R @ shaft_local; want = to / n
    ang = float(np.degrees(np.arccos(np.clip(np.dot(u, want), -1, 1))))
    if ang < 0.2:
        return v, 0.0
    step = min(max_deg, ang)
    axis = np.cross(u, want); axis /= np.linalg.norm(axis)
    out = np.array(v, float)
    out[off + 3:off + 7] = R2q(_rotvec_R(axis * np.radians(step)) @ R, v[off + 3:off + 7])
    return out, step


def rcm_project(v: np.ndarray, portals: dict, shafts: dict, near_mm: float, max_total_deg: float = 5.0) -> np.ndarray:
    """v with each tool turned about its tip until its shaft line passes through its portal (rcm without the per-step
    cap; the roll about the shaft is kept) — a plan target put on the pivot before the step limit looks at it (10-16).
    A tool is turned at most max_total_deg (one turn about the axis rcm takes — continuous in the plan, unlike a cut-off,
    which in the dry run of 10-16 switched on and off around its threshold and made 15° steps); the per-step rcm after the
    limit does the rest as before. 5°: the robot runs' plans were ~2° off (order_sim, fitted to their RCM turning); a
    dry run whose stand-in drill was 72° off and turned 15° a row toward the pivot made the safety gate slow (0.39 s a
    check) and the stream late. Robot runs: commands 0° (median), < 1° (p90) off the pivot."""
    out = np.array(v, float)
    for tool, off in (("endo", 0), ("drill", 7)):
        out, _ = rcm(out, off, portals[tool], shafts[tool], near_mm, max_total_deg)
    return out


def retract(v: np.ndarray, portals: dict, shafts: dict, out_mm: float, own_axis: bool = True) -> np.ndarray:
    """The frame with each tool pulled straight back along its own shaft until the tip is `out_mm` beyond its portal
    (outside the skin), orientation unchanged, trigger off — the robot-feasible ready pose / exit move."""
    out = np.array(v, float)
    for tool, off in (("endo", 0), ("drill", 7)):
        u = q2R(out[off + 3:off + 7]) @ shafts[tool]                    # portal → tip direction
        d = float(np.dot(out[off:off + 3] - portals[tool], u)) + out_mm  # depth along the shaft + the margin outside
        if d > 0:
            out[off:off + 3] = out[off:off + 3] - u * d
    out[14] = 0.0
    return out


def batch_episodes(ds: Path, ep: int) -> list[int]:
    """The hand episodes recorded with the same phantom registration (header T0) as episode ep — one sitting. The surgeon
    enters through other skin points from sitting to sitting (09-28 against 10-06: the endoscope's portal 14–17 mm away),
    and with the portal the whole approach turns (15 mm at 55 mm depth = 15°)."""
    eps = [json.loads(l) for l in (ds / "meta/spine_il/episodes.jsonl").read_text().splitlines() if l.strip()]
    def t0(e):
        with open(e["recording"]) as fh:
            h = next(d for d in map(json.loads, fh) if d.get("k") == "hdr")
        T = (h.get("registration") or {}).get("T0") or {}
        return tuple(np.round(np.asarray(T.get("R", []), float).reshape(-1), 6)) + tuple(np.round(np.asarray(T.get("t", []), float).reshape(-1), 3))
    cache: dict = {}
    key = lambda e: cache.setdefault(e["recording"], t0(e))
    mine = key(eps[ep])
    return [i for i, e in enumerate(eps) if e.get("source") == "hand" and key(e) == mine]


def demo_portals(ds: Path, only=None) -> dict:
    """The demos' skin portals (scene mm), the median per tool — over the hand episodes, or over `only` (episode indices)."""
    eps = [json.loads(l) for l in (ds / "meta/spine_il/episodes.jsonl").read_text().splitlines() if l.strip()]
    if only is not None:
        eps = [eps[i] for i in only]
    return {t: np.median(np.asarray([e["phase_info"]["portal_mm"][t] for e in eps if e.get("source") == "hand"
                                     and ((e.get("phase_info") or {}).get("portal_mm") or {}).get(t)], float), axis=0)
            for t in ("endo", "drill")}


# The endoscope's screen (image circle per medium) lives in eval/scope_screen.py — one place for the harness and the evaluation.
_sys_path_ev = str(Path(__file__).resolve().parents[1] / "eval")
if _sys_path_ev not in _sys.path:
    _sys.path.insert(0, _sys_path_ev)
from scope_screen import SCOPE, SCOPE_CIRCLE, on_screen, tip_on_screen, tip_seen  # noqa: E402,F401


def _in_view(meta: dict) -> tuple[dict, float | None]:
    b = ((meta or {}).get("meta") or {}).get("bridge") or {}
    return (b.get("drill") or {}).get("inView") or {}, (b.get("endoCamera") or {}).get("viewAngle")


def drill_on_screen(meta: dict, optics: str | None = None) -> bool:
    """The drill tip is on the endoscope's screen — inside the viewer's field of view and, with `optics` (a session opened
    in that medium — the water-trained policies), inside the image circle shown. Hidden behind bone or not."""
    iv, va = _in_view(meta)
    return tip_on_screen(iv, va, optics)


def drill_visible(meta: dict, optics: str | None = None) -> bool:
    """The stricter test (the trigger rule until 10-07): the drill tip on the screen and not hidden (bridge.drill.inView
    inFov and visible; with `optics` also inside the image circle — in air the two agreed: all 7,543 drilling frames
    judged visible in 18 runs were inside the circle)."""
    iv, va = _in_view(meta)
    return tip_seen(iv, va, optics)


TRIGGER_RULES = ("screen", "tip")


def trigger_blind(meta: dict, optics: str | None, rule: str = "screen") -> bool:
    """Is the trigger to be switched off at this observation (--visible-trigger)? rule "screen" (default; user, 10-07:
    "드릴이 없으면" = the tip has left the screen, nothing else): only when the tip is off the endoscope's screen.
    "tip": also when bone hides it — the rule of every run before 10-07 16:00; it blocked a burr sitting in the cavity it
    had cut (5 of the 7 blocks of P24C_010000_92c5_d566), which the demonstrations do in up to 17 % of their cutting frames."""
    return not (drill_on_screen(meta, optics) if rule == "screen" else drill_visible(meta, optics))


class PhaseTracker:
    """Online phase (task text) from the viewer's per-step signals, the same rules as robot/dataset/phases.py where they
    can run online: 1 → 2 when the commanded tips move (> 20 mm/s for 0.3 s) · 2 → 3 when the endoscope is inside
    (bridge.endoPose.depth > 0 and its entry within 20 mm of the working portal = median of the demos' portals) ·
    3 → 4 at the first bone cut (drill proxy cutting / removedMm3 rising) · 4 → 5 after `pause_s` without cutting and the
    trigger off (1 s; 10 s for phase-conditioned policies — the demos pause the trigger while drilling: 36 pauses, median
    1.8 s, longest 8.3 s, and the dataset labels those pauses phase 4), or — the drilling is done — when the removed volume reaches
    `vol_target` or the drilling has lasted `drill_max_s` (then phase 5 stays) · 5 → 6 when the endoscope is outside.
    `stall_s`, `stall_mm3` (runs without a drilling time limit): the drilling is also done when the last `stall_s` seconds of
    it removed less than `stall_mm3`, trigger on or not — the policy has nothing left to cut where it is. A volume, not
    "no cut at all": on 10-07 a run held the trigger for 25 s and scraped off 33 mm³, a voxel every few chunks, each one
    starting the old no-cut timer again. Default 30 mm³ in 25 s. The demos (the 90 training recordings, eval/deep_1018/
    stall_basis.py, 10-23): the least any of them removes in 25 s of its drilling is 17.9 mm³, and 3 of the 90 have a 25 s
    stretch under 30 mm³ (6.5–28.9 mm³/s over the whole phase). The basis first given here, 61.6 mm³ in any 25 s of 24 demos,
    does not reproduce.
    `end` says which rule ended the drilling: pause | stall | volume | time — or `policy`: a --done-output policy said so
    itself (policy_done; the rules above then are only bounds, and a run they end is one the policy did not end)."""

    def __init__(self, portal_endo, dt_s: float, vol_target: float | None = None, drill_max_s: float | None = None,
                 pause_s: float = 1.0, stall_s: float | None = None, stall_mm3: float = 30.0):
        self.portal = np.asarray(portal_endo, float); self.dt = dt_s; self.pause_s = pause_s; self.stall_s = stall_s; self.end = None
        self.stall_mm3, self.hist = stall_mm3, []   # hist: (seconds of drilling, removed mm³) per drilling frame
        self.vol_target, self.drill_max_s, self.drill_s, self.done_drilling = vol_target, drill_max_s, 0.0, False
        self.ph, self.moving_s, self.nocut_s, self.vol = 1, 0.0, 0.0, 0.0
        self.prev = None

    def policy_done(self) -> bool:
        """The policy's own \"the drilling is over\" (done_step): inside the skin (phase 3 or 4) and before any rule ended the
        drilling, the phase becomes 5 and `end` = policy. False when it comes too late or outside the skin."""
        if self.ph in (3, 4) and not self.done_drilling:
            self.ph = 5; self.done_drilling = True; self.end = "policy"
            return True
        return False

    def update(self, meta: dict, v: np.ndarray) -> int:
        b = (meta.get("meta") or {}).get("bridge") or {}
        ep = b.get("endoPose") or {}
        entry = ep.get("entry")
        inside = bool(ep.get("depth") is not None and ep["depth"] > 0 and entry is not None
                      and np.linalg.norm(np.asarray(entry, float) - self.portal) < 20.0)
        vol = float(b.get("removedMm3") or 0.0)
        cutting = bool(((b.get("drill") or {}).get("proxy") or {}).get("cutting")) or vol > self.vol + 1e-6
        self.vol = max(self.vol, vol)
        trig = v[14] > 0.5
        if self.prev is not None:
            sp = max(np.linalg.norm(v[0:3] - self.prev[0:3]), np.linalg.norm(v[7:10] - self.prev[7:10])) / self.dt
            self.moving_s = self.moving_s + self.dt if sp > 20.0 else 0.0
        self.prev = v.copy()
        self.nocut_s = 0.0 if cutting else self.nocut_s + self.dt
        if self.ph == 1 and (self.moving_s >= 0.3 or inside): self.ph = 2
        if self.ph == 2 and inside: self.ph = 3
        if self.ph == 3 and cutting: self.ph = 4
        if self.ph == 4:
            self.drill_s += self.dt
            self.hist.append((self.drill_s, self.vol))
            while len(self.hist) > 1 and self.hist[1][0] <= self.drill_s - (self.stall_s or 0.0):   # hist[0] = the window's start
                self.hist.pop(0)
            stalled = bool(self.stall_s) and self.hist[0][0] <= self.drill_s - self.stall_s + 1e-9 \
                and self.vol - self.hist[0][1] < self.stall_mm3
            why = "pause" if self.nocut_s >= self.pause_s and not trig else "stall" if stalled else \
                "volume" if self.vol_target and self.vol >= self.vol_target else "time" if self.drill_max_s and self.drill_s >= self.drill_max_s else None
            if why:
                self.ph = 5; self.done_drilling = True; self.end = why
        elif self.ph == 5 and cutting and not self.done_drilling: self.ph = 4
        if self.ph == 5 and not inside: self.ph = 6
        return self.ph


def demo_portal(ds: Path, only=None) -> list[float]:
    eps = [json.loads(l) for l in (ds / "meta/spine_il/episodes.jsonl").read_text().splitlines() if l.strip()]
    if only is not None:
        eps = [eps[i] for i in only]
    P = [e["phase_info"]["portal_mm"]["endo"] for e in eps if e.get("source") == "hand" and (e.get("phase_info") or {}).get("portal_mm")]
    return np.median(np.asarray(P, float), axis=0).tolist()


def prepare(a) -> dict:
    """Everything before the viewer session opens: the episode, the step session's flags, the safety gate walked to the
    start pose and — with --cycle — the robot-feasible ready pose and the scripted entry (shared with live_loop.py)."""
    ds = Path(a.dataset).expanduser()
    meta, states = episode(ds, a.episode)
    with open(meta["recording"]) as fh:
        hdr = next(d for d in map(json.loads, fh) if d.get("k") == "hdr")
    fe = hdr.get("fe") or {}
    flags = [f"{k}={fe[k]}" for k in ("bite", "meltk", "meltv", "touchoff", "meltmax", "scopecal") if fe.get(k) is not None]
    dt_ms = 1000.0 / a.fps
    k0 = a.start_frame
    # the step session opens only with the endoscope through the skin: warm up on the demo's first inside pose, then
    # step to the start pose (render_step.py does the same for the training images)
    # default: open at the start pose itself, outside the skin if it is (allow_outside) — the viewer then picks its skin
    # entry from the path, like the live page and the robot session. --warm: the old way (open on the demo's first
    # inside pose; that pins the drive's entry to the demo's portal and hides a policy that enters elsewhere)
    batch = batch_episodes(ds, a.episode) if getattr(a, "portal_from", "all") == "batch" else None
    portals = demo_portals(ds, batch); shafts = shaft_axes(meta["recording"])
    if batch is not None:
        print(f"[rollout] skin portals of the episode's own sitting ({len(batch)} episodes): "
              + ", ".join(f"{t} {np.round(p, 1).tolist()}" for t, p in portals.items()), flush=True)
    gate, q_ready, cyc = None, None, None
    # a recording of another rig (the 10-06 demos): on the robot its scene is seen through a registration fitted to the arms'
    # tools (eval/fit_registration.py) instead of the recording's own camera axes. The gate, the robot link and the policy
    # recordings (through their header: the robot session's registration) take it; the step session keeps the recording's
    # T0 — the same scene either way, and the images stay those of the training renders.
    reg, hdr_robot = None, hdr
    if getattr(a, "gate_registration", None):
        reg = json.loads(Path(a.gate_registration).expanduser().read_text())
        if reg.get("recording") and Path(reg["recording"]).name != Path(meta["recording"]).name:
            raise SystemExit(f"--gate-registration is for {Path(reg['recording']).name}, the episode is {Path(meta['recording']).name}")
        hdr_robot = {**hdr, "registration": {**hdr["registration"], "T0": {"R": reg["R"], "t": reg["t"]}}}
        reg = {"file": str(a.gate_registration), "T0": {"R": reg["R"], "t": reg["t"]}, "anchor_s": reg.get("anchor_s"),
               "arms_of": reg.get("arms_of"), "fit": reg.get("fit")}
        print(f"[rollout] robot registration fitted to the arms' tools ({(reg['fit'] or {}).get('turn_deg')}° from the recording's)", flush=True)
    s_given = None
    if a.start_state:   # a start pose that is not a demo frame (e.g. random within the demos' start range, make_starts.py)
        s_given = np.asarray(json.loads(Path(a.start_state).read_text())["state"], float)[:15]
    if a.gate_pair:   # the gate is anchored at the pair's window start: states[k0], or --gate-anchor-s
        from safety_gate import RobotGate
        pj = json.loads(Path(a.gate_pairs_file).expanduser().read_text())   # {rid: pair}, or a search's pair_clear_<rid>.json
        pj = pj["best"] if "best" in pj else pj[a.gate_pair]
        k_an = k0
        if a.gate_anchor_s is not None:   # the pair's window start on the session clock (a trimmed episode starts earlier)
            k_an = int(round((a.gate_anchor_s - (meta["t_frame0_ms"] - meta["t_exp_first_ms"]) / 1000.0) * a.fps))
            if not 0 <= k_an < len(states):
                raise SystemExit(f"--gate-anchor-s {a.gate_anchor_s}: frame {k_an} is outside the episode (0..{len(states) - 1})")
        # the pose the gate is anchored at: a frame of the base episode — or, for a base that is not one recording
        # (eval/sitting_base.py: the representative pose of a sitting), the state in --gate-anchor-state
        anchor_state = states[k_an]
        if getattr(a, "gate_anchor_state", None):
            anchor_state = np.asarray(json.loads(Path(a.gate_anchor_state).expanduser().read_text())["state"], float)[:15]
            if s_given is None:
                raise SystemExit("--gate-anchor-state needs --start-state (there is no demo frame to walk to from a pose that is not one)")
        gate = RobotGate(anchor_state, hdr_robot["registration"]["T0"], pj["q_left"], pj["q_right"])
        gate.turn_wait = bool(getattr(a, "gate_turn_wait", False))
        if s_given is not None:   # straight from the anchor to the given start (positions linear, orientations slerp)
            from safety import _slerp as _sl0
            a0 = np.array(anchor_state, float)
            nw = max(2, int(np.ceil(max(np.linalg.norm(s_given[o:o + 3] - a0[o:o + 3]) for o in (0, 7)) / 0.5)))
            for j in range(1, nw + 1):
                f = a0 + (s_given - a0) * (j / nw)
                for o in (3, 10):
                    f[o:o + 4] = _sl0(a0[o:o + 4], s_given[o:o + 4], j / nw)
                ok, why, q = gate.check(f)
                if not ok:
                    raise SystemExit(f"--start-state is not reachable from the pair's anchor (step {j}/{nw}): {why}")
                gate.q.update(q)
        else:
            for k in range(k_an, k0, 1 if k0 > k_an else -1):   # walk the arms along the demo from the anchor to the start pose
                ok, why, q = gate.check(states[k + (1 if k0 > k_an else -1)])
                if not ok:
                    raise SystemExit(f"start frame {k0} is not reachable from the pair's anchor (frame {k_an}): frame {k} — {why}")
                gate.q.update(q)
        print(f"[rollout] safety gate on: robot start pair of {a.gate_pair}" + (f", anchored at the state of {Path(a.gate_anchor_state).name}"
              if getattr(a, "gate_anchor_state", None) else f", anchored at frame {k_an}" if k_an != k0 else ""), flush=True)
    s0 = s_given.copy() if s_given is not None else states[k0].copy()
    if a.start_jitter_mm > 0:   # a different sample of the same start: the closed loop is chaotic — a 1/255 change in 0.14 % of
        # the pixels flipped the 25k ACT between a stall (7 mm³) and 40 s of drilling (145 mm³), 10-02 — so one rollout is one draw
        rng = np.random.default_rng(a.seed)
        s0[0:3] += rng.uniform(-a.start_jitter_mm, a.start_jitter_mm, 3); s0[7:10] += rng.uniform(-a.start_jitter_mm, a.start_jitter_mm, 3)
    start_v = s0.copy()
    if getattr(a, "portal_from", "all") == "run":
        # the pivot point is the run's own: where each tool's shaft crosses the skin of the model at the start pose (the hole
        # it enters through), not the demonstrations' median entry (user, 10-07: the pivot point must not be forced). The
        # crossing comes from the viewer's own answers in the recordings (eval/skin_surface.py: within 0.2 mm of castSkinRay).
        import skin_surface as SK
        pts = SK.load()
        forced = {t: np.asarray(p_, float) for t, p_ in portals.items()}
        portals = {}
        for tool, o in (("endo", 0), ("drill", 7)):
            e = SK.entry_point(s0[o:o + 3], q2R(s0[o + 3:o + 7]) @ shafts[tool], pts)
            if e is None:
                raise SystemExit(f"--portal-from run: no skin sample near the {tool}'s shaft at the start pose — the model's skin is not known there")
            portals[tool] = e
        print("[rollout] pivot points = this run's own skin entries: " + ", ".join(
            f"{t} {np.round(p_, 1).tolist()} ({np.linalg.norm(p_ - forced[t]):.1f} mm from the demonstrations' median)" for t, p_ in portals.items()), flush=True)
    if a.cycle:   # robot-feasible full cycle: ready pose above the portal → axial entry → policy → axial exit → ready
        work = s0.copy(); work[14] = 0.0
        ready = retract(work, portals, shafts, a.ready_out_mm)
        n_in = max(2, int(np.ceil(max(np.linalg.norm(ready[o:o + 3] - work[o:o + 3]) for o in (0, 7)) / (a.entry_speed_mms / a.fps))))
        out_path = [work + (ready - work) * (j / n_in) for j in range(1, n_in + 1)]     # work → ready (the pre-pass)
        for f in out_path:
            f[3:7] = work[3:7]; f[10:14] = work[10:14]
        if gate is not None:   # joint solution at the ready pose = the robot's start pose for the playback
            for j, f in enumerate(out_path):
                ok, why, q = gate.check(f)
                if not ok:   # the arms do not reach the full retraction: the ready pose is the last safe frame, if outside
                    reached = min(a.ready_out_mm - (1 - j / n_in) * (np.linalg.norm(ready[o:o + 3] - work[o:o + 3]))
                                  for o in (0, 7)) if j else -1.0
                    if reached < a.ready_min_out_mm:
                        raise SystemExit(f"ready pose not reachable inside the safety gate ({reached:.0f} mm outside): {why}")
                    print(f"[rollout] ready pose {reached:.0f} mm outside the skin (not {a.ready_out_mm:g}): {why}", flush=True)
                    ready = out_path[j - 1].copy(); out_path = out_path[:j]; n_in = j
                    break
                gate.q.update(q)
            q_ready = {arm: (gate.q[arm] - gate.K[arm].off).round(2).tolist() for arm in gate.q}
            # the arm may not reach the working orientation at the ready pose (3868: left 5° short, at a joint limit).
            # The ready pose takes the orientation the arm actually has there, and the entry turns it into the working
            # orientation — so the playback, anchored at the ready pose, asks for exactly what the gate checked.
            for arm, (tool, off) in {"left": ("endo", 0), "right": ("drill", 7)}.items():
                k = gate.K[arm]; Rbc = k.Z[:3, :3]; Ta = k.ik.fk(gate.q[arm])
                dR = Rbc.T @ Ta[:3, :3] @ gate.FK0[arm][:3, :3].T @ Rbc
                ready[off + 3:off + 7] = R2q(gate.R0s @ (dR @ gate.H0[arm][0]), work[off + 3:off + 7])
        from safety import _slerp as _sl
        entry = []
        for j in range(n_in - 1, -1, -1):   # ready → work: position linear, orientation slerp
            f = work + (ready - work) * (j / n_in)
            for o in (3, 10):
                f[o:o + 4] = _sl(work[o:o + 4], ready[o:o + 4], j / n_in)
            f[14] = 0.0; entry.append(f)
        cyc = {"ready": ready, "entry": entry}
        start_v = ready.copy()
        print(f"[rollout] cycle: ready pose {a.ready_out_mm:g} mm outside the skin, entry {n_in} frames, robot start {q_ready}", flush=True)
    return {"ds": ds, "meta": meta, "states": states, "hdr": hdr, "flags": flags, "dt_ms": dt_ms, "k0": k0, "portals": portals,
            "shafts": shafts, "gate": gate, "q_ready": q_ready, "cyc": cyc, "start_v": start_v, "hdr_robot": hdr_robot, "registration": reg,
            "portal_endo": (np.asarray(portals["endo"], float).tolist() if getattr(a, "portal_from", "all") == "run" else demo_portal(ds, batch)),
            "portal_batch": batch,
            "optics": getattr(a, "optics", None) or policy_optics(getattr(a, "policy_path", None))}


async def main_async(a) -> None:
    P = prepare(a)
    ds, meta, states, hdr, flags, dt_ms, k0 = (P[k] for k in ("ds", "meta", "states", "hdr", "flags", "dt_ms", "k0"))
    portals, shafts, gate, q_ready, cyc, start_v = (P[k] for k in ("portals", "shafts", "gate", "q_ready", "cyc", "start_v"))
    ph3 = (meta.get("phases") or {}).get("3")
    k_warm = min(len(states) - 1, int(round((ph3[0] + 0.3) * a.fps))) if (ph3 and a.warm) else k0
    ws, r = await session(a.api, hdr, states[k_warm] if a.warm else start_v, a.size, flags, allow_outside=not a.warm,
                          optics=P["optics"])
    try:
        img, m = await step(ws, start_v, dt_ms)
        if a.check_image:   # feed the episode's own states from k0 (the drive builds its history) and compare frames
            from PIL import Image
            at = sorted({int(x) for x in a.check_at.split(",")}) if a.check_at else [k0]
            for k in range(k0, max(at) + 1):
                if k > k0:
                    img, m = await step(ws, states[k], dt_ms)
                if k in at:
                    ref = dataset_frame(ds, a.episode, k)
                    diff = np.abs(img.astype(float) - ref.astype(float))
                    print(json.dumps({"frame": k, "mean_abs_diff": round(float(diff.mean()), 2),
                                      "p95": round(float(np.percentile(diff, 95)), 1)}), flush=True)
                    if a.frames_dir:
                        d = Path(a.frames_dir); d.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(np.concatenate([ref, img], 1)).save(d / f"check_{k:04d}.png")
            return
        if a.demo_replay:
            pol = pre = post = None; n_state = 15
        else:
            import torch
            pol, pre, post = load_policy(a.policy_path)
            # inference-side filters on the chunk execution (ACT): how much of each predicted chunk is executed
            if a.temporal_ensemble is not None:   # a new chunk every step, overlapping predictions averaged (ACT paper: 0.01)
                from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
                pol.config.temporal_ensemble_coeff = a.temporal_ensemble; pol.config.n_action_steps = 1
                pol.temporal_ensembler = ACTTemporalEnsembler(a.temporal_ensemble, pol.config.chunk_size)
            elif a.n_action_steps:
                pol.config.n_action_steps = int(a.n_action_steps)
            pol.reset()
            n_state = int(pol.config.input_features["observation.state"].shape[0])
            rp = Path(a.policy_path) / "action_rep.json"   # chunk-relative policy (train/lerobot-train-rel.py)
            chunk_rel = rp.exists() and json.loads(rp.read_text()).get("rep") == "chunk_relative"
            step_norm = load_step_norm(a.policy_path)
            relq: list = []
        end_kind, end_level = (None, None) if a.demo_replay else end_output(policy_rep(a.policy_path))
        if end_kind and getattr(a, "end_level", None):
            end_level = float(a.end_level)
        done_out = end_kind is not None   # the policy ends the drilling itself (a flag or a progress value)
        done_pending, policy_done_at = False, None
        v = start_v.copy(); traj = [v.tolist()]; ph = 1; n_clamped = 0; n_gated = 0; vis = []; n_aimed = 0; aim_deg = []; n_rcm = 0
        gate_log, prev_v = [], v.copy()
        dlog: list = []; n_guard = [0]   # per step: contact gate, burr penetration (mm), in FOV, visible, cutting · surface-guard pushes
        pause_s = a.drill_pause_s if a.drill_pause_s is not None else (10.0 if n_state == 21 else 1.0)
        tracker = PhaseTracker(P["portal_endo"], dt_ms / 1000.0, a.drill_target_mm3, a.drill_max_s, pause_s, a.drill_stall_s, a.drill_stall_mm3); phases = [1]; vols = [0.0]
        fd = Path(a.frames_dir) if a.frames_dir else None
        if fd: fd.mkdir(parents=True, exist_ok=True)
        segs = []   # [name, first frame index]
        step_i = 0

        def view(vv: np.ndarray) -> np.ndarray:
            """The pose the viewer draws: with --arm-model, the tools as they sit on the arms (the robot session's page)."""
            if not (a.arm_model and gate is not None):
                return vv
            f = gate.achieved(vv, physical=True); f[14] = vv[14]
            return f

        async def commit(vv: np.ndarray, scripted: bool):
            """Gate → trigger rules → the viewer steps → bookkeeping. Returns the frame that was actually sent."""
            nonlocal prev_v, img, m, ph, n_gated, step_i
            if gate is not None:   # only frames the two arms can play back (safety_gate.py, the pair search's checks)
                vv, gi = gate.filter(prev_v, vv)
                if not gi["ok"] or gi["scale"] < 1.0 or gi.get("turn") is not None:
                    gate_log.append([step_i, gi["scale"], (gi["why"] or "")[:80]] + ([f"turn {gi['turn']}"] if gi.get("turn") is not None else []))
                if a.arm_model:   # what the arms make of it: the tip there, the tool turned only as far as their IK turns it
                    trig = vv[14]; vv = gate.achieved(vv); vv[14] = trig
            prev_v = vv.copy()
            if scripted or tracker.ph >= 5:   # no trigger on the scripted entry/exit or once the drilling is over
                vv[14] = 0.0
            if a.visible_trigger and vv[14] > 0.5 and trigger_blind(m, P["optics"], a.visible_trigger_rule):   # no drilling off screen
                vv[14] = 0.0; n_gated += 1
            if a.surface_guard is not None and not scripted and vv[14] < 0.5:
                # bone is solid: with the trigger off the burr may not sink into it. The viewer has no reaction force, and a
                # policy that creeps forward without cutting buries the burr behind bone (hidden → it never triggers again,
                # 25k ACT 10-02). Push the drill back toward the viewer's contact proxy when it is deeper than the limit.
                dr = ((m.get("meta") or {}).get("bridge") or {}).get("drill") or {}
                px = dr.get("proxy") or {}
                if px.get("penMm") is not None and px["penMm"] > a.surface_guard and px.get("q") and dr.get("B"):
                    c = np.asarray(px["q"], float) - np.asarray(dr["B"], float)
                    n = float(np.linalg.norm(c))
                    if n > 1e-6:
                        vv[7:10] += c / n * min(n, a.surface_guard_step_mm); n_guard[0] += 1
                        prev_v = vv.copy()
            img, m = await step(ws, view(vv), dt_ms)
            traj.append(vv.tolist())
            dr = ((m.get("meta") or {}).get("bridge") or {}).get("drill") or {}
            px, ivw = dr.get("proxy") or {}, dr.get("inView") or {}
            dlog.append([px.get("gate"), None if px.get("penMm") is None else round(px["penMm"], 2), int(bool(ivw.get("inFov"))),
                         int(bool(ivw.get("visible"))), int(bool(px.get("cutting")))])
            ph_seen = tracker.update(m, vv)
            if a.phase_rules:
                ph = ph_seen
            phases.append(ph_seen); vols.append(tracker.vol)
            iv = (((m.get("meta") or {}).get("bridge") or {}).get("drill") or {}).get("inView") or {}
            vis.append([int(bool(iv.get("inFov") and iv.get("visible"))), iv.get("x"), iv.get("y"), int(vv[14] > 0.5)])
            if fd and step_i % 15 == 0:
                from PIL import Image
                Image.fromarray(img).save(fd / f"{step_i:05d}.png")
            if step_i % 50 == 0:
                print(f"[rollout] step {step_i} phase {ph_seen} trig {int(vv[14] > 0.5)} removed {tracker.vol:.0f} mm3 drill tip {np.round(vv[7:10], 1)}"
                      + (f" | gate shrunk {gate.n_shrunk} held {gate.n_held}" + (f" turn cut {gate.n_turn_cut}" if gate.turn_wait else "") if gate else ""), flush=True)
            step_i += 1
            return vv

        if cyc:   # scripted axial entry: ready pose → the working pose
            segs.append(["entry", len(traj)])
            for f in cyc["entry"]:
                v = await commit(f.copy(), True)
        segs.append(["policy", len(traj)])
        if a.demo_replay:   # the demo's own in-body frames in place of the policy (cycle mechanics / teacher playback)
            ph5 = (meta.get("phases") or {}).get("5")
            k_end = min(len(states) - 1, int(round(ph5[0] * a.fps))) if ph5 else len(states) - 1
            for k in range(k0 + 1, k_end + 1):
                v = await commit(states[k].copy(), False)
        gain_ph = {int(x) for x in a.action_gain_phases.split(",") if x}
        ph_prev = tracker.ph
        for i in range(0 if a.demo_replay else a.steps):
            if a.replan_on_phase and tracker.ph != ph_prev:   # queued actions were predicted for the previous phase
                pol.reset(); relq.clear()
            ph_prev = tracker.ph
            # --state-phase policies; a --done-output policy was never told that the drilling is over (its input folds 5 into 4)
            sv = v if n_state == 15 else np.concatenate([v, np.eye(6)[(4 if done_out and tracker.ph == 5 else tracker.ph) - 1]])
            obs = {"observation.state": torch.tensor(sv, dtype=torch.float32)[None],
                   "observation.images.endo": torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)[None] / 255.0,
                   "task": [PHASES[ph - 1]]}
            with torch.no_grad():
                if chunk_rel:   # predict a cumulative chunk, execute its first n_action_steps as per-step deltas
                    if not relq:
                        ch = predict_chunk(pol, pre, post, obs, step_norm)
                        n_run = int(pol.config.n_action_steps)
                        if done_out and (jd := done_step(ch[:, 13], n_run, level=end_level)) is not None:
                            n_run, done_pending = jd, True   # the steps before the policy's "done", then its part is over
                        relq = AC.cum_to_steps(ch[:n_run, :13])
                        if done_pending and not relq:        # "done" at the chunk's first step: nothing to execute before it
                            if tracker.policy_done():
                                policy_done_at = len(traj) - 1
                            break
                    act = torch.tensor(relq.pop(0))[None]
                else:
                    act = post(pol.select_action(pre(obs)))
            raw = np.asarray(act.squeeze(0).cpu().numpy(), float)
            if raw.shape[0] == len(AC.DELTA_NAMES):   # delta-action policy: each action moves from the current state
                if a.action_gain != 1.0 and tracker.ph in gain_ph:   # undo the regression-to-still shrinkage (openloop_eval.py)
                    raw[0:3] *= a.action_gain; raw[6:9] *= a.action_gain
                raw = AC.from_delta(v, raw)
            v, hit = limit_step(v, raw, a.max_step_mm, a.max_step_deg)   # chunk seams jump (safety.py)
            n_clamped += hit
            if a.rcm:   # both shafts through their portals once near the skin
                for tool, off in (("endo", 0), ("drill", 7)):
                    v, dg = rcm(v, off, portals[tool], shafts[tool], a.rcm_near_mm, a.rcm_max_deg)
                    n_rcm += dg > 0
            if a.aim and tracker.ph >= 3:   # inside the skin: keep the drill tip centred, pivoting about the endo portal
                v, deg = aim_endo(v, m, a.aim_gain, a.aim_max_deg, a.aim_dead_deg,
                                  pivot=portals["endo"] if a.rcm else None)
                n_aimed += deg > 0; aim_deg.append(round(deg, 2))
            v = await commit(v, False)
            if done_pending and not relq:       # the steps before the policy's "done" are executed
                if tracker.policy_done():
                    policy_done_at = len(traj) - 1
                break
            if cyc and tracker.done_drilling:   # the drilling phase is over (target volume / time cap / policy stopped)
                break
        if cyc:   # scripted exit: each tool straight back along its own shaft to outside the skin, then to the ready pose
            segs.append(["exit", len(traj)])
            v_out = retract(v, portals, shafts, a.ready_out_mm, own_axis=True)
            n1 = max(2, int(np.ceil(max(np.linalg.norm(v_out[o:o + 3] - v[o:o + 3]) for o in (0, 7)) / (a.entry_speed_mms / a.fps))))
            v_from = v.copy()
            for j in range(1, n1 + 1):
                f = v_from + (v_out - v_from) * (j / n1); f[3:7] = v_from[3:7]; f[10:14] = v_from[10:14]
                v = await commit(f, True)
            segs.append(["return", len(traj)])
            v_from = v.copy(); tgt = cyc["ready"]
            n2 = max(2, int(np.ceil(max(np.linalg.norm(tgt[o:o + 3] - v_from[o:o + 3]) for o in (0, 7)) / (a.entry_speed_mms / a.fps))))
            from safety import _slerp
            for j in range(1, n2 + 1):
                f = v_from + (tgt - v_from) * (j / n2)
                for o in (3, 10):
                    f[o:o + 4] = _slerp(v_from[o:o + 4], tgt[o:o + 4], j / n2)
                v = await commit(f, True)
        out = {"fps": a.fps, "frames": traj, "ref_recording": meta["recording"],
               "meta": {"policy": "demo-replay" if a.demo_replay else a.policy_path, "dataset": str(ds), "episode": a.episode, "start_frame": k0,
                        "steps": a.steps, "phase_rules": a.phase_rules, "limit": [a.max_step_mm, a.max_step_deg],
                        "filters": {"n_action_steps": a.n_action_steps, "temporal_ensemble": a.temporal_ensemble,
                                    "action_gain": a.action_gain, "action_gain_phases": a.action_gain_phases,
                                    "replan_on_phase": a.replan_on_phase, "drill_pause_s": pause_s,
                                    "surface_guard_mm": a.surface_guard,
                                    "start_jitter_mm": a.start_jitter_mm, "seed": a.seed if a.start_jitter_mm > 0 else None,
                                    "start_state": a.start_state},
                        "surface_guard_frames": n_guard[0], "drill_log": json.dumps(dlog),
                        "done_output": done_out, "end_output": end_kind, "end_level": end_level,
                        "policy_done_frame": policy_done_at, "drill_end": tracker.end,
                        "clamped_frames": n_clamped, "trigger_gated_frames": n_gated, "aimed_frames": n_aimed, "rcm_frames": n_rcm,
                        "cycle": None if not cyc else {"segments": segs, "ready_out_mm": a.ready_out_mm, "robot_start": q_ready},
                        "registration": P.get("registration"), "arm_model": bool(a.arm_model), "portal_from": a.portal_from,
                        "safety_gate": None if gate is None else {"pair_of": a.gate_pair, "checked": gate.n_checked,
                                                                   "shrunk": gate.n_shrunk, "held": gate.n_held,
                                                                   "turn_wait": gate.turn_wait, "turn_cut": gate.n_turn_cut,
                                                                   "events": gate_log[:200]},
                        "rcm": {"on": a.rcm, "near_mm": a.rcm_near_mm, "max_deg": a.rcm_max_deg,
                                "portals": {k: np.round(p, 1).tolist() for k, p in portals.items()}}, "warm": a.warm,
                        "aim": {"on": a.aim, "gain": a.aim_gain, "max_deg": a.aim_max_deg, "dead_deg": a.aim_dead_deg},
                        "visible_trigger": a.visible_trigger, "visible_trigger_rule": a.visible_trigger_rule, "gate_anchor_state": getattr(a, "gate_anchor_state", None), "phases_seen": phases, "removed_mm3": vols,
                        "drill_in_view": vis}}
        Path(a.out).write_text(json.dumps(out))
        print(json.dumps({"out": a.out, "frames": len(traj)}))
    finally:
        try:
            await ws.send(json.dumps({"op": "close"}))
        except Exception:
            pass
        await ws.close()


def build_parser(doc: str | None = None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=doc or __doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-path")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--api", default="ws://127.0.0.1:18780/v1/step")
    ap.add_argument("--optics", choices=("air", "water"), default=None,
                    help="endoscope medium of the viewer session; default = the one the policy was trained on (its action_rep.json), "
                         "else the reference recording's flag (air)")
    ap.add_argument("--phase-rules", action="store_true")
    ap.add_argument("--demo-replay", action="store_true",
                    help="play the demo's own frames from --start-frame to the end of its drilling phase instead of a policy")
    ap.add_argument("--cycle", action="store_true",
                    help="robot-feasible full cycle: start at a ready pose above the portal (tools retracted along their "
                         "shafts), scripted axial entry to --start-frame's pose, the policy, then scripted exit and return")
    ap.add_argument("--ready-out-mm", type=float, default=20.0, help="how far outside the skin the ready pose sits")
    ap.add_argument("--entry-speed-mms", type=float, default=10.0)
    ap.add_argument("--drill-target-mm3", type=float, default=None, help="end the drilling phase at this removed volume")
    ap.add_argument("--drill-max-s", type=float, default=None, help="end the drilling phase after this long")
    ap.add_argument("--drill-stall-s", type=float, default=None, help="end the drilling phase when its last this-many seconds removed "
                    "less than --drill-stall-mm3 (trigger on or not)")
    ap.add_argument("--drill-stall-mm3", type=float, default=30.0, help="… this volume; the 90 training demos remove at least 17.9 mm³ in any 25 s of drilling, 3 of them less than 30 (eval/deep_1018/stall_basis.py)")
    ap.add_argument("--warm", action="store_true",
                    help="open the session on the demo's first inside pose (old way; biases the drive's entry)")
    ap.add_argument("--gate-pair", default=None, help="recording id whose robot start pair the safety gate uses "
                    "(the rollout must start at that pair's window start, e.g. --start-frame 255 for 3868)")
    ap.add_argument("--ready-min-out-mm", type=float, default=10.0, help="--cycle with the gate: accept a ready pose at "
                    "least this far outside the skin when the arms do not reach --ready-out-mm")
    ap.add_argument("--start-state", default=None, help="JSON {'state': 15 values}: start here instead of --start-frame "
                    "(the gate walks straight from its anchor to it)")
    ap.add_argument("--start-jitter-mm", type=float, default=0.0, help="uniform ± noise on both tool tips at the start pose "
                    "(with --seed): repeated draws of the same start, to see how often a policy stalls")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--surface-guard", type=float, default=None, help="with the trigger off, keep the burr from sinking more than "
                    "this many mm into the bone: the drill is pushed back toward the viewer's contact proxy (off by default)")
    ap.add_argument("--surface-guard-step-mm", type=float, default=1.0, help="largest push-back per step")
    ap.add_argument("--drill-pause-s", type=float, default=None, help="the drilling phase ends after this long without cutting "
                    "and the trigger off (default 1 s; 10 s for a phase-conditioned policy, whose training labels keep "
                    "trigger pauses inside phase 4)")
    ap.add_argument("--end-level", type=float, default=None, help="the level above which the policy's end value means the drilling is "
                    "over (default by its form: 0.5 for the done flag, 0.91 for the progress — END_LEVEL)")
    ap.add_argument("--n-action-steps", type=int, default=None, help="execute this many steps of each predicted chunk "
                    "(default: the policy's own, 16 of 40)")
    ap.add_argument("--temporal-ensemble", type=float, default=None, help="ACT temporal ensembling coefficient (0.01): predict "
                    "a chunk every step and average the overlapping predictions")
    ap.add_argument("--action-gain", type=float, default=1.0, help="scale the position deltas of a delta-action policy")
    ap.add_argument("--action-gain-phases", default="3,4", help="phases in which --action-gain applies")
    ap.add_argument("--replan-on-phase", action="store_true", help="drop the queued actions when the phase changes")
    ap.add_argument("--gate-anchor-s", type=float, default=None, help="session time (s) of the pair's window start when the "
                    "rollout starts elsewhere — e.g. frame 0 of a trimmed episode (export_lerobot.py --trim robot); the arms "
                    "are walked along the demo from there to the start frame")
    ap.add_argument("--gate-turn-wait", action="store_true", help="safety gate: a step that fails only on the IK's orientation "
                    "keeps its position and as much of its turn as passes (safety_gate.py turn_wait) instead of being shrunk whole")
    ap.add_argument("--portal-from", choices=("all", "batch", "run"), default="all", help="the skin portals the shafts pivot about and "
                    "the phases are judged by: the median of all hand episodes, or of the base episode's own sitting (the same "
                    "phantom registration) — the 10-06 demos enter 14–17 mm from where the 09-28 demos did; run = this run's own: "
                    "where each tool's shaft crosses the model's skin at the start pose (eval/skin_surface.py) — not a point fixed "
                    "beforehand (user, 10-07)")
    ap.add_argument("--arm-model", action="store_true", help="with the gate: the simulated tools go where the ARMS would take "
                    "them — the tip as asked, the tool turned only as far as the arms' IK turns it (about a tenth) — and that "
                    "pose is the policy's next state, as the measured pose is in live_loop.py; the viewer draws the tools as "
                    "they sit on the arms (the pair's orientation error included). Without it the tools follow every command")
    ap.add_argument("--gate-anchor-state", default=None, help="a base that is not one recording (eval/sitting_base.py base_<NAME>.json): "
                    "the gate is anchored at this state instead of a frame of --episode; with --start-state, --gate-registration and "
                    "the pair found for it (--gate-pairs-file). --episode then only names the sitting (header, skin portals)")
    ap.add_argument("--gate-registration", default=None, help="eval/fit_registration.py's file for the episode's recording: "
                    "the scene as this rig's camera sees it (a recording made on another rig) — gate, robot link, policy recordings")
    ap.add_argument("--gate-pairs-file", default="~/il-robot/runs/jitter_iter/work/pairstart/pairs_by_recording_chosen.json",
                    help="{recording id: {q_left, q_right}} — or one search result, pairstart/pair_clear_<rid>.json ({'best': pair})")
    ap.add_argument("--rcm", action=argparse.BooleanOptionalAction, default=True,
                    help="near the skin, turn each tool so its shaft passes through its portal (default on)")
    ap.add_argument("--rcm-near-mm", type=float, default=70.0)
    ap.add_argument("--rcm-max-deg", type=float, default=2.0)
    ap.add_argument("--aim", action=argparse.BooleanOptionalAction, default=True,
                    help="inside the skin, pivot the endoscope about the skin entry toward the drill tip (default on)")
    ap.add_argument("--aim-gain", type=float, default=0.3)
    ap.add_argument("--aim-max-deg", type=float, default=1.5, help="per step (the motion limiter allows 3°)")
    ap.add_argument("--aim-dead-deg", type=float, default=6.0, help="no correction while the drill is within this of the view axis")
    ap.add_argument("--visible-trigger", action=argparse.BooleanOptionalAction, default=True,
                    help="trigger only while the drill tip is on the endoscope's screen (default on)")
    ap.add_argument("--visible-trigger-rule", choices=TRIGGER_RULES, default="screen",
                    help="screen: the trigger is off only while the tip is off the screen (user, 10-07) · tip: also while bone "
                         "hides it (every run before 10-07 16:00)")
    ap.add_argument("--max-step-mm", type=float, default=2.0, help="per-frame tip move limit (safety.py)")
    ap.add_argument("--max-step-deg", type=float, default=3.0)
    ap.add_argument("--check-image", action="store_true")
    ap.add_argument("--check-at", default=None, help="check mode: frames to compare, e.g. 0,100,250 (states fed from --start-frame)")
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("-o", "--out", default="traj.json")
    return ap


def main() -> None:
    ap = build_parser()
    a = ap.parse_args()
    if not a.check_image and not a.policy_path and not a.demo_replay:
        ap.error("--policy-path needed (or --check-image)")
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
