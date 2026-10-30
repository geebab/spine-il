"""FE recorder v2 sessions (hand demos + robot-origin replays) → a LeRobot v3 dataset (lerobot 0.6.1) for GR00T N1.7.

One episode = one session (mode full, the primary) or one replay window (mode windows: hand first-cut − 2 s + 30 s,
the windows the robot runs replayed — for robot-vs-hand evaluation). Robot sessions only cover the replay window, so
they are the same episode in both modes.

Per frame (fps, default 15 = the FE live drill loop rate, see README):
  observation.images.endo  video 256² — the FE viewer's endoscope screen of that camera frame (render_cache.py)
  observation.state        (15,) float32, bone/scene frame (LPS mm, registration T0 of the session):
                           endo tip xyz, endo quat wxyz, drill tip xyz, drill quat wxyz, drill trigger (0/1)
  action                   (15,) = observation.state of the next frame (absolute next pose + trigger)
                           --done-output: (16,) — plus drill.done, 1 when that next frame lies after the session's last cut
                           (phase ≥ 5): the model says itself that the drilling is over. Frames after the last cut then
                           keep the drilling instruction as their task (the text must not give the end away), and
                           observation.state gets a 16th value pad.zero that is always 0: LeRobot's relative-action code
                           (GR00T, pi0: action − state) needs a state as long as the action, and a 15-value state under a
                           16-value action stops it ("size of tensor a (15) must match … (16)")
                           --progress-output: the same with drill.progress as the 16th value = cut.progress of that next
                           frame (user 10-09: "끝 출력을 0/1 대신 진행도로 바꾼다") — a target on every frame instead of a
                           flag that is 1 in 3.6 % of them; the loop ends the drilling when the predicted value stays at
                           the end level
  phase.index              (1,) int64 1..6 (phases.py); the frame's LeRobot task = that phase's instruction
                           (--instructions lang1 / lang2: in plain words, naming the recording's level and extent —
                           instructions.py; lang2 --cond-dropout P: a share P of the drilling frames names no extent;
                           --decision-copies K: the frames where the goals part again as K episodes, meta decision_copy)
  cut.progress             (1,) float32 0..1: the bone removed up to this frame / the session's final removed volume —
                           the replay render's own removedMm3 per frame (render cache; what the evaluation measures), or
                           without cached frames the recording's cut rows weighted by the voxels each changed (up to
                           0.12 off the render's fraction). 0 before the first cut, 1 from the last cut on
  track.valid              (2,) endo, drill: 1 = a tracker sample within 125 ms (else the pose is held)
  robot.joints             (12,) encoder deg, left arm (endo) 6 + right arm (drill) 6 — robot episodes, else 0
  robot.joints_target      (12,) the goal the arm server held at the *next* frame time (commanded joint goal)
  robot.joints_valid       (1,) 1 = robot episode
  time.src_s               (1,) s since the session's first camera frame, on the session's own clock
Keys without an observation./action. prefix are stored but not fed to policies (GR00T reads observation.state,
observation.images.*, task; any action.* key would become a policy output).

Pose source: the tracker's filtered tip (f.scene.tip = T0·tip) and filtered constellation quaternion rotated into the
scene (T0.R · R(quat)); quaternions are put in the hemisphere of a fixed per-tool reference (Q_REF), so q and −q never
alternate (all rev.4 samples are < 35° from it). Between tracker samples: position linear, rotation nlerp (samples are
11–33 ms apart: < 0.01° from slerp); gaps > 250 ms are held, not bridged. Lost and jump-hold rows are not samples.

Robot episodes: frame k is at hand time k/fps (the replay speed is undone: render and sampling use fps × speed on the
robot session's clock) unless --robot-time robot. Arm logs are put on the tExp axis via tr·1000 + clock.off (sources.py).

--layout openh writes the same episodes in the Open-H-Embodiment v2 layout instead (openh.py: metres, xyzw, one
[x_m, y_m, z_m, qx, qy, qz, qw, gripper] block per tool, observation.meta.*, instruction.text, splits, camera intrinsics,
meta/README.md) — a dataset of its own for the contribution; the training datasets keep the layout above.

Usage:
  python export_lerobot.py --out ~/il-robot/datasets/spine_rev4_20260929 --mode full --include hand,robot --fps 15
  python export_lerobot.py --validate-only --out <dir>
  python export_lerobot.py --layout openh --out <dir> --include hand --fps 20 --size 480 --cache <step cache of that fps and size>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import phases as PH  # noqa: E402
    import instructions as INS  # noqa: E402
    import render_cache as RC  # noqa: E402
    import sources as S  # noqa: E402
    import actions as AC  # noqa: E402
    import openh as OH  # noqa: E402
    import trim as TR  # noqa: E402
else:
    from . import actions as AC
    from . import openh as OH
    from . import phases as PH
    from . import instructions as INS
    from . import render_cache as RC
    from . import sources as S
    from . import trim as TR

from lerobot_robot_spinearm.recording import parse_rigid, quat_to_R, read_rows

ROBOT_TYPE = "tracked_endo_drill_v1"
MAX_GAP_MS = 250.0
VALID_MS = 125.0
Q_REF = {"endo": np.array([0.5617, -0.6076, -0.519, 0.2145]), "drill": np.array([0.6341, 0.6398, 0.4009, 0.1668])}
ACTION_MODE = ["absolute"]   # set from --action-mode in main()
TRIM = [None]                # --trim robot: {"pre_s", "post_s", "spans": {short id: [t_a, t_b]}} — hand episodes only (trim.py)
STATE_PHASE = [False]        # --state-phase: observation.state += one-hot of the phase (6) — for policies without language
DONE_OUTPUT = [False]        # --done-output: action += drill.done (1 = the frame the action leads to is after the last cut); the
                             # frames after the last cut keep the drilling instruction as their task (see main)
DONE_NAME = "drill.done"
INSTRUCTIONS = ["phase1"]    # --instructions: the set of texts a frame's task comes from (dataset/instructions.py)
# --wording (lang1): one wording per episode (the first exports of 10-13/14) or per frame — L30a, trained per episode, answered
# an unseen wording of its own goal like an unrelated sentence (lang_check 10-14): it had tied sentences to episodes, not meanings
WORDING = ["frame"]
# --cond-dropout P (lang2): a drilling frame's sentence names no extent with probability P, drawn per frame and fixed by the
# frame's id — the model learns the goal-free answer too (what guidance pushes away from; 10-14)
COND_DROPOUT = [0.0]
# --decision-copies K (lang1 / lang2): where the two goals part, the frames before and after are written K more times as
# episodes of their own (meta "decision_copy"). 10-14: those frames were 2 % of the data (1,240 of 62,952) — a contralateral
# recording at its first drilling frame past the midline − CROSS_MM, an ipsilateral one at its last drilling frame. A
# neutral recording has no such point. GR00T masks the actions past an episode's end (action_is_pad), so a copy's last
# frames teach no stop
DECISION = [{"copies": 0, "frames": 40}]
PROGRESS_OUTPUT = [False]    # --progress-output: action += drill.progress (cut.progress of the frame the action leads to)
PROGRESS_NAME = "drill.progress"
PAD_NAME = "pad.zero"        # --done-output: the state's 16th value, always 0 (see the header)
PH_WITHDRAW = 5              # phases.py: 4 = drilling (first … last cut), 5 = withdraw — "after the last cut" starts here
LAYOUT = ["spine"]           # --layout: spine = ours (below) | openh = Open-H-Embodiment v2 (openh.py)
EXCLUDE = [{}]               # --exclude: {session id | short id: [[a_ms, b_ms], …]} stamp ranges (tExp) left out — the episode is split there
VIDEO_FILE_MB = 60   # see write_dataset: a video file must stay well under 1,024 s
MIN_PIECE = 15               # frames: a shorter piece between two excluded ranges is dropped
PHASE_NAMES = [f"phase.{k}" for k in range(1, 7)]
TOOL_NAMES = [f"{tool}.{c}" for tool in ("endo", "drill") for c in ("x", "y", "z", "qw", "qx", "qy", "qz")]
STATE_NAMES = TOOL_NAMES + ["drill.trigger"]
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
JOINT_NAMES = [f"left_{j}" for j in JOINTS] + [f"right_{j}" for j in JOINTS]


def end_names() -> list[str]:
    """The 16th action value of this export, if any: the end flag or the progress."""
    return [DONE_NAME] if DONE_OUTPUT[0] else [PROGRESS_NAME] if PROGRESS_OUTPUT[0] else []


def features(views=("endo",), size: int = 256, image_dtype: str = "video", action_mode: str = "absolute") -> dict:
    f = {}
    for v in views:
        f[f"observation.images.{v}"] = {"dtype": image_dtype, "shape": (size, size, 3),
                                        "names": ["height", "width", "channels"]}
    f.update({
        "observation.state": ({"dtype": "float32", "shape": (len(STATE_NAMES) + len(end_names()),), "names": STATE_NAMES + [PAD_NAME] * len(end_names())}
                              if not STATE_PHASE[0] else
                              {"dtype": "float32", "shape": (len(STATE_NAMES) + 6,), "names": STATE_NAMES + PHASE_NAMES}),
        "action": ({"dtype": "float32", "shape": (len(STATE_NAMES) + len(end_names()),), "names": STATE_NAMES + end_names()}
                   if action_mode == "absolute"
                   else {"dtype": "float32", "shape": (len(AC.DELTA_NAMES),), "names": AC.DELTA_NAMES}),
        "phase.index": {"dtype": "int64", "shape": (1,), "names": None},
        "cut.progress": {"dtype": "float32", "shape": (1,), "names": None},
        "track.valid": {"dtype": "float32", "shape": (2,), "names": ["endo", "drill"]},
        "robot.joints": {"dtype": "float32", "shape": (12,), "names": JOINT_NAMES},
        "robot.joints_target": {"dtype": "float32", "shape": (12,), "names": JOINT_NAMES},
        "robot.joints_valid": {"dtype": "float32", "shape": (1,), "names": None},
        "time.src_s": {"dtype": "float32", "shape": (1,), "names": None},
    })
    return f


# ---------------------------------------------------------------- sampling
@dataclass
class ToolSamples:
    t: np.ndarray      # epoch ms (tExp)
    pos: np.ndarray    # (N, 3) scene mm
    quat: np.ndarray   # (N, 4) wxyz scene, hemisphere of Q_REF


def tool_samples(rows: list[dict], tool: str, T0) -> ToolSamples:
    R0, t0 = T0
    t, p, q = [], [], []
    for r in rows:
        if r.get("k") != "f" or (r.get("tool") or "endo") != tool or r.get("lost") or r.get("hold"):
            continue
        te, quat = r.get("tExp"), r.get("quat")
        tip = (r.get("scene") or {}).get("tip")
        if tip is None and r.get("tip") is not None:
            tip = (R0 @ np.asarray(r["tip"], float) + t0).tolist()
        if not isinstance(te, (int, float)) or tip is None or quat is None or len(quat) != 4:
            continue
        t.append(te)
        p.append(tip)
        q.append(quat)
    if not t:
        raise ValueError(f"no valid {tool} samples")
    t = np.asarray(t, float)
    o = np.argsort(t, kind="stable")
    t, p, q = t[o], np.asarray(p, float)[o], np.asarray(q, float)[o]
    keep = np.r_[True, np.diff(t) > 0]
    t, p, q = t[keep], p[keep], q[keep]
    Rs = np.einsum("ij,njk->nik", R0, quat_to_R(q))
    qs = rot_to_quat(Rs)
    qs *= np.where(qs @ Q_REF[tool] < 0, -1.0, 1.0)[:, None]
    return ToolSamples(t, p, qs)


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """(N, 3, 3) → (N, 4) wxyz unit quaternions (Shepperd, vectorised)."""
    R = np.asarray(R, float)
    n = R.shape[0]
    q = np.empty((n, 4))
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    d = np.stack([tr, R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]], 1)
    c = np.argmax(d, 1)
    for k in range(4):
        m = c == k
        if not m.any():
            continue
        r = R[m]
        if k == 0:
            s = 2.0 * np.sqrt(1.0 + tr[m])
            q[m] = np.stack([s / 4, (r[:, 2, 1] - r[:, 1, 2]) / s, (r[:, 0, 2] - r[:, 2, 0]) / s,
                             (r[:, 1, 0] - r[:, 0, 1]) / s], 1)
        elif k == 1:
            s = 2.0 * np.sqrt(np.maximum(1.0 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2], 1e-300))
            q[m] = np.stack([(r[:, 2, 1] - r[:, 1, 2]) / s, s / 4, (r[:, 0, 1] + r[:, 1, 0]) / s,
                             (r[:, 0, 2] + r[:, 2, 0]) / s], 1)
        elif k == 2:
            s = 2.0 * np.sqrt(np.maximum(1.0 + r[:, 1, 1] - r[:, 0, 0] - r[:, 2, 2], 1e-300))
            q[m] = np.stack([(r[:, 0, 2] - r[:, 2, 0]) / s, (r[:, 0, 1] + r[:, 1, 0]) / s, s / 4,
                             (r[:, 1, 2] + r[:, 2, 1]) / s], 1)
        else:
            s = 2.0 * np.sqrt(np.maximum(1.0 + r[:, 2, 2] - r[:, 0, 0] - r[:, 1, 1], 1e-300))
            q[m] = np.stack([(r[:, 1, 0] - r[:, 0, 1]) / s, (r[:, 0, 2] + r[:, 2, 0]) / s,
                             (r[:, 1, 2] + r[:, 2, 1]) / s, s / 4], 1)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def sample_tool(ts: ToolSamples, tq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pose at query times tq (epoch ms) → (pos (M,3), quat (M,4), valid (M,)). Linear / nlerp between neighbours
    ≤ MAX_GAP_MS apart, else the earlier sample is held (the first sample before the recording's first)."""
    t = ts.t
    i = np.clip(np.searchsorted(t, tq, side="right") - 1, 0, len(t) - 1)
    j = np.minimum(i + 1, len(t) - 1)
    span = t[j] - t[i]
    bridge = (j > i) & (tq > t[i]) & (span <= MAX_GAP_MS) & (tq >= t[0])
    a = np.where(bridge, (tq - t[i]) / np.where(span > 0, span, 1.0), 0.0)[:, None]
    pos = ts.pos[i] + a * (ts.pos[j] - ts.pos[i])
    qj = ts.quat[j] * np.where(np.sum(ts.quat[i] * ts.quat[j], 1) < 0, -1.0, 1.0)[:, None]
    q = ts.quat[i] + a * (qj - ts.quat[i])
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    near = np.minimum(np.abs(tq - t[i]), np.where(j > i, np.abs(t[j] - tq), np.inf))
    valid = (near <= VALID_MS) | (bridge & (span <= 2 * VALID_MS))
    return pos, q, valid.astype(np.float32)


def trigger_series(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Drill trigger (btn = 1 and not btnStale) per drill row → (t epoch ms, 0/1), time-sorted."""
    t, v = [], []
    for r in rows:
        if r.get("k") == "f" and r.get("tool") == "drill" and isinstance(r.get("tExp"), (int, float)):
            b = r.get("btn")
            t.append(r["tExp"])
            v.append(1.0 if (b == 1 and not r.get("btnStale")) else 0.0)
    if not t:
        return np.zeros(0), np.zeros(0)
    o = np.argsort(t, kind="stable")
    return np.asarray(t, float)[o], np.asarray(v, float)[o]


def zoh(t_src, v_src, tq, default=0.0):
    t_src = np.asarray(t_src, float)
    if len(t_src) == 0:
        return np.full((len(tq),) + np.asarray(v_src).shape[1:], default, float)
    i = np.searchsorted(t_src, tq, side="right") - 1
    v = np.asarray(v_src, float)[np.clip(i, 0, len(t_src) - 1)]
    v[i < 0] = default
    return v


def duty(t_src, v_src, a_ms, b_ms) -> float | None:
    """Time-weighted fraction of [a, b) with the trigger on (the source-side duty cycle)."""
    if len(t_src) == 0 or b_ms <= a_ms:
        return None
    g = np.arange(a_ms, b_ms, 5.0)
    return float(zoh(t_src, v_src, g).mean())


# ---------------------------------------------------------------- episodes
@dataclass
class Episode:
    sess: "S.Session"
    kind: str                 # full | window
    plan: "RC.RenderPlan"
    ks: list[int]             # grid frames
    window: dict | None = None
    meta: dict = field(default_factory=dict)


def loader_origin_ms(rows: list[dict]) -> float:
    """recording.load_recording's origin: the first valid pose row of either tool (tExp)."""
    return min(r["tExp"] for r in rows if r.get("k") == "f" and not r.get("lost") and not r.get("hold")
               and isinstance(r.get("tExp"), (int, float)) and r.get("pos") and r.get("quat"))


def plan_episodes(sessions, mode: str, fps: float, robot_time: str, log=print) -> tuple[list[Episode], list[dict]]:
    eps, skipped = [], []
    for s in sessions:
        try:
            tl = RC.timeline(s.path)
            plan = plan_for(s, fps, robot_time, tl)
        except (ValueError, KeyError, OSError) as e:
            skipped.append({"session": s.session_id, "why": f"계획 실패: {e}"})
            continue
        if s.source == "hand" and mode == "windows":
            _, rows, _ = read_rows(s.path, frozenset({"k", "r", "tool", "lost", "hold", "pos", "quat", "tExp"}))
            origin = loader_origin_ms(rows)
            cut0 = first_bone_cut_ms(s.path)
            from_s = S.HAND_WINDOWS_FROM_S.get(s.sid)
            src = "rev.4 표"
            if from_s is None:
                from_s, src = float(round((cut0 - origin) / 1000.0 - 2.0)), "첫 절삭 − 2 s"
            a_ms, b_ms = origin + from_s * 1000.0, origin + (from_s + S.WINDOW_LEN_S) * 1000.0
            k0 = max(plan.k0, math.ceil((a_ms - plan.t_exp_first) / plan.step_ms - 1e-9))
            k1 = min(plan.k1, math.floor((b_ms - plan.t_exp_first) / plan.step_ms + 1e-9))
            eps.append(Episode(s, "window", plan, list(range(k0, k1 + 1)),
                               {"from_s": from_s, "len_s": S.WINDOW_LEN_S, "source": src,
                                "first_cut_minus_2_s": round((cut0 - origin) / 1000.0 - 2.0, 3)}))
        else:
            eps.append(Episode(s, "full" if s.source == "hand" else "replay", plan,
                               list(range(plan.k0, plan.k1 + 1))))
        eps[-1].meta["robot_time"] = robot_time if s.source == "robot" else None
    return eps, skipped


def plan_for(s, fps, robot_time, tl):
    if s.source == "robot" and robot_time == "robot":
        speed = s.meta["speed"]
        s.meta = {**s.meta, "speed": 1.0}  # render/sample at fps on the robot clock
        try:
            return RC.plan_for(s, fps, tl)
        finally:
            s.meta["speed"] = speed
    return RC.plan_for(s, fps, tl)


def first_bone_cut_ms(path) -> float:
    best = math.inf
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if '"cut"' in line[:24] and '"bone"' in line:
                d = json.loads(line)
                if d.get("k") == "cut" and d.get("target") == "bone" and isinstance(d.get("tExp"), (int, float)):
                    best = min(best, d["tExp"])
    return best


def build_frames(ep: Episode) -> dict:
    """All per-frame arrays of an episode (no images) + its metadata."""
    s, plan = ep.sess, ep.plan
    hdr, rows, _ = read_rows(s.path)
    T0 = parse_rigid(hdr["registration"]["T0"])
    tq = np.array([plan.t_of(k) for k in ep.ks], float)
    st, valid, src_t = [], [], []
    for tool in ("endo", "drill"):
        ts = tool_samples(rows, tool, T0)
        pos, q, v = sample_tool(ts, tq)
        st += [pos, q]
        valid.append(v)
        src_t.append(ts.t[np.clip(np.searchsorted(ts.t, tq, side="right") - 1, 0, len(ts.t) - 1)])   # the sample each pose starts from
    tt, tv = trigger_series(rows)
    trig = zoh(tt, tv, tq)
    state = np.concatenate(st + [trig[:, None]], 1).astype(np.float32)
    # phases on the whole session (seconds from its first camera frame)
    e, d, cuts = PH.from_rows(rows, plan.t_exp_first, T0)
    t_end = max(float(e.t[-1]) if len(e.t) else 0.0, float(d.t[-1]) if len(d.t) else 0.0)
    ph = PH.segment(e, d, cuts, 0.0, t_end, expect_full=(s.source == "hand"))
    t_rel = (tq - plan.t_exp_first) / 1000.0
    phase = ph.phase_at(t_rel)
    # bone removed so far as a fraction of the session's total, from the recording's own cut rows (each weighted by `mod`,
    # the voxels it changed; by 1 when a recording has none). write_dataset replaces it with the replay render's removed
    # volume when the frames come from the render cache — the cut rows are up to 0.12 off that fraction (09-28 sessions)
    ct = sorted((float(r["tExp"]), float(r.get("mod") or 1.0)) for r in rows
                if r.get("k") == "cut" and r.get("target") == "bone" and isinstance(r.get("tExp"), (int, float)))
    # where the recording cut bone (scene frame): the level and the extent its instruction names (instructions.py)
    cc = np.array([c for r in rows if r.get("k") == "cut" and r.get("target") == "bone" and (r.get("mod") or r.get("dv"))
                   for c in (r.get("c") or [])], float).reshape(-1, 3)
    goal = {"level": "L4", "extent": None, "cut_x_p5_mm": round(float(np.percentile(cc[:, 0], 5)), 1) if len(cc) else None}
    if INSTRUCTIONS[0] in ("lang1", "lang2", "lang3"):
        # the level from the cuts themselves; the header's target (recorder, from 10-13: {side, level, procedure}) is the
        # operator's intent — a recording where the two disagree is refused, not labelled by either. (10-15: this ran for
        # lang1 only — a lang2 export called every recording L4, with L4's words and midline; found on the first L3 export)
        goal["level"], goal["level_check"] = INS.level_of(cc)
        want = (hdr.get("target") or {}).get("level")
        if want and want != goal["level"]:
            raise ValueError(f"{s.session_id}: the header says {want}, the cuts lie on {goal['level']} ({goal['level_check']['share']})")
    if len(cc):   # against the midline of the recording's own level (instructions.MIDLINE_X_MM); lang2: clear ones only
        goal["extent"] = INS.extent_label(INSTRUCTIONS[0], cc[:, 0], goal["level"])
        goal["past_midline_mm"] = round(INS.MIDLINE_X_MM[goal["level"]] - float(np.percentile(cc[:, 0], 5)), 1)
    progress = np.zeros(len(tq))
    if ct:
        cw = np.cumsum([w for _, w in ct])
        progress = zoh([t for t, _ in ct], cw / cw[-1], tq, 0.0)
    inside = np.zeros((len(tq), 2), bool)   # per tool: inside the skin at its working portal (trim.py)
    for c, (name, tr) in enumerate((("endo", e), ("drill", d))):
        portal = (ph.info.get("portal_mm") or {}).get(name)
        if portal is not None and len(tr.t):
            inside[:, c] = PH._zoh(tr.t, PH.inside_mask(tr, np.asarray(portal, float)), t_rel, False).astype(bool)
    n = len(tq)
    joints = np.zeros((n, 12), np.float32)
    target = np.zeros((n, 12), np.float32)
    jvalid = np.zeros((n, 1), np.float32)
    align = None
    if s.source == "robot":
        tq_next = np.r_[tq[1:], tq[-1] + plan.step_ms]
        cols, tcols, align = [], [], {}
        for arm in ("endo", "drill"):  # left arm = endoscope, right arm = drill
            tl = S.arm_timeline(s.meta["arm_dirs"][arm])
            if tl["joints"] != JOINTS:
                raise ValueError(f"{arm} arm joints {tl['joints']} ≠ {JOINTS}")
            ts_, q_ = np.asarray(tl["t_state"]), np.asarray(tl["q"])
            cols.append(np.stack([np.interp(tq, ts_, q_[:, c]) for c in range(6)], 1))
            tcols.append(zoh(tl["t_goal"], np.asarray(tl["goal"]).reshape(-1, 6), tq_next, np.nan)
                         if tl["goal"] else np.full((n, 6), np.nan))
            gap = np.diff(ts_)
            align[arm] = {"off_ms": round(tl["off_ms"], 3), "off_spread_ms": round(tl["off_spread_ms"], 4),
                          "state_dt_ms_p50": round(float(np.median(gap)), 2),
                          "state_dt_ms_max": round(float(gap.max()), 1),
                          "frames_outside_state_log": int(((tq < ts_[0]) | (tq > ts_[-1])).sum())}
        joints = np.concatenate(cols, 1).astype(np.float32)
        tgt = np.concatenate(tcols, 1)
        tgt = np.where(np.isnan(tgt), joints, tgt)  # before the first goal: the measured pose
        target = tgt.astype(np.float32)
        jvalid[:] = 1.0
    ftr = next((r for r in rows if r.get("k") == "ftr"), {})
    meta = {
        "source": s.source, "session_id": s.session_id, "recording": str(s.path), "kind": ep.kind,
        "recorder": hdr.get("recorderVersion"), "operator": hdr.get("operator"),
        "registration": (hdr.get("registration") or {}).get("method"),
        "drill_size": (hdr.get("drill") or {}).get("sizeKey"), "drill_radius_mm": (hdr.get("drill") or {}).get("radiusMm"),
        "operator_verdict": ftr.get("operatorVerdict"),
        "t_exp_first_ms": plan.t_exp_first, "t_frame0_ms": float(tq[0]), "t_frame_last_ms": float(tq[-1]),
        "grid": {"k0": ep.ks[0], "k1": ep.ks[-1], "render_fps": plan.render_fps, "step_ms": plan.step_ms},
        "window": ep.window, "phases": ph.table(), "phase_flags": ph.flags, "phase_info": ph.info,
        "task_episode": S.TASK_EPISODE,
        "trigger_duty_source": duty(tt, tv, float(tq[0]), float(tq[-1]) + plan.step_ms),
        "track_valid_frac": [round(float(v.mean()), 4) for v in valid],
        "progress_source": "cut rows (mod-weighted)" if ct else "no cut rows",
        "goal": goal,
    }
    if s.source == "robot":
        m = s.meta
        meta.update({"replay_of": s.replay_of, "run": m.get("run"), "tag": m.get("tag"), "speed": m.get("speed"),
                     "robot_time": ep.meta.get("robot_time"),
                     "replay_from_s": m.get("from_s"), "replay_max_s": m.get("max_s"),
                     "overrides": m.get("overrides"), "anchor": m.get("anchor"), "cut_agreement": m.get("iou"),
                     "contaminated": bool(m.get("contaminated")), "contaminated_reason": m.get("contaminated_reason"),
                     "joint_alignment": align})
    return {"tq": tq, "state": state, "valid": np.stack(valid, 1), "phase": phase, "progress": np.asarray(progress, np.float32), "joints": joints,
            "target": target, "jvalid": jvalid, "t_rel": t_rel.astype(np.float32), "meta": meta,
            "inside": inside, "t_cut0": ph.t_cut0, "t_out": ph.t_out, "trig_src": (tt, tv), "src_t": np.stack(src_t, 1)}


def frame_task(meta: dict, phase: int, i: int, copy: int = 0) -> str:
    """The task of frame i of a session: its goal's sentence for the phase (instructions.py), in a wording drawn per frame
    (or once per episode, --wording episode); --cond-dropout turns a drilling frame's into the neutral one. copy > 0 = a
    --decision-copies episode: its wordings and dropouts are drawn anew."""
    g, sid, tail = meta["goal"], meta["session_id"], f"/c{copy}" if copy else ""
    extent = g["extent"] or INS.DEFAULT_EXTENT
    if COND_DROPOUT[0] and int(phase) == 4:
        u = int(hashlib.sha256(f"{sid}/{i}{tail}/drop".encode()).hexdigest()[:13], 16) / 16 ** 13
        extent = INS.NEUTRAL if u < COND_DROPOUT[0] else extent
    return INS.text(INSTRUCTIONS[0], phase, g["level"], extent, key=sid + ("" if WORDING[0] == "episode" else f"/{i}") + tail,
                    legacy=PH.PHASES)


def decision_frame(fr: dict) -> tuple[int | None, str | None]:
    """Where a clearly labelled recording's two goals part (--decision-copies): a contralateral one at its first drilling
    frame whose drill tip lies past the midline − CROSS_MM with the trigger on ("fork"), an ipsilateral one at its last
    drilling frame ("end"). (None, None) for a neutral or unlabelled one, or a contralateral one that never gets there."""
    g, ph = fr["meta"]["goal"], fr["phase"]
    if g.get("extent") == "contralateral":
        k = np.flatnonzero((fr["state"][:, 14] > 0.5) & (ph == 4) & (fr["state"][:, 7] < INS.EXTENT_X_MM[g["level"]]))
        return (int(k[0]), "fork") if len(k) else (None, None)
    if g.get("extent") == "ipsilateral":
        k = np.flatnonzero(ph == 4)
        return (int(k[-1]), "end") if len(k) else (None, None)
    return None, None


# ---------------------------------------------------------------- writing
def write_dataset(eps: list[Episode], out: Path, fps: float, views, size: int, images: str, image_dtype: str,
                  cache: Path, repo_id: str, log=print) -> tuple[list[dict], list[dict]]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image

    use_views = views if images != "none" else ()
    openh = LAYOUT[0] == "openh"
    joints = any(ep.sess.source == "robot" for ep in eps)   # openh: joint features only when robot episodes are exported
    feats = OH.features(use_views, size, image_dtype, joints) if openh else features(use_views, size, image_dtype, ACTION_MODE[0])
    # video files of at most VIDEO_FILE_MB: lerobot finds a frame by its time inside the file, as float32 with a tolerance
    # of 0.1 ms, and above 1,024 s float32 steps are 0.12 ms — the 24-session set in lerobot's default 200 MB files (26 min
    # each at 256 px) failed its first batch with FrameTimestampError (10-07). 60 MB keeps a 256 px file under ~8 min.
    ds = LeRobotDataset.create(repo_id=repo_id, fps=int(round(fps)), features=feats, root=out,
                               robot_type=ROBOT_TYPE, use_videos=(image_dtype == "video"), video_files_size_in_mb=VIDEO_FILE_MB,
                               **({"tolerance_s": OH.TOLERANCE_S} if openh else {}))
    written, skipped = [], []
    try:
        for ep in eps:
            name = f"{ep.sess.source} {ep.sess.sid} {ep.sess.meta.get('tag', '')}".strip()
            frames = None
            if use_views:
                frames = RC.cached_frames(cache, ep.plan, views, size)
                if frames is None:
                    skipped.append({"session": ep.sess.session_id, "why": "렌더 캐시 미완(render_cache.py 대기)"})
                    log(f"[export] skip {name}: render cache incomplete")
                    continue
            try:
                fr = build_frames(ep)
            except (ValueError, KeyError, OSError) as e:
                skipped.append({"session": ep.sess.session_id, "why": f"{type(e).__name__}: {e}"})
                log(f"[export] skip {name}: {e}")
                continue
            if frames is not None and not openh:   # the replay render's own removed volume per frame (what the evaluation measures)
                try:
                    met = RC.cached_metrics(cache, ep.plan, views, size)
                except (OSError, KeyError, ValueError):   # a cache without the viewer's per-frame metrics: the cut rows stay
                    met = {}
                vol = lambda r: float((r.get("bridge") or {}).get("removedMm3") or (r.get("seek") or {}).get("removedMm3") or 0.0)
                fin = max((vol(r) for r in met.values()), default=0.0)   # the session's total, also for an episode that is a part of it
                if fin > 0 and all(k in met for k in ep.ks):
                    fr["progress"] = np.clip(np.array([vol(met[k]) for k in ep.ks], np.float32) / np.float32(fin), 0.0, 1.0)
                    fr["meta"]["progress_source"] = "render removedMm3"; fr["meta"]["removed_mm3_final"] = round(fin, 3)
            n = len(ep.ks) - 1  # the last grid frame only serves as the previous frame's action
            if n < 2:
                skipped.append({"session": ep.sess.session_id, "why": f"프레임 {n + 1}개"})
                continue
            i0 = 0
            if TRIM[0] and ep.sess.source == "hand" and ep.kind == "full":   # the part the arms can play back
                try:
                    ia, ib, tinfo = TR.span(fr["t_rel"], fr["inside"], fr["t_cut0"], fr["t_out"], TRIM[0]["pre_s"],
                                            TRIM[0]["post_s"], (TRIM[0].get("spans") or {}).get(ep.sess.sid))
                except ValueError as e:
                    skipped.append({"session": ep.sess.session_id, "why": f"trim: {e}"})
                    log(f"[export] skip {name}: trim: {e}")
                    continue
                i0, n = ia, min(ib, n)        # frames i0 … n−1 are written, frame n is the last action's target
                fr["meta"]["trim"] = tinfo
                fr["meta"]["t_frame0_ms"], fr["meta"]["t_frame_last_ms"] = float(fr["tq"][i0]), float(fr["tq"][n])
            if openh:
                fr["pose16"], tools = OH.pose16(fr["state"]), OH.tool_text(fr["meta"])
                fr["meta"]["split"] = ep.meta.get("split")
            if frames is not None:  # the render API's replay verdict of every chunk used
                dirs = {p[use_views[0]].parent.parent.parent for p in frames.values()}
                st = [json.loads((d / "DONE").read_text() or "{}") for d in sorted(dirs)]
                fr["meta"]["render"] = {"pass": all(x.get("pass") for x in st),
                                        "fail_reasons": sorted({r for x in st for r in (x.get("failReasons") or [])}),
                                        "jobs": len(st)}
            # an episode per run of kept frames: one, unless stamp ranges of the session are excluded (--exclude)
            pieces, ex = [(i0, n)], EXCLUDE[0].get(ep.sess.session_id) or EXCLUDE[0].get(ep.sess.sid)
            if ex:
                ok = np.zeros(len(fr["tq"]), bool); ok[i0:n + 1] = True
                for a_ms, b_ms in ex:
                    ok &= ~((fr["tq"] >= a_ms) & (fr["tq"] <= b_ms))
                e = np.flatnonzero(np.diff(np.r_[False, ok, False].astype(np.int8)))
                runs = [(int(a), int(b) - 1) for a, b in zip(e[::2], e[1::2])]       # frames a … b−1 written, b = the last target
                pieces = [x for x in runs if x[1] - x[0] >= MIN_PIECE]
                log(f"[export] {name}: {len(ex)} stamp range(s) excluded → {len(pieces)} episode(s)"
                    + (f", {len(runs) - len(pieces)} piece(s) under {MIN_PIECE} frames dropped" if len(runs) > len(pieces) else ""))
            def add(i: int, copy: int = 0) -> None:   # frame i of the session (copy > 0: of a --decision-copies episode)
                if openh:
                    f = OH.frame(fr, i, tools, joints)
                else:
                    act = fr["state"][i + 1] if ACTION_MODE[0] == "absolute" else AC.to_delta(fr["state"][i], fr["state"][i + 1])
                    ph_i = int(fr["phase"][i])
                    if DONE_OUTPUT[0] or PROGRESS_OUTPUT[0]:
                        # the end of the drilling as a 16th action value, about the frame this action leads to: the flag
                        # "it lies after the last cut" (phase ≥ 5; the target train/lerobot-train-rel.py --done-output
                        # gives ACT) or the progress there (bone removed / the session's total). The frame's task must
                        # not give the answer away: after the last cut it stays the drilling instruction
                        end = fr["phase"][i + 1] >= PH_WITHDRAW if DONE_OUTPUT[0] else fr["progress"][i + 1]
                        act = np.concatenate([np.asarray(act, np.float32), [np.float32(end)]])
                        ph_i = min(ph_i, PH_WITHDRAW - 1) if ph_i >= PH_WITHDRAW else ph_i
                    obs_state = fr["state"][i]
                    if DONE_OUTPUT[0] or PROGRESS_OUTPUT[0]:
                        obs_state = np.concatenate([np.asarray(obs_state, np.float32), [np.float32(0.0)]])   # pad.zero
                    if STATE_PHASE[0]:
                        oh = np.zeros(6, np.float32); oh[int(fr["phase"][i]) - 1] = 1.0
                        obs_state = np.concatenate([np.asarray(obs_state, np.float32), oh])
                    f = {"observation.state": obs_state, "action": act,
                         "phase.index": np.array([fr["phase"][i]], np.int64),
                         "cut.progress": np.array([fr["progress"][i]], np.float32), "track.valid": fr["valid"][i],
                         "robot.joints": fr["joints"][i], "robot.joints_target": fr["target"][i],
                         "robot.joints_valid": fr["jvalid"][i], "time.src_s": np.array([fr["t_rel"][i]], np.float32),
                         "task": frame_task(fr["meta"], ph_i, i, copy)}
                for v in use_views:
                    im = Image.open(frames[ep.ks[i]][v]).convert("RGB")
                    if im.size != (size, size):
                        im = im.resize((size, size))
                    f[OH.image_key(v) if openh else f"observation.images.{v}"] = np.asarray(im)
                ds.add_frame(f)

            first = len(written)
            for part, (pa, pb) in enumerate(pieces):
                for i in range(pa, pb):
                    add(i)
                ds.save_episode()
                m = dict(fr["meta"])
                m.update(episode_index=len(written), n_frames=pb - pa,
                         trigger_duty_dataset=round(float(fr["state"][pa:pb, -1].mean()), 4),
                         phase_counts={int(k): int(v) for k, v in zip(*np.unique(fr["phase"][pa:pb], return_counts=True))})
                if pa or "trim" in m or ex:   # the source duty of the kept part, for validate()
                    m["trigger_duty_source"] = duty(*fr["trig_src"], float(fr["tq"][pa]), float(fr["tq"][pb]))
                if ex:
                    m.update(part=[part + 1, len(pieces)], excluded_ms=ex, t_frame0_ms=float(fr["tq"][pa]), t_frame_last_ms=float(fr["tq"][pb]))
                written.append(m)
                log(f"[export] ep {m['episode_index']:3d} {name}: {pb - pa} frames, phases {m['phase_counts']}"
                    + (f", trimmed {m['trim']['t_a']:.1f}–{m['trim']['t_b']:.1f} s" if "trim" in m else "")
                    + (f", part {part + 1}/{len(pieces)} ({fr['t_rel'][pa]:.1f}–{fr['t_rel'][pb]:.1f} s)" if ex else ""))
            # where the two goals part, DECISION copies more times — inside the written piece that holds that frame
            k_dec, kind = decision_frame(fr) if DECISION[0]["copies"] and not openh else (None, None)
            hit = [j for j, (pa, pb) in enumerate(pieces) if k_dec is not None and pa <= k_dec < pb]
            if DECISION[0]["copies"] and not openh and not hit:
                log(f"[export] {name}: no decision copies (goal {fr['meta']['goal'].get('extent')}"
                    + (", its point lies outside the written frames)" if k_dec is not None else ", no point where the goals part)"))
            for j in hit:
                pa, pb = pieces[j]
                a_, b_ = max(pa, k_dec - DECISION[0]["frames"]), min(pb, k_dec + DECISION[0]["frames"])
                for c in range(1, DECISION[0]["copies"] + 1):
                    for i in range(a_, b_):
                        add(i, c)
                    ds.save_episode()
                    m = dict(fr["meta"])
                    m.update(episode_index=len(written), n_frames=b_ - a_,
                             trigger_duty_dataset=round(float(fr["state"][a_:b_, -1].mean()), 4),
                             phase_counts={int(k): int(v) for k, v in zip(*np.unique(fr["phase"][a_:b_], return_counts=True))},
                             trigger_duty_source=duty(*fr["trig_src"], float(fr["tq"][a_]), float(fr["tq"][b_])),
                             t_frame0_ms=float(fr["tq"][a_]), t_frame_last_ms=float(fr["tq"][b_]),
                             decision_copy={"copy": c, "of": first + j, "kind": kind, "frames_of": [a_ - pa, b_ - pa],
                                            "decision_frame": k_dec - a_})
                    written.append(m)
                log(f"[export] {name}: {DECISION[0]['copies']} decision copies of frames {a_ - pa}–{b_ - pa} of ep {first + j} "
                    f"({kind} at {k_dec - pa}, goal {fr['meta']['goal']['extent']})")
    finally:
        ds.finalize()
    if openh:   # after finalize(): its writers own meta/
        sids = sorted({m["session_id"] for m in written})
        OH.finish(ds, out, written, fps, use_views, size, image_optics(cache, sids), log, cuts=image_cuts(cache, sids))
    return written, skipped


def image_optics(cache: Path, sessions: list[str]) -> str | None:
    """The endoscope medium of the cached images of these sessions (chunk manifest "optics", render_step.py --optics):
    air | water, "recording" for chunks rendered with the recording's own flag, "mixed" if the chunks disagree."""
    seen = set()
    for s in sessions:
        for f in Path(cache).glob(f"{s}/*/c*/manifest.json"):
            seen.add(json.loads(f.read_text()).get("optics") or "recording")
    return None if not seen else seen.pop() if len(seen) == 1 else "mixed"


def image_cuts(cache: Path, sessions: list[str]) -> str | None:
    """Whose bone the cached images show: "recorded" — the recording replayed with its own cut rows (render_cache.py: the
    bone as the operator saw it) · "simulated" — the dataset poses fed to a step session that cut again (render_step.py) ·
    "mixed" · None without images. Training images are replay renders since 10-06 (demo-images-from-recorded-cuts)."""
    seen = set()
    for s in sessions:
        for f in Path(cache).glob(f"{s}/*/c*/manifest.json"):
            seen.add("recorded" if json.loads(f.read_text()).get("cuts") == "recorded" else "simulated")
    return None if not seen else seen.pop() if len(seen) == 1 else "mixed"


def write_sidecar(out: Path, args: dict, written: list[dict], skipped: list[dict]) -> None:
    d = out / "meta" / "spine_il"
    d.mkdir(parents=True, exist_ok=True)
    openh = LAYOUT[0] == "openh"
    if openh:   # a dataset meant to leave the lab: no operator names, no local paths
        ops: dict = {}
        written = [OH.scrub(m, ops) for m in written]
        args = {k: v.replace(str(Path.home()), "~") for k, v in args.items()}
    with open(d / "episodes.jsonl", "w") as fh:
        for m in written:
            fh.write(json.dumps(m, ensure_ascii=False, default=float) + "\n")
    splits = {"hand": [m["episode_index"] for m in written if m["source"] == "hand"],
              "robot": [m["episode_index"] for m in written if m["source"] == "robot"],
              "robot_clean": [m["episode_index"] for m in written if m["source"] == "robot" and not m.get("contaminated")],
              "contaminated": [m["episode_index"] for m in written if m.get("contaminated")],
              "decision_copies": [m["episode_index"] for m in written if m.get("decision_copy")]}
    (d / "splits.json").write_text(json.dumps(splits, indent=1))
    info = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "args": args, "layout": LAYOUT[0],
            "state_names": OH.STATE_NAMES if openh else STATE_NAMES,
            "units": {"position": ("m" if openh else "mm") + ", scene/bone frame (LPS), registration T0 of each session",
                      "quat": ("xyzw" if openh else "wxyz") + ", tool constellation → scene; hemisphere of q_ref (wxyz)",
                      "trigger": "0/1", "joints": ("rad" if openh else "deg") + ", arm angles (server encoder / goal)"},
            "q_ref": {k: v.tolist() for k, v in Q_REF.items()}, "phases": PH.PHASES,
            "instructions": {**INS.table(INSTRUCTIONS[0], PH.PHASES), "wording": WORDING[0],
                             **({"cond_dropout": COND_DROPOUT[0]} if COND_DROPOUT[0] else {}),
                             **({"decision_copies": {**DECISION[0], "episodes": sum(1 for m in written if m.get("decision_copy"))}}
                                if DECISION[0]["copies"] else {})},
            "task_episode": S.TASK_EPISODE,
            "action": "absolute: action[k] = observation.state[k+1]; relative (delta) = action − observation.state "
                      "for xyz, q_state⁻¹ ⊗ q_action for rotation, trigger stays absolute",
            **({"done_output": {"name": DONE_NAME, "index": len(STATE_NAMES),
                                "is": "1 when the frame the action leads to (k+1) has phase.index >= 5, i.e. lies after the session's "
                                      "last cut; 0 before. Absolute like the trigger (never a relative action). A model reading it "
                                      "says the drilling is over; the robot loop ends the drilling when it stays above 0.5 for 5 steps",
                                "task": "frames after the last cut carry the drilling instruction (phase 4's) as their task — the "
                                        "instruction must not tell the model that the drilling is over. phase.index keeps the truth",
                                "state_pad": f"observation.state has a 16th value {PAD_NAME} that is always 0 — only so that the state is "
                                             "as long as the action (LeRobot's relative-action code subtracts state[:16] from "
                                             "action[:16]; drill.trigger and drill.done must be excluded from relative actions: "
                                             "relative_exclude_joints=[\"trigger\", \"done\"]). At inference send 0"}}
               if DONE_OUTPUT[0] else {}),
            **({"progress_output": {"name": PROGRESS_NAME, "index": len(STATE_NAMES),
                                    "is": "cut.progress of the frame the action leads to (k+1): bone removed up to that frame / the "
                                          "session's final removed volume, 0 before the first cut … 1 from the last cut on. Absolute "
                                          "like the trigger (never a relative action). The robot loop ends the drilling when the "
                                          "predicted value stays at its end level (inference/rollout.py END_LEVEL) for 5 steps",
                                    "source": sorted({m.get("progress_source") for m in written}),
                                    "task": "frames after the last cut carry the drilling instruction (phase 4's) as their task — the "
                                            "instruction must not tell the model that the drilling is over. phase.index keeps the truth",
                                    "state_pad": f"observation.state has a 16th value {PAD_NAME} that is always 0 — only so that the state "
                                                 "is as long as the action (LeRobot's relative-action code subtracts state[:16] from "
                                                 "action[:16]; drill.trigger and drill.progress must be excluded from relative actions: "
                                                 "relative_exclude_joints=[\"trigger\", \"progress\"]). At inference send 0"}}
               if PROGRESS_OUTPUT[0] else {}),
            "skipped": skipped,
            # which endoscope medium the images show — train/lerobot-train-rel.py copies it into the policy (action_rep.json)
            # and inference/rollout.py opens its viewer session with it
            "image_optics": image_optics(Path(args["cache"]).expanduser(), sorted({m["session_id"] for m in written})),
            # … and whose bone: the recording's own cuts (replay render) or a re-simulation of the cutting (step render)
            "image_cuts": image_cuts(Path(args["cache"]).expanduser(), sorted({m["session_id"] for m in written}))}
    (d / "export.json").write_text(json.dumps(info, ensure_ascii=False, indent=1, default=float))


# ---------------------------------------------------------------- validation
def validate(out: Path, repo_id: str, log=print) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=out)
    side = [json.loads(x) for x in (out / "meta/spine_il/episodes.jsonl").read_text().splitlines() if x.strip()]
    hf = ds.hf_dataset.with_format("numpy")
    st = np.asarray(hf["observation.state"], float)
    ac = np.asarray(hf["action"], float)
    epi = np.asarray(hf["episode_index"]).reshape(-1)
    ph = np.asarray(hf["phase.index"]).reshape(-1)
    rep = {"episodes": ds.meta.total_episodes, "frames": ds.meta.total_frames, "fps": ds.meta.fps,
           "tasks": len(ds.meta.tasks), "features": {k: list(v["shape"]) for k, v in ds.meta.features.items()}}
    qn = [np.linalg.norm(st[:, a:a + 4], axis=1) for a in (3, 10)]
    rep["quat_norm_err_max"] = float(max(np.abs(q - 1).max() for q in qn))
    rep["nan"] = {"state": int(np.isnan(st).sum()), "action": int(np.isnan(ac).sum())}
    rep["state_min"] = np.round(st.min(0), 2).tolist()
    rep["state_max"] = np.round(st.max(0), 2).tolist()
    names = list(ds.meta.features["action"]["names"] or [])
    pr = np.asarray(hf["cut.progress"], float).reshape(-1) if "cut.progress" in ds.meta.features else None
    if pr is not None:   # the removed fraction: inside 0..1, never falling within an episode, (nearly) 1 after the last cut
        rep["progress"] = {"min": round(float(pr.min()), 4), "max": round(float(pr.max()), 4),
                           "never_falls": bool(all(np.all(np.diff(pr[epi == e]) >= -1e-6) for e in np.unique(epi))),
                           "min_after_last_cut": round(float(pr[ph >= PH_WITHDRAW].min()), 4) if (ph >= PH_WITHDRAW).any() else None,
                           "max_before_first_cut": round(float(pr[ph < 4].max()), 4) if (ph < 4).any() else None,
                           "source": sorted({m.get("progress_source") for m in side if m.get("progress_source")})}
    if names[len(STATE_NAMES):] == [PROGRESS_NAME]:   # --progress-output: the 15 values are the next state, the 16th its progress
        same, flag = [], []
        for e in np.unique(epi):
            ix = np.flatnonzero(epi == e)
            same.append(np.allclose(ac[ix[:-1], :15], st[ix[1:], :15], atol=1e-4))
            flag.append(np.allclose(ac[ix[:-1], 15], pr[ix[1:]], atol=1e-6))
        rep["action_is_next_state"] = bool(all(same)); rep["progress_is_the_target_frames"] = bool(all(flag))
        rep["tasks_text"] = sorted(str(t) for t in ds.meta.tasks.index) if hasattr(ds.meta.tasks, "index") else None
    elif names[len(STATE_NAMES):] == [DONE_NAME]:   # --done-output: the 15 values are the next state, the 16th the end flag
        same, flag = [], []
        for e in np.unique(epi):
            ix = np.flatnonzero(epi == e)
            same.append(np.allclose(ac[ix[:-1], :15], st[ix[1:], :15], atol=1e-4))
            flag.append(np.array_equal(ac[ix[:-1], 15], (ph[ix[1:]] >= PH_WITHDRAW).astype(float)))
        rep["action_is_next_state"] = bool(all(same)); rep["done_is_target_frame_after_last_cut"] = bool(all(flag))
        rep["done_frames"] = int((ac[:, 15] > 0.5).sum()); rep["done_values"] = sorted(set(np.unique(ac[:, 15]).tolist()))
        rep["tasks_text"] = sorted(str(t) for t in ds.meta.tasks.index) if hasattr(ds.meta.tasks, "index") else None
    elif ac.shape[1] == st.shape[1]:
        same = [np.allclose(ac[np.flatnonzero(epi == e)[:-1]], st[np.flatnonzero(epi == e)[1:]], atol=1e-4)
                for e in np.unique(epi)]
        rep["action_is_next_state"] = bool(all(same))
    else:   # delta actions: from_delta(state, action) must give the next state
        errs = []
        for e in np.unique(epi):
            ix = np.flatnonzero(epi == e)
            for i in ix[:-1][::50]:
                nx = AC.from_delta(st[i], ac[i]); errs.append(float(np.abs(nx[:3] - st[i + 1][:3]).max()))
        rep["action_mode"] = "delta"
        rep["delta_roundtrip_mm_max"] = round(max(errs), 6) if errs else None
    rows = []
    for m in side:
        w = epi == m["episode_index"]
        p = ph[w]
        rows.append({"ep": m["episode_index"], "src": m["source"], "id": short_id(m), "n": int(w.sum()),
                     "duty_ds": round(float(st[w, 14].mean()), 3),   # the trigger (a --state-phase state is longer)
                     "duty_src": None if m["trigger_duty_source"] is None else round(m["trigger_duty_source"], 3),
                     "phase_monotonic": bool(np.all(np.diff(p) >= 0)), "phases": sorted(set(p.tolist())),
                     "contaminated": m.get("contaminated", False)})
    rep["per_episode"] = rows
    rep["duty_abs_err_max"] = max((abs(r["duty_ds"] - r["duty_src"]) for r in rows if r["duty_src"] is not None),
                                  default=None)
    item = ds[0]
    rep["sample_keys"] = {k: list(getattr(v, "shape", ())) for k, v in item.items() if hasattr(v, "shape")}
    rep["sample_task"] = item.get("task")
    return rep


def short_id(m: dict) -> str:
    return S.short_id(m["session_id"]) + (f"/{m['tag']}" if m.get("tag") else "")


# ---------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["full", "windows"], default="full")
    ap.add_argument("--include", default="hand,robot")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--state-phase", action="store_true",
                    help="append the phase one-hot (6) to observation.state — phase conditioning for policies that do not "
                         "read the task text (ACT); at inference the rollout's PhaseTracker supplies it")
    ap.add_argument("--done-output", action="store_true",
                    help="absolute actions only: a 16th action value drill.done — 1 when the frame the action leads to is after "
                         "the session's last cut (phase ≥ 5). For models that end the drilling themselves (user 10-07: the policy "
                         "ends the drilling; 10-09: also in the GR00T data). The frames after the last cut then carry the drilling "
                         "instruction as their task, so the text does not give the end away")
    ap.add_argument("--wording", choices=["frame", "episode"], default="frame",
                    help="lang1: the wording of an instruction drawn per frame (default) or once per episode")
    ap.add_argument("--instructions", choices=INS.SETS, default="phase1",
                    help="the texts of a frame's task: phase1 = the phase sentences every model so far was trained with; lang1 = "
                         "plain English naming the level and the extent of the laminectomy, several wordings; lang2 = lang1 "
                         "naming only a clear extent, a neutral sentence for the recordings near the midline (instructions.py)")
    ap.add_argument("--cond-dropout", type=float, default=0.0,
                    help="lang2: the share of the drilling frames whose sentence names no extent (drawn per frame, fixed by the "
                         "frame's id) — the goal-free answer guidance needs; 0.15 suggested")
    ap.add_argument("--decision-copies", type=int, default=0,
                    help="lang1 / lang2: K more episodes per clearly labelled recording, each a copy of the frames around where "
                         "the goals part (contralateral: first drilling frame past the midline − 2 mm; ipsilateral: last "
                         "drilling frame) — 2 %% of the frames before")
    ap.add_argument("--decision-frames", type=int, default=40,
                    help="--decision-copies: frames before and after that point (default 40 = GR00T's chunk)")
    ap.add_argument("--progress-output", action="store_true",
                    help="as --done-output, but the 16th action value is drill.progress: the bone removed up to the frame the "
                         "action leads to / the session's final removed volume (cut.progress). A target on every frame (user "
                         "10-09: the end output becomes a progress value)")
    ap.add_argument("--action-mode", choices=["absolute", "delta"], default="absolute",
                    help="delta: per-tool position change (mm) + rotation vector (tool frame) + trigger, 13 values (actions.py)")
    ap.add_argument("--trim", choices=["none", "robot"], default="none",
                    help="robot: hand episodes keep only the part the arms can play back — first cut − pre s (both tools "
                         "inside the skin) … both tools out (trim.py); the cut ends are scripted on the robot")
    ap.add_argument("--trim-pre-s", type=float, default=TR.PRE_S)
    ap.add_argument("--trim-post-s", type=float, default=TR.POST_S)
    ap.add_argument("--trim-spans", type=Path, default=None,
                    help="JSON {short id: [t_a, t_b]} of gate-verified spans (inference/trim_gate.py) that narrow the rule")
    ap.add_argument("--views", default="endo")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--images", choices=["require", "none"], default="require",
                    help="require: episodes whose render cache is incomplete are skipped (listed); none: no images")
    ap.add_argument("--image-dtype", choices=["video", "image"], default="video")
    ap.add_argument("--robot-time", choices=["hand", "robot"], default="hand")
    ap.add_argument("--sessions", type=Path, default=None, help="default ~/il-robot/viewer-data/sessions")
    ap.add_argument("--cache", type=Path, default=None, help="default ~/il-robot/datasets/render_cache")
    ap.add_argument("--repo-id", default="local/spine_rev4")
    ap.add_argument("--only", default="", help="comma list of session short ids / run tags")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep-lerobot-quantiles", action="store_true",
                    help="leave meta/stats.json's q01 … q99 as LeRobot aggregates them (mean of per-episode quantiles) instead of "
                         "rewriting them as quantiles over all frames (global_quantiles.py)")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--layout", choices=["spine", "openh"], default="spine",
                    help="openh: the Open-H-Embodiment v2 layout (openh.py) — absolute actions, no --state-phase; "
                         f"Open-H suggests --fps ≥ {OH.MIN_FPS} and --size ≥ {OH.MIN_SIZE} (a render cache of that fps and size)")
    ap.add_argument("--exclude", type=Path, default=None,
                    help="JSON of stamp ranges (tExp, epoch ms) to leave out — the episode is split around them: the viewer's "
                         "qc_trim.json (sessions.<session id>.exclude_tExp_ms) or {session id | short id: [[a_ms, b_ms], …]}")
    ap.add_argument("--val", default="", help="openh: comma list of short ids / run tags of the validation split")
    ap.add_argument("--test", default="", help="openh: … of the test split (neither given: the last tenth each, at least one)")
    a = ap.parse_args(argv)
    if a.layout == "openh" and (a.action_mode != "absolute" or a.state_phase):
        ap.error("--layout openh holds absolute poses only: no --action-mode delta, no --state-phase")
    if (a.done_output or a.progress_output) and (a.action_mode != "absolute" or a.layout != "spine" or a.state_phase):
        ap.error("--done-output / --progress-output go with --action-mode absolute and --layout spine, without --state-phase")
    if a.done_output and a.progress_output:
        ap.error("--done-output and --progress-output are two forms of one 16th action value: give one")
    if a.cond_dropout and (a.instructions not in INS.NEUTRAL_SETS or not 0.0 < a.cond_dropout < 1.0):
        ap.error("--cond-dropout P (0 < P < 1) needs --instructions lang2 / lang3 — only they have a sentence that names no extent")
    if a.decision_copies and (a.instructions not in ("lang1", "lang2", "lang3") or a.layout != "spine" or a.decision_copies < 0 or a.decision_frames < 1):
        ap.error("--decision-copies K needs --instructions lang1 or lang2 (the goal says where the goals part) and --layout spine")
    ACTION_MODE[0] = a.action_mode
    STATE_PHASE[0] = a.state_phase
    DONE_OUTPUT[0] = a.done_output
    PROGRESS_OUTPUT[0] = a.progress_output
    INSTRUCTIONS[0] = a.instructions; WORDING[0] = a.wording
    COND_DROPOUT[0] = float(a.cond_dropout); DECISION[0] = {"copies": int(a.decision_copies), "frames": int(a.decision_frames)}
    LAYOUT[0] = a.layout
    TRIM[0] = None
    EXCLUDE[0] = {}
    if a.exclude:
        x = json.loads(a.exclude.expanduser().read_text())
        EXCLUDE[0] = {k: v["exclude_tExp_ms"] for k, v in x["sessions"].items() if v.get("exclude_tExp_ms")} if "sessions" in x else x
    if a.trim == "robot":
        TRIM[0] = {"pre_s": a.trim_pre_s, "post_s": a.trim_post_s,
                   "spans": json.loads(a.trim_spans.expanduser().read_text()) if a.trim_spans else {}}
    out = a.out.expanduser()
    a.cache = a.cache or RC.CACHE_ROOT

    def log(msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    if not a.validate_only:
        if out.exists():
            if not a.overwrite:
                ap.error(f"{out} exists — --overwrite to rebuild")
            shutil.rmtree(out)
        include = tuple(x for x in a.include.split(",") if x)
        used, skipped_s = RC.usable_sessions(include, a.sessions, set(a.only.split(",")) if a.only else None)
        used.sort(key=lambda s: (s.source != "hand", s.session_id))
        skipped = [{"session": s.session_id, "source": s.source, "why": s.skip or f"--include 에 {s.source} 없음"}
                   for s in skipped_s]
        eps, sk = plan_episodes(used, a.mode, a.fps, a.robot_time, log)
        skipped += sk
        if a.layout == "openh":   # info.json splits are index ranges: write train, then val, then test
            eps = OH.assign_splits(eps, {x for x in a.val.split(",") if x}, {x for x in a.test.split(",") if x}, log)
        log(f"[export] {len(eps)} episodes planned ({a.mode}), {len(skipped)} sessions skipped")
        views = tuple(v for v in a.views.split(",") if v)
        written, sk = write_dataset(eps, out, a.fps, views, a.size, a.images, a.image_dtype, a.cache, a.repo_id, log)
        skipped += sk
        args = {k: str(v) for k, v in vars(a).items()}
        write_sidecar(out, args, written, skipped)
        log(f"[export] wrote {len(written)} episodes → {out}; skipped {len(skipped)}")
        if a.layout != "openh" and not a.keep_lerobot_quantiles:
            # LeRobot's dataset quantiles are the mean of per-episode quantiles; GR00T scales and clips with them
            # (10-10: 50 % of the endoscope's depth values on the clip limit) — global_quantiles.py
            import global_quantiles as GQ
            q = GQ.fix(out, log=lambda *_: None)
            log(f"[export] meta/stats.json quantiles rewritten over all frames: {', '.join(q['changed']) or 'nothing to change'}")
        for x in skipped:
            log(f"  skip {x['session']}: {x['why']}")
    side = out / "meta/spine_il/export.json"
    if a.validate_only and side.exists():   # the dataset's own layout
        LAYOUT[0] = json.loads(side.read_text()).get("layout", "spine")
    rep = (OH.validate if LAYOUT[0] == "openh" else validate)(out, a.repo_id, log)
    print(json.dumps({k: v for k, v in rep.items() if k != "per_episode"}, ensure_ascii=False, indent=1))
    for r in rep["per_episode"]:
        print(json.dumps(r, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
