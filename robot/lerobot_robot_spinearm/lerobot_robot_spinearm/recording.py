"""FE session recorder v2 (rec-2.x JSONL) → the hand motion `endo_follow.py --mode replay` plays. Pure: numpy only.

File contract (FE `sessionRecordSchemaV2*.ts`, `trackerRecorderSupport.ts`, `trackerRecorderT0.ts`):
  - one session per file, `k` = hdr | f | ev | cut | ftr. The row order authority is `r` (hdr 0, +1 per written
    row), not the line order: a missing `r` = a lost chunk, the same `r` twice = a re-sent chunk (keep the first).
  - f row = one tracker message of one tool (`endo` | `drill`): pos / quat (wxyz, tool → camera) / tip = the
    tracker's filtered output; raw{pos, quat, tip} = the optical fix before the filters, still, jump hold and gyro
    fusion (T0 trackers only). Camera = D435 optical frame, mm. lost=1 = pose_lost (no pose); hold=1 = a jump-hold
    re-send (not a new measurement). vis = constellation balls seen.
  - time: tExp = the camera frame stamp (epoch ms), the dataset time axis ('v2-t0' files); tsd = its source.
    'v2-core' pilot files have no tExp; tSrc (tracker send time, ~+24.5 ms and ~7 ms jitter after tExp) is the
    fallback there.
  - hdr.registration.T0 = T_scene←cam {R row-major 9, t mm}; dynamic=true puts the applied `T` on every f row.
    hdr.toolCalib.<tool>.sha256_12 = the first 12 hex of the sha256 of the tool model file the relay loaded.

`load_recording()` returns the dict `endo_follow.load_replay()` returns — t (s), pos (m, camera), R (tool →
camera), vis, H0_tip (m), H0_R — plus where it came from; `sample_pose()` samples it between recorded frames.
endo_follow glue: `add_replay_args()` (its recorder flags), `replay_input()` (read before the arm is touched),
`provenance()` (the run's `replay_src` log row).
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from pathlib import Path

import numpy as np

TOOLS = ("endo", "drill")
MIN_SCHEMA = 2
MAX_GAP_S = 0.25  # s of recording: recorded frames further apart are held, not bridged (see sample_pose)
ROW_KINDS = ("hdr", "f", "ev", "cut", "ftr")
F_KEYS = frozenset({"k", "r", "tool", "lost", "hold", "pos", "quat", "raw", "tExp", "tSrc", "tsd", "vis", "T"})


class RecordingError(ValueError):
    """The file cannot be replayed (not a recorder v2 file, no usable samples, or a strict check failed)."""


# ---------------------------------------------------------------- rotations
def quat_to_R(q) -> np.ndarray:
    """wxyz (normalised here) → rotation; (4,) → (3, 3), (N, 4) → (N, 3, 3). Same as handeye.quat_to_R."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    R = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                  2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                  2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1)
    return R.reshape(q.shape[:-1] + (3, 3))


def R_to_quat(R) -> np.ndarray:
    """Rotation → unit quaternion wxyz with w ≥ 0 (Shepperd: the largest component is taken from the diagonal)."""
    R = np.asarray(R, float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * math.sqrt(tr + 1.0)
        q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1e-300))
        q = [(R[2, 1] - R[1, 2]) / s, s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1e-300))
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1e-300))
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4]
    q = np.asarray(q) / np.linalg.norm(q)
    return -q if q[0] < 0 else q


def slerp(q0, q1, a: float) -> np.ndarray:
    """Unit-quaternion slerp along the shorter arc (q and −q are the same rotation)."""
    q0 = np.asarray(q0, float) / np.linalg.norm(q0)
    q1 = np.asarray(q1, float) / np.linalg.norm(q1)
    d = float(q0 @ q1)
    if d < 0.0:
        q1, d = -q1, -d
    if d > 1.0 - 1e-9:  # < 0.005° apart: 1/sin θ loses precision, the normalised chord is the arc to < 1e-12 rad
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1.0 - a) * th) * q0 + math.sin(a * th) * q1) / math.sin(th)


def sample_pose(rp: dict, t: float, interp: str = "linear",
                max_gap_s: float | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """The recorded pose at replay time t (s, on rp's own axis) → (R tool → camera, pos m, vis).

      zoh     the last sample at or before t — the replay before 2026-09-27 (a step at every recorded frame)
      linear  position linear, rotation slerp between the two neighbours; vis = the lower of the two
    max_gap_s (s on rp's axis: divide by --replay-speed once rp["t"] is stretched): neighbours further apart are not
    bridged — the earlier one is held until the later one comes (an occlusion is not turned into a straight line
    the hand never drew). None / 0 = no limit. Outside the recording the end sample is held: no extrapolation."""
    if interp not in ("zoh", "linear"):
        raise ValueError(f"interp {interp!r}: zoh | linear")
    ts = rp["t"]
    i = max(0, int(np.searchsorted(ts, t, side="right")) - 1)
    if interp == "zoh" or i >= len(ts) - 1 or t <= ts[i] or (max_gap_s and ts[i + 1] - ts[i] > max_gap_s):
        return rp["R"][i], rp["pos"][i], int(rp["vis"][i])
    a = (t - ts[i]) / (ts[i + 1] - ts[i])
    p = rp["pos"][i] + a * (rp["pos"][i + 1] - rp["pos"][i])
    R = quat_to_R(slerp(R_to_quat(rp["R"][i]), R_to_quat(rp["R"][i + 1]), a))
    return R, p, min(int(rp["vis"][i]), int(rp["vis"][i + 1]))


# ---------------------------------------------------------------- file, header, registration
def _obj(line: str) -> dict | None:
    try:
        d = json.loads(line)
    except ValueError:
        return None
    return d if isinstance(d, dict) else None


def _int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _schema(d: dict) -> int:
    v = d.get("schemaVersion")
    return int(v) if _num(v) else 0


def read_header(path) -> dict | None:
    """The recorder header of a file, or None if it is not a recorder v2 file. The first valid line (a JSON object;
    blank or torn lines skipped) decides: a header (k == 'hdr', schemaVersion ≥ 2) → that; a recorder row (an int
    `r`, k = f | ev | cut | ftr: the first chunk arrived late) → the header further down; anything else → None."""
    p = Path(path)
    if not p.is_file():
        return None
    with open(p, encoding="utf-8", errors="replace") as fh:
        d = next((o for o in (_obj(line) for line in fh if line.strip()) if o is not None), None)
        if d is not None and d.get("k") != "hdr":
            if not (_int(d.get("r")) and d.get("k") in ROW_KINDS):
                return None
            d = next((o for o in (_obj(line) for line in fh if '"hdr"' in line) if o and o.get("k") == "hdr"), None)
    return d if d is not None and _schema(d) >= MIN_SCHEMA else None


def is_recorder_file(path) -> bool:
    return read_header(path) is not None


def read_rows(path, f_keys=None) -> tuple[dict, list[dict], dict]:
    """All rows ordered by `r`, a re-sent `r` dropped (the first in the file is kept) → (hdr, rows, counts).
    f_keys: keep only these keys of f rows and only k / r of ev and cut rows (imu, drawn, proxy and cut stamps are
    most of a long session's memory)."""
    seen: set[int] = set()
    rows: list[dict] = []
    n = {"bad_lines": 0, "no_r": 0, "dup_r": 0}
    first_k = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = _obj(line)
            if d is None:
                n["bad_lines"] += 1
                continue
            if first_k is None:
                first_k = str(d.get("k"))
            r = d.get("r")
            if not _int(r):
                n["no_r"] += 1
                continue
            if r in seen:
                n["dup_r"] += 1
                continue
            seen.add(r)
            if f_keys is not None and d.get("k") == "f":
                d = {k: v for k, v in d.items() if k in f_keys}
            elif f_keys is not None and d.get("k") in ("ev", "cut"):
                d = {"k": d["k"], "r": r}
            rows.append(d)
    rows.sort(key=lambda d: d["r"])
    hdr = next((d for d in rows if d.get("k") == "hdr"), None)
    if hdr is None or _schema(hdr) < MIN_SCHEMA:
        raise RecordingError(f"{path}: no recorder header (k=hdr, schemaVersion ≥ {MIN_SCHEMA})")
    n["missing_r"] = rows[-1]["r"] - rows[0]["r"] + 1 - len(rows)
    n["hdr_late"] = int(first_k != "hdr")
    return hdr, rows, n


def parse_rigid(obj) -> tuple[np.ndarray, np.ndarray]:
    """T_scene←cam → (R 3×3, t mm) from {R: row-major 9 | 3×3, t: mm}, {T0: …}, {registration: …} or a 4×4 (mm)."""
    if isinstance(obj, dict):
        if "R" in obj and "t" in obj:
            R, t = np.asarray(obj["R"], float).reshape(3, 3), np.asarray(obj["t"], float).reshape(3)
        elif "T0" in obj:
            return parse_rigid(obj["T0"])
        elif isinstance(obj.get("registration"), dict):
            return parse_rigid(obj["registration"])
        else:
            raise RecordingError("registration: expected {R: row-major 9, t: mm} (or {T0: …}, a 4×4)")
    else:
        M = np.asarray(obj, float)
        if M.shape != (4, 4):
            raise RecordingError(f"registration: a 4×4 or {{R, t}}, got shape {M.shape}")
        R, t = M[:3, :3], M[:3, 3]
    if not (np.isfinite(R).all() and np.isfinite(t).all()) or np.abs(R.T @ R - np.eye(3)).max() > 1e-3 \
            or abs(np.linalg.det(R) - 1.0) > 1e-3:
        raise RecordingError("registration: R is not a rotation (R must be row-major 9 or 3×3)")
    return R, t


def _rigid_or_none(obj):
    try:
        return parse_rigid(obj) if obj else None
    except (RecordingError, ValueError, TypeError):
        return None


def load_registration(arg) -> tuple[np.ndarray, np.ndarray] | None:
    """`--replay-registration`: inline JSON, a JSON file, or a recorder file (its hdr.registration.T0). None → None."""
    if arg is None:
        return None
    s = str(arg).strip()
    if not s:
        raise RecordingError("--replay-registration is empty — give today's T_scene←cam or leave the flag out")
    if s[0] in "{[":
        return parse_rigid(json.loads(s))
    hdr = read_header(s)
    if hdr is not None:
        if not (hdr.get("registration") or {}).get("T0"):
            raise RecordingError(f"{s}: this recording has no registration T0")
        return parse_rigid(hdr["registration"]["T0"])
    return parse_rigid(json.loads(Path(s).read_text()))


def tool_model(path) -> tuple[np.ndarray, str]:
    """Tracker tool model (tool_calib.json) → (tip m, constellation frame; the ball centroid if it has no tip,
    as handeye.tool_calib) and the sha256_12 the relay stamps into hdr.toolCalib for the same file."""
    raw = Path(path).read_bytes()
    d = json.loads(raw)
    tip = d.get("tip_local")
    if tip is None and not d.get("points_local"):
        raise RecordingError(f"tool model {path}: neither tip_local nor points_local")
    tip_m = (np.asarray(tip, float) if tip is not None else np.asarray(d["points_local"], float).mean(0)) / 1000.0
    return tip_m, hashlib.sha256(raw).hexdigest()[:12]


# ---------------------------------------------------------------- the loader
def _vec(v, n: int) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == n and all(_num(x) for x in v)


def _pose(d: dict, raw: bool):
    src = d.get("raw") if raw else d
    if not isinstance(src, dict) or not _vec(src.get("pos"), 3) or not _vec(src.get("quat"), 4):
        return None
    return src["pos"], src["quat"]


def _increasing(v) -> list[int]:
    """Indices of a longest strictly increasing subsequence of v (patience sorting): an isolated time outlier costs
    its own row, not every row after it."""
    tails: list[float] = []
    tail_i: list[int] = []
    prev = [-1] * len(v)
    for i, x in enumerate(v):
        k = bisect.bisect_left(tails, x)
        if k == len(tails):
            tails.append(x)
            tail_i.append(i)
        else:
            tails[k], tail_i[k] = x, i
        prev[i] = tail_i[k - 1] if k else -1
    out, i = [], (tail_i[-1] if tail_i else -1)
    while i >= 0:
        out.append(i)
        i = prev[i]
    return out[::-1]


def load_recording(path, tool: str = "endo", tip_local_m=None, calib_path=None, *, t_start_s: float | None = None,
                   t_end_s: float | None = None, T_now=None, pose: str = "auto", strict: bool = False,
                   max_gap_s: float | None = MAX_GAP_S) -> dict:
    """A recorder v2 file → the replay dict of `endo_follow.load_replay()` (+ provenance).

    tool            'endo' | 'drill' (the f rows' `tool`; untagged = endo, as the FE)
    tip_local_m     the robot-side tip (m, constellation frame); None → from calib_path, else the recording's own
                    hdr.toolCalib tipLocal (then nothing is cross-checked)
    calib_path      the robot-side tool model file: its sha256_12 must equal hdr.toolCalib.<tool>.sha256_12
    t_start_s/t_end_s  window, s after the file's first valid pose row (either tool); t = 0 and H0 = the first valid
                    sample in the window
    T_now           absolute mode: today's T_scene←cam (R, t mm) — every pose is moved into today's camera through
                    the scene, p_now = T_now⁻¹·(T_rec·p_rec) (R alike; T_rec = hdr T0, or the row's own T when the
                    registration is dynamic — a row without one is dropped like a lost row). None = relative mode:
                    the recorded camera axes are used as they are
    pose            'auto' = raw if the file has it (rows without raw are dropped), else the filtered pos/quat;
                    'raw' | 'filtered' to force one
    strict          refuse (RecordingError) instead of warning when the tool model or the tool identity is not
                    confirmed (sha mismatch or missing, tracker cfg has endo2, tracker and relay models differ)
    max_gap_s       holes longer than this (s of recording) are counted and warned about — the replay holds across
                    them (sample_pose max_gap_s) instead of drawing a straight line; None / 0 = not counted
    Samples: f rows of the tool, not lost, not hold, with a pose and a time; t strictly increasing (an out-of-order
    time drops the fewest rows that restore the order)."""
    if tool not in TOOLS:
        raise RecordingError(f"tool {tool!r}: one of {TOOLS}")
    if pose not in ("auto", "raw", "filtered"):
        raise RecordingError(f"pose {pose!r}: auto | raw | filtered")
    if read_header(path) is None:
        raise RecordingError(f"{path}: not a recorder v2 file (no k=hdr row with schemaVersion ≥ 2)")
    hdr, rows, n = read_rows(path, F_KEYS)
    n.update(no_raw=0, no_T=0, vis_unknown=0, gap_max_ms=0.0, gaps_over_max=0, gap_jump_max_mm=0.0)
    warn: list[str] = []
    refuse: list[str] = []

    spec = hdr.get("timingSpec")
    if spec != "v2-t0":
        warn.append(f"timingSpec={spec!r} — 본 녹화(v2-t0) 형식이 아니다(파일럿 v2-core 등)")
    if n["hdr_late"]:
        warn.append("hdr 가 첫 줄이 아니었다(첫 청크가 늦게 도착) — r 순서로 찾았다")
    if n["missing_r"] > 0:
        warn.append(f"r 빈 번호 {n['missing_r']}개 = 청크 유실")
    if n["dup_r"]:
        warn.append(f"같은 r 재전송 {n['dup_r']}행 — 첫 것만 썼다")
    if n["bad_lines"] or n["no_r"]:
        warn.append(f"읽지 못한 줄 {n['bad_lines']}개, r 없는 행 {n['no_r']}개 — 버렸다")

    # tool identity (robot-rig recordings can carry the arm's own markers under the same names — audit H5)
    tcfg = ((hdr.get("tracker") or {}).get("cfg") or {}).get("tools") or {}
    calib_hdr = (hdr.get("toolCalib") or {}).get(tool) or {}
    if "endo2" in tcfg:
        refuse.append("트래커 설정에 endo2 가 있다(로봇 리그 녹화) — 로봇 마커 블록이 endo/drill 이름으로 섞였을 수 있다")
    trk_sha = ((tcfg.get(tool) or {}).get("calib") or {}).get("sha12")
    if trk_sha and calib_hdr.get("sha256_12") and trk_sha != calib_hdr["sha256_12"]:
        refuse.append(f"트래커가 쓴 {tool} 모델 {trk_sha} ≠ 릴레이가 헤더에 넣은 {calib_hdr['sha256_12']}")

    # tool model: the tip that turns the constellation pose into the tool tip
    rec_sha, rec_tip = calib_hdr.get("sha256_12"), calib_hdr.get("tipLocal")
    robot_sha = None
    if calib_path is not None:
        file_tip, robot_sha = tool_model(calib_path)
        tip_src = f"로봇 모델 {Path(calib_path).name}"
        if tip_local_m is None:
            tip_local_m = file_tip
    elif tip_local_m is not None:
        tip_src = "호출자가 준 팁"
    elif _vec(rec_tip, 3):
        tip_local_m, tip_src = np.asarray(rec_tip, float) / 1000.0, "녹화 헤더 toolCalib"
    else:
        raise RecordingError(f"no {tool} tip: pass the tool model (tool_calib.json) — the header has no tipLocal")
    tip_local_m = np.asarray(tip_local_m, float).reshape(3)
    tip_diff = float(np.linalg.norm(tip_local_m * 1000.0 - np.asarray(rec_tip, float))) if _vec(rec_tip, 3) else None
    if calib_path is None:
        refuse.append(f"로봇 쪽 도구 모델 파일 없음 — sha 대조 생략(팁 = {tip_src})")
    elif not rec_sha:
        refuse.append(f"녹화 헤더에 toolCalib.{tool}.sha256_12 가 없다 — 도구 모델 대조 불가")
    elif rec_sha != robot_sha:
        dt = f", 팁 차 {tip_diff:.2f} mm" if tip_diff is not None else ""
        refuse.append(f"도구 모델 불일치: 녹화 toolCalib.{tool}.sha256_12={rec_sha} ≠ {Path(calib_path).name} "
                      f"{robot_sha}{dt}")
    if strict and refuse:
        raise RecordingError("strict: " + " / ".join(refuse))
    warn.extend(refuse)

    reg = hdr.get("registration") or {}
    if T_now is not None and not reg.get("T0"):
        raise RecordingError("absolute mode needs hdr.registration.T0 — this recording has none")
    dyn_abs = T_now is not None and bool(reg.get("dynamic"))

    # samples
    f_rows = [d for d in rows if d.get("k") == "f"]
    mine = [d for d in f_rows if (d.get("tool") or "endo") == tool]
    n["tool_rows"] = len(mine)
    n["lost"] = sum(1 for d in mine if d.get("lost"))
    n["hold"] = sum(1 for d in mine if not d.get("lost") and d.get("hold"))
    good = [d for d in mine if not d.get("lost") and not d.get("hold") and _pose(d, False)]
    has_raw = [_pose(d, True) is not None for d in good]
    use_raw = pose == "raw" or (pose == "auto" and any(has_raw))
    if use_raw and not any(has_raw):
        raise RecordingError(f"{tool}: raw poses asked for, the file has none (a v2-core file?) — use pose='filtered'")
    if use_raw:
        n["no_raw"] = has_raw.count(False)
        good = [d for d, h in zip(good, has_raw) if h]
        if n["no_raw"]:
            warn.append(f"raw 없는 {tool} 행 {n['no_raw']}개를 뺐다(필터값을 섞으면 계단이 생긴다)")
    elif pose == "auto":
        warn.append("raw 가 없는 파일 — 트래커 필터 출력(pos/quat)으로 재생한다")
    row_T: dict[int, tuple] = {}
    if dyn_abs:  # a dynamic session's rows each carry their own T; T0 is only the first one
        row_T = {d["r"]: Tr for d in good if (Tr := _rigid_or_none(d.get("T"))) is not None}
        n["no_T"] = len(good) - len(row_T)
        good = [d for d in good if d["r"] in row_T]
        if n["no_T"]:
            warn.append(f"동적 정합인데 T 가 없는 {tool} 행 {n['no_T']}개 — 절대 모드라 lost 처럼 뺐다(필름 드라이브가 "
                        f"쉬던 구간: T0 로 메우면 필름 이동량만큼 틀린다)")
    tf = "tExp" if any(_num(d.get("tExp")) for d in good) else "tSrc"
    if tf == "tSrc":
        warn.append("tExp 가 없다 — 시간 축 = tSrc(트래커 송신, tExp 보다 ~24.5 ms 늦고 ~7 ms 흔들림)")
    n["no_time"] = sum(1 for d in good if not _num(d.get(tf)))
    good = [d for d in good if _num(d.get(tf))]
    kept = [good[i] for i in _increasing([d[tf] for d in good])]  # r order; the interpolator needs rising time
    n["non_monotonic"] = len(good) - len(kept)
    if n["non_monotonic"]:
        warn.append(f"{tf} 가 앞뒤와 어긋난 {tool} 행 {n['non_monotonic']}개를 뺐다(순서를 되살리는 최소 개수)")
    tsd = sorted({str(d["tsd"]) for d in kept if d.get("tsd") is not None})
    if tf == "tExp" and len(tsd) > 1:
        warn.append(f"tExp 원천(tsd)이 섞였다 {tsd} — 원천이 바뀐 곳은 시간이 튈 수 있다")
    if not kept:
        raise RecordingError(f"{path}: no valid {tool} samples (rows {n['tool_rows']}, lost {n['lost']}, "
                             f"hold {n['hold']}, no T {n['no_T']})")
    by_tool: dict[str, list[float]] = {}  # the file origin = the first valid pose row of either tool (its own
    for d in f_rows:                      # time outliers left out, as for the samples)
        if not d.get("lost") and not d.get("hold") and _pose(d, False) and _num(d.get(tf)):
            by_tool.setdefault(str(d.get("tool") or "endo"), []).append(d[tf])
    origin = min(v[_increasing(v)[0]] for v in by_tool.values())
    ts = (np.array([d[tf] for d in kept]) - origin) / 1000.0
    win = (ts >= (-math.inf if t_start_s is None else t_start_s)) & (ts <= (math.inf if t_end_s is None else t_end_s))
    if not win.any():
        raise RecordingError(f"no valid {tool} samples in [{t_start_s}, {t_end_s}] s (recording 0–{ts[-1]:.1f} s)")
    n["out_of_window"] = int((~win).sum())
    sel = [d for d, w in zip(kept, win) if w]
    P = np.array([_pose(d, use_raw)[0] for d in sel], float)  # mm, recorded camera
    R = quat_to_R(np.array([_pose(d, use_raw)[1] for d in sel], float))
    vis = [int(d["vis"]) if _num(d.get("vis")) else 4 for d in sel]
    n["vis_unknown"] = sum(1 for d in sel if not _num(d.get("vis")))
    if n["vis_unknown"]:
        warn.append(f"vis 가 없거나 숫자가 아닌 {tool} 행 {n['vis_unknown']}개 — 4볼로 봤다")

    if T_now is not None:
        Rn, tn = parse_rigid({"R": T_now[0], "t": T_now[1]}) if not isinstance(T_now, dict) else parse_rigid(T_now)
        T0 = parse_rigid(reg["T0"])
        for k, d in enumerate(sel):
            Rr, tr = row_T[d["r"]] if dyn_abs else T0
            P[k] = Rn.T @ (Rr @ P[k] + tr - tn)
            R[k] = Rn.T @ Rr @ R[k]
        coord = "absolute"
    else:
        coord = "relative"
        warn.append("상대 모드: 녹화 당시 카메라 배치를 무시한다 — 녹화 카메라 축의 변위를 현 리그 R_bc 로 돌리므로, 그 뒤 "
                    "카메라가 θ 돌았으면 경로도 θ 돈다(--replay-registration 으로 scene 경유 보정)")

    t = ts[win] - ts[win][0]
    if len(t) > 1:  # holes (occlusions, dropped rows): replayed as hold-then-jump, not as a straight line
        gap = np.diff(t)
        tip_mm = R @ tip_local_m * 1000.0 + P
        n["gap_max_ms"] = round(float(gap.max()) * 1000.0, 1)
        if max_gap_s:
            big = np.flatnonzero(gap > max_gap_s)
            n["gaps_over_max"] = int(len(big))
            if len(big):
                jump = np.linalg.norm(tip_mm[big + 1] - tip_mm[big], axis=1)
                n["gap_jump_max_mm"] = round(float(jump.max()), 1)
                warn.append(f"{max_gap_s * 1000:.0f} ms 보다 긴 공백 {len(big)}개(최장 {n['gap_max_ms']:.0f} ms, 그 사이 팁 "
                            f"점프 최대 {n['gap_jump_max_mm']:.1f} mm) — 보간하지 않고 앞 샘플을 유지하다 다음 샘플로 넘어간다"
                            f"(사람이 하지 않은 직선 이동을 만들지 않게)")
    pos = [p / 1000.0 for p in P]
    t0_ms = float(sel[0][tf])
    return {
        "t": t, "pos": pos, "R": list(R), "vis": vis,
        "H0_tip": R[0] @ tip_local_m + pos[0], "H0_R": R[0].copy(),
        "source": "recorder-v2", "tool": tool, "pose_source": "raw" if use_raw else "filtered",
        "time_source": tf, "coord": coord, "t0_ms": t0_ms, "t_offset_s": (t0_ms - origin) / 1000.0,
        "max_gap_s": max_gap_s, "tip_local_m": tip_local_m, "tip_source": tip_src,
        "calib": {"recorded_sha12": rec_sha, "robot_sha12": robot_sha,
                  "match": (rec_sha == robot_sha) if (rec_sha and robot_sha) else None, "tip_diff_mm": tip_diff},
        "recording": {"path": str(path), "sessionId": hdr.get("sessionId"),
                      "recorderVersion": hdr.get("recorderVersion"), "timingSpec": spec,
                      "registration": {"method": reg.get("method"), "dynamic": bool(reg.get("dynamic"))}},
        "counts": n, "warnings": warn,
    }


def describe(rp: dict) -> list[str]:
    """Operator lines for a loaded replay: one summary line, then one per warning."""
    if rp.get("source") != "recorder-v2":
        gap = float(np.diff(rp["t"]).max()) * 1000.0 if len(rp["t"]) > 1 else 0.0
        return [f"[replay] follow 런: {len(rp['t'])} 손 포즈(tap raw), {float(rp['t'][-1]):.1f} s · 좌표=상대(녹화 카메라 축)"
                f" · 최장 공백 {gap:.0f} ms"]
    meta, c = rp.get("recording", {}), rp.get("calib", {})
    model = {True: "일치", False: "불일치", None: "대조 안 함"}[c.get("match")]
    coord = "절대(scene 경유 현 카메라 — 방향만, 시작점은 팔의 engage 위치)" if rp.get("coord") == "absolute" \
        else "상대(녹화 카메라 축)"
    head = (f"[replay] 레코더 v2 {meta.get('sessionId')} ({meta.get('recorderVersion')}): {rp['tool']} {len(rp['t'])} 포즈, "
            f"{float(rp['t'][-1]):.1f} s (녹화 {rp['t_offset_s']:.1f} s 부터) · 포즈={rp['pose_source']} · "
            f"시간={rp['time_source']} · 좌표={coord} · 도구모델 {model} · 최장 공백 {rp['counts']['gap_max_ms']:.0f} ms")
    return [head] + [f"[replay] ⚠ {w}" for w in rp.get("warnings", [])]


# ---------------------------------------------------------------- endo_follow glue
def add_replay_args(ap) -> None:
    """endo_follow's flags for replaying an FE recorder file (beside its own --replay-run / -max-s / -speed / …)."""
    ap.add_argument("--replay-tool", choices=list(TOOLS), default="endo",
                    help="--replay-run may also be an FE recorder v2 session file (.jsonl: raw poses on tExp); whose "
                         "motion to replay from it (endo on the left arm, drill on the right: --arm right)")
    ap.add_argument("--replay-interp", choices=["zoh", "linear"], default="linear",
                    help="replay between recorded frames: linear (position; rotation slerp) or zoh (hold, the old way)")
    ap.add_argument("--replay-max-gap-s", type=float, default=MAX_GAP_S,
                    help="linear: recorded frames further apart than this (s of recording — an occlusion) are not "
                         "bridged: the last one is held until the next one comes (0 = always bridge)")
    ap.add_argument("--replay-registration", default=None,
                    help="recorder file, absolute mode: today's T_scene←cam (JSON {R row-major 9, t mm}, a file with "
                         "it, or a recorder file of today's rig) — the recording goes through the scene into today's "
                         "camera. Direction only: the replay still starts where the arm is at engage. Default "
                         "relative: the recording's camera axes are taken as today's")
    ap.add_argument("--replay-from-s", type=float, default=None, help="recorder file: start this far into it (s)")


def replay_input(a, load_dir) -> tuple[dict | None, str | None]:
    """`endo_follow --mode replay`: the hand motion, read before the arm is touched → (rp, None) after printing what
    was read, or (None, message) to stop with. load_dir = endo_follow.load_replay (a follow run dir). A drill replay
    stops here with what the loader read: the right arm's IK and the drill tip model are not wired."""
    run, tool = a.replay_run, a.replay_tool
    if not run:
        return None, "--mode replay 에는 --replay-run(follow 런 디렉터리 또는 FE 레코더 v2 .jsonl)이 필요하다"
    p = Path(run)
    try:
        if p.is_dir():
            if tool != "endo" or a.replay_registration or a.replay_from_s is not None:
                return None, (f"{run}: follow 런 디렉터리 재생은 내시경·상대 모드·처음부터만 — --replay-tool drill / "
                              f"--replay-registration / --replay-from-s 는 레코더 v2 파일에만 쓴다")
            rp = load_dir(run)
        elif not p.exists():
            return None, f"[replay] {run}: 없다"
        elif not is_recorder_file(p):
            return None, (f"{run}: 레코더 v2 파일이 아니다(k=hdr·schemaVersion≥2 행이 없다) — follow 런 디렉터리나 "
                          f"FE 세션 JSONL 을 줄 것")
        elif (tool == "drill") != (getattr(a, "arm", "left") == "right"):  # endo rides the left arm, drill the right
            return None, (f"[replay] {tool} 재생은 {'오른팔(--arm right)' if tool == 'drill' else '왼팔(--arm left)'} 로만 "
                          f"한다 — 지금 --arm {getattr(a, 'arm', 'left')}. 팔에는 아무것도 보내지 않았다.")
        elif tool not in ("endo", "drill"):  # the recording's own tip: nothing on the robot side is wired for it yet
            rp = load_recording(p, tool, t_start_s=a.replay_from_s, T_now=load_registration(a.replay_registration),
                                max_gap_s=a.replay_max_gap_s)
            return None, (f"[replay] {tool} 재생은 아직 팔에 배선되지 않았다: 로더는 읽었다 — {tool} {len(rp['t'])} 포즈"
                          f"({rp['pose_source']}, {rp['time_source']}), {float(rp['t'][-1]):.1f} s. 오른팔 IK·드릴 팁 모델"
                          f" 연결은 로봇 세션 몫이다(endo_follow 는 왼팔 내시경 팁만 안다). 팔에는 아무것도 보내지 않았다.")
        else:
            rp = load_recording(p, tool, calib_path=a.tool_calib, t_start_s=a.replay_from_s,
                                T_now=load_registration(a.replay_registration), max_gap_s=a.replay_max_gap_s)
    except Exception as e:  # noqa: BLE001 — whatever is wrong with the input stops the run before the arm moves
        return None, f"[replay] {run}: {type(e).__name__}: {e}"
    print("\n".join(describe(rp)), flush=True)
    return rp, None


def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    return o


def provenance(rp: dict, a=None) -> dict:
    """What a replay run replayed, JSON-ready — endo_follow logs it as {"k": "replay_src", …}: the loader's describe()
    lines and fields, plus the replay flags (interp, max gap, speed, clean, window, registration)."""
    t = rp["t"]
    out = {"source": rp.get("source", "follow-run"), "n": len(t), "duration_s": round(float(t[-1]), 3),
           "lines": describe(rp)}
    for k in ("tool", "pose_source", "time_source", "coord", "t0_ms", "t_offset_s", "max_gap_s", "tip_source",
              "calib", "counts", "warnings", "recording"):
        if k in rp:
            out[k] = rp[k]
    if a is not None:
        out["flags"] = {k: getattr(a, k, None) for k in ("replay_run", "replay_tool", "replay_interp",
                                                         "replay_max_gap_s", "replay_speed", "replay_clean_hz",
                                                         "replay_from_s", "replay_max_s", "replay_registration")}
    return _plain(out)
