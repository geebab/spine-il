"""Recorder v2 → arm replay loader (recording.py) — synthetic recorder rows built here, no hardware.

SPINE_REC_SAMPLE=<a real rec-2.x .jsonl> additionally loads that file (e.g. the 2026-09-24 0551 session).
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot_robot_spinearm import handeye as he
from lerobot_robot_spinearm import recording as rec

TIP = [-154.9, 11.3, -59.2]  # mm, constellation frame
PTS = [[0, 0, 0], [145.8, 0, 0], [32.7, 31.1, 0], [89.0, -21.0, 1.0]]
T_EXP0 = 1790175900146.187
DT_MS = 1000.0 / 90.0


def q_axis(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    h = np.radians(deg) / 2
    return np.r_[np.cos(h), np.sin(h) * axis]


def q_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


Q_BASE = {"endo": q_axis([0.3, -0.8, 0.5], 110.0), "drill": q_axis([-0.6, 0.2, 0.7], 40.0)}
R_REC = he.quat_to_R(q_axis([0.2, 0.9, -0.1], 97.0))
T_REC = np.array([-458.5, 280.5, 986.9])


def raw_pose(tool, ts):
    """Hand motion (mm, wxyz) at ts s: a straight line + a constant-rate turn about z — interpolation-exact."""
    p = np.array([100 + 50 * ts, -40 + 20 * ts, 600 - 30 * ts]) if tool == "endo" else \
        np.array([-60 - 25 * ts, -20 + 40 * ts, 614 + 10 * ts])
    return p, q_mul(Q_BASE[tool], q_axis([0, 0, 1], 30.0 * ts))


def calib_file(tmp_path, tip=TIP, name="tool_calib.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"points_local": PTS, "tip_local": tip}))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def session(n=30, tools=("endo", "drill"), sha="aaaaaaaaaaaa", raw=True, t_exp=True, **hdr_over):
    hdr = {"k": "hdr", "r": 0, "schemaVersion": 2, "recorderVersion": "rec-2.3.0-t0",
           "timingSpec": "v2-t0" if t_exp else "v2-core", "sessionId": "s_test", "quatOrder": "wxyz",
           "registration": {"method": "fid3", "T0": {"R": R_REC.flatten().tolist(), "t": T_REC.tolist()},
                            "dynamic": False},
           "toolCalib": {t: {"sha256_12": sha, "tipLocal": TIP} for t in ("endo", "drill")},
           "tracker": {"cfg": {"tools": {t: {"calib": {"sha12": sha}} for t in ("endo", "drill")}}}}
    hdr.update(hdr_over)
    rows, r = [hdr], 1
    for k in range(n):
        te = T_EXP0 + k * DT_MS
        for tool in tools:
            p, q = raw_pose(tool, k * DT_MS / 1000.0)
            d = {"k": "f", "r": r, "i": r - 1, "seq": 100 + k, "tool": tool, "tSrc": int(te + 24.5),
                 "tMono": 5000.0 + k * DT_MS, "pos": (p + [0.5, 0, 0]).round(3).tolist(),  # filtered ≠ raw
                 "quat": q.round(6).tolist(), "vis": 4}
            if t_exp:
                d["tExp"] = te
            if raw:
                d["raw"] = {"pos": p.round(3).tolist(), "quat": q.round(6).tolist()}
            rows.append(d)
            r += 1
    rows.append({"k": "ftr", "r": r, "endReason": "completed"})
    return rows


def write(tmp_path, rows, name="s.jsonl", lines=None):
    path = tmp_path / name
    path.write_text("".join(json.dumps(d) + "\n" for d in rows) if lines is None else lines)
    return path


def f_of(rows, tool):
    return [d for d in rows if d["k"] == "f" and d["tool"] == tool]


# ---------------------------------------------------------------- rotations
def test_quat_R_matches_handeye_and_roundtrips():
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = rec.quat_to_R(q)
        assert np.allclose(R, he.quat_to_R(q), atol=1e-12)
        q2 = rec.R_to_quat(R)
        assert q2[0] >= 0 and abs(np.linalg.norm(q2) - 1) < 1e-12
        assert np.allclose(q2, q if q[0] >= 0 else -q, atol=1e-9)
    Q = rng.normal(size=(5, 4))
    assert np.allclose(rec.quat_to_R(Q), [he.quat_to_R(q) for q in Q], atol=1e-12)


def test_slerp_unit_constant_rate_shortest_arc():
    q0, q1 = q_axis([0, 0, 1], 10.0), q_axis([0, 0, 1], 100.0)
    for a in np.linspace(0, 1, 11):
        q = rec.slerp(q0, q1, a)
        assert abs(np.linalg.norm(q) - 1) < 1e-12
        assert he.R_angle_deg(rec.quat_to_R(q).T @ rec.quat_to_R(q0)) == pytest.approx(90.0 * a, abs=1e-6)
        assert np.allclose(rec.quat_to_R(rec.slerp(q0, -q1, a)), rec.quat_to_R(q), atol=1e-12)  # −q = q


# ---------------------------------------------------------------- file order, exclusions, time
def test_rows_ordered_by_r_and_resent_r_keeps_first(tmp_path):
    rows = session(n=12, tools=("endo",))
    f = f_of(rows, "endo")
    first3 = dict(f[3])
    resent = dict(f[3], raw={"pos": [9e3, 9e3, 9e3], "quat": f[3]["raw"]["quat"]})  # same r, arrives later
    gone = f[6]["r"]
    order = [rows[0]] + f[::-1] + [resent, rows[-1]]  # lines reversed: r is the authority, not the line order
    order = [d for d in order if d["r"] != gone]  # a lost chunk
    rp = rec.load_recording(write(tmp_path, order), "endo", tip_local_m=np.array(TIP) / 1000)
    assert rp["counts"]["dup_r"] == 1 and rp["counts"]["missing_r"] == 1
    assert len(rp["t"]) == 11 and np.all(np.diff(rp["t"]) > 0)
    k3 = int(np.argmin(np.abs(rp["t"] - 3 * DT_MS / 1000)))
    assert np.allclose(rp["pos"][k3] * 1000, first3["raw"]["pos"])  # the first copy of r won
    assert any("r 빈 번호" in w for w in rp["warnings"]) and any("재전송" in w for w in rp["warnings"])


def test_lost_and_hold_rows_excluded_vis_kept(tmp_path):
    rows = session(n=10, tools=("endo", "drill"))
    e = f_of(rows, "endo")
    for d in e[:2]:  # the recording starts with the endoscope lost: t = 0 is its first real fix
        for key in ("pos", "quat", "raw", "vis"):
            d.pop(key)
        d["lost"] = 1
    e[5]["hold"] = 1
    e[7]["vis"] = 3
    rp = rec.load_recording(write(tmp_path, rows), "endo", tip_local_m=np.array(TIP) / 1000)
    c = rp["counts"]
    assert (c["tool_rows"], c["lost"], c["hold"], len(rp["t"])) == (10, 2, 1, 7)
    assert rp["t"][0] == 0.0 and rp["t_offset_s"] == pytest.approx(2 * DT_MS / 1000, abs=1e-6)  # drill at 0
    assert rp["vis"] == [4, 4, 4, 4, 3, 4, 4]
    assert np.allclose(rp["t"], (np.array([2, 3, 4, 6, 7, 8, 9]) - 2) * DT_MS / 1000, atol=1e-6)


def test_tool_selection(tmp_path):
    rows = session(n=8)
    path = write(tmp_path, rows)
    for tool in ("endo", "drill"):
        rp = rec.load_recording(path, tool, tip_local_m=np.array(TIP) / 1000)
        want = [d["raw"]["pos"] for d in f_of(rows, tool)]
        assert rp["tool"] == tool and np.allclose(np.array(rp["pos"]) * 1000, want)


def test_raw_preferred_rows_without_raw_dropped(tmp_path):
    rows = session(n=10, tools=("endo",))
    f_of(rows, "endo")[4].pop("raw")
    rp = rec.load_recording(write(tmp_path, rows), "endo", tip_local_m=np.array(TIP) / 1000)
    assert rp["pose_source"] == "raw" and rp["counts"]["no_raw"] == 1 and len(rp["t"]) == 9
    rpf = rec.load_recording(write(tmp_path, rows), "endo", tip_local_m=np.array(TIP) / 1000, pose="filtered")
    assert rpf["pose_source"] == "filtered" and len(rpf["t"]) == 10
    assert np.allclose((rpf["pos"][0] - rp["pos"][0]) * 1000, [0.5, 0, 0])


def test_pilot_file_without_raw_or_tExp_falls_back(tmp_path):
    rows = session(n=10, tools=("endo",), raw=False, t_exp=False)
    rp = rec.load_recording(write(tmp_path, rows), "endo", tip_local_m=np.array(TIP) / 1000)
    assert (rp["pose_source"], rp["time_source"]) == ("filtered", "tSrc")
    ts = np.array([d["tSrc"] for d in f_of(rows, "endo")], float)
    assert np.allclose(rp["t"], (ts - ts[0]) / 1000)
    w = " ".join(rp["warnings"])
    assert "timingSpec" in w and "tExp 가 없다" in w and "raw 가 없는" in w
    with pytest.raises(rec.RecordingError):
        rec.load_recording(write(tmp_path, rows), "endo", tip_local_m=np.array(TIP) / 1000, pose="raw")


def test_H0_is_the_first_valid_sample_of_the_window(tmp_path):
    rows = session(n=40, tools=("endo", "drill"))
    tip = np.array(TIP) / 1000
    path = write(tmp_path, rows)
    rp = rec.load_recording(path, "endo", tip_local_m=tip)
    assert np.allclose(rp["H0_R"], rp["R"][0]) and np.allclose(rp["H0_tip"], rp["R"][0] @ tip + rp["pos"][0])
    d0 = f_of(rows, "endo")[0]["raw"]
    assert np.allclose(rp["H0_tip"] * 1000, he.quat_to_R(d0["quat"]) @ TIP + d0["pos"], atol=1e-6)
    w = rec.load_recording(path, "endo", tip_local_m=tip, t_start_s=0.105, t_end_s=0.305)  # frames 10 … 27
    k = 10
    dk = f_of(rows, "endo")[k]["raw"]
    assert w["t"][0] == 0 and w["t_offset_s"] == pytest.approx(k * DT_MS / 1000, abs=1e-6) and len(w["t"]) == 18
    assert w["counts"]["out_of_window"] == 40 - 18
    assert np.allclose(w["H0_tip"] * 1000, he.quat_to_R(dk["quat"]) @ TIP + dk["pos"], atol=1e-6)
    with pytest.raises(rec.RecordingError):
        rec.load_recording(path, "endo", tip_local_m=tip, t_start_s=99.0)


def test_header_found_by_r_when_the_first_chunk_came_late(tmp_path):
    rows = session(n=10, tools=("endo",))
    lines = "".join(json.dumps(d) + "\n" for d in rows[3:] + rows[:3])  # the hdr (r 0) three rows down
    path = write(tmp_path, rows, name="late.jsonl", lines=lines)
    assert rec.is_recorder_file(path) and rec.read_header(path)["r"] == 0
    rp = rec.load_recording(path, "endo", tip_local_m=np.array(TIP) / 1000)
    assert len(rp["t"]) == 10 and rp["counts"]["hdr_late"] == 1
    assert any("hdr 가 첫 줄이 아니었다" in w for w in rp["warnings"])


def test_one_time_outlier_costs_one_row_and_a_tsd_mix_is_flagged(tmp_path):
    rows = session(n=200, tools=("endo",))
    f = f_of(rows, "endo")
    f[50]["tExp"] += 10000.0  # a +10 s stamp used to drop every row after it (and leave a 10 s hole)
    f[120]["tExp"] -= 10000.0  # a −10 s one would have become the file's time origin
    for d in f:
        d["tsd"] = "g"
    f[150]["tsd"] = "s"
    rp = rec.load_recording(write(tmp_path, rows), "endo", tip_local_m=np.array(TIP) / 1000)
    assert len(rp["t"]) == 198 and rp["counts"]["non_monotonic"] == 2 and rp["t_offset_s"] == 0.0
    assert rp["t"][-1] == pytest.approx(199 * DT_MS / 1000, abs=1e-6) and np.diff(rp["t"]).max() < 0.03
    w = " ".join(rp["warnings"])
    assert "어긋난" in w and "tsd" in w


def test_read_rows_keeps_only_what_the_loader_needs(tmp_path):
    rows = session(n=3)
    for d in f_of(rows, "endo") + f_of(rows, "drill"):
        d["imu"], d["drawn"] = {"g": [[1, 2, 3]] * 50}, {"tip": [0.0, 0.0, 0.0]}
    rows.insert(-1, {"k": "cut", "r": rows[-1]["r"], "c": [[0.0, 0.0, 0.0]] * 100})
    rows[-1]["r"] += 1
    hdr, got, n = rec.read_rows(write(tmp_path, rows), rec.F_KEYS)
    assert all(set(d) <= rec.F_KEYS for d in got if d["k"] == "f") and hdr == rows[0]
    assert [d for d in got if d["k"] == "cut"] == [{"k": "cut", "r": rows[-2]["r"]}] and n["missing_r"] == 0


def test_bad_inputs_give_clear_errors(tmp_path):
    rows = session(n=5, tools=("endo",))
    f_of(rows, "endo")[2]["vis"] = None
    rp = rec.load_recording(write(tmp_path, rows), "endo", tip_local_m=np.array(TIP) / 1000)
    assert rp["vis"] == [4, 4, 4, 4, 4] and rp["counts"]["vis_unknown"] == 1
    with pytest.raises(rec.RecordingError, match="empty"):
        rec.load_registration("  ")
    bad = tmp_path / "bad_calib.json"
    bad.write_text(json.dumps({"residual_mm": 0.3}))
    with pytest.raises(rec.RecordingError, match="neither"):
        rec.tool_model(bad)


# ---------------------------------------------------------------- tool model and identity
def test_tool_model_sha_checked(tmp_path):
    calib, sha = calib_file(tmp_path)
    ok = rec.load_recording(write(tmp_path, session(n=5, sha=sha)), "endo", calib_path=calib, strict=True)
    assert ok["calib"]["match"] is True and ok["calib"]["tip_diff_mm"] == pytest.approx(0.0)
    assert np.allclose(ok["tip_local_m"] * 1000, TIP) and not any("도구 모델" in w for w in ok["warnings"])
    other, sha2 = calib_file(tmp_path, tip=[-150.0, 11.3, -59.2], name="other.json")
    bad = rec.load_recording(write(tmp_path, session(n=5, sha=sha)), "endo", calib_path=other)
    assert bad["calib"]["match"] is False and bad["calib"]["tip_diff_mm"] == pytest.approx(4.9)
    assert any("도구 모델 불일치" in w and sha2 in w for w in bad["warnings"])
    with pytest.raises(rec.RecordingError, match="strict"):
        rec.load_recording(write(tmp_path, session(n=5, sha=sha)), "endo", calib_path=other, strict=True)
    no_hdr = session(n=5, sha=sha, toolCalib=None, tracker=None)
    assert any("toolCalib.endo.sha256_12 가 없다" in w
               for w in rec.load_recording(write(tmp_path, no_hdr), "endo", calib_path=calib)["warnings"])


def test_tool_identity_warnings(tmp_path):
    calib, sha = calib_file(tmp_path)
    rows = session(n=5, sha=sha)
    rows[0]["tracker"]["cfg"]["tools"]["endo2"] = {"calib": {"sha12": sha}}
    rows[0]["tracker"]["cfg"]["tools"]["drill"]["calib"]["sha12"] = "162a00282c56"
    rp = rec.load_recording(write(tmp_path, rows), "drill", calib_path=calib)
    w = " ".join(rp["warnings"])
    assert "endo2" in w and "162a00282c56" in w
    with pytest.raises(rec.RecordingError):
        rec.load_recording(write(tmp_path, rows), "drill", calib_path=calib, strict=True)


# ---------------------------------------------------------------- interpolation
def test_sample_pose_linear_slerp_and_held_ends(tmp_path):
    rows = session(n=20, tools=("endo",))
    rp = rec.load_recording(write(tmp_path, rows), "endo", tip_local_m=np.array(TIP) / 1000)
    rp["vis"][7] = 3
    ts = rp["t"]
    for x in np.linspace(ts[0], ts[-1], 97):
        R, p, vis = rec.sample_pose(rp, x)
        p_true, q_true = raw_pose("endo", x)
        assert np.allclose(p * 1000, p_true, atol=2e-3)  # the motion is linear: exact up to the 3-dp rounding
        assert he.R_angle_deg(R.T @ he.quat_to_R(q_true)) < 1e-3  # constant-rate turn: slerp is exact
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12) and np.linalg.det(R) == pytest.approx(1.0)
        i = max(0, int(np.searchsorted(ts, x, side="right")) - 1)
        Rz, pz, vz = rec.sample_pose(rp, x, "zoh")  # the old replay, unchanged
        assert Rz is rp["R"][i] and pz is rp["pos"][i] and vz == rp["vis"][i]
    mid = (ts[7] + ts[8]) / 2
    assert rec.sample_pose(rp, mid)[2] == 3 and rec.sample_pose(rp, ts[8])[2] == 4
    for mode in ("zoh", "linear"):  # no extrapolation either side
        for x, k in ((ts[0] - 5.0, 0), (ts[-1] + 5.0, -1)):
            R, p, _ = rec.sample_pose(rp, x, mode)
            assert np.allclose(p, rp["pos"][k]) and np.allclose(R, rp["R"][k])
    with pytest.raises(ValueError):
        rec.sample_pose(rp, 0.1, "cubic")


def test_long_hole_is_held_not_bridged(tmp_path):
    rows = session(n=100, tools=("endo",))
    for d in f_of(rows, "endo")[20:80]:  # a 678 ms occlusion
        for key in ("pos", "quat", "raw", "vis"):
            d.pop(key)
        d["lost"] = 1
    rp = rec.load_recording(write(tmp_path, rows), "endo", tip_local_m=np.array(TIP) / 1000)
    c, ts, tip = rp["counts"], rp["t"], np.array(TIP) / 1000
    k = int(np.argmax(np.diff(ts)))  # the hole: sample k = frame 19, sample k + 1 = frame 80
    jump = np.linalg.norm((rp["R"][k + 1] @ tip + rp["pos"][k + 1]) - (rp["R"][k] @ tip + rp["pos"][k])) * 1000
    assert k == 19 and c["gaps_over_max"] == 1 and c["gap_max_ms"] == pytest.approx(61 * DT_MS, abs=0.1)
    assert c["gap_jump_max_mm"] == pytest.approx(jump, abs=0.1) and any("공백" in w for w in rp["warnings"])
    mid = (ts[k] + ts[k + 1]) / 2
    for sp in (1.0, 0.2):  # on the replay's stretched axis the limit is divided by the speed as well
        rps = {**rp, "t": ts / sp}
        R, p, _ = rec.sample_pose(rps, mid / sp, "linear", rec.MAX_GAP_S / sp)
        assert np.linalg.norm(p - rp["pos"][k]) == 0.0 and R is rp["R"][k]  # held: no invented motion
        assert rec.sample_pose(rps, ts[k + 1] / sp, "linear", rec.MAX_GAP_S / sp)[1] is rp["pos"][k + 1]
        _, pb, _ = rec.sample_pose(rps, mid / sp, "linear")  # no limit: a straight line the hand never drew
        assert np.linalg.norm(pb - rp["pos"][k]) * 1000 > 20.0
    x = (ts[5] + ts[6]) / 2  # an ordinary 11 ms step is still interpolated
    assert np.allclose(rec.sample_pose(rp, x, "linear", rec.MAX_GAP_S)[1], (rp["pos"][5] + rp["pos"][6]) / 2)


# ---------------------------------------------------------------- coordinates
def test_absolute_mode_goes_through_the_scene(tmp_path):
    rows = session(n=10, tools=("endo",))
    path = write(tmp_path, rows)
    tip = np.array(TIP) / 1000
    rel = rec.load_recording(path, "endo", tip_local_m=tip)
    assert rel["coord"] == "relative" and any("상대 모드" in w for w in rel["warnings"])
    same = rec.load_recording(path, "endo", tip_local_m=tip, T_now=(R_REC, T_REC))
    assert same["coord"] == "absolute" and not any("상대 모드" in w for w in same["warnings"])
    assert np.allclose(same["pos"], rel["pos"], atol=1e-12) and np.allclose(same["R"], rel["R"], atol=1e-12)
    Rn, tn = he.quat_to_R(q_axis([1, -0.3, 0.4], 11.85)) @ R_REC, T_REC + [60.0, -50.0, 55.0]
    ab = rec.load_recording(path, "endo", tip_local_m=tip, T_now=(Rn, tn))
    for p_now, R_now, p_rec, R_rec in zip(ab["pos"], ab["R"], rel["pos"], rel["R"]):
        assert np.allclose(Rn @ (p_now * 1000) + tn, R_REC @ (p_rec * 1000) + T_REC, atol=1e-9)  # same scene point
        assert np.allclose(Rn @ R_now, R_REC @ R_rec, atol=1e-12)
        back = R_REC.T @ (Rn @ (p_now * 1000) + tn - T_REC)  # and back into the recording's camera
        assert np.allclose(back, p_rec * 1000, atol=1e-9)
    assert np.allclose(ab["H0_tip"], ab["R"][0] @ tip + ab["pos"][0])


def test_absolute_mode_uses_the_row_T_when_dynamic(tmp_path):
    rows = session(n=6, tools=("endo",))
    rows[0]["registration"]["dynamic"] = True
    T_row = {"R": he.quat_to_R(q_axis([0, 1, 0], 5.0)).flatten().tolist(), "t": [1.0, 2.0, 3.0]}
    for d in f_of(rows, "endo"):
        d["T"] = T_row
    ab = rec.load_recording(write(tmp_path, rows), "endo", tip_local_m=np.array(TIP) / 1000,
                            T_now=(np.eye(3), np.zeros(3)))
    d0 = f_of(rows, "endo")[0]["raw"]
    assert np.allclose(ab["pos"][0] * 1000, np.reshape(T_row["R"], (3, 3)) @ d0["pos"] + T_row["t"], atol=1e-9)


def test_dynamic_rows_without_T_are_dropped_in_absolute_mode(tmp_path):
    rows = session(n=6, tools=("endo",))
    rows[0]["registration"]["dynamic"] = True
    f = f_of(rows, "endo")
    for d in f[::2]:
        d["T"] = {"R": np.eye(3).flatten().tolist(), "t": [100.0, 0.0, 0.0]}
    f[4]["T"] = {"R": [1, 0, 0], "t": [0, 0, 0]}  # a broken T counts as none
    path = write(tmp_path, rows)
    ab = rec.load_recording(path, "endo", tip_local_m=np.array(TIP) / 1000, T_now=(np.eye(3), np.zeros(3)))
    assert ab["counts"]["no_T"] == 4 and len(ab["t"]) == 2  # frames 0 and 2
    raw = np.array([f[k]["raw"]["pos"] for k in (0, 2)])
    assert np.allclose(np.array(ab["pos"]) * 1000 - raw, [[100.0, 0, 0]] * 2, atol=1e-9)  # no T0 fill-in zigzag
    assert any("T 가 없는" in w for w in ab["warnings"])
    rel = rec.load_recording(path, "endo", tip_local_m=np.array(TIP) / 1000)  # relative mode needs no T
    assert rel["counts"]["no_T"] == 0 and len(rel["t"]) == 6


def test_registration_forms(tmp_path):
    R, t = R_REC, T_REC
    js = json.dumps({"R": R.flatten().tolist(), "t": t.tolist()})
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, t
    (tmp_path / "reg.json").write_text(js)
    rec_file = write(tmp_path, session(n=2), name="today.jsonl")
    for arg in (js, str(tmp_path / "reg.json"), str(rec_file), json.dumps({"T0": M.tolist()}),
                json.dumps({"registration": {"T0": {"R": R.tolist(), "t": t.tolist()}}})):
        R2, t2 = rec.load_registration(arg)
        assert np.allclose(R2, R) and np.allclose(t2, t)
    assert rec.load_registration(None) is None
    with pytest.raises(rec.RecordingError):
        rec.load_registration(json.dumps({"R": R.T.flatten().tolist()[:8] + [2.0], "t": [0, 0, 0]}))


# ---------------------------------------------------------------- endo_follow wiring
def test_recorder_file_detection(tmp_path):
    rows = session(n=6)
    path = write(tmp_path, rows)
    assert rec.is_recorder_file(path)
    assert rec.is_recorder_file(write(tmp_path, rows, name="torn.jsonl",
                                      lines="\n{\"k\": \"hd\n" + "".join(json.dumps(d) + "\n" for d in rows)))
    assert not rec.is_recorder_file(write(tmp_path, [dict(rows[0], schemaVersion=1)] + rows[1:], name="v1.jsonl"))
    run = tmp_path / "run"
    run.mkdir()
    (run / "events.jsonl").write_text(json.dumps({"k": "clock", "wall": 1.0}) + "\n")
    assert not rec.is_recorder_file(run / "events.jsonl") and not rec.is_recorder_file(run)


def _args(run, **kw):
    a = {"replay_run": None if run is None else str(run), "replay_tool": "endo", "replay_interp": "linear",
         "replay_max_gap_s": rec.MAX_GAP_S, "replay_registration": None, "replay_from_s": None, "replay_speed": 0.2,
         "replay_clean_hz": None, "replay_max_s": None, "tool_calib": None}
    return SimpleNamespace(**{**a, **kw})


def test_replay_input_reads_and_describes_before_the_arm(tmp_path, capsys):
    path = write(tmp_path, session(n=6))
    calib, _ = calib_file(tmp_path)
    rp, msg = rec.replay_input(_args(path, tool_calib=calib), None)
    out = capsys.readouterr().out
    assert msg is None and len(rp["t"]) == 6 and out.startswith("[replay] 레코더 v2 s_test (rec-2.3.0-t0): endo 6")
    assert "포즈=raw" in out and "도구모델 불일치" in out and "상대 모드" in out  # the header's sha is a dummy
    run = tmp_path / "run"
    run.mkdir()
    (run / "x.json").write_text("{}\n")
    fake = {"t": np.array([0.0, 0.5]), "pos": [np.zeros(3)] * 2, "R": [np.eye(3)] * 2, "vis": [4, 4]}
    rp, msg = rec.replay_input(_args(run), lambda d: fake)
    assert rp is fake and msg is None and "follow 런: 2 손 포즈" in capsys.readouterr().out
    stops = {"필요하다": _args(None), "--replay-registration": _args(run, replay_registration="{}"),
             "레코더 v2 파일이 아니다": _args(run / "x.json"), "없다": _args(tmp_path / "nope.jsonl"),
             "99": _args(path, tool_calib=calib, replay_from_s=99.0),
             "RecordingError": _args(path, tool_calib=calib, replay_registration=" "),
             "IndexError": _args(run)}  # e.g. load_replay on a dir without an engage
    for key, a in stops.items():
        rp, msg = rec.replay_input(a, lambda d: [][0])
        assert rp is None and key in msg, (key, msg)
    # each tool on its own arm: endo left, drill right — the wrong pairing stops before the arm is touched
    rp, msg = rec.replay_input(_args(path, replay_tool="drill"), None)
    assert rp is None and "오른팔(--arm right)" in msg and "아무것도 보내지 않았다" in msg
    rp, msg = rec.replay_input(_args(path, replay_tool="endo", arm="right"), None)
    assert rp is None and "왼팔(--arm left)" in msg
    rp, msg = rec.replay_input(_args(path, replay_tool="drill", arm="right", tool_calib=calib), None)
    assert msg is None and len(rp["t"]) == 6 and rp["tool"] == "drill"


def test_provenance_is_a_json_row(tmp_path):
    rp = rec.load_recording(write(tmp_path, session(n=6)), "endo", tip_local_m=np.array(TIP) / 1000)
    row = json.loads(json.dumps({"k": "replay_src", **rec.provenance(rp, _args("s.jsonl", replay_interp="zoh"))}))
    assert row["flags"]["replay_interp"] == "zoh" and row["flags"]["replay_max_gap_s"] == rec.MAX_GAP_S
    assert row["lines"] == rec.describe(rp) and row["n"] == 6 and row["pose_source"] == "raw"
    assert row["counts"]["gaps_over_max"] == 0 and row["max_gap_s"] == rec.MAX_GAP_S
    follow = {"t": np.array([0.0, 1.5]), "pos": [np.zeros(3)] * 2, "R": [np.eye(3)] * 2, "vis": [4, 4]}
    assert rec.provenance(follow)["source"] == "follow-run"


def _endo_follow(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))  # robot/
    return pytest.importorskip("endo_follow")


def test_output_is_what_endo_follow_replays(tmp_path, monkeypatch):
    ef = _endo_follow(monkeypatch)
    rp = rec.load_recording(write(tmp_path, session(n=60, tools=("endo",))), "endo",
                            tip_local_m=np.array(TIP) / 1000)
    assert isinstance(rp["t"], np.ndarray) and rp["t"].dtype == float
    assert all(p.shape == (3,) for p in rp["pos"]) and all(R.shape == (3, 3) for R in rp["R"])
    assert all(isinstance(v, int) for v in rp["vis"]) and rp["H0_tip"].shape == (3,) and rp["H0_R"].shape == (3, 3)
    cr = ef.clean_replay(rp, 4.0)  # the existing offline clean takes it as it is
    assert len(cr["t"]) == 60 and cr["H0_tip"] is rp["H0_tip"]


def test_endo_follow_replay_flags(monkeypatch):
    ef = _endo_follow(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["endo_follow.py", "--mode", "replay", "--replay-run", "x.jsonl"])
    a = ef.parse()
    assert (a.replay_interp, a.replay_tool, a.replay_registration, a.replay_from_s, a.replay_max_gap_s) == \
        ("linear", "endo", None, None, rec.MAX_GAP_S)
    monkeypatch.setattr(sys, "argv", ["endo_follow.py", "--mode", "replay", "--replay-run", "x.jsonl",
                                      "--replay-interp", "zoh", "--replay-tool", "drill", "--replay-from-s", "11.4",
                                      "--replay-max-gap-s", "0"])
    a = ef.parse()
    assert (a.replay_interp, a.replay_tool, a.replay_from_s, a.replay_max_gap_s) == ("zoh", "drill", 11.4, 0.0)


# ---------------------------------------------------------------- a real recording (optional)
@pytest.mark.skipif(not os.environ.get("SPINE_REC_SAMPLE"), reason="SPINE_REC_SAMPLE (a rec-2.x .jsonl) not set")
def test_real_recording():
    path = os.environ["SPINE_REC_SAMPLE"]
    ftr = [json.loads(line) for line in open(path) if '"k":"ftr"' in line.replace(" ", "")]
    rng = np.random.default_rng(0)
    for tool in ("endo", "drill"):
        rp = rec.load_recording(path, tool)
        c = rp["counts"]
        if ftr and c["missing_r"] == 0:  # every pose row the recorder wrote is a sample or has a reason not to be
            assert len(rp["t"]) + c["hold"] + c["no_raw"] + c["no_time"] + c["non_monotonic"] == \
                ftr[0]["tools"][tool]["nFrames"] - ftr[0]["tools"][tool]["nLost"]
        if "0551" in path:
            assert len(rp["t"]) == {"endo": 6451, "drill": 6450}[tool]
            assert (rp["pose_source"], rp["time_source"]) == ("raw", "tExp")
            assert c["gaps_over_max"] == 0 and c["gap_max_ms"] < 250  # its longest hole is 156 ms
        assert np.all(np.diff(rp["t"]) > 0)
        # replay-time sampling (a 90 Hz loop, random phase) vs the recorded tip interpolated linearly, and a
        # hold-out check: every other sample removed and predicted from the rest
        tip, ts = rp["tip_local_m"], rp["t"]
        tips = np.array([R @ tip + p for R, p in zip(rp["R"], rp["pos"])]) * 1000
        live = np.arange(rng.uniform(0, 1 / 90), ts[-1], 1 / 90)
        ref = np.array([np.interp(live, ts, tips[:, k]) for k in range(3)]).T
        half = {"t": ts[::2], "pos": rp["pos"][::2], "R": rp["R"][::2], "vis": rp["vis"][::2]}
        err = {}
        for mode in ("zoh", "linear"):
            got = np.array([R @ tip + p for R, p, _ in (rec.sample_pose(rp, x, mode) for x in live)]) * 1000
            e = np.linalg.norm(got - ref, axis=1)
            ho = np.array([R @ tip + p for R, p, _ in (rec.sample_pose(half, x, mode) for x in ts[1::2])]) * 1000
            h = np.linalg.norm(ho - tips[1::2], axis=1)
            err[mode] = (np.sqrt((e ** 2).mean()), e.max(), np.sqrt((h ** 2).mean()), h.max())
        print(f"\n{tool}: {len(ts)} samples, {ts[-1]:.1f} s; replay-time error vs the recorded tip "
              f"(RMS / max mm, then hold-out RMS / max): zoh {np.round(err['zoh'], 3)}, "
              f"linear {np.round(err['linear'], 3)}")
        assert err["linear"][0] < 0.1 * err["zoh"][0] and err["linear"][2] < err["zoh"][2]
