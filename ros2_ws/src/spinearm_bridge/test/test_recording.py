"""recording.py (viewer recorder v2 JSONL → replay events) and replay_timeline.py (pacing, cut markers) — no ROS.

SPINE_REC_SAMPLE=<path to a real recording> adds a check on it (counts, time range, registration direction)."""

import ast
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from spinearm_bridge import recording as rc
from spinearm_bridge import replay_timeline as tl
from spinearm_bridge.convert import quat_wxyz_to_xyzw, split_stamp_ms


def _R_wxyz(q):   # Hamilton quaternion [w, x, y, z] → rotation (tool → camera), the tracker's convention
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _rot(ax_deg, az_deg):
    a, b = math.radians(ax_deg), math.radians(az_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Rz = np.array([[math.cos(b), -math.sin(b), 0], [math.sin(b), math.cos(b), 0], [0, 0, 1]])
    return Rz @ Rx


R0 = _rot(20.0, 30.0)
T0_MM = {"R": R0.ravel().tolist(), "t": [-458.46, 280.46, 986.94]}   # T_scene←cam, R row-major as the recorder writes


def _hdr(dynamic=False, **kw):
    h = {"k": "hdr", "r": 0, "schemaVersion": 2, "recorderVersion": "rec-2.3.0-t0", "timingSpec": "v2-t0",
         "quatOrder": "wxyz", "sessionId": "s_test",
         "registration": {"method": "film" if dynamic else "fid3", "T0": T0_MM, "dynamic": dynamic},
         "drill": {"radiusMm": 4},
         "replay": {"cutRows": True, "drawn": True, "version": 1, "carvedBeforeStart": {"bone": False, "soft": False},
                    "drillVersionStart": 0}}
    h.update(kw)
    return h


def _f(r, tool, t_exp, pos=(100.0, -20.0, 500.0), quat=(1.0, 0.0, 0.0, 0.0), **kw):
    tip = [pos[0] + 10.0, pos[1], pos[2] + 50.0]
    row = {"k": "f", "r": r, "i": r - 1, "seq": 1000 + r, "tool": tool, "tSrc": int(t_exp + 25), "tMono": float(r),
           "tExp": t_exp, "tEmit": t_exp + 25.4, "pos": list(pos), "quat": list(quat), "tip": tip, "vis": 4,
           "rmse": 0.4, "still": 0, "hold": 0, "raw": {"pos": [pos[0] + 0.1, pos[1], pos[2]], "quat": list(quat),
                                                        "tip": tip},
           "scene": {"tip": (R0 @ np.array(tip) + T0_MM["t"]).tolist()}}
    row.update(kw)
    return row


def _cut(r, t_exp=None, target="bone", c=((1.0, 2.0, 3.0),), r_mm=4, mod=1, **kw):
    row = {"k": "cut", "r": r, "i": 0, "tMono": float(r), "tool": "drill", "target": target, "dv": 0, "mod": mod}
    if target == "bone":
        row.update(label="vertebrae_bone_t000", fn="smooth", smooth=1, fresh=1)
    for key, v in (("tExp", t_exp), ("c", None if c is None else [list(p) for p in c]), ("rMm", r_mm)):
        if v is not None:   # a soft `reset` row has no c / rMm
            row[key] = v
    row.update(kw)
    return row


def _lines(rows):
    return [json.dumps(r) for r in rows]


def test_rows_ordered_by_r_first_copy_wins_gaps_counted():
    rows = [_hdr(), _f(1, "endo", 1000.0), _f(2, "drill", 1000.0), _f(4, "endo", 1011.1), {"k": "ftr", "r": 5}]
    resent = dict(rows[2], pos=[9.0, 9.0, 9.0])     # r=2 again (a retransmitted chunk arriving later)
    lines = _lines([rows[3], rows[0], rows[2], rows[4], rows[1]]) + [json.dumps(resent), "", "{truncated"]
    got, st = rc.order_rows(lines)
    assert [d["r"] for d in got] == [0, 1, 2, 4, 5]
    assert got[2]["pos"] == rows[2]["pos"]
    assert (st["dup_r"], st["missing_r"], st["gap_ranges"], st["bad_lines"]) == (1, 1, [(3, 3)], 1)
    got2, _ = rc.order_rows(_lines([rows[0], resent, rows[2]]))
    assert got2[1]["pos"] == [9.0, 9.0, 9.0]         # "first" = file order, whatever the content
    rec = rc.parse(lines)
    assert [e.r for e in rec.events] == [1, 2, 4]
    assert any("repeated r" in w for w in rec.warnings) and any("missing" in w for w in rec.warnings)


def test_frame_fields_units_quat_order_flags():
    q = (0.5, 0.5, -0.5, 0.5)
    row = _f(1, "drill", 1790000000100.25, quat=q, still=1, hold=1, btn=1,
             raw={"pos": [101.0, -20.0, 500.0], "quat": [0.0, 1.0, 0.0, 0.0], "tip": [1.0, 0.0, 666.0]})
    f = rc.frame_fields(row, row["tExp"])
    assert f["pos"] == pytest.approx([0.1, -0.02, 0.5]) and f["pos_raw"] == pytest.approx([0.101, -0.02, 0.5])
    assert f["quat"] == pytest.approx((0.5, -0.5, 0.5, 0.5)) and f["quat_raw"] == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert np.allclose(rc.quat_xyzw_to_R(f["quat"]), _R_wxyz(q))   # same rotation, not just a permutation
    assert f["tip"] == pytest.approx([0.11, -0.02, 0.55]) and f["tip_raw"] == pytest.approx([0.001, 0.0, 0.666])
    assert f["stamp_ms"] == 1790000000100.25 and f["t_send_ms"] == pytest.approx(1790000000125.65, abs=1e-3)
    assert (f["tool"], f["frame"], f["visible"], f["still"], f["held"], f["lost"]) == ("drill", 1001, 4, True, True,
                                                                                       False)
    assert f["rmse"] == pytest.approx(0.4) and f["has_raw"]
    lost = rc.frame_fields({"k": "f", "r": 2, "tool": "endo", "tExp": 5.0, "tSrc": 30, "lost": 1}, 5.0)
    assert lost["lost"] and "pos" not in lost and lost["t_send_ms"] == 30.0
    no_raw = rc.frame_fields({k: v for k, v in row.items() if k != "raw"}, 1.0)
    assert not no_raw["has_raw"] and no_raw["pos_raw"] == no_raw["pos"] and no_raw["tip_raw"] is None
    rec = rc.parse(_lines([_hdr(), row, dict(_f(2, "endo", 1790000000101.0), lost=1)]))
    assert (rec.stats["lost"], rec.stats["hold"], rec.stats["f_rows"]) == (1, 1, {"drill": 1, "endo": 1})


def test_registration_tf_direction_point_round_trip():
    T = rc.to_metres(rc.rigid_to_matrix(T0_MM))
    p_cam = np.array([75.6, 121.4, 642.2])                       # mm, camera frame
    p_scene = R0 @ p_cam + np.array(T0_MM["t"])                   # the viewer: scene = R·cam + t (composeToolPose)
    scene_parent, tr, q = rc.registration_tf(T, "scene")          # frame_id scene, child camera
    assert scene_parent
    p = rc.quat_xyzw_to_R(q) @ (p_cam / 1000.0) + tr               # TF applies a transform child → parent
    assert np.allclose(p * 1000.0, p_scene, atol=1e-9)
    scene_parent, tr2, q2 = rc.registration_tf(T, "camera")      # frame_id camera, child scene
    assert not scene_parent
    back = rc.quat_xyzw_to_R(q2) @ p + tr2
    assert np.allclose(back * 1000.0, p_cam, atol=1e-9)
    wrong = rc.quat_xyzw_to_R(q2) @ (p_cam / 1000.0) + tr2        # the edge the wrong way round: metres off
    assert np.linalg.norm(wrong * 1000.0 - p_scene) > 100.0
    row = _f(1, "endo", 1000.0)                                    # a parsed recording agrees with the row's scene.tip
    rec = rc.parse(_lines([_hdr(), row]))
    _, tr, q = rc.registration_tf(rec.T0, "scene")
    assert np.allclose((rc.quat_xyzw_to_R(q) @ np.array(row["tip"]) / 1000.0 + tr) * 1000.0, row["scene"]["tip"],
                       atol=1e-9)
    with pytest.raises(ValueError):
        rc.rigid_to_matrix({"R": [1, 0, 0, 0, 1, 0, 0, 0, -1], "t": [0, 0, 0]})   # a reflection is not a registration
    with pytest.raises(ValueError):
        rc.registration_tf(T, "tracker_cam")


def test_dynamic_registration_one_tf_per_stamp_before_its_pose():
    eye = np.eye(3).ravel().tolist()
    T1, T2 = {"R": eye, "t": [0.0, 0.0, 100.0]}, {"R": eye, "t": [0.0, 0.0, 200.0]}
    rows = [_hdr(dynamic=True), _f(1, "drill", 1000.0, T=T1), _f(2, "drill", 1011.1, T=T2),
            _f(3, "endo", 1000.0, T=T1),                           # endo written late (held for the `real` join)
            _f(4, "endo", 1011.1, T={"R": eye, "t": [0.0, 0.0, 250.0]})]   # disagrees with the drill row's T
    rec = rc.parse(_lines(rows))
    tfs = [e for e in rec.events if e.kind == "tf"]
    assert [e.t for e in tfs] == [1000.0, 1011.1] and tfs[0].data["T"][2, 3] == pytest.approx(0.1)
    assert rec.stats["tf_conflict"] == 1 and rec.dynamic
    seq = [(e.t, e.kind, e.r) for e in rec.events]
    assert seq == [(1000.0, "tf", 1), (1000.0, "pose", 1), (1000.0, "pose", 3), (1011.1, "tf", 2),
                   (1011.1, "pose", 2), (1011.1, "pose", 4)]
    assert not any(e.kind == "tf" for e in rc.parse(_lines([_hdr(), rows[1]])).events)   # static: no per-row TF


def test_cut_rows_become_scene_points_on_the_texp_axis():
    drawn = {"seq": 1, "tSrc": 990, "tExp": 980.0, "lost": 0, "drillOn": 1, "B": [1.0, 2.0, 3.0], "u": [0.0, 0.0, 1.0],
             "entry": [0.0, 0.0, -50.0], "entryFallback": 0}
    rows = [_hdr(), _f(1, "drill", 1000.0, drawn=drawn),
            _cut(2, c=[(12.0, 22.0, 32.0)], mod=5),               # no tExp → the previous drill row's drawn.tExp
            _cut(3, 990.0, c=[(10.0, 20.0, 30.0), (11.0, 21.0, 31.5)], mod=21),
            _cut(4, mod=2),                                         # no tExp, not before the previous cut
            _cut(5, 1001.0, target="soft", r_mm=2, band=[0, 1], mod=3),
            _cut(6, 1002.0, target="soft", op="reset", c=None, r_mm=None, mod=0),
            _f(7, "drill", 1011.1)]
    rec = rc.parse(_lines(rows))
    cuts = [e for e in rec.events if e.kind == "cut"]
    assert [(e.r, e.t) for e in cuts] == [(2, 980.0), (3, 990.0), (4, 990.0), (5, 1001.0), (6, 1002.0)]
    assert np.allclose(cuts[1].data["points"], [[0.010, 0.020, 0.030], [0.011, 0.021, 0.0315]])
    assert cuts[1].data["radius_m"] == pytest.approx(0.004) and cuts[1].data["label"] == "vertebrae_bone_t000"
    assert cuts[4].data["op"] == "reset" and cuts[4].data["points"].shape == (0, 3)
    s = rec.stats
    assert (s["cut_rows"], s["cut_points"], s["cut_mod"], s["cut_t_fallback"], s["soft_resets"]) == (5, 5, 31, 2, 1)
    acc = tl.CutAccumulator()
    for e in cuts:
        acc.apply(e.data)
    groups = dict(acc.groups())
    assert list(groups) == [("bone", "vertebrae_bone_t000", 0.004)]   # the soft stamps were reset
    assert groups[("bone", "vertebrae_bone_t000", 0.004)].shape == (4, 3)
    assert acc.take_removed() == []                  # the soft group was never handed out: nothing to delete
    assert [(mid, len(p)) for mid, _, p in acc.take_new()] == [(0, 4)] and acc.take_new() == []
    assert (acc.n_rows, acc.n_points, acc.mod) == (5, 5, 31)
    acc.clear()
    assert acc.groups() == [] and acc.take_removed() == [0]
    early = rc.parse(_lines([_hdr(), _cut(1), _f(2, "endo", 1000.0)]))   # a cut before any f row gets its time
    assert [(e.kind, e.t) for e in early.events] == [("cut", 1000.0), ("pose", 1000.0)]


def test_drawn_events_only_on_change_with_screen_lag():
    d1 = {"seq": 1, "tSrc": 990, "tExp": 980.0, "lost": 0, "drillOn": 0, "B": [10.0, 0.0, 0.0], "u": [0.0, 0.0, 2.0],
          "entry": [0.0, 0.0, -50.0], "entryFallback": 0}
    d2 = dict(d1, seq=3, tExp=1000.0, drillOn=1, B=[11.0, 0.0, 0.0])
    e1 = {"seq": 1, "tSrc": 990, "tExp": 990.0, "lost": 0, "tip": [0.0, 5.0, 0.0], "viewDir": [0.0, 1.0, 0.0],
          "depth": 10.0, "roll": 100.0, "entry": [0.0, -5.0, 0.0]}
    rows = [_hdr(), _f(1, "drill", 1000.0, drawn=d1), _f(2, "endo", 1000.0, drawn=e1),
            _f(3, "drill", 1011.1, drawn=d1), _f(4, "drill", 1022.2, drawn=d2)]
    rec = rc.parse(_lines(rows))
    dr = [e for e in rec.events if e.kind == "drawn"]
    assert [(e.t, e.data["tool"]) for e in dr] == [(1000.0, "drill"), (1000.0, "endo"), (1022.2, "drill")]
    assert dr[0].data["p"] == pytest.approx([0.01, 0.0, 0.0]) and dr[0].data["dir"] == pytest.approx([0.0, 0.0, 1.0])
    assert not dr[0].data["drill_on"] and dr[2].data["drill_on"] and dr[2].data["lag_ms"] == pytest.approx(22.2)
    assert dr[1].data["p"] == pytest.approx([0.0, 0.005, 0.0]) and dr[1].data["roll_deg"] == 100.0
    poses = [e for e in rec.events if e.kind == "pose"]
    assert [e.data["screen_lag_ms"] for e in poses] == pytest.approx([20.0, 10.0, 31.1, 22.2])
    assert not any(e.kind == "drawn" for e in rc.parse(_lines(rows), drawn=False).events)


def test_ev_rows_take_the_previous_f_time_and_v2core_falls_back_to_tsrc():
    a, b = _f(2, "endo", 1000.0), _f(4, "drill", 1011.1)
    del a["tExp"], b["tExp"]
    rows = [_hdr(timingSpec="v2-core"), {"k": "ev", "r": 1, "i": 0, "tMono": 1.0, "ev": "note"}, a,
            {"k": "ev", "r": 3, "i": 1, "tMono": 2.0, "ev": "zr_down"}, b]
    rec = rc.parse(_lines(rows))
    assert [(e.t, e.kind, e.r) for e in rec.events] == [(1025.0, "ev", 1), (1025.0, "pose", 2), (1025.0, "ev", 3),
                                                        (1036.0, "pose", 4)]
    assert rec.stats["t_fallback"] == 2 and rec.stats["ev_names"] == {"note": 1, "zr_down": 1}
    assert any("tSrc" in w for w in rec.warnings) and any("v2-core" in w for w in rec.warnings)


def test_header_checks_and_warnings():
    with pytest.raises(ValueError):
        rc.parse(_lines([_f(1, "endo", 1000.0)]))                   # no hdr (r=0)
    with pytest.raises(ValueError):
        rc.parse(_lines([_hdr(quatOrder="xyzw"), _f(1, "endo", 1000.0)]))
    carved = {"cutRows": True, "drawn": True, "version": 1, "carvedBeforeStart": {"bone": True, "soft": False}}
    hdr = _hdr(replay=carved, tracker={"cfg": {"tools": {"endo": {}, "endo2": {}, "drill": {}}}})
    rec = rc.parse(_lines([hdr, _f(1, "endo", 1000.0), {"k": "ftr", "r": 2, "tools": {"endo": {"nFrames": 2}}}]))
    w = " | ".join(rec.warnings)
    assert "carved before start (bone)" in w and "endo2" in w and "nFrames 2" in w
    assert rec.footer["r"] == 2 and rec.drill_radius_m == pytest.approx(0.004) and rec.tools == ["endo"]


def test_screen_axis_for_drawn_cut_and_ev():
    hdr = _hdr(t0Wall=1000.0, t0Mono=500.0)          # viewer clock: epoch = tMono + 500
    d1 = {"seq": 1, "tSrc": 990, "tExp": 980.0, "lost": 0, "drillOn": 1, "B": [1.0, 2.0, 3.0], "u": [0.0, 0.0, 1.0],
          "entry": [0.0, 0.0, -50.0], "entryFallback": 0}
    rows = [hdr,
            _f(1, "drill", 1000.0, tMono=540.0, drawn=d1),                     # reached the recorder at 1040
            _cut(2, 990.0, tMono=560.0),                                        # the engine stamped at 1060
            {"k": "ev", "r": 3, "i": 1, "tMono": 545.0, "ev": "zr_down"},      # 1045
            _f(4, "drill", 1011.1, tMono=505.0, drawn=dict(d1, tExp=1000.0))]  # 1005 < its tExp: never before it
    rec = rc.parse(_lines(rows))
    assert [(e.t, e.kind, e.r) for e in rec.events] == [
        (1000.0, "pose", 1), (1011.1, "pose", 4), (1011.1, "drawn", 4), (1040.0, "drawn", 1), (1045.0, "ev", 3),
        (1060.0, "cut", 2)]
    assert [e.data["lag_ms"] for e in rec.events if e.kind == "drawn"] == pytest.approx([11.1, 20.0])   # tExp − drawn
    assert rec.stats["screen_fallback"] == 0 and not any("t0Wall" in w for w in rec.warnings)
    old = rc.parse(_lines([_hdr()] + rows[1:]))     # no t0 in the header: everything on the tracker axis
    assert [(e.t, e.kind, e.r) for e in old.events if e.kind != "pose"] == [
        (990.0, "cut", 2), (1000.0, "drawn", 1), (1000.0, "ev", 3), (1011.1, "drawn", 4)]
    assert old.stats["screen_fallback"] == 4 and any("t0Wall" in w for w in old.warnings)


def test_malformed_nested_rows_are_skipped_not_fatal():
    rows = [_hdr(dynamic=True), _f(1, "endo", 1000.0),
            dict(_f(2, "drill", 1000.0), drawn=[1, 2]),                  # drawn not an object: row skipped
            dict(_f(3, "endo", 1011.1), raw="x"),                         # raw not an object: row skipped
            dict(_f(4, "drill", 1011.1), T=[1, 2, 3]),                    # T not a RigidT
            dict(_f(5, "endo", 1022.2), quat=[0.0, 0.0, 0.0, 0.0]),      # not a rotation
            dict(_f(6, "drill", 1022.2), drawn={"tExp": 1000.0, "B": "x"}),   # drawn bad inside: the pose still plays
            {"k": "cut", "r": 7, "c": "abc", "rMm": 4, "tExp": 1000.0},   # centres unusable
            {"k": "ev", "r": 8, "ev": "note", "tMono": "x"},              # tMono unusable: tracker axis
            {"k": "f", "r": 9, "tool": "endo"}]                            # no time at all
    rec = rc.parse(_lines(rows))
    assert [(e.kind, e.r) for e in rec.events] == [("pose", 1), ("pose", 6), ("ev", 8)]
    assert rec.stats["bad_rows"] == 7 and any("7 malformed rows skipped" in w for w in rec.warnings)
    for reg in ([1, 2], {"T0": {"R": "bad", "t": [0, 0, 0]}, "dynamic": False}):
        r2 = rc.parse(_lines([_hdr(registration=reg), _f(1, "endo", 1000.0)]))
        assert r2.T0 is None and r2.static_T is None and any("T0" in w for w in r2.warnings)


def test_dynamic_registration_without_row_T_falls_back_to_T0():
    rec = rc.parse(_lines([_hdr(dynamic=True), _f(1, "endo", 1000.0), _f(2, "drill", 1000.0)]))
    assert rec.dynamic and not any(e.kind == "tf" for e in rec.events)
    assert rec.static_T is not None and np.allclose(rec.static_T, rec.T0)
    assert any("T0 used as a static registration" in w for w in rec.warnings)
    assert rc.parse(_lines([_hdr(dynamic=True), _f(1, "endo", 1000.0, T=T0_MM)])).static_T is None   # rows carry T
    assert rc.parse(_lines([_hdr(), _f(1, "endo", 1000.0)])).static_T is not None                      # static


def test_drill_size_change_resizes_the_drawn_drill():
    d = {"seq": 1, "tSrc": 990, "tExp": 980.0, "lost": 0, "drillOn": 0, "B": [1.0, 2.0, 3.0], "u": [0.0, 0.0, 1.0]}
    e = {"seq": 1, "tSrc": 990, "tExp": 990.0, "lost": 0, "tip": [0.0, 5.0, 0.0], "viewDir": [0.0, 1.0, 0.0]}
    rows = [_hdr(), _f(1, "drill", 1000.0, drawn=d),
            {"k": "ev", "r": 2, "i": 1, "tMono": 2.0, "ev": "drill_size_changed", "d": 2.5, "size": "small"},
            _f(3, "drill", 1011.1, drawn=dict(d, tExp=1000.0)), _f(4, "endo", 1011.1, drawn=e),
            {"k": "ev", "r": 5, "i": 3, "tMono": 5.0, "ev": "drill_size_changed", "d": "big"},   # unusable: ignored
            _f(6, "drill", 1022.2, drawn=dict(d, tExp=1011.1))]
    rec = rc.parse(_lines(rows))
    got = [(e.data["tool"], e.data["radius_m"]) for e in rec.events if e.kind == "drawn"]
    assert got == [("drill", 0.004), ("drill", 0.0025), ("endo", None), ("drill", 0.0025)]
    assert rec.stats["drill_size_changes"] == 1


def test_cut_markers_grow_by_chunk_and_hand_out_each_stamp_once():
    acc = tl.CutAccumulator(chunk=3)
    pts = np.arange(30, dtype=float).reshape(10, 3)

    def bone(a, b):
        return {"target": "bone", "label": "b", "points": pts[a:b], "radius_m": 0.004, "mod": b - a}

    acc.apply(bone(0, 2))
    first = acc.take_new()
    assert [(mid, len(p)) for mid, _, p in first] == [(0, 2)] and acc.take_new() == []   # nothing new: no work
    acc.apply(bone(2, 5))
    acc.apply(bone(5, 7))
    second = acc.take_new()
    assert [(mid, len(p)) for mid, _, p in second] == [(0, 1), (1, 3), (2, 1)]   # fill chunk 0, then 1, then 2
    handed = np.concatenate([p for _, _, p in first + second])
    assert np.array_equal(handed, pts[:7])                                       # every stamp once, in order
    soft = {"target": "soft", "label": None, "points": np.zeros((4, 3)), "radius_m": 0.002, "mod": 1}
    acc.apply(soft)
    assert [(mid, len(p)) for mid, _, p in acc.take_new()] == [(10000, 3), (10001, 1)]
    acc.apply({"target": "soft", "op": "reset"})
    acc.apply(dict(soft, points=np.zeros((1, 3))))     # re-created: starts again at chunk 0 (the player re-adds it)
    assert acc.take_removed() == [10000, 10001] and [(m, len(p)) for m, _, p in acc.take_new()] == [(10000, 1)]
    assert (acc.n_rows, acc.n_points, acc.mod) == (6, 12, 9)
    acc.clear()
    assert acc.take_removed() == [0, 1, 2, 10000] and acc.groups() == [] and acc.take_new() == []
    with pytest.raises(ValueError):
        tl.CutAccumulator(chunk=0)


def test_texp_stamp_split_keeps_microseconds():
    sec, ns = split_stamp_ms(1790175900146.187)
    assert sec == 1790175900 and abs(ns - 146_187_000) < 1_000
    s = tl.Schedule([1790175900146.187], 1790175900146.187, 1790175900146.187, 0.5)
    s.start(10.0, 5_000_000_000_000.0)
    assert split_stamp_ms(s.wall_ms(1790175900146.187 + 100.0)) == (5_000_000_000, 200_000_000)   # 100 ms at 0.5×


def test_schedule_paces_by_rate_and_catches_up_in_order():
    s = tl.Schedule([1000.0, 1010.0, 1020.0, 2000.0], 1000.0, 2000.0, rate=2.0)
    s.start(100.0, 0.0)
    assert list(s.due(100.0)) == [0]
    assert list(s.due(100.004)) == []                # 8 recorded ms
    assert list(s.due(100.0051)) == [1]              # 10.2 recorded ms at 2×
    assert list(s.due(100.6, limit=1)) == [2]        # late: everything due, in order, `limit` per call
    assert list(s.due(100.6)) == [3]
    assert s.due_mono_s(2000.0) == pytest.approx(100.5)
    assert s.finished(100.6) and not s.finished(100.49)
    assert s.position_ms(101.0) == 2000.0 and s.progress(100.25) == pytest.approx(0.5)
    s.start(200.0, 0.0)                              # loop: from the top again
    assert list(s.due(200.0)) == [0] and not s.finished(200.0)
    for bad in (([1.0], 0.0, 1.0, 0.0), ([2.0, 1.0], 0.0, 2.0, 1.0), ([1.0], 2.0, 1.0, 1.0)):
        with pytest.raises(ValueError):
            tl.Schedule(*bad)


def test_window_applies_earlier_cuts_and_carries_the_dynamic_transform():
    T = {"R": np.eye(3).ravel().tolist(), "t": [0.0, 0.0, 100.0]}
    rows = [_hdr(dynamic=True), _f(1, "drill", 1000.0, T=T), _cut(2, 1000.0), _f(3, "drill", 1500.0, T=T),
            _cut(4, 1500.0), _f(5, "drill", 2000.0, T=T), _f(6, "drill", 3000.0, T=T)]
    rec = rc.parse(_lines(rows))
    pre, inside, a, b = tl.window(rec, start_s=0.7, end_s=1.5)
    assert (a, b) == (1700.0, 2500.0) and [e.t for e in pre] == [1000.0, 1500.0]
    assert [(e.t, e.kind) for e in inside] == [(1700.0, "tf"), (2000.0, "tf"), (2000.0, "pose")]
    pre, inside, a, b = tl.window(rec)
    assert (a, b, pre) == (1000.0, 3000.0, []) and len(inside) == len(rec.events)
    with pytest.raises(ValueError):
        tl.window(rec, start_s=5.0, end_s=4.0)


def test_ball_model_and_balls_in_camera(tmp_path):
    p = tmp_path / "tool_calib.json"
    p.write_bytes(json.dumps({"points_local": [[0, 0, 0], [30, 0, 0], [0, 40, 0], [0, 0, 50]],
                              "sphere_radius_mm": 7.0}).encode())
    pts, radius, sha = rc.load_ball_model(p)
    assert radius == pytest.approx(0.007) and pts.shape == (4, 3)
    assert sha == hashlib.sha256(p.read_bytes()).hexdigest()[:12]
    q = (math.cos(0.35), 0.0, 0.0, math.sin(0.35))   # 40° about z
    got = rc.balls_in_camera(pts, [0.1, 0.2, 0.3], quat_wxyz_to_xyzw(q))
    assert np.allclose(got, pts @ _R_wxyz(q).T + [0.1, 0.2, 0.3])


def test_topic_namespace():
    assert [rc.topic_namespace(s) for s in ("/replay", "replay", "/replay/", "", "/", " a/b ")] == \
        ["/replay", "/replay", "/replay", "", "", "/a/b"]


def test_parsing_and_timeline_modules_are_ros_free():
    for mod in (rc, tl):
        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        mods = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not [m for m in mods if m.startswith(("rclpy", "tf2", "rosgraph")) or m.endswith("_msgs")], mod


def test_player_node_and_launch_compile():   # they need ROS to run; here at least they must parse
    pkg = Path(__file__).resolve().parents[1]   # the package source (tests run from the source tree)
    for p in (pkg / "spinearm_bridge/recording_player.py", pkg / "launch/replay.launch.py"):
        compile(p.read_text(encoding="utf-8"), str(p), "exec")


SAMPLE = os.environ.get("SPINE_REC_SAMPLE", "")


@pytest.mark.skipif(not (SAMPLE and Path(SAMPLE).is_file()), reason="SPINE_REC_SAMPLE not set (a real recording)")
def test_real_recording():
    rec = rc.load(SAMPLE)
    s = rec.stats
    print("\n" + "\n".join(rc.summary(rec) + [f"WARN {w}" for w in rec.warnings]))
    print(f"timeline {rec.t_first:.3f} … {rec.t_last:.3f} epoch ms = {rec.duration_s:.3f} s")
    ts = np.array([e.t for e in rec.events])
    assert np.all(np.diff(ts) >= 0) and s["events"]["pose"] == sum(s["f_rows"].values())
    pose_t = {e.r: e.t for e in rec.events if e.kind == "pose"}
    span = (max(pose_t.values()) - min(pose_t.values())) / 1000.0
    print(f"poses (tracker axis) span {span:.3f} s; with the screen axis (drawn / cut / ev) {rec.duration_s:.3f} s")
    assert all(e.t >= pose_t[e.r] for e in rec.events if e.kind == "drawn")      # the screen never shows the future
    assert all(e.data["t_exp"] is None or e.t >= e.data["t_exp"] for e in rec.events if e.kind == "cut")
    for tool in rec.tools:   # tExp is monotone per tool
        tt = np.array([e.t for e in rec.events if e.kind == "pose" and e.data["fields"]["tool"] == tool])
        assert np.all(np.diff(tt) > 0)
    with open(SAMPLE, encoding="utf-8") as fh:
        rows, _ = rc.order_rows(fh)
    tf_at = {e.t: e.data["T"] for e in rec.events if e.kind == "tf"}
    worst, n = 0.0, 0
    for row in rows:   # the camera tip through the published scene←camera edge = the viewer's own scene.tip
        if row.get("k") == "f" and row.get("scene") and row.get("tip") and row.get("tExp") is not None:
            T = tf_at.get(float(row["tExp"])) if rec.dynamic else rec.T0
            if T is None:
                continue
            _, tr, q = rc.registration_tf(T, "scene")
            p = rc.quat_xyzw_to_R(q) @ (np.asarray(row["tip"], float) / 1000.0) + tr
            worst, n = max(worst, float(np.linalg.norm(p * 1000.0 - row["scene"]["tip"]))), n + 1
    print(f"scene.tip vs TF(camera tip): {n} rows, max {worst:.4f} mm")
    assert n > 0 and worst < 0.01
    if "_0551" in Path(SAMPLE).name:   # the recording audited on 2026-09-26
        assert s["f_rows"] == {"endo": 6451, "drill": 6450} and s["events"]["ev"] == 669
        assert (s["cut_rows"], s["cut_points"], s["cut_mod"], s["dup_r"], s["missing_r"]) == (428, 459, 33392, 0, 0)
        assert 98.7 < span < 98.9 and s["screen_fallback"] == 0 and rec.static_T is not None
