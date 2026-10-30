"""Synthetic recorder v2 (rec-2.3.0-t0) sessions for the fidelity tests — generated here, never copied from a recording.

Two tools on a 90 Hz camera clock; every frame has tExp (shared by both tools), tSrc = tExp + 24.5 ms (tracker send),
tMono (recorder socket); poses are built in the scene frame and written in camera frame through the registration
(or, for dynamic registrations, through a drifting per-row T). `drawn` lags the rows by LAG frames.
"""

from __future__ import annotations

import json

import numpy as np

from spine_render.geom import Registration, matrix_to_quat, rotvec_to_matrix
from spine_render.script import rig_default

RIG = rig_default()
REG = Registration.from_dict(RIG["registration"])
DT = 1000.0 / 90.0
T_EXP0 = 1_790_175_900_146.187
T_MONO0 = 140_045.0
LAG = 2
R_ENDO = rotvec_to_matrix([0.3, -1.2, 0.4])
R_DRILL = rotvec_to_matrix([1.1, 0.2, -0.5])
BASE = {"endo": np.array([30.0, 139.0, 1052.0]), "drill": np.array([38.0, 93.0, 975.0])}


def scene_tip(tool: str, k: int) -> np.ndarray:
    return BASE[tool] + np.array([0.1 * k, 0.05 * k, -0.02 * k])


def T_row(k: int) -> tuple[np.ndarray, np.ndarray]:
    """Drifting film-reference registration for dynamic sessions: T_k = T0 turned 0.05°/frame, moved 0.2 mm/frame."""
    return rotvec_to_matrix([0, 0, np.radians(0.05 * k)]) @ REG.R, REG.t + np.array([0.2 * k, -0.1 * k, 0.0])


def header(**kw) -> dict:
    h = {"k": "hdr", "r": 0, "schemaVersion": 2, "recorderVersion": "rec-2.3.0-t0", "timingSpec": "v2-t0",
         "coordConvention": "scene=LPS_mm; cam=D435_optical_mm(+Y down)", "sessionId": "s_synth", "meshId": "altician_lumbar1",
         "quatOrder": "wxyz", "registration": {"method": "fid3", "T0": REG.to_dict(), "dynamic": False},
         "drill": {"radiusMm": 4, "sizeKey": "large", "proxy": {"rule": "surface", "enabled": True}},
         "fe": {"rawabs": False, "noproxy": False, "proxy2": False, "bite": 1, "meltk": 2, "meltv": 8, "touchoff": 0.5,
                "meltmax": 40, "film": False, "fid3": 5, "scopecal": "air", "look": None,
                "cfg": {"rawabs": False, "noproxy": False, "proxy2": False, "bite": 1, "meltk": 2, "meltv": 8,
                        "touchoff": 0.5, "meltmax": 40, "film": False, "fid3": True, "scopecal": "air", "look": None},
                "cfgHash": "a4c7b9adfc50"},
         "toolCalib": {"endo": {"sha256_12": "2d48a98bb499", "tipLocal": RIG["tools"]["endo"]["tipLocal"]},
                       "drill": {"sha256_12": "340b2ac02ad2", "tipLocal": RIG["tools"]["drill"]["tipLocal"],
                                 "gravityUpCam": [0.048748, -0.9785, -0.200402]}},
         "tracker": {"cfgHash": "8d76f7834c32", "cfg": {"tools": {"endo": {"calib": {"sha12": "2d48a98bb499"}},
                                                                   "drill": {"calib": {"sha12": "340b2ac02ad2"}}}}},
         "baseline": None, "viewState": {"portalMode": "tracker", "optics": {"autoExposure": {"tauUpS": 0.3}}},
         "replay": {"cutRows": True, "drawn": True, "version": 1, "carvedBeforeStart": {"bone": False, "soft": False}},
         "build": {"gitSha": "7d51bc0c11b1", "gitDirty": False, "gitAt": "server-start"}, "t0Mono": T_MONO0 - 20}
    h.update(kw)
    return h


def make_session(n: int = 60, lost: dict | None = None, btn=lambda k: 0, dynamic: bool = False, cuts: tuple = (10, 20, 40),
                 texp: bool = True, drawn: bool = True, **hdr) -> list[dict]:
    """Rows in r order. lost = {tool: set(frame)}; btn(k) → drill bits or None (unknown → no key); cuts = frames after
    which a cut row (mod 5, dv = index // 2) is written."""
    lost = lost or {}
    rows = [header(**hdr)]
    if dynamic:
        rows[0]["registration"]["dynamic"] = True
    if not texp:
        rows[0]["timingSpec"] = "v2-core"
        rows[0]["replay"] = {"cutRows": True, "drawn": False, "version": 1}
    shown: dict[str, list] = {"endo": [], "drill": []}
    for k in range(n):
        t_exp = T_EXP0 + k * DT
        for j, tool in enumerate(("endo", "drill")):
            calib = RIG["tools"][tool]
            R_s = R_ENDO if tool == "endo" else R_DRILL
            tip_s = scene_tip(tool, k)
            pos_s = tip_s - R_s @ np.asarray(calib["tipLocal"])
            Rr, tr = T_row(k) if dynamic else (REG.R, REG.t)
            R_c = Rr.T @ R_s
            pos_c = Rr.T @ (pos_s - tr)
            r = {"k": "f", "r": len(rows), "i": 2 * k + j, "seq": 1000 + k, "tool": tool, "tSrc": int(t_exp + 24.5),
                 "tMono": round(T_MONO0 + k * DT + 30 + 0.4 * j, 3)}
            if texp:
                r.update(tExp=t_exp, tsd="g", tExpSrc=t_exp, tEmit=t_exp + 24.9)
            if k in lost.get(tool, ()):
                r["lost"] = 1
            else:
                r.update(pos=pos_c.tolist(), quat=matrix_to_quat(R_c).tolist(),
                         tip=(pos_c + R_c @ np.asarray(calib["tipLocal"])).tolist(),
                         shaft=(R_c @ np.asarray(calib["shaftLocal"])).tolist(), vis=4, still=0,
                         scene={"tip": tip_s.tolist()})
                shown[tool].append((t_exp, tip_s, R_s @ np.asarray(calib["shaftLocal"])))
                if dynamic:
                    r["T"] = {"R": Rr.ravel().tolist(), "t": tr.tolist()}
            if tool == "drill":
                b = btn(k)
                if b is not None:
                    r["btn"] = b
            if texp and drawn and len(shown[tool]) > LAG:
                te, tip_d, u = shown[tool][-1 - LAG]
                d = {"seq": 1000 + k - LAG, "tSrc": int(te + 24.5), "tExp": te, "lost": 0}
                if tool == "drill":
                    d.update(B=(tip_d - 4 * u).tolist(), u=u.tolist(), entry=(tip_d - 30 * u).tolist())
                else:
                    d.update(tip=tip_d.tolist(), viewDir=u.tolist(), depth=30.0, entry=(tip_d - 30 * u).tolist())
                r["drawn"] = d
            rows.append(r)
        if k in cuts:
            c = scene_tip("drill", k)
            cut = {"k": "cut", "r": len(rows), "i": 2 * k + 2, "tMono": round(T_MONO0 + k * DT + 31, 3), "tool": "drill",
                   "target": "bone", "label": "vertebrae_bone_t000", "c": [c.tolist()], "rMm": 4, "fn": "smooth",
                   "smooth": 1, "fresh": 1, "dv": cuts.index(k) // 2, "mod": 5}
            if texp:
                cut["tExp"] = t_exp - LAG * DT
            rows.append(cut)
    rows.append({"k": "ftr", "r": len(rows), "endReason": "completed", "durationS": round(n * DT / 1000, 2)})
    return rows


def jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)
