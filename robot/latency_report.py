#!/usr/bin/env python3
"""Stage latencies, end-to-end delay and command-vs-actual error of an `endo_follow.py` run.

  python robot/latency_report.py ~/il-robot/runs/latency/<run>      → <run>/summary.json, report.md, *.png

Definitions (all positions: endoscope tip, tracker camera frame, mm):
  target(t)  follow: R0 + (hand tip(t) − H0) at the hand frame's mid-exposure time
             steps : R0 + scripted offset, at the command time
  robot(t)   the arm-carried constellation's raw tip at its frame's mid-exposure time
  τ_phys     delay that best aligns robot(t + τ) with target(t) (hand moves → arm moves)
  loop       τ_phys + the time the tracker needs to report the arm's frame (arm moves → system knows)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from lerobot_robot_spinearm import handeye as he
from lerobot_robot_spinearm import latency as L
from lerobot_robot_spinearm.protocol import JOINTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    run = Path(a.run).expanduser()
    cfg = json.loads((run / "config.json").read_text())
    ev = L.load(run / "events.jsonl")
    mode = cfg["args"]["mode"]
    tip_m = np.asarray(cfg["tip_local_m"])
    Zi = he.inv(np.asarray(cfg["T_base_cam"]))
    offs = [c["off"] for c in ev["clock"]] or [cfg["clock_wall_minus_mono_ms"]]
    off = float(np.median(offs))  # wall = mono + off (ms)
    out: dict = {"run": str(run), "mode": mode, "clock_off_spread_ms": round(float(np.ptp(offs)), 3),
                 "args": {k: cfg["args"].get(k) for k in ("ik", "step_deg", "max_joint_dps", "ori_weight", "post_weight",
                                                          "source", "pos_only", "ik_iters", "amp_mm")}}

    # ---------------- roles per frame (the tracker's names swap when the markers cross)
    role = {f["fn"]: {"robot": f["robot"], "hand": f["hand"]} for f in ev["frame"]}
    # only while following (the move to the start pose leaves the tap socket unread for ~1–2 s)
    t_eng0 = min((e["t"] for e in ev["engage"]), default=0.0)
    taps = [m for m in ev["tap"] if (m.get("ts") or 0) >= t_eng0 - 200]
    by_role: dict[str, list[dict]] = {"hand": [], "robot": []}
    for m in taps:
        r = role.get(m.get("fn"))
        if not r or m.get("lost"):
            continue
        for k in ("robot", "hand"):
            if r[k] == m["tool"]:
                by_role[k].append(m)

    # ---------------- stage table
    st: dict = {}
    for k, ms in by_role.items():
        st[k] = {"1_expo_to_arrival": L.stats([m["ta"] - m["ts"] for m in ms]),
                 "2_arrival_to_pose": L.stats([m["tp"] - m["ta"] for m in ms]),
                 "3_pose_to_send": L.stats([m["tsend"] - m["tp"] for m in ms]),
                 "4_udp": L.stats([m["trecv"] - m["tsend"] for m in ms])}
    cmds = ev["cmd"]
    cmd_by_seq = {c["seq"]: c for c in cmds}
    st["follower"] = {"5a_frame_to_ik_start": L.stats([c["tik0"] - c["trecv"] for c in cmds]),
                      "5b_ik": L.stats([c["tik"] - c["tik0"] for c in cmds]),
                      "6_goal_rpc_roundtrip": L.stats([c["tack"] - c["tik"] for c in cmds]),
                      "6a_goal_send_to_server_rx": L.stats([c["rx"] * 1000 + off - c["tik"] for c in cmds
                                                            if c.get("rx") is not None])}
    # first bus write of each goal only: with no newer goal the server rewrites the same one every tick
    writes, seen_seq = [], set()
    for s_ in ev["state"]:
        if s_.get("tw") is not None and s_.get("gseq") in cmd_by_seq and s_["gseq"] not in seen_seq:
            seen_seq.add(s_["gseq"])
            writes.append(s_)
    ticks = [s for s in ev["state"] if s.get("tr") is not None]
    st["server"] = {
        "7_goal_rx_to_bus_write": L.stats([(s["tw"] - s["grx"]) * 1000 for s in writes]),
        "7a_tick_start_to_pos_read_done": L.stats([(s["tr"] - s["t"]) * 1000 for s in ticks]),
        "7b_pos_read_done_to_write_done": L.stats([(s["tw"] - s["tr"]) * 1000 for s in writes]),
        "tick_period": L.stats(np.diff([s["t"] * 1000 for s in ev["state"]])),
        "goals_per_write": round(len(cmds) / max(len(writes), 1), 2)}
    if mode == "follow":
        st["e2e"] = {"hand_expo_to_bus_write": L.stats([s["tw"] * 1000 + off - cmd_by_seq[s["gseq"]]["ts_ref"]
                                                        for s in writes if cmd_by_seq[s["gseq"]].get("ts_ref")])}
    else:
        st["e2e"] = {"command_to_bus_write": L.stats([s["tw"] * 1000 + off - cmd_by_seq[s["gseq"]]["tik"]
                                                      for s in writes])}
    out["stages_ms"] = st

    # ---------------- trajectories
    rob = sorted(by_role["robot"], key=lambda m: m["ts"])
    t_rob = np.array([m["ts"] for m in rob])
    x_rob = np.array([m["raw_tip"] for m in rob]) if rob else np.zeros((0, 3))
    engaged_cmds = [c for c in cmds if not c.get("clamped")]
    if mode == "follow":
        t_tgt = np.array([c["ts_ref"] for c in engaged_cmds])
    else:
        t_tgt = np.array([c["tik"] for c in engaged_cmds])
    x_tgt = np.array([c["target_tip_cam_mm"] for c in engaged_cmds]) if engaged_cmds else np.zeros((0, 3))
    out["n"] = {"frames": len(ev["frame"]), "robot_fixes": len(rob), "hand_fixes": len(by_role["hand"]),
                "cmds": len(cmds), "clamped": len(cmds) - len(engaged_cmds), "writes": len(writes),
                "engages": len(ev["engage"]), "disengages": [d["reason"] for d in ev["disengage"]]}
    if len(t_tgt) < 30 or len(t_rob) < 30:
        out["error"] = "too few samples"
        (run / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
        print(json.dumps(out, indent=1, ensure_ascii=False))
        sys.exit(1)

    delays: dict = {"phys_cmd_target_to_robot": L.delay_fit(t_tgt, x_tgt, t_rob, x_rob)}
    if mode == "follow":
        # hand raw (true motion) → target built from it the same way, so the tracker filter lag is included
        hand = sorted(by_role["hand"], key=lambda m: m["ts"])
        eng = ev["engage"]
        t_h = np.array([m["ts"] for m in hand])
        raw_tgt, filt_tip = [], []
        for m in hand:
            e = max((e for e in eng if e["t"] <= m["trecv"]), key=lambda e: e["t"], default=None)
            if e is None:
                raw_tgt.append([np.nan] * 3)
                filt_tip.append([np.nan] * 3)
                continue
            raw_tgt.append(np.array(e["R0_tip_mm"]) + np.array(m["raw_tip"]) - np.array(e["H0_tip_mm"]))
            R = he.quat_to_R(m["quat"])
            filt_tip.append(R @ (tip_m * 1000) + np.array(m["pos"]))
        raw_tgt, filt_tip = np.array(raw_tgt), np.array(filt_tip)
        ok = np.isfinite(raw_tgt).all(1)
        rng = (t_h >= t_tgt.min()) & (t_h <= t_tgt.max()) & ok
        delays["phys_hand_raw_to_robot"] = L.delay_fit(t_h[rng], raw_tgt[rng], t_rob, x_rob)
        raw_tip = np.array([m["raw_tip"] for m in hand])
        delays["tracker_filter_raw_to_output"] = L.delay_fit(t_h[rng], raw_tip[rng], t_h[rng], filt_tip[rng], max_ms=300)
    # encoder path: goal FK (at command time) → encoder FK (at the position read) → marker
    from lerobot_robot_spinearm.ee import MarkerKinematics

    kin = MarkerKinematics(urdf=cfg["urdf"], frame=cfg["frame"])
    kin.set_limits({j: tuple(v) for j, v in cfg["bounds"].items()})
    joff = np.asarray(cfg.get("joint_offset_deg") or [0.0] * 6)  # model angle = arm angle + joff (q_goal is model)
    t_goal = np.array([c["tik"] for c in engaged_cmds])
    x_goal = np.array([(Zi @ kin.fk(c["q_goal"]))[:3, 3] * 1000 for c in engaged_cmds])
    ts_enc = [s for s in ev["state"] if s.get("tr") is not None and s.get("pos")]
    t_enc = np.array([(s["t"] + s["tr"]) / 2 * 1000 + off for s in ts_enc])
    x_enc = np.array([(Zi @ kin.fk(np.array([s["pos"][j] for j in JOINTS]) + joff))[:3, 3] * 1000 for s in ts_enc])
    m_enc = (t_enc >= t_goal.min() - 500) & (t_enc <= t_goal.max() + 1000)
    t_enc, x_enc = t_enc[m_enc], x_enc[m_enc]
    delays["goal_to_encoder"] = L.delay_fit(t_goal, x_goal, t_enc, x_enc)
    delays["encoder_to_marker"] = L.delay_fit(t_enc, x_enc, t_rob, x_rob, max_ms=200)
    out["delays_ms"] = delays

    tau = delays["phys_cmd_target_to_robot"].get("tau_ms") or 0.0
    rp = st["robot"]
    perc = sum((rp[k] or {}).get("median", 0) for k in ("1_expo_to_arrival", "2_arrival_to_pose", "3_pose_to_send", "4_udp"))
    out["loop_ms"] = {"phys": tau, "robot_perception": round(perc, 2), "closed_loop": round(tau + perc, 1)}
    if mode == "follow" and delays.get("phys_hand_raw_to_robot", {}).get("tau_ms") is not None:
        out["loop_ms"]["phys_incl_filter"] = delays["phys_hand_raw_to_robot"]["tau_ms"]
        out["loop_ms"]["closed_loop_incl_filter"] = round(delays["phys_hand_raw_to_robot"]["tau_ms"] + perc, 1)

    # ---------------- errors (question 2)
    y0 = L.interp3(t_tgt, t_rob, x_rob)
    yt = L.interp3(t_tgt + tau, t_rob, x_rob)
    e0, et = y0 - x_tgt, yt - x_tgt
    errs = {"total_at_same_time": L.err_stats(e0), "shape_after_delay": L.err_stats(et)}
    if engaged_cmds and "target_raw_tip_cam_mm" in engaged_cmds[0]:  # vs the human motion before any filter
        x_raw = np.array([c["target_raw_tip_cam_mm"] for c in engaged_cmds])
        d_h = L.delay_fit(t_tgt, x_raw, t_rob, x_rob)
        delays["human_raw_to_robot"] = d_h
        errs["vs_human_same_time"] = L.err_stats(y0 - x_raw)
        errs["vs_human_after_delay"] = L.err_stats(L.interp3(t_tgt + (d_h.get("tau_ms") or 0), t_rob, x_rob) - x_raw)
        errs["filter_change"] = L.err_stats(x_tgt - x_raw)
    wins = L.still_windows(t_tgt, x_tgt, 5.0, 0.5)
    ss = []
    for (t0, t1) in wins:
        tq = np.array([t1])
        y = L.interp3(tq, t_rob, x_rob)
        k = int(np.searchsorted(t_tgt, t1)) - 1
        ss.append(y[0] - x_tgt[k])
    errs["steady_state"] = L.err_stats(np.array(ss)) if ss else None
    # decomposition: IK residual, servo (goal vs encoder, aligned), model/calibration (encoder FK vs marker)
    errs["ik_residual"] = L.stats([c["ik_err_mm"] for c in engaged_cmds])
    g_at_enc = L.interp3(t_enc - (delays["goal_to_encoder"].get("tau_ms") or 0), t_goal, x_goal)
    errs["servo_goal_vs_encoder_after_delay"] = L.err_stats(x_enc - g_at_enc)
    r_at_enc = L.interp3(t_enc, t_rob, x_rob)
    errs["model_encoderFK_vs_marker"] = L.err_stats(x_enc - r_at_enc)
    # displacement gain/direction: least-squares map target Δ → robot Δ (after delay)
    mk = np.isfinite(yt).all(1)
    if mk.sum() > 30:
        A = x_tgt[mk] - x_tgt[mk].mean(0)
        B = yt[mk] - yt[mk].mean(0)
        G, *_ = np.linalg.lstsq(A, B, rcond=None)
        U, S, Vt = np.linalg.svd(G)
        errs["gain_svals"] = np.round(S, 3).tolist()
        errs["direction_err_deg"] = round(he.R_angle_deg(U @ Vt), 2)
    # orientation (follow, not pos-only): target R = dR·R0 vs robot R
    if mode == "follow" and not cfg["args"]["pos_only"]:
        ang = []
        eng = ev["engage"]
        rob_R = {m["fn"]: he.quat_to_R(m["raw_quat"]) for m in rob}
        for c in engaged_cmds:
            e = max((e for e in eng if e["t"] <= c["tik"]), key=lambda e: e["t"], default=None)
            if e is None:
                continue
            dR = he.rotvec_to_R(c["drot"])
            Rt = dR @ np.array(e["R0_R"])
            fr = [f for f in rob if f["ts"] >= c["ts_ref"] + tau][:1]
            if fr:
                ang.append(he.R_angle_deg(Rt.T @ rob_R[fr[0]["fn"]]))
        errs["orientation_after_delay_deg"] = L.stats(ang)
    # per robot-base axis (x, y horizontal, z = up against gravity): error RMS and the robot's own jitter
    # (tip minus its 0.3 s centred moving average)
    R_bc = np.asarray(cfg["T_base_cam"])[:3, :3]
    eb = e0[np.isfinite(e0).all(1)] @ R_bc.T
    k = 27
    xb = x_rob @ R_bc.T
    if len(xb) > 3 * k:
        sm = np.array([np.convolve(xb[:, i], np.ones(k) / k, mode="same") for i in range(3)]).T
        errs["base_xyz_rms"] = np.round(np.sqrt((eb ** 2).mean(0)), 2).tolist()
        errs["robot_jitter_base_xyz"] = np.round((xb - sm)[k:-k].std(0), 2).tolist()
    out["errors_mm"] = errs

    # ---------------- steps mode: step responses and sine
    if mode == "steps":
        e0t = ev["engage"][0]["t"]
        segs = []
        A = cfg["args"]["amp_mm"]
        tt = np.array([c["tik"] for c in cmds])
        dp = np.array([c["dp_mm"] for c in cmds])
        rel = (tt - e0t) / 1000.0
        bounds_s = [1.0 + 1.5 * i for i in range(13)]
        for i in range(12):
            t_change = e0t + bounds_s[i] * 1000
            t_end = e0t + bounds_s[i + 1] * 1000
            ax = i // 4
            m = L.step_metrics(tt, x_tgt if len(x_tgt) == len(tt) else np.array([c["target_tip_cam_mm"] for c in cmds]),
                               t_rob, x_rob, ax, t_change, t_end, A)
            if m:
                m["axis"] = "xyz"[ax]
                segs.append(m)
        out["steps"] = segs
        sines = []
        for f, s0 in ((0.5, 19.0), (1.0, 25.0)):
            mc = (rel >= s0 + 1) & (rel < s0 + 6)
            mr = (t_rob >= e0t + (s0 + 1) * 1000) & (t_rob < e0t + (s0 + 6) * 1000)
            if mc.sum() > 50 and mr.sum() > 50:
                a_c, p_c = L.sine_fit(tt[mc] - e0t, np.array([c["target_tip_cam_mm"][0] for c in np.array(cmds)[mc]]), f)
                a_r, p_r = L.sine_fit(t_rob[mr] - e0t, x_rob[mr, 0], f)
                lag_deg = (p_c - p_r) % 360
                sines.append({"f_hz": f, "amp_ratio": round(a_r / a_c, 3), "phase_lag_deg": round(lag_deg, 1),
                              "lag_ms": round(lag_deg / 360 / f * 1000, 1)})
        out["sines"] = sines
        _ = dp

    (run / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    if not a.no_plots:
        plots(run, mode, t_tgt, x_tgt, t_rob, x_rob, tau, e0, et)
    (run / "report.md").write_text(report_md(out))
    print((run / "report.md").read_text())


def plots(run, mode, t_tgt, x_tgt, t_rob, x_rob, tau, e0, et):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = t_tgt.min()
    fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for i, n in enumerate("xyz"):
        ax[i].plot((t_tgt - t0) / 1000, x_tgt[:, i], lw=1.2, label="target (hand / command)")
        ax[i].plot((t_rob - t0) / 1000, x_rob[:, i], lw=1.0, label="robot marker (observed)")
        ax[i].set_ylabel(f"tip {n} [mm]")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[3].plot((t_tgt - t0) / 1000, np.linalg.norm(e0, axis=1), lw=0.8, label="error, same time")
    ax[3].plot((t_tgt - t0) / 1000, np.linalg.norm(et, axis=1), lw=0.8, label=f"error, after delay {tau:.0f} ms")
    ax[3].set_ylabel("|error| [mm]")
    ax[3].set_xlabel("time [s]")
    ax[3].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{mode}: target vs robot tip (camera frame)")
    fig.tight_layout()
    fig.savefig(run / "trajectory.png", dpi=110)
    plt.close(fig)


def _row(name, s):
    if not s:
        return f"| {name} | – | – | – |\n"
    return f"| {name} | {s['median']} | {s['p95']} | {s['max']} |\n"


def report_md(o: dict) -> str:
    st = o["stages_ms"]
    md = [f"# 지연·오차 측정 — {o['mode']}\n\n", f"- 실행: `{o['run']}`\n", f"- 설정: {o['args']}\n",
          f"- 표본: {o['n']}\n\n", "## 단계별 시간 (ms)\n\n| 단계 | 중앙값 | p95 | 최대 |\n|---|---|---|---|\n"]
    names = {"1_expo_to_arrival": "노출 중간 → 앱 도착", "2_arrival_to_pose": "도착 → 포즈 계산",
             "3_pose_to_send": "포즈 → UDP 송신", "4_udp": "UDP 전달"}
    for role, label in (("hand", "손"), ("robot", "로봇")):
        for k, n in names.items():
            md.append(_row(f"[{label}] {n}", st[role].get(k)))
    fl = {"5a_frame_to_ik_start": "수신 → IK 시작(식별·FK)", "5b_ik": "IK", "6_goal_rpc_roundtrip": "goal RPC 왕복",
          "6a_goal_send_to_server_rx": "goal 송신 → 서버 수신"}
    for k, n in fl.items():
        md.append(_row(n, st["follower"].get(k)))
    sv = {"7_goal_rx_to_bus_write": "서버 수신 → 버스 쓰기(틱 대기 포함)",
          "7a_tick_start_to_pos_read_done": "틱 시작 → 위치 읽기 끝", "7b_pos_read_done_to_write_done": "위치 읽기 끝 → 쓰기 끝",
          "tick_period": "틱 주기"}
    for k, n in sv.items():
        md.append(_row(n, st["server"].get(k)))
    for k, s in st["e2e"].items():
        md.append(_row({"hand_expo_to_bus_write": "손 촬영 → 버스 쓰기 (합계)",
                        "command_to_bus_write": "명령 → 버스 쓰기 (합계)"}[k], s))
    md.append("\n## 끝-끝 지연 (궤적 정렬)\n\n| 구간 | τ (ms) | 정렬 후 RMS (mm) | 정렬 전 RMS | 움직임 크기 |\n|---|---|---|---|---|\n")
    dn = {"phys_cmd_target_to_robot": "명령 목표 → 로봇 마커(물리)", "phys_hand_raw_to_robot": "손(원시) → 로봇 마커(필터 포함)",
          "tracker_filter_raw_to_output": "트래커 필터 지연(원시 → 출력)",
          "human_raw_to_robot": "사람 원래 움직임(필터 전) → 로봇 마커", "goal_to_encoder": "goal → 엔코더",
          "encoder_to_marker": "엔코더 → 마커"}
    for k, d in o["delays_ms"].items():
        md.append(f"| {dn.get(k, k)} | {d.get('tau_ms')} | {d.get('rms_at_tau')} | {d.get('rms_at_0')} | "
                  f"{d.get('motion_std_mm')} |\n")
    md.append(f"\n폐루프: {o['loop_ms']}\n\n## 명령 대비 실제 오차 (팁, mm)\n\n")
    for k, v in o["errors_mm"].items():
        md.append(f"- {k}: {v}\n")
    if o.get("steps"):
        md.append("\n## 스텝 응답\n\n| 축 | 크기 | 데드타임 | 상승(10–90%) | 오버슈트 % | 정착(±1mm) | 정상상태 오차 |\n|---|---|---|---|---|---|---|\n")
        for s in o["steps"]:
            md.append(f"| {s['axis']} | {s['size_mm']} | {s['dead_ms']} | {s['rise_ms']} | {s['overshoot_pct']} | "
                      f"{s['settle_ms']} | {s['ss_err_mm']} |\n")
        for s in o.get("sines", []):
            md.append(f"\n사인 {s['f_hz']} Hz: 진폭비 {s['amp_ratio']}, 위상 지연 {s['phase_lag_deg']}° ({s['lag_ms']} ms)\n")
    return "".join(md)


if __name__ == "__main__":
    main()
