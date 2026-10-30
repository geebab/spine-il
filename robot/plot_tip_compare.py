#!/usr/bin/env python3
"""Recorded tool tip vs the tip estimated from the arm's marker, over the replayed window (scene mm, recording time).

    python robot/plot_tip_compare.py hand.json robot.json -o fig.png [--title ...]

Inputs are replay_to_render.py trajectories of the same window: `--hand` (the recording) and the robot one. Per tool
the arm drove: x / y / z of both, then the distance between them — at the same recording time, and after shifting
the robot by the delay that minimises it (the replay lags the recording). Drill trigger intervals are shaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REC, ROB, ERR, ERR2 = "#2a78d6", "#eb6834", "#52514e", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def track(obj: dict, tool: str):
    f = [x for x in obj["frames"] if tool in x and "tip" in x[tool]]
    t = np.array([x["t"] for x in f])
    p = np.array([x[tool]["tip"] for x in f])
    trig = np.array([bool(x[tool].get("trig")) for x in f])
    return t, p, trig


def resample(t, p, tt):
    return np.stack([np.interp(tt, t, p[:, i]) for i in range(3)], 1)


def best_delay(th, ph, tr, pr, grid):
    best = (np.inf, 0.0)
    for d in grid:
        tt = th[(th >= tr[0] - d) & (th <= tr[-1] - d)]
        if len(tt) < 20:
            continue
        e = np.linalg.norm(ph[np.searchsorted(th, tt)] - resample(tr, pr, tt + d), axis=1)
        r = float(np.sqrt((e ** 2).mean()))
        if r < best[0]:
            best = (r, d)
    return best[1]


def stats(e):
    return f"RMS {np.sqrt((e ** 2).mean()):.2f} · p95 {np.percentile(e, 95):.2f} · 최대 {e.max():.2f} mm"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hand")
    ap.add_argument("robot")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    H, R = json.loads(Path(a.hand).read_text()), json.loads(Path(a.robot).read_text())
    driven = [t for t, v in R["_source"]["tools"].items() if v["source"].startswith("robot")]
    for fam in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "UnDotum"):
        if any(fam in f.name for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = fam
            break
    plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(4, len(driven), figsize=(7.5 * len(driven), 10), sharex=True, squeeze=False)
    summary = {}
    for c, tool in enumerate(driven):
        th, ph, trig = track(H, tool)
        tr, pr, _ = track(R, tool)
        tt = th[(th >= tr[0]) & (th <= tr[-1])]
        ph_t, pr_t = ph[np.searchsorted(th, tt)], resample(tr, pr, tt)
        e0 = np.linalg.norm(ph_t - pr_t, axis=1)
        d = best_delay(th, ph, tr, pr, np.arange(0.0, 0.5, 0.005))
        tt2 = th[(th >= tr[0] - d) & (th <= tr[-1] - d)]
        e1 = np.linalg.norm(ph[np.searchsorted(th, tt2)] - resample(tr, pr, tt2 + d), axis=1)
        summary[tool] = {"same_time": stats(e0), "delay_s": d, "after_delay": stats(e1)}
        on = trig[np.searchsorted(th, tt)] if tool == "drill" else np.zeros(len(tt), bool)
        for i, lab in enumerate("xyz"):
            ax = axs[i, c]
            ax.plot(tt, ph_t[:, i], color=REC, lw=1.6, label="원본 녹화 (손)")
            ax.plot(tt, pr_t[:, i], color=ROB, lw=1.2, label="로봇 마커 추정")
            ax.set_ylabel(f"{lab} (mm)")
            if i == 0:
                ax.set_title(f"{'내시경 (왼팔)' if tool == 'endo' else '드릴 (오른팔)'} 팁 — scene 좌표", loc="left",
                             fontsize=11)
                ax.legend(loc="upper right", frameon=False, fontsize=9)
        ax = axs[3, c]
        ax.plot(tt, e0, color=ERR, lw=1.0, label=f"같은 시각: {stats(e0)}")
        ax.plot(tt2, e1, color=ERR2, lw=1.0, label=f"지연 {d * 1000:.0f} ms 보정: {stats(e1)}")
        ax.set_ylabel("팁 거리 (mm)")
        ax.set_xlabel("녹화 시각 (s, 구간 시작 기준)")
        ax.legend(loc="upper right", frameon=False, fontsize=8)
        if on.any():   # drill trigger held
            edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
            for s_, e_ in zip(edges[::2], edges[1::2]):
                for r_ in range(4):
                    axs[r_, c].axvspan(tt[s_], tt[min(e_, len(tt) - 1)], color="#eda100", alpha=0.12, lw=0)
            axs[3, c].text(0.01, 0.95, "노란 띠 = 드릴 방아쇠", transform=axs[3, c].transAxes, fontsize=8,
                           color=MUTED, va="top")
    src = R["_source"]
    fig.suptitle(a.title or f"{Path(src['recording']).stem} · {src['from_s']:g}–{src['from_s'] + src['len_s']:g} s",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    Path(a.out).with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
