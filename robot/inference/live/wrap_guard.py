#!/usr/bin/env python3
"""Keep a robot run clear of the camera's stamp-counter wrap (2026-10-06).

    python robot/inference/live/wrap_guard.py --fit         # the tracker's journal → stamp_wrap.json, prints the episodes
    python robot/inference/live/wrap_guard.py --wait 780    # if a wrap is due within 780 s: sleep until 90 s after it
    python robot/inference/live/wrap_guard.py               # the next wraps of the stored fit

The D435's hardware frame stamp is a 32-bit microsecond counter: it wraps every 2^32 µs (71.58 min). At about one wrap in
four the tracker's librealsense global_time stamps then drift 0.1-1.7 s (once 9.8 s) from the host clock for a minute
("[t0] ★tExp−host …" in the journal of ube-tracker): 17 of the 22 episodes of 10-02…10-06 start within 1.1 s of one grid of
period 4294.64 s (2^32 µs − 75 ppm; rms 0.4 s over 81 h). While the stamps are behind, the followers skip every frame as
stale and the arms hold (live_loop.py waits). The other episodes (5, all under 0.6 s) have no known trigger.
The grid's phase belongs to the camera's power-up: the fit is made from the journal since the tracker's start, and --wait
does nothing (and says so) when the tracker was restarted after the fit or fewer than 4 episodes lie on one grid.
--fit reads the whole journal of the tracker (about 40 s, most of it system time): it is refused while the arms follow
(endo_follow.py) and within 4 minutes of a wrap of the stored fit — on 10-06 a fit running across the 09:37:32 wrap was
followed by a three-minute episode (−0.7 s). --wait reads the stored fit only.
"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np

P0 = 2 ** 32 / 1e6   # s


def tracker_start() -> str:
    return subprocess.run(["systemctl", "--user", "show", "ube-tracker", "-p", "ActiveEnterTimestamp", "--value"],
                          capture_output=True, text=True).stdout.strip()


def episodes(since: str) -> list:
    """[start (unix s), seconds, extreme tExp−host (ms)] of each run of warnings (gaps over 120 s apart)."""
    out = subprocess.run(["journalctl", "--user", "-u", "ube-tracker", "--since", since, "--no-pager", "-o", "short-unix"],
                         capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        m = re.search(r"★tExp−host ([+\-−]?[0-9.]+)ms", line)
        if m:
            rows.append((float(line.split()[0]), float(m.group(1).replace("−", "-"))))
    eps = []
    for t, v in rows:
        if eps and t - eps[-1][1] <= 120:
            eps[-1][1] = t; eps[-1][2] = max(eps[-1][2], v, key=abs)
        else:
            eps.append([t, t, v])
    return [[a, b - a, v] for a, b, v in eps]


def fit(eps: list) -> dict | None:
    st = np.array([e[0] for e in eps])
    best = None
    for a in range(len(st)):   # anchor on an episode, look for the period (within ±150 ppm of 2^32 µs) with the most starts on one grid
        for ppm in np.arange(-150, 150.5, 0.5):
            P = P0 * (1 + ppm * 1e-6)
            on = np.abs(st - (st[a] + np.round((st - st[a]) / P) * P)) < 6
            if best is None or on.sum() > best[0]:
                best = (int(on.sum()), a, P, on)
    if best is None or best[0] < 4:
        return None
    n, a, P, on = best
    k = np.round((st - st[a]) / P)
    (Pf, t0), *_ = np.linalg.lstsq(np.c_[k[on], np.ones(n)], st[on], rcond=None)
    res = st - (t0 + np.round((st - t0) / Pf) * Pf)
    return {"t0": float(t0), "period_s": float(Pf), "on_grid": n, "episodes": len(st), "rms_s": round(float(np.sqrt((res[on] ** 2).mean())), 2),
            "residual_s": [round(float(r), 1) for r in res]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--wait", type=float, default=None, metavar="S", help="sleep past a wrap that is due within S seconds")
    ap.add_argument("--margin-s", type=float, default=90.0, help="a wrap's episode lasts about a minute")
    ap.add_argument("--file", default="~/il-robot/runs/act_smoke/live/stamp_wrap.json")
    ap.add_argument("--now", type=float, default=None, help="pretend this unix time (test)")
    a = ap.parse_args()
    f = Path(a.file).expanduser()
    old = json.loads(f.read_text()) if f.exists() else {}
    near = old.get("period_s") and old.get("tracker_start") == tracker_start() and \
        abs((time.time() - old["t0"] + old["period_s"] / 2) % old["period_s"] - old["period_s"] / 2) < 240
    if a.fit and (near or subprocess.run(["pgrep", "-f", r"endo_follow\.py"], capture_output=True).returncode == 0):
        print("[wrap] no fit now: " + ("a wrap is within 4 minutes" if near else "the arms are following") + " (reading the journal loads the system)")
    elif a.fit:
        ts = tracker_start(); eps = episodes(ts)
        g = fit(eps) if len(eps) >= 4 else None
        f.write_text(json.dumps({"tracker_start": ts, "fitted": time.strftime("%Y-%m-%d %H:%M:%S"), **(g or {"on_grid": 0, "episodes": len(eps)})}))
        if g and a.wait is None:
            for (t, d, v), r in zip(eps, g["residual_s"]):
                print(time.strftime("%m-%d %H:%M:%S", time.localtime(t)), f"{d:4.0f} s  extreme {v:+8.1f} ms  " + (f"on the grid ({r:+.1f} s)" if abs(r) < 6 else f"off the grid ({r:+.0f} s)"))
    g = json.loads(f.read_text()) if f.exists() else {}
    now = a.now or time.time()
    if not g.get("period_s") or g.get("tracker_start") != tracker_start():
        print(f"[wrap] no usable fit ({g.get('on_grid', 0)} of {g.get('episodes', 0)} episodes on a grid, tracker up since {tracker_start()}) — not waiting")
        return
    P, t0 = g["period_s"], g["t0"]
    nxt = t0 + np.ceil((now - a.margin_s - t0) / P) * P   # the wrap whose minute is not over yet
    if a.wait is None:
        print(f"[wrap] period {P:.2f} s ({g['on_grid']} of {g['episodes']} episodes on the grid, rms {g.get('rms_s')} s); next wraps: "
              + " ".join(time.strftime("%H:%M:%S", time.localtime(nxt + i * P)) for i in range(8)))
    elif nxt - now < a.wait:
        w = nxt + a.margin_s - now
        print(f"[wrap] {time.strftime('%T')} the stamp counter wraps at {time.strftime('%T', time.localtime(nxt))} — waiting {w:.0f} s", flush=True)
        if a.now is None:
            time.sleep(max(0.0, w))
    else:
        print(f"[wrap] {time.strftime('%T')} next stamp-counter wrap at {time.strftime('%T', time.localtime(nxt))}, {nxt - now:.0f} s away — clear")


if __name__ == "__main__":
    main()
