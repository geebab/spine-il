#!/usr/bin/env python3
"""How each demonstration ends its drilling — against the recording rule of 2026-10-09.

User, 10-09: "끝내는 동작을 같게". The hand recordings still to be made end the drilling with the same gesture, so that the
end is something a policy can see and do (the 24 recordings before the rule release the trigger and pull the drill out at
once: 5 mm away after 0.8–1.3 s, out of the portal 4–5 s later — and the 10-06 ones release the trigger 23 times on the way,
each release looking like the beginning of that). The rule, in my wording of the user's decision:

  1  after the last cut the trigger is released and not pulled again
  2  the drill is drawn back along its shaft by about 1 cm (counted: 5–25 mm from where it cut last)
  3  both tools rest there for about 3 s (counted: 2 s or more with the drill tip and the endoscope tip each within 2 mm)
  4  then the tools are taken out as before
  and while drilling the gesture does not occur: a pause keeps the drill at the bone (counted: no rest of 2 s with the
  trigger off and the drill 5 mm or more away from its last cut before the drilling is over)

    python dataset/end_gesture.py --dataset DS [--episodes 25-40] [-o out.json]

Reads the dataset's own columns (tool state at 15 Hz, cut.progress); whether the drill tip stays on the endoscope's
screen during the rest is not looked at here (the render's per-frame metrics have it). A recording that does not follow
the rule is reported, not dropped: what to do with it is the user's decision.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "inference"))

RETRACT_MM = (5.0, 25.0)
REST_S, REST_MM = 2.0, 2.0


def rests(tip: np.ndarray, other: np.ndarray, fps: float, min_s: float = REST_S, tol_mm: float = REST_MM) -> list[tuple[int, int]]:
    """[a, b) frame ranges of at least min_s in which both tips stay within tol_mm of where the range began."""
    out, a, n = [], 0, len(tip)
    while a < n:
        b = a + 1
        while b < n and np.linalg.norm(tip[b] - tip[a]) <= tol_mm and np.linalg.norm(other[b] - other[a]) <= tol_mm:
            b += 1
        if (b - a) / fps >= min_s:
            out.append((a, b)); a = b
        else:
            a += 1
    return out


def check(progress: np.ndarray, state: np.ndarray, first_cut: int, last_cut: int, fps: float = 15.0) -> dict:
    """One demonstration against the rule. state: (n, 15) endo xyz · quat, drill xyz · quat, trigger."""
    n = len(state); trig = state[:, 14] > 0.5; drill = state[:, 7:10]; endo = state[:, 0:3]
    after = slice(last_cut, n)
    repull = int(np.sum(np.diff(trig[after].astype(int)) == 1))
    off = next((i for i in range(last_cut, n) if not trig[i:].any()), None)
    r = [(a + last_cut, b + last_cut) for a, b in rests(drill[after], endo[after], fps)]
    rest = next(((a, b) for a, b in r if RETRACT_MM[0] <= np.linalg.norm(drill[a] - drill[last_cut]) <= RETRACT_MM[1] and not trig[a:b].any()), None)
    # while drilling: where the drill cut last, frame by frame — and rests away from there with the trigger off
    cut = np.diff(progress, prepend=progress[0]) > 1e-6
    last_at = np.maximum.accumulate(np.where(cut, np.arange(n), 0))
    early = [(a + first_cut, b + first_cut) for a, b in rests(drill[first_cut:last_cut], endo[first_cut:last_cut], fps)]
    early = [(a, b) for a, b in early if not trig[a:b].any() and np.linalg.norm(drill[a] - drill[last_at[a]]) >= RETRACT_MM[0]]
    out = {"after_last_cut_s": round((n - 1 - last_cut) / fps, 2), "trigger_pulled_again": repull,
           "trigger_off_after_s": None if off is None else round((off - last_cut) / fps, 2),
           "drill_away_at_end_mm": round(float(np.linalg.norm(drill[-1] - drill[last_cut])), 1),
           "rest": None if rest is None else {"from_s": round((rest[0] - last_cut) / fps, 2), "for_s": round((rest[1] - rest[0]) / fps, 2),
                                              "drawn_back_mm": round(float(np.linalg.norm(drill[rest[0]] - drill[last_cut])), 1)},
           "end_like_rests_while_drilling": len(early)}
    miss = []
    if repull or off is None:
        miss.append("1 trigger pulled again after the last cut" if repull else "1 trigger never released")
    if rest is None:
        miss.append("2-3 no rest of 2 s with the drill 5–25 mm back")
    if early:
        miss.append(f"while drilling: {len(early)} rest(s) that look like the end")
    out["follows_the_rule"] = not miss; out["misses"] = miss
    return out


def main(argv=None) -> None:
    import end_level as EL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True); ap.add_argument("--episodes", default=None, help="e.g. 25-40 or 3,7,9 (default: all)")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args(argv)
    ds = Path(a.dataset).expanduser(); eps = EL.load(ds)
    fps = float(json.loads((ds / "meta/info.json").read_text())["fps"])
    want = None
    if a.episodes:
        want = set()
        for part in a.episodes.split(","):
            lo, _, hi = part.partition("-")
            want |= set(range(int(lo), int(hi or lo) + 1))
    res = {}
    for e, d in sorted(eps.items()):
        if want is not None and e not in want:
            continue
        if d["progress"][-1] < 0.99:
            res[e] = {"session": d["session"], "skipped": "the episode does not contain the end of its drilling"}
            continue
        res[e] = {"session": d["session"], **check(d["progress"], d["state"], d["first_cut"], d["last_cut"], fps)}
    ok = [e for e, r in res.items() if r.get("follows_the_rule")]
    for e, r in res.items():
        if "skipped" in r:
            print(f"{e:3d} {r['session'][-4:]}  — {r['skipped']}"); continue
        rest = r["rest"]
        print(f"{e:3d} {r['session'][-4:]}  {'OK ' if r['follows_the_rule'] else 'no '} after the last cut {r['after_last_cut_s']:5.1f} s · trigger off after {r['trigger_off_after_s']} s"
              f" · rest {('%.1f s, %.0f mm back' % (rest['for_s'], rest['drawn_back_mm'])) if rest else '—'} · drill {r['drill_away_at_end_mm']:.0f} mm away at the end"
              + ("" if r["follows_the_rule"] else "   ← " + "; ".join(r["misses"])))
    print(f"{len(ok)} of {sum('skipped' not in r for r in res.values())} follow the rule")
    if a.out:
        Path(a.out).write_text(json.dumps({"dataset": str(ds), "rule": {"retract_mm": RETRACT_MM, "rest_s": REST_S, "rest_mm": REST_MM}, "episodes": res}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
