#!/usr/bin/env python3
"""One table of a checkpoint sweep (ckpt_sweep.sh): python ckpt_sweep.py SWEEP_DIR [--json OUT]

Per step, from <step>_heldout.json (inference/end_level.py on the held-out demonstrations) and <step>_replay.json
(inference/remote/replay_obs.py on robot runs): how the model moves and reads the progress where it was never trained, and
what it says on the observations the robot actually met — the two can disagree (10-12: the level computed on held-out
demonstrations, 0.91, was reached by one answer in 67, one in 253 and none in 399 on the robot)."""
import argparse
import json
import re
from pathlib import Path

import numpy as np


def heldout(f: Path) -> dict:
    j = json.loads(f.read_text()); o = (j.get("open_loop") or {}).get("drilling") or {}
    d, e = o.get("drill_H16") or {}, o.get("endo_H16") or {}
    hi = (j.get("open_loop") or {}).get("progress_error_where_true_progress_is_0.9_or_more") or {}
    p = next(iter((j.get("predictors") or {}).values()), {})
    return {"drill_direction": d.get("cos_median"), "drill_gain": d.get("gain_slope"), "drill_mm": [d.get("pred_mm_median"), d.get("demo_mm_median")],
            "endo_direction": e.get("cos_median"), "trigger_agree": o.get("trigger_agree_16"),
            "progress_mae": (o.get("progress_error") or {}).get("mean_abs"), "progress_at_end": [hi.get("mean"), hi.get("sd")],
            "best_level": (p.get("best") or {}).get("level"), "level_range": [(p.get("indistinguishable") or {}).get("lowest"), (p.get("indistinguishable") or {}).get("highest")]}


def replay(f: Path) -> dict:
    j = json.loads(f.read_text()); out = {}
    for name, r in (j.get("runs") or {}).items():
        late = [v for k, v in r["passes"][0]["progress_by_mm3"].items() if int(k.split("-")[0]) >= 1000]
        mid = [v for k, v in r["passes"][0]["progress_by_mm3"].items() if 600 <= int(k.split("-")[0]) < 1000]
        ends = {lv: [p["ends_at_mm3"].get(lv) for p in r["passes"]] for lv in ("0.7", "0.8", "0.91")}
        out[name] = {"progress_600_1000": round(float(np.mean([v[0] for v in mid])), 2) if mid else None,
                     "progress_over_1000": round(float(np.mean([v[0] for v in late])), 2) if late else None,
                     "max": max(v[1] for v in r["passes"][0]["progress_by_mm3"].values()), "ends": ends,
                     "spread_sd": (r.get("spread_of_one_observation") or {}).get("sd_median"),
                     "turn_over_3deg": r.get("answers_with_a_turn_over_3deg"), "observations": r.get("observations"),
                     "travel16_mm": r.get("travel16_mm_median")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("dir"); ap.add_argument("--json", default=None)
    a = ap.parse_args(); D = Path(a.dir).expanduser()
    steps = sorted({re.match(r"(\d+)_", f.name).group(1) for f in D.glob("*_*.json") if re.match(r"\d+_", f.name)})
    T = {}
    for s in steps:
        h, r = D / f"{s}_heldout.json", D / f"{s}_replay.json"
        T[s] = {"heldout": heldout(h) if h.exists() else None, "replay": replay(r) if r.exists() else None}
    f2 = lambda x: "–" if x is None else f"{x:.2f}"
    print("held-out demonstrations (the two the training never saw)")
    print("  step   | drill direction · gain · travel model/demo mm | endoscope direction | trigger agree | progress MAE | progress where true ≥ 0.9 | best level (range)")
    for s, t in T.items():
        h = t["heldout"]
        if not h:
            print(f"  {s} | no held-out result"); continue
        print(f"  {s} | {f2(h['drill_direction'])} · {f2(h['drill_gain'])} · {f2(h['drill_mm'][0])}/{f2(h['drill_mm'][1])} | {f2(h['endo_direction'])} | {f2(h['trigger_agree'])} | "
              f"{f2(h['progress_mae'])} | {f2(h['progress_at_end'][0])} ± {f2(h['progress_at_end'][1])} | {f2(h['best_level'])} ({f2(h['level_range'][0])}–{f2(h['level_range'][1])})")
    runs = sorted({n for t in T.values() if t["replay"] for n in t["replay"]})
    for n in runs:
        print(f"\nrobot observations of {n}, asked again")
        print("  step   | progress at 600–1,000 mm3 · over 1,000 mm3 · highest | a level ends the drilling at mm3, per pass: 0.7 · 0.8 · 0.91 | one observation's sd | answers with a turn over 3° (endo/drill) | drill travel over 16 rows")
        for s, t in T.items():
            r = (t["replay"] or {}).get(n)
            if not r:
                print(f"  {s} | –"); continue
            e = lambda lv: "/".join("–" if x is None else f"{x:.0f}" for x in r["ends"][lv])
            tv = r.get("turn_over_3deg") or {}
            print(f"  {s} | {f2(r['progress_600_1000'])} · {f2(r['progress_over_1000'])} · {f2(r['max'])} | {e('0.7')} · {e('0.8')} · {e('0.91')} | {f2(r['spread_sd'])} | "
                  f"{tv.get('endo')}/{tv.get('drill')} of {r.get('observations')} | {f2((r.get('travel16_mm') or {}).get('drill'))} mm")
    if a.json:
        Path(a.json).expanduser().write_text(json.dumps(T, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
