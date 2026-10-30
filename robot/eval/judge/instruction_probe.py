#!/usr/bin/env python3
"""Judge check (b): does the instruction sentence alone change what the policy answers? (policy endpoint only; ~2.5 min)

Takes the 8 drilling observations the policy really saw in robot run 17 (the 256 px endoscope PNGs and the measured 15-value
tool state of the same frame, from run17_live/) and asks the policy server /act with the SAME image + state and three
drilling sentences:
  contra   "Take down the left L4 lamina and carry on over to the opposite side."   (run 17's sentence, unseen in training)
  ipsi     "Take down the left L4 lamina only, leaving the right side untouched."    (run 19's sentence, unseen in training)
  neutral  "Left L4 laminectomy."                                                      (a training sentence)
Your own sentences: --sentence "…" (repeatable) adds them to the same frames; --examples lists sentences that were NOT used
in training (checked against the dataset's 131 tasks) — robot/eval/judge/sentence_loop.py then drives a simulated closed
loop with one of them and writes the endoscope and 3D videos.
GR00T samples its answer (flow matching), so each question is asked N times. Per answer, over the 40 predicted rows: the
drill tip's x (scene frame, mm; the surgeon works from the left at x > 21.9 mm, so "toward / past the midline" is DECREASING
x) and the predicted drill.progress (0 … 1; the policy ends drilling when it passes 0.85).

What we got (Nebius RTX PRO 6000, 8 frames × 3 sentences × 5; once on the VM, once through the Serverless Endpoint, which
takes ~2.5 min for the 120 calls): with the contralateral sentence the predicted progress was lower than with the
ipsilateral one in all 8 frames both times (by 0.11 ± 0.08 and 0.12 ± 0.08) — the policy "thinks it is less done" and keeps
drilling; the tip position after 40 rows differed by +0.77 ± 1.70 and +0.42 ± 1.85 mm toward the midline, which is within
the run-to-run spread of the answers (sd ≈ 1 mm), i.e. not a measurable difference. The answers are sampled, so your numbers
will differ slightly. One run, one start pose, 8 frames: an illustration, not a study.

    POLICY_URL=https://<policy endpoint> ENDPOINT_TOKEN=<token> python robot/eval/judge/instruction_probe.py [RUN_DIR] [--k 8] [--n 5] [-o out.json]
    (RUN_DIR default: $SPINE_IL_DATA/run17_live, fetched by fetch_robot_runs.py)"""
import argparse
import base64
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA, auth  # noqa: E402
from sentence_loop import EXAMPLES  # noqa: E402

MID = 21.9
TASKS = {"contra": "Take down the left L4 lamina and carry on over to the opposite side.",
         "ipsi": "Take down the left L4 lamina only, leaving the right side untouched.",
         "neutral": "Left L4 laminectomy."}
URL = (os.environ.get("POLICY_URL") or "http://127.0.0.1:8900").rstrip("/")


def act(png: bytes, state, task, seq):
    body = {"protocol": "spine-policy/1", "seq": seq, "run": "instruction_probe", "image_png_b64": base64.b64encode(png).decode(),
            "state": [float(v) for v in state], "task": task}
    r = urllib.request.Request(URL + "/act", data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **auth()})
    return np.asarray(json.loads(urllib.request.urlopen(r, timeout=60).read())["action"], float)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", default=str(DATA / "run17_live"))
    ap.add_argument("--k", type=int, default=8); ap.add_argument("--n", type=int, default=5); ap.add_argument("-o", default=None)
    ap.add_argument("--sentence", action="append", default=[], help="your own sentence (repeatable), asked next to the three")
    ap.add_argument("--examples", action="store_true", help="print example sentences (none used in training) and exit")
    a = ap.parse_args()
    if a.examples:
        for k, ss in EXAMPLES.items():
            print(f"{k}:"); [print(f'  "{s}"') for s in ss]
        print("\nNone of these was a training sentence.")
        return
    tasks = {**TASKS, **{f"yours{i + 1}": s for i, s in enumerate(a.sentence)}}
    R = Path(a.run_dir)
    if not (R / "live.json").exists():
        sys.exit(f"missing {R / 'live.json'} — run robot/eval/judge/fetch_robot_runs.py first")
    d = json.loads((R / "live.json").read_text()); F = np.asarray(d["frames"], float); m = d["meta"]
    # drilling-phase observations the policy (not the script) answered, whose logged PNG is present
    looks = [p[0] for p, c in zip(m["page_observations"], m["chunks"]) if c.get("phase") == 4 and not c.get("scripted")
             and (R / "frames" / f"{p[0]:05d}.png").exists()]
    pick = [looks[int(round(i))] for i in np.linspace(0, len(looks) - 1, min(a.k, len(looks)))]
    rows, seq, t0 = [], 0, time.time()
    for f in pick:
        png = (R / "frames" / f"{f:05d}.png").read_bytes(); s = F[f, :15]; x0 = s[7]
        row = {"frame": int(f), "x_now": round(float(x0), 2), "removed_mm3_then": round(float((m.get("removed_mm3") or [0])[f]), 1)}
        for k, task in tasks.items():
            shift, mean_shift, xlast, prog = [], [], [], []
            for _ in range(a.n):
                seq += 1; A = act(png, s, task, seq); x = A[:, 7]
                shift.append(x0 - x[-1]); mean_shift.append(float(np.mean(x0 - x))); xlast.append(x[-1])
                prog.append(A[-1, 15] if A.shape[1] > 15 else np.nan)
            row[k] = {"medial_shift_mm": [round(float(np.mean(shift)), 2), round(float(np.std(shift)), 2)],
                      "mean_medial_shift_mm": [round(float(np.mean(mean_shift)), 2), round(float(np.std(mean_shift)), 2)],
                      "x_last": round(float(np.mean(xlast)), 2), "past_midline_last": bool(np.mean(xlast) < MID),
                      "progress_last": round(float(np.nanmean(prog)), 3)}
        rows.append(row); print(json.dumps(row), flush=True)
    dc = [r["contra"]["medial_shift_mm"][0] - r["ipsi"]["medial_shift_mm"][0] for r in rows]
    dm = [r["contra"]["mean_medial_shift_mm"][0] - r["ipsi"]["mean_medial_shift_mm"][0] for r in rows]
    dp = [r["contra"]["progress_last"] - r["ipsi"]["progress_last"] for r in rows]
    summary = {"frames": len(rows), "calls": seq, "seconds": round(time.time() - t0, 1),
               "contra_minus_ipsi_medial_shift_mm": {"mean": round(float(np.mean(dc)), 2), "sd": round(float(np.std(dc)), 2),
                                                     "frames_contra_more_medial": int(sum(v > 0 for v in dc))},
               "contra_minus_ipsi_mean_medial_shift_mm": {"mean": round(float(np.mean(dm)), 2), "sd": round(float(np.std(dm)), 2)},
               "contra_minus_ipsi_progress": {"mean": round(float(np.nanmean(dp)), 3), "sd": round(float(np.nanstd(dp)), 3),
                                              "frames_contra_lower": int(sum(v < 0 for v in dp))}}
    summary["per_sentence"] = {k: {"sentence": t, "progress_last_mean": round(float(np.nanmean([r[k]["progress_last"] for r in rows])), 3),
                                   "medial_shift_mm_mean": round(float(np.mean([r[k]["medial_shift_mm"][0] for r in rows])), 2)}
                               for k, t in tasks.items()}
    print(json.dumps(summary))
    if a.o:
        Path(a.o).write_text(json.dumps({"rows": rows, "summary": summary, "tasks": tasks}, indent=1))


if __name__ == "__main__":
    main()
