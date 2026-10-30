#!/usr/bin/env python3
"""Recorded robot observations through a policy server — what would THIS model have answered there? (2026-10-12)

    python robot/inference/remote/replay_obs.py RUN_DIR [RUN_DIR …] --policy-url URL --policy-token-file FILE
           [--every 1] [--repeat 1] [--workers 4] [--lead 16] [--period 8] [--json OUT]

A robot run keeps the image of every look (runs/act_smoke/live/<run>/frames/<record frame>.png) and the measured frames
(live.json). This asks a server for each of them again and reads the answer as the loop would:

  progress   the 16th value over the rows a stop-and-go chunk (rows 0–15) and a continuous window (rows L−1 … L+P−2)
             would execute, by the removed volume of that moment, and where a level would have ended the drilling
             (rollout.done_step: 5 rows in a row above it, from a row the chunk / window executes);
  spread     the same observation asked --repeat times: the model samples its answer. Runs 1 and 2 of 10-12 were ended by
             ONE answer above the level (0.94 after 0.38, 0.92 after 0.60) — is that the model's judgement or its noise?
             With --repeat R every pass through the run is one draw of "where the drilling would have ended";
  motion     the tips' travel over 16 rows, and the largest turn and tip step between two rows (the loop limits a frame
             to 3° and 2 mm; a frame it shortens counts against the run: eval/run_criteria.py, criterion 6).

For: the checkpoints of one training compared on what the robot actually met (train/nebius/ckpt_sweep.sh), and a new
model against an old run. The server only — no robot, no arms. The state sent is the record's measured frame at the look
(a continuous run fed the ±0.3 s median around the image's time: the same pose within the arms' wander), the trigger as
commanded then, the task of the harness's phase then."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import remote_policy as RP   # noqa: E402

HOLD = 5                                    # rollout.DONE_HOLD
LEVELS = (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.91)
BINS = (0, 200, 400, 600, 800, 1000, 1200, 1400, 1700, 2200)


def done_step(p: np.ndarray, n_exec: int, level: float) -> int | None:
    """rollout.done_step: the first of the n_exec rows from which the value stays above `level` for HOLD rows."""
    d = np.asarray(p, float).reshape(-1) > level
    for j in range(min(int(n_exec), len(d))):
        w = d[j:j + HOLD]
        if len(w) and bool(w.all()):
            return j
    return None


def turn_deg(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    d = np.abs((q0 * q1).sum(-1) / (np.linalg.norm(q0, axis=-1) * np.linalg.norm(q1, axis=-1)))
    return np.degrees(2 * np.arccos(np.clip(d, 0, 1)))


def observations(run: Path, every: int) -> tuple[list, dict]:
    """[(record frame, image path, state 15, task phase, removed mm3)] of the run's policy part, and its meta."""
    j = json.loads((run / "live.json").read_text()); m = j["meta"]
    meas, cmd = np.asarray(j["frames"], float), np.asarray(j["frames_cmd"], float)
    kinds = m["frame_kind"]; pol = [i for i, k in enumerate(kinds) if k == "policy"]
    if not pol:
        return [], m
    fed = {int(f): int(p) for f, p in zip(m.get("fed_frames") or [], m.get("phases_seen") or [])}
    rm = m.get("removed_mm3") or []
    out = []
    for f in sorted((run / "frames").glob("*.png")):
        i = int(f.stem)
        if not pol[0] - 1 <= i <= pol[-1]:        # the entry's last look is the policy's first observation
            continue
        s = meas[i][:15].copy(); s[14] = float(cmd[i][14] > 0.5)
        out.append((i, f, s, fed.get(i, 4), float(rm[i]) if i < len(rm) else float("nan")))
    return out[::max(1, every)], m


def ask(pol: RP.RemotePolicy, done_out: bool, ob, lead: int, period: int) -> dict:
    from PIL import Image
    i, f, s, ph, vol = ob
    img = np.asarray(Image.open(f).convert("RGB"), np.uint8)
    rows = np.asarray(pol.act(img, s, pol.task(ph)), float)
    p = rows[:, 15] if rows.shape[1] > 15 else np.full(len(rows), np.nan)
    a, b = lead - 1, min(lead - 1 + period, len(rows))
    r = {"frame": i, "mm3": vol, "p_chunk": float(np.nanmax(p[:16])), "p_window": float(np.nanmax(p[a:b])),
         "end_chunk": {str(lv): done_step(p, 16, lv) is not None for lv in LEVELS},
         "end_window": {str(lv): done_step(p[a:], b - a, lv) is not None for lv in LEVELS}}
    for tool, o in (("endo", 0), ("drill", 7)):
        tip = np.vstack([s[o:o + 3], rows[:b, o:o + 3]]); q = np.vstack([s[o + 3:o + 7], rows[:b, o + 3:o + 7]])
        r[tool] = {"travel16_mm": float(np.linalg.norm(rows[min(15, len(rows) - 1), o:o + 3] - s[o:o + 3])),
                   "tip_step_max_mm": float(np.linalg.norm(np.diff(tip, axis=0), axis=1)[1:].max()),      # between rows (not state → row 0)
                   "turn_step_max_deg": float(turn_deg(q[1:-1], q[2:]).max()), "first_turn_deg": float(turn_deg(q[0], q[1]))}
    return r


def summarize(rs: list, key: str) -> dict:
    """rs: the answers of one pass, in the run's order. key: "chunk" or "window"."""
    pk = "p_" + key; ek = "end_" + key
    p = np.array([r[pk] for r in rs]); vol = np.array([r["mm3"] for r in rs])
    by = {}
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sel = (vol >= lo) & (vol < hi)
        if sel.any():
            by[f"{lo}-{hi}"] = [round(float(np.median(p[sel])), 2), round(float(p[sel].max()), 2), int(sel.sum())]
    ends = {}
    for lv in LEVELS:
        k = next((k for k, r in enumerate(rs) if r[ek][str(lv)]), None)
        ends[str(lv)] = None if k is None else round(float(vol[k]), 0)
    return {"progress_by_mm3": by, "ends_at_mm3": ends, "above": {str(lv): int((p > lv).sum()) for lv in LEVELS}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+"); ap.add_argument("--policy-url", required=True); ap.add_argument("--policy-token-file", default=None)
    ap.add_argument("--policy-ca-file", default=None)
    ap.add_argument("--every", type=int, default=1, help="every n-th look of a run")
    ap.add_argument("--repeat", type=int, default=1, help="ask every observation this many times (the model samples its answer)")
    ap.add_argument("--workers", type=int, default=4, help="requests on their way at once (the server answers one at a time)")
    ap.add_argument("--lead", type=int, default=16); ap.add_argument("--period", type=int, default=8)
    ap.add_argument("--image", default="png", help="png | jpeg[:quality] — how the observation is sent (to compare a model's answers from both)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    tok = Path(a.policy_token_file).expanduser().read_text().strip() if a.policy_token_file else None
    pol = RP.RemotePolicy(a.policy_url, tok, timeout_s=20.0, retries=1, run="replay_obs", ca_file=a.policy_ca_file, image=a.image)
    info = pol.info(); done_out = bool(info.get("done_output"))
    print(f"server: {info.get('model')}, {info.get('action_horizon')} rows, end output {info.get('end_output')}")
    out = {"model": info.get("model"), "lead": a.lead, "period": a.period, "repeat": a.repeat, "image": a.image, "runs": {}}
    for r in a.runs:
        run = Path(r).expanduser(); obs, m = observations(run, a.every)
        if not obs:
            print(f"{run.name}: no policy observations with an image — skipped"); continue
        key = "window" if m.get("continuous") else "chunk"
        passes = []
        with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
            for _ in range(a.repeat):
                passes.append(list(ex.map(lambda ob: ask(pol, done_out, ob, a.lead, a.period), obs)))
        S = [summarize(ps, key) for ps in passes]
        P = np.array([[x["p_" + key] for x in ps] for ps in passes])                # (repeat, observations)
        rec = {"observations": len(obs), "read_as": key, "ended_by": m.get("drill_end"), "passes": S,
               "turn_step_max_deg": {t: [round(float(np.percentile([x[t]["turn_step_max_deg"] for x in passes[0]], q)), 2) for q in (50, 90, 100)] for t in ("endo", "drill")},
               "answers_with_a_turn_over_3deg": {t: int(sum(x[t]["turn_step_max_deg"] > 3.0 for x in passes[0])) for t in ("endo", "drill")},
               "tip_step_max_mm": {t: [round(float(np.percentile([x[t]["tip_step_max_mm"] for x in passes[0]], q)), 2) for q in (50, 90, 100)] for t in ("endo", "drill")},
               "travel16_mm_median": {t: round(float(np.median([x[t]["travel16_mm"] for x in passes[0]])), 2) for t in ("endo", "drill")}}
        if a.repeat > 1:
            sd = P.std(axis=0)
            rec["spread_of_one_observation"] = {"sd_median": round(float(np.median(sd)), 3), "sd_p90": round(float(np.percentile(sd, 90)), 3),
                                                "range_max": round(float((P.max(0) - P.min(0)).max()), 3)}
        # against what the run's own model said then (done_log: [record frames so far, the executed rows' values, end row])
        dl = m.get("done_log") or []
        if dl:
            then = {int(e[0]) - 1: max(e[1]) for e in dl if e[1]}
            pairs = [(then[x["frame"]], x["p_" + key]) for x in passes[0] if x["frame"] in then]
            if len(pairs) > 5:
                t0, t1 = np.array(pairs).T
                rec["against_the_run"] = {"n": len(pairs), "then_median": round(float(np.median(t0)), 2), "now_median": round(float(np.median(t1)), 2),
                                          "difference_sd": round(float(np.std(t1 - t0)), 3), "correlation": round(float(np.corrcoef(t0, t1)[0, 1]), 2)}
        out["runs"][run.name] = rec
        print(f"\n{run.name}: {len(obs)} observations × {a.repeat}, read as a {key}; the run was ended by {m.get('drill_end')}")
        print("   progress by removed mm3 (median, max, n): " + "  ".join(f"{k}: {v[0]}/{v[1]} ({v[2]})" for k, v in S[0]["progress_by_mm3"].items()))
        print("   a level would end the drilling at (mm3), per pass: " + "  ".join(
            f"{lv}: " + "/".join("–" if s["ends_at_mm3"][str(lv)] is None else f"{s['ends_at_mm3'][str(lv)]:.0f}" for s in S) for lv in LEVELS))
        if "spread_of_one_observation" in rec:
            print(f"   the same observation asked {a.repeat} times: sd {rec['spread_of_one_observation']['sd_median']} (median), {rec['spread_of_one_observation']['sd_p90']} (90th), "
                  f"largest range {rec['spread_of_one_observation']['range_max']}")
        if "against_the_run" in rec:
            g = rec["against_the_run"]
            print(f"   against the run's own answers ({g['n']}): then {g['then_median']}, now {g['now_median']} (medians), difference sd {g['difference_sd']}, correlation {g['correlation']}")
        print(f"   largest turn between two rows (median/90th/max): endo {rec['turn_step_max_deg']['endo']}°, drill {rec['turn_step_max_deg']['drill']}°; "
              f"answers with one over 3°: {rec['answers_with_a_turn_over_3deg']}; tip travel over 16 rows (median): {rec['travel16_mm_median']} mm")
    if pol.log:
        out["round_trip_s_median"] = round(float(np.median([x["round_trip_s"] for x in pol.log])), 3)
        print(f"\nround trip {out['round_trip_s_median']} s (median of {len(pol.log)}, {a.workers} at once, image {a.image})")
    if a.json:
        Path(a.json).expanduser().write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
