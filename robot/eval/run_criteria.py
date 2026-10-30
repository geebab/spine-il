"""Pass / fail criteria for closed-loop robot runs, set from the demonstrations themselves (2026-10-06, user: "새로운 로봇
평가 기준 네가 세워" — the drilling is no longer cut off by a time limit, and the demo set grows from 8 to 24).

    python run_criteria.py [--dataset DS | --demos 8 | --demos ID,ID,… | --date YYYY-MM-DD] [--beyond-date YYYY-MM-DD]
                           [--make] [LIVE_RUN_DIR …] [--json OUT]

The demo set is the hand sessions of --dataset (its meta/spine_il/episodes.jsonl) of --level (default L4) — the
demonstrations a model was trained on are the ones its runs are measured against. --demos 8 = the 09-28 set.
User, 10-15 ("09-28을 빼자"): the default is the 90-recording L3–L5 dataset without 09-28 (another tracker calibration), so the
reference is its 30 L4 recordings (10-06 16 · 10-12 6 · 10-14 8); until then it was the 24 of spine_hand24_base_f15 (09-28 8 ·
10-06 16). Without the 09-28 recordings the weakest tenth moves up: coverage ≥ 0.71 (was 0.297), demo volumes 1,055–2,161 mm³.
--beyond-date 2026-09-28 has no meaning against this reference (no recording of that day in it).

Reference = the removed-voxel sets of the hand demos (each replayed by the render API with its own cut rows, cached as
iou_cache_hfull_<session>.npz; --make renders the missing ones, about 4 min each, renderer needed):
    core    voxels removed by at least half of the demos — what the task removes
    union   voxels removed by any demo, grown by 2 voxels (1.6 mm) — outside it no demonstration ever cut
Every demo is measured against the reference built from the OTHER demos; that is the human spread of each measure, and
a run passes a measure when it is no worse than the weakest tenth of the demos:

  1 complete    the policy ended the drilling itself and nothing aborted. User, 10-07 23:20: "정책이 끝을 직접 내도록 …
                '드릴링 끝' 표시를 정책의 출력으로 학습" — a policy trained with train/lerobot-train-rel.py --done-output
                has a 14th action value that says the drilling is over; the harness follows it (live.json
                meta.done_output, drill_end = policy). For such a policy every rule of the harness is only a bound and
                a run a bound ended fails: pause (15 s since the last cut and the trigger off at that moment), stall
                (under 30 mm³ in 25 s, trigger on or not), time (the drilling bound, 240 s — above the longest demo,
                218 s), steps, volume, deadline, an abort or the run guard.
                A policy WITHOUT that output (every model up to 10-07) cannot end anything: its runs keep the reading
                they were reported with — drill_end = pause counts as complete — and are marked `legacy end`. That
                reading is a harness timer like stall (10-07 22:50: one run was ended by stall 2.5 s before pause
                would have fired), so those verdicts say less than they seem to about how the policy ended.
                (Until 10-07 noon stall also counted as the policy's own end.)
  2 coverage    share of the core the run removed                       ≥ the demos' 10th percentile
  3 precision   share of the run's removal that lies inside the core    ≥ the demos' 10th percentile
  4 off-target  removed volume outside the grown union                  ≤ the demos' 90th percentile (at least 5 mm³)
  5 visible     drill tip on the endoscope's screen while cutting       ≥ 0.90 — on the screen = in the field of view and
                inside the image circle, hidden behind bone or not (view.on_screen_while_cutting; user, 10-07: "드릴이
                없으면" = the tip has left the screen, nothing else). Until 10-07 the stricter "and not hidden" share
                was used (the demos: 0.83–1.00); a view file without the screen share falls back to it and says so.
  6 unassisted  the harness did not change a motion the policy asked for: no step refused (gate held), shortened (gate
                shrunk), stopped at the travel limit or clamped by the per-frame motion limit. User, 10-07: such a run is a
                failure — "하네스라는게 결국 정책 중간 중간에 알고리즘 처럼 필터링을 두는거" — as in the literature on safety
                filters, where what the filter repaired is the filter's credit and its activations are counted.
     volume      removed volume against the demos' range — reported, not a pass condition (2 and 3 bound it from both sides)
A run passes when 1–6 hold. A model is judged on its runs from the five start poses: how many pass, the median of each
measure, and how well the runs agree with each other (run × run IoU) next to the demos' own agreement (demo × demo IoU).
Runs made under the old 40 s drilling limit fail "complete" by construction; their other measures are still comparable.

The 24-demo set holds two extents of the procedure: the 8 recordings of 09-28 remove 250–1,158 mm³ on the approach side,
the 16 of 10-06 remove 1,056–2,161 mm³ and go on to the far side of the lamina. The user, 10-07, on the video of a passing
run of 874 mm³ ("드릴된 양이 적고 반대편 라미나는 거의 절삭이 없어서 아쉬움") and then: "반대편은 하면 좋고 안해도 합격" —
the far side is welcome, not required. So the verdict stays with the whole demo set (both extents are the task), and the
far side is reported beside it:
  --beyond-date 2026-09-28  report `beyond`: of the core of the recordings NOT of that day (the 16 of 10-06), the part no
                            recording of that day touched (the far side: 387 mm³) — how much of it a run / each of those
                            demos removed. Reported, not a pass condition.
  --date 2026-10-06         judge against the recordings of one day only (their spread sets the thresholds: coverage ≥ 0.80
                            for 10-06) — the stricter reading; not the verdict the user asked for

What the harness does in every run is not held against the policy, but it bounds what a pass means: the entry, the exit
after the drilling and the return are scripted, the phase the policy is told (3 · 4 · 5) is the harness's, and so is the
call that the drilling is over — the policy's part is from about 7 s before the first cut to the end of the drilling.
What the harness changed inside that part is counted per run and printed next to the verdict (`harness` in the json): the
safety gate's refused steps (held), shortened steps (shrunk) and cut turns, frames with the trigger forced off (drill out
of the endoscope's view), frames stopped at the travel limit, frames clamped by the per-frame motion limit. Held, shrunk,
travel-stopped and clamped frames are criterion 6 (until 10-07 21:40 a pass with them was only marked PASS*).
Counted and shown, but not a failure:
  - the trigger switched off while the drill tip is off the endoscope's screen — a rule of the run (user, 10-07 13:45:
    "화면에 드릴이 없으면 방아쇠 트리거 안하는 것은 런은 유지하되 트리거 차단하되 실패는 아님");
  - cut turns (gate turn_wait): the arms make about a tenth of a commanded tool turn whatever is sent, so every run has
    them (9–50) — a limit of this robot, stated with each result. Whether they should fail a run is the user's to say.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import iou_eval as E
import metrics_ext as X
from policy_eval import REPLAY

DATASET = Path.home() / "il-robot/datasets/spine_hand90_replay_entry_water_progress_lang2"   # 10-15: 09-28 out (user); was spine_hand24_base_f15
DEMOS8 = ["3868", "91cb", "bef5", "d340", "bad1", "0338", "919d", "6639"]
PCT, VISIBLE_MIN, OFF_FLOOR_MM3 = 10, 0.90, 5.0


def demo_files(ds: Path, level: str | None = None) -> dict:
    """{short id: recording path} of the dataset's hand sessions (of one level: a multi-level dataset's goal.level; recordings
    without a goal are L4; export_lerobot.py --decision-copies episodes are not recordings)."""
    out = {}
    for line in (ds / "meta/spine_il/episodes.jsonl").read_text().splitlines():
        m = json.loads(line) if line.strip() else {}
        if m.get("source") == "hand" and not m.get("decision_copy") and (level in (None, "any") or (m.get("goal") or {}).get("level", "L4") == level):
            out[m["session_id"].rsplit("_", 1)[-1]] = m["recording"]
    return out


def demo_cut(path: str, make: bool) -> dict | None:
    key = "hfull_" + Path(path).stem
    if not make and not (E.OUT / f"iou_cache_{key}.npz").exists():
        return None
    return E.cached(key, lambda: E.voxels(E.render(path, dict(REPLAY))))


def reference(cuts: list, dims) -> tuple[np.ndarray, np.ndarray]:
    """(core = removed by at least half, union grown by 2 voxels) of a list of voxel index arrays."""
    u, n = np.unique(np.concatenate(cuts).astype(np.int64), return_counts=True)
    return u[n >= int(np.ceil(len(cuts) / 2))], X.dilate(X.dilate(u, dims), dims)


def measure(c: np.ndarray, core: np.ndarray, union: np.ndarray, v: float) -> dict:
    c = c.astype(np.int64)
    return {"mm3": round(len(c) * v, 1), "coverage": round(float(np.isin(core, c).mean()), 3) if len(core) else None,
            "precision": round(float(np.isin(c, core).mean()), 3) if len(c) else None,
            "off_target_mm3": round(float((~np.isin(c, union)).sum() * v), 1)}


POLICY_ENDS = ("policy",)   # the policy's own end of the drilling: a --done-output policy's "the drilling is over"
LEGACY_ENDS = ("pause",)    # a policy without that output: the pause timer, read as its end in the runs up to 10-07


def interventions(meta: dict) -> dict:
    """What the harness changed in the policy's part of a run (live.json meta): the safety gate's counts, the frames with
    the trigger forced off, stopped at the travel limit or clamped by the motion limit. `blocked` = the frames where a
    motion the policy asked for was replaced by a stop (gate held + travel stop) — a pass with any of them is a helped
    pass. The trigger forced off while the drill is not on screen is a rule of the run (user, 10-07): counted, not in
    `blocked`. The gate's counts are those of the POLICY's frames when the run's gate log allows telling them apart
    (`gate_on_scripted_frames` = what it did on the scripted entry, exit and return; None = not known, all counted)."""
    g = meta.get("safety_gate") or {}
    fk = meta.get("frame_kind") or []
    held, shrunk, turn = int(g.get("held") or 0), int(g.get("shrunk") or 0), int(g.get("turn_cut") or 0)
    scripted = None
    ev = g.get("events") or []
    if fk and ev and len(ev) == held + shrunk + turn and all(0 <= int(e[0]) < len(fk) for e in ev):
        # the gate's log is complete (it is cut at 200 events): what it refused on the SCRIPTED entry / exit / return is not
        # the policy's — 10-08, first run on a sitting base: the scripted return to a ready pose at the edge of the gate's
        # 80 mm sphere was shortened twice and refused once, after a policy part without a hold
        kind = lambda e: "turn" if len(e) > 3 and str(e[3]).startswith("turn") else "held" if float(e[1]) == 0.0 else "shrunk"
        scr = [kind(e) for e in ev if fk[int(e[0])] != "policy"]
        scripted = {"held": scr.count("held"), "shrunk": scr.count("shrunk"), "turn_cut": scr.count("turn")}
        held, shrunk, turn = held - scripted["held"], shrunk - scripted["shrunk"], turn - scripted["turn_cut"]
    out = {"policy_frames": int(sum(k == "policy" for k in fk)) or None, "gate_checked": g.get("checked"),
           "gate_held": held, "gate_shrunk": shrunk, "gate_turn_cut": turn, "gate_on_scripted_frames": scripted,
           "trigger_off_frames": int(meta.get("trigger_gated_frames") or 0), "travel_stopped_frames": int(meta.get("travel_limited_frames") or 0),
           "clamped_frames": int(meta.get("clamped_frames") or 0)}
    out["blocked"] = out["gate_held"] + out["travel_stopped_frames"]
    out["overrides"] = out["gate_held"] + out["gate_shrunk"] + out["travel_stopped_frames"] + out["clamped_frames"]   # criterion 6
    return out


def ended_by_policy(drill_end, abort, done_output: bool = False) -> bool:
    """Criterion 1. done_output = the run's policy has the "drilling is over" value (live.json meta.done_output): then only
    its own end counts. Without it the run keeps the legacy reading (pause)."""
    return drill_end in (POLICY_ENDS if done_output else LEGACY_ENDS) and not abort


def run_facts(run: Path) -> dict:
    """What the run's own files say: how the drilling ended, aborts, the drill's visibility (policy_eval.py's replay)."""
    out = {"drill_end": None, "abort": None, "drill_s": None, "visible": None, "visible_kind": None, "seen": None, "harness": None,
           "done_output": False}
    f = run / "live.json"
    if f.exists():
        m = json.loads(f.read_text())["meta"]
        fk, ph = np.array(m.get("frame_kind") or []), np.array(m.get("phases_seen") or [])
        out["abort"] = m.get("abort")
        out["done_output"] = bool(m.get("done_output"))
        out["harness"] = interventions(m)
        out["drill_s"] = m.get("drill_s") or (round(float(((ph == 4) & (fk == "policy")).sum()) / 15.0, 1) if len(fk) == len(ph) and len(ph) else None)
        out["drill_end"] = m.get("drill_end") or ("deadline" if m.get("deadline_hit") else "time (old 40 s limit)")
    vw = None
    rs = run / "robot_session"   # the session's own view file (view_recount.py keeps it current), else the evaluation's copy
    vf = E.OUT / f"view_rofull_{Path(rs.read_text().strip()).stem}.json" if rs.exists() else None
    ev = E.OUT / f"policy_eval_live_{run.name}.json"
    if vf is not None and vf.exists():
        vw = json.loads(vf.read_text())
    elif ev.exists():
        vw = next((c.get("view") for k, c in json.loads(ev.read_text()).get("cuts", {}).items() if k.startswith("robot")), None)
    if vw:
        out["visible"] = vw.get("on_screen_while_cutting", vw.get("visible_while_cutting"))
        out["visible_kind"] = "on screen" if "on_screen_while_cutting" in vw else "seen (old)"
        out["seen"] = vw.get("visible_while_cutting")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--dataset", default=str(DATASET), help="the demo set = this dataset's hand sessions")
    ap.add_argument("--demos", default="all", help="all = every session of --dataset (of --level) · 8 = the 09-28 set · a comma list of ids")
    ap.add_argument("--level", default="L4", help="the reference is the recordings of this level (the robot runs are L4); any = all")
    ap.add_argument("--date", default=None, help="only the recordings of this day (the date folder of their path) are the reference")
    ap.add_argument("--beyond-date", default=None, help="report the part of the reference's core no recording of that day touched")
    ap.add_argument("--make", action="store_true", help="render the missing demo cuts (render API)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    files = demo_files(Path(a.dataset).expanduser(), a.level)
    ids = DEMOS8 if a.demos == "8" else list(files) if a.demos == "all" else a.demos.split(",")
    day = lambda i: Path(files[i]).parent.name if i in files else None
    other = [i for i in files if a.beyond_date and day(i) == a.beyond_date]
    if a.date:
        ids = [i for i in ids if day(i) == a.date]
        if not ids:
            raise SystemExit(f"no recording of {a.date} in the demo set")
    dem, missing, oth = {}, [], {}
    for i in ids + [j for j in other if j not in ids]:
        c = demo_cut(files[i], a.make) if i in files else None
        if c is None:
            missing.append(i)
            continue
        if i in ids:
            dem[i] = c
        if i in other:
            oth[i] = c
    if missing:
        print(f"no cached cut for {len(missing)} demo(s): {' '.join(missing)} (--make renders them)")
    hdr = next(iter(dem.values()))["hdr"]; v, dims = hdr.get("voxelMm3", 0.512), hdr["dims"]
    fin = {i: d["final"].astype(np.int64) for i, d in dem.items()}
    core, union = reference(list(fin.values()), dims)
    # --beyond-date: of the core of the recordings NOT of that day, the part no recording of that day touched (the far side)
    share = lambda c_, region: round(float(np.isin(region, c_).mean()), 3) if len(region) else None
    far_ids = [i for i in fin if i not in oth]
    u_oth = np.unique(np.concatenate([d["final"].astype(np.int64) for d in oth.values()])) if oth else None
    def beyond_region(keys):
        return np.setdiff1d(reference([fin[k] for k in keys], dims)[0], u_oth) if (u_oth is not None and keys) else None
    loo = {}
    for i in fin:   # each demo against the reference of the others: the human spread
        c, u = reference([x for j, x in fin.items() if j != i], dims)
        loo[i] = measure(fin[i], c, u, v)
        if u_oth is not None and i in far_ids and len(far_ids) > 1:   # a far-side demo against the far side of the others
            loo[i]["beyond"] = share(fin[i], beyond_region([k for k in far_ids if k != i]))
    P = lambda k, q: float(np.percentile([m[k] for m in loo.values()], q))
    thr = {"coverage_min": round(P("coverage", PCT), 3), "precision_min": round(P("precision", PCT), 3),
           "off_target_max_mm3": round(max(OFF_FLOOR_MM3, P("off_target_mm3", 100 - PCT)), 1), "visible_min": VISIBLE_MIN,
           "volume_range_mm3": [min(m["mm3"] for m in loo.values()), max(m["mm3"] for m in loo.values())]}
    dxd = [E.iou(fin[x], fin[y])[0] for x, y in itertools.combinations(fin, 2)]
    beyond = beyond_region(far_ids)
    if beyond is not None:
        bd = [m["beyond"] for m in loo.values() if m.get("beyond") is not None]
        thr["beyond_demos"] = [min(bd), round(float(np.percentile(bd, PCT)), 3), max(bd)] if bd else None
    out = {"demos": list(fin), "missing": missing, "date": a.date, "beyond_date": a.beyond_date,
           "beyond_mm3": round(len(beyond) * v, 1) if beyond is not None else None,
           "core_mm3": round(len(core) * v, 1), "union_grown_mm3": round(len(union) * v, 1),
           "core_rule": f"removed by ≥ {int(np.ceil(len(fin) / 2))} of {len(fin)}", "thresholds": thr, "demo_loo": loo,
           "demo_x_demo_iou": {"median": round(float(np.median(dxd)), 3), "min": round(min(dxd), 3), "max": round(max(dxd), 3)}}
    print(f"reference: {len(fin)} demos, core {out['core_mm3']} mm³ ({out['core_rule']}), grown union {out['union_grown_mm3']} mm³")
    print("demos against the others (id: mm³ · coverage · precision · off-target mm³): "
          + "  ".join(f"{i} {m['mm3']:.0f}·{m['coverage']:.2f}·{m['precision']:.2f}·{m['off_target_mm3']:.0f}" for i, m in loo.items()))
    if beyond is not None and thr.get("beyond_demos"):
        print(f"beyond — optional, not a pass condition (core of the {len(far_ids)} recordings not of {a.beyond_date} that no recording of that day "
              f"touched): {out['beyond_mm3']} mm³; those demos removed {thr['beyond_demos'][0]:.2f}–{thr['beyond_demos'][2]:.2f} of it "
              f"(weakest tenth {thr['beyond_demos'][1]:.2f})")
    print(f"thresholds (weakest tenth of the demos): coverage ≥ {thr['coverage_min']}, precision ≥ {thr['precision_min']}, off-target ≤ "
          f"{thr['off_target_max_mm3']} mm³, visible ≥ {VISIBLE_MIN}; demo volumes {thr['volume_range_mm3'][0]:.0f}–{thr['volume_range_mm3'][1]:.0f} mm³; "
          f"demo × demo IoU median {out['demo_x_demo_iou']['median']}")
    runs, cuts = {}, {}
    for r in a.runs:
        r = Path(r).expanduser(); f = r / "robot_session"
        key = "rofull_" + Path(f.read_text().strip()).stem if f.exists() else ""
        if not key or not (E.OUT / f"iou_cache_{key}.npz").exists():
            print(f"{r.name}: no cached robot cut — skipped")
            continue
        c = E.cached(key, None)["final"].astype(np.int64); cuts[r.name] = c
        m = {**measure(c, core, union, v), **run_facts(r)}
        if beyond is not None:
            m["beyond"] = share(c, beyond)
        ok = {"complete": ended_by_policy(m["drill_end"], m["abort"], m["done_output"]),
              "coverage": m["coverage"] is not None and m["coverage"] >= thr["coverage_min"],
              "precision": m["precision"] is not None and m["precision"] >= thr["precision_min"],
              "off_target": m["off_target_mm3"] <= thr["off_target_max_mm3"],
              "visible": m["visible"] is not None and m["visible"] >= VISIBLE_MIN,
              "unassisted": (m.get("harness") or {}).get("overrides", 0) == 0}
        m["ok"], m["pass"] = ok, all(ok.values())
        m["in_volume_range"] = thr["volume_range_mm3"][0] <= m["mm3"] <= thr["volume_range_mm3"][1]
        runs[r.name] = m
        h = m.get("harness") or {}
        m["helped"] = bool(h.get("blocked"))
        who = "the policy" if m["drill_end"] in POLICY_ENDS else "the harness's pause timer — legacy end, a policy without the done output" \
            if m["drill_end"] in LEGACY_ENDS and not m["done_output"] else "the harness"
        print(f"  {r.name:22s} {'PASS' if m['pass'] else 'fail'} | " + " ".join(f"{k}{'✓' if x else '✗'}" for k, x in ok.items())
              + f" | {m['mm3']:.0f} mm³ · cover {m['coverage']:.2f} · prec {m['precision']:.2f} · off {m['off_target_mm3']:.0f} mm³ · {m.get('visible_kind') or 'visible'} {m['visible']} · "
                f"drilling {m['drill_s']} s, ended by {who} ({m['drill_end']}{', abort: ' + str(m['abort'])[:60] if m['abort'] else ''})"
              + (f" · beyond {m['beyond']:.2f}" if m.get("beyond") is not None else "")
              + (f" | harness: gate held {h['gate_held']} shrunk {h['gate_shrunk']} turn-cut {h['gate_turn_cut']} of {h['gate_checked']} checks, "
                 f"trigger off {h['trigger_off_frames']} fr, travel stop {h['travel_stopped_frames']} fr, clamped {h['clamped_frames']} fr" if h else ""))
    if runs:
        rxr = [E.iou(cuts[x], cuts[y])[0] for x, y in itertools.combinations(cuts, 2)]
        out["runs"] = runs
        out["summary"] = {"passed": sum(m["pass"] for m in runs.values()), "of": len(runs),
                          "passed_unhelped": sum(m["pass"] and not m["helped"] for m in runs.values()),
                          "ended_by_policy": sum(ended_by_policy(m["drill_end"], m["abort"], m["done_output"]) for m in runs.values()),
                          "with_done_output": sum(bool(m["done_output"]) for m in runs.values()),
                          **{k: round(float(np.median([m[k] for m in runs.values()])), 3) for k in ("mm3", "coverage", "precision", "off_target_mm3")},
                          "run_x_run_iou_median": round(float(np.median(rxr)), 3) if rxr else None}
        print(f"runs: {out['summary']['passed']} of {len(runs)} pass ({out['summary']['passed_unhelped']} without a blocked frame; "
              f"{out['summary']['ended_by_policy']} complete — {out['summary']['with_done_output']} of the runs with a policy that ends the drilling itself, "
              f"the others by the legacy reading) | medians: {out['summary']['mm3']:.0f} mm³ · cover {out['summary']['coverage']} · "
              f"prec {out['summary']['precision']} · off {out['summary']['off_target_mm3']} mm³ | run × run IoU median {out['summary']['run_x_run_iou_median']} "
              f"(demo × demo {out['demo_x_demo_iou']['median']})")
    if a.json:
        Path(a.json).expanduser().write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
