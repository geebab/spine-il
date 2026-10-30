#!/usr/bin/env python3
"""What an intake of a policy server found, in a few lines, and the commands for the robot runs — printed, never run.

    python intake_summary.py DIR --tag TAG --url URL [--token-file F] [--ca-file F]

DIR holds check.json (remote/connect_check.py), model_eval.json (end_level.py --policy-url) and, after the dry run,
dry.json (live/dry_remote.sh). Written by live/remote_intake.sh after each step.

The end level for the robot runs (a model with the progress output): the level computed before a model existed
(rollout.END_LEVEL: 0.91, the smallest worst-case loss over predictors between a perfect and a poor one) as long as the
model's own held-out sweep cannot tell it from its best level — two demonstrations are a small sample, one standard error
is wide; outside that range, the end of the range nearest to it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

DEFAULT_LEVEL = 0.91        # rollout.END_LEVEL["progress"]; read from rollout when it can be imported (below)


def default_level() -> float:
    try:
        import rollout as RO
        return float(RO.END_LEVEL["progress"])
    except Exception:
        return DEFAULT_LEVEL


def level_to_use(sweep: dict, default: float) -> dict:
    lv = [round(float(x), 3) for x in sweep["levels"]]; U = sweep["U"]; rows = sweep["rows"]
    lo, hi = float(sweep["indistinguishable"]["lowest"]), float(sweep["indistinguishable"]["highest"])
    best = float(sweep["best"]["level"])
    use = default if lo - 1e-9 <= default <= hi + 1e-9 else hi if default > hi else lo
    i = min(range(len(lv)), key=lambda k: abs(lv[k] - use))
    why = (f"{default:g} (computed before a model existed) lies inside the range this model's sweep cannot tell from its best level {best:g} ({lo:g}–{hi:g})"
           if use == default else f"this model's best level is {best:g} and {default:g} lies outside what its sweep cannot tell from it ({lo:g}–{hi:g}): the nearest end")
    return {"level": lv[i], "why": why, "U": round(float(U[i]), 3), "missed": round(float(rows[i]["missed"]), 3),
            "best_level": best, "U_at_best": round(float(sweep["best"]["U"]), 3), "range": [lo, hi]}


def load(d: Path, name: str):
    f = d / name
    try:
        return json.loads(f.read_text()) if f.is_file() else None
    except ValueError:
        return None


def summarize(d: Path, tag: str, url: str, token_file: str | None = None, ca_file: str | None = None) -> tuple[list[str], dict]:
    L: list[str] = []; out: dict = {"tag": tag, "url": url}
    chk, ev, dry = load(d, "check.json"), load(d, "model_eval.json"), load(d, "dry.json")
    if chk:
        out["check"] = chk.get("verdict"); tm = chk.get("timing") or {}; info = chk.get("info") or {}
        L.append(f"1 connection: {str(chk.get('verdict')).upper()} — {info.get('model')!r}, {info.get('action_horizon')} rows an answer, end output {chk.get('end_output')}"
                 + (f"; answers in {tm.get('median_s')} s (slowest {tm.get('slowest_s')} s, first {tm.get('first_s')} s; limit {tm.get('limit_s')} s)" if tm else ""))
        for c in chk.get("checks", []):
            if c["ok"] is False:
                L.append(f"    !! {c['check']}: {c['detail']}")
            elif c["ok"] is None and c["check"] in ("address", "auth", "timing"):
                L.append(f"    -- {c['check']}: {c['detail']}")
    else:
        L.append("1 connection: not checked")
    level = None
    if ev:
        ol = ev.get("open_loop") or {}; dr = (ol.get("drilling") or {}); d16 = dr.get("drill_H16") or {}
        line = f"2 model on the held-out demonstrations: {ev.get('model')}"
        if d16:
            line += (f" — drilling, drill over 16 steps: direction {d16.get('cos_median')}, gain {d16.get('gain_slope')} "
                     f"(demonstration {d16.get('demo_mm_median')} mm, model {d16.get('pred_mm_median')} mm); trigger agreement {dr.get('trigger_agree_16')}")
        L.append(line)
        out["open_loop_drilling"] = {"direction": d16.get("cos_median"), "gain": d16.get("gain_slope"), "trigger": dr.get("trigger_agree_16")}
        sw = next(iter((ev.get("predictors") or {}).values()), None)
        if sw:
            lv = level_to_use(sw, default_level()); level = lv["level"]; out["end_level"] = lv
            pe = sw.get("predictor", {}).get("mae"); late = ol.get("progress_error_where_true_progress_is_0.9_or_more")
            L.append(f"    progress error {pe:.3f} (mean absolute)" + (f"; where the true progress is 0.9 or more: {late['mean']:+.3f} ± {late['sd']:.3f}" if late else ""))
            L.append(f"    end level for the robot runs: {lv['level']:g} — {lv['why']}; at this level the model ends {1 - lv['missed']:.0%} of the held-out walks, U {lv['U']} (its best: {lv['U_at_best']})")
            if lv["level"] > default_level() + 1e-9:
                L.append(f"    -- that is above the level computed before a model existed ({default_level():g}): it rests on {sw.get('predictor', {}).get('answers')} answers over "
                         f"{len(sw.get('episodes') or [])} demonstrations — look at where the dry run and the first robot run end before a series")
            if lv["missed"] > 0:
                L.append(f"    !! at {lv['level']:g} the model never ends the drilling in {lv['missed']:.0%} of the walks — such a run is ended by the harness and fails by rule")
        else:
            L.append("    no progress output: no end level — the harness's bounds end the drilling, the run cannot count as \"ended by the policy\"; the cut is still compared")
    else:
        L.append("2 model evaluation: not done")
    if dry:
        out["dry"] = dry
        L.append(f"3 dry run: drilling ended by {dry.get('drill_end')!r} after {dry.get('drill_s')} s, {dry.get('removed_mm3')} mm³ in the step simulation"
                 + (f", ABORT: {dry.get('abort')}" if dry.get("abort") else "") + f" — {(dry.get('remote_policy') or {}).get('requests')} requests")
    else:
        L.append("3 dry run: not done (DRY=1 bash inference/live/remote_intake.sh … — once per new model path)")
    ready = bool(chk and chk.get("verdict") == "ready" and ev)
    out["ready_for_robot_runs"] = ready and bool(dry) and not (dry or {}).get("abort")
    env = (f"LIVE_END_LEVEL={level:g} " if level is not None else "") + (f"LIVE_POLICY_TOKEN_FILE={token_file} " if token_file and not token_file.endswith("/policy.token") else "") \
        + (f"LIVE_POLICY_CA_FILE={ca_file} " if ca_file else "")
    L.append("robot runs — printed, not started (no robot run is started on the Lenovo until the robot's owner lifts the hold of 10-09):")
    for base in ("C1006", "A0928"):
        L.append(f"    cd ~/il-robot/runs/act_smoke/live && {env}bash run_remote_base.sh {base} {url} {tag} r0      # then r1 … r4")
    out["robot_env"] = env.strip()
    return L, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir"); ap.add_argument("--tag", required=True); ap.add_argument("--url", required=True)
    ap.add_argument("--token-file", default=None); ap.add_argument("--ca-file", default=None)
    a = ap.parse_args(argv)
    d = Path(a.dir).expanduser()
    lines, out = summarize(d, a.tag, a.url, a.token_file or None, a.ca_file or None)
    (d / "SUMMARY.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    (d / "SUMMARY.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
