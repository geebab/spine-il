"""PAPER_FACTS section 3 figures on the 90 training recordings (H90g, spine_hand90 lang3), user 10-23: 09-28 left out."""
import json, sqlite3, collections, numpy as np, time
DS = "/srv/il-robot/datasets/spine_hand90_replay_entry_water_progress_lang3"
eps, seen = [], set()
for l in open(f"{DS}/meta/spine_il/episodes.jsonl"):
    e = json.loads(l)
    if e.get("source") == "hand" and e["session_id"] not in seen:
        seen.add(e["session_id"]); eps.append(e)
C = collections.Counter
out = {"n": len(eps)}
hdrs, ftrs, cutdt = {}, {}, []
for e in eps:
    h = f = None; tc = []
    for line in open(e["recording"]):
        if line.startswith('{"k":"hdr"'): h = json.loads(line)
        elif line.startswith('{"k":"ftr"'): f = json.loads(line)
        elif line.startswith('{"k":"cut"'): tc.append(json.loads(line)["tExp"])
    hdrs[e["session_id"]], ftrs[e["session_id"]] = h, f
    d = np.diff(tc); d = d[d < 500]
    if len(d): cutdt.append(float(np.median(d)))
g = lambda o, *ks: (lambda v: v)(__import__("functools").reduce(lambda a, k: (a or {}).get(k) if isinstance(a, dict) else None, ks, o))
day = lambda s: s[2:10]
out["by_day"] = C(day(e["session_id"]) for e in eps)
out["by_level"] = C(e["goal"]["level"] for e in eps)
out["level_by_day"] = C((day(e["session_id"]), e["goal"]["level"]) for e in eps)
out["extent"] = C(e["goal"]["extent"] for e in eps)
out["extent_by_day_level"] = C((day(e["session_id"]), e["goal"]["level"], e["goal"]["extent"]) for e in eps)
out["past_range"] = [min(e["goal"]["past_midline_mm"] for e in eps), max(e["goal"]["past_midline_mm"] for e in eps)]
out["drill_radius"] = C((e.get("drill_radius_mm"), e.get("drill_size")) for e in eps)
out["operator"] = C(e.get("operator") for e in eps); out["verdict"] = C(e.get("operator_verdict") for e in eps)
out["registration"] = C(e.get("registration") for e in eps); out["recorder"] = C(e.get("recorder") for e in eps)
out["case"] = C(e["session_id"].split("_", 2)[2].rsplit("_trk", 1)[0] for e in eps)
out["scopecal"] = C(g(hdrs[s], "fe", "scopecal") for s in hdrs)
out["light"] = C(json.dumps(g(hdrs[s], "viewState", "optics", "light"), sort_keys=True) for s in hdrs)
out["build"] = C((day(s), g(hdrs[s], "build", "gitSha")) for s in hdrs)
out["tracker"] = C(json.dumps({k: g(hdrs[s], "tracker", "cfg", k) for k in ("w", "h", "fps")}) for s in hdrs)
out["endReason"] = C(g(ftrs[s], "endReason") for s in ftrs)
dur = [g(ftrs[s], "durationS") for s in ftrs if g(ftrs[s], "durationS")]
out["duration_s"] = {"median": round(float(np.median(dur)), 1), "min": round(min(dur), 1), "max": round(max(dur), 1), "n": len(dur)}
rate = [g(ftrs[s], "tools", "endo", "rateHz") for s in ftrs if g(ftrs[s], "tools", "endo", "rateHz")]
out["pose_rate_hz_endo"] = {"median": round(float(np.median(rate)), 1), "min": round(min(rate), 1), "max": round(max(rate), 1), "n": len(rate)}
out["cut_interval_ms"] = {"median_of_medians": round(float(np.median(cutdt)), 1), "min": round(min(cutdt), 1), "max": round(max(cutdt), 1), "n": len(cutdt)}
T0 = C(json.dumps(g(hdrs[s], "registration", "T0", "t")) for s in hdrs)
sit = collections.defaultdict(list)
for s in hdrs: sit[json.dumps(g(hdrs[s], "registration", "T0", "t"))].append(s)
out["sittings"] = sorted([(day(v[0]), len(v), sorted(C(e["goal"]["level"] for e in eps if e["session_id"] in v).items())) for v in sit.values()])
t = [e["t_exp_first_ms"] for e in eps]
out["time_kst_by_day"] = {d: [time.strftime("%H:%M", time.gmtime(min(x for x, e in zip(t, eps) if day(e["session_id"]) == d) / 1000 + 9 * 3600)),
                              time.strftime("%H:%M", time.gmtime(max(x for x, e in zip(t, eps) if day(e["session_id"]) == d) / 1000 + 9 * 3600))] for d in out["by_day"]}
db = sqlite3.connect(__import__("os").path.expanduser("~/il-robot/runs/deep_1018/deep_1018_master.sqlite"))
q = db.execute("select drilling_s, final_removed_mm3, session_id from demos where in_H90g_training = 1").fetchall()
out["drilling_s"] = {"min": min(r[0] for r in q), "max": max(r[0] for r in q), "mean": round(float(np.mean([r[0] for r in q])), 1), "n": len(q),
                     "max_session": max(q, key=lambda r: r[0])[2][-4:]}
out["final_removed_mm3"] = {"min": min(r[1] for r in q), "max": max(r[1] for r in q), "mean": round(float(np.mean([r[1] for r in q])), 1),
                            "min_session": min(q, key=lambda r: r[1])[2][-4:], "max_session": max(q, key=lambda r: r[1])[2][-4:]}
q2 = db.execute("select level, class_2mm, count(*) from demos where in_H90g_training = 1 group by level, class_2mm").fetchall()
out["master_class_2mm"] = q2
lv_ipsi = db.execute("select drilling_s, final_removed_mm3 from demos where in_H90g_training = 1 and level = 'L4' and label_lang2_lang3 = 'ipsilateral'").fetchall()
lv_ipsi98 = db.execute("select drilling_s, final_removed_mm3 from demos where level = 'L4' and label_lang2_lang3 = 'ipsilateral'").fetchall()
out["L4_ipsi_drill_s_mm3_mean_90"] = [round(float(np.mean([r[0] for r in lv_ipsi])), 1), round(float(np.mean([r[1] for r in lv_ipsi])), 1), len(lv_ipsi)]
out["L4_ipsi_drill_s_mm3_mean_98"] = [round(float(np.mean([r[0] for r in lv_ipsi98])), 1), round(float(np.mean([r[1] for r in lv_ipsi98])), 1), len(lv_ipsi98)]
for k, v in out.items():
    print(k, dict(v) if isinstance(v, collections.Counter) else v)
