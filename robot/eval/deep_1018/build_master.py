"""deep_1018: one file with every basic datum of the extent-language robot study (user 10-20: "앞으로 이걸로 논문을 쓸꺼라서,
기초가되는 모든 데이터가 하나의 파일에 존재해야한다"). Everything is read again from the primary records, nothing is copied by hand.

  runs        every H90a / H90g robot run folder (runs/act_smoke/live/R_H90[ag]_*): series, date and time, level, start pose,
              instruction group, the drilling sentence actually sent (live.log), end level, who ended the drilling, abort, drilling
              seconds, live and evaluated removed volume, share of the removed volume past the level's midline, medial reach,
              the independent tracker-tip share (stage 1c), whether the run is in the PREREG v4 design and why not
  demos       the 90 hand recordings H90g was trained on (spine_hand90 … lang3 episodes.jsonl + stage1.json): day, level, labels,
              drilling seconds, final volume, cut-centre and voxel reach (user 10-23: the paper and the hackathon use these 90
              only; the 09-28 eight, all L4 ipsilateral, are in no table — the first versions held the 98 of spine_hand98 lang2)
  starts      the start poses used (runs/act_smoke/iter/starts): tips, offsets, generator, the offline preparation result
  sentences   the lang3 wordings (training) per level and extent, the novel sentences (never trained), the neutral sentence
  results     the analysis outputs the abstract uses (lv_analysis.json, pooled_analysis.json, p3_analysis_p3.json, stage1c.json)
  readme      what each table and column is, sources, the commit of the code

  python build_master.py OUT.sqlite"""
import glob, json, os, re, sqlite3, subprocess, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dataset import instructions as INS          # noqa: E402

H = os.path.expanduser("~"); LV = f"{H}/il-robot/runs/act_smoke/live"; C = f"{H}/il-robot/runs/jitter_iter"; D = f"{H}/il-robot/runs/deep_1018"
ST = f"{H}/il-robot/runs/act_smoke/iter/starts"; PF = f"{H}/il-robot/runs/groot_prep/proof_1016"
DS98 = "/srv/il-robot/datasets/spine_hand98_replay_entry_water_progress_lang2"; DS90 = "/srv/il-robot/datasets/spine_hand90_replay_entry_water_progress_lang3"
OUT = sys.argv[1]
PAT = re.compile(r"^R_(H90[ag])_(?:(pf|rf|p3|e965|g1)_)?(L[35])?(ncontra|nipsi|neutral|contra|ipsi)(\d+)(b?)_C1006_(r\w+?)(\.failed_\w+)?$")
V4 = {"L4": ["r0", "r1", "r2", "r4"], "L3": ["rL3_0", "rL3_1", "rL3_2", "rL3_3"], "L5": ["rL5_2", "rL5_3", "rL5_6", "rL5_7"]}
NOVEL = {"ncontra", "nipsi"}


def vox(sid, mid):
    f = f"{C}/iou_cache_rofull_{sid}.npz"
    if not os.path.exists(f):
        return None, None, None
    z = np.load(f, allow_pickle=True); h = json.loads(str(z["hdr"])); idx = z["final"].astype(np.int64)
    if not len(idx):
        return 0.0, 0.0, None
    x = h["origin"][0] + (idx % h["dims"][0]) * h["spacing"][0]
    return float(len(idx) * h["voxelMm3"]), float(100 * np.mean(x < mid)), float(mid - np.percentile(x, 5))


tip = {}
if os.path.exists(f"{D}/stage1c.json"):
    for w in json.load(open(f"{D}/stage1c.json"))["rows"]:
        tip[w["run"]] = (w.get("tip_past"), w.get("tip_reach"))

runs = []
for r in sorted(glob.glob(f"{LV}/R_H90[ag]_*")):
    name = os.path.basename(r)
    if not os.path.isdir(r):
        continue
    m = PAT.match(name)
    if not m:
        continue
    model, tag, lvl, group, k, rep, start, failed = m.groups(); level = lvl or "L4"
    row = {"run": name, "model": model, "level": level, "start": start, "group": group, "k": int(k), "repeat": int(bool(rep)),
           "wording": "novel" if group in NOVEL else "neutral" if group == "neutral" else "trained", "tag": tag or ""}
    lj = f"{r}/live.json"; sess = open(f"{r}/robot_session").read().strip() if os.path.exists(f"{r}/robot_session") else ""
    sid = os.path.basename(sess).replace(".jsonl", ""); row["session_id"] = sid
    t = re.search(r"s_(\d{8})T(\d{6})", sid)
    row["datetime"] = f"{t.group(1)[:4]}-{t.group(1)[4:6]}-{t.group(1)[6:]} {t.group(2)[:2]}:{t.group(2)[2:4]}:{t.group(2)[4:]}" if t else ""
    if not row["datetime"]:
        row["datetime"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(r)))
    sent = ""
    if os.path.exists(f"{r}/live.log"):
        mm = re.findall(r"drilling '([^']*)'", open(f"{r}/live.log", errors="replace").read())
        sent = mm[-1] if mm else ""
    row["sentence_sent"] = sent
    if not os.path.exists(lj):
        row.update(status="no record", excluded_reason=(open(f"{r}/DONE").read().strip() if os.path.exists(f"{r}/DONE") else "no live.json") + (" (folder kept as " + failed[1:] + ")" if failed else ""))
        runs.append(row); continue
    meta = json.load(open(lj))["meta"]; mid = INS.MIDLINE_X_MM[level]
    vol, past, reach = vox(sid, mid)
    end = meta.get("drill_end"); ab = meta.get("abort")
    row.update(status="aborted" if ab else "done", end_level=meta.get("end_level"), ended_by=end, harness_end=int(end in ("time", "stall", "volume")),
               abort=ab or "", drill_s=meta.get("drill_s"), removed_mm3_live=(meta.get("removed_mm3") or [None])[-1], removed_mm3_eval=vol,
               midline_x_mm=mid, past_midline_share_pct=past, medial_reach_mm=reach, clamped_frames=meta.get("clamped_frames"),
               tip_share_pct=(tip.get(name) or (None, None))[0], tip_reach_mm=(tip.get(name) or (None, None))[1])
    d = row["datetime"][:10]
    if model == "H90g" and tag == "e965":
        series = "2d end level 0.965 (10-18)"
    elif model == "H90g" and tag == "pf":
        series = "L4 r0 (10-17)"
    elif model == "H90g" and tag == "p3":
        series = "P3 L4 r1/r2/r4 (10-19)"
    elif model == "H90g" and level in ("L3", "L5"):
        series = "L3/L5 start A (10-17)" if d <= "2026-10-17" else "v4 L3/L5 (10-19/20)"
    else:
        series = f"H90a {tag or 'base'} ({d[5:]})"
    row["series"] = series
    inv4 = model == "H90g" and tag != "e965" and start in V4[level] and not ab and failed is None
    row["in_v4_design"] = int(inv4)
    row["excluded_reason"] = "" if inv4 else ("aborted: " + ab if ab else "end level 0.965 (2d counterfactual)" if tag == "e965" else
                                              "model H90a" if model == "H90a" else "start pose outside the v4 design" if start not in V4[level] else "")
    runs.append(row)

# demos
eps = {}
for l in open(f"{DS90}/meta/spine_il/episodes.jsonl"):
    e = json.loads(l)
    if e.get("source") == "hand":
        eps.setdefault(e["session_id"], e)
in90 = {json.loads(l)["session_id"] for l in open(f"{DS90}/meta/spine_il/episodes.jsonl")}
s1 = {d["sid"]: d for d in json.load(open(f"{D}/stage1.json"))["demos"]} if os.path.exists(f"{D}/stage1.json") else {}
# the two-class (2 mm) label as the study used it: the lang1 datasets (10-06 / 10-12) and the intake checks (10-14); 09-28 computed
cls2 = {}
for f in glob.glob("/srv/il-robot/datasets/spine_hand30_replay_entry_water_progress_lang1/meta/spine_il/episodes.jsonl"):
    for l in open(f):
        e_ = json.loads(l); cls2.setdefault(e_["session_id"], (e_["goal"].get("extent"), "lang1 dataset"))
for f in glob.glob(f"{H}/il-robot/runs/hand_incoming_20261014*/check/classify.json"):
    for c in json.load(open(f)):
        if c.get("extent"):
            for sid_ in eps:
                if sid_.endswith("_" + c["id"]) and sid_ not in cls2:
                    cls2[sid_] = (c["extent"], "intake check " + os.path.basename(os.path.dirname(os.path.dirname(f))))
demos = []
for sid, e in sorted(eps.items()):
    g = e["goal"]; v = s1.get(sid, {}); reach_c = g.get("past_midline_mm")
    lang1, lang1_src = cls2.get(sid) or (INS.extent_of([g["cut_x_p5_mm"]], g["level"]), "computed from cut_x_p5_mm (rounded)")
    demos.append({"session_id": sid, "day": sid[2:10], "level": g["level"], "label_lang2_lang3": g.get("extent") or "unspecified", "class_2mm": lang1, "class_2mm_source": lang1_src,
                  "cut_centre_reach_mm": reach_c, "voxel_reach_mm": v.get("reach"), "past_midline_share_pct": v.get("past"),
                  "final_removed_mm3": e.get("removed_mm3_final"), "drilling_s": round(e["phases"]["4"][1] - e["phases"]["4"][0], 2),
                  "in_H90g_training": int(sid in in90), "operator": e.get("operator"), "recording": e.get("recording")})

# starts
prep_fail = {"start_C1006rand_L5_0.json": "ready pose not reachable (drill entry 72 mm off the demonstrations' median)",
             "start_C1006rand_L5_1.json": "ready pose not reachable (drill entry 68 mm off the demonstrations' median)"}
starts = []
for f in sorted(glob.glob(f"{ST}/start_C1006rand_*.json")):
    b = os.path.basename(f)
    if b.endswith("_starts.json"):
        continue
    s = json.load(open(f)); st = s["state"]; key = "r" + b[len("start_C1006rand_"):-5]
    starts.append({"start": key, "file": b, "endo_tip": json.dumps([round(x, 2) for x in st[0:3]]), "drill_tip": json.dumps([round(x, 2) for x in st[7:10]]),
                   "tips_sep_mm": s.get("tips_sep_mm"), "nearest_demo_start_mm": json.dumps(s.get("nearest_demo_start_mm")),
                   "generator": s.get("from"), "seed": s.get("seed"), "try": s.get("try"), "offline_preparation": prep_fail.get(b, "passed or not checked"),
                   "used_in_v4": int(any(key == x for v in V4.values() for x in v))})

# sentences
sentences = []
for lv in INS.LEVELS:
    for ex in ("ipsilateral", "contralateral", "unspecified"):
        for i, t in enumerate(INS.wordings(4, lv, ex, "lang3"), 1):
            sentences.append({"level": lv, "extent": ex, "kind": "trained (lang3)", "index": i, "text": t, "sent_in_robot_runs": int(i == 1)})
for ex, fn in (("contralateral", "novel_contra"), ("ipsilateral", "novel_ipsi")):
    for lv in ("L3", "L4", "L5"):
        f = f"{PF}/{fn}.txt" if lv == "L4" else f"{D}/{fn}_{lv}.txt"
        if os.path.exists(f):
            sentences.append({"level": lv, "extent": ex, "kind": "novel (never trained)", "index": 1, "text": open(f).read().strip(), "sent_in_robot_runs": 1})

# results
results = []
def add(src, key, val):
    results.append({"source": src, "key": key, "value": json.dumps(val) if not isinstance(val, (int, float, str)) else val})
if os.path.exists(f"{D}/stage1.json"):          # the demos under the robot's metric (1a cells, ① rule, 1d contra target)
    for k in ("cells", "rule_1", "voxel_vs_cut", "target"):
        add("stage1.json", k, json.load(open(f"{D}/stage1.json"))["res"].get(k))
if os.path.exists(f"{D}/stall_basis_hand90_replay_entry_water_progress_lang3.json"):   # stall_basis.py: least removed in any 25 s / 15 s
    add("stall_basis.json", "demos90", json.load(open(f"{D}/stall_basis_hand90_replay_entry_water_progress_lang3.json")))
for fn in ("lv_analysis.json", "pooled_analysis.json", "p3_analysis_p3.json"):
    p = f"{D}/{fn}"
    if os.path.exists(p):
        j = json.load(open(p))
        for k, v in j.items():
            if k != "rows" and k != "runs":
                add(fn, k, v)
if os.path.exists(f"{D}/stage1c.txt"):
    add("stage1c.txt", "summary", open(f"{D}/stage1c.txt").read())

commit = subprocess.run(["git", "-C", os.path.join(os.path.dirname(__file__), ".."), "log", "-1", "--format=%h %ad", "--date=iso"], capture_output=True, text=True).stdout.strip()
readme = [
    ("about", "deep_1018 extent-language robot study (H90g = GR00T N1.7 fine-tuned on 90 hand recordings, lang3 instructions). Built by robot/eval/deep_1018/build_master.py at " + time.strftime("%Y-%m-%d %H:%M") + ", code commit " + commit),
    ("runs", "one row per robot run folder. status done / aborted / no record. in_v4_design = 1: the 105 runs of the PREREG v4 design (H90g, end level 0.85, four start poses per level, not aborted). past_midline_share_pct = share of the run's removed bone volume (replay voxels, 0.8 mm grid) lying past the level's midline plane (midline_x_mm); medial_reach_mm = midline − 5th percentile of removed-voxel x. harness_end = 1: ended by the harness (time / stall / volume) = a failed run by the user's rule, still analysed for the side. sentence_sent = the drilling sentence the loop sent (live.log). datetime = robot session start (local time)."),
    ("demos", "the 90 hand recordings H90g was trained on (10-06 … 10-14, one operator; the 09-28 eight are left out, user 10-23). label_lang2_lang3 = the sentence class used in training (not past the midline → ipsilateral, > 5 mm past → contralateral, in between → unspecified / neutral sentence; cut centres). class_2mm = the two-class split (> 2 mm past = contralateral). in_H90g_training = 1 for every row (kept for queries written against the 98-row versions)."),
    ("starts", "start poses: tips in the scene frame (mm), generator text, offline preparation (live_loop.py --prepare-out) result. rL5_0/1 failed it and were replaced by rL5_6/7 (PREREG v4.1)."),
    ("sentences", "lang3 drilling wordings (15 per extent, 10 neutral) per level; index 1 is the one the robot runs sent; the novel sentences (10-16, level word changed) were never in training."),
    ("results", "the analysis outputs (JSON as text) behind the abstract; p values from permutation tests described in PREREG.md (v1 … v4.1) and pooled_analysis.py (post hoc)."),
    ("sources", f"runs: {LV}/R_*/(live.json, live.log, robot_session) · voxels: {C}/iou_cache_rofull_<session>.npz · demos: {DS90} and {D}/stage1.json · starts: {ST} · sentences: dataset/instructions.py, {PF}/novel_*.txt, {D}/novel_*_L?.txt · analyses: {D}/*.json"),
]

if os.path.exists(OUT):
    os.remove(OUT)
con = sqlite3.connect(OUT)
for name, rows in (("runs", runs), ("demos", demos), ("starts", starts), ("sentences", sentences), ("results", results)):
    cols = []
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    con.execute(f"CREATE TABLE {name} (" + ", ".join(f'"{c}"' for c in cols) + ")")
    con.executemany(f"INSERT INTO {name} VALUES (" + ",".join("?" * len(cols)) + ")", [[r.get(c) for c in cols] for r in rows])
con.execute("CREATE TABLE readme (topic, text)"); con.executemany("INSERT INTO readme VALUES (?, ?)", readme)
con.commit()
for t in ("runs", "demos", "starts", "sentences", "results"):
    print(t, con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
print("v4 runs", con.execute("SELECT COUNT(*) FROM runs WHERE in_v4_design=1").fetchone()[0],
      "| by level/group", con.execute("SELECT level, group_concat(n) FROM (SELECT level, \"group\"||':'||COUNT(*) n FROM runs WHERE in_v4_design=1 GROUP BY level, \"group\") GROUP BY level").fetchall())
con.close()
