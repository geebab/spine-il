import json, glob
from pathlib import Path
P = Path(__file__).parent
ids = ["3868", "91cb", "bef5", "d340", "bad1", "0338", "919d", "6639"]
out = {}
for rid in ids:
    f = P / f"pair_{rid}.json"
    if not f.exists(): out[rid] = None; continue
    out[rid] = json.load(open(f))["best"]
json.dump(out, open(P / "pairs_by_recording.json", "w"), indent=1)
for rid, b in out.items():
    if b is None: print(rid, "MISSING"); continue
    print(rid, b["q_left"], b["q_right"], b["vec_target_mm"], b["vec_err_mm"], b["sep_mm"], b["ori_err_deg"], b["arms_min_mm_start"],
          b["arms_min_mm_replay"], b["replay_unsafe"], b["replay_n_checked"], b["joint_margin_min_deg"], b["joint_margin_min_joint"],
          b["paths_ok_relaxed"], b["paths"]["strict"]["left_second_see_both"], b["dq_from_current"], b["seed"])
