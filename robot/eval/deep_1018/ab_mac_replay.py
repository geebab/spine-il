"""Mac render node check before it evaluates robot runs (10-19, user: "평가는 맥세션에서", "맥 세션에서 분석가능한지 미리 확인해"):
replay one robot session that the Lenovo already evaluated, through the Mac node (RENDER_API_URL=http://127.0.0.1:1878x,
RENDER_API_INLINE=1), with the evaluation's own replay options (policy_eval REPLAY, fps 5, optics water), and compare the
removed voxels with the Lenovo's cache. Nothing is cached or overwritten. Pass = identical voxel set (IoU 1, same volume).

  cd ~/il-robot/runs/jitter_iter/work && RENDER_API_URL=http://127.0.0.1:18785 RENDER_API_INLINE=1 python ab_mac_replay.py RUN_NAME"""
import json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.home() / "il-robot/runs/jitter_iter/work"))
import iou_eval as E                                     # noqa: E402
from policy_eval import REPLAY                           # noqa: E402

LV = Path.home() / "il-robot/runs/act_smoke/live"; name = sys.argv[1]; O = LV / name
sf = Path(open(O / "robot_session").read().strip()); run = Path(open(O / "run_dir").read().strip())
sfe = run / "thin" / sf.name
sfe = sfe if sfe.exists() else sf
sid = sf.stem; ref = E.OUT / f"iou_cache_rofull_{sid}.npz"
if not ref.exists():
    raise SystemExit(f"{name}: no Lenovo evaluation to compare with ({ref})")
print(f"{name}: API {os.environ.get('RENDER_API_URL')} inline {os.environ.get('RENDER_API_INLINE')}, session {sfe.name}")
t0 = time.time(); z = E.render(str(sfe), {**REPLAY, "fps": 5.0, "optics": "water"}); v = E.voxels(z); dt = time.time() - t0
r = np.load(ref, allow_pickle=True); hr = json.loads(str(r["hdr"])); a = set(np.asarray(v["final"]).tolist()); b = set(r["final"].tolist())
same_grid = (v["hdr"].get("sdfSha256"), v["hdr"].get("gridSha256")) == (hr.get("sdfSha256"), hr.get("gridSha256"))
iou = len(a & b) / max(1, len(a | b))
res = {"run": name, "seconds": round(dt), "same_grid": same_grid, "mac_voxels": len(a), "lenovo_voxels": len(b), "iou": iou,
       "pass": bool(same_grid and a == b)}
print(json.dumps(res)); (Path.home() / "il-robot/runs/deep_1018" / f"ab_mac_{name}.json").write_text(json.dumps(res, indent=1))
sys.exit(0 if res["pass"] else 1)
