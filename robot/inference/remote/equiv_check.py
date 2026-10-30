#!/usr/bin/env python3
"""Does the remote path give the loop the chunk a local model gives it? One ACT checkpoint, loaded once, asked two ways
for the same dataset frames: directly (rollout.predict_chunk, what a local run does) and through the reference server
on localhost (image → PNG → JSON → server → absolute rows → JSON → remote_policy.abs_to_chunk, what a remote run does).

    python remote/equiv_check.py --policy-path CKPT/pretrained_model --dataset DS [--episodes 2,18] [--n 20] [-o out.json]

The two chunks must agree to rounding (the wire carries 6 decimals; a position is a difference of two numbers near 1000):
under 0.01 mm and 1e-4 rad. Anything more is a convention that differs between the two sides (quaternion order, units,
which frame a rotation is expressed in, the trigger's or the done value's column). No renderer, no robot.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import policy_server as PS    # noqa: E402
import remote_policy as RP    # noqa: E402
import rollout as RO          # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episodes", default="2,18")
    ap.add_argument("--n", type=int, default=20, help="frames, spread over the episodes")
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    import torch
    be = PS.LerobotBackend(SimpleNamespace(policy_path=a.policy_path, torch_threads=a.torch_threads, device="cpu", post="auto", optics="water", end_output="progress"))
    if be.rep != "chunk_relative":
        raise SystemExit("equiv_check compares with rollout.predict_chunk — a chunk-relative checkpoint (our ACT)")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), PS.make_handler(be, "equiv", log_every=10 ** 9))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    rp = RP.RemotePolicy(f"http://127.0.0.1:{srv.server_address[1]}", "equiv", timeout_s=120, run="equiv_check")
    inf = rp.info()
    ds = Path(a.dataset); eps = [int(x) for x in a.episodes.split(",")]
    rows = []
    for e in eps:
        _, states = RO.episode(ds, e)
        for k in np.linspace(10, len(states) - 50, max(1, a.n // len(eps))).astype(int):
            v = np.asarray(states[k], float); img = RO.dataset_frame(ds, e, int(k))
            for ph in ((3, 4) if be.n_state == 21 else (4,)):
                task = RP.TASKS[ph]
                sv = v if be.n_state == 15 else np.concatenate([v, np.eye(6)[ph - 1]])
                obs = {"observation.state": torch.tensor(sv, dtype=torch.float32)[None],
                       "observation.images.endo": torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)[None] / 255.0, "task": [task]}
                with torch.no_grad():
                    local = RO.predict_chunk(be.pol, be.pre, be.post, obs, be.step_norm)
                remote = rp.chunk(img, v, task)
                n = min(len(local), len(remote)); d = np.abs(local[:n, :remote.shape[1]] - remote[:n])
                rows.append({"episode": e, "frame": int(k), "phase": ph, "rows": n,
                             "pos_mm": float(max(d[:, 0:3].max(), d[:, 6:9].max())), "rot_rad": float(max(d[:, 3:6].max(), d[:, 9:12].max())),
                             "trigger": float(d[:, 12].max()), "done": float(d[:, 13].max()) if d.shape[1] > 13 else None,
                             "local_drill_move_mm": float(np.linalg.norm(local[n - 1, 6:9]))})
    srv.shutdown()
    worst = {k: max(r[k] for r in rows if r[k] is not None) for k in ("pos_mm", "rot_rad", "trigger") + (("done",) if inf["done_output"] else ())}
    ok = worst["pos_mm"] < 0.01 and worst["rot_rad"] < 1e-4 and worst["trigger"] < 1e-4 and worst.get("done", 0) < 1e-4
    out = {"policy": a.policy_path, "dataset": str(ds), "episodes": eps, "requests": len(rows), "server": inf, "worst_difference": worst,
           "agree": bool(ok), "round_trip": rp.summary()["round_trip_s"], "frames": rows}
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("requests", "worst_difference", "agree", "round_trip")}, ensure_ascii=False))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
