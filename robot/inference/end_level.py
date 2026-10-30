#!/usr/bin/env python3
"""The level at which a progress prediction ends the drilling — computed (user, 10-09: "끝 수준은 네가 최적값 계산").

A policy with the progress output (drill.progress: bone removed so far / the demonstration's final volume) ends the
drilling when its prediction stays above a level (rollout.done_step: 5 steps in a row, beginning within the steps about to
be executed). The level trades two losses against each other:

  too low    the drilling ends early and bone is left — at the level L a perfect predictor leaves 1 − L of the volume
  too high   a prediction that falls short near the end never gets there — a harness bound ends the drilling and the run
             fails (user, 10-07: a harness-forced end is a failed run)

so the level to use is the one with the largest
    U(L) = mean over demonstrations of [ fraction of the demonstration's volume removed when the rule fires · 0 if it never fires ]
measured on demonstrations the predictor has not seen. The rule is walked as the loop executes it: a prediction every
n_exec frames (every phase of that grid), a chunk of future values, rollout.done_step — from the episode's first frame:
the loop takes the policy's end as soon as the tools are inside the skin, before the first cut too (PhaseTracker.policy_done;
seen in the dry run of 10-09: a server that answered "progress 1.0" to the first question ended the run with 0 mm³).

What predicts
  --policy-url URL | --policy-path DIR   the model itself (the trained GR00T behind remote/policy_server.py, an ACT
                                         checkpoint) on --episodes it was not trained on. THE calibration that counts:
                                             python end_level.py --dataset DS --policy-url URL --policy-token-file F --episodes 2,18
  --probe                                before a model exists: a linear read-out of the progress 1…T frames ahead from a
                                         frozen image embedding (ResNet-18, ImageNet) and the tool state; every
                                         demonstration is predicted by a read-out fitted on the others (and, harsher, by
                                         one fitted on the other recording day only). A stand-in: how well the end can be
                                         told from what a policy sees, and where the optimum goes when the prediction is poor.
  --oracle                               the label itself — a perfect predictor: the cost side alone.
With more than one predictor the level with the smallest worst-case loss among them is reported too (minimax regret):
the trained model is expected between the perfect predictor and the stand-in.

    python end_level.py --dataset ~/il-robot/datasets/spine_hand24_replay_trim_water_progress --oracle --probe -o OUT.json

CPU only; the embedding of 47k frames takes a quarter of an hour on four cores (cached under --cache). No renderer, no robot.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rollout as RO          # noqa: E402

LEVELS = np.round(np.arange(0.50, 0.9951, 0.005), 3)


# ---------------------------------------------------------------- the demonstrations
def load(ds: Path) -> dict[int, dict]:
    """episode → {progress (n,), state (n,15), phase (n,), first_cut, last_cut, session, day} from the dataset's own columns."""
    import pyarrow.parquet as pq
    cols = ["episode_index", "frame_index", "cut.progress", "observation.state", "phase.index"]
    rows: dict[int, list] = {}
    for f in sorted((ds / "data").glob("*/*.parquet")):
        t = pq.read_table(f, columns=cols).to_pydict()
        for e, k, p, s, ph in zip(*(t[c] for c in cols)):
            rows.setdefault(int(e), []).append((int(k), float(p[0] if isinstance(p, list) else p), s, int(ph[0] if isinstance(ph, list) else ph)))
    meta = [json.loads(l) for l in (ds / "meta/spine_il/episodes.jsonl").read_text().splitlines() if l.strip()]
    out = {}
    for e, r in sorted(rows.items()):
        if e < len(meta) and meta[e].get("decision_copy"):   # export_lerobot.py --decision-copies: a part again, not a recording
            continue
        r.sort(key=lambda x: x[0])
        p = np.asarray([x[1] for x in r], float)
        up = np.nonzero(np.diff(p, prepend=p[0]) > 1e-6)[0]
        sid = meta[e].get("session_id", "") if e < len(meta) else ""
        out[e] = {"progress": p, "state": np.asarray([x[2] for x in r], float)[:, :15], "phase": np.asarray([x[3] for x in r], int),
                  "first_cut": int(up[0]) if len(up) else len(p) - 1, "last_cut": int(up[-1]) if len(up) else len(p) - 1,
                  "session": sid, "day": sid[2:10] if sid.startswith("s_") else sid[:8],
                  "goal": (meta[e].get("goal") if e < len(meta) else None) or None}   # level / extent of the recording (lang1 texts)
    return out


def video_map(ds: Path) -> dict[int, tuple[Path, int, int]]:
    """episode → (video file, first frame in the file, number of frames)."""
    import pyarrow.parquet as pq
    info = json.loads((ds / "meta/info.json").read_text())
    key = next(k for k, v in info["features"].items() if v.get("dtype") == "video")
    fps = info["fps"]
    out = {}
    for f in sorted((ds / "meta/episodes").glob("*/*.parquet")):
        t = pq.read_table(f).to_pydict()
        for i, e in enumerate(t["episode_index"]):
            path = ds / "videos" / key / f"chunk-{t[f'videos/{key}/chunk_index'][i]:03d}" / f"file-{t[f'videos/{key}/file_index'][i]:03d}.mp4"
            out[int(e)] = (path, int(round(t[f"videos/{key}/from_timestamp"][i] * fps)), int(t["length"][i]))
    return out


def frames(ds: Path, episodes=None, stride: int = 1):
    """(episode, frame index, HxWx3 uint8) for every `stride`-th frame, each video file decoded once from its start."""
    import av
    vm = video_map(ds)
    by_file: dict[Path, list] = {}
    for e, (path, a, n) in vm.items():
        if episodes is None or e in episodes:
            by_file.setdefault(path, []).append((a, n, e))
    for path, eps in sorted(by_file.items()):
        eps.sort()
        want = {a + k: (e, k) for a, n, e in eps for k in range(0, n, stride)}
        last = max(want)
        with av.open(str(path)) as c:
            for i, fr in enumerate(c.decode(video=0)):
                if i in want:
                    yield (*want[i], fr.to_ndarray(format="rgb24"))
                if i >= last:
                    break


def targets(progress: np.ndarray, horizon: int) -> np.ndarray:
    """(n, horizon): the progress 1…horizon frames ahead of every frame — what the policy's output is trained on (beyond the
    episode's end: its last value)."""
    n = len(progress)
    idx = np.minimum(np.arange(n)[:, None] + np.arange(1, horizon + 1)[None, :], n - 1)
    return progress[idx]


# ---------------------------------------------------------------- the rule, walked along a demonstration
def walk(chunks: dict[int, np.ndarray], true: np.ndarray, start: int, n_exec: int, level: float, offset: int = 0,
         hold: int = RO.DONE_HOLD) -> int | None:
    """The frame at which the loop's rule ends the drilling, None if never. `chunks`: frame t → the prediction made there
    (values for frames t+1 …). A prediction is made at offset, offset + n_exec, … (the nearest frame at or after it that has
    one); the rule counts from frame `start` on — the first frame of the policy's part (0 in a trimmed episode: the loop
    honours an end before the first cut as well, and such a run has removed nothing)."""
    n, t = len(true), offset
    have = np.asarray(sorted(chunks))
    while t < n:
        i = int(np.searchsorted(have, t))
        if i >= len(have):
            break
        t = int(have[i])
        if t >= start:
            j = RO.done_step(chunks[t], n_exec, hold, level)
            if j is not None:
                return min(t + j, n - 1)
        t += n_exec
    return None


def sweep(eps: dict[int, dict], pred: dict[int, dict[int, np.ndarray]], n_exec: int, levels=LEVELS, offsets=None, fps: float = 15.0) -> dict:
    """U(L) and what it is made of, over the episodes that have predictions and the grid phases `offsets`."""
    offsets = list(range(0, n_exec, max(1, n_exec // 4))) if offsets is None else list(offsets)
    out = {"levels": [float(x) for x in levels], "episodes": sorted(pred), "offsets": offsets, "n_exec": n_exec, "rows": []}
    per_ep = np.zeros((len(levels), len(pred)))
    for li, L in enumerate(levels):
        u, miss, frac, dt = [], 0, [], []
        for ei, e in enumerate(sorted(pred)):
            d = eps[e]; ue = []
            for o in offsets:
                k = walk(pred[e], d["progress"], d.get("start", 0), n_exec, float(L), o)
                if k is None:
                    miss += 1; ue.append(0.0)
                else:
                    ue.append(float(d["progress"][k])); frac.append(float(d["progress"][k])); dt.append((k - d["last_cut"]) / fps)
            per_ep[li, ei] = np.mean(ue); u += ue
        out["rows"].append({"level": float(L), "U": float(np.mean(u)), "missed": miss / len(u), "removed_when_ended": float(np.mean(frac)) if frac else None,
                            "end_vs_last_cut_s": {"median": float(np.median(dt)), "min": float(np.min(dt)), "max": float(np.max(dt))} if dt else None})
    U = np.asarray([r["U"] for r in out["rows"]])
    best = int(np.argmax(U))
    se = float(per_ep[best].std(ddof=1) / np.sqrt(per_ep.shape[1])) if per_ep.shape[1] > 1 else 0.0
    near = np.nonzero(U >= U[best] - se)[0]
    out.update(best={"level": float(levels[best]), **{k: out["rows"][best][k] for k in ("U", "missed", "removed_when_ended", "end_vs_last_cut_s")}},
               se_of_best=se, indistinguishable={"lowest": float(levels[near[0]]), "highest": float(levels[near[-1]])}, U=[float(x) for x in U])
    # per episode (rows = levels, columns = out["episodes"]): a level for a subset (one extent, 10-16) without asking again
    out["U_per_episode"] = [[round(float(x), 4) for x in row] for row in per_ep]
    return out


def minimax(sweeps: dict[str, dict]) -> dict:
    """The level whose worst loss against each predictor's own best is smallest."""
    names = sorted(sweeps); levels = sweeps[names[0]]["levels"]
    R = np.asarray([np.max(sweeps[m]["U"]) - np.asarray(sweeps[m]["U"]) for m in names])
    worst = R.max(0); i = int(np.argmin(worst))
    return {"level": levels[i], "worst_loss": float(worst[i]), "loss": {m: float(R[k, i]) for k, m in enumerate(names)},
            "own_best": {m: sweeps[m]["best"]["level"] for m in names}}


# ---------------------------------------------------------------- predictors
def oracle(eps: dict[int, dict], horizon: int, stride: int = 1) -> dict:
    # ONE target array per episode, its rows shared. (10-09 12:38: `targets(...)[t]` inside the loop kept a full (n, horizon)
    # array alive behind every row — n² values an episode, 4 GB on the real dataset; the kernel killed the process and the
    # camera pipeline stalled with it: the tracker restarted twice.)
    out = {}
    for e, d in eps.items():
        T = targets(d["progress"], horizon)
        out[e] = {t: T[t] for t in range(0, len(d["progress"]), stride)}
    return out


def embed(ds: Path, cache: Path, stride: int, threads: int = 4, log=print) -> dict:
    """Frozen ResNet-18 (ImageNet) embedding of every `stride`-th frame: {X (m, 512), ep (m,), k (m,)} — cached."""
    f = cache / f"feat_{ds.name}_s{stride}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in ("X", "ep", "k")}
    import time
    import torch
    import torchvision
    torch.set_num_threads(threads)
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity(); net.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    X, E, K, buf, t0 = [], [], [], [], time.time()

    def flush():
        if buf:
            x = torch.from_numpy(np.stack(buf)).permute(0, 3, 1, 2).float() / 255.0
            with torch.no_grad():
                X.append(net((x - mean) / std).numpy().astype(np.float32))
            buf.clear()
    for e, k, img in frames(ds, stride=stride):
        buf.append(img); E.append(e); K.append(k)
        if len(buf) == 64:
            flush()
            if len(E) % 1920 == 0:
                log(f"[embed] {len(E)} frames, {len(E) / (time.time() - t0):.1f} /s")
    flush()
    out = {"X": np.concatenate(X), "ep": np.asarray(E), "k": np.asarray(K)}
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **out)
    return out


def probe(eps: dict[int, dict], feat: dict, horizon: int, lams=(1.0, 10.0, 100.0, 1000.0), group=None) -> tuple[dict, dict]:
    """Held-out predictions of a ridge read-out [embedding, tool state] → progress 1…horizon frames ahead. `group(e)` names
    the fold of an episode (default: every episode its own fold = leave one demonstration out); the read-out for a fold is
    fitted on all other folds, features and targets centred on those. The ridge weight is the one of `lams` with the
    smallest held-out error of the prediction itself (not of the level)."""
    group = group or (lambda e: e)
    Z, Y, G = {}, {}, {}
    scale = None
    for e in sorted(eps):
        m = feat["ep"] == e
        k = feat["k"][m]; o = np.argsort(k); k = k[o]
        st = eps[e]["state"][k]
        if scale is None:   # one fixed scale for the state values (mm against unit quaternions), from the first episode
            scale = np.maximum(st.std(0), 1e-3)
        Z[e] = np.concatenate([feat["X"][m][o].astype(np.float64), st / scale], axis=1); Y[e] = targets(eps[e]["progress"], horizon)[k]; G[e] = k
    folds = sorted({group(e) for e in eps}, key=str)
    S = {g: {"n": 0, "sx": 0.0, "sy": 0.0, "xx": 0.0, "xy": 0.0} for g in folds}
    for e in eps:
        s = S[group(e)]
        s["n"] += len(Z[e]); s["sx"] = s["sx"] + Z[e].sum(0); s["sy"] = s["sy"] + Y[e].sum(0); s["xx"] = s["xx"] + Z[e].T @ Z[e]; s["xy"] = s["xy"] + Z[e].T @ Y[e]
    tot = {k: sum(S[g][k] for g in folds) for k in ("n", "sx", "sy", "xx", "xy")}
    best = None
    for lam in lams:
        pred, err = {}, []
        for g in folds:
            n = tot["n"] - S[g]["n"]
            if n <= 0:
                continue
            mx = (tot["sx"] - S[g]["sx"]) / n; my = (tot["sy"] - S[g]["sy"]) / n
            A = (tot["xx"] - S[g]["xx"]) - n * np.outer(mx, mx)
            B = (tot["xy"] - S[g]["xy"]) - n * np.outer(mx, my)
            W = np.linalg.solve(A + lam * np.eye(len(mx)), B)
            for e in eps:
                if group(e) == g:
                    P = (Z[e] - mx) @ W + my
                    pred[e] = {int(t): P[i] for i, t in enumerate(G[e])}
                    err.append(np.abs(P - Y[e]))
        mae = float(np.concatenate(err).mean())
        if best is None or mae < best[0]:
            best = (mae, lam, pred)
    mae, lam, pred = best
    late = np.concatenate([np.stack([pred[e][int(t)] for t in G[e]])[Y[e][:, 0] >= 0.9, 0] - Y[e][Y[e][:, 0] >= 0.9, 0] for e in pred])
    top = [float(max(v[0] for v in pred[e].values())) for e in sorted(pred)]
    info = {"ridge": lam, "folds": len(folds), "mae": mae, "frames": int(sum(len(G[e]) for e in pred)),
            "error_where_true_progress_is_0.9_or_more": {"mean": float(late.mean()), "sd": float(late.std()), "n": int(len(late))},
            "highest_prediction_per_episode": {"min": float(np.min(top)), "median": float(np.median(top)), "max": float(np.max(top))}}
    return pred, info


def from_policy(ds: Path, eps: dict[int, dict], episodes: list[int], n_exec: int, offsets: list[int], predict, log=print, progress: bool = True,
                task=None) -> tuple[dict, dict]:
    """The model's own answers on `episodes`, asked where the loop would ask: frames offset + i·n_exec. Returns the
    progress chunks (for the level; empty for a model without that output, progress=False) and the full answers (rows of
    absolute tool states, for open_loop). The phase text is the one the loop sends such a model (remote_policy.task_for)."""
    import remote_policy as RP
    ask = {e: sorted({t for o in offsets for t in range(o, len(eps[e]["progress"]), n_exec)}) for e in episodes}
    out = {e: {} for e in episodes}; full = {e: {} for e in episodes}
    for e, k, img in frames(ds, set(episodes)):
        if k in ask[e]:
            text = task(int(eps[e]["phase"][k]), eps[e].get("goal")) if task else RP.task_for(int(eps[e]["phase"][k]), progress)
            rows = np.asarray(predict(img, eps[e]["state"][k], text), float)
            if progress and rows.shape[1] < 16:
                raise SystemExit(f"the policy answers {rows.shape[1]} values a row — no progress output (16th value)")
            if progress:
                out[e][k] = rows[:, 15]
            full[e][k] = rows
            if len(full[e]) % 20 == 0:
                log(f"[policy] episode {e}: {len(full[e])}/{len(ask[e])}")
    return out, full


def open_loop(eps: dict[int, dict], answers: dict[int, dict[int, np.ndarray]]) -> dict:
    """Does the model's answer move like the demonstration does? The scores of openloop_eval.py (the ACT numbers of
    10-09), from the answers a server gave: per phase (3 approach, 4 drilling) and tool, over H = 1, 16, 40 steps —
    gain = least-squares slope of the predicted move along the demonstration's direction (≪ 1: the model stands still),
    cos = median cosine between the two moves, both over the frames where the demonstration actually moved;
    trigger = agreement over the first 16 steps; progress = error of the 16th value against the demonstration's."""
    HS = (1, 16, 40); acc: dict = {}; trig: dict = {}; perr: dict = {}
    for e, d in answers.items():
        st, pr, n = eps[e]["state"], eps[e]["progress"], len(eps[e]["state"])
        for t, rows in d.items():
            ph = 3 if t < eps[e]["first_cut"] else 4 if t <= eps[e]["last_cut"] else 5
            for H in HS:
                if H <= len(rows) and t + H < n:
                    for tool, off in (("endo", 0), ("drill", 7)):
                        acc.setdefault((ph, tool, H), []).append((rows[H - 1, off:off + 3] - st[t, off:off + 3], st[t + H, off:off + 3] - st[t, off:off + 3]))
            m = min(16, len(rows), n - 1 - t)
            if m > 0:
                trig.setdefault(ph, []).append(float(((rows[:m, 14] > 0.5) == (st[t + 1:t + 1 + m, 14] > 0.5)).mean()))
                if rows.shape[1] >= 16:
                    perr.setdefault(ph, []).append(rows[:m, 15] - pr[t + 1:t + 1 + m])
    out: dict = {}
    for ph in sorted({k[0] for k in acc}):
        o = {"answers": len(trig.get(ph, [])), "trigger_agree_16": round(float(np.mean(trig[ph])), 3) if ph in trig else None}
        if ph in perr:
            pe = np.concatenate(perr[ph]); o["progress_error"] = {"mean": round(float(pe.mean()), 4), "mean_abs": round(float(np.abs(pe).mean()), 4)}
        for (p_, tool, H), rows in sorted(acc.items(), key=lambda x: (x[0][1], x[0][2])):
            if p_ != ph:
                continue
            P = np.array([r[0] for r in rows]); D = np.array([r[1] for r in rows])
            nd, npr = np.linalg.norm(D, axis=1), np.linalg.norm(P, axis=1)
            moved = nd > 0.3 * (H ** 0.5)                                   # the demonstration actually moved (mm)
            along = np.einsum("ij,ij->i", P, D) / np.maximum(nd, 1e-9)
            o[f"{tool}_H{H}"] = {"demo_mm_median": round(float(np.median(nd)), 3), "pred_mm_median": round(float(np.median(npr)), 3),
                                 "gain_slope": round(float((along[moved] * nd[moved]).sum() / max((nd[moved] ** 2).sum(), 1e-9)), 3) if moved.any() else None,
                                 "cos_median": round(float(np.median(along[moved] / np.maximum(npr[moved], 1e-9))), 3) if moved.any() else None,
                                 "moved_frac": round(float(moved.mean()), 3)}
        out[{3: "approach", 4: "drilling", 5: "after the last cut"}[ph]] = o
    late = [rows[:min(16, len(rows), len(eps[e]["progress"]) - 1 - t), 15] - eps[e]["progress"][t + 1:t + 1 + min(16, len(rows), len(eps[e]["progress"]) - 1 - t)]
            for e, d in answers.items() for t, rows in d.items() if eps[e]["progress"][t] >= 0.9 and rows.shape[1] >= 16]
    if late:
        lv = np.concatenate(late); out["progress_error_where_true_progress_is_0.9_or_more"] = {"mean": round(float(lv.mean()), 4), "sd": round(float(lv.std()), 4), "n": int(len(lv))}
    return out


def print_open_loop(res: dict) -> None:
    if "open_loop" not in res:
        return
    for ph, o in res["open_loop"].items():
        if isinstance(o, dict) and "drill_H16" in o:
            d16, e16 = o["drill_H16"], o.get("endo_H16", {})
            print(f"open loop, {ph}: drill over 16 steps — direction {d16['cos_median']}, gain {d16['gain_slope']} (demonstration {d16['demo_mm_median']} mm, model {d16['pred_mm_median']} mm); "
                  f"endoscope direction {e16.get('cos_median')}, gain {e16.get('gain_slope')}; trigger agreement {o['trigger_agree_16']}; progress error {o.get('progress_error')}")
    if "progress_error_where_true_progress_is_0.9_or_more" in res["open_loop"]:
        print("progress error where the true progress is 0.9 or more:", res["open_loop"]["progress_error_where_true_progress_is_0.9_or_more"])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--oracle", action="store_true"); ap.add_argument("--probe", action="store_true")
    ap.add_argument("--policy-url"); ap.add_argument("--policy-token-file"); ap.add_argument("--policy-path")
    ap.add_argument("--episodes", default=None, help="for a policy: the demonstrations it was not trained on (e.g. 2,18)")
    ap.add_argument("--n-exec", type=int, default=16); ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--offsets", default=None, help="grid phases, e.g. 0,4,8,12 (default: four, evenly; a policy: 0 only)")
    ap.add_argument("--stride", type=int, default=3, help="probe: every n-th frame is embedded")
    ap.add_argument("--blend", default="0.1,0.25,0.5,0.75", help="probe: predictors with this share of the stand-in's error (\"\" = none)")
    ap.add_argument("--cache", default=str(Path.home() / "il-robot/runs/act_smoke/iter/end_level"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args(argv)
    ds = Path(a.dataset).expanduser(); eps = load(ds)
    # an episode that does not contain the end of its drilling (a window of a longer session: its progress stops short of 1)
    # says nothing about ending — it stays in the stand-in's fit (its frames are training data like any other), not in the walk
    partial = sorted(e for e, d in eps.items() if d["progress"][-1] < 0.99)
    info = json.loads((ds / "meta/info.json").read_text()); fps = float(info["fps"])
    offs = [int(x) for x in a.offsets.split(",")] if a.offsets else None
    res = {"dataset": str(ds), "episodes": len(eps), "frames": int(sum(len(d["progress"]) for d in eps.values())), "rule": {"n_exec": a.n_exec, "hold": RO.DONE_HOLD, "horizon": a.horizon},
           "level_in_use": RO.END_LEVEL["progress"], "episodes_without_the_end": {e: round(float(eps[e]["progress"][-1]), 3) for e in partial}, "predictors": {}}
    whole = lambda pred: {e: v for e, v in pred.items() if e not in partial}       # noqa: E731
    sweeps = {}
    if a.oracle:
        sweeps["perfect"] = sweep(eps, whole(oracle(eps, a.horizon)), a.n_exec, offsets=offs, fps=fps)
    if a.probe:
        feat = embed(ds, Path(a.cache), a.stride, a.threads)
        grid = offs if offs is not None else [o for o in range(0, a.n_exec, max(1, a.n_exec // 4))]
        for name, group in (("stand-in, one demonstration left out", None), ("stand-in, the other recording day left out", lambda e: eps[e]["day"])):
            pred, pinfo = probe(eps, feat, a.horizon, group=group)
            sweeps[name] = sweep(eps, whole(pred), a.n_exec, offsets=grid, fps=fps)
            sweeps[name]["predictor"] = pinfo
            if group is None:
                # predictors between the perfect one and the stand-in: the stand-in's error, scaled — where the best level
                # goes as the prediction gets better (a trained model's held-out error tells which row is its own)
                for share in [float(x) for x in a.blend.split(",") if x]:
                    mix, err = {}, []
                    for e, ch in pred.items():
                        T = targets(eps[e]["progress"], a.horizon)
                        mix[e] = {t: T[t] + share * (v - T[t]) for t, v in ch.items()}
                        err += [np.abs(mix[e][t] - T[t]) for t in ch]
                    nm = f"between: {share:g} of the stand-in's error"
                    sweeps[nm] = sweep(eps, whole(mix), a.n_exec, offsets=grid, fps=fps)
                    sweeps[nm]["predictor"] = {"mae": float(np.mean(err))}
    if a.policy_url or a.policy_path:
        if not a.episodes:
            raise SystemExit("--episodes: the demonstrations the model was not trained on")
        held = [int(x) for x in a.episodes.split(",")]
        if a.policy_url:
            import remote_policy as RP
            rp = RP.RemotePolicy(a.policy_url, RP.read_token(a.policy_token_file), timeout_s=60, run="end_level")
            inf = rp.info()
            predict, name, task = rp.act, f"model {inf.get('model')}", rp.task
            if rp.end_kind != "progress":
                # no level to compute: the harness's bounds end such a model's drilling. Its answers are still scored
                # against the demonstrations (the comparison the two trial GR00T models of 10-08 need).
                o = offs if offs is not None else [0]
                _, answers = from_policy(ds, eps, held, a.n_exec, o, predict, progress=False, task=task)
                res.update(model=name, end_output=rp.end_kind, open_loop=open_loop(eps, answers), answers=int(sum(len(d) for d in answers.values())),
                           round_trip_s=rp.summary()["round_trip_s"])
                res["predictors"] = {}
                if a.out:
                    Path(a.out).write_text(json.dumps(res, ensure_ascii=False))
                print(f"{name}: no progress output (end output: {rp.end_kind}) — no end level; the harness's bounds end its drilling")
                print_open_loop(res)
                return
        else:
            sys.path.insert(0, str(HERE / "remote"))
            from types import SimpleNamespace
            import policy_server as PS
            import remote_policy as RP_
            be = PS.LerobotBackend(SimpleNamespace(instructions_file=None, policy_path=a.policy_path, torch_threads=a.threads, device="cpu", post="auto", optics="water", end_output="progress"))
            predict, name = be.predict, f"model {Path(a.policy_path).parent.parent.parent.name}"
            task = lambda ph, goal: RP_.task_for(ph, True, be.instructions, goal)   # noqa: E731
        o = offs if offs is not None else [0]
        chunks, answers = from_policy(ds, eps, held, a.n_exec, o, predict, task=task)
        sweeps[name] = sweep(eps, whole(chunks), a.n_exec, offsets=o, fps=fps)
        res["open_loop"] = open_loop(eps, answers)
        allerr = np.concatenate([v[:min(len(v), len(eps[e]["progress"]) - 1 - t)] - eps[e]["progress"][t + 1:t + 1 + min(len(v), len(eps[e]["progress"]) - 1 - t)]
                                 for e, d in chunks.items() for t, v in d.items() if t < len(eps[e]["progress"]) - 1])
        sweeps[name]["predictor"] = {"mae": float(np.abs(allerr).mean()), "answers": int(sum(len(d) for d in chunks.values())), "rows_per_answer": int(len(next(iter(next(iter(chunks.values())).values()))))}
        res.update(model=name, end_output="progress")
        if a.policy_url:
            res["round_trip_s"] = rp.summary()["round_trip_s"]
    if not sweeps:
        raise SystemExit("nothing to do: --oracle, --probe, --policy-url or --policy-path")
    res["predictors"] = sweeps
    if len(sweeps) > 1:
        res["minimax"] = minimax(sweeps)
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False))
    for m, s in sweeps.items():
        b = s["best"]
        print(f"{m}: best level {b['level']:.3f} — U {b['U']:.3f}, missed {b['missed']:.0%}, removed when ended {b['removed_when_ended']:.3f}, "
              f"end {b['end_vs_last_cut_s']['median']:+.1f} s vs the last cut; not distinguishable from it: {s['indistinguishable']['lowest']:.3f}–{s['indistinguishable']['highest']:.3f}"
              + (f"; prediction error {s['predictor']['mae']:.3f}" if "predictor" in s else ""))
    print_open_loop(res)
    if "minimax" in res:
        print(f"smallest worst-case loss: level {res['minimax']['level']:.3f} (loss ≤ {res['minimax']['worst_loss']:.3f}; " + ", ".join(f"{m} {v:.3f}" for m, v in res["minimax"]["loss"].items()) + ")")


if __name__ == "__main__":
    main()
