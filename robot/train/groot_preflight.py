#!/usr/bin/env python3
"""Before a GR00T N1.7 fine-tuning is started: what LeRobot's GR00T code will make of a dataset (user, 10-10: "그루트
학습을 위한 준비를 해라"). No model, no GPU — the dataset side only, in seconds to minutes on a CPU:

    python train/groot_preflight.py --dataset DS [--chunk-size 40] [--exclude trigger,progress] [--held-out 2,18]
                                    [--lerobot] [-o out.json]

LeRobot 0.6.1 (policies/groot, GR00T N1.7) with use_relative_actions does this with a "new_embodiment" dataset, and each
step can go wrong silently:

  layout     state and action are subtracted dimension by dimension — they must have the same length and order
             (15 against 16 stops the training; hence the dataset's 16th state value pad.zero)
  groups     the action dimensions whose name contains a --exclude word stay absolute, each a group of its own; the
             others form the relative group(s). The STATE is cut into the same groups by the same indices.
  statistics state and absolute action groups are scaled to [-1, 1] with meta/stats.json's q01/q99 and CLIPPED there.
             LeRobot writes those quantiles as the count-weighted mean of per-episode quantiles — with episodes recorded
             at different places that range is far narrower than the data, and whole sittings sit on the clip limit.
             Relative groups are scaled with the min/max of (action − state) per chunk step over all complete 40-step
             windows: one outlier (a quaternion that flips sign, a tracking jump) flattens every normal move.
  sampling   a frame is a training sample only if a complete chunk follows it: the last chunk_size − 1 frames of every
             episode are never an input. The frames after the last cut — where the progress is 1 and the end has to be
             learnt — are exactly those.
  progress   the largest progress the model can output is the q99 the absolute "progress" group is scaled with; it must
             lie above the level at which the loop ends the drilling (rollout.END_LEVEL).

--lerobot runs LeRobot's own functions on the dataset (the relative statistics as the trainer computes them, the groups
and the modality config it derives) and compares; it needs the lerobot of the training environment and a few minutes.
Exit code 1 when something would stop or silently damage a training; the report says what and how to mend it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "inference"))

NATIVE_H = 40          # GR00T N1.7's action horizon (lerobot processor_groot.N1_7_NATIVE_ACTION_HORIZON)
CLIP_OK = 0.03         # a true 1 %/99 % range clips 2 % of the values; more than this share is a wrong range


def load(ds: Path) -> dict:
    import pyarrow.parquet as pq
    info = json.loads((ds / "meta/info.json").read_text())
    cols = ["episode_index", "frame_index", "observation.state", "action", "task_index"] + (["cut.progress"] if "cut.progress" in info["features"] else [])
    rows = {c: [] for c in cols}
    for f in sorted((ds / "data").glob("*/*.parquet")):
        t = pq.read_table(f, columns=cols)
        for c in cols:
            rows[c] += t.column(c).to_pylist()
    ep = np.asarray(rows["episode_index"]); fi = np.asarray(rows["frame_index"]); o = np.lexsort((fi, ep))
    out = {"info": info, "ep": ep[o], "state": np.asarray(rows["observation.state"], np.float64)[o], "action": np.asarray(rows["action"], np.float64)[o],
           "task": np.asarray(rows["task_index"])[o].reshape(-1),
           "progress": np.asarray(rows["cut.progress"], np.float64).reshape(-1)[o] if "cut.progress" in rows else None,
           "stats": json.loads((ds / "meta/stats.json").read_text())}
    out["state_names"] = list(info["features"]["observation.state"]["names"]); out["action_names"] = list(info["features"]["action"]["names"])
    return out


def groups_of(action_names: list[str], exclude: list[str]) -> list[dict]:
    """As lerobot's _infer_n1_7_action_groups: excluded names are absolute groups of one, the runs between them relative."""
    out, cur = [], []

    def flush():
        if cur:
            out.append({"key": "single_arm" if not any(g["key"] == "single_arm" for g in out) else f"single_arm_{len(out)}", "indices": list(cur), "relative": True})
            cur.clear()
    for i, n in enumerate(action_names):
        if any(t and (t == n.lower() or t in n.lower()) for t in (x.lower() for x in exclude)):
            flush(); out.append({"key": n.removesuffix(".pos").split(".")[-1].replace(" ", "_") or "action", "indices": [i], "relative": False})
        else:
            cur.append(i)
    flush()
    return out


def windows(d: dict, episodes=None, horizon: int = NATIVE_H) -> tuple[np.ndarray, np.ndarray]:
    """(state at t, actions t … t+horizon−1) for every frame that has a complete window inside its episode."""
    S, A = [], []
    for e in np.unique(d["ep"]):
        if episodes is not None and int(e) not in episodes:
            continue
        m = np.nonzero(d["ep"] == e)[0]
        n = len(m) - horizon + 1
        if n <= 0:
            continue
        idx = m[0] + np.arange(n)[:, None] + np.arange(horizon)[None, :]
        S.append(d["state"][m[:n]]); A.append(d["action"][idx])
    return np.concatenate(S), np.concatenate(A)


def clip_share(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Share of the values that lie outside [lo, hi] by more than a hundredth of the range (a 0/1 value against a q99 of
    0.99999 is not clipped in any sense that matters)."""
    tol = 0.01 * np.maximum(hi - lo, 1e-12)
    return ((x < lo - tol) | (x > hi + tol)).mean(0)


def report(d: dict, chunk: int, exclude: list[str], held: list[int], end_level: float | None) -> dict:
    sn, an = d["state_names"], d["action_names"]
    eps = [int(e) for e in np.unique(d["ep"])]
    train = [e for e in eps if e not in held]
    sel = np.isin(d["ep"], train)
    R: dict = {"episodes": len(eps), "frames": int(len(d["ep"])), "held_out": held, "chunk_size": chunk, "problems": [], "notes": []}
    bad, note = R["problems"].append, R["notes"].append

    # ---- layout
    R["layout"] = {"state": sn, "action": an}
    if len(sn) != len(an):
        bad(f"state has {len(sn)} values, action {len(an)}: LeRobot subtracts them dimension by dimension (to_relative_actions) and stops")
    G = groups_of(an, exclude)
    R["groups"] = [{**g, "action_names": [an[i] for i in g["indices"]], "state_names": [sn[i] for i in g["indices"] if i < len(sn)]} for g in G]
    for g in G:
        for i in g["indices"]:
            if g["relative"] and i < len(sn) and sn[i] != an[i]:
                bad(f"relative dimension {i}: action '{an[i]}' is subtracted from state '{sn[i]}' — not the same quantity")
    if not any(g["relative"] for g in G):
        bad("no relative group: every action name matches an --exclude word")

    # ---- statistics of the state and of the absolute action groups: q01/q99 of meta/stats.json, clipped
    st = d["stats"]; R["scaling"] = {}
    for kind, names, X in (("observation.state", sn, d["state"][sel]), ("action", an, d["action"][sel])):
        s = st.get(kind, {})
        if "q01" not in s or "q99" not in s:
            note(f"{kind}: no q01/q99 in meta/stats.json — LeRobot falls back to min/max (no clipping, outliers set the scale)")
            continue
        lo, hi = np.asarray(s["q01"], float), np.asarray(s["q99"], float)
        tlo, thi = np.quantile(X, 0.01, axis=0), np.quantile(X, 0.99, axis=0)
        dims = range(len(names)) if kind == "observation.state" else [i for g in G if not g["relative"] for i in g["indices"]]
        share = clip_share(X, lo, hi)
        per_ep = np.stack([clip_share(d[kind.split(".")[-1]][d["ep"] == e], lo, hi) for e in train])
        rows = []
        for i in dims:
            rows.append({"name": names[i], "stats_q01_q99": [round(float(lo[i]), 4), round(float(hi[i]), 4)], "true_q01_q99": [round(float(tlo[i]), 4), round(float(thi[i]), 4)],
                         "clipped_share": round(float(share[i]), 4), "episodes_more_than_half_clipped": int((per_ep[:, i] > 0.5).sum())})
            if share[i] > CLIP_OK and hi[i] > lo[i]:
                bad(f"{kind} '{names[i]}': {share[i]:.0%} of the values lie outside the q01–q99 of meta/stats.json [{lo[i]:.5g}, {hi[i]:.5g}] and are clipped "
                    f"(true 1–99 % range [{tlo[i]:.5g}, {thi[i]:.5g}]; {int((per_ep[:, i] > 0.5).sum())} of {len(train)} training episodes more than half clipped)")
        R["scaling"][kind] = rows

    # ---- relative groups: min/max of (action − state) per chunk step over complete 40-step windows
    S, A = windows(d, train)
    R["relative"] = {"windows": int(len(S)), "horizon": NATIVE_H, "groups": []}
    for g in (g for g in G if g["relative"]):
        idx = g["indices"]; D = A[:, :, idx] - S[:, None, idx]
        mn, mx = D.min(0), D.max(0); q1, q99 = np.quantile(D, 0.01, axis=0), np.quantile(D, 0.99, axis=0)
        half = np.maximum((mx - mn) / 2, 1e-12); mid = (mx + mn) / 2
        typical = np.median(np.abs((D - mid) / half), axis=0)          # how much of [-1, 1] an ordinary value uses
        rows = []
        for k, i in enumerate(idx):
            row = {"name": an[i]}
            for h in sorted({0, min(chunk, NATIVE_H) - 1, NATIVE_H - 1}):
                row[f"step{h + 1}"] = {"min_max": [round(float(mn[h, k]), 4), round(float(mx[h, k]), 4)], "q01_q99": [round(float(q1[h, k]), 4), round(float(q99[h, k]), 4)],
                                       "typical_normalised": round(float(typical[h, k]), 3),
                                       "range_over_q_range": round(float((mx[h, k] - mn[h, k]) / max(q99[h, k] - q1[h, k], 1e-12)), 1)}
            rows.append(row)
        R["relative"]["groups"].append({"key": g["key"], "dims": rows})
        # quaternions are subtracted component by component: a sign flip between two frames is a jump of up to 2
        quat = [k for k, i in enumerate(idx) if an[i].split(".")[-1] in ("qw", "qx", "qy", "qz")]
        if quat:
            worst = float(np.abs(D[:, :, quat]).max())
            R["relative"]["max_quaternion_component_change"] = round(worst, 4)
            if worst > 0.7:
                bad(f"a quaternion component changes by {worst:.2f} within one chunk — a sign flip (q and −q are one rotation, their difference is not small): "
                    "the min/max of the relative statistics is set by it and every ordinary rotation is scaled to nothing")
        ratio = np.array([[r[f"step{min(chunk, NATIVE_H)}"]["range_over_q_range"] for r in rows]]).max()
        if ratio > 4:
            note(f"relative group '{g['key']}': min–max is up to {ratio:.1f}× the 1–99 % range at chunk step {min(chunk, NATIVE_H)} — rare large values set the scale "
                 "(LeRobot scales relative groups with min/max, not percentiles)")

    # ---- sampling: the tail of every episode is never an input
    drop = min(chunk, NATIVE_H) - 1
    lens = np.array([(d["ep"] == e).sum() for e in train])
    R["sampling"] = {"episodes_for_training": len(train), "frames": int(lens.sum()), "dropped_tail_frames_per_episode": drop,
                     "training_samples": int(np.maximum(lens - drop, 0).sum())}
    if d["progress"] is not None:
        after = {}
        for c in sorted({16, 24, 32, 40, chunk}):
            left = []
            for e in train:
                p = d["progress"][d["ep"] == e]
                if p[-1] < 0.99:
                    continue
                up = np.nonzero(np.diff(p, prepend=p[0]) > 1e-6)[0]
                last = int(up[-1]) if len(up) else len(p) - 1
                left.append(max(0, (len(p) - 1 - last) - (min(c, NATIVE_H) - 1)))
            after[c] = {"min": int(np.min(left)), "median": float(np.median(left)), "episodes_with_none": int(np.sum(np.asarray(left) == 0)), "total": int(np.sum(left))}
        R["sampling"]["input_frames_after_the_last_cut_by_chunk_size"] = after
        mine = after[chunk]
        if mine["episodes_with_none"]:
            bad(f"chunk_size {chunk}: {mine['episodes_with_none']} episodes have no input frame after their last cut (the tail of {drop} frames is dropped) — "
                "the model never sees 'the drilling is over' there; a smaller chunk_size keeps them")
        elif mine["median"] < 30:
            note(f"chunk_size {chunk}: a median of {mine['median']:.0f} input frames after the last cut ({mine['median'] / d['info']['fps']:.1f} s) — chunk_size 16 leaves {after[16]['median']:.0f}")

    # ---- the progress output and the level the loop ends the drilling at
    pg = next((g for g in G if not g["relative"] and "progress" in g["key"]), None)
    if pg is not None:
        i = pg["indices"][0]; s = st.get("action", {})
        top = float(s["q99"][i]) if "q99" in s else float(s["max"][i])
        x = d["action"][sel][:, i]
        R["progress"] = {"largest_value_the_model_can_output": round(top, 4), "end_level_in_use": end_level, "share_of_frames_at_or_above_the_level": None if end_level is None else round(float((x >= end_level).mean()), 4),
                         "share_of_frames_at_1": round(float((x >= 0.999).mean()), 4)}
        if end_level is not None and top < end_level + 0.03:
            bad(f"the progress output is clipped at {top:.3f} (q99 of meta/stats.json) — at or under the end level {end_level}: the policy could never end the drilling")
        if i < len(sn) and not np.allclose(d["state"][:, i], 0.0):
            note(f"state '{sn[i]}' is not constant 0 — at run time the loop sends 0 there (it does not know the true progress)")
    elif any("progress" in n for n in an):
        bad("the action has a progress value but 'progress' is not in --exclude: it would be made relative to the state's 16th value")

    # ---- language and video
    tasks = json.loads((Path(d["path"]) / "meta/tasks.json").read_text()) if (Path(d["path"]) / "meta/tasks.json").exists() else None
    if tasks is None:
        try:
            import pyarrow.parquet as pq
            t = pq.read_table(Path(d["path"]) / "meta/tasks.parquet").to_pydict()
            key = next(k for k in t if k != "task_index"); tasks = dict(zip(t["task_index"], t[key]))
        except Exception:  # noqa: BLE001
            tasks = {}
    cnt = {int(k): int((d["task"] == k).sum()) for k in np.unique(d["task"])}
    R["language"] = [{"task_index": k, "frames": v, "text": str(tasks.get(k, tasks.get(str(k), "?")))} for k, v in sorted(cnt.items())]
    v = next((f for f in d["info"]["features"].values() if f.get("dtype") == "video"), None)
    if v:
        R["video"] = {"shape": v.get("shape"), "codec": (v.get("info") or {}).get("video.codec"), "fps": d["info"]["fps"]}
        if (v.get("info") or {}).get("video.codec") == "av1":
            note("the videos are AV1: the training machine's decoder must read it (torchcodec / PyAV with libdav1d — the two trial trainings did)")
    return R


def lerobot_side(ds: Path, d: dict, chunk: int, exclude: list[str]) -> dict:
    """LeRobot's own view: the groups and modality config it derives, and its relative statistics against ours."""
    import lerobot
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.groot import processor_groot as PG
    from lerobot.policies.groot.configuration_groot import GrootConfig
    try:
        from lerobot.utils.feature_utils import dataset_to_policy_features
    except ImportError:                                           # where it lived before lerobot 0.6
        from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.configs.types import FeatureType
    meta = LeRobotDatasetMetadata("local/" + ds.name, root=str(ds))
    feats = dataset_to_policy_features(meta.features)
    cfg = GrootConfig(use_relative_actions=True, relative_exclude_joints=exclude, chunk_size=chunk, n_action_steps=chunk, device="cpu",
                      input_features={k: f for k, f in feats.items() if f.type is not FeatureType.ACTION},
                      output_features={k: f for k, f in feats.items() if f.type is FeatureType.ACTION})
    out = {"lerobot": lerobot.__version__, "action_delta_indices": [cfg.action_delta_indices[0], cfg.action_delta_indices[-1]], "drop_n_last_frames": cfg.drop_n_last_frames}
    stats = PG._make_relative_action_training_stats_from_dataset_meta(cfg, meta)
    assets = PG._build_n1_7_relative_action_processor_assets(cfg, stats, meta, base_assets=None)
    try:      # the task sentences as the model gets them (formalize_language: lower case, punctuation off)
        from lerobot.policies.groot.utils import prepare_n1_7_language_batch
        import pyarrow.parquet as pq
        t = pq.read_table(ds / "meta/tasks.parquet").to_pydict(); key = next(k for k in t if k != "task_index")
        out["language_as_the_model_gets_it"] = {str(x): prepare_n1_7_language_batch([str(x)], 1, formalize_language=assets.formalize_language)[0] for x in t[key]}
    except Exception as e:  # noqa: BLE001
        out["language_as_the_model_gets_it"] = f"not shown ({type(e).__name__}: {e})"
    out.update(modality_config=assets.modality_config, use_percentiles=assets.use_percentiles, valid_action_horizon=assets.valid_action_horizon,
               max_action_horizon=assets.max_action_horizon, clip_outliers=assets.clip_outliers, video_modality_keys=assets.video_modality_keys)
    S, A = windows(d, None)                                                # LeRobot computes them over the whole dataset
    rel = [g for g in groups_of(d["action_names"], exclude) if g["relative"]]
    idx = [i for g in rel for i in g["indices"]]
    D = A[:, :, idx] - S[:, None, idx]
    theirs_min = np.asarray(stats["action"]["min"])[:, idx]; theirs_max = np.asarray(stats["action"]["max"])[:, idx]
    out["round_trip"] = round_trip(PG, cfg, assets, d, chunk)
    out["relative_stats_agree_with_ours"] = {"max_abs_difference_min": float(np.abs(theirs_min - D.min(0)).max()), "max_abs_difference_max": float(np.abs(theirs_max - D.max(0)).max()),
                                             "windows_theirs": int(np.asarray(stats["action"]["count"])[0]), "windows_ours": int(len(S))}
    return out


def round_trip(PG, cfg, assets, d: dict, chunk: int, n: int = 256) -> dict:
    """The dataset's own actions through LeRobot's two steps that frame the model: GrootN17PackInputsStep (what the model
    is trained to output: relative to the state, scaled, padded to 132 values × 40 steps) and GrootN17ActionDecodeStep
    (what the server does with the model's answer). What comes back must be the dataset's absolute actions — a wrong
    group, a wrong state reference or a clipped value shows here, without a model."""
    import torch
    from lerobot.processor import batch_to_transition, policy_action_to_transition
    from lerobot.lerobot_types import TransitionKey
    pack = PG.GrootN17PackInputsStep(
        state_horizon=1, action_horizon=assets.max_action_horizon, valid_action_horizon=assets.valid_action_horizon, video_horizon=None,
        max_state_dim=cfg.max_state_dim, max_action_dim=cfg.max_action_dim, language_key="task", formalize_language=assets.formalize_language,
        embodiment_tag=cfg.embodiment_tag, embodiment_mapping=assets.embodiment_mapping, normalize_min_max=True, training=True, state_dropout_prob=0.0,
        stats=assets.stats, clip_outliers=assets.clip_outliers, video_modality_keys=assets.video_modality_keys, raw_stats=assets.raw_stats,
        use_percentiles=assets.use_percentiles, modality_config=assets.modality_config)
    decode = PG.GrootN17ActionDecodeStep(env_action_dim=len(d["action_names"]), raw_stats=assets.raw_stats, modality_config=assets.modality_config,
                                         use_percentiles=assets.use_percentiles, use_relative_action=True, pack_step=pack, action_decode_transform=None)
    nd = len(d["action_names"])
    pos = [i for i, nm in enumerate(d["action_names"]) if nm.split(".")[-1] in ("x", "y", "z")]
    quat = [i for i, nm in enumerate(d["action_names"]) if nm.split(".")[-1] in ("qw", "qx", "qy", "qz")]
    rest = [i for i in range(nd) if i not in pos + quat]
    pose_state = [i for i, nm in enumerate(d["state_names"]) if nm.split(".")[-1] in ("x", "y", "z", "qw", "qx", "qy", "qz")]

    def run(S, A, seed):
        pick = np.random.default_rng(seed).choice(len(S), size=min(n, len(S)), replace=False)
        batch = {"observation.state": torch.tensor(S[pick], dtype=torch.float32), "observation.images.endo": torch.zeros(len(pick), 3, 32, 32),
                 "action": torch.tensor(A[pick], dtype=torch.float32), "task": ["4 drill"] * len(pick), "action_is_pad": torch.zeros(len(pick), chunk, dtype=torch.bool)}
        tr = pack(batch_to_transition(batch))
        packed = tr[TransitionKey.ACTION]; st = tr[TransitionKey.OBSERVATION]["state"]
        back = decode(policy_action_to_transition(packed[:, :chunk]))[TransitionKey.ACTION].numpy()
        return pick, tr, packed, st, np.abs(back - A[pick])
    # (a) frames that have a complete NATIVE window (40 steps): the windows LeRobot's relative statistics are made of
    S40, A40 = windows(d, None, horizon=NATIVE_H)
    pick, tr, packed, st, err = run(S40, A40[:, :chunk], 0)
    lang = tr[TransitionKey.COMPLEMENTARY_DATA].get("language")
    out = {"samples": int(len(pick)), "packed_action_shape": list(packed.shape), "packed_state_shape": list(st.shape),
           "packed_action_abs_max_in_valid_part": float(packed[:, :chunk, :nd].abs().max()),
           "packed_beyond_valid_is_zero": bool(all(x.numel() == 0 or float(x.abs().max()) == 0 for x in (packed[:, chunk:], packed[:, :, nd:]))),
           "pose_state_values_on_the_clip_limit": float((st[:, 0, pose_state].abs() >= 1.0).float().mean()),
           "language_as_the_model_gets_it": lang[0] if isinstance(lang, (list, tuple)) else str(lang)[:80],
           "max_error_position_mm": float(err[:, :, pos].max()), "max_error_quaternion": float(err[:, :, quat].max()),
           "max_error_other": {d["action_names"][i]: float(err[:, :, i].max()) for i in rest}}
    out["exact"] = bool(out["max_error_position_mm"] < 0.01 and out["max_error_quaternion"] < 1e-4 and all(v < 1e-3 for v in out["max_error_other"].values()))
    # (b) with chunk_size < 40 the trainer also samples the frames whose window is complete for the chunk but not for 40
    # steps: the last frames of every episode. They are in no statistic, and a move larger than any in the statistics is
    # clipped in the target (the tools leaving after the last cut are the fastest moves of an episode)
    if chunk < NATIVE_H:
        St, At = [], []
        for e in np.unique(d["ep"]):
            m = np.nonzero(d["ep"] == e)[0]
            lo, hi = max(0, len(m) - NATIVE_H + 1), len(m) - chunk + 1           # starts with a chunk-complete, not 40-complete window
            if hi > lo:
                idx = m[0] + np.arange(lo, hi)[:, None] + np.arange(chunk)[None, :]
                St.append(d["state"][m[lo:hi]]); At.append(d["action"][idx])
        St, At = np.concatenate(St), np.concatenate(At)
        _, _, _, _, errt = run(St, At, 1)
        out["tail_windows"] = {"per_episode": NATIVE_H - chunk, "samples": int(min(n, len(St))), "share_with_a_clipped_position_over_0.1mm": float((errt[:, :, pos].max((1, 2)) > 0.1).mean()),
                               "max_clipped_mm": float(errt[:, :, pos].max()), "max_error_other": {d["action_names"][i]: float(errt[:, :, i].max()) for i in rest}}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--chunk-size", type=int, default=NATIVE_H, help="--policy.chunk_size of the training (LeRobot's default: 40, the model's own horizon)")
    ap.add_argument("--exclude", default="trigger,progress", help="relative_exclude_joints, comma separated")
    ap.add_argument("--held-out", default="2,18", help="episodes kept out of the training (--dataset.episodes lists the others)")
    ap.add_argument("--end-level", type=float, default=None, help="the level at which the robot loop ends the drilling (default: the loop's own, 0.91)")
    ap.add_argument("--lerobot", action="store_true", help="also run LeRobot's own functions on the dataset and compare (minutes)")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args(argv)
    ds = Path(a.dataset).expanduser(); d = load(ds); d["path"] = str(ds)
    exclude = [x for x in a.exclude.split(",") if x]; held = [int(x) for x in a.held_out.split(",") if x != ""]
    level = a.end_level
    if level is None:
        try:
            import rollout as RO
            level = RO.END_LEVEL["progress"]
        except Exception:  # noqa: BLE001 — alone on a training machine (no robot code beside it): the level of 10-09
            level = 0.91
    R = report(d, a.chunk_size, exclude, held, level)
    if a.lerobot:
        R["lerobot_side"] = L = lerobot_side(ds, d, a.chunk_size, exclude)
        agree = L["relative_stats_agree_with_ours"]
        if max(agree["max_abs_difference_min"], agree["max_abs_difference_max"]) > 1e-3 or agree["windows_theirs"] != agree["windows_ours"]:
            R["problems"].append(f"LeRobot's relative statistics differ from this tool's ({agree}) — the report above does not describe what the trainer will use")
        if not L["round_trip"]["exact"]:
            R["problems"].append(f"the dataset's actions do not come back through LeRobot's pack → decode steps: {L['round_trip']}")
        t = L["round_trip"].get("tail_windows")
        if t and t["share_with_a_clipped_position_over_0.1mm"] > 0.02:
            R["notes"].append(f"chunk_size {a.chunk_size} < 40: the last {t['per_episode']} sampled frames of each episode are in none of LeRobot's relative statistics (they are made of "
                              f"complete 40-step windows); {t['share_with_a_clipped_position_over_0.1mm']:.0%} of them have a position target clipped (up to {t['max_clipped_mm']:.1f} mm) — "
                              "the fast moves out of the body after the last cut. The progress and trigger targets of those frames are exact")
        if not L["use_percentiles"]:
            R["notes"].append("LeRobot scales with min/max (no q01/q99 in the statistics it found)")
    train_eps = [e for e in sorted(set(int(x) for x in d["ep"])) if e not in held]
    R["command"] = ("lerobot-train --policy.type=groot --policy.use_relative_actions=true "
                    f"'--policy.relative_exclude_joints={json.dumps(exclude)}' --policy.chunk_size={a.chunk_size} --policy.n_action_steps={a.chunk_size} "
                    f"--dataset.repo_id=local/{ds.name} --dataset.root=<path>/{ds.name} '--dataset.episodes={json.dumps(train_eps)}' …")
    if a.out:
        Path(a.out).write_text(json.dumps(R, indent=1, ensure_ascii=False))
    print(f"{ds.name}: {R['episodes']} episodes, {R['frames']} frames | groups: " + ", ".join(f"{g['key']}[{len(g['indices'])}{' rel' if g['relative'] else ' abs'}]" for g in R["groups"]))
    s = R["sampling"]; print(f"training samples {s['training_samples']} of {s['frames']} frames ({s['dropped_tail_frames_per_episode']} tail frames an episode dropped)")
    if "input_frames_after_the_last_cut_by_chunk_size" in s:
        print("input frames after the last cut (min · median per episode): " + " | ".join(f"chunk {c}: {v['min']} · {v['median']:.0f}" for c, v in s["input_frames_after_the_last_cut_by_chunk_size"].items()))
    if "progress" in R:
        print("progress: " + json.dumps(R["progress"]))
    if "lerobot_side" in R:
        L = R["lerobot_side"]; rt = L["round_trip"]
        print(f"LeRobot {L['lerobot']}: answers {L['valid_action_horizon']} rows (decodes up to {L['max_action_horizon']}), percentiles {L['use_percentiles']}, clip {L['clip_outliers']}, "
              f"video keys {L['video_modality_keys']}; its relative statistics agree with this tool's to {max(L['relative_stats_agree_with_ours']['max_abs_difference_min'], L['relative_stats_agree_with_ours']['max_abs_difference_max']):.1e}")
        print(f"pack → decode on {rt['samples']} samples: position error ≤ {rt['max_error_position_mm']:.2e} mm, quaternion ≤ {rt['max_error_quaternion']:.1e}, "
              + ", ".join(f"{k} ≤ {v:.1e}" for k, v in rt["max_error_other"].items()) + f"; pose state values on the clip limit {rt['pose_state_values_on_the_clip_limit']:.1%}; language: {rt['language_as_the_model_gets_it']!r}")
        if isinstance(L.get("language_as_the_model_gets_it"), dict):
            for k, v in L["language_as_the_model_gets_it"].items():
                print(f"  task {k!r} → {v!r}")
        if "tail_windows" in rt:
            t = rt["tail_windows"]
            print(f"the last {t['per_episode']} sampled frames of each episode (no 40-step window, in no relative statistic): {t['share_with_a_clipped_position_over_0.1mm']:.0%} have a position target "
                  f"clipped by more than 0.1 mm (up to {t['max_clipped_mm']:.1f} mm); " + ", ".join(f"{k} ≤ {v:.1e}" for k, v in t["max_error_other"].items()))
    for n in R["notes"]:
        print("note:", n)
    for p in R["problems"]:
        print("PROBLEM:", p)
    print("no problem found" if not R["problems"] else f"{len(R['problems'])} problem(s)")
    return 1 if R["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
