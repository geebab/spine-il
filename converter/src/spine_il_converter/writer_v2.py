"""Write recorder v2 episodes as a LeRobot v3 dataset (no image streams yet) plus provenance sidecars.

Sidecars under meta/spine_il/:
  episodes.jsonl                         one record per LeRobot episode (source, split, piece, t0_ms, stats)
  conversion_report.json                 config, splits, exclusions, dataset-level warnings
  recordings/<sessionId>/hdr.json        the recording's hdr row, verbatim
  recordings/<sessionId>/cut.jsonl       every `cut` row of the recording, verbatim, in r order (full-precision
                                         stamp centres — re-applying them in order reproduces the carve)
A cut row's frame: k = ceil((cut.tExp − t0_ms) / (1000 / fps)), i.e. the frame whose interval (t_{k−1}, t_k]
holds the stamp's camera frame. Pieces of one recording (--split-on-gap) share its split and its sidecar folder.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from . import __version__, schema
from .episode import ConvertConfig, EpisodeResult
from .episode_v2 import MASK_NAMES, REC_STATE_NAMES, episodes_of
from .recorder_v2 import is_safe_session_id
from .splits import SPLIT_NAMES, assign_splits
from .writer import SIDECAR_DIR


def rec_dataset_features() -> dict:
    vec = lambda names: {"dtype": "float32", "shape": (len(names),), "names": list(names)}  # noqa: E731
    return {
        "observation.state": vec(REC_STATE_NAMES),
        "action": vec(REC_STATE_NAMES),
        "observation.endo_roll_deg": vec(("roll_deg",)),
        "observation.tracker_time": vec(("t_s",)),
        # not policy inputs: LeRobot only maps observation.* / action* keys to policy features
        "label.drill_cut": vec(("drill_cut",)),
        "mask": vec(MASK_NAMES),
        "diag.tsrc_minus_texp_ms": vec(("endo", "drill")),
    }


def write_rec_dataset(
    results: list[EpisodeResult],
    out_dir: Path,
    repo_id: str,
    cfg: ConvertConfig = ConvertConfig(),
    seed: int = 0,
    overwrite: bool = False,
) -> dict:
    from lerobot.datasets.io_utils import write_info
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    results = _drop_duplicate_sessions(results)
    recordings = [r.episode for r in results if r.included]
    if any(getattr(ep, "format", None) != "rec2" for ep in recordings):
        raise ValueError("recorder v2 episodes cannot share a dataset with PLAN-0 fixture episodes")
    if not recordings:
        raise ValueError("no episode passed the filters — nothing to write")
    if any(not is_safe_session_id(ep.session_id) for ep in recordings):  # the loader rejects these; never write them
        raise ValueError("unsafe sessionId among the episodes")
    out_dir = Path(out_dir)
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists (use overwrite=True / --overwrite for a full rebuild)")
        shutil.rmtree(out_dir)
    split_of = assign_splits([ep.session_id for ep in recordings], seed=seed)  # pieces share their recording's split
    episodes = [p for ep in recordings for p in episodes_of(ep)]
    episodes.sort(key=lambda p: (SPLIT_NAMES.index(split_of[p.session_id]), p.session_id, p.piece or 0))

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=cfg.fps,
        features=rec_dataset_features(),
        root=out_dir,
        robot_type=schema.ROBOT_TYPE,
        use_videos=False,
    )
    records = []
    for ep_index, ep in enumerate(episodes):
        for k in range(ep.n_frames):
            ds.add_frame(
                {
                    "observation.state": ep.state[k],
                    "action": ep.action[k],
                    "observation.endo_roll_deg": ep.endo_roll[k],
                    "observation.tracker_time": ep.tracker_time[k],
                    "label.drill_cut": ep.drill_cut[k],
                    "mask": ep.mask[k],
                    "diag.tsrc_minus_texp_ms": ep.t_src_lag[k],
                    "task": ep.task,
                }
            )
        ds.save_episode()
        records.append(
            {
                "episode_index": ep_index,
                "session_id": ep.session_id,
                "source": str(ep.source),
                "split": split_of[ep.session_id],
                "time_axis": ep.time_axis,
                "t0_ms": ep.t0_ms,
                "recording": str(SIDECAR_DIR / "recordings" / ep.session_id),
                **ep.stats,
            }
        )
    ds.finalize()

    counts = {name: sum(r["split"] == name for r in records) for name in SPLIT_NAMES}
    bounds, start = {}, 0
    for name in SPLIT_NAMES:
        if counts[name]:
            bounds[name] = f"{start}:{start + counts[name]}"
        start += counts[name]
    ds.meta.info.splits = bounds
    write_info(ds.meta.info, ds.root)

    sidecar = out_dir / SIDECAR_DIR
    for ep in recordings:
        first = episodes_of(ep)[0]
        folder = sidecar / "recordings" / ep.session_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "hdr.json").write_text(first.header_line + "\n", encoding="utf-8")
        (folder / "cut.jsonl").write_text("".join(line + "\n" for line in first.cut_lines), encoding="utf-8")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "converter_version": __version__,
        "format": "rec2",
        "repo_id": repo_id,
        "config": {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in asdict(cfg).items()},
        "seed": seed,
        "splits": bounds,
        "split_episodes": {name: [r["episode_index"] for r in records if r["split"] == name] for name in SPLIT_NAMES},
        "included": len(records),
        "recordings": len(recordings),
        "warnings": _model_warnings(episodes),
        "excluded": [
            {"session_id": r.session_id, "source": str(r.source), "reason": r.reason, "detail": r.detail}
            for r in results
            if not r.included
        ],
    }
    with (sidecar / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (sidecar / "conversion_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _drop_duplicate_sessions(results: list[EpisodeResult]) -> list[EpisodeResult]:
    """Two files with one sessionId would share a sidecar folder and a split key: keep the first (by path)."""
    seen, out = set(), []
    for r in sorted(results, key=lambda r: str(r.source)):
        if r.included and r.session_id in seen:
            out.append(EpisodeResult(r.session_id, r.source, None, "duplicate_session", "sessionId already converted"))
            continue
        if r.included:
            seen.add(r.session_id)
        out.append(r)
    return out


def _model_warnings(episodes) -> list[str]:
    """Different tracker tool models per tool across a dataset put different marker frames (tool local +z, hence
    the rot6d x axis) into one dataset."""
    out = []
    for tool in ("endo", "drill"):
        models = sorted({str(ep.stats["identity"]["tool_models"][tool]["sha12"]
                             or ep.stats["identity"]["tool_models"][tool]["relay_sha12"]) for ep in episodes})
        if len(models) > 1:
            out.append(f"{tool}: the dataset mixes tool models {models} — check the marker frame matches across them")
    return out
