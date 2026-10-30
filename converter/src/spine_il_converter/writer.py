"""Write included episodes as a LeRobot v3 dataset plus a provenance sidecar under meta/spine_il/."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from . import __version__, schema
from .episode import ConvertConfig, EpisodeResult
from .features import dataset_features
from .splits import SPLIT_NAMES, assign_splits

SIDECAR_DIR = Path("meta") / "spine_il"


def write_dataset(
    results: list[EpisodeResult],
    out_dir: Path,
    repo_id: str,
    cfg: ConvertConfig = ConvertConfig(),
    seed: int = 0,
    overwrite: bool = False,
) -> dict:
    if any(getattr(r.episode, "format", None) == "rec2" for r in results if r.included):
        from .writer_v2 import write_rec_dataset

        return write_rec_dataset(results, out_dir, repo_id, cfg, seed=seed, overwrite=overwrite)
    from lerobot.datasets.io_utils import write_info
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out_dir = Path(out_dir)
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists (use overwrite=True / --overwrite for a full rebuild)")
        shutil.rmtree(out_dir)

    included = [r.episode for r in results if r.included]
    if not included:
        raise ValueError("no episode passed the filters — nothing to write")
    split_of = assign_splits([ep.session_id for ep in included], seed=seed)
    # Contiguous episode ranges per split so meta/info.json `splits` can express them.
    included.sort(key=lambda ep: (SPLIT_NAMES.index(split_of[ep.session_id]), ep.session_id))

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=cfg.fps,
        features=dataset_features(cfg.image_size),
        root=out_dir,
        robot_type=schema.ROBOT_TYPE,
        use_videos=True,
        image_writer_threads=4,
    )
    records = []
    for ep_index, ep in enumerate(included):
        for k in range(ep.n_frames):
            frame = {
                "observation.state": ep.state[k],
                "action": ep.action[k],
                "observation.phase": ep.phase[k],
                "observation.exposure": ep.exposure[k],
                "observation.progress": ep.progress[k],
                "observation.goal": ep.goal,
                "observation.tracker_time": ep.tracker_time[k],
                "task": ep.task,
            }
            for cam in schema.CAMS:
                frame[f"observation.images.{cam}"] = _load_rgb(ep.image_paths[cam][k], cfg.image_size)
            ds.add_frame(frame)
        ds.save_episode()
        records.append(
            {
                "episode_index": ep_index,
                "session_id": ep.session_id,
                "source": str(ep.source),
                "split": split_of[ep.session_id],
                "t0_src_ms": ep.t0_src_ms,
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

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "converter_version": __version__,
        "repo_id": repo_id,
        "config": {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in asdict(cfg).items()},
        "seed": seed,
        "splits": bounds,
        "split_episodes": {name: [r["episode_index"] for r in records if r["split"] == name] for name in SPLIT_NAMES},
        "included": len(records),
        "excluded": [
            {"session_id": r.session_id, "source": str(r.source), "reason": r.reason, "detail": r.detail}
            for r in results
            if not r.included
        ],
    }
    sidecar = out_dir / SIDECAR_DIR
    sidecar.mkdir(parents=True, exist_ok=True)
    with (sidecar / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (sidecar / "conversion_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _load_rgb(path: Path, image_hw: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    if arr.shape[:2] != tuple(image_hw):
        raise ValueError(f"{path}: image is {arr.shape[1]}x{arr.shape[0]}, expected {image_hw[1]}x{image_hw[0]}")
    return arr
