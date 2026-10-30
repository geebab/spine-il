"""spine-il-convert: validate recorder sessions, build a LeRobot v3 dataset, or write synthetic fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .episode import ConvertConfig, build_episode
from .session import load_session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spine-il-convert", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="check session files against the v2 contract and the episode filters")
    p.add_argument("sessions", nargs="+", type=Path)
    p.add_argument("--data-root", type=Path, help="directory holding sessions/ and frames/ (default: inferred)")
    _add_filter_args(p)

    p = sub.add_parser("build", help="convert every session under DATA_ROOT/sessions into a LeRobot v3 dataset")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repo-id", required=True, help="e.g. <hf_user>/spine-lam-lt45-v1")
    p.add_argument("--seed", type=int, default=0, help="split seed")
    p.add_argument("--overwrite", action="store_true", help="delete OUT and rebuild from scratch")
    p.add_argument("--replay-verdicts", type=Path, help="parquet with columns session_id, feasible (PLAN-0 §8.3)")
    _add_filter_args(p)

    p = sub.add_parser("make-fixtures", help="write synthetic v2 sessions (3 includable + 8 rejectable)")
    p.add_argument("out", type=Path)
    p.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)
    return {"validate": _validate, "build": _build, "make-fixtures": _make_fixtures}[args.command](args)


def _add_filter_args(p: argparse.ArgumentParser) -> None:
    d = ConvertConfig()
    p.add_argument("--max-gap-ms", type=float, default=d.max_gap_ms)
    p.add_argument("--align-tol-ms", type=float, default=d.align_tol_ms)
    p.add_argument("--max-lost-frac", type=float, default=d.max_lost_frac)
    p.add_argument("--min-frames", type=int, default=d.min_frames)
    p.add_argument("--allow-core", action="store_true", help="recorder v2: include timingSpec 'v2-core' files (tSrc axis)")
    p.add_argument("--allow-robot-tools", action="store_true",
                   help="recorder v2: include files whose tracker also ran robot-mounted marker blocks (endo2)")
    p.add_argument("--task", help="recorder v2: task text (default: hdr diagnosis/approach, else a generic one)")
    p.add_argument("--max-masked-frac", type=float, default=d.max_masked_frac,
                   help="recorder v2: per tool, masked frames allowed inside an episode")
    p.add_argument("--split-on-gap", action="store_true",
                   help="recorder v2: cut a recording at masked runs > --max-gap-ms into pieces (never bridged)")


def _config(args, replay_infeasible=frozenset()) -> ConvertConfig:
    return ConvertConfig(
        max_gap_ms=args.max_gap_ms,
        align_tol_ms=args.align_tol_ms,
        max_lost_frac=args.max_lost_frac,
        min_frames=args.min_frames,
        replay_infeasible=frozenset(replay_infeasible),
        allow_core=args.allow_core,
        allow_robot_tools=args.allow_robot_tools,
        task=args.task,
        max_masked_frac=args.max_masked_frac,
        split_on_gap=args.split_on_gap,
    )


def _infer_root(session_path: Path) -> Path:
    for parent in session_path.resolve().parents:
        if parent.name == "sessions":
            return parent.parent
    return session_path.resolve().parent


def _validate(args) -> int:
    cfg = _config(args)
    worst = 0
    for path in args.sessions:
        session = load_session(path)
        result = build_episode(session, args.data_root or _infer_root(path), cfg)
        print(f"== {path}  (sessionId={session.session_id})")
        for issue in session.issues:
            print(f"   {issue}")
        if result.included:
            print(f"   INCLUDED  {json.dumps(result.episode.stats, ensure_ascii=False)}")
        else:
            print(f"   EXCLUDED  reason={result.reason}  {result.detail}")
        worst = max(worst, 2 if session.errors else (0 if result.included else 1))
    return worst


def _build(args) -> int:
    from .writer import write_dataset

    infeasible: set[str] = set()
    if args.replay_verdicts:
        import pandas as pd

        df = pd.read_parquet(args.replay_verdicts)
        infeasible = set(df.loc[~df["feasible"].astype(bool), "session_id"].astype(str))
    cfg = _config(args, infeasible)
    paths = sorted((args.data_root / "sessions").rglob("*.jsonl"))
    if not paths:
        print(f"no *.jsonl under {args.data_root / 'sessions'}", file=sys.stderr)
        return 2
    results = [build_episode(load_session(p), args.data_root, cfg) for p in paths]
    for r in results:
        print(f"{'+' if r.included else '-'} {r.session_id:<32} {r.reason or ''} {r.detail}")
    report = write_dataset(results, args.out, args.repo_id, cfg, seed=args.seed, overwrite=args.overwrite)
    for warning in report.get("warnings", []):
        print(f"warning: {warning}", file=sys.stderr)
    print(f"wrote {report['included']} episodes to {args.out}  splits={report['splits']}")
    print(f"excluded {len(report['excluded'])}; details in {args.out}/meta/spine_il/conversion_report.json")
    return 0


def _make_fixtures(args) -> int:
    from .fixtures import write_standard_set

    for spec, path in write_standard_set(args.out, seed=args.seed):
        print(f"{path}  expected: {spec.expected_reason or 'included'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
