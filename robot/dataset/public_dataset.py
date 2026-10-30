#!/usr/bin/env python3
"""Public copy of an exported LeRobot dataset: the operator's name and the local paths out of its side metadata (user 10-23
via robot-mac: the recordings' operator is "surgeon1" in public; Open-H exports do the same through openh.scrub). The
dataset itself is only read: the copy hardlinks every file except the two rewritten ones, whose links are removed first
(never written through).

  meta/spine_il/episodes.jsonl   operator → --operator, recording → the file name (<session>.jsonl, as in the public
                                 recordings' <date>/ folders)
  meta/spine_il/export.json      the home folder → "~"

    python public_dataset.py --dataset DS --out DIR [--operator surgeon1]   → DIR + DIR/../<name>.public.json (grep)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

SIDE, EXPORT = "meta/spine_il/episodes.jsonl", "meta/spine_il/export.json"
LOCAL = re.compile(r"/home/|/Users/|/srv/", re.I)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--operator", default="surgeon1")
    a = ap.parse_args()
    src, out = Path(a.dataset).expanduser().resolve(), Path(a.out).expanduser().resolve()
    if out.exists():
        raise SystemExit(f"{out} exists")
    shutil.copytree(src, out, copy_function=os.link)
    ops: set = set()
    lines = (src / SIDE).read_text().splitlines()
    (out / SIDE).unlink()
    with open(out / SIDE, "w") as fo:
        for line in lines:
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("operator"):
                ops.add(e["operator"]); e["operator"] = a.operator
            if e.get("recording"):
                e["recording"] = Path(e["recording"]).name
            fo.write(json.dumps(e, ensure_ascii=False) + "\n")
    txt = (src / EXPORT).read_text()
    (out / EXPORT).unlink()
    (out / EXPORT).write_text(txt.replace(str(Path.home()), "~"))
    names, hits = ops, {}
    for f in sorted((out / "meta").rglob("*")):
        if f.is_file() and f.suffix in (".json", ".jsonl", ".md"):
            t = f.read_text()
            n = len(LOCAL.findall(t)) + sum(t.lower().count(x.lower()) for x in names)
            if n:
                hits[str(f.relative_to(out))] = n
    rep = {"dataset": str(src), "out": str(out), "episodes": len(lines), "operator": a.operator,
           "linked_to_source": {p: (out / p).stat().st_ino == (src / p).stat().st_ino for p in (SIDE, EXPORT)},
           "meta_hits_local_path_or_old_operator": hits}
    (out.parent / f"{out.name}.public.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
