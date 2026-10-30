#!/usr/bin/env python3
"""Public copies of a dataset's hand recordings, laid out as <date>/<session>.jsonl (eval/skin_surface.py --recordings
reads that folder). The originals in viewer-data/sessions are only read; every row is kept as recorded, the skin crossings
(`drawn.entry`, `real.entry`) included (user 10-23 via robot-mac: the crossings stay public, the skin STL does not; the
first version of this script removed them). `--operator NAME` replaces the header's operator name. Keys whose name
contains "entry" or "skin" are listed in the report.

    python public_recordings.py --dataset DS --out DIR [--operator NAME]
                                                    → DIR/sessions/<date>/<session>.jsonl + DIR/public_recordings.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SUSPECT = re.compile(r"entry|skin", re.I)


def recordings_of(ds: Path) -> list[Path]:
    out = []
    for line in (ds / "meta/spine_il/episodes.jsonl").read_text().splitlines():
        m = json.loads(line) if line.strip() else {}
        if m.get("source") == "hand" and Path(m["recording"]) not in out:
            out.append(Path(m["recording"]))
    return out


def _suspects(o, pre: str, found: set) -> None:
    if isinstance(o, dict):
        for k, v in o.items():
            p = f"{pre}.{k}" if pre else k
            if SUSPECT.search(k):
                found.add(p)
            _suspects(v, p, found)
    elif isinstance(o, list):
        for v in o[:4]:
            _suspects(v, pre + "[]", found)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--operator", default=None, help="the header's operator name in the copies (default: as recorded)")
    a = ap.parse_args()
    out = Path(a.out).expanduser(); recs = recordings_of(Path(a.dataset).expanduser())
    left, files = set(), []
    for src in recs:
        dst = out / "sessions" / src.parent.name / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        h_src = hashlib.sha256(); n = 0
        with open(src, "rb") as fi, open(dst, "w") as fo:
            for raw in fi:
                h_src.update(raw)
                r = json.loads(raw)
                if a.operator is not None and r.get("k") == "hdr" and "operator" in r:
                    r["operator"] = a.operator
                _suspects(r, r.get("k", ""), left)
                fo.write(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n"); n += 1
        files.append({"session": src.stem, "date": src.parent.name, "rows": n, "source_sha256": h_src.hexdigest(),
                      "public_sha256": hashlib.sha256(dst.read_bytes()).hexdigest()})
    rep = {"dataset": str(a.dataset), "recordings": len(files), "operator": a.operator,
           "keys_matching_entry_or_skin": sorted(left), "files": files}
    (out / "public_recordings.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "files"}, indent=1))


if __name__ == "__main__":
    main()
