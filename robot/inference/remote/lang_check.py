"""Does the model listen to the words? (2026-10-13) — the same picture and tool poses, asked with other instructions.

User, 10-13: "vla의 언어를 활용하기 위한 방안을 모색하라". Before and after a training with the goal in words (dataset/
instructions.py, set lang1) this measures how far an answer moves when ONLY the instruction changes, on demonstrations the
model was not trained on, against how far it moves when nothing changes (the same question asked again: the model samples).

  python inference/remote/lang_check.py NAME --dataset DS --episodes 2,18 [-o out.json]

Per held-out episode: one frame of the approach and frames at 10 / 30 / 50 / 70 / 90 % of the drilling. Per frame:
  again      the right instruction once more                       → the noise floor
  wording    the same goal in the set's other wordings             → should stay at the floor (the meaning, not the sentence)
  extent     the other extent (ipsilateral only / with contralateral; lang2: and the sentence that names none), same level
             → should move where the extents part ways, late in the drilling
  level      the other levels (L3 / L4 / L5), same extent          → should move from the approach on (untrained until such recordings exist)
  legacy     the other set's text for the phase · none (empty) · unrelated ("Open the drawer.")
Measured against the answer to the right instruction, at row 16 (1 s ahead): the drill tip's distance (mm), the endoscope
tip's, the progress output's difference; and the drill tip's mean distance over the 16 rows.

A model of the legacy phase texts has no goal: for it the lang1 sentences are the "other set", and extent / level are asked
in lang1's words — its answers to them say how much an untrained sentence moves it (the baseline for the comparison).
Nothing here moves an arm; a few hundred requests.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import connect_check as CC     # noqa: E402
import endpoint as EP          # noqa: E402
import remote_policy as RP     # noqa: E402

INS = RP.INS
UNRELATED = "Open the drawer."
KINDS = ("again", "wording", "extent", "level", "legacy", "none", "unrelated")


def pick(eps: dict, episodes: list[int]) -> list[tuple[int, int, str, int]]:
    """(episode, frame, label, phase) — one approach frame, five through the drilling."""
    out = []
    for e in episodes:
        d = eps[e]; n = len(d["progress"]); fc, lc = int(d["first_cut"]), int(d["last_cut"])
        out.append((e, int(min(max(0, fc - 30), n - 2)), "approach", 3))
        out += [(e, int(min(fc + (lc - fc) * q // 100, n - 2)), f"drilling {q} %", 4) for q in (10, 30, 50, 70, 90)]
    return out


def variants(name: str, phase: int, goal: dict) -> list[tuple[str, str, str]]:
    """(kind, what, text) asked beside the right instruction of a frame."""
    lv, ex = goal["level"], goal["extent"]
    words = name if name in ("lang1", "lang2", "lang3") else "lang1"     # a legacy model is asked in lang1's words
    lang = INS.wordings(phase, lv, ex)     # lang2's three for a lang3 model too (its first three): ratios stay comparable
    if name == "phase1":
        right, other_set = RP.TASKS[phase], lang[0]
    else:
        right, other_set = lang[0], RP.TASKS[phase]
    v = [("again", "the same", right)]
    if name != "phase1":
        v += [("wording", f"wording {i + 1}", t) for i, t in enumerate(lang[1:], 1)]
    if phase == 4:   # lang2: the sentence that names no extent too
        v += [("extent", x, INS.text(words, 4, lv, x)) for x in INS.extents(words) if x != ex or name == "phase1"]
    v += [("level", l, INS.text(words, phase, l, ex)) for l in INS.LEVELS if l != lv or name == "phase1"]
    return v + [("legacy" if name != "phase1" else "lang1", "the other set", other_set), ("none", "empty", ""), ("unrelated", UNRELATED, UNRELATED)]


def diff(ref: np.ndarray, got: np.ndarray, k: int = 15) -> dict:
    n = min(len(ref), len(got)); k = min(k, n - 1)
    d = {"drill_mm": float(np.linalg.norm(got[k, 7:10] - ref[k, 7:10])), "endo_mm": float(np.linalg.norm(got[k, 0:3] - ref[k, 0:3])),
         "drill_path_mm": float(np.linalg.norm(got[:k + 1, 7:10] - ref[:k + 1, 7:10], axis=1).mean())}
    if ref.shape[1] > 15 and got.shape[1] > 15:
        d["progress"] = float(got[k, 15] - ref[k, 15])
    return d


def summarise(rows: list[dict]) -> dict:
    """kind → stage (approach / early = drilling ≤ 30 % / late) → medians; and the verdict against the noise floor."""
    out: dict = {}
    for r in rows:
        stage = "approach" if r["label"] == "approach" else "early" if r["label"] in ("drilling 10 %", "drilling 30 %") else "late"
        out.setdefault(r["kind"], {}).setdefault(stage, []).append(r)
    tab = {k: {s: {"n": len(v), "drill_mm": round(float(np.median([x["drill_mm"] for x in v])), 2), "drill_mm_max": round(float(np.max([x["drill_mm"] for x in v])), 2),
                   "endo_mm": round(float(np.median([x["endo_mm"] for x in v])), 2), "drill_path_mm": round(float(np.median([x["drill_path_mm"] for x in v])), 2),
                   **({"progress_abs": round(float(np.median([abs(x["progress"]) for x in v])), 3),
                       "progress_mean": round(float(np.mean([x["progress"] for x in v])), 3)} if all("progress" in x for x in v) else {})}
               for s, v in st.items()} for k, st in out.items()}
    floor = max([s["drill_mm"] for s in (tab.get("again") or {}).values()] or [0.0])
    hear = {k: round(max(s["drill_mm"] for s in st.values()) / max(floor, 0.05), 1) for k, st in tab.items() if k != "again"}
    return {"table": tab, "noise_floor_drill_mm": round(floor, 2), "times_the_floor": hear}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", default=None, help="an endpoint's name or http(s)://host:port")
    ap.add_argument("--name", default=None); ap.add_argument("--token-file", default=None)
    ap.add_argument("--dataset", required=True); ap.add_argument("--episodes", required=True, help="episodes the model was not trained on")
    ap.add_argument("--timeout-s", type=float, default=60.0); ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args(argv)
    import end_level as EL
    ep = EP.resolve(a.target, a.name)
    rp = RP.RemotePolicy(ep["url"], RP.read_token(a.token_file or ep["token_file"]), a.timeout_s, retries=0, run="lang_check", ca_file=ep.get("ca_file"))
    info = rp.info(); name = (info.get("instructions") or {}).get("name") or "phase1"
    ds = Path(a.dataset).expanduser(); eps = EL.load(ds); episodes = [int(x) for x in a.episodes.split(",") if x]
    qs = pick(eps, episodes); frames = CC.load_frames(ds, [(e, k) for e, k, _, _ in qs])
    rows = []
    for e, k, label, ph in qs:
        g0 = eps[e].get("goal") or {}
        goal = {"level": "L4", "extent": INS.DEFAULT_EXTENT, **{x: y for x, y in g0.items() if x in ("level", "extent") and y}}
        if goal["extent"] == INS.NEUTRAL and name != "lang2":   # a lang2 dataset's neutral recording, asked of an older model: its lang1 class
            goal["extent"] = "contralateral" if (g0.get("past_midline_mm") or 0.0) > INS.CROSS_MM else "ipsilateral"
        st = eps[e]["state"][k]; right = rp.task(ph, goal if name != "phase1" else None)
        ref = np.asarray(rp.act(frames[(e, k)], st, right), float)
        for kind, what, text in variants(name, ph, goal):
            got = np.asarray(rp.act(frames[(e, k)], st, text), float)
            rows.append({"episode": e, "frame": k, "label": label, "kind": kind, "what": what, "text": text, **diff(ref, got)})
        print(f"episode {e} frame {k} ({label}, goal {goal['level']} {goal['extent']}): " + ", ".join(
            f"{r['kind']} {r['what']} {r['drill_mm']:.1f}" for r in rows[-len(variants(name, ph, goal)):]) + "  (drill tip mm, 1 s ahead)", flush=True)
    res = {"model": info.get("model"), "instructions": name, "dataset": str(ds), "episodes": episodes, "rows": rows, **summarise(rows)}
    print(f"\n{info.get('model')} — instructions {name}; the same question again moves the drill tip {res['noise_floor_drill_mm']} mm (the floor)")
    for kind in KINDS + ("lang1",):
        for stage, s in (res["table"].get(kind) or {}).items():
            print(f"  {kind:9s} {stage:8s} n {s['n']:3d}  drill {s['drill_mm']:5.2f} mm (max {s['drill_mm_max']:5.2f})  endoscope {s['endo_mm']:5.2f} mm"
                  + (f"  progress {s['progress_mean']:+.3f}" if "progress_mean" in s else ""))
    print("  × the floor: " + ", ".join(f"{k} {v}" for k, v in res["times_the_floor"].items()))
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
