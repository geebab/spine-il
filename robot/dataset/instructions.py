"""Instruction sets — the language a frame carries (2026-10-13).

User, 10-13: GR00T is a VLA, but its language input was a two-way switch ("3 approach" / "4 drill") that the harness
flips at the first cut, and the picture alone tells the two apart. Language does work only where the SAME scene calls for
a different action depending on the words. Two such forks exist or are coming:

  extent   every recording cuts the left (ipsilateral) L4 lamina first and then moves medially; the recordings of 09-28
           stay on the left of the midline, most later ones cross it under the spinous process (contralateral
           decompression). At "ipsilateral part removed" the picture is the same and the next action differs — the
           words say which: "ipsilateral only" / "with contralateral" (user, 10-14). (The user lifted the 10-09 "no planned-extent input" rule for
           this, 10-13: "바꿔도 된다".) The end of the drilling stays the policy's: the progress output is the bone removed
           over the recording's own total, so it rises to 1 earlier under a smaller goal.
  level    recordings of L3 and L5 from the same skin entry (5 each, to come): above the skin the picture and the pose
           are the same for the three levels.

A set is named and travels with the dataset and the model (meta/spine_il/export.json "instructions"): the models trained
before this (G24a, G30a, G22a, ACT) keep "phase1" = dataset/phases.py PHASES, whose texts never change under them.

  phase1   PHASES[phase] — "3 피부에서 타겟으로 이동 / advance …", "4 L4-5 후궁 드릴링 / drill the L4-5 lamina" (legacy)
  lang1    plain English, no digit prefix, no mixed Korean, several wordings per instruction (one wording per episode and
           phase, picked by the episode's id — a model that met one sentence only learns the sentence, not its meaning).
           All recordings so far are a LEFT L4 laminectomy (user, 10-13) and the texts say so.
  lang2    lang1's texts and a drilling sentence that names no extent (NEUTRAL, 10-14, after the analysis of why the
           words did not steer L30b / L30c2: half the L4 labels sat on the boundary — the 10-06 "ipsilateral" recordings all
           reach 0–2 mm past the midline — so the words were asked to tell apart recordings that do the same). Only a
           clear recording names its extent: medial reach not past the midline = ipsilateral, more than EDGE_MM[1] past
           = contralateral; the band between carries the neutral sentence. Still one goal sentence per recording (user,
           10-14: the harness never switches it by cut position). export_lerobot.py --cond-dropout gives a share of every
           recording's drilling frames the neutral sentence too: the goal-free answer, the one guidance pushes away from.
  lang3    lang2 with five times the drilling wordings (10-17, user: "표현 늘린 방법으로 재학습"): the 10-16 proof showed that
           sentences the model never met steer the robot like trained ones, but offline a paraphrase moved a chunk only 40 %
           as far — three wordings per extent teach the sentences more than their meaning. The first three of each extent
           are lang2's (the robot's sentence stays the same); the extra ones never use the words of the held-out test
           paraphrases (inference/remote proof: "right", "bilateral", "unilateral", "take down", "carry on", "undercut",
           "untouched", "approach", "keep to", "other side"), so those stay a test of meaning, not of vocabulary.
"""
from __future__ import annotations

import hashlib

import numpy as np

SETS = ("phase1", "lang1", "lang2", "lang3")
NEUTRAL_SETS = ("lang2", "lang3")   # the sets with a drilling sentence that names no extent, and clear-only extent labels
LEVELS = ("L3", "L4", "L5")
EXTENTS = ("ipsilateral", "contralateral")
DEFAULT_EXTENT = "contralateral"     # what a goal that names no extent means, and a frame of a recording without cuts
NEUTRAL = "unspecified"              # lang2: the drilling sentence that names no extent (never lang1's)
# The medial reach of a recording = the 5th percentile of its cut centres along x (LPS: +x = the patient's left). The
# midline of L4 in the scene frame is x = 21.9 mm (the tip of its spinous process; the model is not centred on x = 0 — my
# first names "lateral / mid / full" of 10-13 assumed it was and were wrong). User, 10-14: two classes, "ipsilateral only" /
# "with contralateral". A recording is contralateral when its reach lies more than CROSS_MM past the midline:
# 09-28 23.5–27.7 mm (none crosses), 10-06 / 10-12 12.0–22.1 mm. The reaches are continuous around the boundary
# (20.2, 19.9, 19.6, 19.1 mm) — the classes are a cut through a continuum, not two clusters.
# Per level (user, 10-14: Lt. L3 recordings are coming, label them the same way): the x of each level's spinous process tip
# (mean x of the 2 % most posterior vertices of its mesh, calib/level_meshes; spread 1.7–2.0 mm).
MIDLINE_X_MM = {"L3": 22.6, "L4": 21.9, "L5": 21.5}
CROSS_MM = 2.0
EXTENT_X_MM = {lv: round(m - CROSS_MM, 1) for lv, m in MIDLINE_X_MM.items()}   # reach ≥ this = ipsilateral · below = contralateral
# lang2: how far the medial reach lies past the midline — ≤ EDGE_MM[0] clearly ipsilateral, > EDGE_MM[1] clearly
# contralateral, between = the neutral sentence. 10-14 (09-28 left out): L3 ipsi 8 clear / 7 between, contra 14 / 1;
# L4 ipsi 8 / 8, contra 7 / 7; L5 all clear but 8dec and bd4c. The 0 / 5 mm are mine.
EDGE_MM = (0.0, 5.0)

_SIDE = "left"
_EXTENT_TEXT = {
    "ipsilateral": ("{Side} {level} laminectomy, ipsilateral only.",
                    "Remove the {side} {level} lamina on the {side} side only; do not cross the midline.",
                    "Drill the {side} {level} lamina, ipsilateral side only."),
    "contralateral": ("{Side} {level} laminectomy with contralateral decompression.",
                      "Remove the {side} {level} lamina, then cross the midline under the spinous process to the opposite side.",
                      "Drill the {side} {level} lamina and continue across the midline for contralateral decompression."),
    NEUTRAL: ("{Side} {level} laminectomy.",
              "Remove the {side} {level} lamina.",
              "Drill the {side} {level} lamina."),
}
# lang3 (10-17): lang2's three, then twelve more per extent and seven more neutral ones
_EXTENT_TEXT3 = {
    "ipsilateral": _EXTENT_TEXT["ipsilateral"] + (
        "Decompress the {side} side of {level} only and stay short of the midline.",
        "{Side}-sided {level} laminectomy; stop at the midline.",
        "Open the {side} {level} lamina without passing the spinous process.",
        "Remove bone on the {side} side of {level}; leave the contralateral side intact.",
        "Ipsilateral {side} {level} decompression only.",
        "Work only on the {side} {level} lamina and finish before reaching the midline.",
        "Hemilaminectomy of {level} on the {side}, no crossover.",
        "Drill away the {side} half of the {level} lamina and nothing beyond it.",
        "Thin the {side} {level} lamina; the midline is the limit.",
        "Limit the drilling to the {side} {level} lamina.",
        "{Side} {level} hemilaminectomy only.",
        "Clear the {side} {level} lamina up to, but not across, the midline."),
    "contralateral": _EXTENT_TEXT["contralateral"] + (
        "Decompress the {side} side of {level}, then go past the midline and decompress the contralateral side too.",
        "{Side} {level} laminectomy, then cross over beneath the spinous process.",
        "Open the {side} {level} lamina and extend the decompression across the midline.",
        "Remove bone on the {side} side of {level} and continue to the contralateral lamina.",
        "Over-the-top decompression at {level} from the {side}.",
        "Work through the {side} {level} lamina and keep drilling past the spinous process to the far side.",
        "{Side} {level} hemilaminectomy with crossover to the contralateral side.",
        "Drill the {side} half of the {level} lamina, then the base of the spinous process and beyond the midline.",
        "Ipsilateral and contralateral decompression at {level} from the {side}.",
        "Do not stop at the midline; decompress both sides of {level}.",
        "Cross the midline after the {side} {level} lamina and decompress the far side.",
        "{Side} {level} laminectomy extended across the midline."),
    NEUTRAL: _EXTENT_TEXT[NEUTRAL] + (
        "{Side} {level} decompression.",
        "Open the {side} {level} lamina.",
        "Drill bone at the {side} {level} lamina.",
        "Laminectomy at {level} from the {side}.",
        "Work on the {side} {level} lamina.",
        "{Side} {level} lamina, drilling.",
        "Begin drilling the {side} {level} lamina."),
}
_PHASE_TEXT = {
    1: ("Hold the ready pose above the skin.", "Wait in the ready pose."),
    2: ("Move both tools to the skin entry points.", "Bring the endoscope and the drill to their skin portals."),
    3: ("Advance the endoscope and the drill to the {side} {level} lamina.",
        "Go in through the skin to the {side} {level} lamina.",
        "Approach the {side} {level} lamina with both tools."),
    5: ("Withdraw both tools out of the skin.", "Pull the endoscope and the drill back out."),
    6: ("Return to the ready pose.", "Go back to the ready pose above the skin."),
}


def extent_of(cut_x_mm, level: str = "L4") -> str:
    """The extent class of a recording of `level` from the x of its bone cut centres (scene frame, mm)."""
    x = np.asarray(cut_x_mm, float)
    if x.size == 0:
        raise ValueError("no cut centres — the extent of a recording without cuts is not defined")
    reach = float(np.percentile(x, 5))
    return "ipsilateral" if reach >= EXTENT_X_MM[level] else "contralateral"


def extent_label(name: str, cut_x_mm, level: str = "L4") -> str:
    """The extent a recording's drilling sentence names in the set `name`: lang1 its class (extent_of); lang2 only a clear
    one — the neutral sentence for a medial reach between EDGE_MM[0] and EDGE_MM[1] past the midline of its level."""
    if name not in NEUTRAL_SETS:
        return extent_of(cut_x_mm, level)
    x = np.asarray(cut_x_mm, float)
    if x.size == 0:
        raise ValueError("no cut centres — the extent of a recording without cuts is not defined")
    past = MIDLINE_X_MM[level] - float(np.percentile(x, 5))
    return "ipsilateral" if past <= EDGE_MM[0] else "contralateral" if past > EDGE_MM[1] else NEUTRAL


def _stl_points(path) -> np.ndarray:
    b = open(path, "rb").read(); n = int(np.frombuffer(b[80:84], "<u4")[0])
    if len(b) != 84 + 50 * n:
        raise ValueError(f"{path}: not a binary STL")
    tri = np.frombuffer(b[84:], np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), n)
    return np.unique(tri["v"].reshape(-1, 3).round(2), axis=0).astype(float)


def level_of(cut_centres_mm, mesh_dir=None, every: int = 5, min_share: float = 0.8) -> tuple[str, dict]:
    """The level a recording drilled: every cut centre (scene frame, mm) goes to the level whose mesh surface is nearest;
    the recording is that level when at least `min_share` of them agree. The laminae of neighbouring levels interlock
    (their z ranges overlap by 13 mm), so a plane does not separate them. Meshes: the phantom's bones before merging
    (L3 / L4 / L5 / sacrum .stl, same frame as the cut rows — checked 10-13: 7,157 of 7,181 centres of the 34 hand
    recordings on L4, median 1.35 mm from its surface), outside the repository: $UBE_LEVEL_STL or ~/il-robot/calib/level_meshes."""
    import os
    from pathlib import Path
    d = Path(mesh_dir or os.environ.get("UBE_LEVEL_STL") or Path.home() / "il-robot/calib/level_meshes")
    names = [n for n in (*LEVELS, "sacrum") if (d / f"{n}.stl").exists()]
    if not set(LEVELS) <= set(names):
        raise FileNotFoundError(f"level meshes L3/L4/L5 .stl not found in {d}")
    c = np.asarray(cut_centres_mm, float).reshape(-1, 3)[::max(1, int(every))]
    if not len(c):
        raise ValueError("no cut centres — the level of a recording without cuts is not defined")
    dist = np.empty((len(c), len(names)))
    for j, n in enumerate(names):
        v = _stl_points(d / f"{n}.stl")
        for i in range(0, len(c), 256):
            dist[i:i + 256, j] = np.sqrt(((c[i:i + 256, None, :] - v[None]) ** 2).sum(-1).min(1))
    share = np.bincount(dist.argmin(1), minlength=len(names)) / len(c)
    k = int(share.argmax())
    info = {"share": {n: round(float(x), 3) for n, x in zip(names, share)}, "surface_mm_median": round(float(np.median(dist.min(1))), 2)}
    if names[k] not in LEVELS or share[k] < min_share:
        raise ValueError(f"no level holds {min_share:.0%} of the cut centres: {info['share']}")
    return names[k], info


def _pick(options, key: str) -> str:
    return options[int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(options)]


def wordings(phase: int, level: str = "L4", extent: str = DEFAULT_EXTENT, name: str = "lang2") -> list[str]:
    """Every wording a set has for this phase / level / extent (NEUTRAL: the sentence that names none). lang1 / lang2 share
    theirs; lang3 has more drilling wordings, its first three are lang2's."""
    if level not in LEVELS or extent not in (*EXTENTS, NEUTRAL):
        raise ValueError(f"level {level!r} / extent {extent!r}")
    opts = (_EXTENT_TEXT3 if name == "lang3" else _EXTENT_TEXT)[extent] if int(phase) == 4 else _PHASE_TEXT[int(phase)]
    return [o.format(side=_SIDE, Side=_SIDE.capitalize(), level=level) for o in opts]


def text(name: str, phase: int, level: str = "L4", extent: str = DEFAULT_EXTENT, key: str = "", legacy=None) -> str:
    """The instruction of one frame. `key` picks the wording (an episode id: one wording per episode and phase; "" = the
    first wording — what the robot loop sends unless told otherwise). `legacy` = phases.PHASES for the set phase1."""
    if name == "phase1":
        return legacy[int(phase)]
    if name not in ("lang1", "lang2", "lang3"):
        raise ValueError(f"unknown instruction set {name!r} (known: {', '.join(SETS)})")
    if extent == NEUTRAL and name not in NEUTRAL_SETS and int(phase) == 4:
        raise ValueError(f"the drilling sentence that names no extent is lang2's / lang3's — a {name} model never met it")
    opts = wordings(phase, level, extent, name)
    return _pick(opts, f"{key}/{int(phase)}") if key else opts[0]


def extents(name: str) -> list[str]:
    """The extents a set's drilling sentences name (lang2: and the neutral one)."""
    return [*EXTENTS, NEUTRAL] if name in NEUTRAL_SETS else list(EXTENTS)


def table(name: str, legacy=None) -> dict:
    """What a dataset writes beside itself, and a server tells the loop: the set's name and every text it can carry."""
    if name == "phase1":
        return {"name": name, "phases": {str(k): v for k, v in legacy.items()}}
    if name not in SETS:
        raise ValueError(f"unknown instruction set {name!r} (known: {', '.join(SETS)})")
    ex_ = extents(name)
    return {"name": name, "side": _SIDE, "levels": list(LEVELS), "extents": ex_, "extent_x_mm": dict(EXTENT_X_MM), "midline_x_mm": dict(MIDLINE_X_MM),
            **({"edge_mm": list(EDGE_MM)} if name in NEUTRAL_SETS else {}),
            "phases": {str(p): wordings(p) for p in (1, 2, 5, 6)}, "approach": {lv: wordings(3, lv) for lv in LEVELS},
            "drilling": {lv: {ex: wordings(4, lv, ex, name) for ex in ex_} for lv in LEVELS}}


def texts(tab: dict) -> list[str]:
    """Every text of a table() — what a server lists as the instructions its model was trained with."""
    out: list[str] = []

    def walk(v):
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
    for k in ("phases", "approach", "drilling"):
        walk(tab.get(k) or {})
    return list(dict.fromkeys(out))


def parse_goal(s: str | None) -> dict | None:
    """"L4:ipsilateral" / "L5" / "contra" → {"level", "extent"} (what is not said: L4, contralateral; ipsi / contra are
    enough; "unspecified" / "neutral" = lang2's sentence that names no extent). None for nothing."""
    if not s:
        return None
    g = {"level": "L4", "extent": DEFAULT_EXTENT}
    for part in str(s).replace(",", ":").split(":"):
        p = part.strip()
        if p.upper() in LEVELS:
            g["level"] = p.upper()
        elif p.lower() == "neutral":
            g["extent"] = NEUTRAL
        elif p and [x for x in (*EXTENTS, NEUTRAL) if x.startswith(p.lower())] and len(p) >= 4:
            g["extent"] = next(x for x in (*EXTENTS, NEUTRAL) if x.startswith(p.lower()))
        elif p:
            raise ValueError(f"goal {s!r}: {p!r} is neither a level ({', '.join(LEVELS)}) nor an extent ({', '.join((*EXTENTS, NEUTRAL))})")
    return g
