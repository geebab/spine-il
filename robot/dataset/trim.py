"""Trim a hand session to the part the two arms can play back (export_lerobot.py --trim robot).

Why: a hand recording starts and ends at the operator's ready pose — tools resting 45–58 mm beside the portal, at skin
height (rev.4 set). The arms cannot follow that part: from there the drill marker leaves the detection window, and
from the working pose the first seconds after the skin entry ask for an orientation the 5-DOF arm does not reach
(inference/ready_sweep.py, safety_gate.py). Changing how people record is harder than cutting the data, so the front
and the back of every episode are cut before training; on the robot the cut parts are replaced by the scripted axial
entry/exit from the robot's own ready pose (inference/rollout.py --cycle), which is safe for all rev.4 recordings.

Rule (uniform for every recording, no robot model needed):
  start = the first bone cut − pre_s, but not before both tools are inside the skin (the both-inside run that reaches
          the first cut) — the tools are then near the target, in the working orientation
  end   = both tools out of the skin (end of phase 5) + post_s (default −1 s)
Kept frames keep their phase labels (3 advance, 4 drill, 5 withdraw); phases 1, 2 and 6 are not in the trimmed data.

A gate-verified span of the recording (inference/trim_gate.py: every frame through the robot safety gate with that
recording's own start pair) narrows the rule's span when given — the rule alone is checked against those spans.
"""

from __future__ import annotations

import numpy as np

PRE_S = 7.0     # kept before the first bone cut (rev.4: the gate-safe part starts 8.2–20 s before it, trim_gate.py)
POST_S = -1.0   # relative to "both tools out of the skin": the last second of the withdrawal is cut — with the tools out, the
                # drill marker turns > 40° off the camera on 2 of the 8 rev.4 recordings (919d, 3868: gate-safe to 0.6 s before)


def span(t_rel, inside, t_cut0: float, t_out: float, pre_s: float = PRE_S, post_s: float = POST_S,
         gate_span=None) -> tuple[int, int, dict]:
    """(ia, ib, info): frames ia..ib (inclusive) of the grid t_rel (s, session axis) are kept.

    inside: (N, 2) bool — endo, drill inside the skin at each grid time. gate_span: optional (t_a, t_b) s."""
    t = np.asarray(t_rel, float)
    both = np.asarray(inside, bool).all(axis=1)
    info: dict = {"rule": "first cut − pre_s (both tools inside) … both tools out + post_s", "pre_s": pre_s, "post_s": post_s}
    i_cut = int(np.clip(np.searchsorted(t, t_cut0), 0, len(t) - 1))
    i = i_cut
    if not both[i]:                       # a tool reads outside at the first cut (portal flicker): nearest earlier frame
        prev = np.flatnonzero(both[:i + 1])
        info["flag"] = "첫 절삭 순간 두 기구가 모두 안으로 판정되지 않음"
        i = int(prev[-1]) if len(prev) else i_cut
    while i > 0 and both[i - 1]:
        i -= 1
    t_both = float(t[i])
    t_a = max(t_cut0 - pre_s, t_both)
    t_b = t_out + post_s
    info.update(t_both_inside=round(t_both, 2), t_cut0=round(float(t_cut0), 2), t_out=round(float(t_out), 2))
    if gate_span is not None:
        ga, gb = float(gate_span[0]), float(gate_span[1])
        info["gate_span"] = [round(ga, 2), round(gb, 2)]
        info["rule_inside_gate"] = bool(t_a >= ga - 1e-6 and t_b <= gb + 1e-6)
        t_a, t_b = max(t_a, ga), min(t_b, gb)
    ia = int(np.searchsorted(t, t_a - 1e-9))
    ib = int(np.searchsorted(t, t_b + 1e-9, side="right") - 1)
    ia, ib = max(0, min(ia, len(t) - 1)), max(0, min(ib, len(t) - 1))
    if ib - ia < 2:
        raise ValueError(f"trim leaves {ib - ia + 1} frames (t_a {t_a:.2f}, t_b {t_b:.2f})")
    info.update(t_a=round(float(t[ia]), 3), t_b=round(float(t[ib]), 3), frames_in=len(t), frames_kept=ib - ia + 1,
                cut_front_s=round(float(t[ia] - t[0]), 2), cut_back_s=round(float(t[-1] - t[ib]), 2))
    return ia, ib, info
