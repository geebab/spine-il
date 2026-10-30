"""safety_gate.py asks only whether the arms are closer than ARMS_MIN_MM (10 mm). Since 10-12 it gives parts_dist that
distance as its cutoff instead of 70 mm (a check took 162 ms, 146 of them measuring part pairs that could not decide the
question; now 43 ms). The verdict and the reported distance must be the same — on the real arm models, for configurations
that do come closer than 10 mm."""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE.parents[1] / "eval" / "pairstart"))
import safety_gate as SG      # noqa: E402


@pytest.fixture(scope="module")
def arms():
    try:
        import common as C
        import verify_two_arm as V
        from lerobot_robot_spinearm import safezone as sz
        a, _, _ = V.load_models()
    except Exception as e:      # the arm models (meshes, calibration) are on the robot server only
        pytest.skip(f"arm models not here: {type(e).__name__}")
    return a, sz, {k: np.asarray(v, float) for k, v in C.NEUTRAL.items()}


def test_the_cutoff_at_the_threshold_gives_the_same_verdict_and_distance(arms):
    a, sz, neutral = arms
    rng = np.random.default_rng(0); thr = SG.ARMS_MIN_MM / 1000.0
    close = far = 0
    for _ in range(400):
        q = {k: neutral[k] + rng.uniform(-70, 70, len(neutral[k])) for k in ("left", "right")}
        A, B = a["left"].pose(q["left"])["world"], a["right"].pose(q["right"])["world"]
        d70, d10 = sz.parts_dist(A, B, 0.07), sz.parts_dist(A, B, thr)
        assert (d70 < thr) == (d10 < thr)
        if d70 < thr:
            close += 1
            assert d10 == pytest.approx(d70, abs=1e-12)      # the number in "arms N mm apart" is the same
        else:
            far += 1
            assert d10 >= thr
        if close >= 12 and far >= 12:
            break
    assert close >= 5 and far >= 5, (close, far)             # the comparison was made on both sides of the threshold


def test_the_gate_uses_its_threshold_as_cutoff():
    src = (HERE.parent / "safety_gate.py").read_text()
    assert "self.arms_cutoff_m = ARMS_MIN_MM / 1000.0" in src and "parts_dist(po[\"left\"], po[\"right\"], self.arms_cutoff_m)" in src
