"""The endoscope's screen: is a point the viewer's bridge reports (bridge.drill.inView) inside the picture?

One place for the image circle of the viewer's built-in endoscope profile, used by the harness (inference/rollout.py: the
trigger is switched off while the drill tip is off screen) and by the evaluation (policy_eval.py: the drill on screen
while cutting). User, 2026-10-07: "화면에 드릴이 없으면 방아쇠 트리거 안하는 것" — and "드릴이 없으면" means the tip has
left the screen, nothing else: a tip hidden behind bone but inside the picture is on the screen.

Numbers (FE scopeDefaultProfile.ts; water = the air profile refracted, n 1.333) from the viewer session on 10-06; the view
angles they imply (92.93° air, 67.40° water) are the ones the step session reports (bridge.endoCamera.viewAngle).
K = fx, fy, cx, cy and the circle in px of the 1920×1080 calibration image; k = Kannala-Brandt coefficients; theta_max =
the farthest rim point from the axis (deg).
"""
from __future__ import annotations

import numpy as np

SCOPE_CIRCLE = (967.49, 513.57, 665.76)
SCOPE = {"air": {"K": (1042.625, 1042.766, 974.59, 425.98), "k": (-0.128175, 0.048912, -0.252681, 0.349202), "theta_max": 44.4645},
         "water": {"K": (1389.8197, 1390.0074, 974.5898, 425.9778), "k": (-0.107612, 0.395685, -3.140165, 7.502472),
                   "theta_max": 31.7007}}


def on_screen(x: float, y: float, view_angle_deg: float, optics: str) -> bool:
    """Is a point the bridge projects to inView (x, y) inside the image circle shown on screen? x, y are pinhole
    coordinates normalised by tan(viewAngle/2), y up; the bridge's own inFov tests that wider pinhole frame (radius
    0.9616), which reaches beyond the circle — by up to 8° at its top in water."""
    p = SCOPE[optics]; fx, fy, cx, cy = p["K"]; k = p["k"]
    th = float(np.arctan(np.tan(np.radians(view_angle_deg) / 2.0) * np.hypot(x, y)))
    if th > np.radians(p["theta_max"]):   # the polynomial holds inside the calibrated field only
        return False
    thd = th * (1 + k[0] * th ** 2 + k[1] * th ** 4 + k[2] * th ** 6 + k[3] * th ** 8); phi = np.arctan2(y, x)
    u, v = cx + fx * thd * np.cos(phi), cy - fy * thd * np.sin(phi)
    return bool(np.hypot(u - SCOPE_CIRCLE[0], v - SCOPE_CIRCLE[1]) <= SCOPE_CIRCLE[2])


def tip_on_screen(in_view: dict | None, view_angle_deg: float | None = None, optics: str | None = None) -> bool:
    """The drill tip is on the endoscope's screen: inside the viewer's field of view (inView.inFov) and — when the
    medium of the session is known (`optics` air | water, with the camera's view angle) — inside the image circle shown.
    Whether bone hides it (inView.visible) does not matter: a burr buried in the cavity it cut is on the screen."""
    iv = in_view or {}
    ok = bool(iv.get("inFov"))
    if ok and optics in SCOPE and view_angle_deg and iv.get("x") is not None and iv.get("y") is not None:
        ok = on_screen(float(iv["x"]), float(iv["y"]), float(view_angle_deg), optics)
    return ok


def tip_seen(in_view: dict | None, view_angle_deg: float | None = None, optics: str | None = None) -> bool:
    """The stricter test used until 10-07: on the screen and not hidden behind bone."""
    return bool((in_view or {}).get("visible")) and tip_on_screen(in_view, view_angle_deg, optics)
