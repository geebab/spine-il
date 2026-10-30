"""Shared bits of the judge scripts: where the robot-run files live, endpoint URLs and the bearer token."""
import os
from pathlib import Path

REPO = Path(os.environ.get("SPINE_IL_REPO") or Path(__file__).resolve().parents[3])
DATA = Path(os.environ.get("SPINE_IL_DATA") or REPO / "data" / "robot_runs")   # filled by fetch_robot_runs.py
HF_BASE = "https://huggingface.co/datasets/vincent-won/spine-il-hand90/resolve/main/robot_runs"

RUN17 = "s_20261017T182312_altician_lumbar1_trk_1855"   # L4, unseen contralateral sentence (paper Fig. 5 d)
RUN19 = "s_20261017T184609_altician_lumbar1_trk_3854"   # L4, unseen ipsilateral sentence (paper Fig. 5 h)


def token() -> str:
    """ENDPOINT_TOKEN (one token for both servers of the endpoint); the older per-server names still work."""
    for k in ("ENDPOINT_TOKEN", "RENDER_API_TOKEN", "POLICY_TOKEN"):
        if os.environ.get(k):
            return os.environ[k].strip()
    return ""


def auth() -> dict:
    t = token()
    return {"Authorization": f"Bearer {t}"} if t else {}
