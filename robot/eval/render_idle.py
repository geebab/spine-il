"""Exit 0 when the render API has no queued/running job (robot-origin recordings must not share the GPU with renders)."""
import json, sys
import vol_eval as V
jobs = json.loads(V.call("/v1/jobs"))
busy = [j["id"] for j in jobs if j.get("status") in ("queued", "running", "pending", "started")]
sys.exit(1 if busy else 0)
