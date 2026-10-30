# spine-render (bridge) — render API client

The virtual endoscope view and the bone removal come from Altician's viewer, which is closed source. It runs as a hosted
**render API** (a Nebius Serverless Endpoint until 2026-12-15); this package is the client side: input checks, the job and
step protocol, the bearer token, and the removed-bone voxel sets used for IoU / Dice. The server itself (the viewer in
headless Chrome) is not in this repository.

```bash
uv pip install -e "bridge[dev]"
pytest bridge
export RENDER_API_TOKEN=<token from the submission's testing instructions>
python -c "from spine_render.client import render; render('bridge/examples/trajectory.json', 'out.zip', server='https://<render endpoint>')"
```

## Two modes

- **`replay`** — a rig recording (recorder schema v2 JSONL) as it happened: the viewer re-applies the recorded cuts and
  shows the recorded tool poses; only the frame times and the captures are chosen. Training images are made this way.
- **`simulate`** — the viewer's engine drives and cuts from poses it is sent: authored trajectories and start + deltas
  (`examples/`), and `WS /v1/step` (a virtual robot, `robot/inference/rollout.py`). Time is a fake clock and the random
  seed is fixed, so reruns are near byte-identical.

## HTTP and WebSocket API

API: `POST /v1/jobs` (`{"input": ..., "options": {...}}`) → `GET /v1/jobs/{id}` → `GET /v1/jobs/{id}/result.zip`;
`POST /v1/normalize`; `GET /v1/health`; `WS /v1/step` (reset → step → close, one binary message per image; `simulate`).
A step with `"images": false` moves the tools, advances the clock and cuts, and returns the meta without capturing an image
(about 0.2 s instead of 0.4 s): a closed-loop client feeds every measured frame this way and asks for the image only at
its observation (`robot/inference/live_loop.py`).
Options: `fps, views (endo|scene), size, image_format, optics, drill, drill_size, seed, t_start_s, t_end_s, status,
video`, `mode (replay|simulate), diagnostic, tick_hz (1–1000), grid, grid_tool, reset_sdf, replay_timeouts, ae,
flags_override, extra_flags`, `removed_voxels, removed_every_s, drill_rule (r4|r3|r2|off)` . Booleans must be JSON booleans. `tick_hz` applies to `simulate` on a **recording** only
(the viewer's virtual engine ticks); authored inputs ignore it (noted in `flagInfo.notes`). `drill_size`
(`small|medium|large` = burr r 1.5 / 2 / 4 mm) is for authored inputs only — a recording keeps its own size.

`WS /v1/step` (simulate):

```
client → {"op": "reset", "case"?, "registration"?, "options"?: {views, size, image_format, optics, drill, drill_size, seed,
          drill_rule}, "start": {"endo": POSE, "drill": POSE}, "allow_outside"?: bool}
server → {"ok": true, "op": "reset", "mode": "simulate", "views": {...}, "setupS": s, "drillRule": {...}, "warnings": [...]}
client → {"op": "step", "dt_ms": 33.3, "endo": POSE | DELTA | {"lost": true}, "drill": ... (+ "trig": bool), "images": false?}
server → {"ok": true, "op": "step", "i", "t_ms", "meta", "images": [{"view", "format", "bytes"}]}
          then one binary message per image, in that order
client → {"op": "close"}
POSE  = {"tip": [3], "u": [3], "roll": deg} | {"tip": [3], "quat": [w,x,y,z]}          (scene LPS mm, tool→scene)
DELTA = {"dpos": [3], "drot_deg": [3], "frame": "tool" | "scene"}                     (relative to the last pose)
```

## Auth

Every `/v1/*` request, including the `WS /v1/step` handshake, needs `Authorization: Bearer <token>`; a missing or wrong
token gets `401`. `spine_render.client` reads the token from `RENDER_API_TOKEN`, else from
`~/.config/spinearm-render-api/token`.
