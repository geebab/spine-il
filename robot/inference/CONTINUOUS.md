# Continuous closed loop (`live_loop.py --continuous`)

In the stop-and-go loop the arms hold while the next chunk is computed and the view is captured. The continuous loop keeps
the arms moving: it plans each chunk on a timetable (`stream_plan.py`) so that the next answer arrives before the current
chunk runs out, and blends the new chunk into the commands already sent.

## How it works

- Observe every `lead` frames (16 at 15 fps ≈ 1.1 s) while the arms keep executing; the request carries the measured state
  and the latest view (`../page_obs.py` stream: the recorder page's screencast, about 0.05 s).
- The new chunk is anchored at the **commanded** pose of the step it starts from (`--stream-anchor command`, default), not the
  measured pose — anchoring at the measured pose fed the arms' lag back into the next chunk and made the tools oscillate.
- Old and new chunks are blended over the overlap (`γ`), then pass the same step limits and safety gate as stop-and-go.
- The policy's progress output ends drilling (end level from `end_level.py`); harness limits remain as caps.

## Latency budget

Per request: image capture, upload of a 256 px PNG, inference (about 130 ms on an RTX PRO 6000) and the answer. The
timetable takes the measured round trip into account; with a remote endpoint the round trip dominates.

## Dry run (no robot)

```bash
bash robot/inference/live/dry_stream.sh      # simulated arms + stand-in page + the server's demo backend
```

Checks that every chunk arrives in time, that the blended commands stay inside the gate, and that the run ends by the
policy's progress output.
