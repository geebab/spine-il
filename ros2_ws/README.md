# ros2_ws

| package | content |
|---|---|
| `spinearm_msgs` | `TrackedPose` (one tool, one tracker frame), `ArmState` |
| `spinearm_description` | SO-101 URDF + meshes |
| `spinearm_bridge` | arm server / tracker tap bridges, calibration TF, description republisher, recording player |

```bash
colcon build --symlink-install --packages-select spinearm_msgs spinearm_description spinearm_bridge
source install/setup.bash
ros2 launch spinearm_bridge bridge.launch.py arms:=left,right      # live graph + Foxglove (port 8765)
```

## Recording replay (visualisation only)

`recording_player` plays a viewer session recording (recorder v2 `.jsonl`, rec-2.3.x) into the graph next to the live
robot model and tracker. It is not a timing tool and it does not move the arms.

```bash
ros2 launch spinearm_bridge replay.launch.py file:=~/il-robot/sessions/s_….jsonl [rate:=0.5] [loop:=true] \
    [start_s:=10.0 end_s:=40.0] [clock_mode:=wall|sim] [foxglove:=true]   # foxglove:=true only without bridge.launch
python3 -m spinearm_bridge.recording ~/il-robot/sessions/s_….jsonl         # offline summary + warnings, no ROS
```

| topic | type | frame | content |
|---|---|---|---|
| `/replay/tracker/<tool>/tracked` | TrackedPose | `tracker_cam` | filtered + raw pose, tip; stamp = the row's camera-frame time (`tExp`) |
| `/replay/tracker/<tool>/pose` | PoseStamped | `tracker_cam` | filtered pose |
| `/replay/tracker/balls` | MarkerArray | `tracker_cam` | raw pose × ball model (blue endo, yellow drill; live balls are green/magenta) |
| `/replay/drill/cuts` | MarkerArray | `scene` | cumulative drill stamps, sphere radius = the recorded `rMm` (latched, frame-locked, ≤ 2000 stamps per marker) |
| `/replay/drawn` | MarkerArray | `scene` | each tool as the viewer drew it when the row arrived; drill red while cutting, sized by `drill_size_changed` |
| `/tf_static` (`/tf` per row if dynamic) | | `tracker_cam → scene` | the recording's registration `T_scene←cam`, inverted so `tracker_cam` keeps its hand-eye parent |
| `/diagnostics` | | | `spinearm/replay`: progress, pacing lag, screen staleness (`tExp − drawn.tExp`) |

- `namespace:=/` publishes on the live names (`/tracker/…`) — only without `tracker_bridge` (diagnostics turn ERROR
  if another node publishes them).
- Two clocks on one timeline: tracker poses at the camera frame (`tExp`); drawn tools, cuts and events at the viewer's
  clock (`t0Wall + tMono − t0Mono`), so the drawn tool trails the tracked one by the lag the operator saw and a cut
  appears when the drawn drill is there.
- Clock: `wall` (default) stamps like live data (start time + recorded offset / rate), so it mixes with the live graph.
  `sim` stamps the recorded time and publishes `/clock` (rosbag convention) — every consumer, Foxglove bridge included,
  needs `use_sim_time:=true`; loop restarts jump `/clock` back.
- The replayed poses are in the *recording's* camera frame drawn at the current `tracker_cam`: exact on the rig it was
  recorded on, shifted by the camera move otherwise. Registration edge: `tf_parent:=camera` (default) · `scene` with
  `camera_frame:=replay_cam` (scene as root) · `none` (e.g. when `calib_tf registration:=` already publishes one — then
  cuts / drawn follow the current rig's registration while the tracked poses stay in the recording's camera frame).
- Foxglove (3D panel, display frame `left_base_link` or `scene`): enable `/replay/tracker/balls`, `/replay/drill/cuts`,
  `/replay/drawn`. Keep the `…/pose` arrows off (metre-long by default).
