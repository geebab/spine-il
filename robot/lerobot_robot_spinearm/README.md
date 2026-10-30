# lerobot_robot_spinearm

SO-101 **arm server** + LeRobot **proxy robots**. The server is the only process that opens the
serial port; clients hold a lease and stream joint goals (degrees, 6 joints — joint 6 carries the
tracked marker) over a local ZMQ socket. `--sim` runs the same server with a kinematic arm so you
can develop anywhere without hardware.

```
client (your plugin / teleop)  ──REP 5601──▶  spine-arm-server --arm left   ──▶ /dev/so101_left
                               ◀──PUB 5602──   (lease · bounds · step cap · watchdog · faults · home)
```

## Install

```bash
uv pip install -e robot/lerobot_robot_spinearm          # teammates: inside your own venv
spine-arm-server --arm left --sim --cmd-port 6601 --pub-port 6602   # a virtual arm of your own
spine-arm-ctl --cmd-port 6601 status
```

## Use from LeRobot

```bash
lerobot-teleoperate --robot.type=spine_arm --robot.arm=left --robot.cmd_port=6601 ...
```

```python
from lerobot_robot_spinearm import SpineArm, SpineArmConfig
arm = SpineArm(SpineArmConfig(arm="left", cmd_port=6601, owner="alice"))
arm.connect()                      # lease + torque on
arm.send_action({"shoulder_pan.pos": 10.0})   # stream at ≥ 10 Hz; the watchdog holds after 250 ms
arm.get_observation()              # {"shoulder_pan.pos": …, …, "gripper.pos": …}
arm.home(); arm.torque_off(); arm.disconnect()
```

## Rules of the road

- **One lease per arm.** `lease` is refused while someone else holds it; `--force` works only if the
  holder has been idle ≥ 5 s. Leases expire after 15 s without a command.
- **Stream goals at ≥ 10 Hz.** 250 ms without a goal → the arm holds (watchdog).
- **`estop` needs no token** — anyone can stop an arm at any time.
- Faults freeze the arm with torque on. `overload_protection`/`power_cycled` need a 12 V power-cycle,
  then `enable`. `overheat` clears itself below 50 °C. `stall`/`home_stalled`: fix the obstacle, `clear`.
- `home` folds joint 6 first (the marker bracket collides with the arm otherwise), then the rest.
  `torque_off` is only accepted at the neutral pose (or with `--force` while you hold the arm).

## Protocol

JSON over ZMQ REQ/REP. Every command except `lease`, `estop`, `status`, `state` needs the lease `token`.

| cmd | args | reply |
|---|---|---|
| `lease` | owner, force? | token, ttl |
| `enable` / `hold` / `home` / `release` / `renew` / `clear_faults` | token | state |
| `goal` | token, pos {joint: deg} (subset ok) | sent (clamped), state, pos |
| `torque_off` | token, force? | state — refused `not_at_neutral` |
| `set` | token, step_deg (0.5–10) | step_deg |
| `estop` | — | state |
| `status` / `state` | — | snapshot (+ neutral, bounds for status) |
| `fake` | fault, joint? (sim only) | faults |

PUB stream (30 Hz): `state {"t","arm","state","torque","pos","goal","load","temp","lease","faults","bus_ok","tick"}`.

## End-effector helpers (`lerobot_robot_spinearm.ee`)

The server only speaks joints; Cartesian control lives on the client. `MarkerKinematics` (FK/IK on
`assets/so101_marker.urdf`, frame `marker_frame_link` = the marker on joint 6, 54 mm off its axis) and
`EETeleop` (accumulate a target from per-tick deltas inside the workspace box from `assets/ee_bounds.json`,
z ≥ 0.04 m). Set the IK joint limits from the arm — `kin.set_limits(arm.status()["bounds"])` — placo drags
any joint outside its limit back ~10°/solve, and the CAD limits are tighter than the calibrated arm.
The target is a full pose (position 1/m, orientation 0.01/rad), so all six joints take part and the wrist does not
drift; joint jogs (`{"gripper": ±1}`, `{"wrist_roll": ±1}`) move that joint and re-base the target pose. placo velocity
limits (60°/s) and a 2 cm target leash keep the QP from flipping the elbow near singularities / out of reach.
Reference client: `robot/joycon_ee_teleop.py` (`--arm left`, `--cmd-port 6601` for a `--sim` server).

## Running as a service (owner)

```bash
bash robot/lerobot_robot_spinearm/systemd/install-arm-server.sh   # arm-server@left / arm-server@right (systemd --user)
systemctl --user status arm-server@left; journalctl --user -u arm-server@left -f
```
A restart keeps torque: an enabled arm is picked up in `HOLD` (the lease is dropped — lease again).
Nothing else may open `/dev/so101_*` while the service runs (stop it first for the old direct-bus tools).

## Tests

```bash
pytest robot/lerobot_robot_spinearm            # controller + rpc + plugin (spawns --sim servers)
```
