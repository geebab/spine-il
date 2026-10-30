<!--
Open-H-Embodiment dataset README (template v2.0), written by robot/dataset/openh.py of spine-il at export time.
Numbers come from the exported data. Items marked TODO(team) are not known to the exporter: complete them before submission.
-->

# $name - README

---

## 📋 At a Glance

Demonstrations of the drilling step of a biportal endoscopic left L4-5 laminectomy on a registered virtual patient: two optically tracked replica instruments (endoscope, drill), a simulated endoscope view and simulated bone removal.

---

## 📖 Dataset Overview

This export holds $sources. In a demonstration the operator holds an endoscope replica in one hand and a drill replica in the other; a tracking camera measures both instruments, and a viewer shows the virtual patient (a segmented lumbar spine in patient image space) from the endoscope's pose and removes bone voxels where the running burr touches bone. The demonstrations are made on a physical phantom that consists of the skin only and is hollow inside: the instrument motion is real, and everything under the skin — the anatomy, the endoscope image and the bone removal — is simulated. $trim

Phases in this export (per-frame `observation.meta.phase_index` and `instruction.text`): $phases.

| | |
| :--- | :--- |
| **Total Hours** | `$hours` ($frames frames at $fps Hz) |
| **Total Trajectories** | `$episodes` |
| **Data Type** | `[ ] Clinical` `[ ] In-Vivo` `[ ] Ex-Vivo` `[ ] Phantom / Table-Top` `[x] Simulation` (real instrument motion on a hollow skin-only phantom; digital anatomy and imaging) |
| **License** | OpenMDW-1.1 (the Open-H-Embodiment release licence; TODO(team): confirm before submission) |
| **Version** | `0.1` |

---

## 🎯 Tasks & Domain

### Domain

- [x] **Surgical Robotics**
- [ ] **Ultrasound Robotics**
- [ ] **Other Healthcare Robotics**

### Demonstrated Skills

- Holding the endoscopic view on the target while a second instrument works (two portals)
- Advancing the drill from the skin entry to the L4-5 lamina
- Drilling the lamina (bone removal with a spherical burr, trigger on and off)
- Withdrawing both instruments

---

## 🔬 Data Collection Details

### Collection Method

- [ ] **Human Teleoperation**
- $robot_box **Programmatic/State-Machine** (robot episodes: two SO-101 arms play back a recorded or policy-generated instrument trajectory)
- [ ] **AI Policy / Autonomous**
- $hand_box **Other**: direct manipulation — hand-held, optically tracked instruments; no robot and no teleoperation in the loop

### Operator Details

| | Description |
| :--- | :--- |
| **Operator Count** | `$operators` |
| **Operator Skill Level** | TODO(team): Expert / Intermediate / Novice |
| **Collection Period** | From `$period_from` to `$period_to` |

### Recovery Demonstrations

- [ ] **Yes**
- [x] **No**

---

## 💡 Diversity Dimensions

- [ ] **Camera Position / Angle**
- [ ] **Lighting Conditions**
- [ ] **Target Object**
- [ ] **Spatial Layout**
- [ ] **Robot Embodiment**
- [ ] **Task Execution**
- [ ] **Background / Scene**
- [ ] **Other**

No dimension was varied on purpose: one virtual patient, one simulated light and lens model, one task. The repetitions differ by the operator's own execution (path, drilling time, removed bone volume).

---

## 🛠️ Equipment & Setup

### Robotic Platform(s)

- **Hand episodes:** none — hand-held replica instruments (endoscope; drill with a trigger button)
- **Robot episodes (if any):** two SO-ARM101 arms, left = endoscope, right = drill

### Sensors & Cameras

| Type | Model/Details |
| :--- | :--- |
| **Primary Camera** | Simulated endoscope: the viewer's endoscope screen rendered from the tracked endoscope pose, $image; lens and medium model: `$medium`. The frame is circular (black outside the image circle) and includes the viewer's overlays (orientation notch at the rim, burr contact marker). |
| **Instrument tracking** | Intel RealSense D435 (848x480, 90 fps) tracking a marker-sphere constellation on each instrument; filtered poses, recorded at the tracker's output rate (about 30 Hz per instrument in the first recordings); tip offset calibrated per instrument |
| **Force/Torque Sensor** | none |
| **Medical Imager** | none during collection (the virtual patient is a segmented lumbar spine model) |
| **Other** | Drill trigger: a button of a controller mounted on the drill replica (its IMU is fused into the drill's orientation) |

Camera intrinsics: $intrinsics.

---

## 🎯 Action & State Space Representation

### Action Space Representation

**Primary Action Representation:**
- [x] **Absolute Cartesian**
- [ ] **Relative Cartesian**
- [ ] **Joint Space**
- [ ] **Other**

**Orientation Representation:**
- [x] **Quaternions** (x, y, z, w)
- [ ] **Euler Angles**
- [ ] **Axis-Angle**
- [ ] **Rotation Matrix**
- [ ] **Other**

**Reference Frame:**
- [ ] **Robot Base Frame**
- [ ] **Tool/End-Effector Frame**
- [x] **World/Global Frame**: the scene frame = patient image space (LPS axes), fixed to the anatomy. The tracking camera is registered to it once per session with five fiducial points. Hand-held instruments have no robot base.
- [ ] **Camera Frame**
- [ ] **Other**

**Action Dimensions:**

```
action: [endoscope.x_m, endoscope.y_m, endoscope.z_m, endoscope.qx, endoscope.qy, endoscope.qz, endoscope.qw, endoscope.gripper,
         drill.x_m,     drill.y_m,     drill.z_m,     drill.qx,     drill.qy,     drill.qz,     drill.qw,     drill.gripper]
- x_m, y_m, z_m: instrument tip position in the scene frame (metres)
- qx, qy, qz, qw: unit quaternion rotating the instrument's marker frame into the scene frame (sign kept continuous)
- endoscope.gripper: always 0 (the endoscope has nothing to actuate; the slot keeps the block layout)
- drill.gripper: burr trigger, 0 = off, 1 = on
```

**`action[k]` is `observation.state[k+1]`: the pose measured one frame later.** No setpoint was commanded — the instruments are hand-held, and a robot episode replays a trajectory — so the next measured pose (and trigger) is the positional target. It is not a velocity.

### State Space Representation

**State Information Included:**
- $joint_box **Joint Positions** (robot episodes only, auxiliary: `observation.meta.joint_positions`, radians)
- [ ] **Joint Velocities**
- [x] **End-Effector Pose** (Cartesian position/orientation)
- [ ] **Force/Torque Readings**
- [x] **Gripper State** (drill trigger)
- [ ] **Other**

**State Dimensions:**

```
observation.state: the 16 values of the action layout above, measured at the frame's time

observation.meta.phase_index:  1..6, the phase of the frame (auxiliary)
observation.meta.track_valid:  [endoscope, drill] 1 = a tracker sample within 125 ms of the frame (else the pose is held)
observation.meta.tool.<name>:  the instrument as text
```

The marker frame of an instrument is fixed to the instrument but not aligned with its shaft; the orientation is therefore comparable between frames and episodes of the same instrument calibration, not an axis convention.

---

## ⏱️ Data Synchronization Approach

All streams share one clock. The tracker stamps every camera frame with its exposure time on the host's Unix clock (`tExp`, the camera's hardware timestamp mapped to host time by the camera driver) and sends the two instrument poses of that frame; the recorder stores them, the drill trigger and the bone-cut events with that stamp. The dataset's frame timeline is a fixed grid on that clock: frame k of a session is at `tExp(first frame) + k / fps`. Onto it, positions are interpolated linearly and orientations by normalised linear interpolation between the two neighbouring tracker samples (11-33 ms apart, within 0.01 degrees of slerp); gaps over 250 ms are held, not bridged, and flagged in `observation.meta.track_valid`; the trigger is a zero-order hold.

The endoscope image of frame k is rendered by the viewer for the time of frame k, so image and kinematics refer to the same instant by construction (there is no separate video clock). $rendering

Timestamps: `timestamp` = `frame_index / fps`. `observation.meta.host_stamp_ns` is the frame's grid time (Unix-epoch nanoseconds), `observation.meta.endoscope_stamp_ns` and `observation.meta.drill_stamp_ns` are the stamps of the tracker samples the frame's poses start from. Synchronization tolerance `tolerance_s` = $tolerance s.

---

## 📦 Splits

`meta/info.json` splits (episode index ranges): $splits.

---

## 👥 Attribution & Contact

| | |
| :--- | :--- |
| **Dataset Lead** | TODO(team) |
| **Institution** | TODO(team) |
| **Contact Email** | TODO(team) |
| **Citation (BibTeX)** | TODO(team) |
