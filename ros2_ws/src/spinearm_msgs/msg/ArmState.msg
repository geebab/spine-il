# One arm-server tick (lerobot_robot_spinearm server snapshot). Angles in degrees, *arm* angles (as the servos report
# them); add the calibration's joint_offset_deg for model (URDF) angles. Times: wall-clock seconds.
std_msgs/Header header        # stamp = encoder read (middle of the read), frame_id = "<arm>_base_link"
uint64 tick
string state                  # OFF | HOLD | MOVING | HOMING | FAULT | ESTOP
bool torque
bool bus_ok
string[] faults
string[] name                 # joint names (JOINTS order)
float64[] position            # encoder
float64[] goal                # client target after clamping (NaN if none)
float64[] command             # per-tick command the server wrote to the bus (NaN if none)
int32[] load
int32[] temperature
float64 step_deg
int32 accel
float64 t_read                # position read done
float64 t_write               # goal write done (0 if no write this tick)
int64 goal_seq                # client seq of the goal that write carried (-1 if none)
float64 goal_rx               # when the server received that goal
