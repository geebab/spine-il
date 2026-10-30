# One tool of one tracker camera frame (tracker_service.py --tap). Metres, tool → camera.
std_msgs/Header header        # stamp = mid-exposure of the camera frame, frame_id = "tracker_cam"
string tool                   # tracker name: endo | endo2 | drill (endo/endo2 follow screen side — not identity)
uint64 frame                  # camera frame number
bool lost
geometry_msgs/Pose pose       # filtered (what the viewer gets)
geometry_msgs/Pose pose_raw   # before the tracker filter
geometry_msgs/Point tip
geometry_msgs/Point tip_raw
uint8 visible                 # balls seen
float32 rmse
bool still
bool held
float64 t_arrival             # frame reached the tracker (wall s)
float64 t_pose                # this tool's pose computed
float64 t_send                # sent
