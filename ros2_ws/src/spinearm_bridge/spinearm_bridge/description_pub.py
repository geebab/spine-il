"""description_pub — republishes the arm's robot_description every few seconds (same topic, transient-local).

robot_state_publisher publishes robot_description once, latched. A viewer that misses that single delivery — seen
with Foxglove's URDF layer (topic source) over foxglove_bridge on 2026-09-26: "Invalid topic" although the bridge had
subscribed — never gets the model. Repeating the identical string costs nothing on the viewer side (Foxglove skips a
reload when the URDF text is unchanged) and lets a late or missed subscription recover within one period.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class DescriptionPub(Node):
    def __init__(self):
        super().__init__("description_pub")
        self.msg = String(data=self.declare_parameter("robot_description", "").value)
        period = float(self.declare_parameter("period_s", 2.0).value)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(String, "robot_description", qos)
        self.create_timer(period, lambda: self.pub.publish(self.msg))
        self.pub.publish(self.msg)


def main():
    rclpy.init()
    rclpy.spin(DescriptionPub())


if __name__ == "__main__":
    main()
