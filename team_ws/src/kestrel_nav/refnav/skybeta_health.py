"""/estimator/healthy on skybeta (deploy branch kestral_nav934am; not upstream group119).

On the FlyX lab the team's own vision bridge (onboard/vio/ev_bridge.py) publishes this. On skybeta the
organizer's bridge feeds PX4 and a second vision publisher would corrupt the EKF, so this node only
reports health, with the bridge's rule: cuVSLAM odometry passes VioHealth (no impossible jumps or speeds,
a quiet interval after a bad sample), tracking status is success, and the last good sample is fresher
than health_stale_s. It publishes NOTHING to PX4."""
import math
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool

from onboard.vio.vio_health import VioHealth


class SkybetaHealth(Node):
    def __init__(self):
        super().__init__("skybeta_health")
        dp = self.declare_parameter
        self.stale_s = float(dp("health_stale_s", 0.5).value)
        self.health = VioHealth(max_speed_mps=float(dp("max_speed_mps", 5.0).value),
                                jump_m=float(dp("jump_m", 1.0).value), quiet_s=float(dp("health_quiet_s", 5.0).value))
        self.pub = self.create_publisher(Bool, dp("health_topic", "/estimator/healthy").value, 10)
        self.create_subscription(Odometry, dp("odom_topic", "/visual_slam/tracking/odometry").value, self._odom, 10)
        self.tracking_ok, self.last_good, self.reason = None, None, "no samples"
        try:
            from isaac_ros_visual_slam_interfaces.msg import VisualSlamStatus
            self.create_subscription(VisualSlamStatus, "/visual_slam/status", self._status, 10)
        except ImportError:
            self.get_logger().warning("no VisualSlamStatus type: tracking status not checked")
            self.tracking_ok = True
        self.create_timer(0.2, self._publish)

    def _status(self, m):
        self.tracking_ok = int(m.vo_state) == 1

    def _odom(self, m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if t - getattr(self, "_t_eval", -1e9) < 0.25:     # <= 4 Hz: 30 Hz cost ~11 % of a Jetson-sized budget; 4 Hz
                                                          # keeps the newest good sample well inside health_stale_s 0.5
            return
        self._t_eval = t
        p = m.pose.pose.position
        ok, self.reason = self.health.update(t, (p.x, p.y, p.z))
        if ok:
            self.last_good = time.monotonic()

    def _healthy(self):
        return bool(self.tracking_ok and self.last_good is not None and 0 <= time.monotonic() - self.last_good < self.stale_s)

    def _publish(self):
        self.pub.publish(Bool(data=self._healthy()))


def main(args=None):
    rclpy.init(args=args)
    node = SkybetaHealth()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
