"""Disarmed hand-carry check: do VIO and PX4 agree on forward, left, up and yaw?

Run it on the running stack with the vehicle **disarmed**, props off:

    host/run.sh shell
    ros2 run cage_test frame_check

It publishes nothing. The pose when it starts is the reference; restart it to
re-zero. Twice a second it prints the displacement since then, resolved along
the vehicle's heading at the start, from two sources:

    VIO   /visual_slam/tracking/odometry       what hop_test flies on
    PX4   /skybeta/state local_enu + /fmu/out/vehicle_attitude
                                               what EKF2 made of it

Carry the vehicle, keeping it level, and read both lines:

    1 m along the nose          fwd +1.00  left  0.00
    1 m to the right            fwd  0.00  left -1.00
    lift 0.5 m                  up  +0.50
    90 deg clockwise from above yaw -90

If VIO is right but rotated (forward shows as left), the camera mount yaw in
drone.env is wrong. If VIO is right and PX4 is not, the problem is between
the bridge and EKF2. The ``yaw offset`` column is PX4's heading minus VIO's;
with EV yaw fused it should stay constant (ideally 0) through the turn. A
drifting offset is the toilet-bowl mechanism in CLAUDE.md.

``pos offset`` is PX4's local position minus VIO's, absolute (ENU, metres).
hop_test sends absolute VIO positions and flight_interface passes them to PX4
unanchored, so a horizontal offset here is a jump of that size the moment
Offboard starts. Keep it under ~0.1 m, or launch hop_test with
``anchor_setpoints:=true``. A z offset moves the whole hop up or down by it.
"""
import json
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

from nav_msgs.msg import Odometry
from std_msgs.msg import String

from px4_msgs.msg import VehicleAttitude

from skybeta_flight_interface.frames import wrap_pi, yaw_from_wxyz, yaw_ned_to_enu
from skybeta_flight_interface.topics import fmu_topic

from cage_test.hop_plan import along_heading, yaw_from_xyzw


class FrameCheck(Node):

    def __init__(self, **kwargs):
        super().__init__('frame_check', **kwargs)
        px4_sub_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        attitude_topic = fmu_topic('out', 'vehicle_attitude')
        self.create_subscription(
            Odometry, '/visual_slam/tracking/odometry', self.odometry_cb, 10)
        self.create_subscription(String, '/skybeta/state', self.state_cb, 10)
        self.create_subscription(
            VehicleAttitude, attitude_topic, self.attitude_cb, px4_sub_qos)

        self.vio = None          # (position, yaw) now
        self.px4_position = None
        self.px4_yaw = None
        self.vio_ref = None      # (position, yaw) at start
        self.px4_ref = None
        self.armed = False

        self.create_timer(0.5, self.report)
        self.get_logger().info(
            f'frame_check: VIO from /visual_slam/tracking/odometry, PX4 from '
            f'/skybeta/state and {attitude_topic}. Keep the vehicle still '
            f'until both references are captured, then carry it.')

    def odometry_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.vio = ((p.x, p.y, p.z), yaw_from_xyzw(q.x, q.y, q.z, q.w))

    def attitude_cb(self, msg):
        # PX4 q is NED/FRD wxyz; its heading in ENU is pi/2 - yaw_ned.
        self.px4_yaw = yaw_ned_to_enu(yaw_from_wxyz(tuple(msg.q)))

    def state_cb(self, msg):
        try:
            state = json.loads(msg.data)
        except ValueError:
            return
        self.armed = bool(state.get('armed', False))
        local = state.get('local_enu')
        self.px4_position = tuple(local) if local else None

    @staticmethod
    def line(name, now, ref):
        fwd, left, up = along_heading(now[0], ref[0], ref[1])
        yaw = math.degrees(wrap_pi(now[1] - ref[1]))
        return (f'{name}  fwd {fwd:+5.2f}  left {left:+5.2f}  up {up:+5.2f}  '
                f'yaw {yaw:+5.0f}')

    def report(self):
        if self.armed:
            self.get_logger().warn('ARMED -- this is a disarmed hand-carry check')
        if self.vio is not None and self.vio_ref is None:
            self.vio_ref = self.vio
        px4 = None
        if self.px4_position is not None and self.px4_yaw is not None:
            px4 = (self.px4_position, self.px4_yaw)
            if self.px4_ref is None:
                self.px4_ref = px4

        parts = []
        if self.vio_ref is not None:
            parts.append(self.line('VIO', self.vio, self.vio_ref))
        else:
            parts.append('VIO  no odometry (is cuVSLAM tracking?)')
        if px4 is not None:
            parts.append(self.line('PX4', px4, self.px4_ref))
        else:
            parts.append('PX4  no local position / attitude yet')
        if px4 is not None and self.vio is not None:
            offset = math.degrees(wrap_pi(px4[1] - self.vio[1]))
            parts.append(f'yaw offset {offset:+5.1f}')
            d = [a - b for a, b in zip(px4[0], self.vio[0])]
            parts.append(f'pos offset {d[0]:+5.2f} {d[1]:+5.2f} {d[2]:+5.2f}')
        self.get_logger().info('   |   '.join(parts))


def main(args=None):
    rclpy.init(args=args)
    node = FrameCheck()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
