"""Cage hop test: up, forward, back, down. The first flight on a new airframe.

    take off, climb to climb_height_m       (default 1.0 m)
    fly forward_m along the nose            (default 1.0 m)
    fly back to above the start point
    descend to low_height_m                 (default 0.5 m)
    touch down and let PX4 disarm           (touchdown:=false stops at low_height_m
                                             and leaves the landing to the pilot)

Each leg is held for ``hold_seconds`` at its end. Same contract as
competitor_example's ``square_mission.py``: it reads ``/skybeta/state`` and cuVSLAM odometry,
publishes ``/skybeta/setpoint/pose`` (and, for the touchdown only,
``/skybeta/setpoint/twist``), and ``flight_interface`` owns everything else.
The pilot arms with the switch on Altitude, then switches to Position, and
this node takes off from there.

Why it is not just four waypoints
---------------------------------
* **The setpoint moves at ``speed_mps``.** PX4 flies an Offboard position
  setpoint with no trajectory smoothing (``PositionControl::_positionControl``
  is a P controller capped at ``MPC_XY_VEL_MAX`` = 12 m/s on these airframes),
  so the published point slides along each leg instead of jumping to its end.
  It stops sliding while odometry is stale.
* **"Forward" is the nose at takeoff**, from the odometry orientation, not
  odom +x. They differ if the vehicle was turned after the stack came up.
* **Touchdown is a twist.** Every pose is floored at ``min_altitude_m``, and
  in Offboard PX4's land detector only reports ground contact while the
  setpoint commands a descent of at least 1.1 x ``LNDMC_Z_VEL_MAX`` (0.275 m/s
  with the fleet's 0.25), which a pose never does (its velocity is NaN).
  So the last ``low_height_m`` is flown as a straight-down twist at
  ``descend_speed_mps``, refreshed every tick until PX4 disarms
  (``COM_DISARM_LAND`` = 2 s after landing). Horizontally that last part is a
  zero-velocity hold, not a position hold, which is why it starts low.

It starts only on a switch into Offboard it saw happen, so a node restarted in
flight never starts a hop from the air. After a disarm it is ready again: arm,
switch to Position, and it flies the same hop from wherever the vehicle sits.
"""
import json
import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from cage_test.hop_plan import (
    build_legs,
    carrot,
    leg_length,
    yaw_from_xyzw,
)

#: Above this, cuVSLAM odometry (30 Hz) counts as stale and the setpoint stops moving.
ODOM_STALE_S = 0.5
#: PX4's land detector needs > 1.1 x LNDMC_Z_VEL_MAX (0.25 on the fleet) of commanded descent.
MIN_TOUCHDOWN_SPEED_MPS = 0.3


class HopTest(Node):

    def __init__(self, **kwargs):
        super().__init__('hop_test', **kwargs)

        self.declare_parameter('climb_height_m', 1.0)
        self.declare_parameter('forward_m', 1.0)
        self.declare_parameter('low_height_m', 0.5)
        self.declare_parameter('speed_mps', 0.3)
        self.declare_parameter('hold_seconds', 3.0)
        self.declare_parameter('arrival_radius_m', 0.2)
        # Extra time per leg on top of length / speed before giving up on it.
        self.declare_parameter('leg_timeout_s', 15.0)
        self.declare_parameter('touchdown', True)
        self.declare_parameter('descend_speed_mps', 0.4)
        # After this long without PX4 reporting landed, tell the pilot to disarm.
        self.declare_parameter('touchdown_timeout_s', 15.0)
        self.declare_parameter('publish_rate_hz', 10.0)

        def param(name):
            return self.get_parameter(name).value

        self.climb = float(param('climb_height_m'))
        self.forward = float(param('forward_m'))
        self.low = float(param('low_height_m'))
        self.speed = float(param('speed_mps'))
        self.hold_s = float(param('hold_seconds'))
        self.arrival_radius = float(param('arrival_radius_m'))
        self.leg_timeout = float(param('leg_timeout_s'))
        self.touchdown = bool(param('touchdown'))
        self.descend_speed = float(param('descend_speed_mps'))
        self.touchdown_timeout = float(param('touchdown_timeout_s'))
        rate_hz = float(param('publish_rate_hz'))
        self.dt = 1.0 / rate_hz

        if self.speed <= 0.0:
            raise ValueError('speed_mps must be > 0')
        if not 0.0 < self.low < self.climb:
            raise ValueError('need 0 < low_height_m < climb_height_m')
        if self.touchdown and self.descend_speed < MIN_TOUCHDOWN_SPEED_MPS:
            raise ValueError(
                f'descend_speed_mps must be >= {MIN_TOUCHDOWN_SPEED_MPS} or '
                f'PX4 never detects the landing')

        self.pose_pub = self.create_publisher(
            PoseStamped, '/skybeta/setpoint/pose', 10)
        self.twist_pub = self.create_publisher(
            TwistStamped, '/skybeta/setpoint/twist', 10)
        self.create_subscription(
            Odometry, '/visual_slam/tracking/odometry', self.odometry_cb, 10)
        self.create_subscription(String, '/skybeta/state', self.state_cb, 10)

        self.position = None
        self.yaw = None
        self.odom_time = None
        self.offboard_active = False
        self.armed = False
        self.landed = None
        self.px4_link = False
        self.local_valid = False
        self.envelope = None
        self.log_once = set()
        self.last_readiness_log = None
        self.reset()

        self.create_timer(self.dt, self.tick)
        self.get_logger().info(
            f'hop_test ready: up {self.climb:.2f} m, forward {self.forward:.2f} m, '
            f'back, down to {self.low:.2f} m'
            f'{", then touch down" if self.touchdown else ", then hold for the pilot"}'
            f'; {self.speed:.2f} m/s, {self.hold_s:.0f} s holds.\n'
            'Waiting for the pilot to arm (switch on Altitude), then switch to '
            'Position...')

    def reset(self):
        """Back to waiting for the next arm -> Position."""
        self.state = 'WAIT_FOR_OFFBOARD'
        # Only a switch into Offboard seen from outside it starts a hop.
        self.seen_not_offboard = False
        self.start = None
        self.legs = []
        self.leg_index = 0
        self.progress = 0.0
        self.leg_started_at = None
        self.hold_started_at = None
        self.touchdown_started_at = None
        self.hold_target = None

    # ------------------------------------------------------------------
    # inputs
    # ------------------------------------------------------------------
    def odometry_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.position = (float(p.x), float(p.y), float(p.z))
        self.yaw = yaw_from_xyzw(q.x, q.y, q.z, q.w)
        self.odom_time = self.now_s()

    def state_cb(self, msg):
        try:
            state = json.loads(msg.data)
        except ValueError:
            return
        self.offboard_active = bool(state.get('offboard_active', False))
        self.armed = bool(state.get('armed', False))
        self.landed = state.get('landed')
        self.px4_link = bool(state.get('px4_link', False))
        self.local_valid = state.get('local_enu') is not None
        self.envelope = state.get('envelope')
        if state.get('clamped'):
            self.info_once(
                f'clamped:{",".join(state["clamped"])}',
                f'flight_interface is CLAMPING the setpoint '
                f'({", ".join(state["clamped"])}), envelope {self.envelope}: '
                f'a clamped point is never reached. Check climb_height_m and '
                f'low_height_m against it.')

    def info_once(self, key, message):
        if key not in self.log_once:
            self.log_once.add(key)
            self.get_logger().info(message)

    # ------------------------------------------------------------------
    # outputs
    # ------------------------------------------------------------------
    def send_pose(self, x, y, z):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        # All-zero quaternion == "do not command yaw": PX4 keeps its heading.
        msg.pose.orientation.w = 0.0
        self.pose_pub.publish(msg)

    def send_descent(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.twist.linear.z = -self.descend_speed
        self.twist_pub.publish(msg)

    # ------------------------------------------------------------------
    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def odom_fresh(self):
        return (self.odom_time is not None
                and self.now_s() - self.odom_time <= ODOM_STALE_S)

    def arrived_at(self, target):
        return math.dist(self.position, target) <= self.arrival_radius

    def fail(self, message):
        """Stop advancing and hold where the setpoint is now."""
        self.get_logger().error(f'{message} -- holding. Pilot: take over and land.')
        self.state = 'FAILED'

    # ------------------------------------------------------------------
    # the state machine
    # ------------------------------------------------------------------
    def tick(self):
        if self.position is None:
            self.info_once('no_odom', 'no /visual_slam/tracking/odometry yet -- '
                           'nothing is commanded until cuVSLAM is up.')
            return

        if not self.armed and self.state != 'WAIT_FOR_OFFBOARD':
            self.get_logger().info(
                f'disarmed (was {self.state}). Ready for the next hop: arm, '
                f'then switch to Position.')
            self.reset()

        flying = self.state in ('LEG', 'TOUCHDOWN', 'FAILED', 'DONE')
        if flying and not self.offboard_active:
            self.get_logger().warn(
                f'Offboard lost during {self.state} -- the pilot has the '
                f'vehicle. Nothing more until it is disarmed.')
            self.state = 'ABORTED'

        getattr(self, f'_state_{self.state.lower()}')()

    def log_readiness(self):
        """Every 5 s while waiting: what flight_interface's go check will see."""
        now = self.now_s()
        if self.last_readiness_log is not None and now - self.last_readiness_log < 5.0:
            return
        self.last_readiness_log = now
        checks = {
            'px4_link': self.px4_link,
            'vio': self.odom_fresh(),
            'px4_position': self.local_valid,
            'landed': bool(self.landed),
        }
        missing = [name for name, ok in checks.items() if not ok]
        if missing:
            self.get_logger().warn(f'NOT ready: {", ".join(missing)} false')
        else:
            self.get_logger().info(
                'ready: arm (switch on Altitude), then switch to Position. '
                f'armed={self.armed}')

    def _state_wait_for_offboard(self):
        # Armed too: after an auto-disarm PX4 can still report Offboard.
        if not (self.offboard_active and self.armed):
            if not self.offboard_active:
                self.seen_not_offboard = True
            self.log_readiness()
            # Where we are now; the interface floors it at min_altitude_m. This
            # stream is also what lets flight_interface enter Offboard at all.
            self.send_pose(*self.position)
            return
        if not self.seen_not_offboard:
            self.info_once(
                'offboard_at_start',
                'Offboard was already active when this node started -- not '
                'starting a hop from the air. Disarm to reset.')
            self.send_pose(*self.position)
            return
        self.start = self.position
        self.legs = build_legs(self.start, self.yaw, self.climb, self.forward,
                               self.low, self.hold_s)
        self.begin_leg(0)
        fwd = self.legs[1].end
        self.get_logger().info(
            f'Offboard active. Start ENU ({self.start[0]:.2f}, '
            f'{self.start[1]:.2f}, {self.start[2]:.2f}), heading '
            f'{math.degrees(self.yaw):.0f} deg; forward point '
            f'({fwd[0]:.2f}, {fwd[1]:.2f}). Leg 1/{len(self.legs)}: up.')

    def begin_leg(self, index):
        self.state = 'LEG'
        self.leg_index = index
        self.progress = 0.0
        self.leg_started_at = self.now_s()
        self.hold_started_at = None

    def _state_leg(self):
        leg = self.legs[self.leg_index]
        length = leg_length(leg)
        if self.odom_fresh():
            self.progress = min(self.progress + self.speed * self.dt, length)
        else:
            self.info_once(f'stale:{self.leg_index}',
                           f'odometry stale on leg "{leg.name}" -- setpoint '
                           f'paused until it returns')
        self.hold_target = carrot(leg, self.progress)
        self.send_pose(*self.hold_target)

        if self.hold_started_at is None:
            if self.progress >= length and self.arrived_at(leg.end):
                self.hold_started_at = self.now_s()
                self.get_logger().info(
                    f'leg "{leg.name}" reached, holding {leg.hold_s:.0f} s')
            elif self.now_s() - self.leg_started_at > length / self.speed + self.leg_timeout:
                self.fail(
                    f'leg "{leg.name}" not reached in time (still '
                    f'{math.dist(self.position, leg.end):.2f} m off). Check '
                    f'"clamped" in /skybeta/state and cuVSLAM tracking')
            return
        if self.now_s() - self.hold_started_at < leg.hold_s:
            return

        if self.leg_index + 1 < len(self.legs):
            self.begin_leg(self.leg_index + 1)
            self.get_logger().info(
                f'leg {self.leg_index + 1}/{len(self.legs)}: '
                f'{self.legs[self.leg_index].name}')
        elif self.touchdown:
            self.state = 'TOUCHDOWN'
            self.touchdown_started_at = self.now_s()
            self.get_logger().info(
                f'touching down at {self.descend_speed:.2f} m/s; PX4 disarms '
                f'once it detects the landing')
        else:
            self.state = 'DONE'
            self.get_logger().info(
                f'hop complete -- holding at {self.low:.2f} m. Pilot: take '
                f'over and land.')

    def _state_touchdown(self):
        # Refreshed every tick: a twist expires after setpoint_timeout_s and
        # flight_interface then holds the last pose, i.e. climbs back up.
        self.send_descent()
        if self.landed:
            self.info_once('landed', 'PX4 reports landed -- waiting for the '
                           'auto-disarm (COM_DISARM_LAND)')
            return
        # Keep descending rather than fall back to a pose: on the ground with the
        # landing undetected, a pose at low_height_m would take off again.
        if self.now_s() - self.touchdown_started_at > self.touchdown_timeout:
            self.info_once(
                'no_landing',
                f'no landing detected after {self.touchdown_timeout:.0f} s, '
                f'still descending. Pilot: if it is on the ground, disarm.')

    def _state_failed(self):
        self.send_pose(*self.hold_target)

    def _state_done(self):
        self.send_pose(*self.hold_target)

    def _state_aborted(self):
        # PX4 ignores this outside Offboard; it only keeps the stream alive.
        self.send_pose(*self.position)


def main(args=None):
    rclpy.init(args=args)
    node = HopTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
