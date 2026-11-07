"""Fly the planned DroneHacks course through the skybeta flight interface.

The skybeta contract (copied from competitor_example/square_mission.py, the file skybeta says to copy):

    subscribe  /skybeta/state                     JSON: offboard_active, armed, landed, clamped, ...
    subscribe  /visual_slam/tracking/odometry     where we are (ENU, cuVSLAM ``odom`` frame)
    publish    /skybeta/setpoint/pose             where we want to be (ENU, ``odom`` frame)

Nothing here touches /fmu/*, arms, or changes mode. The pilot arms and switches to Position;
``skybeta_flight_interface`` then enters Offboard once, and this node takes off from the ground.

What this node adds on top of the example
-----------------------------------------
1. At start-up it plans the whole course with ``coursenav`` (A* on the published OBJ, every chicane
   option scored, best chosen) from a start pose in the COURSE frame: the ``start_*`` parameters, or
   the course file's prior (the launch mark, heading along the guideline) when they are NaN.
2. When Offboard becomes active it records the vehicle's odom pose ``p0`` and converts the plan into
   odom-frame waypoints: ``planner.to_route`` gives each waypoint as (forward, left) from the start
   pose, which is rotated by the vehicle's odom yaw and offset by ``p0``. The course->odom transform
   is therefore fixed by the start pose unless ``use_localizer`` is true, in which case every
   ``/coursenav/pose`` (course frame, from kestrel_nav.coursenav.loc_node) paired with the odometry at
   that moment re-estimates it.
3. It flies the waypoints at ``cruise_z_m - landed_z_m`` above the odom origin, yaw along each leg (the
   camera looks where it flies), pausing ``hold_seconds`` at gate/tunnel/align legs. Over the pad it
   descends to ``min_altitude_m`` and holds: LANDING IS THE PILOT'S (the interface floors altitude).

Frames: course ENU = the OBJ's frame (origin floor centre). odom ENU = cuVSLAM's, origin at the rig's
power-on pose with x along the vehicle's nose at that moment. Heights: course z is above the floor; odom
z is above the landed body origin, hence ``landed_z_m``.

NOT FLIGHT-TESTED. Written against skybeta 0.3.2 docs and the workspace planner; fly it first with
``mission:=true`` on the bench (props off) and watch /skybeta/state for clamps.
"""
from __future__ import annotations

import json
import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String


def yaw_from_quat(q):
    """ENU yaw (rad) of a geometry_msgs Quaternion."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class CourseMission(Node):
    """Plan once, then a small state machine on a 10 Hz timer."""

    def __init__(self, **kwargs):
        super().__init__('course_mission', **kwargs)
        dp = self.declare_parameter
        self.chicane = str(dp('chicane', 'auto').value)
        self.start_x = float(dp('start_x', float('nan')).value)
        self.start_y = float(dp('start_y', float('nan')).value)
        self.start_yaw_deg = float(dp('start_yaw_deg', float('nan')).value)
        self.landed_z = float(dp('landed_z_m', 0.15).value)
        self.arrival_radius = float(dp('arrival_radius_m', 0.25).value)
        self.hold_seconds = float(dp('hold_seconds', 1.0).value)
        self.use_localizer = bool(dp('use_localizer', False).value)
        self.leg_timeout = float(dp('leg_timeout_s', 45.0).value)
        self.min_alt = float(dp('min_altitude_m', 0.3).value)
        self.max_alt = float(dp('max_altitude_m', 3.0).value)
        self.course_file = str(dp('course_file', '').value)          # '' = the packaged course_v3.json
        rate_hz = float(dp('publish_rate_hz', 10.0).value)

        # --- the contract ---
        self.setpoint_pub = self.create_publisher(PoseStamped, '/skybeta/setpoint/pose', 10)
        self.create_subscription(Odometry, '/visual_slam/tracking/odometry', self.odometry_cb, 10)
        self.create_subscription(String, '/skybeta/state', self.state_cb, 10)
        if self.use_localizer:
            self.create_subscription(PoseWithCovarianceStamped, '/coursenav/pose', self.course_pose_cb, 10)
        self.plan_pub = self.create_publisher(String, '/kestrel_nav/plan', 1)

        # --- state ---
        self.position = None            # odom ENU (x, y, z)
        self.yaw = None                 # odom ENU yaw
        self.offboard_active = False
        self.was_offboard = False
        self.state = 'WAIT_FOR_OFFBOARD'
        self.p0 = None                  # odom pose at mission start (x, y, z, yaw)
        self.T = None                   # course -> odom: (tx, ty, dyaw): odom = R(dyaw) course + t
        self.targets = []               # [(x_odom, y_odom, z_odom, yaw_odom, leg dict)]
        self.k = 0
        self.leg_started = None
        self.hold_until = None
        self.log_once = set()

        # --- plan the course now; a failure here must be loud before anything arms ---
        self.plan = self.plan_course()
        self.create_timer(1.0 / rate_hz, self.tick)

    # ------------------------------------------------------------------ planning
    def plan_course(self):
        from kestrel_nav.coursenav import planner
        from kestrel_nav.coursenav.course import load_course, plan_grids, resolve
        from kestrel_nav.coursenav.objmap import CourseMap

        doc = load_course(self.course_file) if self.course_file else load_course()
        cmap = CourseMap.load(resolve(doc, 'map'))
        obst, blocked, clearance = plan_grids(cmap, doc)
        if math.isnan(self.start_x) or math.isnan(self.start_y) or math.isnan(self.start_yaw_deg):
            sx, sy = doc['start'].get('spawn_xy', doc['start']['xy'])
            syaw = float(doc['start']['yaw_deg'])
            source = 'course-file prior (launch mark)'
        else:
            sx, sy, syaw = self.start_x, self.start_y, self.start_yaw_deg
            source = 'parameters'
        if self.chicane == 'auto':
            res = planner.plan_options(doc, obst, blocked, clearance, start_xy=(sx, sy), cmap=cmap)
            p, reason = res['plan'], res['reason']
        else:
            p = planner.plan(doc, obst, blocked, clearance, start_xy=(sx, sy), chicane=self.chicane, cmap=cmap)
            reason = 'forced'
        wps = planner.flight_waypoints(p['waypoints'])
        legs = planner.to_legs(p['waypoints'])
        cruise = float(doc['cruise_z_m'])
        z_odom = cruise - self.landed_z
        if not (self.min_alt <= z_odom <= self.max_alt):
            self.get_logger().error(
                f'cruise {cruise} m above the floor is {z_odom:.2f} m above the odom origin, outside the '
                f'interface band [{self.min_alt}, {self.max_alt}]: it WILL be clamped. Fix landed_z_m.')
        plan = {'start_course': (float(sx), float(sy), math.radians(syaw)), 'start_source': source,
                'chicane': p.get('chicane'), 'reason': reason, 'length_m': p['length_m'],
                'min_clearance_m': p['min_clearance_m'], 'time_estimate_s': p['time_estimate_s'],
                'predicted_false_blocks': p.get('predicted_false_blocks', []),
                'waypoints_course': [(w['name'], float(w['xy'][0]), float(w['xy'][1])) for w in wps],
                'legs': legs, 'z_odom': z_odom, 'cruise_z_m': cruise,
                'pad_radius_m': float(doc['sequence'][-1].get('radius_m', 0.5))}
        self.get_logger().info(
            f"planned from {source} ({sx:.2f}, {sy:.2f}, {syaw:.1f} deg): chicane {plan['chicane']} "
            f"({reason}); {len(wps) - 1} legs, {p['length_m']:.1f} m, min clearance {p['min_clearance_m']:.2f} m, "
            f"~{p['time_estimate_s']:.0f} s; flying {z_odom:.2f} m above the odom origin")
        for name, x, y in plan['waypoints_course']:
            self.get_logger().info(f'   {name:14s} course ({x:6.2f}, {y:6.2f})')
        self.plan_pub.publish(String(data=json.dumps({k: v for k, v in plan.items()}, default=str)))
        return plan

    # ------------------------------------------------------------------ frames
    def set_transform_from_start(self):
        """course->odom from: vehicle is at course start pose S and at odom pose p0 right now."""
        sx, sy, syaw = self.plan['start_course']
        x0, y0, _, yaw0 = self.p0
        dyaw = wrap(yaw0 - syaw)
        c, s = math.cos(dyaw), math.sin(dyaw)
        self.T = (x0 - (c * sx - s * sy), y0 - (s * sx + c * sy), dyaw)

    def course_pose_cb(self, msg):
        """Localizer says we are at course pose C while odometry says odom pose O: re-fit course->odom."""
        if self.position is None or self.yaw is None:
            return
        cx, cy = msg.pose.pose.position.x, msg.pose.pose.position.y
        cyaw = yaw_from_quat(msg.pose.pose.orientation)
        dyaw = wrap(self.yaw - cyaw)
        c, s = math.cos(dyaw), math.sin(dyaw)
        new = (self.position[0] - (c * cx - s * cy), self.position[1] - (s * cx + c * cy), dyaw)
        if self.T is None:
            self.T = new
        else:   # low-pass: the PF is already filtered, this only removes callback jitter
            a = 0.2
            self.T = (self.T[0] + a * (new[0] - self.T[0]), self.T[1] + a * (new[1] - self.T[1]),
                      wrap(self.T[2] + a * wrap(new[2] - self.T[2])))
        if self.state in ('FLY', 'HOLD_AT') and self.targets:
            self.build_targets()

    def course_to_odom(self, x, y):
        tx, ty, dyaw = self.T
        c, s = math.cos(dyaw), math.sin(dyaw)
        return (c * x - s * y + tx, s * x + c * y + ty)

    def build_targets(self):
        z = self.plan['z_odom']
        pts = [self.course_to_odom(x, y) for _, x, y in self.plan['waypoints_course']]
        out = []
        for i in range(1, len(pts)):
            (xa, ya), (xb, yb) = pts[i - 1], pts[i]
            yaw = math.atan2(yb - ya, xb - xa) if math.hypot(xb - xa, yb - ya) > 0.05 else None
            out.append((xb, yb, z, yaw, self.plan['legs'][i - 1]))
        self.targets = out

    # ------------------------------------------------------------------ inputs
    def odometry_cb(self, msg):
        p = msg.pose.pose.position
        self.position = (float(p.x), float(p.y), float(p.z))
        self.yaw = yaw_from_quat(msg.pose.pose.orientation)

    def state_cb(self, msg):
        try:
            st = json.loads(msg.data)
        except ValueError:
            return
        self.offboard_active = bool(st.get('offboard_active', False))
        if st.get('clamped'):
            self.info_once(f"clamped:{','.join(st['clamped'])}",
                           f"flight interface is CLAMPING our setpoints ({', '.join(st['clamped'])}); "
                           f"envelope {st.get('envelope')}")

    def info_once(self, key, message):
        if key not in self.log_once:
            self.log_once.add(key)
            self.get_logger().info(message)

    # ------------------------------------------------------------------ output
    def send(self, x, y, z, yaw_enu=None):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = float(x), float(y), float(z)
        if yaw_enu is None:
            msg.pose.orientation.w = 0.0            # all-zero quaternion: do not command yaw
        else:
            msg.pose.orientation.z = math.sin(float(yaw_enu) / 2.0)
            msg.pose.orientation.w = math.cos(float(yaw_enu) / 2.0)
        self.setpoint_pub.publish(msg)

    # ------------------------------------------------------------------ 10 Hz
    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def dist_xy(self, t):
        return math.hypot(self.position[0] - t[0], self.position[1] - t[1])

    def tick(self):
        if self.position is None:
            self.info_once('no_odom', 'no /visual_slam/tracking/odometry yet: nothing is commanded until cuVSLAM tracks')
            return
        if self.was_offboard and not self.offboard_active:
            self.get_logger().warn('Offboard lost: the pilot has control. Mission paused (no automatic resume).')
            self.was_offboard = False
            self.state = 'PAUSED'
        if self.offboard_active and not self.was_offboard:
            self.was_offboard = True
        getattr(self, f'_state_{self.state.lower()}')()

    def _state_wait_for_offboard(self):
        if not self.offboard_active:
            self.info_once('waiting', 'holding here until the pilot arms, then switches to Position')
            self.send(*self.position)                     # the stream that lets the interface enter Offboard
            return
        self.p0 = (*self.position, self.yaw)
        if self.T is None:
            self.set_transform_from_start()
        self.build_targets()
        self.k, self.leg_started = 0, self.now_s()
        self.state = 'CLIMB'
        self.get_logger().info(f"Offboard active at odom ({self.p0[0]:.2f}, {self.p0[1]:.2f}, {self.p0[2]:.2f}) "
                               f"yaw {math.degrees(self.p0[3]):.0f} deg; climbing to z={self.plan['z_odom']:.2f}")

    def _state_climb(self):
        target = (self.p0[0], self.p0[1], self.plan['z_odom'])
        self.send(*target)
        if abs(self.position[2] - target[2]) <= self.arrival_radius:
            self.state = 'FLY'
            self.leg_started = self.now_s()
            self.announce_leg()

    def announce_leg(self):
        x, y, z, yaw, leg = self.targets[self.k]
        self.get_logger().info(f"leg {self.k + 1}/{len(self.targets)} -> {leg['name']} at odom ({x:.2f}, {y:.2f}, {z:.2f})"
                               + (f" [{leg['mode']}, hold]" if leg.get('hold') else ''))

    def _state_fly(self):
        x, y, z, yaw, leg = self.targets[self.k]
        self.send(x, y, z, yaw)
        if self.dist_xy((x, y)) <= self.arrival_radius:
            if self.k + 1 >= len(self.targets):
                self.state = 'PAD'
                self.get_logger().info(f'over the pad: descending to {self.min_alt:.2f} m and holding. Pilot lands.')
                return
            if leg.get('hold'):
                self.hold_until = self.now_s() + self.hold_seconds
                self.state = 'HOLD_AT'
                return
            self.k += 1
            self.leg_started = self.now_s()
            self.announce_leg()
        elif self.now_s() - self.leg_started > self.leg_timeout:
            self.get_logger().error(f"leg {self.k + 1} ({leg['name']}) not reached in {self.leg_timeout:.0f} s, still "
                                    f"{self.dist_xy((x, y)):.2f} m away: holding. Check /skybeta/state for a clamp.")
            self.state = 'HOLD'

    def _state_hold_at(self):
        x, y, z, yaw, _ = self.targets[self.k]
        self.send(x, y, z, yaw)
        if self.now_s() >= self.hold_until:
            self.k += 1
            self.leg_started = self.now_s()
            self.state = 'FLY'
            self.announce_leg()

    def _state_pad(self):
        x, y, _, yaw, _ = self.targets[-1]
        self.send(x, y, self.min_alt, yaw)

    def _state_hold(self):
        x, y, z, yaw, _ = self.targets[self.k] if self.targets else (*self.position, None, None)
        self.send(x, y, z, yaw)

    def _state_paused(self):
        self.send(*self.position)


def main(args=None):
    rclpy.init(args=args)
    node = CourseMission()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
