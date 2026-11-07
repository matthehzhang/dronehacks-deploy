"""Cage flight tests: the first hop, and the tuning patterns, with every run logged.

``hop_test`` (default ``sequence:=hop``) is the first flight:

    take off, climb to climb_height_m       (default 1.0 m)
    fly forward_m along the nose            (default 1.0 m)
    fly back to above the start point
    descend to low_height_m                 (default 0.5 m)
    touch down and let PX4 disarm           (touchdown:=false stops at low_height_m
                                             and leaves the landing to the pilot)

``tune_test`` is the same node with ``sequence:=hover,steps,speed,yaw``: any
comma list of the patterns in ``tune_plan`` (hop, hover, steps, speed, yaw,
vertical, square), flown between the same takeoff and landing. See
``tune_plan.py`` for what each one flies.

Same contract as competitor_example's ``square_mission.py``: it reads
``/skybeta/state`` and cuVSLAM odometry, publishes ``/skybeta/setpoint/pose``
(and, for the touchdown only, ``/skybeta/setpoint/twist``), and
``flight_interface`` owns everything else. The pilot arms with the switch on
Altitude, then switches to Position, and this node takes off from there.

Why it is not just waypoints
----------------------------
* **The setpoint moves at the segment's speed** (``move`` segments). PX4 flies
  an Offboard position setpoint with no trajectory smoothing (a P controller,
  ``MPC_XY_P`` = 0.95 on drone 16), so the published point slides along each
  leg. The vehicle trails it by about speed / MPC_XY_P: ~1 m at 1 m/s. ``step``
  segments jump on purpose: that is the step input for tuning.
* **"Forward" is the nose at takeoff**, from the odometry orientation.
* **Yaw is only commanded once a yaw segment starts**, from PX4's own heading
  at that moment (``vehicle_attitude``), so a VIO/PX4 yaw offset causes no
  jump. From then on every pose carries the commanded yaw.
* **Touchdown is a twist** (PX4's land detector needs a commanded descent of
  >= 1.1 x LNDMC_Z_VEL_MAX; a pose never commands one).
* **The sequence stops** (holds, pilot lands) when the vehicle is more than
  ``abort_distance_m`` off the current segment's path. Lagging behind the
  moving setpoint does not count; drifting sideways or overshooting does.

It starts only on a switch into Offboard it saw happen, so a node restarted in
flight never starts a sequence from the air. After a disarm it is ready again.

Every run (arm -> disarm + ``post_roll_s``) is written to
``<log_dir>/<time>_<sequence>/``: commands, VIO, PX4 pose, both IMUs, battery,
events, a per-segment summary, and a rosbag2 mcap of every /fmu topic. See
``run_log.py`` and ``recorder.py``.
"""
import collections
import json
import math
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from rcl_interfaces.msg import ParameterDescriptor

from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import String

from px4_msgs.msg import BatteryStatus, SensorCombined, VehicleAttitude

from skybeta_flight_interface.frames import yaw_from_wxyz, yaw_ned_to_enu
from skybeta_flight_interface.topics import fmu_topic

from cage_test import analysis
from cage_test.recorder import BagRecorder, select_topics
from cage_test.run_log import PREROLL_S, RunLog, default_log_dir, free_gb
from cage_test.tune_plan import (
    DEFAULTS,
    build_plan,
    carrot,
    distance_to_path,
    plan_duration,
    seg_length,
    validate,
    wrap_pi,
    xyzw_from_yaw,
    yaw_from_xyzw,
)

#: Above this, cuVSLAM odometry (30 Hz) counts as stale and the setpoint stops moving.
ODOM_STALE_S = 0.5
#: PX4 attitude older than this is not used as the yaw reference.
ATTITUDE_STALE_S = 0.5
#: PX4's land detector needs > 1.1 x LNDMC_Z_VEL_MAX (0.25 on the fleet) of commanded descent.
MIN_TOUCHDOWN_SPEED_MPS = 0.3

#: Parameters on top of tune_plan.DEFAULTS: name -> default.
NODE_DEFAULTS = {
    'sequence': 'hop',
    'arrival_radius_m': 0.2,
    'yaw_tolerance_deg': 10.0,
    'leg_timeout_s': 15.0,          # extra time per segment before giving up on it
    'abort_distance_m': 0.6,
    'touchdown': True,
    'descend_speed_mps': 0.4,
    'touchdown_timeout_s': 15.0,
    'publish_rate_hz': 10.0,
    'log_runs': True,
    'log_dir': 'auto',             # auto: /root/team_ws/runs in the stack
    'record_bag': True,
    'record_images': False,
    'max_run_s': 900.0,
    'post_roll_s': 3.0,
}


class HopTest(Node):

    def __init__(self, default_sequence='hop', **kwargs):
        super().__init__('hop_test', **kwargs)
        # Any type, coerced below: "speed_mps:=1" (an int) works, so does "0.25,0.5".
        anything = ParameterDescriptor(dynamic_typing=True)
        defaults = dict(DEFAULTS)
        defaults.update(NODE_DEFAULTS)
        defaults['sequence'] = default_sequence
        for name, default in defaults.items():
            self.declare_parameter(name, default, anything)

        def param(name):
            value = self.get_parameter(name).value
            default = defaults[name]
            if isinstance(default, bool):
                return value if isinstance(value, bool) else str(value).lower() == 'true'
            if isinstance(default, float):
                return float(value)
            return '' if value is None else str(value)

        self.cfg = {name: param(name) for name in DEFAULTS}
        self.sequence = param('sequence')
        validate(self.cfg, self.sequence)
        self.climb = self.cfg['climb_height_m']
        self.low = self.cfg['low_height_m']
        self.speed = self.cfg['speed_mps']
        self.arrival_radius = param('arrival_radius_m')
        self.yaw_tol = math.radians(param('yaw_tolerance_deg'))
        self.leg_timeout = param('leg_timeout_s')
        self.abort_distance = param('abort_distance_m')
        self.touchdown = param('touchdown')
        self.descend_speed = param('descend_speed_mps')
        self.touchdown_timeout = param('touchdown_timeout_s')
        self.dt = 1.0 / param('publish_rate_hz')
        self.log_runs = param('log_runs')
        self.log_dir = param('log_dir')
        if self.log_dir in ('', 'auto'):
            self.log_dir = default_log_dir()
        self.record_bag = param('record_bag')
        self.record_images = param('record_images')
        self.max_run_s = param('max_run_s')
        self.post_roll_s = param('post_roll_s')

        if self.touchdown and self.descend_speed < MIN_TOUCHDOWN_SPEED_MPS:
            raise ValueError(
                f'descend_speed_mps must be >= {MIN_TOUCHDOWN_SPEED_MPS} or '
                f'PX4 never detects the landing')
        if self.abort_distance <= self.arrival_radius:
            raise ValueError('abort_distance_m must be > arrival_radius_m')

        px4_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50,
        )
        self.pose_pub = self.create_publisher(PoseStamped, '/skybeta/setpoint/pose', 10)
        self.twist_pub = self.create_publisher(TwistStamped, '/skybeta/setpoint/twist', 10)
        self.event_pub = self.create_publisher(String, '/cage_test/event', 50)
        self.create_subscription(
            Odometry, '/visual_slam/tracking/odometry', self.odometry_cb, 10)
        self.create_subscription(String, '/skybeta/state', self.state_cb, 10)
        self.create_subscription(
            VehicleAttitude, fmu_topic('out', 'vehicle_attitude'), self.attitude_cb, px4_qos)
        self.create_subscription(
            SensorCombined, fmu_topic('out', 'sensor_combined'), self.px4_imu_cb, sensor_qos)
        self.create_subscription(
            BatteryStatus, fmu_topic('out', 'battery_status'), self.battery_cb, px4_qos)
        self.create_subscription(Imu, '/oak/imu/data', self.oak_imu_cb, sensor_qos)

        self.position = None
        self.yaw = None
        self.velocity = None
        self.odom_time = None
        self.px4_yaw = None
        self.attitude_time = None
        self.battery_v = None
        self.offboard_active = False
        self.armed = False
        self.landed = None
        self.px4_link = False
        self.local_enu = None
        self.local_valid = False
        self.clamped = []
        self.envelope = None
        self.log_once = set()
        self.last_readiness_log = None
        self.last_cmd = {}

        # Run logging: ring buffers for the pre-arm baseline, then the open run.
        self.pre_samples = collections.deque()
        self.pre_oak = collections.deque()
        self.pre_px4 = collections.deque()
        self.run = None
        self.run_result = ''
        self.run_legs = []
        self.log_suspended = False
        self.recorder = BagRecorder()
        self.reset()

        self.create_timer(self.dt, self.tick)
        legs = build_plan((0.0, 0.0, 0.0), 0.0, self.cfg, self.sequence)
        self.say('info',
                 f'{self.get_name()} ready: sequence "{self.sequence}", '
                 f'{len(legs)} segments, about {plan_duration(legs):.0f} s in the air'
                 f'{", then touch down" if self.touchdown else ", then hold for the pilot"}. '
                 f'Climb {self.climb:.2f} m, {self.speed:.2f} m/s between patterns. '
                 f'Runs logged to {self.log_dir if self.log_runs else "(off)"}.\n'
                 'Waiting for the pilot to arm (switch on Altitude), then switch to '
                 'Position...')

    def reset(self):
        """Back to waiting for the next arm -> Position."""
        self.state = 'WAIT_FOR_OFFBOARD'
        # Only a switch into Offboard seen from outside it starts a sequence.
        self.seen_not_offboard = False
        self.start = None
        self.legs = []
        self.leg_index = 0
        self.progress = 0.0
        self.leg_started_at = None
        self.hold_started_at = None
        self.touchdown_started_at = None
        self.hold_target = None
        self.cmd_yaw = None
        self.yaw_start = None

    # ------------------------------------------------------------------
    # logging helpers
    # ------------------------------------------------------------------
    def say(self, level, message, **fields):
        getattr(self.get_logger(), level)(message)
        t = self.now_s()
        if self.run is not None:
            self.run.event(t, level, message, state=self.state, **fields)
        msg = String()
        msg.data = json.dumps({'t': round(t, 3), 'level': level, 'state': self.state,
                               'text': message, **fields}, default=str)
        self.event_pub.publish(msg)

    def info_once(self, key, message):
        if key not in self.log_once:
            self.log_once.add(key)
            self.say('info', message)

    # ------------------------------------------------------------------
    # inputs
    # ------------------------------------------------------------------
    def odometry_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.position = (float(p.x), float(p.y), float(p.z))
        self.yaw = yaw_from_xyzw(q.x, q.y, q.z, q.w)
        twist = getattr(msg, 'twist', None)
        if twist is not None:
            v = twist.twist.linear
            self.velocity = (float(v.x), float(v.y), float(v.z))
        self.odom_time = self.now_s()

    def attitude_cb(self, msg):
        # PX4 q is NED/FRD wxyz; its heading in ENU is pi/2 - yaw_ned.
        self.px4_yaw = yaw_ned_to_enu(yaw_from_wxyz(tuple(msg.q)))
        self.attitude_time = self.now_s()

    def battery_cb(self, msg):
        v = float(getattr(msg, 'voltage_v', float('nan')))
        self.battery_v = v if math.isfinite(v) and v > 0.0 else None

    def _imu(self, source, t, stamp, acc, gyro):
        row = (t, acc[0], acc[1], acc[2], gyro[0], gyro[1], gyro[2])
        if self.run is not None:
            self.run.imu(source, (t, stamp) + row[1:])
            (self.run_oak if source == 'oak' else self.run_px4).append(row)
        else:
            ring = self.pre_oak if source == 'oak' else self.pre_px4
            ring.append((t, stamp) + row[1:])
            while ring and ring[0][0] < t - PREROLL_S:
                ring.popleft()

    def oak_imu_cb(self, msg):
        a, g = msg.linear_acceleration, msg.angular_velocity
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._imu('oak', self.now_s(), stamp, (a.x, a.y, a.z), (g.x, g.y, g.z))

    def px4_imu_cb(self, msg):
        self._imu('px4', self.now_s(), msg.timestamp * 1e-6,
                  tuple(msg.accelerometer_m_s2), tuple(msg.gyro_rad))

    def state_cb(self, msg):
        try:
            state = json.loads(msg.data)
        except ValueError:
            return
        self.offboard_active = bool(state.get('offboard_active', False))
        self.armed = bool(state.get('armed', False))
        self.landed = state.get('landed')
        self.px4_link = bool(state.get('px4_link', False))
        local = state.get('local_enu')
        self.local_enu = tuple(local) if local else None
        self.local_valid = local is not None
        self.envelope = state.get('envelope')
        self.clamped = state.get('clamped') or []
        if self.clamped:
            self.info_once(
                f'clamped:{",".join(self.clamped)}',
                f'flight_interface is CLAMPING the setpoint '
                f'({", ".join(self.clamped)}), envelope {self.envelope}: '
                f'a clamped point is never reached. Check climb_height_m and '
                f'low_height_m against it.')

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
        if self.cmd_yaw is None:
            # All-zero quaternion == "do not command yaw": PX4 keeps its heading.
            msg.pose.orientation.w = 0.0
        else:
            q = xyzw_from_yaw(self.cmd_yaw)
            msg.pose.orientation.x, msg.pose.orientation.y = q[0], q[1]
            msg.pose.orientation.z, msg.pose.orientation.w = q[2], q[3]
        self.pose_pub.publish(msg)
        self.last_cmd = {'cmd_kind': 'pose', 'cmd_x': float(x), 'cmd_y': float(y),
                         'cmd_z': float(z), 'cmd_yaw': self.cmd_yaw}

    def send_descent(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.twist.linear.z = -self.descend_speed
        self.twist_pub.publish(msg)
        self.last_cmd = {'cmd_kind': 'twist', 'cmd_vz': -self.descend_speed}

    # ------------------------------------------------------------------
    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def odom_fresh(self):
        return (self.odom_time is not None
                and self.now_s() - self.odom_time <= ODOM_STALE_S)

    def measured_yaw(self):
        """PX4's heading (what its yaw controller tracks) if fresh, else VIO's."""
        if self.attitude_time is not None and self.now_s() - self.attitude_time <= ATTITUDE_STALE_S:
            return self.px4_yaw
        return self.yaw

    def arrived_at(self, target):
        return math.dist(self.position, target) <= self.arrival_radius

    def fail(self, message):
        """Stop advancing and hold where the setpoint is now."""
        self.say('error', f'{message} -- holding. Pilot: take over and land.')
        self.state = 'FAILED'
        self.run_result = f'failed: {message}'

    # ------------------------------------------------------------------
    # the state machine
    # ------------------------------------------------------------------
    def tick(self):
        if self.position is None:
            self.info_once('no_odom', 'no /visual_slam/tracking/odometry yet -- '
                           'nothing is commanded until cuVSLAM is up.')
            return

        self.manage_run()

        if not self.armed and self.state != 'WAIT_FOR_OFFBOARD':
            self.say('info', f'disarmed (was {self.state}). Ready for the next run: arm, '
                     f'then switch to Position.')
            if self.state in ('TOUCHDOWN', 'DONE') and self.run_result == 'flying':
                self.run_result = 'completed'
            self.reset()

        flying = self.state in ('LEG', 'TOUCHDOWN', 'FAILED', 'DONE')
        if flying and not self.offboard_active:
            self.say('warn',
                     f'Offboard lost during {self.state} -- the pilot has the '
                     f'vehicle. Nothing more until it is disarmed.')
            if self.run_result in ('flying', 'completed'):
                self.run_result = f'pilot took over during {self.state}'
                if self.state == 'LEG':
                    self.run_result += f' "{self.legs[self.leg_index].name}"'
            self.state = 'ABORTED'

        self.last_cmd = {}
        getattr(self, f'_state_{self.state.lower()}')()
        self.record_sample()

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
        self.legs = build_plan(self.start, self.yaw, self.cfg, self.sequence)
        self.run_legs = self.legs
        self.run_result = 'flying'
        if self.run is not None:
            self.run.meta['start_enu'] = list(self.start)
            self.run.meta['start_heading_deg'] = math.degrees(self.yaw)
            self.run.meta['segments'] = [s._asdict() for s in self.legs]
            self.run.write_meta()
        self.begin_leg(0)
        fwd = self.legs[1].end if len(self.legs) > 1 else self.legs[0].end
        self.say('info',
                 f'Offboard active. Start ENU ({self.start[0]:.2f}, '
                 f'{self.start[1]:.2f}, {self.start[2]:.2f}), heading '
                 f'{math.degrees(self.yaw):.0f} deg; forward point '
                 f'({fwd[0]:.2f}, {fwd[1]:.2f}). Leg 1/{len(self.legs)}: up.')

    def begin_leg(self, index):
        self.state = 'LEG'
        self.leg_index = index
        leg = self.legs[index]
        self.progress = seg_length(leg) if leg.kind == 'step' else 0.0
        self.leg_started_at = self.now_s()
        self.hold_started_at = None
        if leg.kind == 'yaw':
            if self.cmd_yaw is None:
                self.cmd_yaw = self.measured_yaw()
            self.yaw_start = self.cmd_yaw
        if self.run is not None:
            self.run.event(self.now_s(), 'segment', leg.name, index=index,
                           seg_kind=leg.kind, pattern=leg.pattern)

    def leg_ramp_done(self, leg):
        if leg.kind == 'yaw':
            return self.progress >= abs(leg.yaw_delta)
        return self.progress >= seg_length(leg)

    def leg_arrived(self, leg):
        if not self.arrived_at(leg.end):
            return False
        if leg.kind == 'yaw':
            measured = self.measured_yaw()
            return measured is not None and abs(wrap_pi(self.cmd_yaw - measured)) <= self.yaw_tol
        return True

    def _state_leg(self):
        leg = self.legs[self.leg_index]
        fresh = self.odom_fresh()
        if leg.kind == 'yaw':
            if fresh:
                self.progress = min(self.progress + leg.yaw_rate * self.dt, abs(leg.yaw_delta))
            self.cmd_yaw = wrap_pi(self.yaw_start + math.copysign(self.progress, leg.yaw_delta))
            self.hold_target = tuple(leg.end)
            ramp_s = abs(leg.yaw_delta) / leg.yaw_rate
        else:
            if fresh and leg.kind == 'move':
                self.progress = min(self.progress + leg.speed * self.dt, seg_length(leg))
            self.hold_target = carrot(leg, self.progress)
            ramp_s = seg_length(leg) / leg.speed if leg.kind == 'move' else 0.0
        if not fresh:
            self.info_once(f'stale:{self.leg_index}',
                           f'odometry stale on leg "{leg.name}" -- setpoint '
                           f'paused until it returns')
        self.send_pose(*self.hold_target)

        if fresh:
            off = distance_to_path(self.position, leg)
            if off > self.abort_distance:
                self.fail(f'leg "{leg.name}": {off:.2f} m off its path (limit '
                          f'{self.abort_distance:.2f} m), sequence stopped')
                return

        if self.hold_started_at is None:
            if self.leg_ramp_done(leg) and self.leg_arrived(leg):
                self.hold_started_at = self.now_s()
                self.say('info', f'leg "{leg.name}" reached, holding {leg.hold_s:.0f} s')
            elif self.now_s() - self.leg_started_at > ramp_s + self.leg_timeout:
                self.fail(
                    f'leg "{leg.name}" not reached in time (still '
                    f'{math.dist(self.position, leg.end):.2f} m off). Check '
                    f'"clamped" in /skybeta/state and cuVSLAM tracking')
            return
        if self.now_s() - self.hold_started_at < leg.hold_s:
            return

        if self.leg_index + 1 < len(self.legs):
            self.begin_leg(self.leg_index + 1)
            self.say('info', f'leg {self.leg_index + 1}/{len(self.legs)}: '
                     f'{self.legs[self.leg_index].name}')
        elif self.touchdown:
            self.state = 'TOUCHDOWN'
            self.touchdown_started_at = self.now_s()
            self.say('info',
                     f'touching down at {self.descend_speed:.2f} m/s; PX4 disarms '
                     f'once it detects the landing')
        else:
            self.state = 'DONE'
            self.run_result = 'completed'
            self.say('info', f'sequence complete -- holding at {self.low:.2f} m. '
                     f'Pilot: take over and land.')

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

    # ------------------------------------------------------------------
    # run logging: open at the arm, close post_roll_s after the disarm
    # ------------------------------------------------------------------
    def phase(self):
        if self.state != 'LEG':
            return ''
        leg = self.legs[self.leg_index]
        if self.hold_started_at is not None:
            return 'hold'
        return 'ramp' if not self.leg_ramp_done(leg) else 'settle'

    def record_sample(self):
        now = self.now_s()
        local = self.local_enu or (None, None, None)
        vel = self.velocity or (None, None, None)
        row = {
            't': now, 'state': self.state,
            'seg': self.leg_index if self.state == 'LEG' else -1,
            'seg_name': self.legs[self.leg_index].name if self.state == 'LEG' else '',
            'phase': self.phase(),
            'vio_x': self.position[0], 'vio_y': self.position[1], 'vio_z': self.position[2],
            'vio_yaw': self.yaw, 'vio_vx': vel[0], 'vio_vy': vel[1], 'vio_vz': vel[2],
            'px4_x': local[0], 'px4_y': local[1], 'px4_z': local[2], 'px4_yaw': self.px4_yaw,
            'odom_age': (now - self.odom_time) if self.odom_time is not None else None,
            'armed': self.armed, 'offboard': self.offboard_active,
            'landed': None if self.landed is None else bool(self.landed),
            'clamped': ';'.join(self.clamped), 'battery_v': self.battery_v,
        }
        row.update(self.last_cmd)
        if self.run is not None:
            row['t_run'] = now - self.run_t0
            self.run.sample(row)
            self.run_samples.append(row)
            if len(self.run_samples) % 50 == 0:
                self.run.flush()
        else:
            self.pre_samples.append(row)
            while self.pre_samples and self.pre_samples[0]['t'] < now - PREROLL_S:
                self.pre_samples.popleft()

    def manage_run(self):
        if not self.log_runs:
            return
        now = self.now_s()
        if self.run is None:
            if not self.armed:
                self.log_suspended = False
            elif not self.log_suspended:
                self.open_run(now)
            return
        if self.armed:
            self.run_disarmed_at = None
        elif self.run_disarmed_at is None:
            self.run_disarmed_at = now
        if self.run_disarmed_at is not None and now - self.run_disarmed_at >= self.post_roll_s:
            self.close_run()
        elif now - self.run_t0 > self.max_run_s:
            self.say('warn', f'run longer than max_run_s ({self.max_run_s:.0f} s): '
                     'closing the log; the next arm opens a new one')
            self.close_run()
            self.log_suspended = True

    def open_run(self, now):
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            self.run = RunLog(self.log_dir, self.sequence, {
                'node': self.get_name(),
                'sequence': self.sequence,
                'config': self.cfg,
                'node_params': {k: self.get_parameter(k).value for k in NODE_DEFAULTS},
            })
        except OSError as exc:
            self.say('error', f'cannot open a run log in {self.log_dir}: {exc} -- '
                     'flying without one')
            self.log_runs = False
            return
        self.run_t0 = now
        self.run_disarmed_at = None
        self.run_result = 'armed, no Offboard'
        self.run_samples = []
        self.run_legs = []
        self.run_oak = []
        self.run_px4 = []
        # The pre-arm baseline.
        for row in self.pre_samples:
            row = dict(row, t_run=row['t'] - now)
            self.run.sample(row)
            self.run_samples.append(row)
        for source, ring, dest in (('oak', self.pre_oak, self.run_oak),
                                   ('px4', self.pre_px4, self.run_px4)):
            for r in ring:
                self.run.imu(source, r)
                dest.append((r[0],) + tuple(r[2:]))
            ring.clear()
        self.pre_samples.clear()
        self.say('info', f'armed: logging this run to {self.run.path}')
        self.start_bag()

    def start_bag(self):
        if not self.record_bag:
            return
        if not BagRecorder.available():
            self.say('warn', 'ros2 CLI not found: no bag for this run (CSVs only)')
            return
        gb = free_gb(self.run.path)
        if gb is not None and gb < 1.0:
            self.say('error', f'only {gb:.1f} GB free under {self.log_dir}: no bag '
                     'for this run (CSVs only). Delete old runs.')
            return
        names = [n for n, _ in self.get_topic_names_and_types()]
        topics = select_topics(names, self.record_images)
        path = os.path.join(self.run.path, 'bag')
        try:
            cmd = self.recorder.start(path, topics, max_s=self.max_run_s + 60)
        except OSError as exc:
            self.say('error', f'could not start ros2 bag record: {exc}')
            return
        self.run.meta['bag'] = {'path': path, 'topics': topics, 'command': cmd}
        self.run.write_meta()
        self.say('info', f'recording {len(topics)} topics to {path}'
                 + (f' ({gb:.0f} GB free)' if gb is not None else ''))

    def close_run(self):
        run, self.run = self.run, None
        self.recorder.stop()
        summary, table = [], ''
        try:
            if self.run_legs:
                summary = analysis.summarize(
                    self.run_legs, self.run_samples, self.run_oak, self.run_px4)
                table = analysis.format_table(summary)
        except Exception as exc:  # the log matters more than the summary
            self.get_logger().error(f'summary failed: {exc!r}')
        run.event(self.now_s(), 'result', self.run_result)
        run.close(self.run_result, summary, table)
        self.get_logger().info(
            f'run closed ({self.run_result}): {run.path}'
            + (f'\n{table}' if table else ''))

    def destroy_node(self):
        if self.run is not None:
            self.run_result += ' (node stopped)'
            self.close_run()
        self.recorder.shutdown()
        return super().destroy_node()


def _main(default_sequence, args=None):
    rclpy.init(args=args)
    node = HopTest(default_sequence=default_sequence)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(args=None):
    _main('hop', args)


def tune_main(args=None):
    _main('hover,steps,speed,yaw', args)
