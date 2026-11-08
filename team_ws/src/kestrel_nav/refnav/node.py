"""refnav.node - the ROS 2 shell around refnav.core. No flight decisions here.

What it takes from its surroundings, all through standard ROS interfaces (so the simulator, a
replay of recorded data and the aircraft differ only in their launch files):

  clock        the node clock: use_sim_time:=true follows /clock (the simulator publishes it;
               `ros2 bag play --clock` publishes the recording's), false = system time (aircraft)
  depth        sensor_msgs/Image (32FC1 m or 16UC1 mm) + CameraInfo, topics by parameter
  calibration  TF base_link <- the depth image's frame_id (a static transform; REP-103 optical
               frame), looked up at run time - never read from a simulator file
  health       std_msgs/Bool, true = the estimator feeding PX4 is healthy (topic by parameter)
  PX4          /fmu/out/{vehicle_local_position_v1, vehicle_attitude, vehicle_status_v1}
               in; /fmu/in/{offboard_control_mode, trajectory_setpoint, vehicle_command} out
  status       /refnav/status (status_topic; std_msgs/String, JSON): phase, why, leg, counters, the
               latest depth refusal - what a grader, the interactive runtime or an operator reads;
               the node itself never reads simulator truth
  control      optional (control_topic): std_msgs/String JSON {"run_id", "request": "land"} from the
               interactive runtime; a request for this run_id goes through the mission's own landing

Run:  python3 -m refnav.node --ros-args -p route_file:=scenarios/<x>.json -p depth_topic:=...           -p use_sim_time:=true
"""
from __future__ import annotations
import json
import math
import sys
import time
import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from . import core as C
from . import depth as D
from .trace import LPOS_FIELDS, TraceWriter, fields
PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST, depth=1)
CMD_SET_MODE, CMD_ARM, CMD_LAND = (176, 400, 21)
PX4_MAIN_MODE_OFFBOARD = 6

def _as(default, val):
    """Coerce a ROS parameter value to the type of the Params default ("true"/"false" strings count)."""
    if isinstance(default, bool):
        return val if isinstance(val, bool) else str(val).strip().lower() in ('1', 'true', 'yes')
    return type(default)(val)

def load_route(node):
    path = node.declare_parameter('route_file', '').value
    flat = list(node.declare_parameter('route', [0.0]).value)
    if path:
        with open(path) as f:
            doc = json.load(f)
        pts = doc['route'] if isinstance(doc, dict) else doc
        return ([(float(a), float(b)) for a, b in pts], doc if isinstance(doc, dict) else {})
    if len(flat) >= 2 and len(flat) % 2 == 0:
        return ([(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat), 2)], {})
    raise SystemExit('refnav: give route_file:=<json> or route:=[f1,l1,f2,l2,...] (forward/left metres)')

class RefNav(Node):

    def __init__(self):
        super().__init__('refnav')
        route, doc = load_route(self)
        prm = C.Params(route=route, legs=list(doc.get('legs', [])))
        for k, v in vars(C.Params()).items():
            if k in ('route', 'legs'):
                continue
            default = doc.get('params', {}).get(k, v)
            val = self.declare_parameter(k, default).value
            setattr(prm, k, _as(v, val))
        prm.track_mode = 'path'
        prm.local_horizon = True
        prm.legs = [dict(name=L.get('name', 'waypoint'), mode='cruise', hold=True) for L in prm.legs]
        self.mission = C.Mission(prm)
        self.return_enabled = False
        self.diagnostic_enabled = False
        self.depth_topic = self.declare_parameter('depth_topic', '/camera/depth/image_raw').value
        info = self.declare_parameter('depth_info_topic', '').value
        self.info_topic = info or self.depth_topic.rsplit('/', 1)[0] + '/camera_info'
        self.health_topic = self.declare_parameter('health_topic', '/estimator/healthy').value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        self.exit_when_done = bool(self.declare_parameter('exit_when_done', True).value)
        control = self.declare_parameter('control_topic', '').value
        self.run_id = self.declare_parameter('run_id', '').value
        self.sys_clock = Clock(clock_type=ClockType.SYSTEM_TIME)
        self.trace = TraceWriter(self.declare_parameter('trace_file', '').value, warn=self.get_logger().warning)
        self._lpos_received_ros_s = None
        self._lpos_timestamp = None
        from tf2_ros import Buffer, TransformListener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        from px4_msgs.msg import BatteryStatus, EstimatorStatusFlags, OffboardControlMode, TrajectorySetpoint, VehicleAttitude, VehicleCommand, VehicleLandDetected, VehicleLocalPosition, VehicleStatus
        self._OCM, self._TSP, self._CMD = (OffboardControlMode, TrajectorySetpoint, VehicleCommand)
        self.px4_output = str(self.declare_parameter('px4_output', 'direct').value)
        if self.px4_output not in ('direct', 'skybeta'):
            raise ValueError('px4_output must be direct or skybeta')
        lpos_topic, status_topic_px4, land_topic = ('/fmu/out/vehicle_local_position_v1', '/fmu/out/vehicle_status_v1', '/fmu/out/vehicle_land_detected')
        if self.px4_output == 'skybeta':
            from .skybeta import SkybetaOutput
            self.sky = SkybetaOutput(self)
            self.pub_ocm = self.pub_sp = self.pub_cmd = None
            try:
                from skybeta_flight_interface.topics import resolve_endpoints
                ep = resolve_endpoints()
                lpos_topic, status_topic_px4, land_topic = (ep['vehicle_local_position'], ep['vehicle_status'], ep['vehicle_land_detected'])
            except Exception as e:
                self.get_logger().warning(f'skybeta topic resolver unavailable ({e}); using _v1 names')
            self.get_logger().info(f'skybeta output: twist only, no /fmu/in; PX4 topics {lpos_topic}, {status_topic_px4}, {land_topic}')
        else:
            self.sky = None
            self.pub_ocm = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', PX4_QOS)
            self.pub_sp = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', PX4_QOS)
            self.pub_cmd = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', PX4_QOS)
        status_topic = self.declare_parameter('status_topic', '/refnav/status').value
        self.pub_status = self.create_publisher(String, status_topic, 10)
        self.v = C.Vehicle()
        self.create_subscription(VehicleLocalPosition, lpos_topic, self._on_lpos, PX4_QOS)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self._on_att, PX4_QOS)
        self.create_subscription(VehicleStatus, status_topic_px4, self._on_status, PX4_QOS)
        self.create_subscription(VehicleLandDetected, land_topic, self._on_land, PX4_QOS)
        self._yaw_align = (False, None)
        self._ev_yaw = (False, None)
        self.create_subscription(EstimatorStatusFlags, '/fmu/out/estimator_status_flags', self._on_est_flags, PX4_QOS)
        self.create_subscription(Bool, self.health_topic, self._on_health, 10)
        self.create_subscription(CameraInfo, self.info_topic, self._on_info, qos_profile_sensor_data)
        self.create_subscription(Image, self.depth_topic, self._on_depth, qos_profile_sensor_data)
        # Altitude-hold height: VIO z anchored to PX4's height, corrected slowly (height_tau_s).
        # PX4's height reference here is the downward rangefinder (EKF2_HGT_REF 2), which reads a
        # window sill or any object below as the floor rising; VIO does not.
        from nav_msgs.msg import Odometry
        from .height import AnchoredHeight
        self.height_tau_s = float(self.declare_parameter('height_tau_s', 10.0).value)
        self._vio = None                 # (receive time, z up)
        self._height = AnchoredHeight(self.height_tau_s, float(self.declare_parameter('height_jump_m', 0.15).value))
        self.create_subscription(Odometry, self.declare_parameter('vio_topic', '/visual_slam/tracking/odometry').value,
                                 self._on_vio, qos_profile_sensor_data)
        path_topic = self.declare_parameter('path_topic', '/coursenav/path').value
        if self.mission.p.track_mode == 'path' and path_topic:
            self.create_subscription(String, path_topic, self._on_path, 10)
        if control:
            ctl_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST, depth=1)
            self.create_subscription(String, control, self._on_control, ctl_qos)
        self.K = None
        self.extrinsic = None
        self.rejects = {}
        self.done_at = None
        self._wait_log_t = None
        self.create_timer(0.05, self._tick)
        self.create_timer(0.5, self._publish_status)
        self.get_logger().info(f"route {len(route)} legs; depth {self.depth_topic}; health {self.health_topic}; sim time {self.get_parameter('use_sim_time').value}")

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-09
    EV_YAW_GRACE_S = 2.0
    RESET_MAX_HEADING = math.radians(10.0)
    RESET_MAX_XY_M = 0.3

    def _check_reset(self, m):
        """skybeta: refnav keeps targets, home and path in PX4's local frame and has no reset handling (the lab's
        NED vision rarely resets). A small reset is absorbed by the position/heading loops; a LARGE one (Gazebo
        REALISM, after a 330 ms EV freeze: heading_reset 2->3 with 1.570 -> -2.029 rad) means the frame no longer
        describes the world: land in place through refnav's own landing rather than fly on stale targets."""
        cur = (getattr(m, 'xy_reset_counter', None), getattr(m, 'heading_reset_counter', None))
        prev = getattr(self, '_reset_counters', None)
        self._reset_counters = cur
        if prev is None or cur == prev or self.mission.phase not in C.AIRBORNE:
            return
        dh = float(getattr(m, 'delta_heading', 0.0)) if cur[1] != prev[1] else 0.0
        dxy = math.hypot(*[float(c) for c in getattr(m, 'delta_xy', (0.0, 0.0))[:2]]) if cur[0] != prev[0] else 0.0
        msg = f'EKF reset in flight: heading {math.degrees(dh):+.1f} deg, xy {dxy:.2f} m (counters {prev} -> {cur})'
        if abs(dh) > self.RESET_MAX_HEADING or dxy > self.RESET_MAX_XY_M:
            self.get_logger().warning(msg + ': frame no longer trusted, landing in place')
            self.mission.request_land(msg)
        else:
            self.get_logger().info(msg + ': small, continuing')

    def _on_est_flags(self, m):
        self._yaw_align = (bool(m.cs_yaw_align), self._now())
        self._ev_yaw = (bool(getattr(m, 'cs_ev_yaw', False)), self._now())
        if self._ev_yaw[0]:
            self._ev_yaw_true_t = self._now()

    def _on_lpos(self, m):
        accepted = self._return_observe('position', m.timestamp * 1e-06, eph=float(getattr(m, 'eph', math.nan)), epv=float(getattr(m, 'epv', math.nan)), resets=tuple((getattr(m, k, None) for k in ('xy_reset_counter', 'z_reset_counter', 'heading_reset_counter'))), velocity_resets=tuple((getattr(m, k, None) for k in ('vxy_reset_counter', 'vz_reset_counter'))), velocity_valid=bool(m.v_xy_valid and getattr(m, 'v_z_valid', False)), heading=float(m.heading))
        if not accepted:
            return
        self._lpos_received_ros_s = self._now()
        self._lpos_timestamp = getattr(m, 'timestamp', None)
        if self.trace.enabled:
            self.trace.record('local_position', self._lpos_received_ros_s, data=fields(m, LPOS_FIELDS))
        self.v.pos = (m.x, m.y, self._robust_z(float(m.z)))
        self.v.heading = m.heading
        rng = float(getattr(m, 'dist_bottom', math.nan))
        self.v.rng = rng if math.isfinite(rng) else None
        vel = (m.vx, m.vy, m.vz)
        self.v.vel = vel if bool(m.v_xy_valid) and all((math.isfinite(c) for c in vel)) else None
        ya, ya_t = self._yaw_align
        heading_ok = bool(m.heading_good_for_control) or (ya and ya_t is not None and (self._now() - ya_t <= 3.0))
        if self.px4_output == 'skybeta' and (not heading_ok):
            ev_true_t = getattr(self, '_ev_yaw_true_t', None)
            heading_ok = bool(ev_true_t is not None and self._now() - ev_true_t <= self.EV_YAW_GRACE_S)
        if self.px4_output == 'skybeta':
            self._check_reset(m)
        self.v.pos_ok = bool(m.xy_valid and m.z_valid and heading_ok and all((math.isfinite(c) for c in (m.x, m.y, m.z, m.heading))))

    def _on_vio(self, m):
        z = float(m.pose.pose.position.z)
        if math.isfinite(z):
            self._vio = (self._now(), z)

    def _robust_z(self, z_px4):
        """PX4 NED z replaced by the VIO-anchored height (refnav.height): large PX4 height jumps, e.g. its
        rangefinder reference over a window sill, are disregarded and the height is held through VIO gaps."""
        vio = None if self._vio is None else -self._vio[1]          # ENU up -> NED down
        z = self._height.update(self._now(), z_px4 if math.isfinite(z_px4) else None, vio,
                                None if self._vio is None else self._vio[0])
        return z if math.isfinite(z) else z_px4

    def _on_att(self, m):
        if not self._return_observe('attitude', m.timestamp * 1e-06):
            return
        self.v.q = tuple((float(c) for c in m.q))

    def _on_status(self, m):
        if not self._return_observe('status', m.timestamp * 1e-06):
            return
        self.v.armed = m.arming_state == 2
        self.v.nav_state = int(m.nav_state)

    def _on_land(self, m):
        accepted = self._return_observe('land', m.timestamp * 1e-06)
        if not accepted:
            return
        self.v.landed = bool(m.landed)

    def _return_observe(self, name, stamp, **data):
        return True
        return self.mission.observe(name, stamp, self._now(), time.monotonic(), **data)

    def _on_battery(self, m):
        self._return_observe('battery', m.timestamp * 1e-06, connected=bool(m.connected), remaining=float(m.remaining), warning=int(m.warning), faults=int(m.faults), powering_off=bool(m.is_powering_off), time_remaining_s=float(m.time_remaining_s))

    def _on_localizer(self, m):
        try:
            d = json.loads(m.data)
            self._return_observe('localizer', float(d['t']), **{k: d.get(k) for k in ('ready', 'updates', 'sigma_xy', 'x', 'y', 'resets')})
        except (ValueError, TypeError, KeyError):
            self.mission.samples.pop('localizer', None)

    def _on_health(self, m):
        self.mission.on_health(self._now(), m.data)

    def _on_info(self, m):
        if m.k[0] > 0 and m.k[4] > 0:
            self.K = (m.k[0], m.k[4], m.k[2], m.k[5])

    def _on_path(self, m):
        if not getattr(self.mission, 'accepts_outbound_paths', True):
            return
        try:
            d = json.loads(m.data)
        except ValueError:
            self.trace.record('path_invalid', self._now(), raw=m.data)
            return
        self.trace.record('path', self._now(), data=d)
        mi = self.mission
        if int(d.get('leg', -1)) == mi.leg and d.get('valid') is False:
            had_path = mi.path is not None
            mi.path = None
            if had_path and mi.phase in ('go', 'check', 'turn') and (mi.sp is not None) and (mi.last_v is not None) and (mi.last_v.pos is not None):
                mi.sp[:2] = mi.last_v.pos[:2]
        if mi.targets is None:
            return
        wps = getattr(mi, 'waypoints_ne', None)
        for j, ne in (d.get('targets_ne') or {}).items():
            j = int(j)
            if mi.leg < j < len(mi.targets) and len(ne) == 2 and all((math.isfinite(float(c)) for c in ne)):
                mi.targets[j] = (float(ne[0]), float(ne[1]))
                if wps and j < len(wps):
                    wps[j] = mi.targets[j]
        goal = d.get('goal_ne')
        if (wps and int(d.get('leg', -1)) == mi.leg < len(wps) and goal and len(goal) == 2
                and all((math.isfinite(float(c)) for c in goal))):
            wps[mi.leg] = (float(goal[0]), float(goal[1]))   # true waypoint: centre of any search sweep
        if int(d.get('leg', -1)) == mi.leg and d.get('pts_ne'):
            received = self._now()
            accepted = mi.on_path(received, mi.leg, d['pts_ne'], d.get('v') or [], d.get('stops') or [], pid=d.get('pid', 0), tubes=d.get('tubes'), terminal=d.get('terminal', True))
            self.trace.record('path_applied', received, accepted=accepted, leg=mi.leg, pid=d.get('pid', 0))

    def _on_control(self, m):
        try:
            req = json.loads(m.data)
        except ValueError:
            return
        if req.get('request') == 'land' and (not self.run_id or req.get('run_id') == self.run_id):
            self.get_logger().info(f"landing requested by the runtime ({req.get('why', 'operator')})")
            self.mission.request_land(req.get('why') or 'operator')

    def _reject(self, key, detail=None, record=True):
        """Count a refused depth frame by its reason; log the reason WITH its numbers at the 1st, 10th,
        100th and 1000th time; tell the mission (which reports it while it waits)."""
        detail = detail or key
        if record:
            self.mission.depth_refused(key, detail)
        self.rejects[key] = self.rejects.get(key, 0) + 1
        if self.rejects[key] in (1, 10, 100, 1000):
            self.get_logger().warning(f'depth frame not used ({self.rejects[key]}x): {detail}')

    def _calibration(self, frame_id):
        if self.extrinsic is not None and self.extrinsic[0] == frame_id:
            return self.extrinsic
        from rclpy.time import Time
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, frame_id, Time())
        except Exception as e:
            self._reject('no_tf', f'no TF {self.base_frame} <- {frame_id} ({type(e).__name__})')
            return None
        r, t = (tf.transform.rotation, tf.transform.translation)
        R = D.quat_to_mat(r.w, r.x, r.y, r.z)
        fwd = R[:, 2]
        ang = math.degrees(math.acos(max(-1.0, min(1.0, fwd[0] / np.linalg.norm(fwd)))))
        if ang > 25.0:
            self._reject('camera_not_forward', f"camera {frame_id} looks {ang:.0f} deg off the body's forward axis: this package needs a forward camera")
            return None
        self.extrinsic = (frame_id, R, np.array([t.x, t.y, t.z]))
        self.get_logger().info(f'calibration from TF: {self.base_frame} <- {frame_id} t=({t.x:.3f},{t.y:.3f},{t.z:.3f}) optical axis {ang:.1f} deg off forward')
        return self.extrinsic

    def _on_depth(self, m):
        if self.K is None:
            return self._reject('no_camera_info', f'no camera_info yet on {self.info_topic}')
        cal = self._calibration(m.header.frame_id)
        if cal is None:
            return None
        if self.v.q is None:
            return self._reject('no_attitude', 'no PX4 attitude yet')
        try:
            d = D.decode_depth(m.encoding, m.height, m.width, m.step, bytes(m.data))
        except ValueError as e:
            return self._reject('undecodable', str(e))
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-09
        if not self.mission.on_depth(stamp, d, self.K, cal[1], cal[2], self.v.q, now=self._now()):
            key, detail = self.mission.depth_reject or ('refused', 'refused by the mission')
            self._reject(key, detail, record=False)
        return None

    def _sys_us(self):
        return self.sys_clock.now().nanoseconds // 1000

    def _command(self, cmd, p1=0.0, p2=0.0):
        c = self._CMD()
        c.timestamp = self._sys_us()
        c.command, c.param1, c.param2 = (cmd, float(p1), float(p2))
        c.target_system = c.target_component = c.source_system = c.source_component = 1
        c.from_external = True
        self.pub_cmd.publish(c)

    def _tick(self):
        now = self._now()
        if now <= 0.0:
            return
        diagnostic = False
        if self.sky is not None and self.sky.land_xy is not None and (self.v.armed is False) and (self.mission.phase == 'land'):
            self.mission.land_auto_seen = True
        out = self.mission.step(now, self.v)
        for e in out.events:
            self.get_logger().info(e)
        if self.mission.phase == 'wait' and self.mission.t_phase is not None:
            if self._wait_log_t is None or now - self._wait_log_t >= 10.0:
                if self._wait_log_t is not None:
                    self.get_logger().info(f'not ready after {now - self.mission.t_phase:.0f} s (gives up at {self.mission.p.wait_s:.0f} s): {self.mission.why}')
                self._wait_log_t = now
        if out.events:
            self._publish_status()
        published = None
        if self.sky is not None:
            published = self.sky.publish(now, out, self.v)
            out.commands = []
        if self.sky is None and out.stream and (out.setpoint is not None):
            o = self._OCM()
            o.timestamp = self._sys_us()
            o.position = True
            o.velocity = out.velocity is not None
            o.acceleration = out.accel is not None
            self.pub_ocm.publish(o)
            s = self._TSP()
            s.timestamp = o.timestamp
            s.position = [float(c) for c in out.setpoint[:3]]
            s.velocity = [float(out.velocity[0]), float(out.velocity[1]), 0.0] if out.velocity is not None else [math.nan] * 3
            s.acceleration = [float(out.accel[0]), float(out.accel[1]), math.nan] if out.accel is not None else [math.nan] * 3
            s.yaw = float(out.setpoint[3])
            if getattr(out, 'yawspeed', None) is not None:
                s.yawspeed = float(out.yawspeed)
            self.pub_sp.publish(s)
            if self.trace.enabled:
                published = fields(s, ('timestamp', 'position', 'velocity', 'acceleration', 'jerk', 'yaw', 'yawspeed'))
        for c in out.commands:
            if c == 'offboard':
                self._command(CMD_SET_MODE, 1.0, PX4_MAIN_MODE_OFFBOARD)
            elif c == 'arm':
                self._command(CMD_ARM, 1.0)
            elif c == 'land':
                self._command(CMD_LAND)
            elif c == 'disarm':
                self._command(CMD_ARM, 0.0)
            elif c == 'force_disarm':
                self._command(CMD_ARM, 0.0, 21196.0)
        if out.phase == 'done' and self.done_at is None:
            self.done_at = now
            self.get_logger().info(f'mission done: {self.mission.result}')
        self.trace.record('tick', now, phase=self.mission.phase, leg=self.mission.leg, vehicle=self.v, output=out, published_trajectory=published, lpos_received_ros_s=self._lpos_received_ros_s, lpos_timestamp=self._lpos_timestamp, path=None if self.mission.path is None else {k: self.mission.path[k] for k in ('t', 'leg', 'pid')}, path_s=self.mission.path_s, path_stop_done=self.mission.path_stop_done)

    def _publish_status(self):
        st = self.mission.status(self._now())
        st['rejects'] = dict(self.rejects)
        self.pub_status.publish(String(data=json.dumps(st)))
        self.trace.record('status', self._now(), data=st)
        if self.exit_when_done and self.done_at is not None and (self._now() - self.done_at > 1.5):
            raise SystemExit(0 if self.mission.result == 'complete' else 1)

    def destroy_node(self):
        if hasattr(self, 'trace'):
            self.trace.close()
        return super().destroy_node()

def main(argv=None):
    """argv: a full command line (argv[0] then --ros-args ...), default sys.argv - so a program can run
    the node with its own settings (examples/takeoff_land, runtime/program.refnav_args)."""
    rclpy.init(args=list(argv) if argv is not None else sys.argv)
    node = RefNav()
    code = 1
    try:
        rclpy.spin(node)
    except SystemExit as e:
        code = int(e.code or 0)
    except KeyboardInterrupt:
        code = 130
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(code)
if __name__ == '__main__':
    main()
