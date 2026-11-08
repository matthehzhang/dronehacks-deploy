"""ROS 2 Skybeta adapter for fresh-map search. Never arms, changes mode or reads truth.

Run ``python3 -m kestrel_nav.searchnav.node --ros-args -p use_sim_time:=true``
only alongside the normal calibrated sensor/flight-interface bringup. A pilot
must authorize the vertical launch column; no forward-only sensor can see
every overhead obstruction. A separate explicit allow_takeoff parameter gates
that assumption. No horizontal cell is cleared from the launch fixture.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
import json
import math
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import TwistStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool, String
from px4_msgs.msg import (VehicleLocalPosition, VehicleAttitude, VehicleStatus,
                          VehicleLandDetected, EstimatorStatusFlags)
from tf2_ros import Buffer, TransformListener

from .grid import DepthGrid, GridParams
from .mission import SearchMission, MissionParams, Observation, Command
from .target import DetectorParams, MonoFloorDetector, TargetTracker, RGBBlueDetector, HSVParams, decode_bgr
from .floor import fit_floor, FloorTracker
from .inputs import DepthWatchdog, Px4ClockAnchor, verify_small_reset
from .opening import DepthOpeningDetector, OpeningTracker
from .clearance import DepthClearanceHistory
from .corridor import assess_motion_corridor, CorridorParams
from .launch_volume import LaunchVolume, LaunchVolumeParams

PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)
NED_TO_ENU = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
FLU_TO_FRD = np.diag([1., -1., -1.])
# Same EV-yaw freshness contract as existing refnav. The measured flags source
# publishes every 1.008-1.021s; an extra 1s transport gate was unsatisfiable.
EV_YAW_FRESH_S = 2.0
INPUT_LIMITS = dict(position=.5, attitude=.5, status=1., flags=EV_YAW_FRESH_S, land=2.6,
                    health=.5, vio=.5, state=1., depth=.5, mono=.5)


class CaptureHistoryUnavailable(ValueError):
    """Expected briefly after withdrawing a verified reset's pose history."""


def quat_R(w, x, y, z):
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])


def versioned(name, typ):
    version = int(getattr(typ, 'MESSAGE_VERSION', 0))
    return '/fmu/out/' + name + (f'_v{version}' if version else '')


def stamp(m):
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def decode_depth(m):
    """Respect row padding and byte order; invalid returns stay NaN."""
    if m.encoding.lower() == '32fc1':
        size, code, scale = 4, 'f4', 1.
    elif m.encoding.lower() in ('16uc1', 'mono16'):
        size, code, scale = 2, 'u2', .001
    else:
        raise ValueError('need calibrated 32FC1 metres or 16UC1 millimetres')
    if m.step < m.width * size or m.step % size:
        raise ValueError('invalid image stride')
    dtype = np.dtype(('>' if m.is_bigendian else '<') + code)
    out = np.frombuffer(m.data, dtype=dtype).reshape(m.height, m.step // size)[:, :m.width].astype(float) * scale
    out[~np.isfinite(out) | (out <= 0)] = np.nan
    return out


class SearchNode(Node):
    def __init__(self):
        super().__init__('search_mission')
        dp = self.declare_parameter
        self.base_frame = str(dp('base_frame', 'base_link').value)
        self.frame = 'odom'
        self.allow_takeoff = bool(dp('allow_takeoff', False).value)
        self.max_stamp_age = .5
        self.qualify_s = 3.0
        self.sync_tolerance = .12
        self.depth_hz = float(dp('mapping_hz', 4.).value)
        self.detector_hz = float(dp('detector_hz', 3.).value)
        gp = GridParams(resolution_m=float(dp('resolution_m', .05).value),
                        extent_m=float(dp('map_extent_m', 16.).value))
        mp = MissionParams(cruise_height_m=float(dp('cruise_height_m', .8).value),
                           v_cruise_mps=float(dp('v_cruise_mps', .4).value),
                           v_gap_mps=float(dp('v_gap_mps', .2).value),
                           time_limit_s=float(dp('time_limit_s', 180.).value))
        self.grid = DepthGrid(gp)
        launch_bounds_text = str(dp('launch_volume_bounds_json', '').value)
        launch_bounds = None if not launch_bounds_text else tuple(json.loads(launch_bounds_text))
        self.launch_volume = LaunchVolume(LaunchVolumeParams(
            enabled=bool(dp('pilot_verified_launch',False).value), relative_bounds=launch_bounds,
            duration_s=float(dp('launch_volume_duration_s',30.).value)))
        self.mission = SearchMission(self.grid, mp)
        detector_p = DetectorParams(qualified=bool(dp('target_contrast_qualified', False).value))
        self.detector = MonoFloorDetector(detector_p)
        self.tracker = TargetTracker(detector_p)
        self.rgb_detector = RGBBlueDetector(hsv=HSVParams(
            hue_low=int(dp('blue_hue_low', 90).value), hue_high=int(dp('blue_hue_high', 135).value),
            saturation_min=int(dp('blue_saturation_min', 80).value), value_min=int(dp('blue_value_min', 40).value)))
        SearchNode.reset_rgb(self)
        SearchNode.reset_openings(self)
        self.samples, self.errors, self.counts = {}, {}, {}
        self.pose_history, self.att_history = deque(maxlen=800), deque(maxlen=800)
        self.last_source = {}
        self.position, self.velocity, self.yaw = None, (0., 0., 0.), None
        self.att = None
        self.armed, self.offboard, self.landed = False, False, None
        self.health, self.position_valid, self.link, self.frozen = False, False, False, False
        self.ev_yaw, self.yaw_aligned, self.ev_yaw_last = False, False, -math.inf
        self.state = {}
        self.saw_disarmed = False
        self.qualified_since = None
        self.fault = ''
        self.last_reset = None
        self.frame_epoch_stamp = -math.inf
        self.frame_resets = []
        self.capture_waiting = set()
        self.px4_offset = None
        self.px4_clock = Px4ClockAnchor()
        self.floor_z = None
        self.floor_source = None
        self.floor_tracker = FloorTracker()
        self.depth_watchdog = DepthWatchdog(.5)
        self.last_depth, self.last_depth_run, self.last_mono_run = None, -math.inf, -math.inf
        self.cameras = {}
        self.latest_candidates = []
        self.depth_valid_fraction = 0.
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.twist_pub = self.create_publisher(TwistStamped, '/skybeta/setpoint/twist', 10)
        self.status_pub = self.create_publisher(String, '/searchnav/status', 10)
        self.path_pub = self.create_publisher(Path, '/searchnav/path', 5)
        self.grid_pub = self.create_publisher(OccupancyGrid, '/searchnav/occupancy', 5)
        self.safe_grid_pub = self.create_publisher(OccupancyGrid, '/searchnav/traversability', 5)
        self.candidates_pub = self.create_publisher(String, '/searchnav/candidates', 5)
        self.command_pub = self.create_publisher(String, '/searchnav/command', 10)
        self.target_image_pub = self.create_publisher(Image, '/searchnav/target_image', qos_profile_sensor_data)
        self.blue_mask_pub = self.create_publisher(Image, '/searchnav/blue_mask', qos_profile_sensor_data)
        self.opening_pub = self.create_publisher(String, '/searchnav/opening', 5)
        self.create_subscription(VehicleLocalPosition, versioned('vehicle_local_position', VehicleLocalPosition), self.on_position, PX4_QOS)
        self.create_subscription(VehicleAttitude, versioned('vehicle_attitude', VehicleAttitude), self.on_attitude, PX4_QOS)
        self.create_subscription(VehicleStatus, versioned('vehicle_status', VehicleStatus), self.on_status, PX4_QOS)
        self.create_subscription(VehicleLandDetected, versioned('vehicle_land_detected', VehicleLandDetected), self.on_land, PX4_QOS)
        self.create_subscription(EstimatorStatusFlags, versioned('estimator_status_flags', EstimatorStatusFlags), self.on_flags, PX4_QOS)
        self.create_subscription(Bool, '/estimator/healthy', self.on_health, 10)
        self.create_subscription(String, '/skybeta/state', self.on_state, 10)
        self.create_subscription(Odometry, '/visual_slam/tracking/odometry', self.on_vio, 10)
        depth_topic = str(dp('depth_topic', '/kestrel_nav/depth').value)
        depth_info = str(dp('depth_info_topic', '/kestrel_nav/depth/camera_info').value)
        mono_topic = str(dp('mono_topic', '/oak/left/image_mono').value)
        mono_info = str(dp('mono_info_topic', '/oak/left/camera_info').value)
        self.create_subscription(CameraInfo, depth_info, lambda m: self.on_info('depth', m), qos_profile_sensor_data)
        self.create_subscription(CameraInfo, mono_info, lambda m: self.on_info('mono', m), qos_profile_sensor_data)
        self.create_subscription(Image, depth_topic, self.on_depth, qos_profile_sensor_data)
        self.create_subscription(Image, mono_topic, self.on_mono, qos_profile_sensor_data)
        rgb_topic = str(dp('rgb_topic', '/oak/rgb/image_raw').value)
        rgb_info = str(dp('rgb_info_topic', '/oak/rgb/camera_info').value)
        self.create_subscription(CameraInfo, rgb_info, lambda m: self.on_info('rgb', m), qos_profile_sensor_data)
        self.create_subscription(Image, rgb_topic, self.on_rgb, qos_profile_sensor_data)
        self.create_timer(.05, self.tick)
        self.create_timer(.5, self.publish_diagnostics)
        self.last_command = None
        self.ready = False
        self.ready_reason = 'waiting for inputs'
        self.last_tick_wall = time.monotonic()
        self.readiness_failures = []
        self.source_limits = dict(INPUT_LIMITS)
        self.get_logger().info('Fresh-map search: no prior map/PF/waypoints. Waiting for qualified sensors and pilot. '
                               f'launch column approved={self.allow_takeoff}; mono appearance qualified={detector_p.qualified}')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def reset_rgb(self):
        self.rgb_tracker = TargetTracker(DetectorParams(qualified=True), source='rgb_hsv', color_observed=True)
        self.rgb_candidates, self.rgb_reason = [], 'waiting for calibrated RGB and matched depth'
        self.last_rgb_run, self.rgb_depth_supported_at = -math.inf, -math.inf

    def reset_openings(self):
        self.opening_detector = DepthOpeningDetector()
        self.opening_tracker = OpeningTracker()
        self.clearance_history = DepthClearanceHistory()
        self.clearance_history.launch_volume = getattr(self,'launch_volume',None)
        self.opening_candidates = []
        self.opening_error = ''
        self.corridor_status = {}
        self.slab_replay_queue = deque()
        self.slab_replay_processed = 0

    def start_slab_replay(self, now):
        self.slab_replay_queue = deque(self.clearance_history.replay_frames(now))
        self.slab_replay_processed = 0

    def process_slab_replay(self):
        if self.mission.phase != 'rebuild_height':
            self.slab_replay_queue.clear()
            return
        deadline, processed = time.monotonic()+.03, 0
        while self.slab_replay_queue and processed < 8:
            frame = self.slab_replay_queue.popleft()
            function = self.grid.integrate if frame['full'] else self.grid.integrate_occupied
            function(frame['depth'], frame['K'], frame['R'], frame['t'], frame['stamp'])
            self.slab_replay_processed += 1
            processed += 1
            if frame['full'] or time.monotonic() >= deadline:
                break

    def reject(self, source, reason, *, transient=False):
        self.errors[source] = reason
        self.counts[source + '_rejected'] = self.counts.get(source + '_rejected', 0) + 1
        if self.mission.started is not None and source not in ('mono', 'target') and not transient:
            self.fault = self.fault or reason

    def accept(self, source, capture, limit=.5, *, transient=False):
        now = self.now()
        if not math.isfinite(now) or now <= 0 or not math.isfinite(capture) or capture <= 0:
            self.reject(source, f'{source}: waiting for initialized source/ROS clock', transient=transient)
            return False
        # Only spatial observations are expressed in the reset PX4 frame.
        # VIO is used here solely as an independently framed health stream,
        # never to place map points. Its normally delayed packets can remain
        # fresh across a PX4 correction; rejecting them fabricates a dropout.
        if source in ('position', 'attitude', 'depth', 'mono', 'rgb') and capture < self.frame_epoch_stamp:
            self.counts[source + '_pre_epoch'] = self.counts.get(source + '_pre_epoch', 0) + 1
            return False
        if not math.isfinite(capture) or not -.05 <= now - capture <= limit:
            self.reject(source, f'{source}: capture stale/future', transient=transient)
            return False
        previous = self.samples.get(source)
        if previous is not None and capture <= previous[0]:
            return False
        self.samples[source] = (capture, now)
        self.errors.pop(source, None)
        return True

    def px4_accept(self, source, m, limit=.5):
        boot = float(m.timestamp) * 1e-6
        now = self.now()
        if source == 'position':
            resets = self.px4_clock.resets
            offset = self.px4_clock.observe(now, boot, active=self.mission.started is not None)
            if self.px4_clock.fault:
                self.fault = self.px4_clock.fault
                return None
            if self.px4_clock.resets != resets:
                # A new boot-clock epoch must not remain blocked by packet
                # counters from the previous domain. A plain EKF pose reset
                # does not take this path and retains duplicate protection.
                self.last_source.clear()
                self.reset_preflight_frame(self.last_reset, now, self.position or (0., 0., 0.))
                self.mission.reason = 'ground clock discontinuity: remeasure floor and requalify all inputs'
            self.px4_offset = self.px4_clock.offset
        # PX4 DDS can arrive before the first /clock. Anchoring a min-offset at
        # ROS time zero then permanently poisons every later source timestamp.
        # Neither clock state nor freshness may be changed until both clocks
        # are initialized. Hardware ROS system time is already positive.
        if not math.isfinite(now) or now <= 0 or not math.isfinite(boot) or boot <= 0:
            self.counts[source + '_waiting_clock'] = self.counts.get(source + '_waiting_clock', 0) + 1
            return None
        if boot <= self.last_source.get(source, -math.inf):
            return None
        if self.px4_offset is None:
            return None
        capture = boot + self.px4_offset
        if not self.accept(source, capture, limit):
            return None
        self.last_source[source] = boot
        return capture

    def on_position(self, m):
        captured = self.px4_accept('position', m)
        if captured is None:
            return
        values = (m.x, m.y, m.z, m.vx, m.vy, m.vz, m.heading)
        self.position_valid = bool(m.xy_valid and m.z_valid and m.v_xy_valid and m.v_z_valid
                                   and all(math.isfinite(float(x)) for x in values))
        if not self.position_valid:
            self.reject('position', 'PX4 fused position/velocity invalid')
            return
        reset = tuple(getattr(m, k, None) for k in ('xy_reset_counter', 'z_reset_counter', 'heading_reset_counter'))
        if self.last_reset is not None and reset != self.last_reset:
            if self.mission.started is not None:
                self.handle_airborne_reset(m, reset, captured, (float(m.y), float(m.x), -float(m.z)))
            else:
                self.reset_preflight_frame(reset, captured, (float(m.y), float(m.x), -float(m.z)))
        self.last_reset = reset
        self.position = (float(m.y), float(m.x), -float(m.z))
        self.velocity = (float(m.vy), float(m.vx), -float(m.vz))
        self.yaw = (math.pi / 2 - float(m.heading) + math.pi) % (2 * math.pi) - math.pi
        self.pose_history.append((captured, np.asarray(self.position)))
        distance = float(getattr(m, 'dist_bottom', math.nan))
        if self.mission.started is None and self.landed is True and bool(getattr(m, 'dist_bottom_valid', False)) and .03 < distance < .6:
            self.floor_z = self.position[2] - distance
            self.floor_source = 'px4_dist_bottom'
            if self.grid.height is None:
                # A fresh map begins before arming, at the intended relative
                # flight-height slab. Only actual visible voxels become free.
                self.grid.reset(self.position[:2], self.position[2] + self.mission.p.cruise_height_m)

    def reset_preflight_frame(self, counters, captured, position):
        volume = getattr(self,'launch_volume',None)
        if volume is not None and volume.grant_attempted:
            volume.revoke('ground frame reset after consumed pilot approval; fresh explicit launch required')
        """An EKF frame change invalidates all preflight spatial observations.

        Keep camera-to-body calibration, but no old floor, target, map voxel or
        pose/attitude history can be interpreted in the new local frame. Require
        fresh stream samples, a new measured floor and the full 3s qualification.
        """
        self.frame_resets.append(dict(t=captured, old=self.last_reset, new=counters))
        self.frame_epoch_stamp = captured
        self.pose_history.clear()
        self.att_history.clear()
        self.att = None
        self.samples.clear()
        self.errors.clear()
        self.floor_z, self.floor_source = None, None
        self.floor_tracker = FloorTracker()
        self.depth_watchdog = DepthWatchdog(.5)
        self.last_depth = None
        self.last_depth_run, self.last_mono_run = -math.inf, -math.inf
        self.latest_candidates = []
        self.tracker = TargetTracker(self.tracker.p)
        SearchNode.reset_rgb(self)
        SearchNode.reset_openings(self)
        self.grid.reset(position[:2], None)
        self.mission = SearchMission(self.grid, self.mission.p)
        self.mission.transition('wait', 'ground EKF frame reset: remeasure floor and requalify all inputs', captured)
        self.qualified_since, self.ready, self.saw_disarmed = None, False, False
        self.last_command = None

    def handle_airborne_reset(self, message, counters, captured, position):
        if hasattr(self,'launch_volume'):
            self.launch_volume.revoke('airborne EKF reset: pilot volume cannot transfer to a new frame')
        decision, why = verify_small_reset(self.last_reset, counters, message)
        authorized = bool(self.armed and self.offboard and self.state.get('armed') is True
                          and self.state.get('offboard_active') is True and not self.frozen)
        measured_floor = self.floor_z is not None and math.isfinite(self.floor_z)
        description = f'EKF frame reset {self.last_reset} -> {counters}'
        allowed = bool(decision is not None and authorized and measured_floor and not self.fault
                       and self.mission.phase not in ('yield', 'exhausted'))
        if allowed:
            allowed = self.mission.begin_reset_recovery(captured, position, decision['delta_z_enu'], description)
        event = dict(t=captured, old=self.last_reset, new=counters, recoverable=allowed,
                     decision=decision, reason=why or (self.mission.reason if allowed else 'authority/recovery bound or prior fault'))
        self.frame_resets.append(event)
        # Withdraw every frame-dependent spatial object before another command.
        self.frame_epoch_stamp = captured
        self.pose_history.clear()
        self.att_history.clear()
        self.att = None
        self.last_depth = None
        self.last_depth_run, self.last_mono_run = -math.inf, -math.inf
        self.latest_candidates = []
        self.tracker = TargetTracker(self.tracker.p)
        SearchNode.reset_rgb(self)
        SearchNode.reset_openings(self)
        self.capture_waiting = set()
        self.qualified_since, self.ready = None, False
        self.mission.route = []
        if allowed:
            # The already measured horizontal floor is a scalar Z reference.
            # XY/yaw resets leave it unchanged; a verified Z reset translates
            # it by the exact same delta as the altitude hold. No voxel or
            # target is retained and no new free cell is manufactured.
            self.floor_z += decision['delta_z_enu']
            self.floor_source = 'measured_floor_after_verified_frame_translation'
            self.floor_tracker = FloorTracker()
            self.floor_tracker.reason = 'retained measured horizontal floor scalar under verified translation'
            self.floor_tracker.measurement = dict(floor_z=self.floor_z, source=self.floor_source)
        else:
            self.fault = self.fault or (why or self.mission.reason or description)
            self.floor_z, self.floor_source = None, None
            self.grid.reset(position[:2], None)
        # Keep actual last-good critical-stream timestamps, NOT their spatial
        # content. Missing reset history may pause for only the existing .5s
        # depth/mono deadlines. Health/authority/clock checks remain unchanged.

    def on_attitude(self, m):
        captured = self.px4_accept('attitude', m)
        q = tuple(float(x) for x in m.q)
        if captured is None:
            return
        if not all(math.isfinite(x) for x in q) or not .9 < math.sqrt(sum(x*x for x in q)) < 1.1:
            self.reject('attitude', 'invalid PX4 attitude')
            return
        q = np.asarray(q) / np.linalg.norm(q)
        self.att = NED_TO_ENU @ quat_R(*q) @ FLU_TO_FRD
        self.att_history.append((captured, self.att.copy()))

    def on_status(self, m):
        if self.px4_accept('status', m, 1.) is None:
            return
        self.armed = int(m.arming_state) == 2
        self.offboard = int(m.nav_state) == 14
        self.saw_disarmed |= not self.armed

    def on_land(self, m):
        if self.px4_accept('land', m, 2.6) is not None:
            self.landed = bool(m.landed)

    def on_flags(self, m):
        if self.px4_accept('flags', m, EV_YAW_FRESH_S) is None:
            return
        self.ev_yaw = bool(getattr(m, 'cs_ev_yaw', False))
        self.yaw_aligned = bool(getattr(m, 'cs_yaw_align', False))
        if self.ev_yaw:
            self.ev_yaw_last = self.now()

    def on_health(self, m):
        self.health = bool(m.data)
        self.accept('health', self.now())
        if not self.health:
            self.reject('health', 'VIO health reports unhealthy')

    def on_vio(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        covariance = np.asarray(m.pose.covariance)
        values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        if (not all(math.isfinite(float(x)) for x in values) or not m.header.frame_id
                or not np.isfinite(covariance).all() or (np.diag(covariance.reshape(6, 6)) < 0).any()):
            self.reject('vio', 'invalid VIO pose/frame/covariance')
            return
        frame = (m.header.frame_id, m.child_frame_id)
        if hasattr(self, 'vio_frame') and frame != self.vio_frame:
            self.reject('vio', 'VIO frame changed; restart on ground')
            return
        self.vio_frame = frame
        self.accept('vio', stamp(m))

    def on_state(self, m):
        try:
            state = json.loads(m.data)
            if any(type(state.get(k)) is not bool for k in ('armed', 'offboard_active', 'px4_link', 'frozen')):
                raise ValueError('missing authority flags')
            captured = float(state['stamp'])
            if not self.accept('state', captured, 1.):
                return
            self.state = state
            self.link, self.frozen = state['px4_link'], state['frozen']
        except (ValueError, TypeError, KeyError):
            self.reject('state', 'invalid flight-interface authority state')

    def on_info(self, which, m):
        K = (float(m.k[0]), float(m.k[4]), float(m.k[2]), float(m.k[5]))
        if not m.header.frame_id or m.width <= 0 or m.height <= 0 or not all(math.isfinite(x) for x in K) or min(K[:2]) <= 0:
            self.reject(which, 'invalid CameraInfo', transient=which == 'rgb')
            return
        self.cameras[which] = m

    def calibrated_pose(self, which, image):
        info = self.cameras.get(which)
        if info is None or info.header.frame_id != image.header.frame_id or info.width != image.width or info.height != image.height:
            raise ValueError(f'{which}: image and CameraInfo frame/size disagree')
        captured = stamp(image)
        def nearest(history):
            if not history:
                raise CaptureHistoryUnavailable('capture-time fused pose absent')
            sample = min(history, key=lambda a: abs(a[0] - captured))
            if abs(sample[0] - captured) > self.sync_tolerance:
                raise CaptureHistoryUnavailable('capture-time fused pose/attitude is stale')
            return sample[1]
        body_pos, R_world_body = nearest(self.pose_history), nearest(self.att_history)
        transform = self.tf.lookup_transform(self.base_frame, info.header.frame_id, Time.from_msg(image.header.stamp))
        q, t = transform.transform.rotation, transform.transform.translation
        R_body_cam = quat_R(q.w, q.x, q.y, q.z)
        t_body_cam = np.array([t.x, t.y, t.z])
        return ((info.k[0], info.k[4], info.k[2], info.k[5]),
                R_world_body @ R_body_cam, body_pos + R_world_body @ t_body_cam, info)

    def on_depth(self, m):
        now = self.now()
        map_due = now - self.last_depth_run >= 1. / max(self.depth_hz, .5)
        try:
            captured = stamp(m)
            if captured < self.frame_epoch_stamp:
                return
            previous = self.samples.get('depth')
            if previous is not None and captured <= previous[0]:
                return
            if not -.05 <= now-captured <= .5:
                raise ValueError('depth capture stale/future')
            K, R, t, info = self.calibrated_pose('depth', m)
            if any(abs(float(x)) > 1e-8 for x in info.d):
                raise ValueError('depth must be rectified; nonzero distortion is unsupported')
            depth = decode_depth(m)
            self.depth_valid_fraction = float(np.mean(np.isfinite(depth)))
            if self.depth_valid_fraction < .3:
                raise ValueError('depth valid fraction below existing .30 floor')
            if map_due and self.floor_z is None and self.landed is True and self.mission.started is None:
                body = min(self.pose_history, key=lambda x: abs(x[0]-captured))[1]
                measurement, why = fit_floor(depth, K, R, t, body)
                ground = self.floor_tracker.update(measurement, captured, why)
                if ground is not None:
                    self.floor_z = ground['floor_z']
                    self.floor_source = ground['source']
                    self.grid.reset(body[:2], body[2]+self.mission.p.cruise_height_m)
            if self.grid.height is not None:
                if map_due:
                    changed = self.grid.integrate(depth, K, R, t, captured)
                    self.counts['map_full_frames'] = self.counts.get('map_full_frames', 0) + 1
                else:
                    changed = self.grid.integrate_occupied(depth, K, R, t, captured)
                    self.counts['map_hit_frames'] = self.counts.get('map_hit_frames', 0) + 1
                self.counts['changed_voxels'] = self.counts.get('changed_voxels', 0) + changed
            # Timestamp freshness is committed ONLY for a fully usable frame;
            # failed calibration/TF/decoding/coverage/mapping refreshes nothing.
            if not self.accept('depth', captured, transient=True):
                return
            self.depth_watchdog.usable(captured, self.now())
            self.last_depth = (captured, depth, K, R, t)
            try:
                self.clearance_history.integrate(depth, K, R, t, captured, self.now())
                if map_due:
                    openings = self.opening_detector.detect(depth, K, R, t, captured)
                    self.opening_candidates = [item.status() for item in openings]
                    self.opening_tracker.update(openings, captured)
                self.opening_error = ''
            except Exception as exc:
                # Optional aperture inference never refreshes or bypasses a
                # critical sensor guard. A failed checker grants no 3-D motion.
                self.clearance_history.reset()
                self.opening_tracker.reset()
                self.opening_candidates = []
                self.opening_error = f'{type(exc).__name__}: {exc}'
            self.capture_waiting.discard('depth')
            self.counts['depth_frames'] = self.counts.get('depth_frames', 0) + 1
            if map_due:
                self.last_depth_run = now
        except Exception as exc:  # TF may be unavailable during startup.
            if isinstance(exc, CaptureHistoryUnavailable):
                self.capture_waiting.add('depth')
            self.depth_watchdog.reject(str(exc))
            self.reject('depth', str(exc), transient=True)

    def on_mono(self, m):
        now = self.now()
        try:
            if stamp(m) < self.frame_epoch_stamp:
                return
            if m.encoding.lower() not in ('mono8', '8uc1'):
                raise ValueError('target detector requires raw MONO8; color identities remain unobserved')
            K, R, t, info = self.calibrated_pose('mono', m)
            mono = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width]
            if not self.accept('mono', stamp(m)):
                return
            self.capture_waiting.discard('mono')
            self.counts['mono_frames'] = self.counts.get('mono_frames', 0) + 1
            # Every calibrated usable image supervises freshness; only costly
            # inference/undistortion is rate capped. Throttling the watchdog
            # itself falsely makes a healthy 15Hz camera look stale.
            if now-self.last_mono_run < 1. / max(self.detector_hz, .5):
                return
            self.last_mono_run = now
            if any(abs(float(x)) > 1e-8 for x in info.d):
                mono = cv2.undistort(mono, np.asarray(info.k).reshape(3, 3), np.asarray(info.d))
            if self.last_depth is None or abs(self.last_depth[0] - stamp(m)) > self.sync_tolerance or self.floor_z is None:
                self.latest_candidates = []
                self.tracker.update([], t[:2], stamp(m))
                return
            _, depth, Kd, Rd, td = self.last_depth
            candidates = self.detector.detect(mono, K, R, t, self.floor_z, depth, Kd, Rd, td)
            self.latest_candidates = [asdict(c) for c in candidates]
            self.tracker.update(candidates, t[:2], stamp(m))
        except Exception as exc:
            if isinstance(exc, CaptureHistoryUnavailable):
                self.capture_waiting.add('mono')
            self.reject('mono', str(exc))

    def on_rgb(self, m):
        try:
            captured, now = stamp(m), self.now()
            if captured < self.frame_epoch_stamp:
                return
            K, R, t, info = self.calibrated_pose('rgb', m)
            bgr = decode_bgr(m)
            if not self.accept('rgb', captured, transient=True):
                return
            self.counts['rgb_frames'] = self.counts.get('rgb_frames', 0) + 1
            if now-self.last_rgb_run < 1. / max(self.detector_hz, .5):
                return
            self.last_rgb_run = now
            if any(abs(float(x)) > 1e-8 for x in info.d):
                bgr = cv2.undistort(bgr, np.asarray(info.k).reshape(3, 3), np.asarray(info.d))
            matched = self.last_depth is not None and abs(self.last_depth[0]-captured) <= self.sync_tolerance
            if matched:
                _, depth, Kd, Rd, td = self.last_depth
            else:
                depth, Kd, Rd, td = None, None, None, None
            candidates = self.rgb_detector.detect(bgr, K, R, t, self.floor_z, depth, Kd, Rd, td)
            self.rgb_candidates = [asdict(c) for c in candidates]
            self.rgb_tracker.update(candidates, t[:2], captured)
            self.rgb_reason = ('RGB/depth capture mismatch: no target promotion' if not matched else
                               'no measured floor: no target promotion' if self.floor_z is None else
                               'HSV candidates verified with independent RGB/depth calibration')
            self.rgb_depth_supported_at = captured if matched and self.floor_z is not None else -math.inf
            annotated = bgr.copy()
            for c in candidates:
                x, y, w, h = c.box
                cv2.rectangle(annotated, (x,y), (x+w,y+h), (0,255,0), 2)
            label = 'BLUE CONFIRMED' if self.rgb_tracker.confirmed else f'blue candidates: {len(candidates)}'
            cv2.putText(annotated, label, (10,24), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,0), 2)
            for image_data, encoding, publisher in ((annotated,'bgr8',self.target_image_pub),
                                                   (self.rgb_detector.last_mask,'mono8',self.blue_mask_pub)):
                out = Image()
                out.header, out.height, out.width = m.header, image_data.shape[0], image_data.shape[1]
                out.encoding, out.is_bigendian = encoding, 0
                out.step = out.width*(3 if encoding=='bgr8' else 1)
                out.data = image_data.tobytes()
                publisher.publish(out)
        except Exception as exc:
            self.rgb_reason = str(exc)
            self.reject('rgb', str(exc), transient=True)

    def selected_target(self, now):
        rgb = self.rgb_tracker.confirmed
        if (rgb is not None and 0 <= now-rgb[1] <= .5
                and 0 <= now-self.rgb_depth_supported_at <= .5 and 'rgb' not in self.errors):
            return rgb, 'rgb_hsv'
        mono = self.tracker.confirmed
        if mono is not None and 0 <= now-mono[1] <= self.mission.p.target_fresh_s:
            return mono, 'qualified_mono'
        return None, None

    def readiness(self):
        now = self.now()
        for source, age in self.source_limits.items():
            sample = self.samples.get(source)
            transient_depth = source == 'depth' and self.depth_watchdog.action(now) == 'pause'
            expected_history = (self.mission.phase == 'recover' and source == 'mono'
                                and source in self.capture_waiting)
            if source in self.errors and not transient_depth and not expected_history:
                return False, self.errors[source]
            if sample is None or not -.05 <= now - sample[0] <= age or not 0 <= now - sample[1] <= age:
                return False, f'{source}: absent/stale input'
        if not self.link or self.frozen:
            return False, 'interface link lost or hold requested'
        if self.mission.started is not None and (self.state.get('offboard_active') != self.offboard or self.state.get('armed') != self.armed):
            return False, 'PX4 and interface authority disagree'
        if not self.position_valid or not self.health:
            return False, 'position or VIO health invalid'
        if not (self.yaw_aligned or self.ev_yaw or now - self.ev_yaw_last <= EV_YAW_FRESH_S):
            return False, 'PX4 yaw not aligned or recently fused from vision'
        if self.floor_z is None:
            return False, 'no measured ground-plane height: ' + self.floor_tracker.reason
        if not self.allow_takeoff:
            return False, 'vertical launch column not approved: allow_takeoff=false'
        return True, 'fresh calibrated sensors and fused local pose'

    def input_ages(self, now):
        return {source: dict(capture_age_s=None if source not in self.samples else now-self.samples[source][0],
                             receipt_age_s=None if source not in self.samples else now-self.samples[source][1],
                             max_age_s=limit, error=self.errors.get(source))
                for source, limit in self.source_limits.items()}

    def recovery_spatial_ready(self):
        return bool(self.floor_z is not None and self.grid.height is not None
                    and all(source in self.samples and self.samples[source][0] >= self.frame_epoch_stamp
                            for source in ('position', 'attitude', 'depth', 'mono')))

    def corridor_error(self, velocity):
        """Fresh depth must actually view the commanded stopping cross-section.

        The map may retain older side observations, but they cannot replace the
        current .60 valid-ray/.30 frame-validity guards when moving. A sensor
        guard failure stops and discards the path before requesting another view.
        """
        if self.last_depth is None or self.position is None:
            return 'no calibrated depth for commanded motion corridor'
        captured, depth, K, R, t = self.last_depth
        # A farther view plane supplements the existing near-body certificate;
        # it cannot make unknown launch space traversable.
        if self.mission.phase == 'center_opening':
            decision = self.mission.centering_status
            certified = bool(decision.get('requires_3d_clearance')
                and decision.get('reason') == 'fresh swept 3-D clearance certified'
                and len(decision.get('authorized_velocity', ())) == len(velocity)
                and np.allclose(decision.get('authorized_velocity', ()), velocity))
        else:
            speed = max(math.hypot(*velocity[:2]), math.hypot(*self.velocity[:2]))
            distance = speed*self.mission.p.command_delay_s+speed**2/(2*self.mission.p.braking_mps2)+.12
            direction = np.asarray(velocity[:2],float)/max(math.hypot(*velocity[:2]),1e-9)
            endpoint = np.asarray(self.position[:2])+direction*distance
            certified = bool(self.mission.route
                and self.grid.route_safe(self.mission._remaining_route(self.position[:2]))
                and self.grid.segment_safe(self.position[:2], endpoint))
        sample = self.samples.get('depth')
        pose = self.samples.get('position')
        assessment = assess_motion_corridor(depth, K, R, t, self.position, velocity, self.velocity,
            captured, None if sample is None else sample[1], self.now(),
            pose_stamp=None if pose is None else pose[0], route_certified=certified,
            params=CorridorParams(radius_m=self.grid.p.radius_m,margin_m=self.grid.p.margin_m,
                half_height_m=self.grid.p.half_height_m,braking_mps2=self.mission.p.braking_mps2,
                reaction_delay_s=self.mission.p.command_delay_s,depth_error_m=self.grid.p.depth_error_m))
        self.corridor_status = assessment.status()
        return None if assessment.allowed else assessment.reason

    def update_launch_volume(self, observation):
        """Consume approval only at the qualified pilot ground transition."""
        now = observation.now
        if (self.mission.started is None and self.launch_volume.p.enabled
                and not self.launch_volume.grant_attempted
                and observation.ready and observation.armed and observation.offboard and observation.landed):
            self.launch_volume.grant(observation.position,now,pilot_verified=True,
                saw_disarmed=self.saw_disarmed,allow_takeoff=self.allow_takeoff,pilot_started=True,
                provenance='explicit pilot_verified_launch parameter and fixed launch_volume_bounds_json')
        if self.mission.started is not None and (observation.fault or not observation.armed or not observation.offboard):
            self.launch_volume.revoke('critical fault or pilot/PX4 authority lost')
        self.grid.set_launch_volume(self.launch_volume,now)

    def tick(self):
        now = self.now()
        okay, reason = self.readiness()
        if not okay and (not self.readiness_failures or self.readiness_failures[-1]['reason'] != reason
                         or now-self.readiness_failures[-1]['t'] >= .5):
            self.readiness_failures.append(dict(t=now, reason=reason,
                                                qualified_since=self.qualified_since,
                                                input_ages=self.input_ages(now)))
            self.readiness_failures = self.readiness_failures[-50:]
        spatial_ready = self.mission.phase != 'recover' or self.recovery_spatial_ready()
        if okay and spatial_ready:
            self.qualified_since = now if self.qualified_since is None else self.qualified_since
        else:
            self.qualified_since = None
        qualified = okay and spatial_ready and self.qualified_since is not None and now - self.qualified_since >= self.qualify_s
        depth_pause = self.depth_watchdog.action(now) == 'pause'
        self.ready = bool(qualified and self.saw_disarmed and not depth_pause)
        self.ready_reason = (reason if not okay else
                             'rebuilding post-reset pose/depth/image history' if not spatial_ready else
                             'transient unusable depth: paused' if depth_pause else
                             f'qualifying continuously: {now-self.qualified_since:.2f}/{self.qualify_s:.2f}s'
                             if not qualified else
                             'waiting for a fresh disarmed status' if not self.saw_disarmed else reason)
        # A ground transition may differ between two subscriptions for one tick;
        # airborne disagreement yields immediately rather than guessing authority.
        if self.mission.started is not None and not okay:
            self.fault = self.fault or reason
        if self.position is None or self.yaw is None:
            return
        target, target_source = self.selected_target(now)
        before = self.mission.phase
        if okay and self.ready:
            self.process_slab_replay()
        observation = Observation(now=now, position=self.position, yaw=self.yaw,
            velocity=self.velocity, ready=self.ready,
            offboard=self.offboard and self.state.get('offboard_active') is True,
            armed=self.armed and self.state.get('armed') is True,
            landed=self.landed is True, fault=self.fault,
            target_xy=None if target is None else target[0], target_stamp=-math.inf if target is None else target[1],
            inputs_fresh=okay, opening=self.opening_tracker.current(now),
            certify_sweep=self.clearance_history.certify, slab_ready=not self.slab_replay_queue)
        self.update_launch_volume(observation)
        if depth_pause and self.mission.started is not None and not self.fault and self.mission.phase != 'recover':
            output = Command(self.mission.phase, 'transient unusable depth: pause translation and yaw', (0., 0., 0., 0.))
        else:
            output = self.mission.tick(observation)
        if self.mission.phase in ('yield','exhausted'):
            self.launch_volume.revoke('mission yielded or bounded exploration exhausted')
            self.grid.refresh_launch_volume(now)
        if before == 'center_opening' and self.mission.phase == 'rebuild_height':
            self.start_slab_replay(now)
        if output.velocity is not None and math.hypot(*output.velocity[:2]) > 1e-6:
            corridor = self.corridor_error(output.velocity)
            if corridor:
                # Calibrated depth has already been integrated by on_depth.
                # Invalid rays are left unknown. A corridor refusal must ALSO
                # invalidate the follower: it cannot keep replaying this path.
                self.mission.hold_xy = np.asarray(self.position[:2])
                # Bring the full current capture (including conservative range
                # uncertainty) into A* now, even on a hit-only mapping tick.
                # Timestamp ordering prevents duplicate free-clearing evidence.
                if self.last_depth is not None:
                    captured,depth,K,R,t = self.last_depth
                    self.grid.integrate(depth,K,R,t,captured)
                if self.mission.phase == 'center_opening':
                    self.mission.end_centering(observation, corridor + '; rotate to observe the 3-D stopping corridor')
                    self.start_slab_replay(now)
                elif self.mission.phase in ('move','align'):
                    self.mission.reject_motion_view(observation,corridor)
                # A non-route stationary phase must never be promoted into
                # exploration by a camera refusal (notably during takeoff).
                output = self.mission._hold(Observation(now, self.position, self.yaw, self.velocity))
        if before != output.phase:
            self.get_logger().info(f'{before} -> {output.phase}: {output.reason}')
        self.last_command = dict(t=now, phase=output.phase, reason=output.reason,
                                 velocity_enu=None if output.velocity is None else list(output.velocity),
                                 map_revision=self.grid.revision)
        self.command_pub.publish(String(data=json.dumps(self.last_command)))
        if output.velocity is not None:
            cmd = TwistStamped()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.header.frame_id = self.frame
            cmd.twist.linear.x, cmd.twist.linear.y, cmd.twist.linear.z = map(float, output.velocity[:3])
            cmd.twist.angular.z = float(output.velocity[3])
            self.twist_pub.publish(cmd)

    def publish_diagnostics(self):
        status = self.mission.status()
        status.update(t=self.now(), ready=self.ready, ready_reason=self.ready_reason,
                      qualification_since=self.qualified_since,
                      qualification_elapsed_s=None if self.qualified_since is None else self.now()-self.qualified_since,
                      qualification_required_s=self.qualify_s, input_ages=self.input_ages(self.now()),
                      last_readiness_failures=self.readiness_failures[-10:],
                      fault=self.fault, armed=self.armed, offboard=self.offboard, landed=self.landed,
                      position=self.position, velocity=self.velocity, yaw=self.yaw,
                      depth_valid_fraction=self.depth_valid_fraction, floor_z=self.floor_z,
                      floor_source=self.floor_source, ground_fit=self.floor_tracker.status(),
                      frame_epoch_stamp=self.frame_epoch_stamp if math.isfinite(self.frame_epoch_stamp) else None,
                      frame_resets=self.frame_resets[-10:],
                      recovery_spatial_ready=self.recovery_spatial_ready() if self.mission.phase == 'recover' else None,
                      depth_watchdog=self.depth_watchdog.status(self.now()),
                      px4_clock=self.px4_clock.status(),
                      input_errors=self.errors, sensor_counts=self.counts, map_origin=self.grid.origin.tolist(),
                      radius_m=self.grid.p.radius_m, margin_m=self.grid.p.margin_m,
                      map=self.grid.summary(None if self.position is None else self.position[:2]),
                      launch_column_approved=self.allow_takeoff, target=self.tracker.status())
        selected, selected_source = self.selected_target(self.now())
        status.update(rgb_target=dict(self.rgb_tracker.status(), reason=self.rgb_reason,
                      hsv=asdict(self.rgb_detector.hsv), depth_supported_at=(self.rgb_depth_supported_at
                      if math.isfinite(self.rgb_depth_supported_at) else None)),
                      selected_target=selected, selected_target_source=selected_source)
        opening_status = dict(t=self.now(), candidates=self.opening_candidates,
            detector=self.opening_detector.diagnostics, tracker=self.opening_tracker.status(self.now()),
            centering=self.mission.centering_status, error=self.opening_error,
            clearance=self.clearance_history.status(self.now()),
            slab_replay_remaining=len(self.slab_replay_queue), slab_replay_processed=self.slab_replay_processed)
        status['opening'] = opening_status
        status['motion_corridor'] = self.corridor_status
        status['launch_volume'] = self.launch_volume.status(self.now())
        self.opening_pub.publish(String(data=json.dumps(opening_status)))
        self.status_pub.publish(String(data=json.dumps(status)))
        self.candidates_pub.publish(String(data=json.dumps(dict(t=self.now(), detections=self.latest_candidates,
                    rgb_detections=self.rgb_candidates, rgb=self.rgb_tracker.status(), **self.tracker.status()))))
        g = OccupancyGrid()
        g.header.stamp = self.get_clock().now().to_msg()
        g.header.frame_id = self.frame
        g.info.resolution = self.grid.p.resolution_m
        g.info.width = g.info.height = self.grid.n
        g.info.origin.position.x, g.info.origin.position.y = map(float, self.grid.origin)
        g.info.origin.orientation.w = 1.
        g.data = self.grid.collapsed().ravel().tolist()
        self.grid_pub.publish(g)
        traversability = OccupancyGrid()
        traversability.header, traversability.info = g.header, g.info
        traversability.data = np.where(self.grid.navigation()[0], 0,
                                       np.where(self.grid.collapsed() == -1, -1, 100)).ravel().tolist()
        self.safe_grid_pub.publish(traversability)
        path = Path()
        path.header = g.header
        for xy in self.mission.route:
            p = PoseStamped()
            p.header = g.header
            p.pose.position.x, p.pose.position.y = map(float, xy)
            p.pose.position.z = self.mission.hold_z or 0.
            p.pose.orientation.w = 1.
            path.poses.append(p)
        self.path_pub.publish(path)


def main(args=None):
    rclpy.init(args=args)
    node = SearchNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop publishing: interface/PX4 authority and timeout policy own recovery.
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
