#!/usr/bin/env python3
"""Fly the DroneHacks course on skybeta with livenav: map-primed A* replanning on a live depth grid,
a velocity carrot follower, and ring detection for the pad.

The skybeta contract (competitor guide section 1), nothing else:

    subscribe  /visual_slam/tracking/odometry     where we are (ENU, cuVSLAM ``odom`` frame)
    subscribe  /skybeta/state                     JSON: offboard_active, landed, clamped, ...
    publish    /skybeta/setpoint/twist            ENU world velocity + yaw rate (expires after 1 s)
    preflight  zero twist after input qualification; never stores a fallback pose

Sensors read (all the airframe's own): the OAK depth image + camera_info (depth_topic, enabled by
the team in the camera config), the OAK left mono image for the pad rings, PX4 attitude for
levelling the scan, the downward VL53L1X through PX4 for the height of the scan band.

Nothing here publishes to /fmu/in/*, arms, or changes mode. The pilot arms, then switches to
Position; the flight interface enters Offboard once, and this node takes off (CLIMB) by commanding
an upward twist. On the drone the interface floors altitude at min_altitude_m and has no land
command: DESCEND goes to that floor over the pad and holds. The pilot lands.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from isaac_ros_visual_slam_interfaces.msg import VisualSlamStatus
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from .coursenav.depth import DepthFrame, decode_depth, quat_to_mat
from .coursenav.scan import depth_points_level, level_from_px4_q
from .livenav.follower import FollowParams
from .livenav.mission import LiveMission, MissionParams, Observation
from .livenav.navgrid import NavParams
from .livenav.pad import PadDetector, level_cam_rotation
from .livenav.replanner import ReplanParams
from .livenav.safety import InputSafety, SafetyParams, corridor_error

PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)
R_FLU_OPT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def yaw_of_xyzw(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rpy_from_body_mount(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
    Ry = np.array([[cp, 0, sp], [0, 1.0, 0], [-sp, 0, cp]])
    Rx = np.array([[1.0, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


class LiveMissionNode(Node):
    def __init__(self, **kwargs):
        super().__init__('live_mission', **kwargs)
        dp = self.declare_parameter
        defaults = SafetyParams()
        self.safety = InputSafety(SafetyParams(**{
            name: type(value)(dp(name, value).value)
            for name, value in vars(defaults).items()
        }))
        # --- mission ---
        mp = MissionParams(
            use_pf=bool(dp('use_pf', True).value),
            pf_sigma_xy_m=float(dp('pf_sigma_xy_m', 0.10).value),
            pf_sigma_yaw_deg=float(dp('pf_sigma_yaw_deg', 4.0).value),
            scan_rmax_m=float(dp('scan_rmax_m', 4.0).value),
            cruise_z_m=float(dp('cruise_z_m', 0.8).value),
            landed_z_m=float(dp('landed_z_m', 0.10).value),
            descend_to_z_odom=float(dp('descend_to_z_m', 0.3).value),
            time_limit_s=float(dp('time_limit_s', 170.0).value),
            chicane=str(dp('chicane', 'auto').value),
            depth_max_age_s=self.safety.p.depth_max_age_s,
        )
        sx, sy, syaw = (float(dp('start_x', float('nan')).value), float(dp('start_y', float('nan')).value),
                        float(dp('start_yaw_deg', float('nan')).value))
        if not (math.isnan(sx) or math.isnan(sy)):
            mp.start_xy = (sx, sy)
        if not math.isnan(syaw):
            mp.start_yaw_deg = syaw
        fp = FollowParams(v_cruise=float(dp('v_cruise', 0.4).value), v_gate=float(dp('v_gate', 0.25).value),
                          stop_m=float(dp('stop_m', 0.55).value))
        course_file = str(dp('course_file', '').value) or None
        self.mission = LiveMission(mp, NavParams(), ReplanParams(), fp, course_file=course_file)
        # --- sensors ---
        self.depth_topic = str(dp('depth_topic', '/kestrel_nav/depth').value)
        self.depth_info_topic = str(dp('depth_info_topic', '/kestrel_nav/depth/camera_info').value)
        self.mono_topic = str(dp('mono_topic', '/oak/left/image_mono').value)
        self.mono_info_topic = str(dp('mono_info_topic', '/oak/left/camera_info').value)
        self.base_frame = str(dp('base_frame', 'base_link').value)
        self.use_tf = bool(dp('camera_from_tf', True).value)
        cam = [float(dp(k, v).value) for k, v in (('camera_x', 0.116), ('camera_y', 0.037), ('camera_z', 0.029),
                                                   ('camera_roll', 0.0), ('camera_pitch', 0.0), ('camera_yaw', 0.0))]
        self.cam_t_param = np.array(cam[:3])
        self.cam_R_param = rpy_from_body_mount(*cam[3:]) @ R_FLU_OPT     # body FLU <- optical
        self.rf_offset = float(dp('rangefinder_below_origin_m', 0.027).value)
        self.pad_hz = float(dp('pad_detect_hz', 4.0).value)
        rate_hz = float(dp('publish_rate_hz', 10.0).value)
        self.min_alt = float(dp('min_altitude_m', 0.3).value)

        # --- the contract ---
        self.twist_pub = self.create_publisher(TwistStamped, '/skybeta/setpoint/twist', 10)
        self.create_subscription(Odometry, '/visual_slam/tracking/odometry', self.odom_cb, 10)
        self.create_subscription(VisualSlamStatus, '/visual_slam/status', self.tracking_cb, 10)
        self.create_subscription(String, '/skybeta/state', self.state_cb, 10)
        # --- sensors in ---
        self.create_subscription(CameraInfo, self.depth_info_topic, self.depth_info_cb, qos_profile_sensor_data)
        self.create_subscription(Image, self.depth_topic, self.depth_cb, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, self.mono_info_topic, self.mono_info_cb, qos_profile_sensor_data)
        self.create_subscription(Image, self.mono_topic, self.mono_cb, qos_profile_sensor_data)
        try:
            from px4_msgs.msg import DistanceSensor, VehicleAttitude, VehicleLocalPosition
            self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.att_cb, PX4_QOS)
            # skybeta-14 (2026-11-07): the agent bridges no /fmu/out/distance_sensor; the rangefinder
            # reaches us as EKF2's dist_bottom in vehicle_local_position, spelled WITHOUT _v1 on this
            # firmware although vehicle_status carries _v1. Subscribe to both spellings.
            self.create_subscription(DistanceSensor, '/fmu/out/distance_sensor', self.rf_cb, PX4_QOS)
            for name in ('/fmu/out/vehicle_local_position', '/fmu/out/vehicle_local_position_v1'):
                self.create_subscription(VehicleLocalPosition, name, self.lpos_cb, PX4_QOS)
        except ImportError:
            self.get_logger().warning('px4_msgs not importable: scan levelling uses a level attitude')
        # --- diagnostics out ---
        self.status_pub = self.create_publisher(String, '/kestrel_nav/live_status', 5)
        self.path_pub = self.create_publisher(Path, '/kestrel_nav/path', 5)
        # --- TF (camera extrinsics) ---
        self.tf_buffer = None
        if self.use_tf:
            try:
                from tf2_ros import Buffer, TransformListener
                self.tf_buffer = Buffer()
                self.tf_listener = TransformListener(self.tf_buffer, self)
            except ImportError:
                self.get_logger().warning('tf2_ros not importable: camera pose from parameters')

        # --- state ---
        self.position = None
        self.yaw = None
        self.offboard_active = False
        self.armed = False
        self.px4_link = False
        self.position_valid = False
        self.interface_frozen = False
        self.odom_frame = None
        self.landed = None
        self.clamped = []
        self.K_depth = None
        self.K_mono = None
        self.mono_hw = None
        self.cam_cal = {}             # frame_id -> (R_body_cam, t_body_cam)
        self.att = None
        self.height = None
        self.pts = None
        self.pts_h = 0.8
        self.pts_t = -math.inf
        self.depth_frame = None
        self.pad_fix = None
        self.pad_fix_t = -math.inf
        self.pad_detector = None
        self.last_pad_run = -math.inf
        self.counts = dict(depth=0, depth_bad=0, mono=0, pad=0)
        self.log_once = set()
        self.create_timer(1.0 / rate_hz, self.tick)
        self.create_timer(0.5, self.publish_status)
        plan = self.mission.plan
        self.get_logger().info(
            f"livenav ready: chicane {plan.get('chicane')} ({plan.get('reason', '')[:60]}), {len(self.mission.goals)} goals, "
            f"{plan['length_m']} m; cruise {mp.cruise_z_m} m above the floor = {self.mission.z_odom_cruise:.2f} m above odom; "
            f"v {fp.v_cruise}/{fp.v_gate} m/s; depth from {self.depth_topic}. Waiting for Offboard.")

    # ------------------------------------------------------------------ inputs
    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def monotonic_s(self):
        return time.monotonic()

    @staticmethod
    def stamp_s(m):
        return float(m.header.stamp.sec) + float(m.header.stamp.nanosec) * 1e-9

    def odom_cb(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        covariance = np.asarray(m.pose.covariance, float)
        norm = math.hypot(q.x, q.y, q.z, q.w)
        frame = (m.header.frame_id, m.child_frame_id)
        if (not all(math.isfinite(v) for v in values) or not 0.9 <= norm <= 1.1
                or covariance.size != 36 or not np.isfinite(covariance).all()
                or np.any(covariance.reshape(6, 6).diagonal() < 0)):
            self.safety.reject('odom', 'invalid odometry pose or covariance')
            return
        if not frame[0] or (self.odom_frame is not None and frame != self.odom_frame):
            self.safety.reject('odom', 'odometry frame changed or missing; restart on ground')
            return
        if not self.safety.accept('odom', self.stamp_s(m), self.now_s(), self.monotonic_s()):
            return
        self.odom_frame = frame
        self.position = (float(p.x), float(p.y), float(p.z))
        self.yaw = yaw_of_xyzw(q.x / norm, q.y / norm, q.z / norm, q.w / norm)

    def tracking_cb(self, m):
        if m.vo_state != 1:
            self.safety.reject('tracking', f'cuVSLAM tracking failed (vo_state={m.vo_state})')
            return
        self.safety.accept('tracking', self.stamp_s(m), self.now_s(), self.monotonic_s())

    def state_cb(self, m):
        try:
            st = json.loads(m.data)
            if not isinstance(st, dict) or any(type(st.get(k)) is not bool
                    for k in ('offboard_active', 'armed', 'px4_link', 'frozen')):
                raise ValueError('missing authority fields')
            stamp = float(st['stamp'])
        except (ValueError, TypeError, KeyError):
            self.safety.reject('state', 'invalid flight-interface state')
            return
        if not self.safety.accept('state', stamp, self.now_s(), self.monotonic_s()):
            return
        self.offboard_active = bool(st.get('offboard_active', False))
        self.armed = st['armed']
        self.px4_link = st['px4_link']
        self.interface_frozen = st['frozen']
        local = st.get('local_enu')
        self.position_valid = (isinstance(local, list) and len(local) == 3
                               and all(isinstance(v, (float, int)) and math.isfinite(v) for v in local))
        self.landed = st.get('landed')
        self.clamped = st.get('clamped') or []
        if self.clamped:
            self.info_once(f"clamped:{','.join(self.clamped)}",
                           f"flight interface is CLAMPING our setpoints ({', '.join(self.clamped)}); envelope {st.get('envelope')}")

    def depth_info_cb(self, m):
        if all(math.isfinite(v) for v in m.k) and m.k[0] > 0 and m.k[4] > 0:
            self.K_depth = (m.k[0], m.k[4], m.k[2], m.k[5])
        else:
            self.K_depth = None
            self.safety.reject('depth', 'invalid depth camera calibration')
            self.clear_depth()

    def mono_info_cb(self, m):
        if m.k[0] > 0 and self.K_mono is None:
            self.K_mono = (m.k[0], m.k[4], m.k[2], m.k[5])
            self.mono_hw = (m.height, m.width)
            self.pad_detector = PadDetector(self.K_mono, self.mono_hw)

    def att_cb(self, m):
        self.att = tuple(float(v) for v in m.q)

    def lpos_cb(self, m):
        """EKF2 height above ground (rangefinder-aided): the body origin's height for the scan band."""
        if m.dist_bottom_valid and 0.0 < m.dist_bottom < 5.0:
            self.height = (self.now_s(), float(m.dist_bottom))

    def rf_cb(self, m):
        if 0 < m.current_distance <= m.max_distance:
            tilt = 0.0
            if self.att is not None:
                R = level_from_px4_q(self.att)
                tilt = math.acos(max(-1.0, min(1.0, R[2, 2])))
            self.height = (self.now_s(), m.current_distance * math.cos(tilt) + self.rf_offset)

    def camera_pose(self, frame_id):
        """body FLU <- camera optical rotation and translation, from TF or the parameters."""
        if frame_id in self.cam_cal:
            return self.cam_cal[frame_id]
        if self.tf_buffer is not None:
            try:
                from rclpy.time import Time
                tf = self.tf_buffer.lookup_transform(self.base_frame, frame_id, Time())
                r, t = tf.transform.rotation, tf.transform.translation
                cal = (quat_to_mat(r.w, r.x, r.y, r.z), np.array([t.x, t.y, t.z]))
                self.cam_cal[frame_id] = cal
                self.get_logger().info(f'camera {frame_id} from TF: t={cal[1].round(3).tolist()}')
                return cal
            except Exception:
                pass
        self.info_once(f'tf:{frame_id}', f'no TF {self.base_frame} <- {frame_id}: using camera_* parameters')
        self.cam_cal[frame_id] = (self.cam_R_param, self.cam_t_param)
        return self.cam_cal[frame_id]

    def body_height(self):
        if self.height is not None and self.now_s() - self.height[0] < 1.0:
            return self.height[1]
        if self.position is not None:
            return self.position[2] + self.mission.p.landed_z_m
        return self.mission.p.cruise_z_m

    def depth_cb(self, m):
        if self.K_depth is None:
            self.info_once('no_depth_info', f'no CameraInfo on {self.depth_info_topic} yet')
            return
        try:
            d = decode_depth(m.encoding, m.height, m.width, m.step, bytes(m.data))
        except ValueError as e:
            self.counts['depth_bad'] += 1
            self.safety.reject('depth', f'invalid depth image: {e}')
            self.clear_depth()
            self.info_once('depth_enc', str(e))
            return
        R_body_cam, t_body_cam = self.camera_pose(m.header.frame_id)
        R_level = level_from_px4_q(self.att) if self.att is not None else np.eye(3)
        frame = DepthFrame(d, self.K_depth, R_body_cam, t_body_cam, R_level, stride=4, border_px=16)
        if frame.valid_frac < self.safety.p.min_depth_valid_fraction:
            self.safety.reject('depth', f'insufficient valid depth ({frame.valid_frac:.2f})')
            self.clear_depth()
            return
        if not self.safety.accept('depth', self.stamp_s(m), self.now_s(), self.monotonic_s()):
            return
        self.depth_frame = frame
        self.pts = depth_points_level(d, self.K_depth, R_body_cam, t_body_cam, R_level, stride=4, border_px=16)
        self.pts_h = self.body_height()
        self.pts_t = self.stamp_s(m)
        self.counts['depth'] += 1

    def mono_cb(self, m):
        self.counts['mono'] += 1
        if self.pad_detector is None or self.mission.state not in ('FLY', 'PAD', 'DESCEND'):
            return
        rp = self.mission.replanner
        if rp.k < len(self.mission.goals) - 2:
            return                                                  # only near the end of the course
        now = self.now_s()
        if now - self.last_pad_run < 1.0 / self.pad_hz:
            return
        self.last_pad_run = now
        try:
            if m.encoding in ('bgr8', 'rgb8'):
                img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step // 3, 3)[:, :m.width]
            elif m.encoding in ('mono8', '8UC1'):
                img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width]
            else:
                self.info_once('mono_enc', f'unsupported mono encoding {m.encoding}')
                return
        except ValueError:
            return
        R_body_cam, t_body_cam = self.camera_pose(m.header.frame_id)
        roll = pitch = 0.0
        if self.att is not None:
            R = level_from_px4_q(self.att)
            pitch, roll = -math.asin(max(-1.0, min(1.0, R[2, 0]))), math.atan2(R[2, 1], R[2, 2])
        R_level_cam = level_cam_rotation(R_body_cam, roll, pitch)
        cam_h = self.body_height() + float(t_body_cam[2])
        det = self.pad_detector.detect(np.ascontiguousarray(img), cam_h, R_level_cam)
        if det is not None:
            # detector output is relative to the camera's ground point; shift to the body origin
            self.pad_fix = (det[0] + float(t_body_cam[0]), det[1] + float(t_body_cam[1]))
            self.pad_fix_t = now
            self.counts['pad'] += 1

    def info_once(self, key, message):
        if key not in self.log_once:
            self.log_once.add(key)
            self.get_logger().info(message)

    # ------------------------------------------------------------------ outputs
    def send_twist(self, vx, vy, vz, yaw_rate):
        t = TwistStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.twist.linear.x, t.twist.linear.y, t.twist.linear.z = float(vx), float(vy), float(vz)
        t.twist.angular.z = float(yaw_rate)
        self.twist_pub.publish(t)

    def clear_depth(self):
        self.pts = self.depth_frame = None
        self.mission.scan_nav = self.mission.scan_pf = None

    def stop_mission(self, now, reason):
        self.safety.trip(reason)
        if self.safety.action == 'run':
            self.safety.action = 'stop'
        self.clear_depth()
        self.mission.abort(now, self.safety.fault)
        self.info_once(f'fault:{self.safety.fault}',
                       f'MISSION FAULT: {self.safety.fault}. Pilot takeover required; ground restart to requalify.')

    def publish_path(self, path_course):
        if path_course is None or self.mission.T is None:
            return
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        tx, ty, dyaw = self.mission.T
        c, s = math.cos(dyaw), math.sin(dyaw)
        for x, y in path_course:
            p = PoseStamped()
            p.header = msg.header
            p.pose.position.x, p.pose.position.y = c * x - s * y + tx, s * x + c * y + ty
            p.pose.position.z = self.mission.z_odom_cruise
            p.pose.orientation.w = 1.0
            msg.poses.append(p)
        self.path_pub.publish(msg)

    # ------------------------------------------------------------------ 10 Hz
    def tick(self):
        now = self.now_s()
        action = self.safety.evaluate(now, self.monotonic_s(), offboard=self.offboard_active,
                                      armed=self.armed, landed=self.landed, link=self.px4_link,
                                      position_valid=self.position_valid, frozen=self.interface_frozen)
        if self.safety.stream_error('depth', now, self.monotonic_s()):
            self.clear_depth()
        if self.safety.fault:
            self.stop_mission(now, self.safety.fault)
        if action == 'silent':
            self.info_once(self.safety.reason, self.safety.reason)
            return
        if action in ('ready', 'stop'):
            # No pose target is planted in the interface, even during preflight.
            self.send_twist(0.0, 0.0, 0.0, 0.0)
            return
        obs = Observation(now=now, odom_xyz=self.position, odom_yaw=self.yaw, offboard_active=self.offboard_active,
                          pts_level=self.pts, height_m=self.pts_h, pts_t=self.pts_t,
                          pad_fix=self.pad_fix, pad_fix_t=self.pad_fix_t)
        before = self.mission.state
        try:
            cmd = self.mission.step(obs)
            if not all(math.isfinite(v) for v in (cmd.vx, cmd.vy, cmd.vz, cmd.yaw_rate)):
                raise ValueError('non-finite mission command')
        except Exception as e:
            self.stop_mission(now, f'mission computation failed: {type(e).__name__}: {e}')
            self.send_twist(0.0, 0.0, 0.0, 0.0)
            return
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        fp = self.mission.follower.p
        reason = corridor_error(self.depth_frame, c * cmd.vx + s * cmd.vy, -s * cmd.vx + c * cmd.vy,
                                half_width=fp.corridor_half_m, band=self.mission.doc.get('plan_band_m', 0.25),
                                at_m=fp.stop_m + fp.reactive_ramp_m, params=self.safety.p)
        if cmd.state == 'FAULT' or reason:
            self.stop_mission(now, reason or cmd.note)
            self.send_twist(0.0, 0.0, 0.0, 0.0)
            return
        if cmd.state != before:
            self.get_logger().info(f'{before} -> {cmd.state} {cmd.note}')
        if cmd.state == 'PAUSED':
            return
        self.send_twist(cmd.vx, cmd.vy, cmd.vz, cmd.yaw_rate)
        if cmd.path_course is not None:
            self.publish_path(cmd.path_course)

    def publish_status(self):
        st = self.mission.status()
        st.update(counts=self.counts, scan_age=round(self.now_s() - self.pts_t, 2) if self.pts_t > -math.inf else None,
                  height=self.height[1] if self.height else None, offboard=self.offboard_active, clamped=self.clamped,
                  pad_fix=self.pad_fix, safety_action=self.safety.action,
                  safety_reason=self.safety.reason, safety_fault=self.safety.fault)
        self.status_pub.publish(String(data=json.dumps(st, default=str)))


def main(args=None):
    rclpy.init(args=args)
    node = LiveMissionNode()
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
