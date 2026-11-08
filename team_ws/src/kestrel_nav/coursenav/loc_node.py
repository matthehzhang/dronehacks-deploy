#!/usr/bin/env python3
"""coursenav localizer (ROS 2): VIO odometry + forward depth scan -> course pose.

SHADOW MODE: publishes only /coursenav/* topics. Nothing here publishes to PX4 (/fmu/in/*).
Feeding /coursenav/odometry to the vision bridge is a separate, reviewed step.

In:  /visual_slam/tracking/odometry  nav_msgs/Odometry   (motion: map<-odom correction model)
     /visual_slam/status            VisualSlamStatus      (optional: vo_state 1 = visually tracking)
     depth_topic + CameraInfo       sensor_msgs/Image      (scan)   TF base_link <- camera frame
     /fmu/out/vehicle_attitude      px4_msgs               (levelling)
     /fmu/out/distance_sensor       px4_msgs               (height above floor for the scan band)

Sensors: only the airframe's own - OAK stereo/IMU (through cuVSLAM), OAK depth, the downward
VL53L1X (through PX4) and PX4 attitude. No extra range sensors.
Out: /coursenav/pose      geometry_msgs/PoseWithCovarianceStamped, frame "course" (ENU)
     /coursenav/odometry  nav_msgs/Odometry, frame "course": VIO corrected into the course frame
     /coursenav/status    std_msgs/String JSON (2 Hz)
     /coursenav/occupancy nav_msgs/OccupancyGrid, frame "course" (1 Hz, latched): the LIVE map of this
                          run (coursenav.occgrid), -1 unknown, 0..100 occupancy probability
     logs/coursenav_<stamp>.csv (one row per scan update)
     logs/occ_<stamp>.npz      the live map at shutdown: a RUN RECORD. Rule 6.4: never loaded by a later run.

Live map params: live_map (true) fills it; use_landmarks (false) also lets the PF match its landmarks.

Run:  python3 -m coursenav.loc_node --ros-args -p use_sim_time:=true
"""
from __future__ import annotations

import bisect
import csv
import json
import os
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from refnav.depth import decode_depth, quat_to_mat

from .course import PROJECT, load_course
from .localgrid import LocalGrid, LocalParams
from .frames import compose
from .scan import depth_points_level, level_from_px4_q, scan_from_points

PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)
RANGEFINDER_BELOW_ORIGIN_M = 0.027      # Sky Beta VL53L1X mount z (airframe file); override by parameter


def stamp_s(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def yaw_of_xyzw(x, y, z, w):
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class LocNode(Node):
    def __init__(self):
        super().__init__("coursenav_loc")
        dp = self.declare_parameter
        doc = load_course(dp("course_file", os.environ.get("KESTREL_COURSE_FILE", str(PROJECT / "course" / "course_v3.json"))).value)
        self.depth_only = True
        st = doc['start']
        sxy = st.get('spawn_xy',st['xy'])
        self.prior = (float(dp('init_x',float(sxy[0])).value),float(dp('init_y',float(sxy[1])).value),
                      math.radians(float(dp('init_yaw_deg',float(st['yaw_deg'])).value)))
        self.base_frame = dp("base_frame", "base_link").value
        self.rmax = float(dp("scan_rmax", 4.0).value)
        self.cov_floor_xy = float(dp("cov_floor_xy_m", 0.05).value)
        self.cov_floor_yaw = math.radians(float(dp("cov_floor_yaw_deg", 1.0).value))
        self.rf_offset = float(dp("rangefinder_below_origin_m", RANGEFINDER_BELOW_ORIGIN_M).value)
        self.snap_every = float(dp("snapshot_every_s", 0.0).value)   # > 0: save scan snapshots (tools/scan_snapshot.py)
        self.snap_t = None
        self.live = None
        from tf2_ros import Buffer, TransformListener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.K = None
        self.cam_cal = None                    # (frame, R_body_cam, t_body_cam)
        self.att = None                        # PX4 attitude (w,x,y,z)
        self.att_received = -math.inf
        self.local = LocalGrid(LocalParams())
        self.local_stamp = None
        self.local_height = None
        self.band_half_m = float(dp('band_half_m', 0.25).value)   # airframe vertical sweep + margin
        self.band_reset_m = float(dp('band_reset_m', 0.15).value)
        self._h_off, self._h_t = None, 0.
        self.local_error = 'waiting for fresh calibrated depth and odometry'
        self.height = None                     # (stamp_wall, metres of the body origin above the floor)
        self.vio_ok = True
        self.odom_hist = deque(maxlen=200)     # (t, (x, y, yaw), z, msg)
        self.last_odom = None
        self.resets = 0
        self.counts = {"scans": 0, "updates": 0, "empty_scans": 0, "no_odom_match": 0, "beams_depth": 0}
        self.last_est = None
        self.last_update_t = None
        self.T_mean = None                     # (tx, ty, theta) course <- odom, from the latest estimate

        self.pub_pose = self.create_publisher(PoseWithCovarianceStamped, "/coursenav/pose", 10)
        self.pub_odom = self.create_publisher(Odometry, "/coursenav/odometry", 10)
        self.pub_status = self.create_publisher(String, "/coursenav/status", 10)
        # Correction for a PX4-frame mission (refnav correction_topic): PX4 local NED is the VIO frame
        # placed at the declared start fixture with the declared heading (ev_bridge heading reference).
        from geometry_msgs.msg import Vector3Stamped
        self._V3 = Vector3Stamped
        self.nominal = np.array([float(dp("nominal_x", float(sxy[0])).value),
                                 float(dp("nominal_y", float(sxy[1])).value),
                                 math.radians(float(dp("nominal_yaw_deg", float(st["yaw_deg"])).value))])
        self.pub_corr = self.create_publisher(Vector3Stamped, dp("correction_topic", "/coursenav/px4_correction").value, 10)
        self.min_updates_for_corr = int(dp("min_updates_for_correction", 30).value)
        self.publish_zero_drift = bool(dp("publish_zero_drift", False).value)   # skybeta launch: true
        self.adaptive_hz = float(dp("adaptive_hz", 0.0).value)                   # skybeta launch: 4; 0 = every frame
        self._t_proc, self._cost = 0.0, []
        # the odometry -> course correction as an EKF state (coursenav.drift): published for the navigator
        # (coursenav.nav_node), which converts its course-frame paths into PX4's frame with it every tick
        self.pub_drift = self.create_publisher(String, dp("drift_topic", "/coursenav/drift").value, 10)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_occ = self.create_publisher(OccupancyGrid, "/coursenav/occupancy", latched)
        self.pub_local = self.create_publisher(OccupancyGrid, '/coursenav/local_occupancy', 10)
        self.create_subscription(Odometry, dp("odom_topic", "/visual_slam/tracking/odometry").value,
                                 self._on_odom, qos_profile_sensor_data)
        try:
            from isaac_ros_visual_slam_interfaces.msg import VisualSlamStatus
            self.create_subscription(VisualSlamStatus, "/visual_slam/status",
                                     lambda m: setattr(self, "vio_ok", m.vo_state == 1), 10)
        except ImportError:
            pass
        depth_topic = dp("depth_topic", "/camera/depth/image_raw").value
        self.create_subscription(CameraInfo, dp("depth_info_topic", "/camera/depth/camera_info").value,
                                 self._on_info, qos_profile_sensor_data)
        self.create_subscription(Image, depth_topic, self._on_depth, qos_profile_sensor_data)
        from px4_msgs.msg import DistanceSensor, VehicleAttitude
        self.create_subscription(VehicleAttitude, "/fmu/out/vehicle_attitude",
                                 self._on_attitude, PX4_QOS)
        self.create_subscription(DistanceSensor, "/fmu/out/distance_sensor", self._on_rangefinder, PX4_QOS)
        # kestral_nav934am (skybeta): the drone publishes no distance_sensor; its rangefinder reaches ROS only
        # as vehicle_local_position.dist_bottom. height_source "local_position" uses that (default unchanged).
        self.height_source = str(dp("height_source", "rangefinder").value)
        self.landed_z = float(dp("landed_z_m", 0.10).value)
        if self.height_source == "local_position":
            from px4_msgs.msg import VehicleLocalPosition
            topic = "/fmu/out/vehicle_local_position_v1"
            try:
                from skybeta_flight_interface.topics import resolve_endpoints
                topic = resolve_endpoints()["vehicle_local_position"]
            except Exception:                               # noqa: BLE001
                pass
            self.create_subscription(VehicleLocalPosition, topic, self._on_lpos_height, PX4_QOS)
            self.get_logger().info(f"height from {topic} dist_bottom (fallback landed_z_m - z)")

        logdir = Path(dp("log_dir", str(PROJECT / "logs")).value)
        logdir.mkdir(parents=True, exist_ok=True)
        self.log_path = logdir / f"coursenav_{time.strftime('%Y%m%dT%H%M%S')}.csv"
        self.snap_dir = logdir / f"scan_snaps_{time.strftime('%Y%m%dT%H%M%S')}"
        self.occ_path = logdir / f"occ_{time.strftime('%Y%m%dT%H%M%S')}.npz"
        self.log_f = open(self.log_path, "w", newline="")
        self.log = csv.writer(self.log_f)
        self.log.writerow(["t", "x", "y", "yaw", "sx", "sy", "syaw", "vio_x", "vio_y", "vio_yaw", "z_height",
                           "beams_depth", "ess", "vio_ok"])
        self.create_timer(0.5, self._status)
        self.create_timer(0.25, self._publish_local)
        self.get_logger().info(f"VIO-depth mapper: launch fixture {np.round(self.prior, 3)}; "
                               f"depth {depth_topic}; log {self.log_path}")

    # ------------------------------------------------------------------ inputs
    def _on_attitude(self, m):
        q = np.asarray(m.q,float)
        if q.shape == (4,) and np.isfinite(q).all() and abs(np.linalg.norm(q)-1.) < .05:
            self.att = tuple(q)
            self.att_received = self.get_clock().now().nanoseconds*1e-9

    def _on_info(self, m):
        if m.k[0] > 0:
            self.K = (m.k[0], m.k[4], m.k[2], m.k[5])

    def _tf(self, frame):
        from rclpy.time import Time
        tf = self.tf_buffer.lookup_transform(self.base_frame, frame, Time())
        r, t = tf.transform.rotation, tf.transform.translation
        return quat_to_mat(r.w, r.x, r.y, r.z), np.array([t.x, t.y, t.z])

    def _flight_height(self, z_vio):
        """Body height above the floor: VIO z anchored to the rangefinder height and corrected
        slowly (tau 10 s). The rangefinder alone reads a sill or object below as the floor rising."""
        now = time.monotonic()
        rf = self.height[1] if self.height is not None and 0 <= now - self.height[0] <= .5 else None
        if rf is not None:
            if self._h_off is None:
                self._h_off = rf - z_vio
            else:
                dt = max(0., now - self._h_t)
                self._h_off += min(1., dt/10.)*((rf - z_vio) - self._h_off)
            self._h_t = now
        if self._h_off is None:
            return rf if rf is not None else .8
        return z_vio + self._h_off

    def _on_lpos_height(self, m):
        """dist_bottom while plausible (0.02-3 m; PX4 1.16 with range as the height reference may leave
        dist_bottom_valid false although it carries the range), else landed_z_m minus NED z."""
        d = float(m.dist_bottom)
        if math.isfinite(d) and 0.02 < d < 3.0:
            h = d                    # PX4's height of the body above the ground (sim: 0.0999 landed = landed_z)
        elif m.z_valid and math.isfinite(m.z):
            h = self.landed_z - float(m.z)
        else:
            return
        self.height = (time.monotonic(), h)

    def _on_rangefinder(self, m):
        if m.current_distance <= 0 or m.current_distance > m.max_distance:
            return
        tilt = 0.0
        if self.att is not None:
            R = level_from_px4_q(self.att)
            tilt = math.acos(max(-1.0, min(1.0, R[2, 2])))
        self.height = (time.monotonic(), m.current_distance * math.cos(tilt) + self.rf_offset)

    def _on_odom(self, m):
        t = stamp_s(m.header)
        p, q = m.pose.pose.position, m.pose.pose.orientation
        pose = np.array([p.x, p.y, yaw_of_xyzw(q.x, q.y, q.z, q.w)])
        if self.last_odom is not None and math.hypot(*(pose[:2]-self.last_odom[1][:2])) > .5:
            self.resets += 1
            self.local = LocalGrid(self.local.p)
            self.local_stamp = None
            self.local_height = None
            self.local_error = 'VIO reset: observations withdrawn'
        self.last_odom = (t, pose)
        self.odom_hist.append((t, pose, p.z, m))

    def _odom_at(self, t):
        if not self.odom_hist:
            return None
        ts = [h[0] for h in self.odom_hist]
        i = bisect.bisect_left(ts, t)
        cands = [self.odom_hist[k] for k in (i - 1, i) if 0 <= k < len(ts)]
        best = min(cands, key=lambda h: abs(h[0] - t))
        return best if abs(best[0] - t) <= 0.1 else None

    def _scan_rate_now(self):
        if self.get_parameter('use_sim_time').value:
            return self.get_clock().now().nanoseconds * 1e-9
        return time.monotonic()

    def _scan_rate_limited(self):
        elapsed = self._scan_rate_now() - self._t_proc
        return 0 <= elapsed < 1.0 / self.adaptive_hz

    def _on_depth(self, m):
        if self.K is None or self.att is None or self.last_odom is None:
            return
        if self.adaptive_hz > 0 and self._scan_rate_limited():
            return
        if self.cam_cal is None:
            try:
                self.cam_cal = (m.header.frame_id, *self._tf(m.header.frame_id))
            except Exception:
                return
        t = stamp_s(m.header)
        od = self._odom_at(t)
        if od is None:
            self.counts["no_odom_match"] += 1
            return
        try:
            d = decode_depth(m.encoding, m.height, m.width, m.step, bytes(m.data))
        except ValueError:
            return
        self.counts["scans"] += 1
        R_level = level_from_px4_q(self.att)
        h = self._flight_height(od[2])
        pts = depth_points_level(d, self.K, self.cam_cal[1], self.cam_cal[2], R_level)
        # The prior matcher is diagnostic infrastructure only in this mode.
        # Neither its fit nor its correction may influence local geometry.
        pose = od[1]
        if float(np.mean(np.isfinite(d) & (d > 0))) >= .3:
            if self._integrate_local(t,pose,pts,h):
                self._publish_local()
        else:
            self.local_error = 'depth valid fraction below .30; local map not refreshed'
        nominal_body = compose(self.nominal,pose)
        self.pub_drift.publish(String(data=json.dumps(dict(stamp=t,delta=[0.,0.,0.],
            nominal=nominal_body.tolist(),fixture=list(map(float,self.nominal)),height_m=float(h),
            sigma_xy=None,accepted=0,rejected=0,resets=self.resets,uncorrected=True,
            source='vio_only_no_prior_matching'))))
        self._t_proc = self._scan_rate_now()
        return

    def _integrate_local(self, captured, pose, points, height):
        now = self.get_clock().now().nanoseconds*1e-9
        if (not -.05 <= now-captured <= .5 or not 0 <= now-self.att_received <= .5
                or self.height is None or not 0 <= time.monotonic()-self.height[0] <= .5
                or not self.vio_ok or not np.isfinite(pose).all()
                or self.local_stamp is not None and captured <= self.local_stamp):
            self.local_error = 'local map requires fresh depth, attitude, height and healthy capture-matched VIO'
            return False
        # The band follows the aircraft (vertical centring changes height on purpose); walls
        # are vertical, so the map is kept. Surfaces outside the band are handled by the
        # vertical centring, not as 2-D walls. The floor is never in the band.
        if self.local_height is None or abs(height-self.local_height) > self.band_reset_m:
            # The flight band moved on purpose (centring/probing): obstacles mapped in the old band
            # (e.g. a window sill) are not obstacles in this one. Height is sill-proof, so this
            # never fires on a rangefinder step.
            self.local = LocalGrid(self.local.p)
            self.local_height = height
        scan = scan_from_points(points,height,max(.15,height-self.band_half_m),height+self.band_half_m,rmin=.3,rmax=12.)
        self.local.integrate(captured,compose(self.nominal,pose),scan)
        self.local_stamp = captured
        self.local_error = ''
        return True

    def _publish_local(self):
        if self.local.origin is None or self.local_stamp is None:
            return
        self.local.decay(self.get_clock().now().nanoseconds*1e-9)
        origin,res,w,h,data = self.local.grid_fields()
        msg = OccupancyGrid()
        # Re-publication must never make old sensor evidence appear fresh.
        from rclpy.time import Time
        msg.header.stamp = Time(seconds=self.local_stamp).to_msg()
        msg.header.frame_id = 'coursenav_nominal'
        msg.info.resolution = float(res)
        msg.info.width,msg.info.height = w,h
        msg.info.origin.position.x,msg.info.origin.position.y = map(float,origin)
        msg.info.origin.orientation.w = 1.
        msg.data = data.tolist()
        self.pub_local.publish(msg)

    def _status(self):
        self.pub_status.publish(String(data=json.dumps(dict(self.counts,resets=self.resets,
            vio_ok=self.vio_ok,ready=self.last_odom is not None,source='vio_depth_only',
            local_map=dict(self.local.stats(),frame='coursenav_nominal',last_capture=self.local_stamp,error=self.local_error)))))
        self.log_f.flush()

    def destroy_node(self):
        self.log_f.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LocNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
