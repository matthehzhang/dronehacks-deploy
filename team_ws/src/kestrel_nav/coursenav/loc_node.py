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
     /coursenav/local_occupancy nav_msgs/OccupancyGrid, frame "coursenav_nominal" (2 Hz): the LOCAL
                          obstacle grid (coursenav.localgrid), every scan at the raw odometry pose, never
                          gated on the localizer; the navigator adds its obstacles to the path map
     /coursenav/prior_state std_msgs/String JSON (2 Hz): does the prior map describe the view
                          (coursenav.priorstate: placed / dropped, from the scan fit)
     logs/coursenav_<stamp>.csv (one row per scan update)
     logs/occ_<stamp>.npz      the live map at shutdown: a RUN RECORD. Rule 6.4: never loaded by a later run.

Live map params: live_map (true) fills it; use_landmarks (false) also lets the PF match its landmarks.
Local grid params: local_map (true), local_size_m (8), local_decay_s (30), local_max_range_m (3),
local_band_m (0.35: obstacles within this height of the aircraft). Prior monitor: prior_drop_fit (0.30),
prior_drop_s (3), prior_restore_s (2); restore at min_fit_frac.

Run:  python3 -m coursenav.loc_node --ros-args -p use_sim_time:=true
"""
from __future__ import annotations

import bisect
import csv
import json
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

from .drift import DriftEKF
from .course import PROJECT, default_course_path, load_course, resolve, scan_field, scan_grid, see_through_grid
from .objmap import CourseMap
from .localgrid import LocalGrid, LocalParams
from .occgrid import LiveGrid, OccParams
from .priorstate import PriorMonitor, PriorParams
from .pf import CorrectionLimiter, CorrectionPF, PFParams, compose, inverse, project_correction
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
        doc = load_course(dp("course_file", str(default_course_path())).value)
        cmap = CourseMap.load(resolve(doc, "map"))
        self.field_zlo, self.field_zhi = float(dp("scan_zlo", 0.3).value), float(dp("scan_zhi", 1.6).value)
        self.field = scan_grid(cmap, self.field_zlo, self.field_zhi)
        pfp = PFParams(n=int(dp("particles", 600).value),
                       use_cluster=bool(dp("use_cluster", True).value),
                       min_beams=int(dp("min_beams", 8).value),
                       min_fit_frac=float(dp("min_fit_frac", 0.35).value))
        # beams ending on the net say nothing about the position along it (tails, A6)
        net = cmap.band(self.field_zlo, self.field_zhi, only_prefixes=("Net",))
        self.pf = CorrectionPF(self.field, scan_field(self.field, doc), pfp, uninformative=net)
        self.limiter = CorrectionLimiter(float(dp("corr_rate_per_m", 0.05).value), float(dp("corr_rate_per_s", 0.01).value))
        self.corr_sigma_dir = float(dp("corr_project_sigma_m", 0.10).value)
        self.last_corr_odom = None
        self.last_corr_t = None
        st = doc["start"]
        sxy = st.get("spawn_xy", st["xy"])           # declared spawn fixture when the course file has one
        self.prior = (float(dp("init_x", float(sxy[0])).value), float(dp("init_y", float(sxy[1])).value),
                      math.radians(float(dp("init_yaw_deg", float(st["yaw_deg"])).value)))
        self.prior_sigma = (float(dp("init_sigma_xy", float(st["sigma_xy_m"])).value),
                            math.radians(float(dp("init_sigma_yaw_deg", float(st["sigma_yaw_deg"])).value)))
        self.base_frame = dp("base_frame", "base_link").value
        self.rmax = float(dp("scan_rmax", 4.0).value)
        self.cov_floor_xy = float(dp("cov_floor_xy_m", 0.05).value)
        self.cov_floor_yaw = math.radians(float(dp("cov_floor_yaw_deg", 1.0).value))
        self.rf_offset = float(dp("rangefinder_below_origin_m", RANGEFINDER_BELOW_ORIGIN_M).value)
        self.snap_every = float(dp("snapshot_every_s", 0.0).value)   # > 0: save scan snapshots (tools/scan_snapshot.py)
        self.snap_t = None
        self.live = None
        if bool(dp("live_map", True).value):
            self.live = LiveGrid(self.field, OccParams(use_in_pf=bool(dp("use_landmarks", False).value),
                                                       gate_sigma_xy_m=float(dp("live_gate_sigma_xy_m", 0.06).value),
                                                       gate_sigma_yaw_deg=float(dp("live_gate_sigma_yaw_deg", 2.0).value)),
                                 see_through=see_through_grid(cmap, self.field_zlo, self.field_zhi))
        self.prior_dist = self.pf.dist
        self.local = None
        if bool(dp("local_map", True).value):
            self.local = LocalGrid(LocalParams(size_m=float(dp("local_size_m", 8.0).value),
                                               decay_s=float(dp("local_decay_s", 30.0).value),
                                               max_range_m=float(dp("local_max_range_m", 3.0).value)))
        self.local_band = float(dp("local_band_m", 0.35).value)
        self.prior_mon = PriorMonitor(PriorParams(drop_fit=float(dp("prior_drop_fit", 0.30).value),
                                                  drop_s=float(dp("prior_drop_s", 3.0).value),
                                                  restore_fit=float(dp("min_fit_frac", 0.35).value),
                                                  restore_s=float(dp("prior_restore_s", 2.0).value)))

        from tf2_ros import Buffer, TransformListener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.K = None
        self.cam_cal = None                    # (frame, R_body_cam, t_body_cam)
        self.att = None                        # PX4 attitude (w,x,y,z)
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
        self.ekf = DriftEKF()
        self.pub_drift = self.create_publisher(String, dp("drift_topic", "/coursenav/drift").value, 10)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_occ = self.create_publisher(OccupancyGrid, "/coursenav/occupancy", latched)
        self.pub_local = self.create_publisher(OccupancyGrid, "/coursenav/local_occupancy", 10)
        self.pub_prior = self.create_publisher(String, "/coursenav/prior_state", latched)
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
                                 lambda m: setattr(self, "att", tuple(float(c) for c in m.q)), PX4_QOS)
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
        if self.live is not None:
            self.create_timer(1.0, self._publish_occ)
        if self.local is not None:
            self.create_timer(0.5, self._publish_local)
        self.get_logger().info(f"shadow localizer: prior {np.round(self.prior, 3)} sigma {self.prior_sigma}; "
                               f"depth {depth_topic}; log {self.log_path}")

    # ------------------------------------------------------------------ inputs
    def _on_info(self, m):
        if m.k[0] > 0:
            self.K = (m.k[0], m.k[4], m.k[2], m.k[5])

    def _tf(self, frame):
        from rclpy.time import Time
        tf = self.tf_buffer.lookup_transform(self.base_frame, frame, Time())
        r, t = tf.transform.rotation, tf.transform.translation
        return quat_to_mat(r.w, r.x, r.y, r.z), np.array([t.x, t.y, t.z])

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
        if self.last_odom is not None:
            if math.hypot(*(pose[:2] - self.last_odom[1][:2])) > 0.5 and self.pf.ready:
                self.pf.reanchor(self.last_odom[1], pose)          # VIO reset: keep the body continuous
                self.resets += 1
                self.get_logger().warning("VIO jump > 0.5 m: re-anchored the correction")
        self.last_odom = (t, pose)
        self.odom_hist.append((t, pose, p.z, m))
        if not self.pf.ready:
            self.pf.init(self.prior, *self.prior_sigma, pose)
            self.get_logger().info("particle filter initialised from the course start prior")
        if self.T_mean is not None:
            self._publish_odom(m, pose)

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
        if self.K is None or self.att is None or not self.pf.ready:
            return
        # kestral_nav934am (skybeta) adaptive rate: once confident (sigma_xy < 0.05 m, sigma_yaw < 2 deg), process
        # at most adaptive_hz scans/s; full rate when confidence drops. Corrections are rate-limited anyway.
        if self.adaptive_hz > 0 and self.last_est is not None:
            cov = self.last_est[2]
            confident = math.sqrt(max(cov[0, 0], cov[1, 1])) < 0.05 and math.degrees(math.sqrt(cov[2, 2])) < 2.0
            if confident and self._scan_rate_limited():
                return
        t_cpu = time.process_time()
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
        h = self.height[1] if self.height is not None else 0.8
        pts = depth_points_level(d, self.K, self.cam_cal[1], self.cam_cal[2], R_level)
        scan = scan_from_points(pts, h, self.field_zlo, self.field_zhi, rmax=self.rmax)
        n_depth = len(scan)
        self.counts["beams_depth"] = n_depth
        pose = od[1]
        dt = 0.0 if self.last_update_t is None else max(0.0, t - self.last_update_t)
        self.pf.predict(pose, dt, self.vio_ok)
        if len(scan) == 0:
            self.counts["empty_scans"] += 1
        else:
            if self.pf.update(pose, scan):
                self.counts["updates"] += 1
            if self.pf.last_fit is not None:
                was = self.prior_mon.state
                if self.prior_mon.on_fit(t, *self.pf.last_fit) != was:
                    self.get_logger().warning(f"prior map {self.prior_mon.state}: scan fit {self.pf.last_fit[1]:.0%} "
                                              f"(drop below {self.prior_mon.p.drop_fit:.0%}, restore at "
                                              f"{self.prior_mon.p.restore_fit:.0%})")
                    self._publish_prior()
        if self.local is not None:
            # the aircraft's own height band, at the RAW odometry pose (nominal frame): never gated
            lscan = scan_from_points(pts, h, max(0.15, h - self.local_band), h + self.local_band,
                                     rmax=self.local.p.max_range_m)
            self.local.integrate(t, compose(self.nominal, pose), lscan)
        self.last_update_t = t
        mean, cov = self.pf.estimate(pose)
        self.T_mean = compose(mean, inverse(pose))
        if self.live is not None:
            self.live.integrate(mean, scan, cov)              # gated on the localizer's confidence
            if self.live.p.use_in_pf:
                f = self.live.landmark_field()
                self.pf.set_distance(self.prior_dist if f is None else f)
        self.last_est = (t, mean, cov, pose, h, n_depth)
        self._t_proc = self._scan_rate_now()
        self._cost.append(time.process_time() - t_cpu)
        if len(self._cost) >= 50:                  # measured per-frame CPU cost, for the drone's first runs
            self.get_logger().info(f"scan cost: mean {1000 * np.mean(self._cost):.1f} ms CPU/frame "
                                   f"(p90 {1000 * np.percentile(self._cost, 90):.1f} ms) over {len(self._cost)} frames")
            self._cost = []
        self._publish_pose(m.header.stamp, mean, cov, od[2])
        nominal_body = compose(self.nominal, pose)
        self.ekf.predict(nominal_body, dt)
        if self.counts["updates"] >= self.min_updates_for_corr:
            self.ekf.update(mean, cov, nominal_body)
        if self.ekf.ready:
            self.pub_drift.publish(String(data=json.dumps({
                "stamp": t, "delta": [float(v) for v in self.ekf.x], "nominal": [float(v) for v in nominal_body],
                "fixture": [float(v) for v in self.nominal], "sigma_xy": self.ekf.sigma_xy(),
                "accepted": self.ekf.accepted, "rejected": self.ekf.rejected, "resets": self.ekf.resets})))
        elif self.publish_zero_drift:
            # kestral_nav934am (skybeta): no correction yet (the EKF starts after min_updates_for_correction PF
            # updates; on the ground the fit gate can refuse every scan). Publish ZERO drift: VIO + the start
            # fixture is the estimate, as the flight already uses. The navigator plans nothing without a drift
            # message (Gazebo: no leg-1 path for > 40 s on a Jetson-sized CPU budget).
            self.pub_drift.publish(String(data=json.dumps({
                "stamp": t, "delta": [0.0, 0.0, 0.0], "nominal": [float(v) for v in nominal_body],
                "fixture": [float(v) for v in self.nominal], "sigma_xy": None,
                "accepted": 0, "rejected": 0, "resets": 0, "uncorrected": True})))
        if self.counts["updates"] >= self.min_updates_for_corr:
            nominal = compose(self.nominal, pose)                     # where VIO + fixture put the body
            raw = np.array([mean[0] - nominal[0], mean[1] - nominal[1]])   # course ENU drift of that belief
            # only along the directions this scan geometry constrains, and no faster than drift can grow
            _, kept, P = project_correction(raw, cov, self.corr_sigma_dir)
            travelled = 0.0 if self.last_corr_odom is None else math.hypot(*(pose[:2] - self.last_corr_odom[:2]))
            dt_c = 0.0 if self.last_corr_t is None else max(0.0, t - self.last_corr_t)
            self.last_corr_odom, self.last_corr_t = pose, t
            prev = self.limiter.out if self.limiter.out is not None else raw
            target = prev + P @ (raw - prev)                        # move only along the measured directions
            lim = self.limiter.step(target, travelled, dt_c)
            self.counts["corr_kept_dirs"] = kept
            c = self._V3()
            c.header.stamp = m.header.stamp
            c.header.frame_id = "px4_local_ned_target_correction"
            c.vector.x, c.vector.y = float(-lim[1]), float(-lim[0])   # NED shift to ADD to PX4-frame targets
            c.vector.z = float(max(self.cov_floor_xy, math.sqrt(max(cov[0, 0], cov[1, 1]))))
            self.pub_corr.publish(c)
        if self.snap_every > 0 and (self.snap_t is None or t - self.snap_t >= self.snap_every):
            self.snap_t = t
            self._snapshot(t, mean, cov, pose, h, scan)
        sd = np.sqrt(np.maximum(np.diag(cov), 0))
        self.log.writerow([f"{t:.4f}", *(f"{v:.4f}" for v in mean), *(f"{v:.4f}" for v in sd),
                           *(f"{v:.4f}" for v in pose), f"{h:.3f}", n_depth, f"{self.pf.ess():.0f}",
                           int(self.vio_ok)])

    def _snapshot(self, t, mean, cov, odom, h, scan):
        """Everything tools/scan_snapshot.py needs to draw this update's beams over the map."""
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.snap_dir / f"snap_{t:012.3f}.npz", t=t, mean=mean, cov=cov, odom=odom, height=h,
                            bearing=scan.bearing, range=scan.range, sigma=scan.sigma,
                            particles=self.pf.body(odom), weights=self.pf.w)

    # ------------------------------------------------------------------ outputs
    def _cov6(self, cov, z_var=0.01, rp_var=0.0004):
        c = np.zeros((6, 6))
        c[0, 0] = max(cov[0, 0], self.cov_floor_xy ** 2)
        c[1, 1] = max(cov[1, 1], self.cov_floor_xy ** 2)
        c[0, 1] = c[1, 0] = cov[0, 1]
        c[2, 2] = z_var
        c[3, 3] = c[4, 4] = rp_var
        c[5, 5] = max(cov[2, 2], self.cov_floor_yaw ** 2)
        return [float(v) for v in c.ravel()]

    def _publish_pose(self, stamp, mean, cov, z):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "course"
        msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = float(mean[0]), float(mean[1]), float(z)
        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = math.sin(mean[2] / 2), math.cos(mean[2] / 2)
        msg.pose.covariance = self._cov6(cov)
        self.pub_pose.publish(msg)

    def _publish_odom(self, m, pose):
        """The VIO sample re-expressed in the course frame through the current correction."""
        tx, ty, th = self.T_mean
        c, s = math.cos(th), math.sin(th)
        p, q = m.pose.pose.position, m.pose.pose.orientation
        out = Odometry()
        out.header.stamp = m.header.stamp
        out.header.frame_id, out.child_frame_id = "course", m.child_frame_id
        out.pose.pose.position.x = tx + c * p.x - s * p.y
        out.pose.pose.position.y = ty + s * p.x + c * p.y
        out.pose.pose.position.z = p.z
        # q_course = q_yaw(th) * q_odom   (x, y, z, w)
        hw, hz = math.cos(th / 2), math.sin(th / 2)
        out.pose.pose.orientation.w = hw * q.w - hz * q.z
        out.pose.pose.orientation.x = hw * q.x - hz * q.y
        out.pose.pose.orientation.y = hw * q.y + hz * q.x
        out.pose.pose.orientation.z = hw * q.z + hz * q.w
        cov = self.last_est[2] if self.last_est is not None else np.diag([0.09, 0.09, 0.02])
        vio_cov = np.asarray(m.pose.covariance).reshape(6, 6)
        out.pose.covariance = self._cov6(cov, z_var=max(vio_cov[2, 2], 1e-4), rp_var=max(vio_cov[3, 3], 1e-5))
        out.twist = m.twist                         # body-frame twist is frame independent
        self.pub_odom.publish(out)

    def _publish_occ(self):
        g = self.live
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "course"
        nx, ny = g.L.shape
        msg.info.resolution = float(g.prior.res)
        msg.info.width, msg.info.height = nx, ny
        msg.info.origin.position.x, msg.info.origin.position.y = (float(v) for v in g.prior.origin)
        msg.info.origin.orientation.w = 1.0
        prob = np.rint(g.probability() * 100).astype(np.int8)
        prob[g.seen == 0] = -1
        msg.data = prob.T.ravel().tolist()                    # row-major, x fastest
        self.pub_occ.publish(msg)

    def _publish_local(self):
        if self.local.origin is None:
            return
        (ox, oy), res, w, hgt, data = self.local.grid_fields()
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "coursenav_nominal"
        msg.info.resolution = float(res)
        msg.info.width, msg.info.height = int(w), int(hgt)
        msg.info.origin.position.x, msg.info.origin.position.y = float(ox), float(oy)
        msg.info.origin.orientation.w = 1.0
        msg.data = data.tolist()
        self.pub_local.publish(msg)

    def _publish_prior(self):
        self.pub_prior.publish(String(data=json.dumps(self.prior_mon.as_dict())))

    def _status(self):
        st = dict(self.counts, resets=self.resets, vio_ok=self.vio_ok, ready=self.pf.ready,
                  height_m=None if self.height is None else round(self.height[1], 3),
                  ess=round(self.pf.ess()) if self.pf.ready else None,
                  scans_skipped=self.pf.skipped, scan_fit=self.pf.last_fit,
                  cluster_frac=round(self.pf.last_cluster_frac, 2),
                  live_map=self.live.stats() if self.live is not None else None,
                  local_map=self.local.stats() if self.local is not None else None,
                  prior=self.prior_mon.as_dict())
        if self.last_est is not None:
            t, mean, cov, pose, *_ = self.last_est
            st.update(t=round(t, 3), x=round(float(mean[0]), 3), y=round(float(mean[1]), 3),
                      yaw_deg=round(math.degrees(mean[2]), 2),
                      sigma_xy=round(float(math.sqrt(max(cov[0, 0], cov[1, 1]))), 3),
                      sigma_yaw_deg=round(math.degrees(math.sqrt(cov[2, 2])), 2))
        self.pub_status.publish(String(data=json.dumps(st)))
        self._publish_prior()
        self.log_f.flush()

    def destroy_node(self):
        try:
            self.log_f.close()
            if self.live is not None and self.live.scans_in:
                self.live.save(self.occ_path, source="loc_node", log=str(self.log_path))
        except Exception as e:                                # never let a record failure mask shutdown
            self.get_logger().error(f"could not save the live map: {e}")
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
