#!/usr/bin/env python3
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

Run:  python3 -m refnav.node --ros-args -p route_file:=scenarios/<x>.json -p depth_topic:=... \
          -p use_sim_time:=true
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
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String

from . import core as C
from . import depth as D
from .trace import LPOS_FIELDS, TraceWriter, fields

# PX4's uXRCE-DDS endpoints are BEST_EFFORT + TRANSIENT_LOCAL; a different durability never matches
PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)
CMD_SET_MODE, CMD_ARM, CMD_LAND = 176, 400, 21
PX4_MAIN_MODE_OFFBOARD = 6


def _as(default, val):
    """Coerce a ROS parameter value to the type of the Params default ("true"/"false" strings count)."""
    if isinstance(default, bool):
        return val if isinstance(val, bool) else str(val).strip().lower() in ("1", "true", "yes")
    return type(default)(val)


def load_route(node):
    path = node.declare_parameter("route_file", "").value
    flat = list(node.declare_parameter("route", [0.0]).value)
    if path:
        with open(path) as f:
            doc = json.load(f)
        pts = doc["route"] if isinstance(doc, dict) else doc
        return [(float(a), float(b)) for a, b in pts], doc if isinstance(doc, dict) else {}
    if len(flat) >= 2 and len(flat) % 2 == 0:
        return [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat), 2)], {}
    raise SystemExit("refnav: give route_file:=<json> or route:=[f1,l1,f2,l2,...] (forward/left metres)")


class RefNav(Node):
    def __init__(self):
        super().__init__("refnav")
        route, doc = load_route(self)
        # per-leg metadata (gate alignment, passages) travels with the route file, never as a ROS parameter
        prm = C.Params(route=route, legs=list(doc.get("legs", [])), land_columns=doc.get("land_columns"))
        for k, v in vars(C.Params()).items():
            if k in ("route", "legs", "land_columns"):
                continue
            default = doc.get("params", {}).get(k, v)
            val = self.declare_parameter(k, default).value
            setattr(prm, k, _as(v, val))
        if prm.sim_stuck_land_disarm_s > 0 and not bool(self.get_parameter("use_sim_time").value):
            self.get_logger().warning("sim_stuck_land_disarm_s is a SIMULATOR workaround: ignored without use_sim_time")
            prm.sim_stuck_land_disarm_s = 0.0
        self.mission = C.Mission(prm)
        self.return_enabled = "automatic_return" in doc
        self.diagnostic_enabled = "diagnostic" in doc
        if self.return_enabled and self.diagnostic_enabled:
            raise ValueError('diagnostic and automatic return are separate missions')
        self.diagnostic_config = doc.get('diagnostic')
        self._diagnostic_summary_written = False
        if self.return_enabled:
            from coursenav.return_plan import ReturnGeometry
            from .return_home import ReturnMission
            self.mission = ReturnMission(prm, ReturnGeometry(doc["automatic_return"]))
        elif self.diagnostic_enabled:
            from .diagnostic import DiagnosticMission
            self.mission = DiagnosticMission(prm, self.diagnostic_config)
        self.depth_topic = self.declare_parameter("depth_topic", "/camera/depth/image_raw").value
        info = self.declare_parameter("depth_info_topic", "").value
        self.info_topic = info or self.depth_topic.rsplit("/", 1)[0] + "/camera_info"
        self.health_topic = self.declare_parameter("health_topic", "/estimator/healthy").value
        self.base_frame = self.declare_parameter("base_frame", "base_link").value
        self.exit_when_done = bool(self.declare_parameter("exit_when_done", True).value)
        control = self.declare_parameter("control_topic", "").value
        self.run_id = self.declare_parameter("run_id", "").value
        # PX4-bound stamps use the SYSTEM clock in every setting: on the aircraft PX4's DDS client
        # converts agent (system) time with its timesync; the simulator's PX4 boots with
        # UXRCE_DDS_SYNCT=0 and clamps a future stamp to its own "now". Simulation time would
        # be read as seconds-old commands by a real PX4, so it is never used for these.
        self.sys_clock = Clock(clock_type=ClockType.SYSTEM_TIME)
        self.trace = TraceWriter(self.declare_parameter("trace_file", "").value,
                                 warn=self.get_logger().warning)
        self._lpos_received_ros_s = None
        self._lpos_timestamp = None
        if self.diagnostic_enabled:
            self._diagnostic_record('header', config=self.diagnostic_config,
                platform='workspace native PX4; fixed position hold during stationary yaw, velocity-only translation',
                measurement_time_basis='PX4 estimate ROS receipt; raw PX4 source clock retained separately',
                fitting_scope='command-to-received-estimate response, including transport/estimator effects',
                clamp_observation='no companion clamp in protocol; internal PX4 limiting is not observed',
                transit_controller='existing native refnav position/velocity feedforward; excluded from fitting')

        from tf2_ros import Buffer, TransformListener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        from px4_msgs.msg import (BatteryStatus, EstimatorStatusFlags, OffboardControlMode, TrajectorySetpoint, VehicleAttitude,
                                  VehicleCommand, VehicleLandDetected, VehicleLocalPosition, VehicleStatus)
        self._OCM, self._TSP, self._CMD = OffboardControlMode, TrajectorySetpoint, VehicleCommand
        # px4_output: "direct" (FlyX lab: this node owns /fmu/in, arms and sets Offboard) or "skybeta" (the
        # drone: /fmu/in is the organizer flight interface's; the pilot arms, the interface enters Offboard,
        # setpoints go to /skybeta/setpoint/twist; see SkybetaOutput). Default unchanged.
        self.px4_output = str(self.declare_parameter("px4_output", "direct").value)
        if self.px4_output not in ("direct", "skybeta"):
            raise ValueError("px4_output must be direct or skybeta")
        lpos_topic, status_topic_px4, land_topic = ("/fmu/out/vehicle_local_position_v1", "/fmu/out/vehicle_status_v1",
                                                    "/fmu/out/vehicle_land_detected")
        if self.px4_output == "skybeta":
            from .skybeta import SkybetaOutput
            self.sky = SkybetaOutput(self)
            self.pub_ocm = self.pub_sp = self.pub_cmd = None
            try:                                            # the organizer's version-aware topic names
                from skybeta_flight_interface.topics import resolve_endpoints
                ep = resolve_endpoints()
                lpos_topic, status_topic_px4, land_topic = (ep["vehicle_local_position"], ep["vehicle_status"],
                                                            ep["vehicle_land_detected"])
            except Exception as e:                          # noqa: BLE001
                self.get_logger().warning(f"skybeta topic resolver unavailable ({e}); using _v1 names")
            self.get_logger().info(f"skybeta output: twist only, no /fmu/in; PX4 topics {lpos_topic}, "
                                   f"{status_topic_px4}, {land_topic}")
        else:
            self.sky = None
            self.pub_ocm = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", PX4_QOS)
            self.pub_sp = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", PX4_QOS)
            self.pub_cmd = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", PX4_QOS)
        status_topic = self.declare_parameter("status_topic", "/refnav/status").value
        self.pub_status = self.create_publisher(String, status_topic, 10)

        self.v = C.Vehicle()
        self.create_subscription(VehicleLocalPosition, lpos_topic, self._on_lpos, PX4_QOS)
        self.create_subscription(VehicleAttitude, "/fmu/out/vehicle_attitude", self._on_att, PX4_QOS)
        self.create_subscription(VehicleStatus, status_topic_px4, self._on_status, PX4_QOS)
        self.create_subscription(VehicleLandDetected, land_topic, self._on_land, PX4_QOS)
        if self.return_enabled:
            # this PX4 image publishes battery_status_v1 (run015 topic list); subscribe to both names so
            # the return's battery check sees whichever exists (run015: "battery missing or stale")
            for topic in ("/fmu/out/battery_status_v1", "/fmu/out/battery_status"):
                self.create_subscription(BatteryStatus, topic, self._on_battery, PX4_QOS)
            self.create_subscription(String, "/coursenav/status", self._on_localizer, 10)
        self._yaw_align = (False, None)
        self._ev_yaw = (False, None)
        self.create_subscription(EstimatorStatusFlags, "/fmu/out/estimator_status_flags", self._on_est_flags, PX4_QOS)
        self.create_subscription(Bool, self.health_topic, self._on_health, 10)
        if self.mission.p.prior_required:
            # latched on the localizer side: the last state arrives even if refnav starts later
            from rclpy.qos import DurabilityPolicy as _D, QoSProfile as _Q, ReliabilityPolicy as _R
            self.create_subscription(String, "/coursenav/prior_state", self._on_prior,
                                     _Q(depth=1, reliability=_R.RELIABLE, durability=_D.TRANSIENT_LOCAL))
        if self.diagnostic_enabled:
            from sensor_msgs.msg import Imu
            imu_topic = self.declare_parameter('diagnostic_imu_topic', '/stereo/imu').value
            self.create_subscription(Imu, imu_topic, self._on_diagnostic_imu, qos_profile_sensor_data)
            self.create_subscription(String, '/coursenav/drift', self._on_diagnostic_drift, 10)
        self.create_subscription(CameraInfo, self.info_topic, self._on_info, qos_profile_sensor_data)
        self.create_subscription(Image, self.depth_topic, self._on_depth, qos_profile_sensor_data)
        corr_topic = self.declare_parameter("correction_topic", "").value
        if corr_topic:
            # optional map correction of this mission's targets (geometry_msgs/Vector3Stamped: x = north m,
            # y = east m to ADD to the PX4-frame targets, z = 1-sigma m); applied only at leg starts,
            # bounded by the correction_* parameters (off unless correction_max_m > 0)
            from geometry_msgs.msg import Vector3Stamped
            self.create_subscription(Vector3Stamped, corr_topic, lambda m: self.mission.on_correction(
                m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, m.vector.x, m.vector.y, m.vector.z), 10)
        path_topic = self.declare_parameter("path_topic", "/coursenav/path").value
        if self.mission.p.track_mode == "path" and path_topic:
            # the navigator's path for the current leg (coursenav.nav_node, std_msgs/String JSON: leg, pid,
            # pts_ne, v, stops) and the later stops on the same correction (targets_ne). No path = no
            # forward motion on a path leg (refnav waits; path_stale_s).
            self.create_subscription(String, path_topic, self._on_path, 10)
        if control:
            # transient-local: a request made a moment before this node subscribed still arrives
            ctl_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 history=HistoryPolicy.KEEP_LAST, depth=1)
            self.create_subscription(String, control, self._on_control, ctl_qos)

        self.K = None
        self.extrinsic = None           # (frame_id, R_body_cam, t_body_cam)
        self.rejects = {}
        self.done_at = None
        self._wait_log_t = None
        self.create_timer(0.05, self._tick)            # node clock: pauses with a paused simulator
        self.create_timer(0.5, self._publish_status)
        self.get_logger().info(f"route {len(route)} legs; depth {self.depth_topic}; health {self.health_topic}; "
                               f"sim time {self.get_parameter('use_sim_time').value}")

    # ------------------------------------------------------------------ inputs
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    EV_YAW_GRACE_S = 2.0
    RESET_MAX_HEADING = math.radians(10.0)      # larger EKF resets in flight: refnav's NED targets are no longer trusted
    RESET_MAX_XY_M = 0.30

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
        dxy = (math.hypot(*[float(c) for c in getattr(m, 'delta_xy', (0.0, 0.0))[:2]]) if cur[0] != prev[0] else 0.0)
        msg = f"EKF reset in flight: heading {math.degrees(dh):+.1f} deg, xy {dxy:.2f} m (counters {prev} -> {cur})"
        if abs(dh) > self.RESET_MAX_HEADING or dxy > self.RESET_MAX_XY_M:
            self.get_logger().warning(msg + ": frame no longer trusted, landing in place")
            self.mission.request_land(msg)
        else:
            self.get_logger().info(msg + ": small, continuing")

    def _on_est_flags(self, m):
        self._yaw_align = (bool(m.cs_yaw_align), self._now())
        self._ev_yaw = (bool(getattr(m, 'cs_ev_yaw', False)), self._now())
        if self._ev_yaw[0]:
            self._ev_yaw_true_t = self._now()

    def _on_lpos(self, m):
        accepted = self._return_observe('position', m.timestamp * 1e-6,
                eph=float(getattr(m, 'eph', math.nan)), epv=float(getattr(m, 'epv', math.nan)),
                resets=tuple(getattr(m, k, None) for k in ('xy_reset_counter', 'z_reset_counter', 'heading_reset_counter')),
                velocity_resets=tuple(getattr(m, k, None) for k in ('vxy_reset_counter', 'vz_reset_counter')),
                velocity_valid=bool(m.v_xy_valid and getattr(m, 'v_z_valid', False)), heading=float(m.heading))
        if getattr(self, 'diagnostic_enabled', False):
            now = self._now()
            self._diagnostic_record('odom', now=now, source_stamp=now,
                position=[float(m.y), float(m.x), -float(m.z)],
                velocity_enu=[float(m.vy), float(m.vx), -float(m.vz)],
                yaw_enu=float((math.pi / 2 - m.heading + math.pi) % (2 * math.pi) - math.pi),
                px4_source_stamp_us=int(m.timestamp),
                px4_sample_stamp_us=getattr(m, 'timestamp_sample', None),
                accepted=bool(accepted and m.xy_valid and m.z_valid),
                position_valid=bool(m.xy_valid and m.z_valid),
                velocity_valid=bool(m.v_xy_valid and getattr(m, 'v_z_valid', False)),
                xy_reset_counter=getattr(m, 'xy_reset_counter', None),
                heading_reset_counter=getattr(m, 'heading_reset_counter', None))
        if not accepted:
            return
        self._lpos_received_ros_s = self._now()
        self._lpos_timestamp = getattr(m, "timestamp", None)
        if self.trace.enabled:
            self.trace.record("local_position", self._lpos_received_ros_s,
                              data=fields(m, LPOS_FIELDS))
        self.v.pos = (m.x, m.y, m.z)
        self.v.heading = m.heading
        rng = float(getattr(m, "dist_bottom", math.nan))
        self.v.rng = rng if math.isfinite(rng) else None
        # velocity (NED m/s) for pose tracking and the settled-arrival check; None when PX4 flags it invalid
        vel = (m.vx, m.vy, m.vz)
        self.v.vel = vel if bool(m.v_xy_valid) and all(math.isfinite(c) for c in vel) else None
        if getattr(self, 'diagnostic_enabled', False) and not getattr(m, 'v_z_valid', False):
            self.v.vel = None
        # heading: PX4's heading_good_for_control, OR its own yaw alignment. With a COMPASS, PX4 sets
        # heading_good_for_control only after re-aligning the heading in flight above 1.5 m (ekf.h
        # isYawFinalAlignComplete: ground magnetic anomalies), so a compass aircraft would wait on the
        # ground forever (field_two_tags, 2026-10-17); yaw alignment is what PX4's own position
        # controller flies on. A vision heading sets heading_good_for_control on the ground.
        ya, ya_t = self._yaw_align
        heading_ok = bool(m.heading_good_for_control) or (ya and ya_t is not None and self._now() - ya_t <= 3.0)
        if self.px4_output == "skybeta" and not heading_ok:
            # skybeta (kestral_nav934am): vision arrives in an FRD frame (pose_frame 2, EKF2_EV_CTRL 9). PX4 1.16
            # ev_yaw_control resets yaw to EV and leaves yaw_align FALSE for FRD, so heading_good_for_control
            # never sets (deploy CLAUDE.md, drone 16 bench: "it does not gate Position mode"). The heading is
            # the fused vision yaw: accept it while EKF2 reports cs_ev_yaw (fresh) and the heading is finite.
            # grace: a brief EV gap (cuVSLAM stalls 150-430 ms on drone 14) clears cs_ev_yaw for a moment; PX4 keeps
            # fusing on its own timeout, as heading_good_for_control does not flicker on the lab
            ev_true_t = getattr(self, '_ev_yaw_true_t', None)
            heading_ok = bool(ev_true_t is not None and self._now() - ev_true_t <= self.EV_YAW_GRACE_S)
        if self.px4_output == "skybeta":
            self._check_reset(m)
        self.v.pos_ok = bool(m.xy_valid and m.z_valid and heading_ok
                             and all(math.isfinite(c) for c in (m.x, m.y, m.z, m.heading)))

    def _on_att(self, m):
        if not self._return_observe('attitude', m.timestamp * 1e-6):
            return
        self.v.q = tuple(float(c) for c in m.q)

    def _on_status(self, m):
        if not self._return_observe('status', m.timestamp * 1e-6):
            return
        self.v.armed = m.arming_state == 2
        self.v.nav_state = int(m.nav_state)

    def _on_land(self, m):
        accepted = self._return_observe('land', m.timestamp * 1e-6)
        if getattr(self, 'diagnostic_enabled', False):
            now = self._now()
            self._diagnostic_record('land', now=now, source_stamp=now,
                px4_source_stamp_us=int(m.timestamp), landed=bool(m.landed),
                accepted=bool(accepted))
        if not accepted:
            return
        self.v.landed = bool(m.landed)

    def _return_observe(self, name, stamp, **data):
        if not self.return_enabled and not getattr(self, 'diagnostic_enabled', False):
            return True
        return self.mission.observe(name, stamp, self._now(), time.monotonic(), **data)

    def _diagnostic_record(self, kind, now=None, source_stamp=None, frame_id='odom', **data):
        if not getattr(self, 'diagnostic_enabled', False):
            return
        now = self._now() if now is None else now
        row = dict(schema=1, kind=kind, t_ros=now, t_mono=time.monotonic(),
                   source_stamp=source_stamp, frame_id=frame_id,
                   **self.mission.diagnostic_status())
        row.update(data)
        self.trace.record('diagnostic', now, row=row)

    def _on_diagnostic_imu(self, m):
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        now = self._now()
        a, g = m.linear_acceleration, m.angular_velocity
        acceleration, angular = [a.x, a.y, a.z], [g.x, g.y, g.z]
        previous_frame = getattr(self, '_diagnostic_imu_frame', None)
        if not m.header.frame_id or (previous_frame is not None and previous_frame != m.header.frame_id):
            self.mission.abort_diagnostic('diagnostic IMU frame missing or changed', now)
            self._diagnostic_record('imu', now=now, source_stamp=stamp,
                frame_id=m.header.frame_id, acceleration=acceleration, angular_velocity=angular, accepted=False)
            return
        self._diagnostic_imu_frame = m.header.frame_id
        accepted = self.mission.on_imu(stamp, now, time.monotonic(),
                                      linear_acceleration=acceleration, angular_velocity=angular)
        self._diagnostic_record('imu', now=now, source_stamp=stamp,
                                frame_id=m.header.frame_id, acceleration=acceleration,
                                angular_velocity=angular, accepted=bool(accepted))

    def _on_diagnostic_drift(self, m):
        try:
            self.mission.on_drift(json.loads(m.data), self._now(), time.monotonic())
        except (TypeError, ValueError, KeyError):
            self.mission.abort_diagnostic('invalid diagnostic map-frame update', self._now())

    def _on_battery(self, m):
        self._return_observe('battery', m.timestamp * 1e-6, connected=bool(m.connected),
                             remaining=float(m.remaining), warning=int(m.warning), faults=int(m.faults),
                             powering_off=bool(m.is_powering_off), time_remaining_s=float(m.time_remaining_s))

    def _on_localizer(self, m):
        try:
            d = json.loads(m.data)
            self._return_observe('localizer', float(d['t']),
                                 **{k: d.get(k) for k in ('ready', 'updates', 'sigma_xy', 'x', 'y', 'resets')})
        except (ValueError, TypeError, KeyError):
            self.mission.samples.pop('localizer', None)

    def _on_prior(self, m):
        try:
            d = json.loads(m.data)
            self.mission.on_prior_state(self._now(), d["state"], d.get("fit"))
        except (ValueError, TypeError, KeyError):
            pass

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
            if not isinstance(d, dict):
                raise ValueError("path message must be an object")
            leg = d.get("leg", -1)
            if type(leg) is not int:
                raise ValueError("path leg must be an integer")
        except (ValueError, TypeError):
            self.trace.record("path_invalid", self._now(), raw=m.data)
            return
        self.trace.record("path", self._now(), data=d)
        mi = self.mission
        if mi.targets is None:
            return
        received = self._now()
        if d.get("plan_state") in ("pending", "failed"):
            mi.on_plan_result(received, leg, d.get("request_id"),
                              d["plan_state"], d.get("reason", ""))
            return
        if mi.recover is not None and mi.p.recovery_replan:
            if (mi.replan_request is None or d.get("request_id") != mi.replan_request["id"]
                    or leg != mi.leg or d.get("recovery_certified") is not True):
                return
        elif d.get("request_id") is not None:
            return
        try:
            # Validate before mutating any target: a rejected recovery response
            # must not change subsequent legs through its correction metadata.
            targets = []
            for j, ne in (d.get("targets_ne") or {}).items():
                j = int(j)
                if mi.leg < j < len(mi.targets) and len(ne) == 2 and all(math.isfinite(float(c)) for c in ne):
                    targets.append((j, (float(ne[0]), float(ne[1]))))
            if leg == mi.leg and d.get("pts_ne"):
                accepted = mi.on_path(received, mi.leg, d["pts_ne"], d.get("v") or [],
                                      d.get("stops") or [], pid=d.get("pid", 0), tubes=d.get("tubes"),
                                      request_id=d.get("request_id"), recovery_certified=d.get("recovery_certified", False),
                                      hop=d.get("recovery_hop") is True)
                self.trace.record("path_applied", received, accepted=accepted,
                                  leg=mi.leg, pid=d.get("pid", 0))
                if not accepted:
                    return
            elif mi.recover is not None and mi.p.recovery_replan:
                return
            for j, ne in targets:
                mi.targets[j] = ne
        except (ValueError, TypeError, OverflowError, AttributeError):
            self.trace.record("path_invalid", received, data=d)

    def _on_control(self, m):
        try:
            req = json.loads(m.data)
        except ValueError:
            return
        if req.get("request") == "land" and (not self.run_id or req.get("run_id") == self.run_id):
            self.get_logger().info(f"landing requested by the runtime ({req.get('why', 'operator')})")
            self.mission.request_land(req.get("why") or "operator")

    def _reject(self, key, detail=None, record=True):
        """Count a refused depth frame by its reason; log the reason WITH its numbers at the 1st, 10th,
        100th and 1000th time; tell the mission (which reports it while it waits)."""
        detail = detail or key
        if record:
            self.mission.depth_refused(key, detail)
        self.rejects[key] = self.rejects.get(key, 0) + 1
        if self.rejects[key] in (1, 10, 100, 1000):
            self.get_logger().warning(f"depth frame not used ({self.rejects[key]}x): {detail}")

    def _calibration(self, frame_id):
        if self.extrinsic is not None and self.extrinsic[0] == frame_id:
            return self.extrinsic
        from rclpy.time import Time
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, frame_id, Time())
        except Exception as e:                      # not (yet) on the TF tree
            self._reject("no_tf", f"no TF {self.base_frame} <- {frame_id} ({type(e).__name__})")
            return None
        r, t = tf.transform.rotation, tf.transform.translation
        R = D.quat_to_mat(r.w, r.x, r.y, r.z)
        fwd = R[:, 2]                               # the optical axis in the body frame
        ang = math.degrees(math.acos(max(-1.0, min(1.0, fwd[0] / np.linalg.norm(fwd)))))
        if ang > 25.0:
            self._reject("camera_not_forward",
                         f"camera {frame_id} looks {ang:.0f} deg off the body's forward axis: this package needs a forward camera")
            return None
        self.extrinsic = (frame_id, R, np.array([t.x, t.y, t.z]))
        self.get_logger().info(f"calibration from TF: {self.base_frame} <- {frame_id} t=({t.x:.3f},{t.y:.3f},{t.z:.3f}) "
                               f"optical axis {ang:.1f} deg off forward")
        return self.extrinsic

    def _on_depth(self, m):
        if self.K is None:
            return self._reject("no_camera_info", f"no camera_info yet on {self.info_topic}")
        cal = self._calibration(m.header.frame_id)
        if cal is None:
            return None
        if self.v.q is None:
            return self._reject("no_attitude", "no PX4 attitude yet")
        try:
            d = D.decode_depth(m.encoding, m.height, m.width, m.step, m.data)
        except ValueError as e:
            return self._reject("undecodable", str(e))
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if not self.mission.on_depth(stamp, d, self.K, cal[1], cal[2], self.v.q, now=self._now()):
            key, detail = self.mission.depth_reject or ("refused", "refused by the mission")
            self._reject(key, detail, record=False)
        return None

    # ------------------------------------------------------------------ outputs
    def _sys_us(self):
        return self.sys_clock.now().nanoseconds // 1000

    def _command(self, cmd, p1=0.0, p2=0.0):
        c = self._CMD()
        c.timestamp = self._sys_us()
        c.command, c.param1, c.param2 = cmd, float(p1), float(p2)
        c.target_system = c.target_component = c.source_system = c.source_component = 1
        c.from_external = True
        self.pub_cmd.publish(c)

    def _tick(self):
        now = self._now()
        if now <= 0.0:
            return                                   # sim time before the first /clock
        diagnostic = getattr(self, 'diagnostic_enabled', False)
        if diagnostic and (not self.trace.enabled or self.trace.failed or self.trace.dropped):
            self.mission.abort_diagnostic('diagnostic recording unavailable, failed or dropped events', now)
        if (self.sky is not None and self.sky.land_xy is not None and self.v.armed is False
                and self.mission.phase == "land"):
            # skybeta: no AUTO_LAND exists; the interface descent + PX4's auto-disarm (COM_DISARM_LAND) IS the
            # landing, so the disarm after it confirms touchdown like AUTO_LAND -> disarmed does on the lab
            self.mission.land_auto_seen = True
        out = (self.mission.step(now, self.v, wall_now=time.monotonic()) if self.return_enabled or diagnostic
               else self.mission.step(now, self.v))
        if diagnostic and getattr(self.mission, 'protocol_tick', None) is not None:
            # Include the failed tick even when no experiment command publishes.
            # evaluated_phase in the snapshot predates a transition to FAULT.
            self._diagnostic_record('protocol_tick', now=now, source_stamp=now,
                                    evidence=self.mission.protocol_tick)
        for e in out.events:
            self.get_logger().info(e)
        if self.mission.phase == "wait" and self.mission.t_phase is not None:
            if self._wait_log_t is None or now - self._wait_log_t >= 10.0:   # say why it is not flying, every 10 s
                if self._wait_log_t is not None:
                    self.get_logger().info(f"not ready after {now - self.mission.t_phase:.0f} s "
                                           f"(gives up at {self.mission.p.wait_s:.0f} s): {self.mission.why}")
                self._wait_log_t = now
        if out.events:
            self._publish_status()
        published = None
        if self.sky is not None:                            # the drone: everything goes through the interface
            published = self.sky.publish(now, out, self.v)
            out.commands = []
        velocity_only = getattr(out, 'velocity_only', None) if self.sky is None else None
        position_hold = getattr(out, 'position_hold', None)
        if out.stream and (velocity_only is not None or position_hold is not None):
            command = position_hold if position_hold is not None else velocity_only
            if len(command) != 4 or not all(math.isfinite(v) for v in command):
                self.mission.abort_diagnostic('invalid diagnostic velocity command', now)
                return
            o = self._OCM()
            o.timestamp = self._sys_us()
            o.position, o.velocity, o.acceleration = position_hold is not None, True, False
            self.pub_ocm.publish(o)
            s = self._TSP()
            s.timestamp = o.timestamp
            s.position = ([float(x) for x in position_hold[:3]] if position_hold is not None else [math.nan] * 3)
            s.velocity = ([0., 0., 0.] if position_hold is not None else [float(x) for x in velocity_only[:3]])
            s.acceleration = [math.nan] * 3
            s.yaw = math.nan
            s.yawspeed = float(command[3])
            self.pub_sp.publish(s)
            published = fields(s, ('timestamp', 'position', 'velocity', 'acceleration', 'yaw', 'yawspeed'))
            # Stamp the command in the measurement clock as well as preserving
            # the PX4-bound system timestamp. They are not interchangeable.
            vn, ve, vd, yr = s.velocity[0], s.velocity[1], s.velocity[2], s.yawspeed
            self._diagnostic_record('command', now=now, source_stamp=now,
                velocity_enu=[ve, vn, -vd], yaw_rate=-yr,
                control_mode='position_hold' if position_hold is not None else 'velocity_only',
                position_target_enu=[s.position[1], s.position[0], -s.position[2]] if position_hold is not None else None,
                px4_message_system_stamp_us=int(s.timestamp),
                offboard_active=self.v.nav_state == C.NAV_OFFBOARD, armed=self.v.armed,
                health=bool(self.mission.health.arm_ok(now)[0]), clamped=[],
                safety='run', fault=None)
        elif self.sky is None and out.stream and out.setpoint is not None:
            o = self._OCM()
            o.timestamp = self._sys_us()
            o.position = True
            o.velocity = out.velocity is not None          # velocity feed-forward alongside the position
            o.acceleration = out.accel is not None         # acceleration feed-forward (pose tracking)
            self.pub_ocm.publish(o)
            s = self._TSP()
            s.timestamp = o.timestamp
            s.position = [float(c) for c in out.setpoint[:3]]
            s.velocity = ([float(out.velocity[0]), float(out.velocity[1]), 0.0] if out.velocity is not None
                          else [math.nan] * 3)
            # vertical acceleration stays NaN: PX4's altitude loop owns it (0 would fight hover thrust)
            s.acceleration = ([float(out.accel[0]), float(out.accel[1]), math.nan] if out.accel is not None
                              else [math.nan] * 3)
            s.yaw = float(out.setpoint[3])
            if getattr(out, "yawspeed", None) is not None:
                s.yawspeed = float(out.yawspeed)            # turn profile feed-forward (yaw_accel_dps2)
            self.pub_sp.publish(s)
            if self.trace.enabled:
                published = fields(s, ("timestamp", "position", "velocity", "acceleration", "jerk",
                                       "yaw", "yawspeed"))
        for c in out.commands:
            if c == "offboard":
                self._command(CMD_SET_MODE, 1.0, PX4_MAIN_MODE_OFFBOARD)
            elif c == "arm":
                self._command(CMD_ARM, 1.0)
            elif c == "land":
                self._command(CMD_LAND)
            elif c == "disarm":
                self._command(CMD_ARM, 0.0)                 # param2 0: PX4 refuses it in the air
            elif c == "force_disarm":                       # SIM workaround only (Params.sim_stuck_land_disarm_s)
                self._command(CMD_ARM, 0.0, 21196.0)
        if out.phase == "done" and self.done_at is None:
            self.done_at = now
            self.get_logger().info(f"mission done: {self.mission.result}")
        self.trace.record("tick", now, phase=self.mission.phase, leg=self.mission.leg,
                          vehicle=self.v, output=out, published_trajectory=published,
                          lpos_received_ros_s=self._lpos_received_ros_s,
                          lpos_timestamp=self._lpos_timestamp,
                          path=None if self.mission.path is None else
                          {k: self.mission.path[k] for k in ("t", "leg", "pid")},
                          path_s=self.mission.path_s, path_stop_done=self.mission.path_stop_done)

    def _publish_status(self):
        st = self.mission.status(self._now())
        st["rejects"] = dict(self.rejects)
        self.pub_status.publish(String(data=json.dumps(st)))
        self.trace.record("status", self._now(), data=st)
        if self.exit_when_done and self.done_at is not None and self._now() - self.done_at > 1.5:
            raise SystemExit(0 if self.mission.result == "complete" else 1)

    def destroy_node(self):
        if hasattr(self, "trace"):
            if getattr(self, 'diagnostic_enabled', False) and not self._diagnostic_summary_written:
                self._diagnostic_summary_written = True
                self._diagnostic_record('summary', dropped_count=self.trace.dropped,
                    logging_error='trace writer failed' if self.trace.failed else None,
                    result=self.mission.diagnostic_status(),
                    native_result=self.mission.result, native_phase=self.mission.phase,
                    native_landed=self.v.landed, native_armed=self.v.armed)
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


if __name__ == "__main__":
    main()
