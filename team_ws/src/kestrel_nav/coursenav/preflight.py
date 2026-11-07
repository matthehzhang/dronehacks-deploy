"""Preflight: every condition a course run depends on, checked TOGETHER and reported together.

Read-only: subscribes and inspects processes; publishes nothing, commands nothing. Built after the
2026-11-07 live session, where each of these failed one at a time:
  * orphaned onboard services from a previous world kept a second cuVSLAM + the old vision bridge
    alive, and the new bridge refused to start ("another local process owns this vision input");
  * no localizer after a Reset (the default service profile has none);
  * a managed run's read-only project snapshot (checked by fly_course.run_record_dir);
  * a second controller would compete for /fmu/in (never seen, always checked).

    from coursenav.preflight import run_checks, report
    rows = run_checks(seconds=6.0)        # -> [Check(name, ok, hard, detail)]
    ok = report(rows)                     # prints a table; False if any HARD check failed
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

ONBOARD_ROLES = {          # process role -> the cmdline fragment that identifies it
    "cuvslam": "onboard.slam.cuvslam_lib_node",
    "vision_bridge": "onboard.vio.ev_bridge",
    "localizer": "coursenav.loc_node",
}
NAV_OFFBOARD = 14


@dataclass
class Check:
    name: str
    ok: bool
    hard: bool          # a hard failure blocks arming
    detail: str


def onboard_processes():
    """-> {role: [pid, ...]} of live processes on this machine (any owner record, or none)."""
    out = {r: [] for r in ONBOARD_ROLES}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        for role, frag in ONBOARD_ROLES.items():
            if f"-m {frag}" in cmd and "grep" not in cmd:
                out[role].append(int(d.name))
    return out


def _collect(seconds):
    """Spin a listener for `seconds` of wall time. -> (latest messages, arrival counts, publisher counts)."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
                           qos_profile_sensor_data)
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image
    from std_msgs.msg import Bool, String
    from px4_msgs.msg import (DistanceSensor, EstimatorStatusFlags, VehicleLandDetected, VehicleLocalPosition,
                              VehicleStatus)
    px4 = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)
    own = not rclpy.ok()
    if own:
        rclpy.init()
    node = Node("preflight_probe")
    last, count = {}, {}

    def sub(kind, topic, key, qos):
        def cb(m):
            last[key] = m
            count[key] = count.get(key, 0) + 1
        node.create_subscription(kind, topic, cb, qos)

    sub(VehicleStatus, "/fmu/out/vehicle_status_v1", "status", px4)
    sub(VehicleLandDetected, "/fmu/out/vehicle_land_detected", "land", px4)
    sub(VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1", "lpos", px4)
    sub(EstimatorStatusFlags, "/fmu/out/estimator_status_flags", "ekf", px4)
    sub(DistanceSensor, "/fmu/out/distance_sensor", "range", px4)
    sub(Bool, "/estimator/healthy", "healthy", 10)
    sub(Image, "/camera/depth/image_raw", "depth", qos_profile_sensor_data)
    sub(Odometry, "/visual_slam/tracking/odometry", "vio", qos_profile_sensor_data)
    sub(String, "/coursenav/status", "loc", 10)
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < seconds:
            rclpy.spin_once(node, timeout_sec=0.05)
        pubs = {t: node.count_publishers(t) for t in (
            "/fmu/in/vehicle_visual_odometry", "/visual_slam/tracking/odometry", "/fmu/in/trajectory_setpoint",
            "/fmu/in/offboard_control_mode", "/coursenav/px4_correction", "/coursenav/status")}
    finally:
        node.destroy_node()
        if own:
            rclpy.shutdown()
    return last, count, pubs, time.monotonic() - t0


def run_checks(seconds=6.0, allow_armed=False, need_localizer=True, collected=None):
    """-> [Check]. `collected` (last, count, pubs, elapsed) and the process table can be injected (tests)."""
    import json
    last, count, pubs, dt = collected if collected is not None else _collect(seconds)
    procs = onboard_processes() if collected is None else {}
    rate = lambda k: count.get(k, 0) / dt if dt > 0 else 0.0   # noqa: E731
    rows = []

    st = last.get("status")
    if st is None:
        rows.append(Check("PX4 status", False, True, "no /fmu/out/vehicle_status_v1"))
    else:
        armed = st.arming_state == 2
        ok = (allow_armed or not armed) and not st.failsafe
        rows.append(Check("PX4 status", ok, True,
                          f"{'ARMED' if armed else 'disarmed'}, nav_state {st.nav_state}, failsafe {st.failsafe}"))
        if st.nav_state == NAV_OFFBOARD and not armed:
            rows.append(Check("PX4 mode", True, False, "left in Offboard by an earlier run (harmless while disarmed)"))
    ld = last.get("land")
    rows.append(Check("landed", bool(ld and ld.landed) or allow_armed, True,
                      "no land detector" if ld is None else f"landed={ld.landed}"))
    lp = last.get("lpos")
    if lp is None:
        rows.append(Check("PX4 position", False, True, "no local position"))
    else:
        ok = bool(lp.xy_valid and lp.z_valid and lp.heading_good_for_control
                  and all(math.isfinite(v) for v in (lp.x, lp.y, lp.z, lp.heading)))
        rows.append(Check("PX4 position", ok, True, f"xy_valid {lp.xy_valid} z_valid {lp.z_valid} heading_ok "
                          f"{lp.heading_good_for_control} at ({lp.x:+.2f}, {lp.y:+.2f}, {lp.z:+.2f}) NED"))
    ek = last.get("ekf")
    rows.append(Check("EKF fusion", bool(ek and ek.cs_ev_pos and ek.cs_ev_yaw), True,
                      "no estimator flags" if ek is None else
                      f"ev_pos {ek.cs_ev_pos} ev_yaw {ek.cs_ev_yaw} range_height {ek.cs_rng_hgt}"))
    h = last.get("healthy")
    rows.append(Check("estimator health", bool(h and h.data) and rate("healthy") > 1.0, True,
                      "no /estimator/healthy" if h is None else f"{h.data} at {rate('healthy'):.1f} Hz"))
    rows.append(Check("depth stream", rate("depth") >= 5.0, True, f"{rate('depth'):.1f} Hz"))
    rows.append(Check("VIO odometry", rate("vio") >= 5.0, True, f"{rate('vio'):.1f} Hz"))
    rows.append(Check("rangefinder", rate("range") >= 1.0, True, f"{rate('range'):.1f} Hz"))

    # one estimator chain, one vision input to PX4, nobody else commanding PX4
    for role in ("cuvslam", "vision_bridge"):
        if collected is None:
            n = len(procs.get(role, []))
            rows.append(Check(f"{role} processes", n == 1, True,
                              f"{n} running {procs.get(role)}" + ("  <- orphan from an earlier world? stop the extra "
                                                                  "one" if n > 1 else "")))
    rows.append(Check("PX4 vision publishers", pubs.get("/fmu/in/vehicle_visual_odometry") == 1, True,
                      f"{pubs.get('/fmu/in/vehicle_visual_odometry')} on /fmu/in/vehicle_visual_odometry"))
    rows.append(Check("VIO publishers", pubs.get("/visual_slam/tracking/odometry") == 1, True,
                      f"{pubs.get('/visual_slam/tracking/odometry')} on /visual_slam/tracking/odometry"))
    others = pubs.get("/fmu/in/trajectory_setpoint", 0) + pubs.get("/fmu/in/offboard_control_mode", 0)
    rows.append(Check("no other controller", others == 0, True,
                      f"{others} publisher(s) already on trajectory_setpoint/offboard_control_mode"))

    if need_localizer:
        n_loc = pubs.get("/coursenav/status", 0)
        rows.append(Check("localizer instances", n_loc == 1, True, f"{n_loc} publishing /coursenav/status"))
        rows.append(Check("correction publishers", pubs.get("/coursenav/px4_correction", 0) <= 1, True,
                          f"{pubs.get('/coursenav/px4_correction', 0)} on /coursenav/px4_correction"))
        lm = last.get("loc")
        if lm is None:
            rows.append(Check("localizer converged", False, True, "no /coursenav/status"))
        else:
            try:
                s = json.loads(lm.data)
            except ValueError:
                s = {}
            ok = bool(s.get("ready")) and s.get("updates", 0) >= 30 and (s.get("sigma_xy") or 9) <= 0.15
            rows.append(Check("localizer converged", ok, True,
                              f"updates {s.get('updates')}, sigma {s.get('sigma_xy')} m, at ({s.get('x')}, "
                              f"{s.get('y')}) yaw {s.get('yaw_deg')}"))
    return rows


def report(rows, title="PREFLIGHT"):
    """Print every check; -> True when no HARD check failed."""
    w = max(len(r.name) for r in rows) if rows else 10
    print(f"---- {title} ----", flush=True)
    for r in rows:
        mark = "ok  " if r.ok else ("FAIL" if r.hard else "warn")
        print(f"  [{mark}] {r.name:<{w}}  {r.detail}", flush=True)
    bad = [r.name for r in rows if r.hard and not r.ok]
    print(f"---- {'PASS' if not bad else 'BLOCKED: ' + ', '.join(bad)} ----", flush=True)
    return not bad
