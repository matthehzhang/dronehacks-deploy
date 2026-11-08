"""coursenav navigator node (ROS 2): course-frame paths for refnav's path legs, re-planned in flight.

    in   /coursenav/drift      std_msgs/String JSON  the localizer's drift EKF: delta (dx, dy, dth), the
                                                     nominal body pose it is anchored at (coursenav.loc_node)
         /coursenav/occupancy  nav_msgs/OccupancyGrid the live map (landmarks become obstacles)
         /refnav/status        std_msgs/String JSON  the mission's leg and phase
    out  /coursenav/path       std_msgs/String JSON  {stamp, leg, pid, pts_ne, v, stops, targets_ne}

pts_ne is the current leg's path in PX4 local NED, re-expressed with the latest drift state every
tick (the geometry is planned in the course frame); pid changes only when the path is RE-PLANNED.
targets_ne holds every later stop (refnav keeps them on the same correction). Planning: coursenav.
navigator (fused map, A*, corner arcs, time-optimal profile). The route file is fly_course.py's run
record (route.json: nav_legs, navigation). Never commands PX4: refnav does.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .course import load_course, plan_grids, resolve, scan_grid, see_through_grid
from .drift import DriftEKF
from .livemap import LiveLandmarks
from .navigator import Navigator
from .navplan import course_to_ned, smooth_params_for
from .objmap import CourseMap
from .planner import PlanError

AIRBORNE = ("hold", "turn", "check", "go", "blocked", "backoff", "recheck", "align")


class NavNode(Node):
    def __init__(self):
        super().__init__("coursenav_nav")
        dp = self.declare_parameter
        route_file = dp("route_file", "").value
        if not route_file:
            raise SystemExit("coursenav.nav_node: give route_file:=<fly_course run record route.json>")
        with open(route_file) as f:
            rd = json.load(f)
        self.nav_legs = rd["nav_legs"]
        doc = load_course()
        cmap = CourseMap.load(resolve(doc, "map"))
        obst, blocked, clearance = plan_grids(cmap, doc)
        need = float(doc["vehicle"]["radius_m"] + doc["vehicle"]["margin_m"])
        sp = smooth_params_for(rd.get("navigation", {}), rd.get("params", {}), need)
        sg = scan_grid(cmap)
        self.live = None
        if sg.occ.shape == obst.occ.shape and bool(dp("live_map", True).value):
            st = see_through_grid(cmap) if bool(dp("clear_seen_through", True).value) else None
            self.live = LiveLandmarks(sg, see_through=st, plan_obst=obst.occ)
            if st is None:
                self.live.free_prob = -1.0                       # never clear a prior surface
        self.block_replan = bool(dp("replan_on_block", True).value)
        self.nav = Navigator(obst, blocked, clearance, need, sp, live=self.live, gate_lines=rd.get("gate_lines"))
        # the flight frame's heading relative to the VIO origin: 90 deg in the simulator (vision bridge
        # heading_enu_rad), 0 on the skybeta drone (the flight interface permutes odometry ENU directly)
        self.decl = math.radians(float(dp("declared_heading_deg", 90.0).value))
        self.drift = None
        self.status = None
        self.nav_leg, self.pid = None, 0
        self.stats = dict(plans=0, replans=0, plan_errors=0)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(String, dp("drift_topic", "/coursenav/drift").value, self._on_drift, 10)
        self.create_subscription(String, dp("status_topic", "/refnav/status").value, self._on_status, 10)
        self.create_subscription(OccupancyGrid, "/coursenav/occupancy", self._on_occ, latched)
        self.pub = self.create_publisher(String, dp("path_topic", "/coursenav/path").value, 10)
        self.create_timer(1.0 / float(dp("rate_hz", 10.0).value), self._tick)
        self.get_logger().info(f"navigator: {len(self.nav_legs)} legs ({sum(L['path'] for L in self.nav_legs)} "
                               f"path legs), cruise {sp.cruise_mps:.2f} m/s, buffer {need:.2f} m")
        self.cache = self._preplan(rd, route_file)

    def _preplan(self, rd, route_file):
        """kestral_nav934am (skybeta): plan every path leg BEFORE take-off, each from its leg's start (the start
        pose, then the previous stop), and write <run>/navigator_ready. On a Jetson-sized CPU budget (Gazebo,
        cgroup 1.4 cores) the in-flight leg-1 plan never arrived in 40 s; unbudgeted it took 14-24 s."""
        from pathlib import Path
        cache = {}
        start = rd.get("start_xy")
        if start is not None:
            t0 = time.monotonic()
            prev = np.asarray(start, float)
            for j, L in enumerate(self.nav_legs):
                if L["path"]:
                    goals = [np.array(g, float) for g in L["goals"]]
                    try:
                        P = self.nav.plan(prev, goals, t=time.monotonic())
                        cache[j] = (prev.copy(), goals, P)
                        self.get_logger().info(f"pre-planned leg {j + 1}: {P.length:.2f} m, {len(P.stops)} stop corners")
                    except PlanError as e:
                        self.get_logger().warning(f"pre-plan leg {j + 1} failed ({e}); it will be planned in flight")
                prev = np.asarray(L["stop"], float)
            self.nav.path = None
            self.get_logger().info(f"pre-planned {len(cache)} path legs in {time.monotonic() - t0:.1f} s")
        Path(route_file).with_name("navigator_ready").write_text(json.dumps({"legs": sorted(cache)}))
        return cache

    def _on_drift(self, m):
        try:
            self.drift = json.loads(m.data)
        except ValueError:
            pass

    def _on_status(self, m):
        try:
            self.status = json.loads(m.data)
        except ValueError:
            pass

    def _on_occ(self, m):
        if self.live is not None:
            self.live.on_grid(m)

    def _tick(self):
        if self.drift is None or self.status is None:
            return
        d = self.drift
        ekf = DriftEKF.from_state(d["delta"], d["nominal"])
        nb = np.array(d["nominal"], float)
        believed = ekf.body_course(nb)
        leg, phase = int(self.status.get("leg", 0)), self.status.get("phase")

        def to_ned(pts):
            return course_to_ned(ekf.to_nominal(pts, nb), d["fixture"], self.decl)

        targets = {j: [float(v) for v in to_ned([self.nav_legs[j]["stop"]])[0]]
                   for j in range(leg + 1, len(self.nav_legs))}
        out = {"stamp": time.time(), "leg": leg, "pid": self.pid, "targets_ne": targets}
        airborne_path_leg = leg < len(self.nav_legs) and self.nav_legs[leg]["path"] and phase in AIRBORNE

        def publish():
            if airborne_path_leg and self.nav.path is not None and self.nav_leg == leg:
                P = self.nav.path
                out.update(stamp=time.time(), pid=self.pid, pts_ne=to_ned(P.pts).round(4).tolist(),
                           v=P.v.round(3).tolist(), stops=list(P.stops),
                           tubes=None if P.tube is None else P.tube.round(4).tolist())
            self.pub.publish(String(data=json.dumps(out)))

        if not airborne_path_leg:
            publish()
            return
        L = self.nav_legs[leg]
        now = time.monotonic()
        want, why = self.nav_leg != leg, "new leg"
        if not want and self.nav.path is not None:
            prog = self.nav.progress(believed[:2])
            if self.nav.needs_replan(now, prog):
                want, why = True, "live map: obstacle on the path ahead"
            elif (self.block_replan and phase in ("blocked", "backoff", "recheck")
                  and (self.nav.t_plan is None or now - self.nav.t_plan >= self.nav.p.replan_every_s)
                  and self.nav.path_blocked_ahead(prog)):
                want, why = True, f"refnav {phase}: the live map now shows the obstacle"
        c = self.cache.pop(leg, None) if self.nav_leg != leg else None
        if want and c is not None and np.hypot(*(c[0] - believed[:2])) <= 0.5 and not self.nav.map_changed():
            self.nav.path, self.nav.goals, self.nav.t_plan = c[2], c[1], now
            self.pid += 1
            self.nav_leg = leg
            want = False
            self.get_logger().info(f"leg {leg + 1}: pre-planned path {self.nav.path.length:.2f} m")
        # the path refnav already has stays fresh while a (re-)plan runs: a plan on the Jetson can take
        # longer than refnav's path_stale_s, and a stale path stops the aircraft
        publish()
        if not want:
            return
        goals = [np.array(g, float) for g in L["goals"]]
        goals = [g for g in goals[:-1] if np.hypot(*(g - believed[:2])) > 0.3] + [goals[-1]]
        try:
            v0 = float(self.status.get("v_cmd") or 0.0) if self.nav_leg == leg else 0.0
            self.nav.plan(believed[:2], goals, t=now, v_start=v0)
            self.stats["replans" if self.nav_leg == leg else "plans"] += 1
            self.pid += 1
            self.nav_leg = leg
            sig = self.nav._sig
            self.get_logger().info(f"leg {leg + 1}: path {self.nav.path.length:.2f} m, {len(self.nav.path.stops)} "
                                   f"stop corners, est {self.nav.path.duration():.1f} s ({why}; live map: "
                                   f"{sig[0]} landmark cells, {sig[1]} prior cells cleared)")
            publish()
        except PlanError as e:
            self.nav.t_plan = now                       # retry at the re-plan rate, not every tick
            self.stats["plan_errors"] += 1
            self.get_logger().warning(f"leg {leg + 1}: no path ({e}; {why}); refnav keeps the last one"
                                      if self.nav.path is not None else
                                      f"leg {leg + 1}: no path ({e}); refnav holds without one")


def main(args=None):
    rclpy.init(args=args)
    node = NavNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
