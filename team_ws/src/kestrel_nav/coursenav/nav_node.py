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
from scipy import ndimage
from std_msgs.msg import String

from .course import Grid2D, load_course, plan_grids, resolve, scan_grid
from .drift import DriftEKF
from .navigator import Navigator
from .navplan import course_to_ned, headings_to_frame, smooth_params_for
from .objmap import CourseMap
from .planner import PlanError

AIRBORNE = ("hold", "turn", "check", "go", "blocked", "backoff", "recheck", "search", "align")


class LiveLandmarks:
    """The localizer's published occupancy grid as the navigator's live map: confident surfaces more
    than novel_min_m from any prior surface, in clumps of at least min_cells (coursenav.occgrid rules)."""

    def __init__(self, prior: Grid2D, occ_prob=0.88, novel_min_m=0.15, min_cells=4):
        self.prior = prior
        self.prior_dist = prior.distance()
        self.occ_prob, self.novel_min_m, self.min_cells = occ_prob, novel_min_m, min_cells
        self.occupied = np.zeros(prior.occ.shape, bool)

    def on_grid(self, m):
        nx, ny = self.prior.occ.shape
        if m.info.width != nx or m.info.height != ny:
            return False
        prob = np.asarray(m.data, np.int8).reshape(ny, nx).T          # row-major, x fastest
        self.occupied = prob >= int(round(self.occ_prob * 100))
        return True

    def landmarks(self):
        novel = self.occupied & (self.prior_dist > self.novel_min_m)
        if not novel.any() or self.min_cells <= 1:
            return novel
        lab, n = ndimage.label(novel, structure=np.ones((3, 3), bool))
        sizes = ndimage.sum(novel, lab, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, bool)
        keep[1:] = sizes >= self.min_cells
        return keep[lab]


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
        self.live = LiveLandmarks(sg) if sg.occ.shape == obst.occ.shape else None
        self.nav = Navigator(obst, blocked, clearance, need, sp, live=self.live, gate_lines=rd.get("gate_lines"))
        # the flight frame's heading relative to the VIO origin: 90 deg in the simulator (vision bridge
        # heading_enu_rad), 0 on the skybeta drone (the flight interface permutes odometry ENU directly)
        self.decl = math.radians(float(dp("declared_heading_deg", 90.0).value))
        self.drift = None
        self.status = None
        self.nav_leg, self.pid = None, 0
        self.last_attempt = -math.inf
        self.search_replanned = False
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
        if phase != 'search':
            self.search_replanned = False

        def to_ned(pts):
            return course_to_ned(ekf.to_nominal(pts, nb), d["fixture"], self.decl)

        targets = {j: [float(v) for v in to_ned([self.nav_legs[j]["stop"]])[0]]
                   for j in range(leg + 1, len(self.nav_legs))}
        out = {"stamp": time.time(), "leg": leg, "pid": self.pid, "targets_ne": targets}
        if leg < len(self.nav_legs) and self.nav_legs[leg]["path"] and phase in AIRBORNE:
            L = self.nav_legs[leg]
            want = self.nav_leg != leg or self.nav.path is None or (phase == 'search' and not self.search_replanned)
            if not want and self.nav.path is not None:
                want = self.nav.needs_replan(time.monotonic(), 0.0)
            c = self.cache.pop(leg, None) if self.nav_leg != leg else None
            if want and c is not None and np.hypot(*(c[0] - believed[:2])) <= 0.5:
                self.nav.path, self.nav.goals, self.nav.t_plan = c[2], c[1], time.monotonic()
                self.pid += 1
                self.nav_leg = leg
                want = False
                self.get_logger().info(f"leg {leg + 1}: pre-planned path {self.nav.path.length:.2f} m")
            if want and time.monotonic()-self.last_attempt >= self.nav.p.replan_every_s:
                self.last_attempt = time.monotonic()
                self.search_replanned = phase == 'search'
                goals = [np.array(g, float) for g in L["goals"]]
                goals = [g for g in goals[:-1] if np.hypot(*(g - believed[:2])) > 0.3] + [goals[-1]]
                try:
                    v0 = float(self.status.get("v_cmd") or 0.0) if self.nav_leg == leg else 0.0
                    self.nav.plan(believed[:2], goals, t=time.monotonic(), v_start=v0)
                    self.stats["replans" if self.nav_leg == leg else "plans"] += 1
                    self.pid += 1
                    self.nav_leg = leg
                    self.get_logger().info(f"leg {leg + 1}: path {self.nav.path.length:.2f} m, "
                                           f"{len(self.nav.path.stops)} stop corners, est {self.nav.path.duration():.1f} s")
                except PlanError as e:
                    self.nav.path = None
                    self.stats["plan_errors"] += 1
                    self.get_logger().warning(f"leg {leg + 1}: no path ({e}); refnav holds without one")
            if self.nav.path is not None and self.nav_leg == leg:
                P = self.nav.path
                out.update(pid=self.pid, pts_ne=to_ned(P.pts).round(4).tolist(), v=P.v.round(3).tolist(),
                           stops=list(P.stops),
                           tubes=None if P.tube is None else P.tube.round(4).tolist())
                if P.yaw is not None:                   # CRAB: the heading to hold per point (NED rad)
                    out["yaw_ne"] = headings_to_frame(P.pts, P.yaw, to_ned).round(4).tolist()
        out['valid'] = bool(out.get('pts_ne'))
        self.pub.publish(String(data=json.dumps(out)))


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
