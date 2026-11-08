"""coursenav navigator node (ROS 2): course-frame paths for refnav's path legs, re-planned in flight.

    in   /coursenav/drift      std_msgs/String JSON  the localizer's drift EKF: delta (dx, dy, dth), the
                                                     nominal body pose it is anchored at (coursenav.loc_node)
         /coursenav/occupancy  nav_msgs/OccupancyGrid the live map (landmarks become obstacles)
         /coursenav/local_occupancy nav_msgs/OccupancyGrid the LOCAL grid, nominal frame, never gated
                                                     (coursenav.localgrid): its obstacles are added to the
                                                     path map every tick, converted with the same drift
                                                     state as the aircraft's believed pose
         /coursenav/prior_state std_msgs/String JSON  placed / dropped (coursenav.priorstate): while
                                                     dropped, prior surfaces explain no local obstacle
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
from .navplan import course_to_ned, smooth_params_for
from .objmap import CourseMap
from .localgrid import course_mask, occupied_points_from_grid
from .planner import PlanError
from .observed_free import ObservedFree

AIRBORNE = ("hold", "turn", "check", "go", "blocked", "backoff", "recheck", "align", "scan", "replan", "recovery_scan")


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
        self.obst = obst
        self.surf_dist = sg.distance() if sg.occ.shape == obst.occ.shape else obst.distance()
        self.local_pts = None                      # (k, 2) nominal-frame obstacle points from the local grid
        self.local_explain_m = float(dp("local_explain_m", 0.15).value)
        self.prior_state = "unknown"
        self.observed_free = None
        self.recovery = None
        self.recovery_reference = None
        self.recovery_prefix = None
        self.reference_path = None
        self.reference_leg = None
        self.recovery_last_id = -1
        self.recovery_last_attempt = None
        self.recovery_min_interval_s = float(dp("recovery_min_interval_s", 0.5).value)
        self.recovery_grid_max_age_s = float(dp("recovery_grid_max_age_s", 2.0).value)
        # the flight frame's heading relative to the VIO origin: 90 deg in the simulator (vision bridge
        # heading_enu_rad), 0 on the skybeta drone (the flight interface permutes odometry ENU directly)
        self.decl = math.radians(float(dp("declared_heading_deg", 90.0).value))
        self.drift = None
        self.status = None
        self.nav_leg, self.pid = None, 0
        self.normal_last_attempt = None
        self.stats = dict(plans=0, replans=0, plan_errors=0, local_cells=0)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(String, dp("drift_topic", "/coursenav/drift").value, self._on_drift, 10)
        self.create_subscription(String, dp("status_topic", "/refnav/status").value, self._on_status, 10)
        self.create_subscription(OccupancyGrid, "/coursenav/occupancy", self._on_occ, latched)
        self.create_subscription(OccupancyGrid, "/coursenav/local_occupancy", self._on_local, 10)
        self.create_subscription(String, "/coursenav/prior_state", self._on_prior, latched)
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
            status = json.loads(m.data)
            if not isinstance(status, dict):
                return
            request = status.get("recovery_request")
            if isinstance(request, dict) and type(request.get("id")) is int:
                if request["id"] < self.recovery_last_id:
                    return
                if request["id"] == self.recovery_last_id and self.recovery is None:
                    return
            self.status = status
            if not isinstance(request, dict):
                self.recovery = None
                return
            rid, leg = request.get("id"), request.get("leg")
            if (type(rid) is not int or type(leg) is not int or rid < 0
                    or leg != status.get("leg") or request.get("observed_only") is not True):
                return
            if rid > self.recovery_last_id:
                self.recovery_last_id = rid
                self.recovery = dict(request_id=rid, leg=leg, plan_state="pending", reason="awaiting observed-free replan")
                self.recovery_reference = self.reference_path if self.reference_leg == leg else None
                self.recovery_prefix = None
                self.nav.path = None
                self.cache.pop(leg, None)  # a cached global path never satisfies a recovery request
        except ValueError:
            pass

    def _on_occ(self, m):
        if self.live is not None:
            self.live.on_grid(m)

    def _on_local(self, m):
        o = m.info.origin.position
        q = m.info.origin.orientation
        # The producer's contract is an axis-aligned nominal grid. Reject other
        # frames/rotations instead of silently certifying the wrong coordinates.
        if (m.header.frame_id != "coursenav_nominal" or
                not np.isfinite([q.x, q.y, q.z, q.w]).all() or
                abs(q.x) > 1e-8 or abs(q.y) > 1e-8 or abs(q.z) > 1e-8 or abs(abs(q.w)-1.) > 1e-8):
            self.observed_free = None
            return
        try:
            stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            grid = ObservedFree.from_grid((o.x, o.y), m.info.resolution,
                                         m.info.width, m.info.height, m.data, stamp)
            if self.observed_free is not None and stamp < self.observed_free.stamp:
                return  # delayed grids must not replace newer evidence
            self.local_pts = occupied_points_from_grid((o.x, o.y), m.info.resolution,
                                                       m.info.width, m.info.height, m.data)
            self.observed_free = grid
        except (ValueError, TypeError):
            self.observed_free = None

    def _on_prior(self, m):
        try:
            st = json.loads(m.data).get("state", "unknown")
        except ValueError:
            return
        if st not in ("placed", "dropped", "unknown"):
            st = "unknown"
        if st != self.prior_state:
            self.get_logger().warning(f"prior map {st}: local obstacles near prior surfaces "
                                      f"{'are again' if st == 'placed' else 'are no longer'} treated as mapped")
        self.prior_state = st

    def _recovery_plan(self, leg, believed, ekf, nb, now):
        r = self.recovery
        if r is None:
            return
        if r["leg"] != leg:
            self.nav.path = None
            r.update(plan_state="failed", reason="request leg is no longer active")
            return
        grid = self.observed_free
        to_nominal = lambda pts: ekf.to_nominal(pts, nb)
        if r["plan_state"] == "ready":
            if (grid is None or not grid.fresh(now, self.recovery_grid_max_age_s)
                    or self.nav.path is None
                    or self.recovery_prefix is None
                    or not grid.certifies_path(self.recovery_prefix, to_nominal, self.nav.need)):
                self.nav.path = None
                r.update(plan_state="failed", reason="recovery path lost fresh observed-free certification")
            return
        if r["plan_state"] != "pending":
            return  # exactly one attempt per explicit request id
        if (self.recovery_last_attempt is not None
                and now - self.recovery_last_attempt < self.recovery_min_interval_s):
            return
        self.recovery_last_attempt = now
        self.nav.path = None
        try:
            if not (0 <= leg < len(self.nav_legs)) or not self.nav_legs[leg]["path"]:
                raise PlanError("recovery request is not for a path leg")
            if grid is None or not grid.fresh(now, self.recovery_grid_max_age_s):
                raise PlanError("fresh observed-free local grid unavailable")
            path, prefix, join = self.nav.plan_recovery(
                believed[:2], self.recovery_reference, grid, to_nominal, t=time.monotonic(),
                lookahead_m=max(1., float(self.status.get("stop_m") or 0.)))
            if not grid.fresh(self.get_clock().now().nanoseconds * 1e-9, self.recovery_grid_max_age_s):
                raise PlanError("observed-free local grid expired during planning")
            self.recovery_prefix = prefix
            self.nav_leg = leg
            self.pid += 1
            self.stats["replans"] += 1
            r.update(plan_state="ready", reason="observed-free detour rejoins unchanged reference suffix",
                     certified_prefix_m=float(np.linalg.norm(np.diff(prefix, axis=0), axis=1).sum()),
                     rejoin_index=join)
        except PlanError as exc:
            self.nav.path = None
            self.stats["plan_errors"] += 1
            r.update(plan_state="failed", reason=str(exc))
            self.get_logger().warning(f"recovery request {r['request_id']}: {exc}")

    def _tick(self):
        if self.status is None:
            return
        if self.drift is None:
            if self.recovery is not None:
                self.pub.publish(String(data=json.dumps({"stamp": self.get_clock().now().nanoseconds * 1e-9,
                    "leg": self.recovery["leg"], "pid": self.pid, **self.recovery})))
            return
        d = self.drift
        ekf = DriftEKF.from_state(d["delta"], d["nominal"])
        nb = np.array(d["nominal"], float)
        believed = ekf.body_course(nb)
        if self.local_pts is not None:
            # local obstacles: nominal -> course with the SAME state as the believed pose (relative geometry
            # right even when the course pose is not); prior surfaces explain nearby cells only while placed
            explain = self.local_explain_m if self.prior_state == "placed" else 0.0
            mask = course_mask(ekf.to_course(self.local_pts, nb) if len(self.local_pts) else self.local_pts,
                               self.obst, self.surf_dist, explain)
            self.nav.set_local(mask)
            self.stats["local_cells"] = int(mask.sum())
        leg, phase = int(self.status.get("leg", 0)), self.status.get("phase")

        def to_ned(pts):
            return course_to_ned(ekf.to_nominal(pts, nb), d["fixture"], self.decl)

        targets = {j: [float(v) for v in to_ned([self.nav_legs[j]["stop"]])[0]]
                   for j in range(leg + 1, len(self.nav_legs))}
        now = self.get_clock().now().nanoseconds * 1e-9
        out = {"stamp": now, "leg": leg, "pid": self.pid, "targets_ne": targets,
               "request_id": None}
        if self.recovery is not None:
            self._recovery_plan(leg, believed, ekf, nb, now)
            out.update(self.recovery)
            out["recovery_certified"] = self.recovery["plan_state"] == "ready"
            if self.recovery["plan_state"] == "ready" and self.nav.path is not None:
                P = self.nav.path
                out.update(pid=self.pid, pts_ne=to_ned(P.pts).round(4).tolist(), v=P.v.round(3).tolist(),
                           stops=list(P.stops), tubes=None if P.tube is None else P.tube.round(4).tolist())
            self.pub.publish(String(data=json.dumps(out)))
            return
        if 0 <= leg < len(self.nav_legs) and self.nav_legs[leg]["path"] and phase in AIRBORNE:
            L = self.nav_legs[leg]
            want = self.nav_leg != leg or self.nav.path is None
            if not want and self.nav.path is not None:
                want = self.nav.needs_replan(time.monotonic(), 0.0)
            c = self.cache.pop(leg, None) if self.nav_leg != leg else None
            if want and c is not None and np.hypot(*(c[0] - believed[:2])) <= 0.5:
                self.nav.path, self.nav.goals, self.nav.t_plan = c[2], c[1], time.monotonic()
                self.pid += 1
                self.nav_leg = leg
                want = False
                self.get_logger().info(f"leg {leg + 1}: pre-planned path {self.nav.path.length:.2f} m")
            if want:
                if (self.normal_last_attempt is not None and self.normal_last_attempt[0] == leg
                        and now - self.normal_last_attempt[1] < max(0.1, self.nav.p.replan_every_s)):
                    self.nav.path = None
                    out.update(plan_state="failed", reason="awaiting bounded planner retry")
                    self.pub.publish(String(data=json.dumps(out)))
                    return
                self.normal_last_attempt = (leg, now)
                goals = [np.array(g, float) for g in L["goals"]]
                goals = [g for g in goals[:-1] if np.hypot(*(g - believed[:2])) > 0.3] + [goals[-1]]
                try:
                    v0 = float(self.status.get("v_cmd") or 0.0) if self.nav_leg == leg else 0.0
                    self.nav.plan(believed[:2], goals, t=time.monotonic(), v_start=v0)
                    self.stats["replans" if self.nav_leg == leg else "plans"] += 1
                    self.pid += 1
                    self.nav_leg = leg
                    self.get_logger().info(f"leg {leg + 1}: path {self.nav.path.length:.2f} m, "
                                           f"{len(self.nav.path.stops)} stop corners, est {self.nav.path.duration():.1f} s, "
                                           f"{self.stats['local_cells']} local obstacle cells, prior {self.prior_state}")
                except PlanError as e:
                    self.nav.path = None  # never keep streaming a path the new map invalidated
                    out.update(plan_state="failed", reason=str(e))
                    self.stats["plan_errors"] += 1
                    self.get_logger().warning(f"leg {leg + 1}: no path ({e}); refnav holds without one")
            if self.nav.path is not None and self.nav_leg == leg:
                P = self.nav.path
                self.reference_path, self.reference_leg = P, leg
                out.update(plan_state="ready", reason="regular map path")
                out.update(pid=self.pid, pts_ne=to_ned(P.pts).round(4).tolist(), v=P.v.round(3).tolist(),
                           stops=list(P.stops),
                           tubes=None if P.tube is None else P.tube.round(4).tolist())
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
