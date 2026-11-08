"""Depth-only receding-horizon ROS navigator. Global coordinates provide intent only."""
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
from .course import load_course
from .navplan import course_to_ned, smooth_params_for
from .planner import PlanError
from .localgrid import occupied_points_from_grid
from .depth_planner import DepthPlanner
from .course import resolve
from .room import RoomBoundary
AIRBORNE = ('hold', 'turn', 'check', 'go', 'blocked', 'backoff', 'recheck', 'align', 'search')

class NavNode(Node):

    def __init__(self):
        super().__init__('coursenav_nav')
        dp = self.declare_parameter
        route_file = dp('route_file', '').value
        if not route_file:
            raise SystemExit('coursenav.nav_node: give route_file:=<fly_course run record route.json>')
        with open(route_file) as f:
            rd = json.load(f)
        self.nav_legs = rd['nav_legs']
        doc = load_course()
        self.depth_only = True
        need = float(doc['vehicle']['radius_m'] + doc['vehicle']['margin_m'])
        sp = smooth_params_for(rd.get('navigation', {}), rd.get('params', {}), need)
        # Declared enclosure limits: a hard constraint independent of depth (nets may be invisible).
        self.room = RoomBoundary.load(resolve(doc, 'room_boundary_file')) if doc.get('room_boundary_file') else None
        self.depth_nav = DepthPlanner(need, sp, rd.get('start_xy'), float(dp('pilot_verified_launch_radius_m', 0.0).value),
                                      room=self.room)
        self.depth_plan_t = -math.inf
        self.depth_plan_revision = -1
        self.last_obstacle_stops = 0
        self.local_fields = None
        self.local_pts = np.zeros((0, 2))
        self.local_stamp = None
        self.local_required = bool(dp('local_map_required', True).value)
        self.local_error = 'waiting for ungated local map'
        self.decl = math.radians(float(dp('declared_heading_deg', 90.0).value))
        self.drift = None
        self.status = None
        self.nav_leg, self.pid = (None, 0)
        self.goal_progress = {}
        self.last_attempt = -math.inf
        self.last_plan_error = ''
        self.stats = dict(plans=0, replans=0, plan_errors=0)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(String, dp('drift_topic', '/coursenav/drift').value, self._on_drift, 10)
        self.create_subscription(String, dp('status_topic', '/refnav/status').value, self._on_status, 10)
        self.create_subscription(OccupancyGrid, '/coursenav/local_occupancy', self._on_local, 10)
        self.pub_grid = self.create_publisher(OccupancyGrid, '/coursenav/traversability', 10)
        self.create_timer(0.5, self._publish_grid)
        self.pub = self.create_publisher(String, dp('path_topic', '/coursenav/path').value, 10)
        self.create_timer(1.0 / float(dp('rate_hz', 10.0).value), self._tick)
        if self.room is None:
            self.get_logger().warning('navigator: NO room boundary declared; invisible nets are unconstrained')
        else:
            self.get_logger().info(f'navigator: room boundary {self.room.poly.round(3).tolist()} (hard constraint, not depth)')
        self.get_logger().info(f"navigator: {len(self.nav_legs)} legs ({sum((L['path'] for L in self.nav_legs))} path legs), cruise {sp.cruise_mps:.2f} m/s, buffer {need:.2f} m")
        from pathlib import Path
        Path(route_file).with_name('navigator_ready').write_text(json.dumps({'mode': 'depth_only', 'legs': []}))

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

    def _publish_grid(self):
        n = self.depth_nav
        if n.grid is None:
            return
        blocked = n.blocked
        m = OccupancyGrid()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'course'
        m.info.resolution = float(blocked.res)
        m.info.width, m.info.height = blocked.occ.shape
        m.info.origin.position.x, m.info.origin.position.y = map(float, blocked.origin)
        m.info.origin.orientation.w = 1.0
        # 100 (purple) is reserved for measured-obstacle inflation; 99 is the
        # declared room-boundary keep-out (not depth). Incomplete footprint
        # evidence remains unknown (-1), never displayed as an obstacle.
        data = np.full(blocked.occ.shape,-1,np.int8)
        data[~blocked.occ] = 0
        data[n.room_clear < n.need] = 99
        data[n.obstacle_clear < n.need] = 100
        m.data = data.T.ravel().tolist()
        self.pub_grid.publish(m)

    def _tick(self):
        if self.drift is not None and self.status is not None:
            self._tick_depth()

    def _on_local(self, m):
        try:
            captured = m.header.stamp.sec + m.header.stamp.nanosec * 1e-09
            now = self.get_clock().now().nanoseconds * 1e-09
            if m.header.frame_id != 'coursenav_nominal' or not -0.05 <= now - captured <= 0.5 or (self.local_stamp is not None and captured < self.local_stamp) or (not 0 < m.info.resolution <= 0.1) or (m.info.width * m.info.height != len(m.data)) or (not 0 < len(m.data) <= 250000):
                raise ValueError('invalid, stale or mismatched-frame local grid')
            origin = (m.info.origin.position.x, m.info.origin.position.y)
            if not np.isfinite(origin).all():
                raise ValueError('invalid grid origin')
            self.local_pts = occupied_points_from_grid(origin, m.info.resolution, m.info.width, m.info.height, m.data)
            self.local_fields = (origin, float(m.info.resolution), np.asarray(m.data, np.int16).reshape(m.info.height, m.info.width).T.copy())
            self.local_stamp, self.local_error = (captured, '')
        except (ValueError, TypeError) as exc:
            self.local_error = str(exc)

    def _tick_depth(self):
        """Receding observed-space horizon; origin supplies goal coordinates only."""
        d, n = (self.drift, self.depth_nav)
        now = self.get_clock().now().nanoseconds * 1e-09
        leg, phase = (int(self.status.get('leg', 0)), self.status.get('phase'))
        position = np.asarray(d['nominal'][:2], float)

        def to_ned(pts):
            return course_to_ned(pts, d['fixture'], self.decl)
        out = dict(stamp=time.time(), leg=leg, pid=self.pid, valid=False, terminal=False, map_provenance='vio_depth_only_waypoint_intent', mission_preview=[], targets_ne={j: to_ned([L['stop']])[0].tolist() for j, L in enumerate(self.nav_legs) if j > leg})
        if leg < len(self.nav_legs):
            out.update(goal_course=list(map(float, self.nav_legs[leg]['stop'])), goal_ne=to_ned([self.nav_legs[leg]['stop']])[0].tolist())
        if self.room is not None:
            out['room_boundary'] = dict(polygon_course=self.room.poly.round(4).tolist(), source='declared enclosure, not depth',
                                        aircraft_clearance_m=round(float(self.room.clearance(position)), 3),
                                        required_m=round(n.need, 3))
        fresh = self.local_stamp is not None and -0.05 <= now - self.local_stamp <= 0.5 and (not self.local_error)
        out.update(local_map_fresh=bool(fresh), local_map_error=self.local_error)
        if not fresh or self.local_fields is None:
            n.path = None
            out.update(reason='fresh_depth_grid_required', plan_error='fresh_depth_grid_required')
        elif leg < len(self.nav_legs) and phase in AIRBORNE:
            n.update(*self.local_fields)
            start_cell = n.grid.ij(position)
            if n.grid.inside(start_cell):
                out['start_clearance'] = dict(required_m=n.need,
                    room_m=float(n.room_clear[tuple(start_cell)]) if np.isfinite(n.room_clear[tuple(start_cell)]) else None,
                    observed_m=float(n.observed_clear[tuple(start_cell)]),
                    obstacle_m=None if not np.isfinite(n.obstacle_clear[tuple(start_cell)]) else float(n.obstacle_clear[tuple(start_cell)]))
            goal = np.asarray(self.nav_legs[leg]['stop'], float)
            consumed = self.status.get('horizon_reached') == dict(leg=leg, pid=self.pid)
            if consumed and n.path is not None and (not n.terminal):
                n.visits.append(n.path.pts[-1].copy())
                n.path = None
            reached = consumed or n.reached_frontier(position)
            obstacle_stops = int(self.status.get('obstacle_stops',0))
            guard_changed = obstacle_stops != self.last_obstacle_stops
            self.last_obstacle_stops = obstacle_stops
            changed = self.nav_leg != leg
            if changed:
                n.visits.clear()
            unsafe = n.path is not None and (not n.safe(position))
            if unsafe or changed or reached or guard_changed:
                n.path = None
            extend = n.path is not None and (not n.terminal) and (now - self.depth_plan_t >= 1.0) and (n.revision != self.depth_plan_revision)
            near_end = n.path is not None and np.linalg.norm(position - n.path.pts[-1]) < 0.65
            want = n.path is None or (extend and near_end)
            if want and now - self.depth_plan_t >= 0.5:
                self.depth_plan_t = now
                try:
                    n.plan(position, goal, v_start=float(self.status.get('v_cmd') or 0.0))
                    self.depth_plan_revision = n.revision
                    self.pid += 1
                    self.nav_leg = leg
                    self.last_plan_error = ''
                except PlanError as exc:
                    self.last_plan_error = str(exc)
            out.update(local_obstacle_cells=int(n.grid.occ.sum()), plan_error=self.last_plan_error, reason=self.last_plan_error or n.reason,
                       planner=dict(n.diag))
            if n.path is not None:
                P = n.path
                out.update(valid=True, pid=self.pid, terminal=bool(n.terminal), goal_course=goal.tolist(), horizon_course=P.pts[-1].tolist(), pts_course=P.pts.round(4).tolist(), pts_ne=to_ned(P.pts).round(4).tolist(), v=P.v.round(4).tolist(), stops=list(P.stops), tubes=None if P.tube is None else P.tube.round(4).tolist())
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
if __name__ == '__main__':
    main()
