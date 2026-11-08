"""Waypoint intent + observed-depth geometry. No origin-map obstacle input.

Unknown cells are not walls, but are not certified flight space either. Both
the aircraft footprint and every smoothed segment must fit in observed free
space. An optional pilot-verified launch disc is fixed in the odometry frame;
it never overrides a measured obstacle. PURE: no ROS or flight commands.
"""
from dataclasses import replace
import heapq
import math

import numpy as np
from scipy import ndimage

from .course import Grid2D
from .planner import PlanError, astar
from .smooth import Path, build


class DepthPlanner:
    def __init__(self, need, smooth, launch_xy=None, launch_radius=0.):
        if not np.isfinite(launch_radius) or not 0 <= launch_radius <= 2.:
            raise ValueError('pilot-verified launch radius must be between 0 and 2 m')
        self.need, self.sp = float(need), replace(smooth, need_m=float(need))
        self.launch_xy = None if launch_xy is None else np.asarray(launch_xy, float)
        self.launch_radius = float(launch_radius)
        self.path = None
        self.terminal = False
        self.visits = []
        self.reason = 'waiting for depth grid'
        self.grid = self.blocked = self.clear = None
        self.known = self.free = None
        self.revision = 0

    def update(self, origin, res, probabilities):
        prob = np.asarray(probabilities)
        if prob.ndim != 2 or not np.isfinite(prob).all() or not .01 <= res <= .1:
            raise ValueError('invalid local depth grid')
        occupied = prob >= 88
        free = (prob >= 0) & (prob <= 45)
        self.grid = Grid2D(occupied, np.asarray(origin, float), float(res))
        if self.launch_radius > 0 and self.launch_xy is not None:
            xy = self.grid.xy(np.indices(prob.shape).transpose(1, 2, 0))
            free |= np.linalg.norm(xy-self.launch_xy, axis=2) <= self.launch_radius
        free &= ~occupied
        # Pad boundaries: leaving the rolling sensor window is unknown.
        observed_clear = ndimage.distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1]*res
        obstacle_clear = (ndimage.distance_transform_edt(~occupied)*res if occupied.any()
                          else np.full(prob.shape, math.inf))
        self.observed_clear = observed_clear - res*math.sqrt(2)/2
        self.obstacle_clear = obstacle_clear - res*math.sqrt(2)/2
        # Cell-centre raster uncertainty is included once, conservatively.
        self.clear = np.minimum(observed_clear, obstacle_clear) - res*math.sqrt(2)/2
        blocked = self.clear < self.need
        if (self.blocked is None or not np.array_equal(self.blocked.origin,self.grid.origin)
                or not np.array_equal(self.blocked.occ,blocked) or not np.array_equal(self.free,free)):
            self.revision += 1
        self.free, self.known = free, prob >= 0
        self.blocked = Grid2D(blocked, self.grid.origin, res)

    def _safe_points(self, points):
        # Densify even long straight shortcuts; never validate vertices alone.
        pts = np.asarray(points)
        samples = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(1, int(math.ceil(np.linalg.norm(b-a)/(self.grid.res*.4))))
            samples.extend(a+(b-a)*t for t in np.linspace(0, 1, n+1)[1:])
        ij = self.grid.ij(samples)
        return bool(self.grid.inside(ij).all() and not self.blocked.occ[ij[:, 0], ij[:, 1]].any())

    def safe(self, position):
        if self.path is None or self.grid is None:
            return False
        first = max(0, int(np.argmin(np.linalg.norm(self.path.pts-position, axis=1)))-1)
        return self._safe_points(np.vstack([position, self.path.pts[first:]]))

    def _reachable(self, start):
        """Dijkstra on the same 8-neighbour grid, without diagonal corner cuts."""
        dist = np.full(self.blocked.occ.shape, math.inf)
        source = tuple(self.grid.ij(start))
        dist[source] = 0.
        queue = [(0., *source)]
        nx, ny = dist.shape
        while queue:
            d, i, j = heapq.heappop(queue)
            if d > dist[i, j]:
                continue
            for di, dj in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                a, b = i+di, j+dj
                if not (0 <= a < nx and 0 <= b < ny) or self.blocked.occ[a,b]:
                    continue
                if di and dj and (self.blocked.occ[i,b] or self.blocked.occ[a,j]):
                    continue
                nd = d+self.grid.res*math.hypot(di,dj)
                if nd < dist[a,b]:
                    dist[a,b] = nd
                    heapq.heappush(queue, (nd,a,b))
        return dist

    def plan(self, start, goal, v_start=0.):
        self.path = None
        if self.grid is None:
            raise PlanError('waiting_for_depth_grid')
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        ij = self.grid.ij(start)
        if not self.grid.inside(ij) or self.blocked.occ[tuple(ij)]:
            raise PlanError('start_footprint_not_observed_clear: scan; no snapped escape connector')
        dist = self._reachable(start)
        gj = self.grid.ij(goal)
        self.terminal = bool(self.grid.inside(gj) and np.isfinite(dist[tuple(gj)]))
        if self.terminal:
            endpoint = goal
        else:
            # Reachable observation poses just inside the known-space boundary.
            edge = ndimage.distance_transform_edt(np.pad(self.free,1))[1:-1,1:-1]*self.grid.res
            candidates = np.argwhere(np.isfinite(dist) & (dist >= .35)
                                     & (edge <= self.need+.4))
            if not len(candidates):
                raise PlanError('no_reachable_observation_pose: scan other headings')
            xy = self.grid.xy(candidates)
            travel = dist[candidates[:,0],candidates[:,1]]
            score = np.linalg.norm(xy-goal,axis=1)+.4*travel
            for visit in self.visits[-30:]:
                score += 2.*np.exp(-np.linalg.norm(xy-visit,axis=1)**2/.25)
            endpoint = xy[int(np.argmin(score))]
        poly = astar(self.blocked, self.clear, start, endpoint)
        if np.linalg.norm(endpoint-start) <= .05:
            # Already in capture range: still provide a terminal path so the
            # follower can finish the waypoint, rather than scan indefinitely.
            length = float(np.linalg.norm(endpoint-start))
            path = Path(np.vstack([start,endpoint]),np.array([0.,length]),np.zeros(2),
                        np.zeros(2),np.zeros(2),np.full(2,self.clear[tuple(ij)]))
        else:
            path = build(poly, self.grid, self.clear, self.sp, v_start=v_start)
        if not self._safe_points(path.pts):
            raise PlanError('smoothed_path_leaves_observed_clear_space')
        if path.length > .15 and np.interp(.1, path.s, path.v) <= 1e-6:
            raise PlanError('non_executable_zero_speed_prefix')
        self.path = path
        self.reason = 'waypoint_route' if self.terminal else 'exploration_route'
        return path

    def reached_frontier(self, position, capture=.15):
        if self.path is not None and not self.terminal and np.linalg.norm(position-self.path.pts[-1]) <= capture:
            self.visits.append(self.path.pts[-1].copy())
            self.path = None
            return True
        return False
