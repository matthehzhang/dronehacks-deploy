"""Waypoint intent + observed-depth geometry + declared room limits.

No origin-map obstacle input. Unknown cells are not walls, but are not
certified flight space either. Both the aircraft footprint and every smoothed
segment must fit in observed free space, clear of measured obstacles AND
inside the declared room boundary (nets may be invisible to depth, so the
enclosure is a separate hard constraint). An optional pilot-verified launch
disc is fixed in the odometry frame; it never overrides a measured obstacle
or the room boundary. PURE: no ROS or flight commands.

Observation poses (non-terminal horizons) must ADVANCE toward the active
waypoint under an optimistic cost-to-go: a grid distance in which measured
obstacles and the room boundary block, and unknown space is assumed passable.
That optimism is used only to rank where to look next; it never certifies a
cell for flight and never creates an obstacle.
"""
from dataclasses import replace
import math

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .course import Grid2D
from .planner import PlanError, astar
from .smooth import Path, build

_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


def grid_distances(passable, res, sources):
    """Shortest 8-neighbour distance (m) from weighted source cells, no diagonal corner cuts.

    sources: iterable of ((i, j), initial_cost_m). Impassable cells and
    unreachable cells are inf. Runs in C (scipy csgraph) on the whole grid.
    """
    ok = np.asarray(passable, bool)
    nx, ny = ok.shape
    n = nx*ny
    idx = np.arange(n).reshape(nx, ny)
    rows, cols, wts = [], [], []
    for di, dj in _STEPS:
        a = ok[max(0, -di):nx-max(0, di), max(0, -dj):ny-max(0, dj)]
        b = ok[max(0, di):nx-max(0, -di) or None, max(0, dj):ny-max(0, -dj) or None]
        e = a & b
        if di and dj:   # both orthogonal neighbours must be passable
            e &= ok[max(0, di):nx-max(0, -di) or None, max(0, -dj):ny-max(0, dj)]
            e &= ok[max(0, -di):nx-max(0, di), max(0, dj):ny-max(0, -dj) or None]
        src = idx[max(0, -di):nx-max(0, di), max(0, -dj):ny-max(0, dj)][e]
        rows.append(src)
        cols.append(src + di*ny + dj)
        wts.append(np.full(len(src), res*math.hypot(di, dj)))
    seeds = [(int(i)*ny+int(j), max(float(c), 1e-9)) for (i, j), c in sources
             if 0 <= i < nx and 0 <= j < ny and ok[i, j]]
    if not seeds:
        return np.full(ok.shape, math.inf)
    # One virtual node feeds every seed with its initial cost.
    rows.append(np.full(len(seeds), n))
    cols.append(np.array([s for s, _ in seeds]))
    wts.append(np.array([c for _, c in seeds]))
    g = csr_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(n+1, n+1))
    d = dijkstra(g, directed=True, indices=n)[:n].reshape(nx, ny)
    if len(seeds) == 1 and seeds[0][1] <= 1e-9:
        d[np.unravel_index(seeds[0][0], ok.shape)] = 0.
    return d


class DepthPlanner:
    def __init__(self, need, smooth, launch_xy=None, launch_radius=0., room=None,
                 min_progress=.15, frontier_band=.4):
        if not np.isfinite(launch_radius) or not 0 <= launch_radius <= 2.:
            raise ValueError('pilot-verified launch radius must be between 0 and 2 m')
        self.need, self.sp = float(need), replace(smooth, need_m=float(need))
        self.launch_xy = None if launch_xy is None else np.asarray(launch_xy, float)
        self.launch_radius = float(launch_radius)
        self.room = room
        self.min_progress, self.frontier_band = float(min_progress), float(frontier_band)
        self.path = None
        self.terminal = False
        self.visits = []
        self.reason = 'waiting for depth grid'
        self.diag = {}
        self.grid = self.blocked = self.clear = None
        self.known = self.free = None
        self.revision = 0
        self._room_key = None

    def _room_field(self, shape):
        """Signed room clearance at every cell centre (inf without a declared room)."""
        key = (tuple(self.grid.origin), shape, self.grid.res)
        if key != self._room_key:
            self._room_key = key
            if self.room is None:
                self._room_cells = np.full(shape, math.inf)
                self._room_pad = None
            else:
                xy = self.grid.xy(np.indices(shape).transpose(1, 2, 0))
                self._room_cells = self.room.clearance(xy)
                pij = np.indices((shape[0]+2, shape[1]+2)).transpose(1, 2, 0)-1
                self._room_pad = self.room.contains(self.grid.xy(pij))
        return self._room_cells

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
        raster = res*math.sqrt(2)/2       # cell-centre raster uncertainty, once
        # Pad boundaries: leaving the rolling sensor window is unknown.
        observed_clear = ndimage.distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1]*res
        obstacle_clear = (ndimage.distance_transform_edt(~occupied)*res if occupied.any()
                          else np.full(prob.shape, math.inf))
        room_clear = self._room_field(prob.shape)
        self.observed_clear = observed_clear - raster
        self.obstacle_clear = obstacle_clear - raster
        self.room_clear = room_clear - raster
        self.clear = np.minimum(np.minimum(observed_clear, obstacle_clear), room_clear) - raster
        blocked = self.clear < self.need
        if (self.blocked is None or not np.array_equal(self.blocked.origin, self.grid.origin)
                or not np.array_equal(self.blocked.occ, blocked) or not np.array_equal(self.free, free)):
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
        """Travel distance through certified cells from the aircraft."""
        return grid_distances(~self.blocked.occ, self.grid.res, [(tuple(self.grid.ij(start)), 0.)])

    def cost_to_go(self, goal):
        """Optimistic distance to the waypoint: measured obstacles and the room block, unknown passes.

        Ranking only. A waypoint outside the rolling window is reached through the
        window border at straight-line cost from there.
        """
        res = self.grid.res
        ok = np.minimum(self.obstacle_clear, self.room_clear) >= self.need
        shape = ok.shape
        gj = self.grid.ij(goal)
        if self.grid.inside(gj):
            cells = np.argwhere(ok)
            near = np.linalg.norm(self.grid.xy(cells)-goal, axis=1)
            pick = near <= max(.5, 2*res)
            seeds = [(tuple(c), float(r)) for c, r in zip(cells[pick], near[pick])]
        else:
            border = np.zeros(shape, bool)
            border[[0, -1], :] = border[:, [0, -1]] = True
            cells = np.argwhere(border & ok)
            near = np.linalg.norm(self.grid.xy(cells)-goal, axis=1)
            seeds = [(tuple(c), float(r)) for c, r in zip(cells, near)]
        return grid_distances(ok, res, seeds)

    def _frontier_distance(self):
        """Metres to the nearest unobserved cell INSIDE the room (space outside is never worth seeing)."""
        inside = self._room_cells > 0
        unknown = ~self.free & ~self.grid.occ & inside
        pad = np.pad(unknown, 1, constant_values=False)
        if self._room_pad is None:
            pad[[0, -1], :] = pad[:, [0, -1]] = True      # window edge: unknown beyond
        else:
            edge = np.zeros(pad.shape, bool)
            edge[[0, -1], :] = edge[:, [0, -1]] = True
            pad |= edge & self._room_pad
        if not pad.any():
            return np.full(unknown.shape, math.inf)
        return ndimage.distance_transform_edt(~pad)[1:-1, 1:-1]*self.grid.res

    def _goal_state(self, goal, dist):
        gj = self.grid.ij(goal)
        if not self.grid.inside(gj):
            return 'outside_depth_window'
        g = tuple(gj)
        if self.room_clear[g] < self.need:
            return 'room_boundary_clearance'
        if self.obstacle_clear[g] < self.need:
            return 'measured_obstacle_clearance'
        if np.isfinite(dist[g]):
            return 'reachable'
        return 'not_yet_observed_reachable'

    def plan(self, start, goal, v_start=0.):
        self.path = None
        if self.grid is None:
            raise PlanError('waiting_for_depth_grid')
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        ij = self.grid.ij(start)
        self.diag = dict(goal_xy=goal.round(3).tolist())
        if not self.grid.inside(ij):
            raise PlanError('start_footprint_not_observed_clear: outside depth window; no snapped escape connector')
        if self.blocked.occ[tuple(ij)]:
            c = tuple(ij)
            self.diag.update(start_room_m=round(float(self.room_clear[c]), 3),
                             start_observed_m=round(float(self.observed_clear[c]), 3))
            if self.room_clear[c] < self.need:
                raise PlanError('start_violates_room_boundary_clearance: hold; boundary is not depth-derived')
            why = ('measured obstacle' if self.obstacle_clear[c] < self.need else 'unobserved footprint coverage')
            raise PlanError(f'start_footprint_not_observed_clear: {why}; scan; no snapped escape connector')
        if self.room is not None and self.room.clearance(goal) < self.need:
            raise PlanError('waypoint_violates_room_boundary_clearance: waypoint intent inconsistent with room')
        dist = self._reachable(start)
        self.diag['goal_state'] = self._goal_state(goal, dist)
        self.terminal = self.diag['goal_state'] == 'reachable'
        if self.terminal:
            endpoint = goal
        else:
            endpoint = self._observation_pose(start, goal, dist)
        poly = astar(self.blocked, self.clear, start, endpoint)
        if np.linalg.norm(endpoint-start) <= .05:
            # Already in capture range: still provide a terminal path so the
            # follower can finish the waypoint, rather than scan indefinitely.
            length = float(np.linalg.norm(endpoint-start))
            path = Path(np.vstack([start, endpoint]), np.array([0., length]), np.zeros(2),
                        np.zeros(2), np.zeros(2), np.full(2, self.clear[tuple(ij)]))
        else:
            path = build(poly, self.grid, self.clear, self.sp, v_start=v_start)
        if not self._safe_points(path.pts):
            raise PlanError('smoothed_path_leaves_observed_clear_space')
        if path.length > .15 and np.interp(.1, path.s, path.v) <= 1e-6:
            raise PlanError('non_executable_zero_speed_prefix')
        self.path = path
        self.reason = 'waypoint_route' if self.terminal else 'observation_route_toward_waypoint'
        return path

    def _observation_pose(self, start, goal, dist):
        """A reachable pose near unobserved room space that reduces the optimistic cost-to-go."""
        h = self.cost_to_go(goal)
        h0 = float(h[tuple(self.grid.ij(start))])
        if not np.isfinite(h0):
            h0 = float(np.linalg.norm(goal-start))
            hx = None
        else:
            hx = h
        edge = self._frontier_distance()
        candidates = np.argwhere(np.isfinite(dist) & (dist >= .35) & (edge <= self.need+self.frontier_band))
        self.diag.update(cost_to_go_m=round(h0, 3), frontier_candidates=int(len(candidates)))
        if not len(candidates):
            raise PlanError('no_reachable_observation_pose: sweep toward waypoint')
        xy = self.grid.xy(candidates)
        togo = (hx[candidates[:, 0], candidates[:, 1]] if hx is not None
                else np.linalg.norm(xy-goal, axis=1))
        travel = dist[candidates[:, 0], candidates[:, 1]]
        progress = h0-togo
        for need in (self.min_progress, .05):
            keep = np.isfinite(togo) & (progress >= need)
            if keep.any():
                break
        else:
            self.diag['best_progress_m'] = (None if not np.isfinite(progress).any()
                                            else round(float(np.nanmax(np.where(np.isfinite(progress), progress, np.nan))), 3))
            raise PlanError('no_observation_pose_advances_toward_waypoint: sweep toward waypoint')
        xy, togo, travel, progress = xy[keep], togo[keep], travel[keep], progress[keep]
        score = togo+.4*travel
        for visit in self.visits[-30:]:
            score += 2.*np.exp(-np.linalg.norm(xy-visit, axis=1)**2/.25)
        k = int(np.argmin(score))
        self.diag.update(advancing_candidates=int(len(xy)), progress_m=round(float(progress[k]), 3),
                         endpoint_cost_to_go_m=round(float(togo[k]), 3))
        return xy[k]

    def reached_frontier(self, position, capture=.15):
        if self.path is not None and not self.terminal and np.linalg.norm(position-self.path.pts[-1]) <= capture:
            self.visits.append(self.path.pts[-1].copy())
            self.path = None
            return True
        return False
