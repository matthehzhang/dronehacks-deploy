"""Conservative, bounded, depth-only occupancy and observed-free A*.

All coordinates are metres in the current local ENU estimate. ``states`` is a
vertical stack: a 2-D cell is free only when every sampled flight-height layer
has been observed free. Unknown and expired observations are blocked. This is
a sampled slab, not proof against features below depth/map resolution.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import numpy as np
from scipy.ndimage import distance_transform_edt, minimum_filter

UNKNOWN, FREE, OCCUPIED = -1, 0, 100


@dataclass(frozen=True)
class GridParams:
    resolution_m: float = 0.05
    extent_m: float = 16.0
    radius_m: float = 0.294082845
    margin_m: float = 0.12
    half_height_m: float = 0.4
    layer_spacing_m: float = 0.10
    max_range_m: float = 4.0
    max_return_m: float = 12.0
    min_range_m: float = 0.30
    observation_ttl_s: float = 20.0
    occupied_clear_frames: int = 3
    depth_error_m: float = 0.08
    border_px: int = 24

    def __post_init__(self):
        if self.radius_m < 0.294082845 or self.margin_m < 0.12:
            raise ValueError('vehicle envelope and planning reserve may not be reduced')
        for k in ('resolution_m', 'extent_m', 'half_height_m', 'layer_spacing_m',
                  'max_range_m', 'max_return_m', 'min_range_m', 'observation_ttl_s', 'depth_error_m'):
            if not math.isfinite(getattr(self, k)) or getattr(self, k) <= 0:
                raise ValueError(k)
        if self.occupied_clear_frames < 3 or self.max_range_m <= self.min_range_m:
            raise ValueError('invalid range/clearing hysteresis')


class DepthGrid:
    def __init__(self, params=None):
        self.p = params or GridParams()
        self.n = int(math.ceil(self.p.extent_m / self.p.resolution_m))
        self.origin = np.array([-self.p.extent_m / 2] * 2, float)
        self.offsets = np.linspace(-self.p.half_height_m, self.p.half_height_m,
                                  1 + int(math.ceil(2 * self.p.half_height_m / self.p.layer_spacing_m)))
        self.states = np.full((len(self.offsets), self.n, self.n), UNKNOWN, np.int8)
        self.seen = np.full(self.states.shape, -math.inf)
        self.clear_count = np.zeros(self.states.shape, np.uint8)
        self.revision = 0
        self.changed_cells = np.empty((0, 2), int)
        self.changed_count = 0
        self.height = None
        self._cache = None
        self.frontier_details = []
        self.frontier_summary = {}
        self.launch_volume = None
        self._launch_mask = np.zeros(self.states.shape, bool)
        self._launch_signature = None
        self.planning_time = None

    @property
    def inflation_m(self):
        # Cell diagonal reserve ensures no interpolation squeezes between cells.
        return self.p.radius_m + self.p.margin_m + self.p.resolution_m * math.sqrt(2) / 2

    def reset(self, center_xy, height):
        self.origin = np.asarray(center_xy, float) - self.n * self.p.resolution_m / 2
        self.height = None if height is None else float(height)
        self.states.fill(UNKNOWN)
        self.seen.fill(-math.inf)
        self.clear_count.fill(0)
        self.frontier_details = []
        self.frontier_summary = {}
        self._launch_signature = None
        self._launch_mask.fill(False)
        self._changed(np.ones((self.n, self.n), bool))

    def set_launch_volume(self, volume, now):
        self.launch_volume = volume
        self.refresh_launch_volume(now)

    def refresh_launch_volume(self, now):
        """Separate operator evidence; never rewrite measured depth states."""
        self.planning_time = now
        status = {} if self.launch_volume is None else self.launch_volume.status(now)
        bounds = status.get('world_bounds') if status.get('active') else None
        signature = (None if bounds is None else tuple(bounds), self.height, tuple(self.origin))
        if signature == self._launch_signature:
            return
        previous = self._launch_mask.copy()
        self._launch_mask.fill(False)
        if bounds is not None and self.height is not None:
            lo, hi = np.asarray(bounds[:3]), np.asarray(bounds[3:])
            # Query only a small bounding box. The contract validates the
            # complete voxel (including height sampling reserve) itself.
            a, b = self.cell(lo[:2]), self.cell(hi[:2])
            rows=np.arange(max(0,a[0]), min(self.n,b[0]+1))
            cols=np.arange(max(0,a[1]), min(self.n,b[1]+1))
            if len(rows) and len(cols):
                layers, rr, cc=np.meshgrid(np.arange(len(self.offsets)),rows,cols,indexing='ij')
                points=np.column_stack([self.origin[0]+(cc.ravel()+.5)*self.p.resolution_m,
                    self.origin[1]+(rr.ravel()+.5)*self.p.resolution_m,self.height+self.offsets[layers.ravel()]])
                allowed=self.launch_volume.unknown_mask(points,now,
                    (self.p.resolution_m/2,self.p.resolution_m/2,self.p.layer_spacing_m/2))
                self._launch_mask[layers.ravel(),rr.ravel(),cc.ravel()]=allowed
        if np.any(self._launch_mask & (self.states == OCCUPIED)):
            self.launch_volume.revoke('existing measured obstacle contradicts pilot-verified launch volume')
            self._launch_mask.fill(False)
            signature = (None, self.height, tuple(self.origin))
        self._launch_signature = signature
        self._changed(np.any(previous != self._launch_mask, axis=0))

    def planning_collapsed(self):
        occupied = np.any(self.states == OCCUPIED, axis=0)
        usable = np.all((self.states == FREE) | ((self.states == UNKNOWN) & self._launch_mask), axis=0)
        return np.where(occupied,OCCUPIED,np.where(usable,FREE,UNKNOWN)).astype(np.int8)

    def _revoke_contradicted_launch_volume(self, occupied, now):
        if self.launch_volume is not None and np.any(occupied & self._launch_mask):
            self.launch_volume.revoke('depth obstacle contradicts pilot-verified launch volume')
            self.refresh_launch_volume(now)

    def _changed(self, mask):
        if mask.any():
            self.revision += 1
            self.changed_cells = np.argwhere(mask)
            self.changed_count = int(mask.sum())
            self._cache = None

    def cell(self, xy):
        a = np.floor((np.asarray(xy, float) - self.origin) / self.p.resolution_m).astype(int)
        return int(a[1]), int(a[0])

    def inside(self, ij):
        return 0 <= ij[0] < self.n and 0 <= ij[1] < self.n

    def xy(self, ij):
        return self.origin + (np.array([ij[1], ij[0]], float) + 0.5) * self.p.resolution_m

    def collapsed(self):
        occupied = np.any(self.states == OCCUPIED, axis=0)
        free = np.all(self.states == FREE, axis=0)
        return np.where(occupied, OCCUPIED, np.where(free, FREE, UNKNOWN)).astype(np.int8)

    def navigation(self):
        if self._cache is None:
            observed = self.planning_collapsed() == FREE
            # The map boundary itself is blocked, including an all-free test map.
            observed[[0, -1], :] = False
            observed[:, [0, -1]] = False
            clearance = distance_transform_edt(observed) * self.p.resolution_m
            safe = observed & (clearance > self.inflation_m)
            self._cache = safe, clearance
        return self._cache

    def expire(self, now):
        self.refresh_launch_volume(now)
        old = (self.states != UNKNOWN) & (now - self.seen > self.p.observation_ttl_s)
        if old.any():
            self.states[old] = UNKNOWN
            self.clear_count[old] = 0
            self._changed(old.any(axis=0))

    def integrate(self, depth_m, K, R_world_cam, t_world_cam, now):
        """Project slab centres into depth, without clearing invalid/no-return rays.

        R/t are capture-time calibrated camera pose; optical x-right,y-down,z-forward.
        An endpoint splat inserts every valid sampled surface at flight height.
        Occupancy wins within a frame and needs three distinct free frames to clear.
        Returns the number of CHANGED voxels, not a net occupancy-count difference.
        """
        if self.height is None:
            return 0
        d = np.asarray(depth_m, float)
        if d.ndim != 2 or not np.isfinite(np.asarray(K)).all():
            raise ValueError('invalid calibrated depth')
        fx, fy, cx, cy = K
        if fx <= 0 or fy <= 0:
            raise ValueError('invalid focal length')
        R, t = np.asarray(R_world_cam, float), np.asarray(t_world_cam, float)
        if R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all() or not np.isfinite(t).all():
            raise ValueError('invalid camera pose')
        h, w = d.shape
        b = self.p.border_px
        valid = np.isfinite(d) & (d >= self.p.min_range_m) & (d <= self.p.max_return_m)
        valid[:b] = False
        valid[-b:] = False
        valid[:, :b] = False
        valid[:, -b:] = False
        # All 3x3 support pixels must be valid; use closest surface conservatively.
        support = minimum_filter(valid.astype(np.uint8), size=3, mode='constant') > 0
        nearest = minimum_filter(np.where(valid, d, np.inf), size=3, mode='constant', cval=0)
        i0, j0 = self.cell(t[:2] - self.p.max_range_m)
        i1, j1 = self.cell(t[:2] + self.p.max_range_m)
        rows = np.arange(max(0, i0), min(self.n, i1 + 1))
        cols = np.arange(max(0, j0), min(self.n, j1 + 1))
        if not len(rows) or not len(cols):
            return 0
        zz, ii, jj = np.meshgrid(np.arange(len(self.offsets)), rows, cols, indexing='ij')
        zix, iy, ix = zz.ravel(), ii.ravel(), jj.ravel()
        points = np.column_stack([self.origin[0] + (ix + .5) * self.p.resolution_m,
                                  self.origin[1] + (iy + .5) * self.p.resolution_m,
                                  self.height + self.offsets[zix]])
        pc = (points - t) @ R
        # A valid farther return observes its traversed ray from the lens onward.
        # min_range rejects RETURN depths; it must not turn those traversed rays
        # into an artificial blind obstacle surrounding the camera.
        infront = (pc[:, 2] > .02) & (pc[:, 2] < self.p.max_range_m)
        denom = np.maximum(pc[:, 2], 1e-6)
        u = np.rint(fx * pc[:, 0] / denom + cx).astype(int)
        v = np.rint(fy * pc[:, 1] / denom + cy).astype(int)
        visible = infront & (u >= b) & (u < w - b) & (v >= b) & (v < h - b)
        ids = np.flatnonzero(visible)
        ids = ids[support[v[ids], u[ids]]]
        if not len(ids):
            return 0
        sd = nearest[v[ids], u[ids]]
        free_ids = ids[pc[ids, 2] < sd - self.p.depth_error_m]
        occ_ids = ids[np.abs(pc[ids, 2] - sd) <= self.p.depth_error_m]
        free = np.zeros_like(self.states, bool)
        occ = np.zeros_like(self.states, bool)
        free[zix[free_ids], iy[free_ids], ix[free_ids]] = True
        occ[zix[occ_ids], iy[occ_ids], ix[occ_ids]] = True
        # Surface splat catches surfaces that lie between slab-cell centres.
        occ |= self._endpoint_mask(d, K, R, t, valid)
        return self.apply_observation(free, occ, now)

    def _endpoint_mask(self, d, K, R, t, valid=None):
        occ = np.zeros_like(self.states, bool)
        if self.height is None:
            return occ
        fx, fy, cx, cy = K
        h, w = d.shape
        if valid is None:
            valid = np.isfinite(d) & (d >= self.p.min_range_m) & (d <= self.p.max_return_m)
            b = self.p.border_px
            valid[:b] = False
            valid[-b:] = False
            valid[:, :b] = False
            valid[:, -b:] = False
        vs, us = np.mgrid[0:h:3, 0:w:3]
        ok = valid[vs, us] & (d[vs, us] <= self.p.max_range_m)
        ds = d[vs[ok], us[ok]]
        cp = np.column_stack([(us[ok] - cx) / fx * ds, (vs[ok] - cy) / fy * ds, ds])
        wp = cp @ R.T + t
        xy = np.floor((wp[:, :2] - self.origin) / self.p.resolution_m).astype(int)
        layer = np.argmin(np.abs(wp[:, 2, None] - (self.height + self.offsets)[None]), axis=1)
        ok = ((np.abs(wp[:, 2] - self.height) <= self.p.half_height_m + self.p.resolution_m / 2)
              & (xy[:, 0] >= 0) & (xy[:, 0] < self.n) & (xy[:, 1] >= 0) & (xy[:, 1] < self.n))
        occ[layer[ok], xy[ok, 1], xy[ok, 0]] = True
        return occ

    def integrate_occupied(self, depth_m, K, R_world_cam, t_world_cam, now):
        """Insert newly observed hits between slower full free-space projections.

        This lightweight update never clears space or resets other voxels'
        clearing hysteresis. A fresh obstacle changes the planner revision at
        sensor cadence even when the expensive free-volume projection is capped.
        """
        occ = self._endpoint_mask(np.asarray(depth_m, float), K,
                                  np.asarray(R_world_cam, float), np.asarray(t_world_cam, float))
        occ &= now >= self.seen
        self._revoke_contradicted_launch_volume(occ,now)
        changed = occ & (self.states != OCCUPIED)
        self.states[occ] = OCCUPIED
        self.seen[occ] = now
        self.clear_count[occ] = 0
        self._changed(changed.any(axis=0))
        return int(changed.sum())

    def apply_observation(self, free, occupied, now):
        """Sensor-evidence entry point; exposed for deterministic replays/tests."""
        # Height-slab rebuilding may replay older calibrated views while new
        # depth is arriving. Old free rays must never clear a newer obstacle.
        eligible = now > self.seen
        free = np.asarray(free, bool) & ~np.asarray(occupied, bool) & eligible
        occupied = np.asarray(occupied, bool) & (now >= self.seen)
        self._revoke_contradicted_launch_volume(occupied,now)
        before = self.states.copy()
        self.seen[free | occupied] = now
        was_occ = self.states == OCCUPIED
        self.clear_count[free & was_occ] = np.minimum(254, self.clear_count[free & was_occ] + 1)
        self.clear_count[~free & eligible] = 0
        cleared = free & (~was_occ | (self.clear_count >= self.p.occupied_clear_frames))
        self.states[cleared] = FREE
        self.states[occupied] = OCCUPIED
        self.clear_count[occupied] = 0
        changed = before != self.states
        self._changed(changed.any(axis=0))
        return int(changed.sum())

    def segment_safe(self, a, b, safe=None):
        safe = self.navigation()[0] if safe is None else safe
        a, b = np.asarray(a, float), np.asarray(b, float)
        steps = max(1, int(math.ceil(np.linalg.norm(b - a) / (self.p.resolution_m * .25))))
        for point in np.linspace(a, b, steps + 1):
            ij = self.cell(point)
            if not self.inside(ij) or not safe[ij]:
                return False
        return True

    def route_safe(self, points, safe=None):
        points = list(points)
        return bool(points) and all(self.segment_safe(a, b, safe) for a, b in
                                    (zip(points, points[1:]) if len(points) > 1 else [(points[0], points[0])]))

    def route_rejection(self, start_xy, goal_xy):
        safe, _ = self.navigation()
        for label, xy in (('start', start_xy), ('goal', goal_xy)):
            ij = self.cell(xy)
            if not self.inside(ij):
                return f'{label} outside fresh bounded map'
            if self.planning_collapsed()[ij] == UNKNOWN:
                return f'{label} is unknown at required flight-height slab'
            if not safe[ij]:
                return f'{label} full vehicle envelope intersects occupied/unknown cells'
        return 'no connected observed-safe path'

    def summary(self, position_xy=None):
        raw = self.collapsed()
        safe, clearance = self.navigation()
        out = dict(provenance='calibrated_depth_only', prior_loaded=False,
                   resolution_m=self.p.resolution_m, vertical_layers=len(self.offsets),
                   vertical_offsets_m=self.offsets.tolist(), inflation_m=self.inflation_m,
                   revision=self.revision, changed_cells_last_revision=self.changed_count,
                   unknown_cells=int(np.count_nonzero(raw == UNKNOWN)),
                   free_cells=int(np.count_nonzero(raw == FREE)),
                   occupied_cells=int(np.count_nonzero(raw == OCCUPIED)),
                   traversable_cells=int(safe.sum()))
        out['pilot_volume_only_cells'] = int(np.count_nonzero((raw==UNKNOWN)&(self.planning_collapsed()==FREE)))
        out['planning_provenance'] = ('calibrated_depth_plus_explicit_pilot_launch_volume'
                                     if self._launch_mask.any() else 'calibrated_depth_only')
        if position_xy is not None:
            ij = self.cell(position_xy)
            out['start_inside_map'] = self.inside(ij)
            out['start_traversable'] = bool(self.inside(ij) and safe[ij])
            out['start_clearance_m'] = float(clearance[ij]) if self.inside(ij) else 0.
            if self.inside(ij):
                radius = int(math.ceil(self.inflation_m / self.p.resolution_m))
                rows, cols = np.mgrid[max(0, ij[0]-radius):min(self.n, ij[0]+radius+1),
                                      max(0, ij[1]-radius):min(self.n, ij[1]+radius+1)]
                disc = (rows-ij[0])**2 + (cols-ij[1])**2 <= radius**2
                values = raw[rows[disc], cols[disc]]
                out['start_footprint_unknown_cells'] = int(np.count_nonzero(values == UNKNOWN))
                out['start_footprint_occupied_cells'] = int(np.count_nonzero(values == OCCUPIED))
                out['start_footprint_layers'] = [dict(z_m=float(self.height+offset),
                    unknown=int(np.count_nonzero(self.states[k,rows[disc],cols[disc]]==UNKNOWN)),
                    occupied=int(np.count_nonzero(self.states[k,rows[disc],cols[disc]]==OCCUPIED)),
                    pilot_verified_unknown=int(np.count_nonzero(self._launch_mask[k,rows[disc],cols[disc]] &
                        (self.states[k,rows[disc],cols[disc]]==UNKNOWN))))
                    for k,offset in enumerate(self.offsets)] if self.height is not None else []
        return out

    def astar(self, start_xy, goal_xy):
        safe = self.navigation()[0]
        start, goal = self.cell(start_xy), self.cell(goal_xy)
        if not self.inside(start) or not self.inside(goal) or not safe[start] or not safe[goal]:
            return None
        queue = [(0., start)]
        costs, parents = {start: 0.}, {}
        while queue:
            _, cur = heapq.heappop(queue)
            if cur == goal:
                path = [cur]
                while cur != start:
                    cur = parents[cur]
                    path.append(cur)
                xy = [np.asarray(start_xy, float)] + [self.xy(x) for x in path[::-1][1:-1]] + [np.asarray(goal_xy, float)]
                return self.smooth(xy, safe)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nxt = cur[0] + di, cur[1] + dj
                if not self.inside(nxt) or not safe[nxt]:
                    continue
                if di and dj and (not safe[cur[0] + di, cur[1]] or not safe[cur[0], cur[1] + dj]):
                    continue
                g = costs[cur] + math.hypot(di, dj)
                if g < costs.get(nxt, math.inf):
                    costs[nxt], parents[nxt] = g, cur
                    heapq.heappush(queue, (g + math.dist(nxt, goal), nxt))
        return None

    def smooth(self, points, safe=None):
        """Prune only completely checked swept-envelope segments; never spline across a corner."""
        if len(points) < 3:
            return points
        out, i = [points[0]], 0
        while i < len(points) - 1:
            j = len(points) - 1
            while j > i + 1 and not self.segment_safe(points[i], points[j], safe):
                j -= 1
            out.append(points[j])
            i = j
        return out

    def reachable(self, start_xy):
        safe = self.navigation()[0]
        start = self.cell(start_xy)
        dist = np.full((self.n, self.n), np.inf)
        if not self.inside(start) or not safe[start]:
            return dist
        q = [(0., start)]
        dist[start] = 0.
        while q:
            d, cur = heapq.heappop(q)
            if d > dist[cur]:
                continue
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ij = cur[0] + di, cur[1] + dj
                if self.inside(ij) and safe[ij] and d + self.p.resolution_m < dist[ij]:
                    dist[ij] = d + self.p.resolution_m
                    heapq.heappush(q, (dist[ij], ij))
        return dist

    def _frontier_views(self, point, raw):
        """Rank possible observation yaws by first-unknown rays, respecting walls.

        This is a bounded 2-D visibility heuristic, not a claim that unknown
        volume is empty. No ray changes occupancy. Motion still requires the
        full observed-free slab and calibrated current-depth corridor guard.
        """
        headings = np.linspace(-math.pi, math.pi, 12, endpoint=False)
        ray_offsets = np.linspace(-math.pi / 6, math.pi / 6, 25)
        angles = headings[:, None] + ray_offsets
        distances = np.arange(self.p.resolution_m / 2,
                              min(2.5, self.p.max_range_m), self.p.resolution_m / 2)
        xx = point[0] + np.cos(angles)[..., None] * distances
        yy = point[1] + np.sin(angles)[..., None] * distances
        cols = np.floor((xx - self.origin[0]) / self.p.resolution_m).astype(int)
        rows = np.floor((yy - self.origin[1]) / self.p.resolution_m).astype(int)
        inside = (rows >= 0) & (rows < self.n) & (cols >= 0) & (cols < self.n)
        values = raw[np.clip(rows, 0, self.n - 1), np.clip(cols, 0, self.n - 1)]
        stop = ~inside | (values != FREE)
        first = stop.argmax(axis=-1)
        first_values = np.take_along_axis(values, first[..., None], axis=-1)[..., 0]
        first_inside = np.take_along_axis(inside, first[..., None], axis=-1)[..., 0]
        unknown_rays = stop.any(axis=-1) & first_inside & (first_values == UNKNOWN)
        counts = unknown_rays.sum(axis=-1)
        best = int(counts.argmax())
        return float(counts[best]) / len(ray_offsets), float(headings[best]), int(counts[best])

    def frontiers(self, start_xy, visited=(), limit=12, *, heading=None, speed_mps=.4):
        """Reachable observation points: visibility gain versus travel/turn cost.

        Candidate points lie inside observed-safe space, back from unknown.
        A cheap unknown-neighbourhood filter supplies a bounded, spatially
        diverse pool. Visibility then stops at known walls and the FIRST unknown
        cell, so an unseen room behind a solid wall cannot create false gain.
        """
        self.frontier_details = []
        self.frontier_summary = dict(strategy='occlusion_checked_frontiers', candidates=0,
                                     evaluated=0, occluded=0, selected=0)
        raw = self.planning_collapsed()
        safe, clearance = self.navigation()
        travel = self.reachable(start_xy)
        if not np.isfinite(travel).any():
            self.frontier_summary['reason'] = self.route_rejection(start_xy, start_xy)
            return []
        unknown = raw == UNKNOWN
        from scipy.ndimage import uniform_filter
        gain = uniform_filter(unknown.astype(float), size=max(3, int(2. / self.p.resolution_m)))
        unknown_distance = distance_transform_edt(~unknown) * self.p.resolution_m
        options = np.argwhere(safe & np.isfinite(travel) & (travel > .35)
                              & (unknown_distance < 1.2) & (gain > .025))
        self.frontier_summary['candidates'] = len(options)
        candidates = []
        for ij in options[::max(1, int(.2 / self.p.resolution_m))]:
            point = self.xy(ij)
            if any(np.linalg.norm(point - np.asarray(v)) < .45 for v in visited):
                continue
            proxy = float(gain[tuple(ij)]) * 6. - float(travel[tuple(ij)]) * .30
            candidates.append((proxy, point, tuple(ij)))
        candidates.sort(key=lambda x: x[0], reverse=True)
        pool = []
        for _, point, ij in candidates:
            if all(np.linalg.norm(point - other[0]) >= .25 for other in pool):
                pool.append((point, ij))
            if len(pool) >= 64:
                break
        ranked = []
        for point, ij in pool:
            visible_gain, view_yaw, rays = self._frontier_views(point, raw)
            self.frontier_summary['evaluated'] += 1
            if not rays:
                self.frontier_summary['occluded'] += 1
                continue
            travel_m = float(travel[ij])
            travel_s = travel_m / max(float(speed_mps), .05)
            bearing = math.atan2(*(point - np.asarray(start_xy))[::-1])
            turn_s = 0. if heading is None else abs((bearing-heading+math.pi) % (2*math.pi)-math.pi) / math.radians(25.)
            extra = max(0., float(clearance[ij]) - self.inflation_m)
            score = visible_gain * 8. - travel_s * .20 - turn_s * .08 + min(extra, .5) * .30
            detail = dict(score=score, xy=point.tolist(), visible_unknown_fraction=visible_gain,
                          visible_unknown_rays=rays, ray_count=25, view_heading_rad=view_yaw,
                          travel_m=travel_m, estimated_travel_s=travel_s,
                          estimated_turn_s=turn_s, extra_clearance_m=extra)
            ranked.append((score, point, detail))
        ranked.sort(key=lambda x: x[0], reverse=True)
        selected = []
        for score, point, detail in ranked:
            if all(np.linalg.norm(point - other[1]) > .45 for other in selected):
                selected.append((score, point))
                self.frontier_details.append(detail)
            if len(selected) >= limit:
                break
        self.frontier_summary['selected'] = len(selected)
        return selected
