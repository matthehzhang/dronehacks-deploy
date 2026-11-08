"""Prior map + live depth occupancy -> the grid the replanner searches. PURE (numpy + scipy).

The PRIOR is the published course OBJ voxelised into the flight band (coursenav.course.plan_grids):
rule 6.4 allows a preloaded map built from the published layout. The LIVE layer is
coursenav.occgrid.LiveGrid: log-odds occupancy filled from this run's depth scans only, empty at
every launch (rule 6.5: nothing carried between runs).

Combined occupancy uses the prior band, live depth, and the enclosure boundary.
Adaptive mode can clear tagged prior surfaces with recent, repeated depth evidence.
Blocked = combined inflated by (vehicle radius + margin) | the prior keep-outs (boulders).
Clearance = metres from each cell to the nearest combined obstacle.

Net fence: a depth ray that passes through the net returns something outside the enclosure (the
lounge). Those returns are pulled back to the enclosure boundary, so the net reads as a wall and
nothing outside it is ever marked free.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy import ndimage

from ..coursenav.course import Grid2D, plan_grids
from ..coursenav.objmap import CourseMap
from ..coursenav.occgrid import LiveGrid, OccParams
from ..coursenav.scan import Scan


@dataclass
class NavParams:
    inflate_extra_m: float = 0.0      # on top of vehicle radius + margin from the course file
    hard_margin_m: float = 0.10       # validity checks: vehicle radius + this (planning uses the full 0.12 margin);
                                      # offline: 0.05 collided at 0.25-0.27 m with a 0.45 s velocity lag
    live_max_range_m: float = 3.5     # beyond this a return is not trusted (stereo error ~ r^2)
    live_occ_thresh: float = 1.5      # log-odds: >= 2 net hits (0.85 each) makes a cell an obstacle
    novel_min_m: float = 0.22         # a live obstacle counts only if no prior surface lies within this:
                                      # returns the map already explains are the map's walls, seen through a
                                      # start-pose error or VIO drift, and must not thicken them
    adaptive: bool = False
    evidence_age_s: float = 8.0
    rebuild_every_s: float = 0.5      # inflation/EDT recompute cadence while scans arrive


class NavGrid:
    def __init__(self, cmap: CourseMap, doc, params: NavParams = None):
        self.p = replace(params) if params is not None else NavParams()
        self.doc = doc
        self.obst, self.prior_blocked, self.prior_clearance = plan_grids(cmap, doc)
        v = doc["vehicle"]
        self.inflate_m = float(v["radius_m"]) + float(v["margin_m"]) + self.p.inflate_extra_m
        self.hard_inflate_m = float(v["radius_m"]) + self.p.hard_margin_m
        enc = doc.get("enclosure")
        self.enc_lo = np.asarray(enc["lo"], float) if enc else self.obst.origin.copy()
        self.enc_hi = np.asarray(enc["hi"], float) if enc else self.obst.origin + np.array(self.obst.occ.shape) * self.obst.res
        nx, ny = self.obst.occ.shape
        xy = self.obst.xy(np.stack(np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij"), -1))
        self.outside = np.any((xy < self.enc_lo) | (xy > self.enc_hi), axis=-1)
        occ_params = OccParams(max_range_m=self.p.live_max_range_m, occ_thresh=self.p.live_occ_thresh,
                               novel_min_m=self.p.novel_min_m, min_landmark_cells=3)
        self.live = LiveGrid(self.obst, occ_params)
        self._last_build = -math.inf
        self.last_seen = np.full(self.obst.occ.shape, -math.inf) if self.p.adaptive else None
        if self.p.adaptive:
            band = doc.get("plan_band_m", .25)
            z = doc["cruise_z_m"]
            prefixes = tuple(doc.get("adaptive_obstacle_groups", ()))
            mutable = cmap.band(z - band, z + band, only_prefixes=prefixes)
            fixed = cmap.band(z - band, z + band, exclude_prefixes=prefixes)
            self.mutable = mutable & ~fixed & ~self.outside
            self.keepout = np.zeros_like(mutable)
            for item in doc.get("keepout", []):
                cells = cmap.band(item["zlo"], item["zhi"], only_prefixes=item["groups"])
                if cells.any():
                    self.keepout |= ndimage.distance_transform_edt(~cells) * cmap.res < item["radius_m"]

        self._dirty = True
        self.blocked = None
        self.clearance = None
        self.rebuild(force=True)

    # ------------------------------------------------------------------ live updates
    def fence_clip(self, pose, scan: Scan) -> Scan:
        """Returns beyond the enclosure are moved onto its boundary (the net is a wall)."""
        if scan is None or len(scan) == 0:
            return scan
        x0, y0, th = float(pose[0]), float(pose[1]), float(pose[2])
        ang = th + scan.bearing
        ca, sa = np.cos(ang), np.sin(ang)
        tmax = np.full(len(ang), np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            for c, d, lo, hi in ((x0, ca, self.enc_lo[0], self.enc_hi[0]), (y0, sa, self.enc_lo[1], self.enc_hi[1])):
                t_lo = (lo - c) / d
                t_hi = (hi - c) / d
                t = np.where(d > 0, t_hi, np.where(d < 0, t_lo, np.inf))
                tmax = np.minimum(tmax, t)
        r = np.minimum(scan.range, np.maximum(tmax, 0.05))
        return Scan(scan.bearing, r, scan.sigma)

    def integrate(self, pose, scan: Scan, now: float) -> bool:
        """Add one level-body scan taken at course pose (x, y, yaw). Returns True if something changed."""
        scan = self.fence_clip(pose, scan)
        if self.last_seen is not None and scan is not None and len(scan):
            free, occupied = self.live.beam_cells(pose, scan)
            self.expire(now)
            self.live.scans_in += 1
            self.live.update_cells(free, occupied)
            self.last_seen.flat[free] = now
            self.last_seen.flat[occupied] = now
            ok = bool(len(free) or len(occupied))
        else:
            ok = self.live.integrate(pose, scan)
        if ok:
            self._dirty = True
        if self._dirty and now - self._last_build >= self.p.rebuild_every_s:
            self.rebuild(now=now)
        return ok

    def expire(self, now):
        """Discard old evidence even when new depth is empty or absent."""
        if self.last_seen is None:
            return
        expired = np.isfinite(self.last_seen) & (now - self.last_seen > self.p.evidence_age_s)
        if expired.any():
            self.live.L[expired] = 0
            self.live.seen[expired] = 0
            self.last_seen[expired] = -math.inf
            self._dirty = True

    def reset_evidence(self, now):
        self.live.L.fill(0)
        self.live.seen.fill(0)
        self.last_seen.fill(-math.inf)
        self.rebuild(now, force=True)

    def rebuild(self, now: float = 0.0, force: bool = False):
        self.expire(now)
        if not (self._dirty or force):
            return
        if self.p.adaptive:
            # Only tagged, removable geometry may be contradicted. A real return always wins.
            free = self.mutable & self.live.free() & (self.live.seen >= 4)
            occ = (self.obst.occ & ~free) | self.live.occupied() | self.outside
        else:
            occ = self.obst.occ | self.live.landmarks() | self.outside
        dist = ndimage.distance_transform_edt(~occ) * self.obst.res
        self.blocked = Grid2D(self.prior_blocked.occ | (dist < self.inflate_m), self.obst.origin, self.obst.res)
        self.blocked_hard = Grid2D(dist < self.hard_inflate_m, self.obst.origin, self.obst.res)
        self.clearance = np.minimum(self.prior_clearance, dist)
        if self.p.adaptive:
            self.blocked.occ[:] = self.keepout | (dist < self.inflate_m)
            self.blocked_hard.occ[:] |= self.keepout
            self.clearance = np.where(self.keepout, 0., dist)
        self.combined = Grid2D(occ, self.obst.origin, self.obst.res)
        self._last_build = now
        self._dirty = False

    # ------------------------------------------------------------------ queries
    def clearance_at(self, xy) -> float:
        ij = self.obst.ij(xy)
        if not self.obst.inside(ij):
            return 0.0
        return float(self.clearance[ij[0], ij[1]])

    def is_blocked(self, xy, hard=False) -> bool:
        g = self.blocked_hard if hard else self.blocked
        return bool(g.lookup(np.asarray(xy, float)[None])[0])

    def segment_blocked(self, p, q, step=0.025, hard=False) -> bool:
        g = self.blocked_hard if hard else self.blocked
        n = max(2, int(math.ceil(math.dist(p, q) / step)) + 1)
        return bool(g.lookup(np.linspace(p, q, n)).any())

    def nearest_free(self, xy, max_m=0.6, hard=False):
        """The blocked grid can swallow the vehicle's own cell when a live return lands close by.
        -> nearest unblocked cell centre within max_m, or None."""
        xy = np.asarray(xy, float)
        if not self.is_blocked(xy, hard):
            return xy
        r_cells = int(math.ceil(max_m / self.obst.res))
        c = self.obst.ij(xy)
        best, bd = None, np.inf
        occ = (self.blocked_hard if hard else self.blocked).occ
        for di in range(-r_cells, r_cells + 1):
            for dj in range(-r_cells, r_cells + 1):
                i, j = c[0] + di, c[1] + dj
                if 0 <= i < occ.shape[0] and 0 <= j < occ.shape[1] and not occ[i, j]:
                    p = self.obst.xy(np.array([i, j]))
                    d = math.dist(p, xy)
                    if d < bd:
                        best, bd = p, d
        return best if bd <= max_m else None

    def carved(self, xy, radius_m=None) -> Grid2D:
        """The planning grid with a disk around xy unblocked (except hard-blocked cells)."""
        r = self.inflate_m if radius_m is None else radius_m
        occ = self.blocked.occ.copy()
        c = self.obst.ij(np.asarray(xy, float))
        n = int(math.ceil(r / self.obst.res))
        i0, i1 = max(0, c[0] - n), min(occ.shape[0], c[0] + n + 1)
        j0, j1 = max(0, c[1] - n), min(occ.shape[1], c[1] + n + 1)
        ii, jj = np.mgrid[i0:i1, j0:j1]
        disk = np.hypot(ii - c[0], jj - c[1]) * self.obst.res <= r
        sub = occ[i0:i1, j0:j1]
        sub[disk] = self.blocked_hard.occ[i0:i1, j0:j1][disk]
        return Grid2D(occ, self.obst.origin, self.obst.res)

    def live_stats(self):
        stats = self.live.stats()
        if self.p.adaptive:
            stats["cleared_prior_cells"] = int(np.count_nonzero(self.mutable & ~self.combined.occ))
        return stats
