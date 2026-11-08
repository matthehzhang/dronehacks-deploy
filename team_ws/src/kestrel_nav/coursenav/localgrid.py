"""The LOCAL obstacle grid: what depth saw around the aircraft in the last seconds, in the nominal (odometry)
frame, NEVER gated on the localizer. PURE: no ROS.

Why (bag 20261107-201744, real drone): the localizer's scan fit stayed at 10-20 % (chance level) all flight,
so it never updated, its sigma grew past the live map's gate, and the live map (coursenav.occgrid, course
frame, gated on sigma <= 6 cm / 2 deg) stayed empty. Refnav blocked 18 times on an obstacle the navigator
never knew about. Obstacle avoidance does not need the course position: it needs where the obstacle is
relative to the aircraft, which odometry gives to centimetres over the seconds that matter.

  frame    "nominal": the VIO/PX4 frame placed at the declared start fixture (loc_node: compose(nominal,
           vio_pose)). A rigid copy of the odometry frame: the localizer never moves it. The navigator turns
           nominal points into the course frame with the drift EKF it already uses for the aircraft's own
           believed pose, so obstacles stay right RELATIVE to the aircraft even when the course pose is wrong.
  window   size_m x size_m around the aircraft, res_m cells; recentred (cells kept at their world place)
           when the aircraft has moved more than recentre_m from the window centre.
  update   per scan: log-odds +l_occ at each return cell, l_free along each beam up to one cell short of it
           (no return = unknown, never free); valid distant returns carve only within max_range_m, with a 3-sigma margin;
           distant endpoints are never marked occupied. Free-only beams (scan.free_only, the dense free rays)
           carve the same way and never mark their endpoint. free_range_m > max_range_m lets distant returns
           carve further than obstacles are marked.
  body     body_radius_m > 0: cells wholly inside the airframe's disc at an integrated pose are recorded; for
           swept_age_s they are published free (the aircraft was physically there), unless a confident
           obstacle return lands on them afterwards. This is what makes the aircraft's own position, which
           a forward camera never sees, certifiable for a recovery hop.
  decay   every cell relaxes toward unknown with time constant decay_s, so a cell nothing sees any more
           fades (a person who walked off, a stereo artefact); one at l_max stays occupied for
           decay_s * ln(l_max / occ_thresh) after the last return (17 s with the defaults): long enough to
           remember an obstacle the camera turned away from during a detour.
  occupied L >= occ_thresh in a connected clump of >= min_cells cells (specks are stereo noise).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .scan import Scan


@dataclass
class LocalParams:
    res_m: float = 0.05
    size_m: float = 8.0              # window side
    recentre_m: float = 1.5          # recentre when the aircraft is this far from the window centre
    max_range_m: float = 3.0         # do not mark or carve beyond this
    l_occ: float = 0.85              # same evidence weights as coursenav.occgrid
    l_free: float = -0.40
    l_min: float = -2.0
    l_max: float = 3.5
    occ_thresh: float = 2.0          # >= 3 net returns more than pass-throughs
    decay_s: float = 30.0            # log-odds time constant toward unknown
    min_cells: int = 3               # smallest clump that counts as an obstacle
    free_range_m: float = 0.0        # > 0: distant valid returns carve free space out to this (else max_range_m);
                                     # obstacle marking still stops at max_range_m
    body_radius_m: float = 0.0       # > 0: the airframe's own disc (prop tips) at each integrated pose is free:
                                     # the aircraft physically occupies it, so nothing else can be there
    swept_age_s: float = 0.0         # ... and stays published as free this long after the aircraft left it
                                     # (unless a confident obstacle return lands there since)


class LocalGrid:
    def __init__(self, p: LocalParams = None):
        self.p = p or LocalParams()
        self.n = int(round(self.p.size_m / self.p.res_m))
        self.L = np.zeros((self.n, self.n), np.float32)
        self.seen = np.zeros((self.n, self.n), np.uint16)
        # Per-cell pass-through evidence, separate from publication/decay time.
        # -inf means never free; a sensor observation at simulation time zero is valid.
        self.last_free = np.full((self.n, self.n), -np.inf, np.float64)
        # Per-cell time the airframe's own disc last covered the cell (body_radius_m > 0).
        self.last_body = np.full((self.n, self.n), -np.inf, np.float64)
        self.origin = None                 # world (nominal) xy of cell (0, 0)'s corner; cell-aligned
        self.t_last = None
        self.scans_in = 0
        self.recentres = 0

    # ------------------------------------------------------------------ geometry
    def _centre_on(self, xy):
        """Place the window so xy is at its centre, keeping every cell at its world place (cell-aligned)."""
        res = self.p.res_m
        new = np.floor((np.asarray(xy, float) - self.p.size_m / 2) / res) * res
        if self.origin is None:
            self.origin = new
            return
        shift = np.rint((new - self.origin) / res).astype(int)      # cells the window moves by
        if not shift.any():
            return
        self.L = _shift(self.L, shift)
        self.seen = _shift(self.seen, shift)
        self.last_free = _shift(self.last_free, shift, fill=-np.inf)
        self.last_body = _shift(self.last_body, shift, fill=-np.inf)
        self.origin = self.origin + shift * res
        self.recentres += 1

    def centre(self):
        return None if self.origin is None else self.origin + self.p.size_m / 2

    def ij(self, xy):
        return np.floor((np.atleast_2d(np.asarray(xy, float)) - self.origin) / self.p.res_m).astype(int)

    def cell_centres(self, ij):
        return self.origin + (np.asarray(ij, float) + 0.5) * self.p.res_m

    # ------------------------------------------------------------------ update
    def decay(self, t):
        if self.t_last is not None and t > self.t_last and self.p.decay_s > 0:
            self.L *= np.float32(math.exp(-(t - self.t_last) / self.p.decay_s))
        self.t_last = t if self.t_last is None else max(self.t_last, t)

    def integrate(self, t, pose, scan: Scan) -> bool:
        """One scan seen from nominal-frame pose (x, y, yaw) at time t. Never gated."""
        self.scans_in += 1
        if self.origin is None or math.hypot(*(np.asarray(pose[:2]) - self.centre())) > self.p.recentre_m:
            self._centre_on(pose[:2])
        self.decay(t)
        self._mark_body(t, pose)
        if scan is None or len(scan) == 0:
            return False
        valid = np.isfinite(scan.range) & (scan.range > 0) & np.isfinite(scan.bearing)
        keep = valid & (scan.range <= self.p.max_range_m)
        r, b = scan.range[keep], scan.bearing[keep]
        marks = np.ones(len(scan.range), bool) if scan.free_only is None else ~np.asarray(scan.free_only, bool)
        if marks.shape != scan.range.shape:
            raise ValueError("scan.free_only must have one flag per beam")
        marks = marks[keep]
        res, shape = self.p.res_m, np.array(self.L.shape)
        x0, y0, th = float(pose[0]), float(pose[1]), float(pose[2])
        ca, sa = np.cos(th + b), np.sin(th + b)
        ts = np.arange(0.0, self.p.max_range_m, res * 0.5)
        ok = ts[None] < (r[:, None] - res)
        X, Y = x0 + ca[:, None] * ts[None], y0 + sa[:, None] * ts[None]
        fij = np.floor((np.stack([X[ok], Y[ok]], -1) - self.origin) / res).astype(int)
        fij = fij[np.all((fij >= 0) & (fij < shape), axis=1)]
        # free-only beams (dense free rays) carve like any beam but never mark their endpoint
        eij = np.floor((np.stack([x0 + ca[marks] * r[marks], y0 + sa[marks] * r[marks]], -1)
                        - self.origin) / res).astype(int).reshape(-1, 2)
        eij = eij[np.all((eij >= 0) & (eij < shape), axis=1)]
        # Distant measurements certify only the near segment, from the actual
        # camera origin. No origin/return/uncertainty means no free evidence.
        optical = scan.free_origin_xy
        free_range = self.p.free_range_m if self.p.free_range_m > 0 else self.p.max_range_m
        if optical is not None:
            optical = np.asarray(optical, float)
            if (optical.shape == (2,) and np.isfinite(optical).all()
                    and np.linalg.norm(optical) < free_range):
                far = (valid & (scan.range > self.p.max_range_m)
                       & np.isfinite(scan.sigma) & (scan.sigma >= 0))
                endpoints = np.stack([scan.range[far] * np.cos(scan.bearing[far]),
                                      scan.range[far] * np.sin(scan.bearing[far])], axis=1)
                vectors = endpoints - optical
                lengths = np.linalg.norm(vectors, axis=1)
                directions = vectors / lengths[:, None]
                dot = directions @ optical
                horizon = -dot + np.sqrt(dot ** 2 + free_range ** 2
                                          - optical @ optical)
                reach = np.minimum(lengths - 3.0 * scan.sigma[far], horizon) - res
                # Camera can sit behind the body origin; allow the full chord.
                steps = np.arange(0., 2 * free_range, res * .5)
                use = steps[None] < reach[:, None]
                points = optical + directions[:, None, :] * steps[None, :, None]
                c, s = math.cos(th), math.sin(th)
                rotation = np.array([[c, -s], [s, c]])
                world = points[use] @ rotation.T + [x0, y0]
                far_ij = np.floor((world - self.origin) / res).astype(int)
                far_ij = far_ij[np.all((far_ij >= 0) & (far_ij < shape), axis=1)]
                fij = np.concatenate([fij, far_ij])
        free = np.unique(np.ravel_multi_index(fij.T, self.L.shape)) if len(fij) else np.zeros(0, int)
        occ = np.unique(np.ravel_multi_index(eij.T, self.L.shape)) if len(eij) else np.zeros(0, int)
        free = np.setdiff1d(free, occ, assume_unique=True)
        flat, seen = self.L.reshape(-1), self.seen.reshape(-1)
        flat[free] += self.p.l_free
        flat[occ] += self.p.l_occ
        last_free = self.last_free.reshape(-1)
        last_free[free] = t if math.isfinite(t) else -np.inf
        last_free[occ] = -np.inf  # a hit revokes previous free evidence, even before log-odds turn positive
        self.last_body.reshape(-1)[occ] = -np.inf  # ... and the airframe-swept evidence
        np.clip(flat, self.p.l_min, self.p.l_max, out=flat)
        touched = np.concatenate([free, occ])
        seen[touched] = np.minimum(seen[touched].astype(np.int32) + 1, 65535)
        return bool(len(touched))

    def _mark_body(self, t, pose):
        """Record the cells the airframe's own disc covers at this pose (body_radius_m > 0)."""
        rad = self.p.body_radius_m
        if rad <= 0 or self.origin is None or not math.isfinite(t):
            return
        xy = np.asarray(pose[:2], float)
        if not np.isfinite(xy).all():
            return
        res = self.p.res_m
        k = int(math.ceil(rad / res)) + 1
        c = self.ij(xy)[0]
        ii, jj = np.mgrid[c[0] - k:c[0] + k + 1, c[1] - k:c[1] + k + 1]
        ok = (ii >= 0) & (ii < self.n) & (jj >= 0) & (jj < self.n)
        ii, jj = ii[ok], jj[ok]
        # the WHOLE cell must lie inside the disc: farthest corner within rad
        lo = self.origin + np.stack([ii, jj], -1) * res
        far = np.maximum(np.abs(lo - xy), np.abs(lo + res - xy))
        inside = np.hypot(far[:, 0], far[:, 1]) <= rad
        self.last_body[ii[inside], jj[inside]] = t

    # ------------------------------------------------------------------ products
    def occupied(self):
        """Confident obstacle cells, specks removed."""
        occ = self.L >= self.p.occ_thresh
        if not occ.any() or self.p.min_cells <= 1:
            return occ
        lab, n = ndimage.label(occ, structure=np.ones((3, 3), bool))
        sizes = ndimage.sum(occ, lab, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, bool)
        keep[1:] = sizes >= self.p.min_cells
        return keep[lab]

    def obstacle_points(self):
        """(k, 2) nominal-frame centres of the occupied cells."""
        if self.origin is None:
            return np.zeros((0, 2))
        return self.cell_centres(np.argwhere(self.occupied()))

    def probability(self):
        return 1.0 - 1.0 / (1.0 + np.exp(self.L))

    def grid_fields(self, now=None, max_free_age_s=8.0, swept_age_s=None):
        """-> (origin xy, res, width, height, data) for a nav_msgs/OccupancyGrid: row-major with x fastest,
        -1 unknown (never touched), 0..100 probability; occupied-but-speck cells are reported as 50 so a
        reader applying >= 88 (occ_thresh) and the clump rule sees exactly occupied().
        With now, negative/free evidence expires independently of obstacle decay;
        republishing or an empty scan cannot make old free cells fresh again."""
        prob = np.rint(self.probability() * 100).astype(np.int8)
        occ = self.L >= self.p.occ_thresh
        prob[occ & ~self.occupied()] = 50
        prob[self.seen == 0] = -1
        if now is not None:
            age = float(now) - self.last_free
            recent = (np.isfinite(age) & (age >= 0) & (age <= max_free_age_s))
            prob[(self.L < 0) & ~recent] = -1
            swept_age = self.p.swept_age_s if swept_age_s is None else swept_age_s
            if self.p.body_radius_m > 0 and swept_age > 0:
                # cells the airframe itself occupied recently are free, unless a CONFIDENT obstacle return
                # landed there (pose error or something moved in): that evidence always wins
                body_age = float(now) - self.last_body
                swept = np.isfinite(body_age) & (body_age >= 0) & (body_age <= swept_age) & ~occ
                prob[swept] = 0
        return self.origin, self.p.res_m, self.n, self.n, prob.T.ravel()

    def stats(self):
        return {"scans_in": self.scans_in, "occupied": int(self.occupied().sum()),
                "known": int((self.seen > 0).sum()), "recentres": self.recentres,
                "centre": None if self.origin is None else [round(float(v), 2) for v in self.centre()]}


def occupied_points_from_grid(origin, res, width, height, data, occ_prob=88):
    """Reader side (the navigator): OccupancyGrid fields -> (k, 2) frame points of cells >= occ_prob."""
    prob = np.asarray(data, np.int16).reshape(height, width).T
    ij = np.argwhere(prob >= occ_prob)
    return np.asarray(origin, float) + (ij + 0.5) * float(res)


def free_points_from_grid(origin, res, width, height, data, free_prob=23):
    """Fresh, confidently free OccupancyGrid cells -> (k, 2) frame points.

    The publisher must expire per-cell free evidence (LocalGrid.grid_fields(now=...)).
    Unknown and invalid probability cells are never free. The default 23 percent
    corresponds approximately to log-odds -1.2: three pass-only updates from zero,
    not three net passes over hits (weights, saturation and decay differ).
    """
    origin = np.asarray(origin, float)
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("free grid origin must contain two finite coordinates")
    res = float(res)
    if not math.isfinite(res) or res <= 0:
        raise ValueError("free grid resolution must be finite and positive")
    if (not isinstance(width, (int, np.integer)) or not isinstance(height, (int, np.integer))
            or width <= 0 or height <= 0):
        raise ValueError("free grid dimensions must be positive integers")
    free_prob = float(free_prob)
    if not math.isfinite(free_prob) or not 0 <= free_prob < 50:
        raise ValueError("free probability threshold must be in [0, 50)")
    prob = np.asarray(data, float).reshape(height, width).T
    valid = np.isfinite(prob) & (prob >= 0) & (prob <= free_prob)
    valid &= prob == np.floor(prob)  # OccupancyGrid probabilities are integers.
    ij = np.argwhere(valid)
    return origin + (ij + 0.5) * res


def _shift(a, shift, fill=0):
    """Move array contents by -shift cells; drop departing cells and initialize new cells with fill."""
    out = np.full_like(a, fill)
    dx, dy = int(shift[0]), int(shift[1])
    n0, n1 = a.shape
    if abs(dx) >= n0 or abs(dy) >= n1:
        return out
    src = a[max(dx, 0):n0 + min(dx, 0), max(dy, 0):n1 + min(dy, 0)]
    out[max(-dx, 0):n0 + min(-dx, 0), max(-dy, 0):n1 + min(-dy, 0)] = src
    return out


def course_mask(pts_course, grid, prior_dist=None, explain_m=0.0):
    """Local obstacle points already in the course frame -> bool mask on `grid` (coursenav.course.Grid2D).
    explain_m > 0 with prior_dist: drop points within explain_m of a prior surface (the prior already has
    them: a live copy of a mapped wall, a few cm off, would only squeeze the corridors). Use explain_m = 0
    when the prior is dropped: its surfaces are not where the camera sees them, so nothing is explained."""
    mask = np.zeros(grid.occ.shape, bool)
    pts = np.atleast_2d(np.asarray(pts_course, float))
    if pts.size == 0:
        return mask
    ij = grid.ij(pts)
    ok = grid.inside(ij)
    ij = ij[ok]
    if explain_m > 0 and prior_dist is not None and len(ij):
        ij = ij[np.asarray(prior_dist)[ij[:, 0], ij[:, 1]] > explain_m]
    if len(ij):
        mask[ij[:, 0], ij[:, 1]] = True
    return mask
