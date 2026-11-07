"""Particle filter for the map <- odom correction. PURE (numpy).

VIO (cuVSLAM) is locally excellent and drifts slowly, so instead of tracking the body
directly each particle is a hypothesis of the 2D transform T = (tx, ty, theta) that maps
the VIO odom frame into the course frame. Body course pose = T o odom_pose.

  predict(): random walk of T, sized by how far/long VIO has moved since the last update
             (and much larger when VIO says it is not visually tracking);
  update():  score each particle's body pose by how well the scan endpoints land on
             mapped surfaces (likelihood field = distance to the nearest surface);
  estimate():weighted mean and covariance of the body pose in the course frame.

The perturbation is applied to the BODY pose, then T is recomputed: a heading error then
rotates about the aircraft, not about the course origin (no lever-arm blow-up).

Tails (A6, after flight E's p95 0.15 m / max 0.24 m with a 3 cm median):
  * update() SKIPS a scan that fits the map poorly: fewer than min_beams informative beams (a beam
    ending on the see-through net carries nothing: `uninformative` cells), or whose best particle
    explains fewer than min_fit_frac of them (a scan of something not on the map must not pull the
    cloud towards a wrong wall);
  * estimate() uses the DOMINANT CLUSTER of particles (cluster_res_m bins, the heaviest connected
    clump) instead of the weighted mean, so a split cloud (two walls could explain the scan) does
    not report a point between the two;
  * project_correction() keeps a correction only along the directions the scan can measure (the
    particle covariance's eigenvectors with a small sigma): inside a corridor or the tube, across
    the axis only;
  * CorrectionLimiter rate-limits the published correction to what VIO drift physically allows.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .course import Grid2D
from .scan import Scan


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def compose(T, p):
    """T (..,3) o p (..,3): 2D rigid transforms (x, y, theta)."""
    c, s = np.cos(T[..., 2]), np.sin(T[..., 2])
    x = T[..., 0] + c * p[..., 0] - s * p[..., 1]
    y = T[..., 1] + s * p[..., 0] + c * p[..., 1]
    return np.stack([x, y, wrap(T[..., 2] + p[..., 2])], axis=-1)


def inverse(p):
    c, s = np.cos(p[..., 2]), np.sin(p[..., 2])
    x = -(c * p[..., 0] + s * p[..., 1])
    y = -(-s * p[..., 0] + c * p[..., 1])
    return np.stack([x, y, wrap(-p[..., 2])], axis=-1)


@dataclass
class PFParams:
    n: int = 600
    drift_per_m: float = 0.04          # T random walk (m per m travelled, 1 sigma)
    drift_per_s: float = 0.01          # ... and per second: keeps the cloud wide enough to follow drift
                                       # where the scan says nothing (a tube along its axis; flight D)
    yaw_drift_per_m: float = math.radians(1.0)
    yaw_drift_per_s: float = math.radians(0.05)
    lost_factor: float = 8.0           # multiply the walk when VIO is not visually tracking
    clip_m: float = 0.30               # likelihood field saturates here (unexplained returns)
    p_outlier: float = 0.15            # robust mixture weight for a beam the map does not explain
    beam_decorrelation: float = 4.0    # neighbouring depth beams are correlated: divide log-lik by this
    ess_frac: float = 0.5
    roughen_xy: float = 0.008
    roughen_yaw: float = math.radians(0.2)
    max_endpoint_m: float = 4.5
    body_clear_m: float = 0.15         # the body centre cannot be closer than this to a mapped surface
    body_penalty: float = -50.0        # log-likelihood for an impossible body position
    # ---- tails (A6)
    min_beams: int = 8                 # skip a scan with fewer informative beams
    min_fit_frac: float = 0.35         # skip a scan whose best particle explains fewer than this of them
    fit_tol_m: float = 0.12            # ... "explains": the endpoint lands within this of a mapped surface
    informative_m: float = 0.15        # a beam ending within this of an uninformative surface (net) is not counted
    use_cluster: bool = True           # estimate() from the dominant particle cluster
    cluster_res_m: float = 0.10


class CorrectionPF:
    def __init__(self, field: Grid2D, distance: np.ndarray, params: PFParams = None, seed=0, free=None,
                 uninformative=None):
        self.p = params or PFParams()
        self.field = field
        self.dist = distance                      # metres to the nearest scan surface, per cell
        # where the BODY may be: away from surfaces and inside the enclosure (scan_field sets the
        # outside to 0, so it is not free). Without this, a particle beyond the net explains every
        # beam "perfectly" (flight A, 2026-11-06: 1.19 m error).
        self.free = (distance > self.p.body_clear_m) if free is None else free
        # cells the map knows: inside the enclosure (scan_field > 0) or a surface. A beam ending beyond
        # the net lands where the field is 0: unknown, never penalised, and never "explained" either
        self.known = (np.asarray(distance) > 0) | np.asarray(field.occ, bool)
        # cells of surfaces a beam can land on without saying where the aircraft is along them
        # (the nets: 92 % open, their returns are sparse and unmodelled)
        self.uninf_dist = None
        if uninformative is not None and np.asarray(uninformative).any():
            self.uninf_dist = ndimage.distance_transform_edt(~np.asarray(uninformative, bool)) * field.res
        self.rng = np.random.default_rng(seed)
        self.T = None                             # (n,3)
        self.w = None
        self.last_odom = None
        self.updates = 0
        self.skipped = 0                          # scans refused by the fit gate
        self.last_ll = None
        self.last_fit = None                      # (informative beams, fraction explained) of the latest scan
        self.last_cluster_frac = 1.0              # weight share of the dominant cluster in the latest estimate

    def set_distance(self, distance):
        """Swap the likelihood field (same grid): e.g. prior surfaces + live landmarks (coursenav.occgrid)."""
        if distance.shape != self.dist.shape:
            raise ValueError("distance field must share the PF grid")
        self.dist = distance

    # ------------------------------------------------------------ setup
    def init(self, body_prior, sigma_xy, sigma_yaw, odom_pose):
        """body_prior: course (x, y, yaw) guess; odom_pose: the VIO pose at the same instant."""
        n = self.p.n
        b = np.empty((n, 3))
        b[:, 0] = body_prior[0] + self.rng.normal(0, sigma_xy, n)
        b[:, 1] = body_prior[1] + self.rng.normal(0, sigma_xy, n)
        b[:, 2] = wrap(body_prior[2] + self.rng.normal(0, sigma_yaw, n))
        self.T = compose(b, inverse(np.asarray(odom_pose, float))[None].repeat(n, 0))
        self.w = np.full(n, 1.0 / n)
        self.last_odom = np.asarray(odom_pose, float)

    @property
    def ready(self):
        return self.T is not None

    # ------------------------------------------------------------ steps
    def body(self, odom_pose):
        return compose(self.T, np.asarray(odom_pose, float)[None].repeat(len(self.T), 0))

    def predict(self, odom_pose, dt, vio_ok=True):
        odom_pose = np.asarray(odom_pose, float)
        moved = math.hypot(*(odom_pose[:2] - self.last_odom[:2]))
        turned = abs(float(wrap(odom_pose[2] - self.last_odom[2])))
        k = 1.0 if vio_ok else self.p.lost_factor
        sxy = k * (self.p.drift_per_m * moved + self.p.drift_per_s * max(dt, 0.0))
        syaw = k * (self.p.yaw_drift_per_m * (moved + 0.3 * turned) + self.p.yaw_drift_per_s * max(dt, 0.0))
        self._jitter(odom_pose, sxy, syaw)
        self.last_odom = odom_pose

    def _jitter(self, odom_pose, sxy, syaw):
        if sxy <= 0 and syaw <= 0:
            return
        b = self.body(odom_pose)
        n = len(b)
        b[:, 0] += self.rng.normal(0, sxy, n)
        b[:, 1] += self.rng.normal(0, sxy, n)
        b[:, 2] = wrap(b[:, 2] + self.rng.normal(0, syaw, n))
        self.T = compose(b, inverse(odom_pose)[None].repeat(n, 0))

    def body_ok(self, poses):
        ij = np.floor((poses[:, :2] - self.field.origin) / self.field.res).astype(int)
        inside = np.all((ij >= 0) & (ij < np.array(self.free.shape)), axis=-1)
        ij = np.where(inside[:, None], ij, 0)
        return inside & self.free[ij[:, 0], ij[:, 1]]

    def _endpoints_world(self, poses, e):
        c, s = np.cos(poses[:, 2])[:, None], np.sin(poses[:, 2])[:, None]
        X = poses[:, 0:1] + c * e[None, :, 0] - s * e[None, :, 1]
        Y = poses[:, 1:2] + s * e[None, :, 0] + c * e[None, :, 1]
        return X, Y

    def _lookup(self, grid, X, Y, outside=0.0):
        ij = np.floor((np.stack([X, Y], -1) - self.field.origin) / self.field.res).astype(int)
        inside = np.all((ij >= 0) & (ij < np.array(grid.shape)), axis=-1)
        ij = np.where(inside[..., None], ij, 0)
        return np.where(inside, grid[ij[..., 0], ij[..., 1]], outside)

    def beam_distances(self, poses, scan: Scan):
        """(n, m) distance of each kept beam's endpoint to the nearest mapped surface, per pose
        (off the map = beyond the enclosure: 0, unknown and never penalised), and the kept sigmas."""
        e = scan.endpoints()                                     # (m,2) level body
        keep = scan.range <= self.p.max_endpoint_m
        e, sig = e[keep], scan.sigma[keep]
        if len(e) == 0:
            return np.zeros((len(poses), 0)), sig
        X, Y = self._endpoints_world(poses, e)
        return np.minimum(self._lookup(self.dist, X, Y), self.p.clip_m), sig

    def loglik(self, poses, scan: Scan):
        """(n,) log-likelihood of the scan for each body pose (n,3)."""
        d, sig = self.beam_distances(poses, scan)
        if d.shape[1] == 0:
            return np.zeros(len(poses))
        # robust mixture: Gaussian on the distance-to-surface, plus a flat outlier floor
        g = np.exp(-0.5 * (d / sig[None]) ** 2)
        lik = (1 - self.p.p_outlier) * g + self.p.p_outlier * math.exp(-0.5 * (self.p.clip_m / 0.1) ** 2) + 1e-12
        return np.log(lik).sum(axis=1) / self.p.beam_decorrelation + np.where(self.body_ok(poses), 0.0, self.p.body_penalty)

    def informative(self, odom_pose, scan: Scan):
        """Which beams say something about where the aircraft is: endpoints (seen from the current
        estimate) on KNOWN cells (inside the enclosure) and not on an uninformative surface such as
        the net."""
        keep = scan.range <= self.p.max_endpoint_m
        mean, _ = self.estimate(odom_pose)
        X, Y = self._endpoints_world(mean[None], scan.endpoints())
        known = self._lookup(self.known, X, Y, outside=False)[0].astype(bool)
        inf = keep & known
        if self.uninf_dist is not None:
            near_uninf = self._lookup(self.uninf_dist, X, Y, outside=np.inf)[0] <= self.p.informative_m
            inf = inf & ~near_uninf
        return inf

    def scan_fit(self, odom_pose, scan: Scan):
        """-> (n informative beams, fraction of them the BEST particle explains). The fit gate.
        With fewer than min_beams informative beams (a view of nothing but the net), the fraction is
        judged on every known beam instead, so a start scan facing the net still counts."""
        inf = self.informative(odom_pose, scan)
        n_inf = int(inf.sum())
        if n_inf < self.p.min_beams:
            keep = scan.range <= self.p.max_endpoint_m
            mean, _ = self.estimate(odom_pose)
            X, Y = self._endpoints_world(mean[None], scan.endpoints())
            inf = keep & self._lookup(self.known, X, Y, outside=False)[0].astype(bool)
            n_inf = int(inf.sum())
        if n_inf == 0:
            return 0, 0.0
        sub = Scan(scan.bearing[inf], scan.range[inf], scan.sigma[inf])
        b = self.body(odom_pose)
        d, sig = self.beam_distances(b, sub)
        ll = self.loglik(b, sub)
        best = int(np.argmax(ll))
        frac = float(np.mean(d[best] <= self.p.fit_tol_m)) if d.shape[1] else 0.0
        return n_inf, frac

    def update(self, odom_pose, scan: Scan):
        if scan is None or len(scan) == 0:
            return False
        n_inf, frac = self.scan_fit(odom_pose, scan)
        self.last_fit = (n_inf, frac)
        if n_inf < self.p.min_beams or frac < self.p.min_fit_frac:
            self.skipped += 1                                    # the map does not explain this scan: keep the belief
            return False
        ll = self.loglik(self.body(odom_pose), scan)
        ll -= ll.max()
        self.last_ll = ll
        self.w = self.w * np.exp(ll)
        s = self.w.sum()
        if not np.isfinite(s) or s <= 0:
            self.w = np.full(len(self.w), 1.0 / len(self.w))
        else:
            self.w /= s
        self.updates += 1
        if self.ess() < self.p.ess_frac * len(self.w):
            self._resample(odom_pose)
        return True

    def ess(self):
        return float(1.0 / np.sum(self.w ** 2))

    def _resample(self, odom_pose):
        n = len(self.w)
        positions = (self.rng.random() + np.arange(n)) / n
        idx = np.minimum(np.searchsorted(np.cumsum(self.w), positions), n - 1)
        self.T = self.T[idx]
        self.w = np.full(n, 1.0 / n)
        self._jitter(odom_pose, self.p.roughen_xy, self.p.roughen_yaw)

    def reanchor(self, old_odom, new_odom):
        """VIO reset: keep the body estimate continuous across an odom frame jump."""
        b = self.body(old_odom)
        self.T = compose(b, inverse(np.asarray(new_odom, float))[None].repeat(len(b), 0))
        self.last_odom = np.asarray(new_odom, float)

    def dominant(self, b):
        """Mask of the particles in the heaviest connected clump of cluster_res_m cells."""
        res = self.p.cluster_res_m
        ij = np.floor((b[:, :2] - b[:, :2].min(axis=0)) / res).astype(int)
        shape = tuple(ij.max(axis=0) + 1)
        wsum = np.zeros(shape)
        np.add.at(wsum, (ij[:, 0], ij[:, 1]), self.w)
        lab, n = ndimage.label(wsum > 0, structure=np.ones((3, 3), bool))
        if n <= 1:
            self.last_cluster_frac = 1.0
            return np.ones(len(b), bool)
        weights = ndimage.sum(wsum, lab, index=np.arange(1, n + 1))
        best = int(np.argmax(weights)) + 1
        self.last_cluster_frac = float(weights[best - 1] / max(wsum.sum(), 1e-12))
        return lab[ij[:, 0], ij[:, 1]] == best

    def estimate(self, odom_pose):
        """-> (mean (x, y, yaw), cov 3x3) of the body pose in the course frame, from the dominant
        cluster of particles (use_cluster) or the whole cloud."""
        b = self.body(odom_pose)
        w = self.w
        if self.p.use_cluster and len(b) > 1:
            m = self.dominant(b)
            b, w = b[m], w[m] / max(w[m].sum(), 1e-12)
        mx, my = np.sum(w * b[:, 0]), np.sum(w * b[:, 1])
        myaw = math.atan2(np.sum(w * np.sin(b[:, 2])), np.sum(w * np.cos(b[:, 2])))
        d = np.stack([b[:, 0] - mx, b[:, 1] - my, wrap(b[:, 2] - myaw)], axis=1)
        cov = (d * w[:, None]).T @ d
        return np.array([mx, my, myaw]), cov


# ---------------------------------------------------------------- what reaches the mission
def project_correction(dxy, cov_xy, sigma_max_m=0.10):
    """Keep a correction only along the directions the scan has constrained: the eigenvectors of the
    xy covariance whose sigma is at most sigma_max_m. Along an unconstrained direction (a corridor's
    axis, the tube's axis) the correction is dropped. -> (projected (2,), number of kept directions,
    the 2x2 projector onto the kept directions: apply it to update an earlier value only along them)."""
    dxy = np.asarray(dxy, float)
    cov = np.asarray(cov_xy, float)[:2, :2]
    vals, vecs = np.linalg.eigh(0.5 * (cov + cov.T))
    P = np.zeros((2, 2))
    kept = 0
    for val, vec in zip(vals, vecs.T):
        if math.sqrt(max(val, 0.0)) <= sigma_max_m:
            P += np.outer(vec, vec)
            kept += 1
    return P @ dxy, kept, P


class CorrectionLimiter:
    """Rate-limit a published correction to what VIO drift can physically produce: a few cm per
    metre travelled plus a small amount per second. The first value passes unlimited (the start
    offset found on the ground is a one-off, not drift)."""

    def __init__(self, rate_per_m=0.05, rate_per_s=0.01):
        self.rate_per_m, self.rate_per_s = float(rate_per_m), float(rate_per_s)
        self.out = None

    def step(self, target_xy, travelled_m, dt_s):
        target = np.asarray(target_xy, float)
        if self.out is None:
            self.out = target.copy()
            return self.out.copy()
        allowed = self.rate_per_m * max(travelled_m, 0.0) + self.rate_per_s * max(dt_s, 0.0)
        d = target - self.out
        n = float(np.hypot(*d))
        if n > allowed and n > 0:
            d = d * allowed / n
        self.out = self.out + d
        return self.out.copy()
