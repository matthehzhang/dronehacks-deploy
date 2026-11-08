"""Depth image -> a horizontal "laser-style" scan in the level body frame. PURE.

Sensor policy: the only scan source is the forward OAK depth image (an allowed, on-board
sensor). No extra range sensors are used.

Level body frame: x forward, y left, heading-aligned with the body, gravity-levelled
(roll and pitch removed), origin at the body origin. A scan is two arrays: bearings
(rad, +left) and ranges (m) from the body origin, one per bearing bin. Bins without a
return are dropped: no return is unknown, never free space.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# level (forward, right, down) -> level (forward, left, up)
_FRD_TO_FLU = np.diag([1.0, -1.0, -1.0])


@dataclass
class Scan:
    bearing: np.ndarray       # rad, + = left
    range: np.ndarray         # m from the body origin, in the horizontal plane
    sigma: np.ndarray         # m, per-beam measurement std
    free_origin_xy: np.ndarray | None = None  # measured optical origin for far free-only rays
    free_only: np.ndarray | None = None       # per beam: True = carve free space only, never mark its endpoint

    def __len__(self):
        return len(self.range)

    def endpoints(self):
        """(n,2) level-body xy of each return."""
        return np.stack([self.range * np.cos(self.bearing), self.range * np.sin(self.bearing)], axis=1)

    @staticmethod
    def empty():
        z = np.zeros(0)
        return Scan(z, z, z)

    @staticmethod
    def concat(scans):
        scans = [s for s in scans if s is not None and len(s)]
        if not scans:
            return Scan.empty()
        return Scan(*(np.concatenate([getattr(s, f) for s in scans]) for f in ("bearing", "range", "sigma")))


def depth_points_level(depth_m, K, R_body_cam, t_body_cam, R_level_body, stride=4, border_px=16):
    """Depth (H,W) metres (NaN = none) -> (n,3) points in the level FLU body frame."""
    fx, fy, cx, cy = K
    d = np.asarray(depth_m, np.float32)
    h, w = d.shape
    vs, us = np.mgrid[border_px:h - border_px:stride, border_px:w - border_px:stride]
    z = d[vs, us].ravel()
    ok = np.isfinite(z) & (z > 0)
    u, v, z = us.ravel()[ok].astype(float), vs.ravel()[ok].astype(float), z[ok].astype(float)
    p_cam = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=1)
    p_body = p_cam @ np.asarray(R_body_cam).T + np.asarray(t_body_cam)       # body FLU
    return p_body @ np.asarray(R_level_body).T


def level_from_rp(roll, pitch):
    """Level FLU <- body FLU for a FLU body with the given roll and pitch (yaw removed)."""
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Ry @ Rx


def level_from_px4_q(q_wxyz):
    """Level FLU <- body FLU from PX4's vehicle_attitude (FRD body -> NED, w first)."""
    w, x, y, z = q_wxyz
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    yaw = math.atan2(R[1, 0], R[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    Rz_inv = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return _FRD_TO_FLU @ Rz_inv @ R @ _FRD_TO_FLU


def scan_from_points(pts_level, height_m, zlo=0.3, zhi=1.6, rmin=0.4, rmax=4.0, n_bins=48,
                     hfov_rad=math.radians(76), quantile=0.1, min_pts=3, sigma0=0.05, sigma_k=0.02):
    """Level points -> scan. Keeps points whose height above the floor (height_m + z) is in
    [zlo, zhi] and horizontal range in [rmin, rmax]; per bearing bin returns a low quantile
    of the range (nearest surface, robust to single stereo outliers). Stereo depth error grows
    with range^2: sigma = sigma0 + sigma_k * r^2."""
    if pts_level is None or len(pts_level) == 0:
        return Scan.empty()
    zw = height_m + pts_level[:, 2]
    r = np.hypot(pts_level[:, 0], pts_level[:, 1])
    b = np.arctan2(pts_level[:, 1], pts_level[:, 0])
    sel = (zw >= zlo) & (zw <= zhi) & (r >= rmin) & (r <= rmax) & (np.abs(b) <= hfov_rad / 2)
    r, b = r[sel], b[sel]
    if len(r) == 0:
        return Scan.empty()
    edges = np.linspace(-hfov_rad / 2, hfov_rad / 2, n_bins + 1)
    k = np.clip(np.digitize(b, edges) - 1, 0, n_bins - 1)
    order = np.lexsort((r, k))
    k, r, b = k[order], r[order], b[order]
    starts = np.searchsorted(k, np.arange(n_bins), "left")
    ends = np.searchsorted(k, np.arange(n_bins), "right")
    out_b, out_r = [], []
    for i in range(n_bins):
        n = ends[i] - starts[i]
        if n < min_pts:
            continue
        j = starts[i] + int(quantile * (n - 1))
        out_r.append(r[j])
        out_b.append(0.5 * (edges[i] + edges[i + 1]))
    out_r = np.asarray(out_r)
    return Scan(np.asarray(out_b), out_r, sigma0 + sigma_k * out_r ** 2)



def local_scan_from_points(pts_level, height_m, zlo=0.3, zhi=1.6, *,
                           obstacle_range_m=3.0, ray_range_m=12.0,
                           camera_origin_xy=None, n_bins=48,
                           hfov_rad=math.radians(76), min_pts=3,
                           sigma0=0.05, sigma_k=0.02):
    """Keep the existing near scan; add measured distant rays for bounded free carving.

    A near point vetoes distant evidence in its bin even if it cannot form a
    supported obstacle return. Far rays retain their actual direction and optical
    origin; they are never synthetic obstacle endpoints at the local cutoff.
    """
    near = scan_from_points(pts_level, height_m, zlo, zhi, rmax=obstacle_range_m,
                            n_bins=n_bins, hfov_rad=hfov_rad, min_pts=min_pts,
                            sigma0=sigma0, sigma_k=sigma_k)
    if camera_origin_xy is None or pts_level is None or not len(pts_level):
        return near
    origin = np.asarray(camera_origin_xy, float)
    if origin.shape != (2,) or not np.isfinite(origin).all():
        return near
    pts = np.asarray(pts_level, float)
    r = np.hypot(pts[:, 0], pts[:, 1])
    b = np.arctan2(pts[:, 1], pts[:, 0])
    zw = height_m + pts[:, 2]
    valid = (np.isfinite(pts).all(axis=1) & (r > 0) & (zw >= zlo) &
             (zw <= zhi) & (np.abs(b) <= hfov_rad / 2))
    edges = np.linspace(-hfov_rad / 2, hfov_rad / 2, n_bins + 1)
    pts, r, b = pts[valid], r[valid], b[valid]
    bins = np.clip(np.digitize(b, edges) - 1, 0, n_bins - 1)
    veto = set(bins[r <= obstacle_range_m])
    out_b, out_r, out_s = [], [], []
    for i in range(n_bins):
        if i in veto:
            continue
        ids = np.flatnonzero((bins == i) & (r > obstacle_range_m) & (r <= ray_range_m))
        if len(ids) < min_pts:
            continue
        lengths = np.linalg.norm(pts[ids, :2] - origin, axis=1)
        sigmas = sigma0 + sigma_k * lengths ** 2
        # Select a real ray with the most conservative uncertainty-adjusted reach.
        j = int(np.argmin(lengths - 3.0 * sigmas))
        out_b.append(b[ids[j]])
        out_r.append(r[ids[j]])
        out_s.append(sigmas[j])
    return Scan(np.concatenate([near.bearing, out_b]), np.concatenate([near.range, out_r]),
                np.concatenate([near.sigma, out_s]), origin.copy())


def with_dense_free_rays(obstacle_scan, pts_level, height_m, zlo=0.3, zhi=1.6, *, n_bins=160,
                         obstacle_range_m=3.0, ray_range_m=12.0, camera_origin_xy=None,
                         hfov_rad=math.radians(76), min_pts=3, sigma0=0.05, sigma_k=0.02):
    """The obstacle scan, plus FREE-ONLY rays from a finer angular binning of the same points.

    Why: with 48 bins (1.6 deg) neighbouring rays are more than one 5 cm cell apart beyond ~1.8 m,
    so the cells between them stay unknown and every certified-free footprint erodes around them.
    Obstacle marking is unchanged (the coarse scan keeps its robust per-bin quantile and endpoints).
    The dense rays obey the same rules as local_scan_from_points: a near return ends its ray at the
    bin's low quantile, a bin with any near point but too few for a return gives no ray at all and
    vetoes far carving, far rays carve only up to their 3-sigma-shortened length. They never mark
    an endpoint occupied, so a sparser bin cannot invent an obstacle either.
    n_bins <= len of the coarse binning disables them (the obstacle scan is returned unchanged).
    """
    base = obstacle_scan if obstacle_scan is not None else Scan.empty()
    if n_bins is None or n_bins <= 0 or pts_level is None or not len(pts_level):
        return base
    dense = local_scan_from_points(pts_level, height_m, zlo, zhi, obstacle_range_m=obstacle_range_m,
                                   ray_range_m=ray_range_m, camera_origin_xy=camera_origin_xy,
                                   n_bins=int(n_bins), hfov_rad=hfov_rad, min_pts=min_pts,
                                   sigma0=sigma0, sigma_k=sigma_k)
    if not len(dense):
        return base
    flags = np.concatenate([np.zeros(len(base), bool) if base.free_only is None else np.asarray(base.free_only, bool),
                            np.ones(len(dense), bool)])
    origin = base.free_origin_xy if base.free_origin_xy is not None else dense.free_origin_xy
    return Scan(np.concatenate([base.bearing, dense.bearing]), np.concatenate([base.range, dense.range]),
                np.concatenate([base.sigma, dense.sigma]), origin, flags)
