"""Coloured landing-target localisation from the colour camera + depth. PURE (numpy/scipy).

The colour image gives WHERE the target is (an HSV threshold, largest blob);
depth gives HOW FAR: depth points are projected into the colour image through
the calibrated camera transforms, and the ones landing inside the blob are the
target's surface. A floor-plane ray through the blob centre is the fallback
when too few depth points land inside it. Size and floor-height checks reject
other blue things. TargetTracker turns noisy detections into a stable estimate
that may move (the target is re-localised on demand), bounded by a maximum
shift from the declared waypoint.

Frames: body FLU (x forward, y left, z up); level = body with roll/pitch
removed; camera optical (x right, y down, z forward).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage


def hsv_mask(rgb, h_lo=95., h_hi=125., s_min=90., v_min=60.):
    """Boolean mask of pixels in an OpenCV-style hue band (0-180), saturation/value floors (0-255)."""
    a = np.asarray(rgb, np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(axis=-1), a.min(axis=-1)
    d = mx - mn
    with np.errstate(divide='ignore', invalid='ignore'):
        h = np.where(mx == r, (g - b) / d % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 30.
        s = np.where(mx > 0, d / mx * 255., 0.)
    h = np.where(d > 0, h, 0.)
    return (h >= h_lo) & (h <= h_hi) & (s >= s_min) & (mx >= v_min)


def largest_blob(mask, min_px=40):
    """-> (blob mask, pixel count) of the largest 8-connected component, or (None, 0)."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    if n == 0:
        return None, 0
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    k = int(np.argmax(sizes))
    if sizes[k] < min_px:
        return None, int(sizes[k])
    return lab == k + 1, int(sizes[k])


def depth_body_points(depth_m, K, R_body_cam, t_body_cam, stride=4, rmin=0.3, rmax=8.):
    """Depth image (metres, NaN = none) -> (n,3) body-frame points."""
    fx, fy, cx, cy = K
    d = np.asarray(depth_m, np.float32)
    vs, us = np.mgrid[0:d.shape[0]:stride, 0:d.shape[1]:stride]
    z = d[vs, us].ravel()
    ok = np.isfinite(z) & (z > rmin) & (z < rmax)
    u, v, z = us.ravel()[ok], vs.ravel()[ok], z[ok]
    cam = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=1)
    return cam @ np.asarray(R_body_cam).T + np.asarray(t_body_cam)


def locate_target(blob, K_rgb, R_body_rgb, t_body_rgb, body_pts, R_level_body, cam_height,
                  min_depth_pts=12, floor_tol_m=0.15, size_range_m=(0.2, 1.2)):
    """Target centre in the LEVEL frame (x forward, y left), or (None, info).

    blob: colour-image mask; body_pts: depth points in the body frame; cam_height: height
    of the body origin above the floor (fallback ray and floor check)."""
    info = {}
    H, W = blob.shape
    R_lb = np.asarray(R_level_body)
    if len(body_pts):
        cam = (np.asarray(body_pts) - np.asarray(t_body_rgb)) @ np.asarray(R_body_rgb)   # body -> rgb optical
        front = cam[:, 2] > 0.05
        fx, fy, cx, cy = K_rgb
        u = np.round(fx * cam[:, 0] / np.where(front, cam[:, 2], 1) + cx).astype(int)
        v = np.round(fy * cam[:, 1] / np.where(front, cam[:, 2], 1) + cy).astype(int)
        inimg = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        hit = np.zeros(len(cam), bool)
        hit[inimg] = blob[v[inimg], u[inimg]]
        lvl = np.asarray(body_pts)[hit] @ R_lb.T
        info['depth_pts'] = int(hit.sum())
        if len(lvl) >= min_depth_pts:
            z_med = float(np.median(lvl[:, 2]))
            ext = np.percentile(lvl[:, :2], 90, axis=0) - np.percentile(lvl[:, :2], 10, axis=0)
            info.update(method='depth', floor_z_m=round(z_med + cam_height, 3),
                        extent_m=round(float(np.max(ext)), 3))
            if abs(z_med + cam_height) > floor_tol_m:
                info['reject'] = 'not_on_floor'
                return None, info
            if not size_range_m[0] * .5 <= info['extent_m'] <= size_range_m[1]:
                info['reject'] = 'size'
                return None, info
            # Perspective oversamples the near half of a flat target: use the centre of its
            # robust extent, not the median of its points.
            lo, hi = np.percentile(lvl[:, :2], 5, axis=0), np.percentile(lvl[:, :2], 95, axis=0)
            return (lo + hi) / 2, info
    # Fallback: ray through the blob centroid to the floor plane.
    vv, uu = np.nonzero(blob)
    fx, fy, cx, cy = K_rgb
    ray_cam = np.array([(uu.mean() - cx) / fx, (vv.mean() - cy) / fy, 1.])
    ray = R_lb @ (np.asarray(R_body_rgb) @ ray_cam)
    origin = R_lb @ np.asarray(t_body_rgb, float)
    info['method'] = 'floor_ray'
    if ray[2] >= -1e-3 or not np.isfinite(cam_height):
        info['reject'] = 'ray_misses_floor'
        return None, info
    s = -(cam_height + origin[2]) / ray[2]
    p = origin + s * ray
    if not 0.2 < s < 8.:
        info['reject'] = 'range'
        return None, info
    return p[:2], info


def level_to_course(xy_level, pose):
    """Level body (forward, left) -> course xy through the aircraft pose (x, y, yaw)."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.array([pose[0] + c * xy_level[0] - s * xy_level[1], pose[1] + s * xy_level[0] + c * xy_level[1]])


class TargetTracker:
    """Stable target estimate from recent detections; follows the target if it moves.

    An estimate needs `agree` recent detections within `consistent_m` of their median,
    all within `max_shift_m` of the declared waypoint and inside the room."""

    def __init__(self, declared_xy, max_shift_m=2.5, consistent_m=.15, agree=3, window_s=4., keep=8,
                 inside=None):
        self.declared = np.asarray(declared_xy, float)
        self.max_shift, self.consistent, self.agree = float(max_shift_m), float(consistent_m), int(agree)
        self.window, self.keep = float(window_s), int(keep)
        self.inside = inside
        self.obs = []
        self.estimate = None

    def observe(self, t, xy):
        xy = np.asarray(xy, float)
        if not np.isfinite(xy).all() or np.linalg.norm(xy - self.declared) > self.max_shift:
            return False
        if self.inside is not None and not self.inside(xy):
            return False
        self.obs = [(tt, p) for tt, p in self.obs if t - tt <= self.window][-(self.keep - 1):] + [(t, xy)]
        pts = np.array([p for _, p in self.obs])
        med = np.median(pts, axis=0)
        close = np.linalg.norm(pts - med, axis=1) <= self.consistent
        if close.sum() >= self.agree:
            self.estimate = np.median(pts[close], axis=0)
        return True
