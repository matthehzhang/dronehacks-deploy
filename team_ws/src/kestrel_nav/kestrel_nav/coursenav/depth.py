"""Forward clearance from one depth frame. PURE (numpy only).

The frame arrives with its own calibration: the pinhole intrinsics from CameraInfo and the
camera's pose on the body from TF (base_link <- camera optical frame, REP-103: optical x right,
y down, z forward; body x forward, y left, z up). The vehicle attitude levels the points, so
"ahead" and "at flight height" mean the same thing while the airframe is tilted.
"""
from __future__ import annotations

import math

import numpy as np

# body FLU -> body FRD (PX4's body frame)
_FLU_TO_FRD = np.diag([1.0, -1.0, -1.0])


def quat_to_mat(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def level_rotation(q_wxyz):
    """Rotation taking body-FLU vectors into the LEVEL heading frame (x forward horizontal,
    y right horizontal, z down), i.e. PX4's attitude with its yaw removed. q = PX4's
    vehicle_attitude quaternion (FRD body -> NED), w first."""
    R = quat_to_mat(*q_wxyz)                                   # NED <- FRD
    yaw = math.atan2(R[1, 0], R[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    Rz_inv = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return Rz_inv @ R @ _FLU_TO_FRD


def decode_depth(encoding, height, width, step, data):
    """sensor_msgs/Image depth -> float32 metres (NaN = no return). 32FC1 is metres,
    16UC1 is millimetres (0 = invalid); anything else raises ValueError."""
    enc = encoding.lower()
    if enc == "32fc1":
        a = np.frombuffer(data, np.float32).reshape(height, step // 4)[:, :width].astype(np.float32)
        a = a.copy()
        a[~np.isfinite(a) | (a <= 0.0)] = np.nan
        return a
    if enc in ("16uc1", "mono16"):
        a = np.frombuffer(data, np.uint16).reshape(height, step // 2)[:, :width].astype(np.float32) * 1e-3
        a[a <= 0.0] = np.nan
        return a
    raise ValueError(f"unsupported depth encoding {encoding!r} (need 32FC1 metres or 16UC1 millimetres)")


class DepthFrame:
    """One depth frame, sampled every `stride` pixels inside a `border_px` margin, in the LEVEL
    heading frame (forward, right, down) relative to the body origin. Keeps, per sampled pixel, its
    ray and whether the camera returned ANY depth for it - so the frame can say what it observed
    (a return, near or far) apart from what it did not (no return: unknown, never free space)."""

    def __init__(self, depth_m, K, R_body_cam, t_body_cam, R_level_body, stride=8, border_px=0):
        fx, fy, cx, cy = K
        d = np.asarray(depth_m, np.float32)
        h, w = d.shape
        b = int(border_px)
        vs, us = np.mgrid[b:h - b:stride, b:w - b:stride] if b > 0 else np.mgrid[0:h:stride, 0:w:stride]
        z = d[vs, us].ravel()
        u, v = us.ravel().astype(np.float32), vs.ravel().astype(np.float32)
        rays_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=1)    # optical z = 1
        R = np.asarray(R_level_body) @ np.asarray(R_body_cam)
        self.dirs = rays_cam @ R.T                     # a point at optical depth z is origin + z * dir
        self.origin = np.asarray(R_level_body) @ np.asarray(t_body_cam, float)
        self.z = z
        self.valid = np.isfinite(z) & (z > 0.0)
        self.valid_frac = float(self.valid.mean()) if z.size else 0.0

    def points(self, rmin=0.3, rmax=6.0):
        sel = self.valid & (self.z > rmin) & (self.z < rmax)
        return self.origin + self.dirs[sel] * self.z[sel, None]

    def corridor_coverage(self, bearing_rad, half_width_m, band_m, at_m):
        """Fraction of the rays that cross the corridor's cross-section at at_m ahead (along the
        bearing, within +-half_width_m and +-band_m of the body height) that returned depth.
        0.0 when no ray of this camera crosses it (the corridor is out of view)."""
        c, s = math.cos(bearing_rad), math.sin(bearing_rad)
        along_dir = self.dirs[:, 0] * c + self.dirs[:, 1] * s
        ok = along_dir > 1e-6
        t = np.where(ok, (at_m - (self.origin[0] * c + self.origin[1] * s)) / np.where(ok, along_dir, 1.0), -1.0)
        p = self.origin + self.dirs * t[:, None]
        across = -p[:, 0] * s + p[:, 1] * c
        window = ok & (t > 0) & (np.abs(across) < half_width_m) & (np.abs(p[:, 2]) < band_m)
        n = int(window.sum())
        return (float(self.valid[window].mean()) if n else 0.0), n


def level_points(depth_m, K, R_body_cam, t_body_cam, R_level_body, stride=8, rmin=0.3, rmax=6.0, border_px=0):
    """Depth image -> (N,3) points in the level heading frame relative to the body origin:
    columns (forward, right, down) in metres. K = (fx, fy, cx, cy). border_px: ignore that many
    pixels at every image edge, where a stereo matcher has the least support (a real OAK-D and
    the simulator's SGBM both produce false near matches along the top rows)."""
    fx, fy, cx, cy = K
    d = np.array(depth_m, np.float32, copy=True)
    if border_px > 0:
        b = int(border_px)
        d[:b, :] = np.nan
        d[-b:, :] = np.nan
        d[:, :b] = np.nan
        d[:, -b:] = np.nan
    d = d[::stride, ::stride]
    vs, us = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    u = us.astype(np.float32) * stride
    v = vs.astype(np.float32) * stride
    ok = np.isfinite(d) & (d > rmin) & (d < rmax)
    z = d[ok]
    p_cam = np.stack([(u[ok] - cx) / fx * z, (v[ok] - cy) / fy * z, z], axis=1)
    p_body = p_cam @ np.asarray(R_body_cam).T + np.asarray(t_body_cam)
    return p_body @ np.asarray(R_level_body).T


def clearance(points, bearing_rad=0.0, half_width_m=0.6, band_m=0.4, min_hits=3):
    """Distance to the nearest obstacle in a corridor of +-half_width_m around the bearing
    (relative to the body heading, positive to the right) and within +-band_m of the body's
    height: the min_hits-th nearest point along the bearing (single pixels are not walls).
    inf when fewer than min_hits points lie in the corridor."""
    if points is None or len(points) == 0:
        return math.inf
    c, s = math.cos(bearing_rad), math.sin(bearing_rad)
    along = points[:, 0] * c + points[:, 1] * s
    across = -points[:, 0] * s + points[:, 1] * c
    sel = (along > 0.0) & (np.abs(across) < half_width_m) & (np.abs(points[:, 2]) < band_m)
    hits = np.sort(along[sel])
    return float(hits[min_hits - 1]) if len(hits) >= min_hits else math.inf


def in_view(bearing_rad, K, width, margin_deg=5.0):
    """Is a horizontal bearing (relative to the camera's forward axis) inside the image?"""
    fx, _, cx, _ = K
    half = min(math.atan2(cx, fx), math.atan2(width - cx, fx))
    return abs(bearing_rad) < half - math.radians(margin_deg)
