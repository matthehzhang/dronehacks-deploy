"""Measured horizontal launch-floor qualification from calibrated depth only."""
from dataclasses import dataclass, asdict
import math
import cv2
import numpy as np


@dataclass(frozen=True)
class FloorParams:
    min_points: int = 100
    min_area_m2: float = .20
    min_xy_span_m: float = .30
    inlier_distance_m: float = .035
    max_residual_m: float = .025
    max_tilt_deg: float = 3.
    minimum_inlier_fraction: float = .55
    nearest_support_m: float = 1.0
    max_support_range_m: float = 2.5
    stable_frames: int = 3
    stable_time_s: float = .5
    stable_height_m: float = .04


def fit_floor(depth, K, R_world_cam, t_world_cam, body_position, params=None):
    """Return (measurement, rejection reason), with no assumed floor/vehicle height.

    Landed-state qualification is the adapter's responsibility. A plane must be
    below the body and have broad, nearby horizontal support. A vertical wall,
    a small rock top or a hole/no return cannot become an invented ground plane.
    """
    p = params or FloorParams()
    d = np.asarray(depth, float)
    fx, fy, cx, cy = K
    body = np.asarray(body_position, float)
    R, t = np.asarray(R_world_cam, float), np.asarray(t_world_cam, float)
    if d.ndim != 2 or body.shape != (3,) or not np.isfinite(body).all():
        return None, 'invalid floor input'
    h, w = d.shape
    vs, us = np.mgrid[8:max(8, h-8):3, 8:max(8, w-8):3]
    ds = d[vs, us]
    ok = np.isfinite(ds) & (ds > .3) & (ds < 6.)
    ds = ds[ok]
    cam = np.column_stack([(us[ok]-cx)/fx*ds, (vs[ok]-cy)/fy*ds, ds])
    points = cam @ R.T + t
    horizontal_range = np.linalg.norm(points[:, :2] - body[:2], axis=1)
    # "Below the body" derives from measured IMU/world-up and pose, not a
    # nominal landing-gear dimension. Keep nearby support rather than a remote
    # horizontal shelf on the other side of the room.
    points = points[(points[:, 2] < body[2]-.03) & (horizontal_range < p.max_support_range_m)]
    if len(points) < p.min_points:
        return None, f'floor support insufficient: {len(points)} below-body depth points'
    if len(points) > 4000:
        points = points[np.linspace(0, len(points)-1, 4000).astype(int)]
    rng = np.random.default_rng(17)  # repeatable robust fit, independent of maps.
    best, best_n = None, 0
    for _ in range(90):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1]-sample[0], sample[2]-sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal /= norm
        if normal[2] < 0:
            normal *= -1
        if normal[2] < math.cos(math.radians(p.max_tilt_deg)):
            continue
        mask = np.abs((points-sample[0]) @ normal) < p.inlier_distance_m
        n = int(mask.sum())
        if n > best_n:
            best, best_n = mask, n
    if best is None or best_n < p.min_points or best_n / len(points) < p.minimum_inlier_fraction:
        return None, 'no broadly supported horizontal floor plane'
    inliers = points[best]
    A = np.column_stack([inliers[:, 0], inliers[:, 1], np.ones(len(inliers))])
    coefficients, *_ = np.linalg.lstsq(A, inliers[:, 2], rcond=None)
    a, b, c = coefficients
    normal = np.array([-a, -b, 1.])
    normal /= np.linalg.norm(normal)
    tilt = math.degrees(math.acos(float(np.clip(normal[2], -1., 1.))))
    residual = np.abs(A @ coefficients - inliers[:, 2])
    rmse = float(np.sqrt(np.mean(residual**2)))
    area = float(cv2.contourArea(cv2.convexHull(inliers[:, :2].astype(np.float32))))
    spans = np.ptp(inliers[:, :2], axis=0)
    nearest = float(np.min(np.linalg.norm(inliers[:, :2]-body[:2], axis=1)))
    floor_z = float(a*body[0] + b*body[1] + c)
    if tilt > p.max_tilt_deg or rmse > p.max_residual_m:
        return None, f'floor tilt/residual rejected: {tilt:.2f}deg {rmse:.3f}m'
    if area < p.min_area_m2 or spans.min() < p.min_xy_span_m or nearest > p.nearest_support_m:
        return None, f'floor footprint support rejected: area={area:.3f}, spans={spans.tolist()}, nearest={nearest:.3f}'
    if floor_z >= body[2]-.03:
        return None, 'floor plane is not below landed body'
    return dict(floor_z=floor_z, normal=normal.tolist(), tilt_deg=tilt,
                residual_m=rmse, support_points=best_n, support_fraction=best_n/len(points),
                area_m2=area, nearest_support_m=nearest, source='calibrated_depth_plane'), ''


class FloorTracker:
    def __init__(self, params=None):
        self.p = params or FloorParams()
        self.history = []
        self.measurement = None
        self.reason = 'no measured ground plane'

    def update(self, measurement, stamp, reason=''):
        if self.history and stamp <= self.history[-1][0]:
            return self.measurement
        if measurement is None:
            self.history = []
            self.measurement = None
            self.reason = reason or 'floor fit rejected'
            return None
        if self.history and (stamp-self.history[-1][0] > .6 or
                             abs(measurement['floor_z']-self.history[-1][1]['floor_z']) > self.p.stable_height_m):
            self.history = []
        self.history.append((stamp, measurement))
        self.history = self.history[-10:]
        if len(self.history) >= self.p.stable_frames and stamp-self.history[0][0] >= self.p.stable_time_s:
            heights = [m['floor_z'] for _, m in self.history]
            if np.ptp(heights) <= self.p.stable_height_m:
                self.measurement = dict(measurement, floor_z=float(np.median(heights)))
                self.reason = 'stable measured horizontal depth plane'
                return self.measurement
        self.reason = f'qualifying floor plane: {len(self.history)} frames'
        return None

    def status(self):
        return dict(reason=self.reason, stable_frames=len(self.history), measurement=self.measurement)
