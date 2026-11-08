"""Measured depth apertures and a clearance-gated 3-D centering controller.

A hole is supported by valid returns behind a near, four-sided vertical frame.
Invalid stereo returns are unknown, never an opening. Detection alone grants no
motion: the caller must certify each requested swept 3-D vehicle volume against
fresh observed space. A fixed-height 2-D A* map cannot provide that certificate
for an altitude change.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable

import cv2
import numpy as np


@dataclass(frozen=True)
class OpeningParams:
    radius_m: float = .294082845
    margin_m: float = .12
    half_height_m: float = .4
    resolution_m: float = .025
    min_range_m: float = .3
    max_range_m: float = 6.
    plane_tolerance_m: float = .06
    plane_residual_m: float = .035
    max_frame_tilt_deg: float = 10.
    min_plane_points: int = 100
    min_full_depth_fraction: float = .30
    min_opening_depth_fraction: float = .60
    min_side_support: float = .65
    frame_band_m: float = .15
    observed_behind_m: float = .60
    max_plane_models: int = 3
    stable_frames: int = 3
    stable_time_s: float = .4
    max_age_s: float = .5
    association_m: float = .15
    size_stability_m: float = .10
    stop_speed_mps: float = .05
    center_tolerance_m: float = .06
    yaw_tolerance_rad: float = math.radians(5.)
    center_speed_mps: float = .15
    traverse_speed_mps: float = .2
    yaw_speed_rps: float = math.radians(20.)
    lookahead_s: float = .5

    def __post_init__(self):
        if self.radius_m < .294082845 or self.margin_m < .12 or self.half_height_m < .4:
            raise ValueError('vehicle envelope and planning reserve may not be reduced')
        if not (.3 <= self.min_full_depth_fraction <= 1 and
                .6 <= self.min_opening_depth_fraction <= 1 and
                0 < self.min_side_support <= 1):
            raise ValueError('invalid depth/frame coverage requirements')
        for name in ('resolution_m', 'min_range_m', 'max_range_m', 'plane_tolerance_m',
                     'plane_residual_m', 'frame_band_m', 'observed_behind_m', 'max_age_s',
                     'association_m', 'size_stability_m', 'stop_speed_mps', 'lookahead_s'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name)
        if self.stable_frames < 3 or self.stable_time_s < .4 or self.max_plane_models < 1:
            raise ValueError('opening requires at least three frames over .4s')


@dataclass(frozen=True)
class Opening:
    center_world: tuple
    normal_world: tuple
    right_world: tuple
    up_world: tuple
    width_m: float
    height_m: float
    plane_range_m: float
    side_support: tuple
    valid_depth_fraction: float
    behind_fraction: float
    plane_residual_m: float
    stamp: float
    body_fits: bool
    box_px: tuple

    def status(self):
        return asdict(self)


def _calibration(K, R, t):
    k, r, pos = np.asarray(K, float), np.asarray(R, float), np.asarray(t, float)
    return (k.shape == (4,) and np.isfinite(k).all() and np.all(k[:2] > 0) and
            r.shape == (3, 3) and np.isfinite(r).all() and
            np.allclose(r.T @ r, np.eye(3), atol=1e-4) and np.linalg.det(r) > .999 and
            pos.shape == (3,) and np.isfinite(pos).all())


class DepthOpeningDetector:
    def __init__(self, params=None):
        self.p = params or OpeningParams()
        self.diagnostics = {'reason': 'no depth aperture measurement'}

    def detect(self, depth, K, R_world_cam, t_world_cam, stamp):
        """Detect bounded approximately vertical apertures, without map priors.

        Depth must already be undistorted optical-z metres; R/t are calibrated
        camera poses at the image's capture timestamp. All four frame sides must
        be in the image. Cropped doors and open-ended corridors remain unproven.
        """
        p = self.p
        d = np.asarray(depth, float)
        self.diagnostics = dict(reason='', planes=0, rejected=[], candidates=0)
        if d.ndim != 2 or min(d.shape) < 30 or not _calibration(K, R_world_cam, t_world_cam) or not math.isfinite(stamp):
            self.diagnostics['reason'] = 'invalid calibrated depth input'
            return []
        valid = np.isfinite(d) & (d >= p.min_range_m) & (d <= p.max_range_m)
        self.diagnostics['full_valid_fraction'] = float(valid.mean())
        if valid.mean() < p.min_full_depth_fraction:
            self.diagnostics['reason'] = 'full-frame valid depth below .30'
            return []
        R, t = np.asarray(R_world_cam, float), np.asarray(t_world_cam, float)
        fx, fy, cx, cy = K
        v, u = np.mgrid[2:d.shape[0]-2:3, 2:d.shape[1]-2:3]
        ds = d[v, u]
        ok = valid[v, u]
        rays = np.column_stack([(u[ok]-cx)/fx, (v[ok]-cy)/fy, np.ones(ok.sum())]) @ R.T
        points = rays * ds[ok, None] + t
        # Bound plane fitting independently of the camera's image size.
        if len(points) > 3500:
            points = points[np.linspace(0, len(points)-1, 3500).astype(int)]
        rng = np.random.default_rng(31)
        remaining = points
        found = []
        for _ in range(p.max_plane_models):
            if len(remaining) < p.min_plane_points:
                break
            best, count = None, 0
            for _ in range(80):
                xyz = remaining[rng.choice(len(remaining), 3, replace=False)]
                n = np.cross(xyz[1]-xyz[0], xyz[2]-xyz[0])
                norm = np.linalg.norm(n)
                if norm < 1e-6:
                    continue
                n /= norm
                if abs(n[2]) > math.sin(math.radians(p.max_frame_tilt_deg)):
                    continue
                mask = np.abs((remaining-xyz[0]) @ n) <= p.plane_tolerance_m
                if mask.sum() > count:
                    best, count = mask, int(mask.sum())
            if best is None or count < p.min_plane_points:
                break
            support = remaining[best]
            center = support.mean(axis=0)
            _, _, vt = np.linalg.svd(support-center, full_matrices=False)
            n = vt[-1]
            if np.dot(n, center-t) < 0:
                n = -n
            residual = float(np.sqrt(np.mean(((support-center) @ n)**2)))
            offset = float(center @ n)
            remaining = remaining[~best]
            if (abs(n[2]) > math.sin(math.radians(p.max_frame_tilt_deg)) or
                    residual > p.plane_residual_m or np.dot(n, R[:, 2]) < .5):
                continue
            self.diagnostics['planes'] += 1
            found.extend(self._on_plane(d, valid, K, R, t, n, offset, residual, stamp))
        # Multiple plane hypotheses can describe the same physical frame.
        found.sort(key=lambda a: (a.plane_residual_m, -a.behind_fraction))
        unique = []
        for item in found:
            if all(np.linalg.norm(np.asarray(item.center_world)-other.center_world) > .2 for other in unique):
                unique.append(item)
        self.diagnostics['candidates'] = len(unique)
        if not unique:
            self.diagnostics['reason'] = 'no four-sided depth-supported aperture'
        return unique

    def _on_plane(self, d, valid, K, R, t, n, offset, residual, stamp):
        p = self.p
        fx, fy, cx, cy = K
        h, w = d.shape
        # Horizontal right and vertical up on the fitted frame plane.
        right = np.cross(n, (0., 0., 1.))
        right /= np.linalg.norm(right)
        up = np.cross(right, n)
        origin = n * offset
        corners = np.array([[0., 0.], [w-1., 0.], [0., h-1.], [w-1., h-1.]])
        rays = np.column_stack([(corners[:, 0]-cx)/fx, (corners[:, 1]-cy)/fy, np.ones(4)]) @ R.T
        den = rays @ n
        if np.min(den) <= .15:
            return []
        scale = (offset-t@n) / den
        if np.any(scale <= p.min_range_m) or np.max(scale) > p.max_range_m:
            return []
        xyz = rays * scale[:, None] + t - origin
        xs, ys = xyz @ right, xyz @ up
        x0, y0 = float(xs.min()), float(ys.max())
        nw, nh = int(np.ceil(np.ptp(xs)/p.resolution_m))+1, int(np.ceil(np.ptp(ys)/p.resolution_m))+1
        if min(nw, nh) < 5 or max(nw, nh) > 700:
            return []
        vv, uu = np.mgrid[0:nh, 0:nw]
        world = origin + (x0+uu[..., None]*p.resolution_m)*right + (y0-vv[..., None]*p.resolution_m)*up
        cam = (world-t) @ R
        u = np.rint(fx*cam[..., 0]/cam[..., 2]+cx).astype(int)
        v = np.rint(fy*cam[..., 1]/cam[..., 2]+cy).astype(int)
        visible = (cam[..., 2] > 0) & (u >= 2) & (u < w-2) & (v >= 2) & (v < h-2)
        u, v = np.clip(u, 0, w-1), np.clip(v, 0, h-1)
        usable = visible & valid[v, u]
        rayn = ((world-t) @ n) / cam[..., 2]
        beyond = d[v, u]*rayn - (offset-t@n)
        near = usable & (np.abs(beyond) <= p.plane_tolerance_m)
        far = usable & (beyond >= p.observed_behind_m)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(far.astype(np.uint8), 8)
        band = max(2, int(math.ceil(p.frame_band_m/p.resolution_m)))
        reserve = p.resolution_m  # subtract one metric raster cell from each edge.
        found = []
        for label in range(1, count):
            x, y, bw, bh, pixels = map(int, stats[label])
            if pixels < 30 or min(bw, bh) < 6:
                continue
            if x < band or y < band or x+bw+band >= nw or y+bh+band >= nh:
                continue
            interior = labels[y:y+bh, x:x+bw] == label
            coverage = float(usable[y:y+bh, x:x+bw].mean())
            behind_fraction = float(interior.mean())
            if coverage < p.min_opening_depth_fraction or behind_fraction < p.min_opening_depth_fraction:
                self.diagnostics['rejected'].append('opening valid/behind depth coverage insufficient')
                continue
            sides = (near[y:y+bh, x-band:x], near[y:y+bh, x+bw:x+bw+band],
                     near[y-band:y, x:x+bw], near[y+bh:y+bh+band, x:x+bw])
            support = tuple(float(side.mean()) for side in sides)
            if min(support) < p.min_side_support:
                self.diagnostics['rejected'].append('one or more frame sides unobserved')
                continue
            width, height = bw*p.resolution_m-2*reserve, bh*p.resolution_m-2*reserve
            center_world = (origin + (x0+(x+(bw-1)/2)*p.resolution_m)*right +
                            (y0-(y+(bh-1)/2)*p.resolution_m)*up)
            fits = (width >= 2*(p.radius_m+p.margin_m) and
                    height >= 2*(p.half_height_m+p.margin_m))
            cc = (center_world-t) @ R
            box = (float(fx*cc[0]/cc[2]+cx), float(fy*cc[1]/cc[2]+cy),
                   float(width*fx/cc[2]), float(height*fy/cc[2]))
            found.append(Opening(tuple(center_world), tuple(n), tuple(right), tuple(up),
                                 float(width), float(height), float((center_world-t)@n), support,
                                 coverage, behind_fraction, residual, float(stamp), bool(fits), box))
        return found


class OpeningTracker:
    """Confirm a unique opening in world coordinates across distinct captures."""
    def __init__(self, params=None):
        self.p = params or OpeningParams()
        self.reset()

    def reset(self):
        self.history = []
        self.confirmed = None
        self.last_stamp = -math.inf
        self.reason = 'no stable measured aperture'

    def update(self, candidates, stamp):
        if not math.isfinite(stamp) or stamp <= self.last_stamp:
            return self.confirmed
        self.last_stamp = stamp
        good = [o for o in candidates if o.body_fits and abs(o.stamp-stamp) < 1e-6]
        if len(good) != 1:
            self.history, self.confirmed = [], None
            self.reason = 'ambiguous apertures' if len(good) > 1 else 'no aperture fits the vehicle envelope'
            return None
        item = good[0]
        if self.history:
            previous = self.history[-1]
            if (stamp-previous.stamp > self.p.max_age_s or
                    np.linalg.norm(np.asarray(item.center_world)-previous.center_world) > self.p.association_m or
                    abs(item.width_m-previous.width_m) > self.p.size_stability_m or
                    abs(item.height_m-previous.height_m) > self.p.size_stability_m or
                    np.dot(item.normal_world, previous.normal_world) < math.cos(math.radians(5.))):
                self.history = []
                self.confirmed = None
        self.history.append(item)
        self.history = self.history[-20:]
        if len(self.history) >= self.p.stable_frames and stamp-self.history[0].stamp >= self.p.stable_time_s:
            self.confirmed = item
            self.reason = 'stable four-sided measured aperture'
        else:
            self.reason = f'qualifying aperture: {len(self.history)} frames'
        return self.confirmed

    def current(self, now):
        if self.confirmed is None or not 0 <= now-self.confirmed.stamp <= self.p.max_age_s:
            return None
        return self.confirmed

    def status(self, now):
        current = self.current(now)
        return dict(reason=self.reason if current is not None else 'aperture stale or unconfirmed',
                    frames=len(self.history), confirmed=None if current is None else current.status())


@dataclass(frozen=True)
class ClearanceCertificate:
    """Returned by a separate fresh 3-D observed-space swept-volume checker.

    The checker must test the whole cylinder radius+margin and vertical
    half-height+margin along start->end, including braking reserve, against
    unknown and occupied volume. A boolean from a 2-D route is insufficient.
    """
    safe: bool
    stamp: float
    start_world: tuple
    end_world: tuple
    radius_m: float
    half_height_m: float
    reason: str = ''


@dataclass(frozen=True)
class CenteringCommand:
    phase: str
    requested_velocity: tuple  # world ENU vx, vy, vz, yaw rate
    authorized_velocity: tuple
    reason: str
    center_error_m: tuple
    aperture_center_world: tuple | None
    requires_3d_clearance: bool

    def status(self):
        return asdict(self)


class OpeningCenteringController:
    """Stop, face the frame, center in lateral/vertical axes, then pass slowly.

    Only the explicit ``traverse=True`` caller request permits forward motion.
    Call every control tick with a fresh tracked aperture and authority state.
    Every nonzero XYZ command requires a matching current clearance certificate.
    """
    def __init__(self, params=None):
        self.p = params or OpeningParams()
        self.reset()

    def reset(self):
        self.settled = False
        self.aperture_center = None

    def command(self, opening, position, yaw, velocity, now, *, authority,
                certify_sweep: Callable | None = None, traverse=False):
        p = self.p
        zero = (0., 0., 0., 0.)
        error = (0., 0.)
        center = None if opening is None else opening.center_world

        def result(phase, requested=zero, authorized=zero, reason='', requires=False):
            return CenteringCommand(phase, tuple(map(float, requested)), tuple(map(float, authorized)),
                                    reason, error, center, requires)

        pos, vel = np.asarray(position, float), np.asarray(velocity, float)
        if (not authority or pos.shape != (3,) or vel.shape != (3,) or
                not np.isfinite(pos).all() or not np.isfinite(vel).all() or
                not all(math.isfinite(x) for x in (yaw, now))):
            self.reset()
            return result('hold', reason='missing authority or fresh finite pose')
        if opening is None or not 0 <= now-opening.stamp <= p.max_age_s:
            self.reset()
            return result('hold', reason='aperture stale or unconfirmed')
        if not opening.body_fits:
            self.reset()
            return result('hold', reason='opening smaller than full vehicle envelope and margin')
        if (self.aperture_center is None or
                np.linalg.norm(np.asarray(center)-self.aperture_center) > p.association_m):
            self.settled = False
            self.aperture_center = np.asarray(center)
        n, right, up = map(lambda x: np.asarray(x, float),
                           (opening.normal_world, opening.right_world, opening.up_world))
        delta = np.asarray(opening.center_world)-pos
        lateral, vertical = float(delta@right), float(delta@up)
        error = lateral, vertical
        desired_yaw = math.atan2(n[1], n[0])
        yaw_error = math.atan2(math.sin(desired_yaw-yaw), math.cos(desired_yaw-yaw))
        if not self.settled and np.linalg.norm(vel) > p.stop_speed_mps:
            return result('stop', reason='stop before changing alignment or entering a gap')
        self.settled = True
        if abs(yaw_error) > p.yaw_tolerance_rad:
            if np.linalg.norm(vel) > p.stop_speed_mps:
                self.settled = False
                return result('stop', reason='stop before changing frame alignment')
            request = (0., 0., 0., float(np.clip(yaw_error, -p.yaw_speed_rps, p.yaw_speed_rps)))
            return result('align', request, request, 'face the measured frame normal')
        centered = abs(lateral) <= p.center_tolerance_m and abs(vertical) <= p.center_tolerance_m
        if centered and not traverse:
            return result('centered', reason='centered; awaiting explicit traversal request')
        if centered:
            xyz, phase = n*p.traverse_speed_mps, 'traverse'
        else:
            xyz = np.clip(lateral, -p.center_speed_mps, p.center_speed_mps)*right
            xyz += np.clip(vertical, -p.center_speed_mps, p.center_speed_mps)*up
            norm = np.linalg.norm(xyz)
            if norm > p.center_speed_mps:
                xyz *= p.center_speed_mps/norm
            phase = 'center'
        requested = (*xyz, 0.)
        # Certify at least the proposed lookahead and braking/stopping distance.
        distance = np.linalg.norm(xyz)*p.lookahead_s + np.linalg.norm(xyz)**2/(2*.5)
        endpoint = pos + xyz/max(np.linalg.norm(xyz), 1e-9)*distance
        if certify_sweep is None:
            self.settled = False
            return result(phase, requested, reason='fresh swept 3-D clearance checker unavailable', requires=True)
        try:
            cert = certify_sweep(tuple(pos), tuple(endpoint), p.radius_m+p.margin_m,
                                 p.half_height_m+p.margin_m, now)
        except Exception as exc:
            self.settled = False
            return result(phase, requested, reason=f'3-D clearance checker failed: {type(exc).__name__}', requires=True)
        try:
            matches = (isinstance(cert, ClearanceCertificate) and cert.safe and
                       0 <= now-cert.stamp <= p.max_age_s and
                       np.shape(cert.start_world) == (3,) and np.shape(cert.end_world) == (3,) and
                       np.allclose(cert.start_world, pos, atol=1e-6) and
                       np.allclose(cert.end_world, endpoint, atol=1e-6) and
                       math.isfinite(cert.radius_m) and math.isfinite(cert.half_height_m) and
                       cert.radius_m >= p.radius_m+p.margin_m and
                       cert.half_height_m >= p.half_height_m+p.margin_m)
        except (TypeError, ValueError, AttributeError):
            matches = False
        if not matches:
            self.settled = False
            reason = cert.reason if isinstance(cert, ClearanceCertificate) and cert.reason else 'invalid/stale/mismatched swept-volume certificate'
            return result(phase, requested, reason=reason, requires=True)
        return result(phase, requested, requested, 'fresh swept 3-D clearance certified', requires=True)
