"""Calibrated MONO floor-contrast candidates, explicitly not color recognition.

A blue patch and a shadow can have identical monochrome pixels. Geometric and
temporal checks verify a real floor patch, never prove that its color is blue.
Promotion to a navigation goal is opt-in after recorded camera qualification.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import cv2
import numpy as np


@dataclass
class Candidate:
    xy: tuple
    area_m2: float
    contrast: float
    box: tuple
    depth_error_m: float
    reason: str = 'floor contrast; color unobserved'
    source: str = 'mono'


@dataclass(frozen=True)
class DetectorParams:
    min_contrast: float = 25.
    min_area_m2: float = .04
    max_area_m2: float = 2.0
    depth_tolerance_m: float = .12
    min_depth_support: float = .7
    min_frames: int = 4
    minimum_track_s: float = .6
    viewpoint_baseline_m: float = .20
    association_m: float = .20
    track_expiry_s: float = 2.0
    qualified: bool = False


def floor_intersections(uv, K, R_world_cam, t_world_cam, floor_z, max_range=5.):
    uv = np.asarray(uv, float)
    fx, fy, cx, cy = K
    rays = np.column_stack([(uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy, np.ones(len(uv))])
    rays = rays @ np.asarray(R_world_cam, float).T
    t = np.asarray(t_world_cam, float)
    denom = rays[:, 2]
    scale = np.full(len(uv), np.nan)
    ok = denom < -.03
    scale[ok] = (floor_z - t[2]) / denom[ok]
    ok &= (scale > .1) & (scale < max_range)
    return t + rays * scale[:, None], ok


def validate_depth(world_points, depth, K, R_world_depth, t_world_depth, tolerance=.12):
    """Mono floor projections reprojected into their OWN calibrated depth camera."""
    pc = (np.asarray(world_points) - np.asarray(t_world_depth)) @ np.asarray(R_world_depth)
    fx, fy, cx, cy = K
    den = np.maximum(pc[:, 2], 1e-6)
    uv = np.rint(np.column_stack([fx * pc[:, 0] / den + cx, fy * pc[:, 1] / den + cy])).astype(int)
    h, w = depth.shape
    errors = []
    for p, (u, v) in zip(pc, uv):
        if p[2] <= .1 or not 2 <= u < w - 2 or not 2 <= v < h - 2:
            errors.append(math.inf)
            continue
        patch = depth[v - 2:v + 3, u - 2:u + 3]
        good = patch[np.isfinite(patch) & (patch > .1)]
        if len(good) < 15:
            errors.append(math.inf)
            continue
        errors.append(abs(float(np.median(good)) - p[2]))
    errors = np.asarray(errors)
    return float(np.mean(errors < tolerance)), float(np.median(errors))


class MonoFloorDetector:
    def __init__(self, params=None):
        self.p = params or DetectorParams()

    def detect(self, image, K_mono, R_world_mono, t_world_mono, floor_z,
               depth, K_depth, R_world_depth, t_world_depth):
        """Inputs must be time-matched, undistorted mono and calibrated depth."""
        mono = np.asarray(image)
        if mono.ndim != 2 or mono.dtype != np.uint8 or depth is None:
            return []
        h, w = mono.shape
        # Only pixels projected down onto the floor are candidates. This also
        # rejects sky/wall contrast before geometric depth verification.
        vs, us = np.mgrid[0:h, 0:w]
        world, floor_ok = floor_intersections(np.column_stack([us.ravel(), vs.ravel()]), K_mono,
                                             R_world_mono, t_world_mono, floor_z)
        floor_ok = floor_ok.reshape(h, w)
        if floor_ok.sum() < 200:
            return []
        # Median local background rejects an exposure gradient while preserving
        # compact contrast regions. We deliberately do not call dark pixels blue.
        kernel = min(151, max(31, (min(h, w) // 3) | 1))
        background = cv2.medianBlur(mono, kernel)
        # Sensor response, exposure and blue paint can make the target either
        # brighter OR darker than red floor; never hard-code color ordering.
        contrast = np.abs(background.astype(np.float32) - mono.astype(np.float32))
        mask = ((contrast >= self.p.min_contrast) & floor_ok).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []
        for contour in contours:
            area_px = cv2.contourArea(contour)
            if area_px < 40:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            if x < 3 or y < 3 or x + bw >= w - 3 or y + bh >= h - 3:
                continue  # need surrounding floor, not a truncated dark wall.
            cmask = np.zeros_like(mask)
            cv2.drawContours(cmask, [contour], -1, 255, -1)
            pixels = np.argwhere(cmask > 0)
            if len(pixels) < 40:
                continue
            sample = pixels[::max(1, len(pixels) // 40)]
            wp, ok = floor_intersections(sample[:, ::-1], K_mono, R_world_mono, t_world_mono, floor_z)
            if ok.mean() < .95:
                continue
            wp = wp[ok]
            hull = cv2.convexHull(wp[:, :2].astype(np.float32))
            area_m2 = float(cv2.contourArea(hull))
            if not self.p.min_area_m2 <= area_m2 <= self.p.max_area_m2:
                continue
            support, error = validate_depth(wp, depth, K_depth, R_world_depth, t_world_depth,
                                            self.p.depth_tolerance_m)
            if support < self.p.min_depth_support or error > self.p.depth_tolerance_m:
                continue
            xy = np.median(wp[:, :2], axis=0)
            results.append(Candidate(tuple(map(float, xy)), area_m2,
                                     float(np.median(contrast[cmask > 0])), (x, y, bw, bh), error))
        return results


class TargetTracker:
    def __init__(self, params=None, *, source='mono', color_observed=False):
        self.p = params or DetectorParams()
        self.source, self.color_observed = source, color_observed
        self.tracks = []
        self.confirmed = None
        self.last_stamp = -math.inf

    def update(self, candidates, camera_xy, stamp):
        if not math.isfinite(stamp) or stamp <= self.last_stamp:
            return self.confirmed
        self.last_stamp = stamp
        self.tracks = [t for t in self.tracks if stamp - t['last'] <= self.p.track_expiry_s]
        used = set()
        for candidate in candidates:
            xy = np.asarray(candidate.xy, float)
            matches = [(np.linalg.norm(t['xy'] - xy), i) for i, t in enumerate(self.tracks)
                       if i not in used and np.linalg.norm(t['xy'] - xy) < self.p.association_m]
            if matches:
                _, i = min(matches)
                track = self.tracks[i]
                track['xy'] = .7 * track['xy'] + .3 * xy
                track['count'] += 1
                track['last'] = stamp
                track['views'].append(np.asarray(camera_xy, float))
            else:
                i = len(self.tracks)
                track = dict(xy=xy, first=stamp, last=stamp, count=1,
                             views=[np.asarray(camera_xy, float)], geometric=False)
                self.tracks.append(track)
            used.add(i)
            # Need actual translation, not just repeated frames or yaw at a
            # single optical origin. Camera offset around a yaw is insufficient
            # if the measured baseline remains below the threshold.
            baseline = max(np.linalg.norm(v - track['views'][0]) for v in track['views'])
            track['geometric'] = bool(track['count'] >= self.p.min_frames
                                      and stamp - track['first'] >= self.p.minimum_track_s
                                      and baseline >= self.p.viewpoint_baseline_m)
            track['views'] = track['views'][-100:]
        eligible = [t for t in self.tracks if t['geometric'] and stamp - t['last'] < .5]
        # More than one equally valid floor patch is ambiguous, even after
        # optical qualification. No nearest/darkest guess becomes the target.
        self.confirmed = ((tuple(map(float, eligible[0]['xy'])), eligible[0]['last'])
                          if self.p.qualified and len(eligible) == 1 else None)
        return self.confirmed

    def status(self):
        return dict(source=self.source, color_observed=self.color_observed, contrast_qualified=self.p.qualified,
                    confirmed=self.confirmed,
                    tracks=[dict(xy=list(map(float, t['xy'])), frames=t['count'],
                                 geometric=t['geometric'], last=t['last']) for t in self.tracks],
                    reason=('HSV blue with calibrated floor/depth support; multiple patches remain ambiguous'
                            if self.color_observed else
                            'mono cannot independently distinguish blue from a shadow or dark floor marking'))


@dataclass(frozen=True)
class HSVParams:
    # OpenCV hue is 0..179. Broad initial blue/cyan band; retain per-run mask
    # for lighting/exposure qualification rather than learning from map truth.
    hue_low: int = 90
    hue_high: int = 135
    saturation_min: int = 80
    value_min: int = 40

    def __post_init__(self):
        if not 0 <= self.hue_low < self.hue_high <= 179:
            raise ValueError('invalid OpenCV blue hue bounds')
        if not 1 <= self.saturation_min <= 255 or not 1 <= self.value_min <= 255:
            raise ValueError('invalid HSV saturation/value bounds')


def decode_bgr(message):
    """Explicit color encodings only; never reinterpret a mono image as RGB."""
    if message.encoding.lower() not in ('rgb8', 'bgr8', 'rgba8', 'bgra8'):
        raise ValueError('blue detector requires rgb8/bgr8/rgba8/bgra8')
    channels = 4 if 'a8' in message.encoding.lower() else 3
    if message.step < message.width * channels:
        raise ValueError('invalid RGB image stride')
    rows = np.frombuffer(message.data, np.uint8).reshape(message.height, message.step)
    image = rows[:, :message.width*channels].reshape(message.height, message.width, channels)
    conversions = {'rgb8':cv2.COLOR_RGB2BGR, 'rgba8':cv2.COLOR_RGBA2BGR,
                   'bgra8':cv2.COLOR_BGRA2BGR}
    return cv2.cvtColor(image, conversions[message.encoding.lower()]) if message.encoding.lower() in conversions else image.copy()


class RGBBlueDetector:
    """Blue-color proposals verified on the measured floor with separate depth TF.

    A blue wall, sky patch or raised object must not become the floor end target.
    The tracker still requires independent views and rejects multiple blue patches.
    """
    def __init__(self, params=None, hsv=None):
        self.p = params or DetectorParams(qualified=True)
        self.hsv = hsv or HSVParams()
        self.last_mask = None

    def detect(self, bgr, K_rgb, R_rgb, t_rgb, floor_z, depth, K_depth, R_depth, t_depth):
        if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError('expected uint8 BGR image')
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        p = self.hsv
        mask = cv2.inRange(hsv, (p.hue_low, p.saturation_min, p.value_min), (p.hue_high, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        self.last_mask = mask
        if floor_z is None or depth is None:
            return []
        h, w = mask.shape
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = []
        for contour in contours:
            if cv2.contourArea(contour) < 40:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            if x < 3 or y < 3 or x+bw >= w-3 or y+bh >= h-3:
                continue
            region = np.zeros_like(mask)
            cv2.drawContours(region, [contour], -1, 255, -1)
            pixels = np.argwhere(region > 0)
            samples = pixels[::max(1, len(pixels)//40)]
            wp, ok = floor_intersections(samples[:, ::-1], K_rgb, R_rgb, t_rgb, floor_z)
            if ok.mean() < .95:
                continue
            wp = wp[ok]
            area = float(cv2.contourArea(cv2.convexHull(wp[:, :2].astype(np.float32))))
            if not self.p.min_area_m2 <= area <= self.p.max_area_m2:
                continue
            support, error = validate_depth(wp, depth, K_depth, R_depth, t_depth,
                                            self.p.depth_tolerance_m)
            if support < self.p.min_depth_support or error > self.p.depth_tolerance_m:
                continue
            xy = np.median(wp[:, :2], axis=0)
            result.append(Candidate(tuple(map(float, xy)), area,
                float(np.median(hsv[:, :, 1][region > 0])), (x,y,bw,bh), error,
                reason='HSV blue with calibrated floor/depth support', source='rgb_hsv'))
        return result
