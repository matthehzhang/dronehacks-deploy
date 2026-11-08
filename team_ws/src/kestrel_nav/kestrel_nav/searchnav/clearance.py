"""Bounded calibrated-depth evidence for small swept 3-D centering moves.

This is a conservative sampled-volume checker, not a source of assumed free
launch space. Unknown, out-of-view, invalid and expired volume blocks a query.
The .05m XY/.10m Z cover includes a raster reserve around the whole vehicle.
"""
from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from .opening import ClearanceCertificate, _calibration


@dataclass(frozen=True)
class ClearanceParams:
    xy_resolution_m: float = .05
    z_resolution_m: float = .10
    depth_error_m: float = .08
    min_range_m: float = .30
    max_range_m: float = 12.
    ttl_s: float = 20.
    max_age_s: float = .5
    full_valid_fraction: float = .30
    local_valid_fraction: float = .60
    clear_frames: int = 3
    max_frames: int = 640
    max_frame_pixels: int = 19200
    max_query_voxels: int = 20000
    max_cached_voxels: int = 80000
    max_move_m: float = .5

    def __post_init__(self):
        if (self.xy_resolution_m > .05 or self.z_resolution_m > .10 or
                self.depth_error_m < .08 or self.ttl_s > 20. or self.max_age_s > .5 or
                self.full_valid_fraction < .30 or self.local_valid_fraction < .60 or
                self.clear_frames < 3 or self.max_frames < 3):
            raise ValueError('clearance evidence requirements may not be weakened')
        if min(self.xy_resolution_m, self.z_resolution_m, self.max_age_s, self.ttl_s) <= 0:
            raise ValueError('invalid clearance geometry/freshness')


class DepthClearanceHistory:
    """Accept all calibrated captures; certify bounded local sweeps on demand.

    ``integrate(depth, K, R, t, capture_stamp, receipt_stamp)`` stores a copy,
    using nearest-hit conservative block reduction for large camera frames.
    ``certify(start, end, radius, half_height, now)`` has the callback signature
    expected by OpeningCenteringController. Call ``reset()`` on every EKF/local
    frame epoch change. No preloaded map, vehicle truth or launch bubble enters.
    """
    def __init__(self, params=None):
        self.p = params or ClearanceParams()
        self.launch_volume = None
        self.reset()

    def reset(self):
        self.frames = deque(maxlen=self.p.max_frames)
        self.capture = self.receipt = None
        self.error = 'no current calibrated depth evidence'
        self.diagnostics = dict(reason=self.error, frames=0)
        self.cache = {}

    def integrate(self, depth_m, K, R_world_cam, t_world_cam, capture_stamp, receipt_stamp):
        p = self.p
        d = np.asarray(depth_m, float)
        if (d.ndim != 2 or min(d.shape) < 3 or not _calibration(K, R_world_cam, t_world_cam) or
                not all(math.isfinite(s) for s in (capture_stamp, receipt_stamp)) or
                not -.05 <= receipt_stamp-capture_stamp <= p.max_age_s or
                self.capture is not None and capture_stamp <= self.capture):
            self.error = 'invalid, old or uncalibrated depth capture'
            return False
        valid = np.isfinite(d) & (d >= p.min_range_m) & (d <= p.max_range_m)
        if valid.mean() < p.full_valid_fraction:
            self.error = 'full-frame valid depth below .30'
            return False
        factor = max(1, int(math.ceil(math.sqrt(d.size/p.max_frame_pixels))))
        k = np.asarray(K, float).copy()
        if factor > 1:
            # One close hit wins an entire reduced pixel. Invalid support never
            # becomes a depth value through interpolation or hole filling.
            h, w = d.shape
            nh, nw = h//factor, w//factor
            patches = d[:nh*factor, :nw*factor].reshape(nh, factor, nw, factor)
            good = valid[:nh*factor, :nw*factor].reshape(nh, factor, nw, factor)
            support = good.mean(axis=(1, 3))
            d = np.min(np.where(good, patches, np.inf), axis=(1, 3))
            d[support < p.local_valid_fraction] = np.nan
            k[:2] /= factor
            k[2:] = (k[2:]-(factor-1)/2)/factor
        else:
            d = np.where(valid, d, np.nan)
        self.frames.append(dict(depth=d.astype(np.float32, copy=True), K=k,
                                R=np.array(R_world_cam, float, copy=True),
                                t=np.array(t_world_cam, float, copy=True), stamp=float(capture_stamp)))
        self.capture, self.receipt, self.error = float(capture_stamp), float(receipt_stamp), ''
        while self.frames and capture_stamp-self.frames[0]['stamp'] > p.ttl_s:
            self.frames.popleft()
        return True

    def replay_frames(self, now):
        """Yield chronological calibrated frames, with ``full`` at <=4Hz.

        Every intervening raw capture remains a hit-only update. The live caller
        should queue these and process bounded work while commanding zero, then
        resume A* after rebuilding. Its occupancy integrator must ignore older
        queued free evidence after a newer live hit at the same voxel.
        """
        last = -math.inf
        for frame in self.frames:
            if not 0 <= now-frame['stamp'] <= self.p.ttl_s:
                continue
            full = frame['stamp']-last >= .25
            if full:
                last = frame['stamp']
            yield dict(frame, full=full)

    def replay_into(self, grid, now):
        """Offline synchronous replay; live controllers use replay_frames()."""
        full_count = 0
        for frame in self.replay_frames(now):
            if frame['full']:
                grid.integrate(frame['depth'], frame['K'], frame['R'], frame['t'], frame['stamp'])
                full_count += 1
            else:
                grid.integrate_occupied(frame['depth'], frame['K'], frame['R'], frame['t'], frame['stamp'])
        grid.expire(now)
        return full_count

    def _samples(self, start, end, radius, half_height):
        p = self.p
        xy_pad = radius+p.xy_resolution_m*math.sqrt(2)/2
        z_pad = half_height+p.z_resolution_m/2
        scale = np.array([p.xy_resolution_m, p.xy_resolution_m, p.z_resolution_m])
        lo = np.floor((np.minimum(start, end)-[xy_pad, xy_pad, z_pad])/scale).astype(int)
        hi = np.ceil((np.maximum(start, end)+[xy_pad, xy_pad, z_pad])/scale).astype(int)
        if np.prod(hi-lo+1, dtype=np.int64) > p.max_query_voxels:
            return None
        cells = np.stack(np.meshgrid(*(np.arange(a, b+1) for a, b in zip(lo, hi)), indexing='ij'), axis=-1).reshape(-1, 3)
        points = cells*scale  # fixed global lattice; never tied to the drone pose.
        delta = end[:2]-start[:2]
        length2 = delta@delta
        alpha = np.clip((points[:, :2]-start[:2])@delta/max(length2, 1e-12), 0., 1.)
        closest = start[:2]+alpha[:, None]*delta
        mask = (np.linalg.norm(points[:, :2]-closest, axis=1) <= xy_pad)
        mask &= (points[:, 2] >= min(start[2], end[2])-z_pad)
        mask &= (points[:, 2] <= max(start[2], end[2])+z_pad)
        return points[mask]

    def certify(self, start_world, end_world, radius_m, half_height_m, now):
        p = self.p
        start, end = np.asarray(start_world, float), np.asarray(end_world, float)
        original_start, original_end = tuple(start_world), tuple(end_world)

        def result(safe, reason, **details):
            self.diagnostics = dict(safe=bool(safe), reason=reason, frames=len(self.frames),
                                    current_capture_age_s=None if self.capture is None else now-self.capture,
                                    current_receipt_age_s=None if self.receipt is None else now-self.receipt,
                                    **details)
            return ClearanceCertificate(bool(safe), float(now), original_start, original_end,
                                        float(radius_m), float(half_height_m), reason)

        if (start.shape != (3,) or end.shape != (3,) or not np.isfinite(start).all() or
                not np.isfinite(end).all() or not math.isfinite(now) or
                not math.isfinite(radius_m) or not math.isfinite(half_height_m) or
                radius_m < .294082845+.12 or half_height_m < .4+.12):
            return result(False, 'invalid sweep or reduced vehicle envelope')
        if self.error:
            return result(False, self.error)
        if (self.capture is None or self.receipt is None or
                not 0 <= now-self.capture <= p.max_age_s or
                not 0 <= now-self.receipt <= p.max_age_s):
            return result(False, 'current calibrated depth capture/receipt stale')
        if np.linalg.norm(end-start) > p.max_move_m:
            return result(False, 'requested sweep exceeds bounded local lookahead')
        points = self._samples(start, end, radius_m, half_height_m)
        if points is None or not len(points):
            return result(False, 'swept volume exceeds bounded voxel query')
        scale = np.array([p.xy_resolution_m, p.xy_resolution_m, p.z_resolution_m])
        keys = list(map(tuple, np.rint(points/scale).astype(int)))
        if len(self.cache)+len(keys) > p.max_cached_voxels:
            self.cache.clear()  # bounded; rebuilding from retained evidence is conservative.
        records = np.array([self.cache.get(key, (-1, 0, -math.inf, -math.inf)) for key in keys], float)
        states, counts, seen, processed = records.T.copy()
        reserve = .5*math.sqrt(2*p.xy_resolution_m**2+p.z_resolution_m**2)+p.depth_error_m
        for frame in self.frames:
            if not 0 <= now-frame['stamp'] <= p.ttl_s:
                continue
            pending = frame['stamp'] > processed
            if not pending.any():
                continue
            expired = pending & (frame['stamp']-seen > p.ttl_s)
            states[expired], counts[expired] = -1, 0
            candidate_ids = np.flatnonzero(pending)
            processed[pending] = frame['stamp']
            cam = (points[candidate_ids]-frame['t'])@frame['R']
            fx, fy, cx, cy = frame['K']
            positive = cam[:, 2] > p.min_range_m
            denominator = np.maximum(cam[:, 2], 1e-6)
            u = np.rint(fx*cam[:, 0]/denominator+cx).astype(int)
            v = np.rint(fy*cam[:, 1]/denominator+cy).astype(int)
            depth = frame['depth']
            h, w = depth.shape
            inside = positive & (u >= 1) & (u < w-1) & (v >= 1) & (v < h-1)
            local_ids = np.flatnonzero(inside)
            if not len(local_ids):
                continue
            ids = candidate_ids[local_ids]
            patch = np.stack([depth[v[local_ids]+dy, u[local_ids]+dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=1)
            valid = np.isfinite(patch)
            supported = valid.mean(axis=1) >= p.local_valid_fraction
            measured = np.min(np.where(valid, patch, np.inf), axis=1)
            z = cam[local_ids, 2]
            hit = supported & (np.abs(measured-z) <= reserve)
            free = supported & (z+reserve < measured)
            hit_ids, free_ids = ids[hit], ids[free]
            states[hit_ids], counts[hit_ids], seen[hit_ids] = 100, 0, frame['stamp']
            counts[free_ids] = np.minimum(counts[free_ids]+1, p.clear_frames)
            seen[free_ids] = frame['stamp']
            clear = free_ids[counts[free_ids] >= p.clear_frames]
            states[clear], seen[clear] = 0, frame['stamp']
            # Unknown/occluded observations cannot clear a previous hit.
        states[now-seen > p.ttl_s] = -1
        for key, state, count, observed, done in zip(keys, states, counts, seen, processed):
            self.cache[key] = (state, count, observed, done)
        occupied, unknown = int(np.sum(states == 100)), int(np.sum(states == -1))
        if occupied and self.launch_volume is not None:
            contradicted=self.launch_volume.unknown_mask(points[states==100],now,scale/2)
            if contradicted.any():
                self.launch_volume.revoke('depth obstacle contradicts pilot-verified 3-D launch volume')
        pilot_supported = np.zeros(len(points),bool)
        if self.launch_volume is not None:
            pilot_supported = (states == -1) & self.launch_volume.unknown_mask(points,now,scale/2)
        remaining_unknown = int(np.count_nonzero((states == -1)&~pilot_supported))
        if occupied:
            return result(False, 'occupied voxel intersects swept vehicle volume',
                          samples=len(points), occupied=occupied, unknown=unknown)
        if unknown:
            if remaining_unknown:
                return result(False, 'unknown/unsupported voxel intersects swept vehicle volume',
                    samples=len(points), occupied=0, unknown=remaining_unknown,
                    pilot_verified_samples=int(pilot_supported.sum()))
        return result(True, ('fresh depth and explicit pilot launch volume support sampled swept 3-D vehicle volume'
                             if pilot_supported.any() else
                             'fresh calibrated depth supports sampled swept 3-D vehicle volume'),
                      samples=len(points), occupied=0, unknown=0,pilot_verified_samples=int(pilot_supported.sum()))

    def status(self, now=None):
        result = dict(self.diagnostics, stored_bytes=sum(f['depth'].nbytes for f in self.frames),
                      max_frames=self.p.max_frames, ttl_s=self.p.ttl_s,
                      cached_voxels=len(self.cache),
                      depth_error=self.error,
                      xy_resolution_m=self.p.xy_resolution_m, z_resolution_m=self.p.z_resolution_m)
        if now is not None:
            result.update(current_capture_age_s=None if self.capture is None else now-self.capture,
                          current_receipt_age_s=None if self.receipt is None else now-self.receipt,
                          current_usable=bool(not self.error and self.capture is not None and self.receipt is not None
                                              and 0 <= now-self.capture <= self.p.max_age_s
                                              and 0 <= now-self.receipt <= self.p.max_age_s))
        return result
