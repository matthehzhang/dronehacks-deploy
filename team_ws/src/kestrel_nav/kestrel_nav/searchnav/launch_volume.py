"""One-shot pilot-verified launch AABB, separate from measured occupancy.

The volume is an explicit temporary operator assertion in a fixed local ENU
frame. It cannot grow, follow the vehicle, clear depth hits, or regrant itself
after a local-frame reset. Callers may use its mask for UNKNOWN cells only;
observed OCCUPIED cells always win. Raw published occupancy stays measured.
"""
from dataclasses import dataclass
import math
import uuid

import numpy as np


@dataclass(frozen=True)
class LaunchVolumeParams:
    enabled: bool = False
    relative_bounds: tuple | None = None
    duration_s: float = 30.
    max_horizontal_span_m: float = 3.
    max_vertical_span_m: float = 3.

    def __post_init__(self):
        if (not isinstance(self.enabled, bool) or not math.isfinite(self.duration_s) or
                not 0 < self.duration_s <= 60. or
                not math.isfinite(self.max_horizontal_span_m) or
                not 0 < self.max_horizontal_span_m <= 3. or
                not math.isfinite(self.max_vertical_span_m) or
                not 0 < self.max_vertical_span_m <= 3.):
            raise ValueError('launch assertion must expire within 60s and span at most 3m per axis')
        if self.relative_bounds is not None:
            bounds = np.asarray(self.relative_bounds, float)
            if bounds.shape != (6,) or not np.isfinite(bounds).all():
                raise ValueError('relative bounds require finite xmin,ymin,zmin,xmax,ymax,zmax')
            span = bounds[3:]-bounds[:3]
            if (np.any(span <= 0) or np.any(span[:2] > self.max_horizontal_span_m) or
                    span[2] > self.max_vertical_span_m):
                raise ValueError('launch assertion bounds must have positive bounded total spans')
            # The declared volume describes launch space, not a remote route.
            if np.any(bounds[:3] > 0) or np.any(bounds[3:] < 0):
                raise ValueError('launch assertion bounds must contain their fixed launch origin')
            object.__setattr__(self, 'relative_bounds', tuple(map(float, bounds)))


class LaunchVolume:
    def __init__(self, params=None):
        self.p = params or LaunchVolumeParams()
        self.world_bounds = None
        self.origin = None
        self.granted_at = self.expires_at = None
        self.last_now = None
        self.grant_attempted = False
        self.revoked_reason = ''
        self.provenance = None
        self.provenance_id = None
        self.pilot_verified = False
        self.active = False
        self.revision = 0

    def grant(self, origin_xyz, now, *, pilot_verified=False, saw_disarmed=False,
              allow_takeoff=False, pilot_started=False,
              provenance='pilot-verified launch volume'):
        """Consume the one explicit launch grant; never called from a reset.

        The caller supplies independently observed lifecycle states. Rejected
        grants are consumed too, so a changing sensor/authority state cannot
        silently turn an old approval attempt into a later automatic grant.
        A fresh explicit pilot approval needs a deliberately new object.
        """
        if self.grant_attempted:
            self.revoke('launch volume grant already consumed; fresh explicit pilot approval required')
            return False
        self.grant_attempted = True
        self.revision += 1
        try:
            origin = np.asarray(origin_xyz, float)
            valid_clock = (math.isfinite(now) and now > 0 and
                           (self.last_now is None or now >= self.last_now))
        except (TypeError, ValueError):
            origin, valid_clock = np.empty(0), False
        if not self.p.enabled or self.p.relative_bounds is None:
            self.revoked_reason = 'pilot-verified launch volume disabled or bounds absent'
            return False
        if not all(x is True for x in (pilot_verified, saw_disarmed, allow_takeoff, pilot_started)):
            self.revoked_reason = 'explicit pilot verification, disarmed history, launch approval and pilot start required'
            return False
        if origin.shape != (3,) or not np.isfinite(origin).all() or not valid_clock:
            self.revoked_reason = 'finite fixed launch origin and positive local clock required'
            return False
        if not isinstance(provenance, str) or not provenance.strip() or len(provenance) > 160:
            self.revoked_reason = 'explicit bounded operator-verification provenance required'
            return False
        relative = np.asarray(self.p.relative_bounds)
        self.origin = tuple(map(float, origin))
        self.world_bounds = tuple(map(float, np.r_[origin+relative[:3], origin+relative[3:]]))
        self.granted_at = self.last_now = float(now)
        self.expires_at = float(now+self.p.duration_s)
        self.provenance = provenance.strip()
        self.provenance_id = 'pilot-launch-'+uuid.uuid4().hex
        self.pilot_verified, self.active, self.revoked_reason = True, True, ''
        return True

    def revoke(self, reason='launch assertion revoked'):
        if self.active:
            self.revision += 1
        self.active = False
        self.grant_attempted = True
        self.revoked_reason = str(reason)[:240]

    def reset(self, reason='local frame reset'):
        self.revoke(reason)

    def _check(self, now):
        try:
            finite = math.isfinite(now)
        except (TypeError, ValueError):
            finite = False
        if not finite:
            self.revoke('nonfinite local clock')
            return False
        if self.last_now is not None and now < self.last_now:
            self.revoke('local clock moved backwards')
            return False
        self.last_now = float(now)
        if self.active and now >= self.expires_at:
            self.revoke('pilot-verified launch volume expired')
        return self.active

    def unknown_mask(self, points_xyz, now, voxel_half_extents=(0., 0., 0.)):
        """Mask only fully contained point/voxel volumes; apply to UNKNOWN only.

        No raw map states are accepted or modified here. ``voxel_half_extents``
        must include the caller's raster/interpolation reserve on every axis.
        A partly contained voxel is excluded, even if its center is inside.
        """
        points = np.asarray(points_xyz, float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('points_xyz must be Nx3')
        mask = np.zeros(len(points), bool)
        if not self._check(now):
            return mask
        reserve = np.asarray(voxel_half_extents, float)
        if reserve.shape == ():
            reserve = np.repeat(reserve, 3)
        if reserve.shape != (3,) or not np.isfinite(reserve).all() or np.any(reserve < 0):
            raise ValueError('voxel_half_extents must be finite nonnegative XYZ reserves')
        bounds = np.asarray(self.world_bounds)
        return (np.isfinite(points).all(axis=1) &
                np.all(points-reserve >= bounds[:3], axis=1) &
                np.all(points+reserve <= bounds[3:], axis=1))

    def status(self, now):
        active = self._check(now)
        return dict(enabled=self.p.enabled, active=bool(active), revision=self.revision,
                    operator_verified=self.pilot_verified, grant_attempted=self.grant_attempted,
                    origin_xyz=self.origin, relative_bounds=self.p.relative_bounds,
                    world_bounds=self.world_bounds, bounds=self.world_bounds,
                    granted_at=self.granted_at, expires_at=self.expires_at,
                    remaining_s=max(0., self.expires_at-now) if active else 0.,
                    provenance=self.provenance, provenance_id=self.provenance_id,
                    revoked_reason=self.revoked_reason,
                    application='UNKNOWN voxels only; observed obstacles always win; raw map unchanged')
