"""Calibrated adaptive view assessment for an already certified motion route.

This supplements the occupancy/swept-volume certificate. A farther assessment
plane can put the unchanged vehicle cross-section inside a real camera's FOV;
it cannot certify the unknown volume between the vehicle and that plane. The
route certificate must provide that independent near-body safety evidence.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

from .opening import _calibration


@dataclass(frozen=True)
class CorridorParams:
    radius_m: float = .294082845
    margin_m: float = .12
    half_height_m: float = .4
    braking_mps2: float = .5
    reaction_delay_s: float = .25
    stopping_reserve_m: float = .12
    depth_error_m: float = .08
    max_age_s: float = .5
    min_return_m: float = .30
    max_return_m: float = 12.
    max_assessment_m: float = 4.
    min_assessment_m: float = .30
    distance_step_m: float = .025
    border_px: int = 24
    required_fov_fraction: float = .60
    required_finite_fraction: float = .60
    required_full_valid_fraction: float = .30
    across_samples: int = 13
    height_samples: int = 9
    obstacle_samples: int = 3

    def __post_init__(self):
        if (self.radius_m < .294082845 or self.margin_m < .12 or self.half_height_m < .4 or
                not 0 < self.braking_mps2 <= .5 or self.reaction_delay_s < .25 or
                self.stopping_reserve_m < .12 or self.depth_error_m < .08 or
                not 0 < self.max_age_s <= .5 or not 0 < self.max_assessment_m <= 4. or
                self.border_px < 24 or self.required_fov_fraction < .60 or
                self.required_finite_fraction < .60 or self.required_full_valid_fraction < .30 or
                self.across_samples < 13 or self.height_samples < 9 or not 1 <= self.obstacle_samples <= 3):
            raise ValueError('camera, vehicle and stopping guards may not be weakened')
        if (not 0 < self.distance_step_m <= .05 or self.min_assessment_m <= 0 or
                self.max_assessment_m < self.min_assessment_m or
                any(f > 1 for f in (self.required_fov_fraction, self.required_finite_fraction,
                                   self.required_full_valid_fraction))):
            raise ValueError('invalid corridor assessment limits')


@dataclass(frozen=True)
class CorridorAssessment:
    allowed: bool
    reason: str
    plane_distance_m: float | None = None
    minimum_stopping_distance_m: float | None = None
    fov_fraction: float = 0.
    finite_fraction: float = 0.
    full_valid_fraction: float = 0.
    closest_obstacle_m: float | None = None
    samples: int = 0
    covered_samples: int = 0
    valid_samples: int = 0
    obstacle_hits: int = 0
    assessed_planes: int = 0
    route_certified: bool = False
    collision_end_world: tuple | None = None
    measured_collision_end_world: tuple | None = None
    collision_radius_m: float | None = None
    collision_half_height_m: float | None = None
    collision_returns_checked: int = 0
    collision_candidate_returns: int = 0
    collision_scan_complete: bool = True
    closest_obstacle_scope: str = 'minimum among checked intersecting returns; blocked scan may stop early'
    collision_source: str = 'full-frame calibrated returns against stopping swept vehicle volume'

    def status(self):
        return asdict(self)


def assess_motion_corridor(depth, K, R_world_cam, t_world_cam, position,
                           command_velocity, measured_velocity, capture_stamp,
                           receipt_stamp, now, *, pose_stamp=None,
                           route_certified=False, params=None):
    """Return a fresh-view verdict for the existing full 13x9 cross-section.

    R/t are capture-time calibrated optical camera pose; position and velocities
    are current world ENU. ``pose_stamp`` is the current fused-pose capture time.
    A valid previously checked route/swept volume is mandatory. The first plane
    with >=.60 FOV coverage is chosen at/after the true stopping distance, up to
    four metres. Invalid depth is never skipped by searching a different plane.
    The view plane tests visibility and valid support only. Collision returns
    are tested separately against the actual stopping swept vehicle cylinder,
    including horizontal radius+margin, vertical half-height and optical depth
    uncertainty. A farther view plane never extends the commanded travel.
    """
    p = params or CorridorParams()
    details = dict(route_certified=bool(route_certified))

    def result(allowed, reason, **updates):
        return CorridorAssessment(bool(allowed), reason, **dict(details, **updates))

    try:
        requested, measured = np.asarray(command_velocity, float), np.asarray(measured_velocity, float)
        pos = np.asarray(position, float)
    except (TypeError, ValueError):
        return result(False, 'invalid motion or fused pose')
    if (requested.ndim != 1 or len(requested) < 2 or measured.ndim != 1 or len(measured) < 2 or
            not np.isfinite(requested).all() or not np.isfinite(measured).all() or
            pos.shape != (3,) or not np.isfinite(pos).all()):
        return result(False, 'invalid motion or fused pose')
    speed = float(np.linalg.norm(requested[:2]))
    if speed <= 1e-6:
        return result(True, 'no horizontal motion requested')
    if not route_certified:
        return result(False, 'full near-body route/swept-volume certificate required')
    if not _calibration(K, R_world_cam, t_world_cam):
        return result(False, 'invalid calibrated camera geometry')
    try:
        d = np.asarray(depth, float)
        finite_stamps = all(math.isfinite(s) for s in (capture_stamp, receipt_stamp, now, pose_stamp))
    except (TypeError, ValueError):
        return result(False, 'missing capture/receipt or current fused-pose timestamps')
    if not finite_stamps or not 0 <= now-capture_stamp <= p.max_age_s or not 0 <= now-receipt_stamp <= p.max_age_s:
        return result(False, 'motion depth capture/receipt is stale')
    if not 0 <= now-pose_stamp <= p.max_age_s:
        return result(False, 'current fused pose is stale')
    if d.ndim != 2 or min(d.shape) <= p.border_px*2+2:
        return result(False, 'invalid depth image geometry')
    valid = np.isfinite(d) & (d >= p.min_return_m) & (d <= p.max_return_m)
    details['full_valid_fraction'] = float(valid.mean())
    if valid.mean() < p.required_full_valid_fraction:
        return result(False, 'motion depth has less than .30 full-frame valid coverage')

    # Use requested speed as well as measured speed; a just-started command must
    # not receive a shorter certificate merely because the vehicle is at rest.
    commanded_xyz = np.r_[requested[:2], requested[2] if len(requested) >= 3 else 0.]
    measured_xyz = np.r_[measured[:2], measured[2] if len(measured) >= 3 else 0.]
    commanded_speed, measured_speed = np.linalg.norm(commanded_xyz), np.linalg.norm(measured_xyz)
    stopping_speed = max(commanded_speed, measured_speed)
    stopping = (stopping_speed*p.reaction_delay_s + stopping_speed**2/(2*p.braking_mps2)
                + p.stopping_reserve_m)
    details['minimum_stopping_distance_m'] = stopping
    if stopping > p.max_assessment_m:
        return result(False, 'true stopping distance exceeds bounded sensor assessment range')
    direction = requested[:2]/speed
    side = np.array([-direction[1], direction[0]])
    across, heights = np.meshgrid(np.linspace(-p.radius_m-p.margin_m, p.radius_m+p.margin_m, p.across_samples),
                                  np.linspace(-p.half_height_m, p.half_height_m, p.height_samples))
    offsets = np.column_stack([across.ravel()[:, None]*side, heights.ravel()])
    details['samples'] = len(offsets)
    R, t = np.asarray(R_world_cam, float), np.asarray(t_world_cam, float)
    h, w = d.shape
    fx, fy, cx, cy = K
    start = max(stopping, p.min_assessment_m)
    distances = np.arange(start, p.max_assessment_m+1e-9, p.distance_step_m)
    if not len(distances) or distances[-1] < p.max_assessment_m-1e-9:
        distances = np.r_[distances, p.max_assessment_m]
    chosen = None
    best_fraction = 0.
    for count, at in enumerate(distances, 1):
        wp = pos + np.r_[direction*at, 0.] + offsets
        pc = (wp-t)@R
        den = np.maximum(pc[:, 2], 1e-6)
        u = np.rint(fx*pc[:, 0]/den+cx).astype(int)
        v = np.rint(fy*pc[:, 1]/den+cy).astype(int)
        inside = ((pc[:, 2] > .02) & (u >= p.border_px) & (u < w-p.border_px) &
                  (v >= p.border_px) & (v < h-p.border_px))
        ids = np.flatnonzero(inside)
        fraction = len(ids)/len(offsets)
        best_fraction = max(best_fraction, fraction)
        if len(ids) >= 12 and fraction >= p.required_fov_fraction:
            chosen = float(at), pc, u, v, ids
            details['assessed_planes'] = count
            break
    if chosen is None:
        return result(False, 'full motion cross-section remains outside calibrated camera FOV',
                      fov_fraction=best_fraction, assessed_planes=len(distances))
    at, pc, u, v, ids = chosen
    observed = d[v[ids], u[ids]]
    good = valid[v[ids], u[ids]]
    finite_fraction = float(good.mean())
    details.update(plane_distance_m=at, fov_fraction=len(ids)/len(offsets),
                   finite_fraction=finite_fraction, covered_samples=len(ids), valid_samples=int(good.sum()))

    # Full-frame endpoint checking retains close/thin obstacles that the sparse
    # visibility-plane sample could miss. Its horizon is the true stopping
    # distance, not the (possibly farther) plane needed for FOV coverage.
    # Include measured velocity's direction as well: a commanded turn does not
    # instantly cancel existing sideways or vertical momentum.
    commanded_end = pos + commanded_xyz/max(commanded_speed, 1e-9)*stopping
    measured_end = pos + measured_xyz/max(measured_speed, 1e-9)*stopping
    radius = p.radius_m+p.margin_m
    details.update(collision_end_world=tuple(commanded_end),
                   measured_collision_end_world=tuple(measured_end),
                   collision_radius_m=radius, collision_half_height_m=p.half_height_m)
    optical_axis = R[:, 2]
    furthest_z = max(float((point-t)@optical_axis) for point in (pos, commanded_end, measured_end))
    furthest_z += radius*np.linalg.norm(optical_axis[:2])+p.half_height_m*abs(optical_axis[2])+p.depth_error_m
    # This exact optical-z bound only removes returns too far away to intersect
    # either swept cylinder even with depth error. No stride or downsampling.
    collision_returns = np.isfinite(d) & (d > 0.) & (d <= p.max_return_m) & (d <= furthest_z)
    border = p.border_px
    collision_returns[:border] = collision_returns[-border:] = False
    collision_returns[:, :border] = collision_returns[:, -border:] = False
    rows, columns = np.nonzero(collision_returns)
    details['collision_candidate_returns'] = len(rows)
    details['collision_returns_checked'] = 0
    hit_count = 0
    closest = None
    # Chunking retains EVERY pixel for an allowed decision. A blocked decision
    # can finish as soon as the existing same-frame hit threshold is met, so a
    # close wall does not stall the control callback projecting half a million
    # redundant hit pixels. This is early rejection, never a stride or thinning.
    for first in range(0, len(rows), 8192):
        rr, cc = rows[first:first+8192], columns[first:first+8192]
        ux, uy = (cc-cx)/fx, (rr-cy)/fy
        rays = np.column_stack([ux*R[j,0]+uy*R[j,1]+R[j,2] for j in range(3)])
        endpoints = rays*d[rr,cc,None]+t
        details['collision_returns_checked'] += len(rr)
        # A bounded optical-z error propagates along the complete calibrated
        # ray. Expand XY and Z by that uncertainty segment's worst components.
        xy_error = p.depth_error_m*np.linalg.norm(rays[:,:2],axis=1)
        z_error = p.depth_error_m*np.abs(rays[:,2])
        hits = np.zeros(len(rr),bool)
        for end in (commanded_end,measured_end):
            delta = end[:2]-pos[:2]
            length2 = float(delta@delta)
            offset = endpoints[:,:2]-pos[:2]
            alpha = np.clip((offset[:,0]*delta[0]+offset[:,1]*delta[1])/max(length2,1e-12),0.,1.)
            closest_xy = pos[:2]+alpha[:,None]*delta
            within_xy = np.linalg.norm(endpoints[:,:2]-closest_xy,axis=1) <= radius+xy_error
            # Conservative vertical bound on the whole swept cylinder.
            within_z = ((endpoints[:,2] >= min(pos[2],end[2])-p.half_height_m-z_error) &
                        (endpoints[:,2] <= max(pos[2],end[2])+p.half_height_m+z_error))
            hits |= within_xy & within_z
        hit_count += int(hits.sum())
        if hits.any():
            hit_delta = endpoints[hits,:2]-pos[:2]
            minimum = float(np.min(hit_delta[:,0]*direction[0]+hit_delta[:,1]*direction[1]))
            closest = minimum if closest is None else min(closest,minimum)
        if hit_count >= p.obstacle_samples:
            details['collision_scan_complete'] = first+len(rr) >= len(rows)
            break
    details.update(obstacle_hits=hit_count, closest_obstacle_m=closest)
    if hit_count >= p.obstacle_samples:
        return result(False, 'new obstacle intersects actual braking swept vehicle volume')
    if finite_fraction < p.required_finite_fraction:
        return result(False, 'motion corridor has less than .60 finite valid depth coverage')
    return result(True, 'fresh calibrated adaptive corridor view; near-body route separately certified')
