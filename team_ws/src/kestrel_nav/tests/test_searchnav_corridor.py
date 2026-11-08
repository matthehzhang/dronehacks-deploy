"""A longer view plane fixes camera coverage without inventing near-body space."""
from dataclasses import replace
import math

import numpy as np
import pytest

from kestrel_nav.searchnav.corridor import assess_motion_corridor, CorridorParams


R = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
K = (320./math.tan(math.radians(40.)), 200./math.tan(math.radians(25.)), 319.5, 199.5)
POSITION = np.array([0., 0., 1.])
CAMERA = POSITION+(.116, 0., .029)


def assess(depth=6., direction=(.4, 0., 0., 0.), **kwargs):
    d = np.full((400, 640), depth, float) if np.isscalar(depth) else depth
    values = dict(pose_stamp=1., route_certified=True)
    values.update(kwargs)
    return assess_motion_corridor(d, K, R, CAMERA, POSITION, direction, (0., 0., 0.),
                                  1., 1.01, 1.1, **values)


def test_default_calibrated_mount_needs_farther_plane_and_passes_open_room():
    result = assess()
    assert result.allowed, result.status()
    assert .60 < result.plane_distance_m < 1.2
    assert result.plane_distance_m >= result.minimum_stopping_distance_m
    assert result.fov_fraction >= .60 and result.finite_fraction == 1.
    assert result.samples == 13*9 and result.closest_obstacle_m is None


def test_near_wall_before_adaptive_plane_still_blocks_and_reports_true_distance():
    result = assess(depth=.45-.116)
    assert not result.allowed and 'obstacle' in result.reason
    assert result.plane_distance_m > .60
    assert result.obstacle_hits >= 3
    assert result.closest_obstacle_m == pytest.approx(.45)


def test_wall_before_far_view_plane_but_beyond_true_braking_vehicle_volume_passes():
    # At this low speed the body-center braking path is ~.123m. Even allowing
    # full .414m radius+margin and .08m optical depth error, a wall at .67m is
    # outside it. The camera still needs a farther ~.7m plane for FOV coverage.
    result = assess(depth=.67-.116, direction=(.01,0.,0.,0.))
    assert result.allowed, result.status()
    assert result.minimum_stopping_distance_m < .13
    assert .67 < result.plane_distance_m
    assert result.obstacle_hits == 0
    assert result.collision_radius_m == pytest.approx(.294082845+.12)


def image_with_point(world, size=3):
    image = np.full((400,640),6.)
    pc = (np.asarray(world)-CAMERA)@R
    u = int(round(K[0]*pc[0]/pc[2]+K[2]))
    v = int(round(K[1]*pc[1]/pc[2]+K[3]))
    half = size//2
    assert 24+half < u < 640-24-half and 24+half < v < 400-24-half
    image[v-half:v+half+1,u-half:u+half+1] = pc[2]
    return image


def test_small_return_between_sparse_view_rays_blocks_immediately_inside_actual_tube():
    d = image_with_point((.55,0.,1.))
    result = assess(d,direction=(.01,0.,0.,0.))
    assert not result.allowed and 'actual braking' in result.reason
    assert result.obstacle_hits >= 3
    assert result.closest_obstacle_m == pytest.approx(.55)
    # A near return below the depth sensor's qualified free-return range also
    # cannot be silently dropped and called free.
    near = image_with_point((.36,0.,1.))
    assert not assess(near,direction=(.01,0.,0.,0.)).allowed


def test_visible_return_laterally_outside_stopping_capsule_does_not_block():
    # Same early forward distance as the blocking point above, but the whole
    # return patch is outside the radius+margin+propagated depth uncertainty.
    d = image_with_point((.55,.31,1.))
    result = assess(d,direction=(.01,0.,0.,0.))
    assert result.allowed, result.status()
    assert result.collision_returns_checked >= 9 and result.obstacle_hits == 0


def test_every_candidate_pixel_is_checked_before_allowing_and_late_thin_hit_blocks():
    d = np.full((400,640),6.)
    # More than a processing chunk of near returns, all safely beside the tube.
    d[24:376,24:52] = .499
    safe = assess(d,direction=(.01,0.,0.,0.))
    assert safe.allowed and safe.collision_candidate_returns > 8192
    assert safe.collision_scan_complete
    assert safe.collision_returns_checked == safe.collision_candidate_returns
    # Put a nine-pixel real obstacle after the first processing chunk. No
    # stride, first-chunk shortcut or sparse plane sampling may miss it.
    patch = image_with_point((.55,0.,.87))
    d[patch < 6.] = patch[patch < 6.]
    blocked = assess(d,direction=(.01,0.,0.,0.))
    assert not blocked.allowed and blocked.obstacle_hits >= 3
    assert blocked.collision_returns_checked > 8192


def test_depth_uncertainty_reserve_is_included_outside_nominal_vehicle_tube():
    result = assess(depth=.60-.116,direction=(.01,0.,0.,0.))
    nominal_front = result.minimum_stopping_distance_m + result.collision_radius_m
    assert .60 > nominal_front  # beyond the nominal cylinder but inside its error reserve.
    assert not result.allowed and result.obstacle_hits >= 3


def test_high_measured_speed_catches_wall_beyond_low_command_braking_tube():
    d = np.full((400,640),.9-.116)
    slow = assess(d,direction=(.01,0.,0.,0.))
    assert slow.allowed, slow.status()
    result = assess_motion_corridor(d,K,R,CAMERA,POSITION,(.01,0.,0.,0.),(.8,0.,0.),
                                   1.,1.01,1.1,pose_stamp=1.,route_certified=True)
    assert not result.allowed and result.obstacle_hits >= 3
    assert result.minimum_stopping_distance_m == pytest.approx(.8*.25+.8**2/(2*.5)+.12)


def test_existing_lateral_momentum_is_included_in_current_collision_union():
    d = image_with_point((.55,.30,1.))
    result = assess_motion_corridor(d,K,R,CAMERA,POSITION,(.01,0.,0.,0.),(0.,.8,0.),
                                   1.,1.01,1.1,pose_stamp=1.,route_certified=True)
    assert not result.allowed and result.obstacle_hits >= 3
    assert result.measured_collision_end_world[1] > .9


@pytest.mark.parametrize('direction', ((0., .4, 0., 0.), (-.4, 0., 0., 0.)))
def test_sideways_and_backward_motion_cannot_find_front_camera_view(direction):
    result = assess(direction=direction)
    assert not result.allowed and 'FOV' in result.reason
    assert result.plane_distance_m is None


def test_sparse_or_invalid_depth_remains_blocked_without_trying_different_plane():
    sparse = np.full((400, 640), np.nan)
    sparse[:100] = sparse[-100:] = 6.
    result = assess(sparse)
    assert result.full_valid_fraction == .5 and result.finite_fraction < .60
    assert not result.allowed and '.60' in result.reason
    sparse[:] = np.nan
    assert not assess(sparse).allowed
    assert '.30' in assess(sparse).reason


def test_no_route_certificate_or_stale_pose_depth_is_a_separate_refusal():
    assert not assess(route_certified=False).allowed
    assert 'near-body' in assess(route_certified=False).reason
    assert not assess(pose_stamp=.1).allowed
    assert not assess(pose_stamp=None).allowed
    d = np.full((400, 640), 6.)
    stale = assess_motion_corridor(d, K, R, CAMERA, POSITION, (.4,0), (0,0), .1, .2, 1.,
                                   pose_stamp=1., route_certified=True)
    assert not stale.allowed and 'stale' in stale.reason


def test_oversized_height_and_stopping_distance_beyond_sensor_range_stay_blocked():
    large = assess(params=CorridorParams(half_height_m=5.))
    assert not large.allowed and 'FOV' in large.reason
    fast = assess(direction=(3.,0.,0.,0.))
    assert not fast.allowed and fast.minimum_stopping_distance_m > 4.
    with pytest.raises(ValueError):
        CorridorParams(half_height_m=.3)
    with pytest.raises(ValueError):
        CorridorParams(required_fov_fraction=.5)


def test_measured_speed_controls_true_braking_distance_even_when_command_slows():
    result = assess_motion_corridor(np.full((400,640),6.), K, R, CAMERA, POSITION,
                                    (.1,0.,0.,0.), (1.,0.,0.), 1.,1.01,1.1,
                                    pose_stamp=1.,route_certified=True)
    assert result.allowed
    assert result.minimum_stopping_distance_m == pytest.approx(1.*.25+1.**2/(2*.5)+.12)
    assert result.plane_distance_m >= result.minimum_stopping_distance_m


def test_assessment_rotates_with_actual_calibrated_camera_and_world_pose():
    # World yaw is arbitrary. Neither FOV nor the body envelope changes.
    angle = .7
    c, s = math.cos(angle), math.sin(angle)
    turn = np.array([[c,-s,0.], [s,c,0.], [0.,0.,1.]])
    result = assess_motion_corridor(np.full((400,640),6.), K, turn@R,
                                    turn@CAMERA, turn@POSITION, (.4*c,.4*s,0.,0.), (0.,0.,0.),
                                    1.,1.01,1.1,pose_stamp=1.,route_certified=True)
    assert result.allowed
    assert result.plane_distance_m == pytest.approx(assess().plane_distance_m)


def test_no_horizontal_command_needs_no_fabricated_view_certificate():
    assert assess(direction=(0.,0.,0.,.2), route_certified=False).allowed
