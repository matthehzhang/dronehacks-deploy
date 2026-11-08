"""Pilot launch assertions stay explicit, bounded, fixed and revocable."""
from dataclasses import replace

import numpy as np
import pytest

from kestrel_nav.searchnav.launch_volume import LaunchVolume, LaunchVolumeParams
from kestrel_nav.searchnav.grid import DepthGrid, GridParams, UNKNOWN, FREE, OCCUPIED
from kestrel_nav.searchnav.mission import SearchMission, Observation
from kestrel_nav.searchnav.clearance import DepthClearanceHistory

BOUNDS = (-1., -1., -.2, 1., 1., 1.8)
FLAGS = dict(pilot_verified=True, saw_disarmed=True, allow_takeoff=True, pilot_started=True)


def granted(bounds=BOUNDS, origin=(2., -1., .3), duration=30.):
    volume = LaunchVolume(LaunchVolumeParams(True, bounds, duration))
    assert volume.grant(origin, 10., **FLAGS)
    return volume


def test_default_is_disabled_and_grant_needs_every_explicit_lifecycle_flag():
    volume = LaunchVolume()
    assert not volume.grant((0.,0.,0.), 1., **FLAGS)
    assert not volume.status(1.)['active']
    for flag in FLAGS:
        volume = LaunchVolume(LaunchVolumeParams(True, BOUNDS))
        args = dict(FLAGS, **{flag:False})
        assert not volume.grant((0.,0.,0.), 1., **args)
        assert not volume.status(1.)['active']


def test_world_bounds_fixed_from_one_origin_and_only_whole_voxels_fit():
    volume = granted()
    expected = (1., -2., .1, 3., 0., 2.1)
    assert np.allclose(volume.status(10.)['world_bounds'], expected)
    points = np.array([[2.,-1.,1.], [1.01,-1.,1.], [1.1,-1.,1.], [2.,-1.,2.05]])
    assert volume.unknown_mask(points, 10.1, (.05,.05,.1)).tolist() == [True,False,True,False]
    volume.unknown_mask(np.array([[100.,100.,100.]]), 11., .01)
    assert np.allclose(volume.status(11.)['world_bounds'], expected)
    assert volume.status(11.)['provenance_id'].startswith('pilot-launch-')


def test_narrow_volume_does_not_certify_full_vehicle_even_when_center_is_inside():
    volume = granted((-.2,-.2,-.2,.2,.2,.2), origin=(0.,0.,0.))
    assert volume.unknown_mask([[0.,0.,0.]], 10.1, (0.,0.,0.)).all()
    assert not volume.unknown_mask([[0.,0.,0.]], 10.1, (.414082845,.414082845,.52)).any()


def test_expiry_revocation_clock_reverse_and_frame_reset_require_fresh_explicit_object():
    volume = granted(duration=2.)
    assert volume.status(11.99)['active']
    revision = volume.revision
    assert not volume.status(12.)['active'] and volume.revision > revision
    assert not volume.grant((0.,0.,0.), 13., **FLAGS)
    for cause in ('pilot authority lost', 'EKF reset', 'pilot hold requested'):
        volume = granted()
        volume.revoke(cause)
        assert not volume.unknown_mask([[2.,-1.,1.]], 11., 0.).any()
        assert cause in volume.status(11.)['revoked_reason']
    volume = granted()
    volume.reset()
    assert not volume.status(11.)['active']
    assert not volume.grant((0.,0.,0.), 12., **FLAGS)
    volume = granted()
    assert not volume.status(9.)['active']
    assert 'clock' in volume.status(11.)['revoked_reason']


@pytest.mark.parametrize('bounds', ((-2,-1,-.2,2,1,1.8), (-1,-1,-1,1,1,3),
                                    (0,0,0,0,1,1), (1,-1,-1,2,1,1),
                                    (-1,-1,-1,1,float('nan'),1)))
def test_invalid_or_remote_or_over_three_metre_total_span_bounds_rejected(bounds):
    with pytest.raises(ValueError):
        LaunchVolumeParams(True, bounds)


def test_finite_clock_origin_and_duration_required_and_failed_attempt_cannot_auto_regrant():
    with pytest.raises(ValueError):
        LaunchVolumeParams(True, BOUNDS, 61.)
    for origin, now in (((0.,0.,float('nan')),1.), ((0.,0.,0.),0.)):
        volume = LaunchVolume(LaunchVolumeParams(True, BOUNDS))
        assert not volume.grant(origin, now, **FLAGS)
        assert not volume.grant((0.,0.,0.), 2., **FLAGS)
    volume = granted()
    assert not volume.status(float('nan'))['active']
    volume = LaunchVolume(LaunchVolumeParams(True, BOUNDS))
    volume.status(5.)
    assert not volume.grant((0.,0.,0.), 4., **FLAGS)


def test_mask_rejects_invalid_reserves_and_unknown_only_semantics_are_explicit():
    volume = granted()
    for bad in ((-.1,.1,.1), (float('inf'),0.,0.), (.1,.1)):
        with pytest.raises(ValueError):
            volume.unknown_mask([[2.,-1.,1.]], 10.1, bad)
    assert not volume.unknown_mask([[float('nan'),-1.,1.]], 10.2).any()
    assert 'UNKNOWN' in volume.status(10.3)['application']


def launch_grid(duration=30.):
    volume = granted((-1.5,-1.5,-1.5,1.5,1.5,1.5), origin=(0.,0.,1.), duration=duration)
    grid = DepthGrid(GridParams(extent_m=6.))
    grid.reset((0.,0.),1.)
    grid.set_launch_volume(volume,10.)
    return grid, volume


def test_planning_uses_separate_whole_volume_assertion_without_changing_raw_depth():
    grid,volume = launch_grid()
    assert np.all(grid.states == UNKNOWN) and np.all(grid.collapsed() == UNKNOWN)
    assert grid.planning_collapsed()[grid.cell((0.,0.))] == FREE
    assert grid.segment_safe((0.,0.),(.6,0.))
    assert not grid.segment_safe((0.,0.),(1.3,0.))  # inflated footprint leaves fixed box
    assert grid.summary((0.,0.))['pilot_volume_only_cells'] > 0
    assert not np.any(grid.collapsed() != UNKNOWN)


def test_observed_hit_revokes_assertion_instead_of_ever_overwriting_hit():
    grid,volume = launch_grid()
    occupied = np.zeros_like(grid.states,dtype=bool)
    ij = grid.cell((.5,0.))
    occupied[:,ij[0],ij[1]] = True
    revision = grid.revision
    grid.apply_observation(np.zeros_like(occupied),occupied,10.1)
    assert grid.revision > revision and not volume.status(10.1)['active']
    assert grid.collapsed()[ij] == OCCUPIED and grid.planning_collapsed()[ij] == OCCUPIED
    assert not grid.navigation()[0].any()


def test_existing_obstacle_also_rejects_new_pilot_assertion():
    grid = DepthGrid(GridParams(extent_m=6.))
    grid.reset((0.,0.),1.)
    occupied = np.zeros_like(grid.states,dtype=bool)
    occupied[:,grid.cell((.5,0.))[0],grid.cell((.5,0.))[1]] = True
    grid.apply_observation(np.zeros_like(occupied),occupied,9.)
    volume = granted((-1.5,-1.5,-1.5,1.5,1.5,1.5),origin=(0.,0.,1.))
    grid.set_launch_volume(volume,10.)
    assert not volume.status(10.)['active'] and not grid.navigation()[0].any()
    assert grid.collapsed()[grid.cell((.5,0.))] == OCCUPIED


def test_expiry_invalidates_active_route_and_stops_before_replanning():
    grid,volume = launch_grid(duration=1.)
    mission = SearchMission(grid)
    mission.started,mission.phase,mission.phase_t = 10.,'move',10.
    mission.hold_xy,mission.hold_z = np.array((0.,0.)),1.
    mission.route = [np.array((0.,0.)),np.array((.6,0.))]
    mission.route_goal,mission.goal_kind = np.array((.6,0.)),'frontier'
    mission.route_index,mission.route_revision = 1,grid.revision
    revision = grid.revision
    result = mission.tick(Observation(now=11.,position=(0.,0.,1.),yaw=0.,ready=True,
        armed=True,offboard=True,landed=False))
    assert not volume.status(11.)['active'] and grid.revision > revision
    assert not mission.route and mission.counts['invalidations'] == 1
    assert result.velocity[:2] == (0.,0.)


def test_height_rebuild_cannot_move_fixed_launch_bounds_or_fill_outside_height():
    grid,volume = launch_grid()
    bounds = volume.world_bounds
    grid.reset((.2,.2),1.4)
    grid.refresh_launch_volume(10.1)
    assert grid.segment_safe((0.,0.),(.5,0.))
    grid.reset((.2,.2),2.3)  # full slab is no longer inside the assertion
    grid.refresh_launch_volume(10.2)
    assert not grid.navigation()[0].any()
    assert volume.world_bounds == bounds and np.all(grid.states == UNKNOWN)


def test_disabled_volume_never_changes_planner_and_authority_loss_withdraws_support():
    grid,volume = launch_grid()
    mission = SearchMission(grid)
    mission.started,mission.phase,mission.hold_z = 10.,'move',1.
    result = mission.tick(Observation(now=10.1,position=(0.,0.,1.),yaw=0.,ready=True,
        armed=True,offboard=False,landed=False))
    assert result.phase == 'yield' and not volume.status(10.1)['active']
    assert not grid.navigation()[0].any()
    grid.set_launch_volume(LaunchVolume(),10.2)
    assert not grid.navigation()[0].any() and np.all(grid.collapsed() == UNKNOWN)


def blind_history(duration=30.):
    history = DepthClearanceHistory()
    volume = granted((-1.5,-1.5,-1.5,1.5,1.5,1.5),origin=(0.,0.,1.),duration=duration)
    history.launch_volume = volume
    optical = np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])
    for stamp in (10.,10.1,10.2):
        assert history.integrate(np.full((120,160),6.),(100.,100.,79.5,59.5),
                                 optical,(0.,0.,1.),stamp,stamp+.01)
    return history,volume,optical


def certify_launch(history,now=10.3):
    return history.certify((0.,0.,1.),(.08,0.,1.),.294082845+.12,.4+.12,now)


def test_3d_assertion_supports_blind_unknown_but_still_requires_fresh_depth_and_whole_bounds():
    history,volume,optical = blind_history()
    assert certify_launch(history).safe
    assert history.status()['pilot_verified_samples'] > 0
    assert any(record[0] == -1 for record in history.cache.values())  # never caches assumed FREE
    assert not history.certify((-1.2,0.,1.),(-1.28,0.,1.),.414082845,.52,10.3).safe
    assert not certify_launch(history,11.).safe  # assertion cannot replace fresh sensor


def test_cached_3d_support_expires_and_new_depth_hit_revokes_entire_assertion():
    history,volume,optical = blind_history(duration=.4)
    assert certify_launch(history).safe
    assert history.integrate(np.full((120,160),6.),(100.,100.,79.5,59.5),
                             optical,(0.,0.,1.),10.4,10.41)
    assert not certify_launch(history,10.42).safe and not volume.status(10.42)['active']
    history,volume,optical = blind_history()
    assert certify_launch(history).safe
    assert history.integrate(np.full((120,160),.4),(100.,100.,79.5,59.5),
                             optical,(0.,0.,1.),10.35,10.36)
    assert not certify_launch(history,10.37).safe
    assert not volume.status(10.37)['active']
