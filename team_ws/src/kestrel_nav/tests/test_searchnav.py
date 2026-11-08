"""Behavioral regressions for map-free search, including changed-obstacle replans."""
import builtins
from dataclasses import replace
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from kestrel_nav.searchnav.grid import DepthGrid, GridParams, FREE, OCCUPIED, UNKNOWN
from kestrel_nav.searchnav.mission import SearchMission, MissionParams, Observation
from kestrel_nav.searchnav.target import (Candidate, DetectorParams, MonoFloorDetector,
                                           TargetTracker, floor_intersections, validate_depth)
from kestrel_nav.searchnav.floor import fit_floor, FloorTracker
from kestrel_nav.searchnav.inputs import DepthWatchdog, Px4ClockAnchor


def grid_free(extent=6.):
    g = DepthGrid(GridParams(extent_m=extent))
    g.reset((0., 0.), 1.)
    g.apply_observation(np.ones(g.states.shape, bool), np.zeros(g.states.shape, bool), 1.)
    return g


def mark(g, xy, value, *, stamp=2.):
    ij = g.cell(xy)
    mask = np.zeros(g.states.shape, bool)
    mask[:, ij[0], ij[1]] = True
    if value == OCCUPIED:
        g.apply_observation(np.zeros_like(mask), mask, stamp)
    elif value == UNKNOWN:
        g.states[mask] = UNKNOWN
        g._changed(mask.any(axis=0))
    return mask


def active_mission(g):
    m = SearchMission(g)
    m.started, m.phase, m.phase_t = 1., 'move', 1.
    m.hold_xy, m.hold_z = np.array([-1.5, 0.]), 1.
    m.route = [np.array([-1.5, 0.]), np.array([1.5, 0.])]
    m.route_index, m.route_revision = 1, g.revision
    m.route_goal, m.goal_kind = np.array([1.5, 0.]), 'target'
    return m


def obs(t=2., **kw):
    values = dict(now=t, position=(-1.5, 0., 1.), yaw=0., ready=True, offboard=True,
                  armed=True, landed=False, target_xy=(1.5, 0.), target_stamp=t)
    values.update(kw)
    return Observation(**values)


def test_starts_unknown_and_preserves_full_vehicle_envelope():
    g = DepthGrid()
    assert np.all(g.collapsed() == UNKNOWN)
    assert not g.navigation()[0].any()
    assert g.p.radius_m == .294082845 and g.p.margin_m == .12
    assert g.inflation_m * 2 < .9144  # .05m raster permits a surveyed 3ft gap.
    with pytest.raises(ValueError):
        GridParams(radius_m=.2)


def test_unknown_is_never_a_shortcut_even_after_smoothing():
    g = grid_free()
    for y in np.arange(-.8, .81, .05):
        mark(g, (0., y), UNKNOWN)
    path = g.astar((-1.5, 0.), (1.5, 0.))
    assert path and len(path) > 2
    assert not g.segment_safe((-1.5, 0.), (1.5, 0.))
    assert g.route_safe(path)
    assert max(abs(p[1]) for p in path) > 1.1


def test_inflation_rejects_gap_narrower_than_vehicle_and_margin():
    g = grid_free()
    for y in (-.35, .35):
        for x in np.arange(-2., 2., .05):
            mark(g, (x, y), OCCUPIED)
    assert not g.segment_safe((-1., 0.), (1., 0.))


def test_one_free_layer_cannot_clear_unseen_vehicle_height():
    g = DepthGrid(GridParams(extent_m=4.))
    g.reset((0., 0.), 1.)
    free = np.zeros(g.states.shape, bool)
    free[len(g.offsets)//2] = True
    g.apply_observation(free, np.zeros_like(free), 1.)
    assert np.all(g.collapsed() == UNKNOWN)
    assert not g.navigation()[0].any()


def test_expiration_blocks_old_free_space_and_increments_revision():
    g = grid_free()
    before = g.revision
    g.expire(22.)
    assert g.revision > before
    assert not g.navigation()[0].any()


def test_equal_count_obstacle_motion_invalidates_cached_route_immediately():
    g = grid_free()
    old = mark(g, (0., 2.), OCCUPIED, stamp=1.5)
    m = active_mission(g)
    old_count = np.count_nonzero(g.states == OCCUPIED)
    # Prepare two real free observations; old occupied cell remains blocked.
    for t in (1.8, 1.9):
        g.apply_observation(old, np.zeros_like(old), t)
    new = np.zeros(g.states.shape, bool)
    ij = g.cell((0., 0.))
    new[:, ij[0], ij[1]] = True
    before = g.revision
    g.apply_observation(old, new, 2.)
    assert np.count_nonzero(g.states == OCCUPIED) == old_count
    assert g.revision > before
    command = m.tick(obs())
    assert m.route == []
    assert m.phase == 'plan'
    assert m.counts['invalidations'] == 1
    assert command.velocity[:2] == (0., 0.)
    # Next settled tick recomputes a different observed-safe path around it.
    m.tick(obs(2.1))
    assert m.route and g.route_safe(m.route)
    assert len(m.route) > 2
    assert not any(g.segment_safe((-1.5, 0.), (1.5, 0.)) for _ in range(1))


def test_new_depth_wall_flows_into_grid_revision_and_stops_existing_follower():
    g = grid_free()
    m = active_mission(g)
    before = g.revision
    R, _ = optical_pose()
    camera = np.array([-1.5, 0., 1.])
    changed = g.integrate(np.full((120, 160), 1.5), (100., 100., 79.5, 59.5), R, camera, 2.)
    assert changed > 0 and g.revision > before
    assert g.collapsed()[g.cell((0., 0.))] == OCCUPIED
    result = m.tick(obs())
    assert not m.route and result.phase == 'plan'
    assert result.velocity[:2] == (0., 0.)


def test_fast_obstacle_hits_update_revision_without_waiting_for_free_projection():
    g=grid_free()
    m=active_mission(g)
    R,_=optical_pose()
    before=g.revision
    g.integrate_occupied(np.full((120,160),1.5),(100.,100.,79.5,59.5),R,(-1.5,0.,1.),2.)
    assert g.revision>before and g.collapsed()[g.cell((0.,0.))]==OCCUPIED
    assert m.tick(obs()).phase=='plan' and not m.route


def test_fast_hit_only_updates_preserve_other_voxels_free_clearing_hysteresis():
    g=grid_free()
    old=mark(g,(-1.,2.),OCCUPIED,stamp=1.5)
    R,t=optical_pose()
    for i in range(3):
        g.apply_observation(old,np.zeros_like(old),2.+i*.3)
        g.integrate_occupied(np.full((120,160),2.),(100.,100.,79.5,59.5),R,t,2.1+i*.3)
    assert np.all(g.states[old]==FREE)


def test_failed_replan_clears_old_route_and_rotates_instead_of_replaying_it():
    g = grid_free()
    m = active_mission(g)
    # A complete new barrier partitions the already observed map.
    for y in np.arange(-3., 3., .05):
        mark(g, (0., y), OCCUPIED)
    m.tick(obs())
    assert not m.route
    m.tick(obs(2.1))
    assert not m.route
    assert m.phase == 'scan'
    turn = m.tick(obs(2.2))
    assert turn.velocity[:2] == (0., 0.)
    assert turn.velocity[3] > 0.


def test_sensor_loss_authority_loss_and_clock_reset_yield_without_commands():
    for kwargs in (dict(ready=False), dict(offboard=False), dict(armed=False),
                   dict(fault='depth stale')):
        m = active_mission(grid_free())
        result = m.tick(obs(**kwargs))
        assert result.phase == 'yield' and result.velocity is None and not m.route
        assert m.tick(obs(3.)).velocity is None  # latched until ground restart.
    m = active_mission(grid_free())
    m.last_t = 5.
    assert m.tick(obs(2.)).phase == 'yield'


def test_empty_map_scan_is_bounded_and_does_not_seed_a_free_bubble():
    g = DepthGrid(GridParams(extent_m=4.))
    g.reset((0., 0.), 1.)
    m = SearchMission(g, MissionParams(scan_timeout_s=1., max_blocked_scans=2))
    m.started, m.hold_z, m.hold_xy = 0., 1., np.array([0., 0.])
    m.begin_scan(obs(0., position=(0., 0., 1.)), 'test')
    for t in (1.1, 1.2, 2.3, 2.4, 3.5, 3.6):
        command = m.tick(obs(t, position=(0., 0., 1.), target_xy=None))
        assert command.velocity[:2] == (0., 0.)
    assert m.phase == 'exhausted' and not m.route
    assert np.all(g.states == UNKNOWN)


def test_launch_is_relative_to_current_local_pose_and_never_has_course_waypoints():
    m = SearchMission()
    position = (20., -10., 3.)
    ground = Observation(1., position, .4, ready=True, landed=True)
    assert m.tick(ground).phase == 'arm'
    launched = m.tick(replace(ground, now=1.1, offboard=True, armed=True))
    assert launched.phase == 'takeoff'
    assert launched.velocity[:2] == (0., 0.)
    assert launched.velocity[2] > 0.
    assert np.allclose(m.hold_xy, position[:2])
    assert m.hold_z == pytest.approx(3.8)
    assert np.all(m.grid.states == UNKNOWN) and not m.route
    assert np.allclose(m.grid.origin + m.grid.n * m.grid.p.resolution_m / 2, position[:2])


def test_mission_time_bound_holds_without_landing_or_repeating_route():
    m = active_mission(grid_free())
    result = m.tick(obs(183.))
    assert result.phase == 'exhausted' and not m.route
    assert result.velocity == (0., 0., 0., 0.)


def test_scan_holds_xyz_while_measured_yaw_accumulates():
    g = grid_free()
    m = SearchMission(g)
    m.started, m.hold_z = 0., 1.
    m.begin_scan(obs(0.), 'test')
    for i in range(1, 31):
        command = m.tick(obs(i*.1, yaw=((i*.22+math.pi) % (2*math.pi)-math.pi), target_xy=None))
        assert command.velocity[:3] == (0., 0., 0.)
        if m.phase == 'plan':
            break
    assert m.phase == 'plan'


def test_stationary_yaw_scan_does_not_issue_uncertified_lateral_hold_corrections():
    m = active_mission(grid_free())
    m.begin_scan(obs(), 'stationary')
    # Even an observed-free hold segment is not an active route to fly during
    # the yaw scan. PX4 receives zero XY velocity; the drift bound remains.
    command=m._hold(obs(2.1,position=(-1.45,.03,1.)),yaw_rate=.2)
    assert command.velocity == (0.,0.,0.,.2)


def test_takeoff_stays_vertical_under_ground_pose_jitter_and_requires_airborne_height():
    mission=SearchMission()
    start=Observation(1.,(0.,0.,0.),0.,ready=True,armed=True,offboard=True,landed=True)
    assert mission.tick(start).phase == 'takeoff'
    for i in range(1,10):
        command=mission.tick(replace(start,now=1.+i*.05,position=(.01,-.015,-.005)))
        assert command.phase == 'takeoff' and command.velocity[:2] == (0.,0.)
        assert command.velocity[2]>0 and mission.counts['scans']==0
    at_height=replace(start,now=2.,position=(.01,-.015,.8))
    assert mission.tick(at_height).phase == 'takeoff'  # landed cannot authorize scanning
    assert mission.tick(replace(at_height,now=2.1,landed=False)).phase == 'scan'


def test_refused_frontier_view_gets_cooldown_and_cannot_restart_planner_every_tick():
    m = active_mission(grid_free())
    m.goal_kind = 'frontier'
    goal = np.array(m.route_goal)
    m.scan_plan_t = 2.
    m.reject_motion_view(obs(2.1),'camera refuses braking corridor')
    assert not m.route and m.retry_goal is None
    assert any(np.allclose(f['xy'],goal) and f['until'] > 2.1 for f in m.failed_frontiers)
    assert m.scan_plan_t == 2. and m.phase == 'scan'
    before=m.counts['scan_plan_checks']
    command=m.tick(obs(2.2,target_xy=None))
    assert m.counts['scan_plan_checks'] == before
    assert command.velocity[:2] == (0.,0.)


def test_frontier_chooser_uses_reachable_side_of_maze_and_avoids_visited_views():
    g = grid_free(8.)
    # Unknown room to north; wall separates east room from launch side.
    g.states[:, g.n//2 + 25:, :] = UNKNOWN
    for y in np.arange(-4., 4., .05):
        mark(g, (.5, y), OCCUPIED)
    g._changed(np.ones((g.n, g.n), bool))
    choices = g.frontiers((-1.5, 0.))
    assert choices and all(point[0] < .1 for _, point in choices)
    assert all(g.astar((-1.5, 0.), point) is not None for _, point in choices)
    visited = [point for _, point in choices]
    next_choices = g.frontiers((-1.5, 0.), visited)
    assert all(all(np.linalg.norm(point-v) >= .45 for v in visited) for _, point in next_choices)


def optical_pose(yaw=0., z=1.):
    optical_in_flu = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) @ optical_in_flu
    return R, np.array([0., 0., z])


def test_invalid_depth_never_clears_and_real_wall_endpoints_block():
    g = DepthGrid(GridParams(extent_m=6., border_px=4))
    g.reset((0., 0.), 1.)
    R, t = optical_pose()
    K = (100., 100., 79.5, 59.5)
    g.integrate(np.full((120, 160), np.nan), K, R, t, 1.)
    assert np.all(g.states == UNKNOWN)
    g.integrate(np.full((120, 160), 2.), K, R, t, 2.)
    assert (g.collapsed() == OCCUPIED).any()
    assert g.collapsed()[g.cell((2., 0.))] == OCCUPIED
    assert g.collapsed()[g.cell((1., 0.))] == FREE
    assert g.collapsed()[g.cell((2.5, 0.))] == UNKNOWN


def test_forward_yaw_scan_exposes_near_body_blind_slab_without_inventing_free_cells():
    # Nominal OAK 80deg HFOV, 640x400 mono downsampled by .5 in host_depth;
    # camera mount x=.116,y=.0375,z=.029m from the simulated sensor fixture.
    # This synthetic camera has the same forward-only near-body coverage issue.
    g = DepthGrid(GridParams(extent_m=6., border_px=4))
    g.reset((0., 0.), 1.)
    focal = 320. / (2 * math.tan(math.radians(80) / 2))
    K = (focal, focal, 159.5, 99.5)
    for i, yaw in enumerate(np.linspace(0., 2*math.pi, 24, endpoint=False)):
        R, t = optical_pose(yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        mount = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) @ np.array([.116, .0375, .029])
        g.integrate(np.full((200, 320), 2.), K, R, t+mount, 1.+i*.05)
    assert np.count_nonzero(g.collapsed() == FREE) > 300
    assert not g.navigation()[0][g.cell((0., 0.))]
    assert not g.frontiers((0., 0.))
    summary = g.summary((0., 0.))
    assert summary['start_footprint_unknown_cells'] > 0
    assert summary['start_footprint_occupied_cells'] == 0
    assert not summary['start_traversable']
    # This is a correct bounded hold for this sensor placement, not a claim of
    # complete course capability. A wide-angle/down/up view is needed.


def downward_scene(patch_value=40):
    h = w = 160
    K = (100., 100., 79.5, 79.5)
    # Optical x world-x, y world-minus-y, z down; camera 2m over floor.
    R = np.diag([1., -1., -1.])
    t = np.array([0., 0., 2.])
    mono = np.full((h, w), 130, np.uint8)
    mono[65:95, 65:95] = patch_value
    depth = np.full((h, w), 2., float)
    return mono, K, R, t, depth


@pytest.mark.parametrize('intensity', [30, 230])
def test_mono_detector_proposes_both_bright_and_dark_floor_contrast(intensity):
    image, K, R, t, depth = downward_scene(intensity)
    results = MonoFloorDetector().detect(image, K, R, t, 0., depth, K, R, t)
    assert len(results) == 1
    assert np.linalg.norm(results[0].xy) < .05
    assert results[0].contrast > 50
    assert results[0].reason == 'floor contrast; color unobserved'


def test_blank_gradient_missing_depth_and_raised_dark_rock_are_not_targets():
    image, K, R, t, depth = downward_scene()
    detector = MonoFloorDetector()
    for blank in (np.full_like(image, 130), np.tile(np.arange(160, dtype=np.uint8), (160, 1))):
        assert not detector.detect(blank, K, R, t, 0., depth, K, R, t)
    missing = np.full_like(depth, np.nan)
    assert not detector.detect(image, K, R, t, 0., missing, K, R, t)
    raised = depth.copy()
    raised[60:100, 60:100] = 1.5
    assert not detector.detect(image, K, R, t, 0., raised, K, R, t)


def test_depth_and_mono_have_separate_extrinsics_and_intrinsics():
    R = np.diag([1., -1., -1.])
    world = np.array([[0., 0., 0.], [.1, .1, 0.]])
    Kd = (80., 90., 80., 60.)
    td = np.array([.3, 0., 2.])
    support, error = validate_depth(world, np.full((120, 160), 2.), Kd, R, td)
    assert support == 1. and error == 0.
    wp, ok = floor_intersections([[80., 60.]], Kd, R, td, 0.)
    assert ok[0] and np.allclose(wp[0], [.3, 0., 0.])


def candidate(x=0., y=0.):
    return Candidate((x, y), .3, 70., (1, 1, 20, 20), .01)


def test_repeated_frames_or_pure_yaw_cannot_confirm_a_target():
    tracker = TargetTracker(DetectorParams(qualified=True))
    for i in range(10):
        assert tracker.update([candidate()], (0., 0.), i*.2+1.) is None
    assert not tracker.tracks[0]['geometric']


def test_geometric_shadow_remains_candidate_without_mono_qualification():
    tracker = TargetTracker()
    for i in range(6):
        assert tracker.update([candidate()], (i*.06, 0.), i*.2+1.) is None
    assert tracker.tracks[0]['geometric']
    assert tracker.status()['color_observed'] is False


def test_qualified_target_needs_viewpoint_and_multiple_targets_stay_ambiguous():
    tracker = TargetTracker(DetectorParams(qualified=True))
    for i in range(6):
        tracker.update([candidate()], (i*.06, 0.), i*.2+1.)
    assert tracker.confirmed is not None
    for i in range(6, 12):
        tracker.update([candidate(), candidate(1.)], (i*.06, 0.), i*.2+1.)
    assert tracker.confirmed is None
    tracker.update([], (2., 0.), 10.)
    assert not tracker.tracks and tracker.confirmed is None


def test_new_strategy_imports_and_operates_with_no_course_file_access(monkeypatch):
    original = builtins.open
    def guarded(path, *args, **kwargs):
        if any(token in str(path).lower() for token in ('course', '.obj', 'waypoint', 'maps/')):
            raise AssertionError('origin-map access')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', guarded)
    g = grid_free()
    m = active_mission(g)
    assert m.tick(obs()).velocity is not None
    root = Path(__file__).resolve().parents[1] / 'kestrel_nav/searchnav'
    for file in root.glob('*.py'):
        text = file.read_text()
        assert 'import coursenav' not in text and 'from coursenav' not in text
        assert 'sim/truth' not in text and 'VehicleCommand' not in text


def test_depth_integration_does_not_require_map_particle_filter_or_its_confidence():
    # Actual-flight regression: the retired localizer gated892/894 scans on PF
    # sigma and left occupied=0. The new entry point takes no PF/map-confidence
    # input: calibrated depth integrates independently of any localizer process.
    import inspect
    assert list(inspect.signature(DepthGrid.integrate).parameters) == [
        'self', 'depth_m', 'K', 'R_world_cam', 't_world_cam', 'now']
    g = DepthGrid(GridParams(extent_m=6., border_px=4))
    g.reset((0., 0.), 1.)
    R, t = optical_pose()
    changed = g.integrate(np.full((120, 160), 2.), (100., 100., 79.5, 59.5), R, t, 2.)
    assert changed > 0 and (g.collapsed() == OCCUPIED).any()
    assert g.summary()['prior_loaded'] is False


def forward_floor_depth(tilt_deg=0.):
    h, w = 120, 160
    K = (100., 100., 79.5, 59.5)
    R, t = optical_pose(z=.18)
    vs, us = np.mgrid[0:h, 0:w]
    rays = np.stack([(us-K[2])/K[0], (vs-K[3])/K[1], np.ones((h,w))], axis=-1) @ R.T
    # Measured scene plane z=tan(tilt)*x, not a fitter prior.
    slope = math.tan(math.radians(tilt_deg))
    denom = rays[:,:,2]-slope*rays[:,:,0]
    depth = np.full((h,w), np.nan)
    good = denom < -.02
    depth[good] = -(t[2]-slope*t[0])/denom[good]
    depth[(depth<.3)|(depth>6.)] = np.nan
    return depth, K, R, t, np.array([0.,0.,.15])


def test_floor_fallback_measures_horizontal_ground_with_no_nominal_height():
    depth,K,R,t,body = forward_floor_depth()
    measurement,why = fit_floor(depth,K,R,t,body)
    assert measurement is not None, why
    assert abs(measurement['floor_z']) < .01
    assert measurement['area_m2'] >= .2 and measurement['normal'][2] > .99
    # Changing local origin changes the measured floor coordinate equally.
    shift=np.array([20.,-10.,3.])
    moved,why=fit_floor(depth,K,R,t+shift,body+shift)
    assert moved is not None and moved['floor_z'] == pytest.approx(3.,abs=.01)


def test_floor_fallback_rejects_wall_blank_and_nonhorizontal_surface():
    depth,K,R,t,body = forward_floor_depth()
    for bad in (np.full_like(depth,np.nan), np.full_like(depth,2.)):
        measurement,_=fit_floor(bad,K,R,t,body)
        assert measurement is None
    sloped,K,R,t,body=forward_floor_depth(8.)
    measurement,_=fit_floor(sloped,K,R,t,body)
    assert measurement is None


def test_floor_qualification_needs_stable_distinct_frames_and_reset_after_failure():
    depth,K,R,t,body = forward_floor_depth()
    measurement,_=fit_floor(depth,K,R,t,body)
    tracker=FloorTracker()
    assert tracker.update(measurement,1.) is None
    assert tracker.update(measurement,1.) is None
    assert tracker.update(measurement,1.25) is None
    assert tracker.update(measurement,1.5) is not None
    tracker.update(None,1.75,'no floor depth')
    assert tracker.measurement is None and not tracker.history


def test_transient_rejected_depth_pauses_without_refreshing_usable_deadline():
    guard=DepthWatchdog(.5)
    assert guard.action(1.)=='stale'
    assert guard.usable(1.,1.)
    guard.reject('sparse stereo')
    assert guard.action(1.25)=='pause'
    guard.reject('still sparse')
    assert guard.action(1.51)=='stale'
    assert guard.capture==1.
    assert guard.usable(1.6,1.6)
    assert guard.action(1.7)=='run'
    assert not guard.usable(1.6,1.8)  # replay cannot refresh receipt.
    assert guard.action(2.11)=='stale'


def test_depth_watchdog_rejects_stale_future_and_backwards_clocks():
    guard=DepthWatchdog(.5)
    assert not guard.usable(1.,2.)
    assert not guard.usable(3.,2.)
    assert guard.usable(2.,2.)
    assert guard.action(1.9)=='stale'


def test_px4_clock_anchor_rejects_zero_held_tick_and_initial_forward_clock_jump():
    clock=Px4ClockAnchor()
    assert clock.observe(0.,2.) is None
    assert clock.observe(.5,3.) is None
    for boot in (3.1,3.2,3.3):
        assert clock.observe(.5,boot) is None
    assert clock.offset is None
    for t in (17.,17.0625,17.125,17.1875,17.25):
        clock.observe(t,t)
    assert clock.offset==0.
    assert clock.first_ros==0. and clock.anchor['span_s']>=.25
    assert clock.observe(17.25,17.25) is None  # duplicate does not move offset.
    assert clock.offset==0.


def test_ground_clock_discontinuity_requires_new_anchor_but_airborne_yields():
    clock=Px4ClockAnchor()
    for t in (1.,1.0625,1.125,1.1875,1.25):
        clock.observe(t,t)
    assert clock.offset==0.
    assert clock.observe(10.,1.3) is None
    assert clock.offset is None and clock.resets==1
    for t in (10.1,10.2,10.3,10.4):
        clock.observe(t,t-8.7)
    assert clock.offset==pytest.approx(8.7)
    assert clock.observe(11.5,1.75,active=True) is None
    assert clock.fault and clock.offset==pytest.approx(8.7)  # never accepts new origin in flight.
