"""Small EKF resets may rebuild spatial state; they never excuse sensor loss."""
from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

from kestrel_nav.searchnav.inputs import verify_small_reset
from kestrel_nav.searchnav.grid import DepthGrid, GridParams, FREE, OCCUPIED, UNKNOWN
from kestrel_nav.searchnav.mission import SearchMission, MissionParams, Observation


def observation(t=2., **kw):
    values = dict(now=t, position=(0., 0., 1.), yaw=0., ready=True,
                  armed=True, offboard=True, landed=False)
    values.update(kw)
    return Observation(**values)


def mission():
    grid = DepthGrid(GridParams(extent_m=6.))
    grid.reset((0., 0.), 1.)
    grid.apply_observation(np.ones_like(grid.states, bool), np.zeros_like(grid.states, bool), 1.)
    m = SearchMission(grid)
    m.started, m.phase, m.phase_t = 1., 'move', 1.
    m.hold_xy, m.hold_z = np.zeros(2), 1.
    m.route = [np.zeros(2), np.array([2., 0.])]
    m.route_goal, m.goal_kind = np.array([2., 0.]), 'frontier'
    m.route_revision = grid.revision
    return m


def test_actual_small_xy_reset_ignores_retained_ground_yaw_delta():
    message = SimpleNamespace(delta_xy=[-.01190124, -.00908326],
                              delta_z=float('nan'), delta_heading=1.5683)
    decision, reason = verify_small_reset((3, 0, 1), (4, 0, 1), message)
    assert decision is not None, reason
    assert decision['delta_heading_ned'] == decision['delta_z_enu'] == 0.
    assert math.hypot(*decision['delta_xy_ned']) == pytest.approx(.0149714, abs=1e-6)


def test_uint8_wrap_is_one_reset_but_skipped_large_and_nonfinite_resets_refuse():
    message = SimpleNamespace(delta_xy=[.01, 0.], delta_z=.02, delta_heading=.01)
    assert verify_small_reset((255, 255, 255), (0, 0, 0), message)[0] is not None
    for counters, changed in [((2, 0, 0), {}), ((1, 0, 0), {'delta_xy':[.11, 0.]}),
                              ((0, 1, 0), {'delta_z':float('nan')}),
                              ((0, 0, 1), {'delta_heading':math.radians(6.)})]:
        m = SimpleNamespace(**(vars(message) | changed))
        assert verify_small_reset((0, 0, 0), counters, m)[0] is None


def test_recovery_withdraws_route_map_and_waits_for_qualification_without_yaw():
    m = mission()
    assert m.begin_reset_recovery(2., (.01, 0., 1.02), .02, 'verified reset')
    assert not m.route and not m.retry_goal
    assert np.all(m.grid.states == UNKNOWN) and m.hold_z == pytest.approx(1.02)
    assert m.started == 1.  # Total mission deadline is never restarted.
    waiting = m.tick(observation(2.1, ready=False, position=(.01, 0., 1.02)))
    assert waiting.phase == 'recover' and waiting.velocity == (0., 0., 0., 0.)
    resumed = m.tick(observation(5.1, position=(.01, 0., 1.02)))
    assert resumed.phase == 'scan' and resumed.velocity == (0., 0., 0., 0.)
    assert not m.grid.navigation()[0].any()  # Requalification cannot invent free cells.


def test_repeated_resets_cannot_extend_recovery_timeout_and_authority_fault_latches():
    m = mission()
    assert m.begin_reset_recovery(2., (0., 0., 1.), 0., 'first')
    assert m.begin_reset_recovery(4., (0., 0., 1.), 0., 'second')
    assert m.recovery_started == 2.
    result = m.tick(observation(7.01, ready=False))
    assert result.phase == 'yield' and result.velocity is None
    assert not m.begin_reset_recovery(7.1, (0., 0., 1.), 0., 'too late')
    for change in ({'inputs_fresh':False}, {'offboard':False}, {'armed':False}, {'fault':'VIO invalid'}):
        m = mission()
        m.begin_reset_recovery(2., (0., 0., 1.), 0., 'reset')
        assert m.tick(observation(2.1, ready=False, **change)).velocity is None
        assert m.phase == 'yield'


def test_reset_count_bound_and_original_mission_deadline_still_apply():
    m = mission()
    for i in range(3):
        assert m.begin_reset_recovery(2.+i*.1, (0., 0., 1.), 0., 'reset')
    assert not m.begin_reset_recovery(2.4, (0., 0., 1.), 0., 'fourth')
    assert m.phase == 'yield'
    m = mission()
    m.begin_reset_recovery(180., (0., 0., 1.), 0., 'late reset')
    assert m.tick(observation(181.1, ready=False)).phase == 'exhausted'


def test_scan_can_choose_frontier_before_completing_rotation():
    m = mission()
    # The only unknown room lies east, with reachable observed-safe stand-off views.
    m.grid.states[:, :, m.grid.n//2+35:] = UNKNOWN
    m.grid._changed(np.ones((m.grid.n, m.grid.n), bool))
    m.begin_scan(observation(), 'new view')
    result = m.tick(observation(2.1, yaw=.05))
    assert result.phase == 'align' and m.route and m.goal_kind == 'frontier'
    assert m.scan_angle < .1 and m.grid.route_safe(m.route)
    assert m.counts['scan_plan_checks'] == 1


def test_unknown_start_checks_are_rate_limited_and_do_not_restart_scan_deadline():
    m = mission()
    m.grid.reset((0., 0.), 1.)
    m.begin_scan(observation(), 'unobserved start')
    for i in range(1, 15):
        # New revisions alone cannot force the expensive chooser every timer tick.
        m.grid.revision += 1
        result = m.tick(observation(2.+i*.05, yaw=i*.02))
        assert result.velocity[:2] == (0., 0.) and not m.route
    assert 1 <= m.counts['scan_plan_checks'] <= 2
    assert m.phase_t == 2. and m.scan_angle > .2
    assert m.plan_rejections[0]['kind'] == 'start'
    assert 'unknown' in m.grid.frontier_summary['reason']


def test_new_obstacle_detours_to_same_frontier_before_selecting_another():
    m = mission()
    goal = m.route_goal.copy()
    hit = np.zeros_like(m.grid.states, bool)
    ij = m.grid.cell((1., 0.))
    hit[:, ij[0], ij[1]] = True
    m.grid.apply_observation(np.zeros_like(hit), hit, 2.)
    stopped = m.tick(observation(2.))
    assert stopped.velocity[:2] == (0., 0.) and not m.route
    # Do not plan until measured speed has settled.
    assert m.tick(observation(2.1, velocity=(.1, 0., 0.))).phase == 'plan'
    assert not m.route
    m.tick(observation(2.2))
    assert m.route and len(m.route) > 2 and m.grid.route_safe(m.route)
    assert np.array_equal(m.route_goal, goal) and m.goal_kind == 'frontier'
    assert not m.rejected_goals and not m.failed_frontiers
