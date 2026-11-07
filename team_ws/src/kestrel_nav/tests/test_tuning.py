"""Pure bounded experiment, synthetic vehicle only. No ROS, DDS, or aircraft."""
import copy
import math

import numpy as np
import pytest

from kestrel_nav.livenav.tuning import TuningController, TuningObservation, validate_config


def config():
    return dict(start_course=[0., 0., 0.], speeds_mps=[.1], time_limit_s=160., candidates=[
        dict(name='synthetic_open', kind='open', start_xy=[0., 0.], end_xy=[2., 0.],
             height_m=.8, half_width_m=.7, start_turn_radius_m=.7, end_turn_radius_m=.7)])


def observation(now=1., **kw):
    values = dict(now=now, position=(0., 0., .8), yaw=0., velocity=(0., 0., 0.),
                  front_clearance=4., coverage=1., transit_complete=True, yaw_rate=0.)
    values.update(kw)
    return TuningObservation(**values)


@pytest.mark.parametrize('missing', ['start_course', 'candidates'])
def test_required_geometry(missing):
    c = config(); del c[missing]
    with pytest.raises(ValueError): validate_config(c)


@pytest.mark.parametrize('key,value', [('end_xy', [.5, 0]), ('half_width_m', .3),
                                      ('start_turn_radius_m', .1), ('height_m', math.nan)])
def test_unsafe_lane_refused(key, value):
    c = config(); c['candidates'][0][key] = value
    with pytest.raises(ValueError): validate_config(c)


def test_tube_requires_explicit_open_ends():
    c = config(); c['candidates'][0]['kind'] = 'tube'
    with pytest.raises(ValueError, match='open'): validate_config(c)
    c['candidates'][0].update(start_open=True, end_open=True)
    assert validate_config(c)['candidates'][0]['kind'] == 'tube'


def test_trial_order_and_budget_refused_before_flight():
    c = config(); c['speeds_mps'] = [.2]
    with pytest.raises(ValueError, match='speeds'): validate_config(c)
    c = config(); c['candidates'][0]['end_xy'] = [8., 0.]
    with pytest.raises(ValueError, match='budget'): validate_config(c)


def test_rotation_schedule_counts_against_same_flight_budget():
    c = config()
    with_yaw = validate_config(c)
    c['yaw_rates_rps'] = []
    translation_only = validate_config(c)
    assert with_yaw['estimated_lane_s'] == translation_only['estimated_lane_s']+26
    assert with_yaw['yaw_rates_rps'] == [.15, .25]
    for rates in ([.2], [.15, .3], [.15, .1], [.1, .15, .2], [math.nan]):
        c['yaw_rates_rps'] = rates
        with pytest.raises(ValueError, match='yaw rates'):
            validate_config(c)


@pytest.mark.parametrize('change', [dict(imu_ok=False), dict(offboard=False), dict(clamped=True),
                                   dict(tracking_ok=False), dict(front_clearance=math.nan),
                                   dict(front_clearance=-math.inf)])
def test_bad_inputs_latch_zero_without_retry(change):
    m = TuningController(config())
    cmd = m.step(observation(**change))
    assert cmd.state == 'FAULT' and (cmd.vx, cmd.vy, cmd.vz, cmd.yaw_rate) == (0, 0, 0, 0)
    assert m.step(observation(2.)).state == 'FAULT'


def test_transit_does_not_invent_a_straight_line_navigation_command():
    m = TuningController(config())
    cmd = m.step(observation(transit_complete=False, position=(-2., 0., .8)))
    assert cmd.state == 'TRANSIT' and cmd.vx == cmd.vy == 0


def test_turn_at_stopped_offset_stays_inside_declared_clear_disk():
    m = TuningController(config()); m.state = 'TURN'; m.after_turn = 'RETURN'; m.phase_t = 1.; m.started = 1.
    cmd = m.step(observation(1.1, position=(1.94, 0., .8), front_clearance=.64))
    assert cmd.state == 'TURN' and cmd.vx == cmd.vy == 0 and cmd.yaw_rate != 0
    assert m.step(observation(1.2, position=(1.94, 0., .8), front_clearance=.4)).state == 'FAULT'


@pytest.mark.parametrize('phase', ['TURN', 'YAW_PULSE'])
def test_turn_disk_uses_radial_distance_not_forward_projection(phase):
    m = TuningController(config()); m.state = phase; m.phase_t = m.started = 1.
    # A diagonal wall can have forward projection below the disk radius while
    # remaining outside the disk. It must not cause a false turn refusal.
    cmd = m.step(observation(1.1, yaw=.1, front_clearance=.62, turn_clearance=.72))
    assert cmd.state != 'FAULT'
    # Conversely, a visible side obstacle cannot be ignored by a clear front.
    cmd = m.step(observation(1.2, front_clearance=4., turn_clearance=.5))
    assert cmd.state == 'FAULT'


def test_invalid_radial_turn_measurement_fails_closed():
    for value in (math.nan, -1., -math.inf):
        m = TuningController(config())
        assert m.step(observation(turn_clearance=value)).state == 'FAULT'


def test_forward_camera_required_for_return():
    m = TuningController(config()); m.state = 'RETURN'; m.phase_t = m.started = 1.
    cmd = m.step(observation(1.1, position=(1., 0., .8), yaw=0.))
    assert cmd.state == 'FAULT' and cmd.vx == 0


def test_complete_pair_with_response_lag_and_no_blind_reverse():
    m = TuningController(config())
    position, velocity, yaw, yaw_rate = np.array([0., 0., .8]), np.zeros(3), 0., 0.
    phases, trial_ids, pulse_positions = set(), set(), {}
    rotation_ids, translation_started = set(), False
    for now in np.arange(.1, 159., .1):
        cmd = m.step(observation(float(now), position=tuple(position), velocity=tuple(velocity), yaw=yaw, yaw_rate=yaw_rate))
        phases.add(cmd.state)
        if cmd.state == 'PULSE':
            translation_started = True
            trial_ids.add(m.trial)
            pulse_positions.setdefault(m.trial, position.copy())
        if cmd.state.startswith('YAW_'):
            assert not translation_started
            assert np.linalg.norm(position[:2]) < .15
            assert cmd.vx == cmd.vy == 0
            if cmd.yaw_rate:
                rotation_ids.add(m.rotation_trial)
                assert math.copysign(1., cmd.yaw_rate) == (1. if m.rotation_trial % 2 == 0 else -1.)
        if math.hypot(cmd.vx, cmd.vy) > 1e-6:
            assert cmd.vx*math.cos(yaw)+cmd.vy*math.sin(yaw) >= -1e-8
        velocity += .1/.3*(np.array([cmd.vx, cmd.vy, cmd.vz])-velocity)
        position += velocity*.1
        yaw_rate = cmd.yaw_rate
        yaw += yaw_rate*.1
        if cmd.state in ('DONE', 'FAULT'): break
    assert cmd.state == 'DONE', (cmd.state, cmd.note, m.status(), position)
    assert trial_ids == {0, 1} and len(m.completed) == 2
    assert len(m.completed_yaw) == 4
    assert rotation_ids == {0, 1, 2, 3}
    assert {'PULSE', 'BRAKE', 'TURN', 'RETURN', 'TRANSFER', 'TRANSFER_SETTLE'} <= phases
    assert pulse_positions[1][0] > pulse_positions[0][0]+.4
    assert np.linalg.norm(position[:2]) < .15


@pytest.mark.parametrize('change', [dict(yaw_rate=None), dict(yaw_rate=math.nan), dict(yaw_rate=.7),
                                   dict(yaw=.81), dict(position=(.16, 0., .8)),
                                   dict(position=(2., 0., .8)), dict(velocity=(.07, 0., 0.)),
                                   dict(front_clearance=.5), dict(coverage=.59), dict(tracking_ok=False)])
def test_rotational_trials_fail_closed_outside_open_start_envelope(change):
    m = TuningController(config())
    m.state, m.started, m.phase_t = 'YAW_PULSE', 1., 1.
    m.yaw_started, m.yaw_reference = 1., 0.
    cmd = m.step(observation(1.1, **change))
    assert cmd.state == 'FAULT' and cmd.yaw_rate == cmd.vx == cmd.vy == 0
    assert m.step(observation(1.2)).state == 'FAULT'


def test_rotational_stage_has_its_own_time_bound_and_requires_actual_excitation():
    m = TuningController(config())
    m.state, m.started, m.phase_t = 'YAW_SETTLE', 1., 27.
    m.yaw_started, m.yaw_reference = 1., 0.
    assert m.step(observation(27.1)).state == 'FAULT'
    m = TuningController(config())
    m.state, m.started, m.phase_t = 'YAW_BRAKE', 1., 1.
    m.yaw_brake_start = 0.
    m.step(observation(1.1))
    cmd = m.step(observation(2.2))
    assert cmd.state == 'FAULT' and 'excitation' in cmd.note


def test_out_of_lane_and_timeout_stop_instead_of_recovering():
    m = TuningController(config()); m.step(observation())
    assert m.step(observation(2., position=(0., .5, .8))).state == 'FAULT'
    m = TuningController(config()); m.step(observation())
    assert m.step(observation(162.)).state == 'FAULT'


def map_fixture():
    """Synthetic test fixture on published map, NOT measured deployment configuration."""
    c = config()
    c['start_course'] = [-2.325, -.125, -83.]
    c['candidates'][0].update(start_xy=[-2.325, -.275], end_xy=[-2.075, -2.325])
    return c


def test_configured_altitude_must_match_map_slice():
    from kestrel_nav.livenav.tuning import TuningProgram
    from kestrel_nav.livenav.mission import MissionParams
    from kestrel_nav.livenav.follower import FollowParams
    c = map_fixture(); c['candidates'][0]['height_m'] = .9
    with pytest.raises(ValueError, match='map planning-band height'):
        TuningProgram(c, MissionParams(use_pf=False), FollowParams())


def test_synthetic_rehearsal_config_is_rejected_by_physical_program():
    from kestrel_nav.livenav.tuning import TuningProgram
    c = config(); c['synthetic'] = True
    with pytest.raises(ValueError, match='synthetic rehearsal geometry'):
        TuningProgram(c, None, None)


@pytest.mark.xfail(strict=False, reason=(
    "branch kestral_nav934am: the diagnostic tuning mission is out of scope here; its synthetic lane "
    "fixture conflicts with the surface-measured published map (ported from workspace matthew930am)."))
def test_transit_handoff_precedes_final_waypoint_descent():
    from types import SimpleNamespace
    from kestrel_nav.livenav.tuning import TuningProgram
    from kestrel_nav.livenav.mission import MissionParams, Observation
    from kestrel_nav.livenav.follower import FollowParams
    m = TuningProgram(map_fixture(), MissionParams(use_pf=False), FollowParams())
    m.set_transform((0., 0.), 0.)
    m.state = 'FLY'; m.imu_ok = True
    m.frame = SimpleNamespace(corridor_coverage=lambda *args: (1., 100))
    # At 10 cm to the lane start: inside handoff, outside navigator's 8 cm arrival.
    dx, dy = m.course_vec_to_odom(0., -.05)
    out = m.step(Observation(now=10., odom_xyz=(dx, dy, .7), odom_yaw=0., offboard_active=True,
                             pts_level=np.array([[3., 0., 0.]]), pts_t=10.))
    assert out.state == 'SETTLE' and m.controller.state == 'SETTLE'
    assert out.vx == out.vy == 0 and out.vz >= 0
    m.controller.state = 'DONE'
    m.controller.reason = 'diagnostic complete; pilot lands'
    stopped = m.abort(11., 'Offboard authority lost; pilot owns vehicle')
    assert stopped.state == 'DONE' and m.controller.state == 'DONE' and m.fault is None
