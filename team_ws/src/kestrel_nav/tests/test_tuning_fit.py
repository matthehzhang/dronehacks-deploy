"""Offline response identification rejects unsupported fits; never commands ROS."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from fit_tuning import analyze, prepare_trials, response_basis
from rehearse_tuning import synthetic_recording, _forward_distance, rehearse


@pytest.fixture(scope='module')
def clean():
    return synthetic_recording(tau_s=.3, delay_s=.15)


def test_exact_zoh_response_integrates_steps_between_observations():
    t = np.array([0., .7, 1.3])
    v, p = response_basis(t, np.array([-1., .2]), np.array([0., 1.]), .4, .1, 0.)
    duration = t[1:] - .3
    assert v[1:, 1] == pytest.approx(1 - np.exp(-duration/.4))
    assert p[1:, 1] == pytest.approx(duration - .4*(1 - np.exp(-duration/.4)))


def test_known_response_recovered_on_independent_trials(clean):
    result = analyze(clean)
    assert result['accepted'], result
    model = result['candidates']['synthetic_lane']['model']
    assert model['tau_s'] == pytest.approx(.3, abs=.02)
    assert model['delay_s'] == pytest.approx(.15, abs=.02)
    assert model['gain'] == pytest.approx(1., abs=.02)
    assert abs(model['bias_mps']) < .002


def test_changed_validation_dynamics_are_rejected():
    result = analyze(synthetic_recording(tau_s=.3, delay_s=.15, validation_tau_s=.7))
    assert not result['accepted']
    assert any('validation' in x for x in result['candidates']['synthetic_lane']['reasons'])


def test_validation_does_not_select_fitted_model(clean):
    base = analyze(clean)['candidates']['synthetic_lane']['model']
    rows = copy.deepcopy(clean)
    for row in rows:
        if row.get('split') == 'validation' and row['kind'] == 'odom':
            row['position'][0] *= 3
    changed = analyze(rows)
    assert changed['candidates']['synthetic_lane']['model'] == base


@pytest.mark.parametrize('mutation', ['missing_summary', 'missing_header', 'dropped', 'missing_split',
                                     'clamp', 'fault', 'unhealthy', 'unaccepted_odom',
                                     'frame_change', 'duplicate_stamp', 'nonfinite'])
def test_invalid_evidence_rejected(clean, mutation):
    rows = copy.deepcopy(clean)
    odom = [r for r in rows if r['kind'] == 'odom']
    command = next(r for r in rows if r['kind'] == 'command' and r['phase'] == 'PULSE')
    if mutation == 'missing_summary':
        rows.pop()
    elif mutation == 'missing_header':
        rows.pop(0)
    elif mutation == 'dropped':
        rows[-1]['dropped_count'] = 1
    elif mutation == 'missing_split':
        del command['split']
    elif mutation == 'clamp':
        command['clamped'] = ['speed']
    elif mutation == 'fault':
        command['phase'] = 'FAULT'
    elif mutation == 'unhealthy':
        command['health'] = False
    elif mutation == 'unaccepted_odom':
        next(r for r in odom if r['phase'] == 'PULSE')['accepted'] = False
    elif mutation == 'frame_change':
        odom[2]['frame_id'] = 'base_link'
    elif mutation == 'duplicate_stamp':
        odom[2]['source_stamp'] = odom[1]['source_stamp']
    elif mutation == 'nonfinite':
        odom[2]['position'][0] = float('nan')
    result = analyze(rows)
    assert not result['accepted']


def test_no_validation_trials_rejected(clean):
    rows = [r for r in clean if r.get('split') != 'validation']
    result = analyze(rows)
    assert not result['accepted']
    assert 'separate train' in result['candidates']['synthetic_lane']['reasons'][0]


def test_waiting_disarmed_before_trial_is_not_an_active_fault(clean):
    rows = copy.deepcopy(clean)
    for r in rows:
        if r.get('phase') == 'SETTLE' and r.get('trial') == 0:
            r.update(offboard_active=False, health=False, phase='TRANSIT')
    assert analyze(rows)['accepted']


def test_trial_without_excitation_is_rejected(clean):
    rows = copy.deepcopy(clean)
    for r in rows:
        if r['kind'] == 'command':
            r['velocity_enu'] = [0., 0., 0.]
    assert not analyze(rows)['accepted']


def test_matching_body_frames_are_not_mistaken_for_fixed_enu(clean):
    rows = copy.deepcopy(clean)
    for r in rows:
        if r['kind'] in ('odom', 'command'):
            r['frame_id'] = 'base_link'
    assert not analyze(rows)['accepted']


def test_malformed_records_are_rejected_without_fitting():
    assert not analyze([[]])['accepted']


def test_missing_command_heartbeat_is_not_extrapolated_as_zoh(clean):
    rows = [r for r in clean if not (r.get('kind') == 'command' and 12.4 < r['t_ros'] < 12.6)]
    result = analyze(rows)
    assert not result['accepted']
    assert any('command gap' in x['reason'] for x in result['rejected_trials'])


def test_clean_shutdown_mid_diagnostic_cannot_be_accepted(clean):
    rows = copy.deepcopy(clean)
    rows[-1]['terminal_phase'] = 'BRAKE'
    result = analyze(rows)
    assert not result['accepted']
    assert any('finish DONE' in x for x in result['errors'])


def test_done_label_cannot_hide_missing_configured_trials(clean):
    rows = copy.deepcopy(clean)
    rows[0]['config']['speeds_mps'].append(.3)
    result = analyze(rows)
    assert not result['accepted']
    assert any('schedule' in x for x in result['errors'])


def test_truth_is_not_a_fitting_input(clean):
    rows = copy.deepcopy(clean)
    for r in rows:
        r['truth'] = {'position': [9999., -9999., 0.], 'velocity': [500., 500., 0.]}
    assert analyze(rows) == analyze(clean)


def test_return_braking_not_mixed_with_pulse(clean):
    rows = copy.deepcopy(clean)
    # Same trial ID returns later; absurd return data must not enter the fit.
    end_index = next(i for i, r in enumerate(rows) if r.get('trial') == 1)
    end_t = rows[end_index - 1]['t_ros']
    extra = copy.deepcopy(rows[end_index - 2])
    extra.update(t_ros=end_t + .05, source_stamp=end_t + .05, phase='CREEP')
    rows.insert(end_index, extra)
    bad_command = copy.deepcopy(extra)
    bad_command.update(t_ros=end_t + .1, source_stamp=end_t + .1, phase='BRAKE')
    bad_odom = copy.deepcopy(rows[end_index - 1])
    bad_odom.update(t_ros=end_t + .1, source_stamp=end_t + .1, phase='BRAKE',
                    position=[999., 999., .8])
    rows[end_index + 1:end_index + 1] = [bad_command, bad_odom]
    # Parser cuts the outbound response at CREEP rather than selecting all BRAKE.
    old, _, _ = prepare_trials(clean)
    new, errors, rejected = prepare_trials(rows)
    assert not errors and not rejected
    assert new[0].velocity == pytest.approx(old[0].velocity)


def test_lane_sensor_geometry_preserves_open_end_and_side_wall():
    lane = {'start_xy': [0., 0.], 'end_xy': [3., 0.], 'half_width_m': .5,
            'start_turn_radius_m': .7, 'end_turn_radius_m': .7}
    assert _forward_distance(np.array([1.5, 0.]), 0., lane) == pytest.approx(2.2)
    assert _forward_distance(np.array([1.5, 0.]), np.pi/2, lane) == pytest.approx(.5)


def test_production_diagnostic_core_completes_synthetic_lane_and_produces_heldout_data():
    # Synthetic geometry is a software fixture, not a measured course candidate.
    config = {'start_course': [0., 0., 0.], 'speeds_mps': [.1], 'time_limit_s': 170.,
              'candidates': [{'name': 'fixture', 'kind': 'open', 'start_xy': [0., 0.],
                              'end_xy': [2., 0.], 'height_m': .8, 'half_width_m': .6,
                              'start_turn_radius_m': .8, 'end_turn_radius_m': .8}]}
    rows = rehearse(config, tau_s=.45, delay_s=.15)
    assert rows[-1]['terminal_phase'] == 'DONE', rows[-1]
    assert len(rows[-1]['completed']) == 2
    result = analyze(rows)
    assert result['accepted'], result
    model = result['candidates']['fixture']['model']
    assert model['tau_s'] == pytest.approx(.45, abs=.02)
    assert model['delay_s'] == pytest.approx(.15, abs=.02)
