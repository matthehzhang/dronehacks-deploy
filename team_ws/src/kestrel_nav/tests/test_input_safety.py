"""Watchdog failure injection: no ROS, aircraft or simulator process."""
import math

import numpy as np
import pytest

from kestrel_nav.coursenav.depth import DepthFrame
from kestrel_nav.livenav.safety import InputSafety, SafetyParams, corridor_error


def feed(safety, now, skip=()):
    for stream in ('state', 'odom', 'tracking', 'depth'):
        if stream not in skip:
            safety.accept(stream, now, now, now + 1000)


def decide(safety, now, **overrides):
    inputs = dict(offboard=False, armed=False, landed=True, link=True, position_valid=True)
    inputs.update(overrides)
    return safety.evaluate(now, now + 1000, **inputs)


def ready():
    safety = InputSafety()
    for now in np.arange(100, 101.21, .2):
        feed(safety, now)
        decide(safety, now)
    assert safety.action == 'ready'
    return safety


def flying():
    safety = ready()
    feed(safety, 101.3)
    assert decide(safety, 101.3, offboard=True, armed=True) == 'run'
    return safety


@pytest.mark.parametrize('missing', ['state', 'odom', 'tracking', 'depth'])
def test_missing_stream_never_qualifies(missing):
    safety = InputSafety()
    for now in (100, 100.4, 101, 102):
        feed(safety, now, skip=(missing,))
        assert decide(safety, now) == 'silent'


@pytest.mark.parametrize('stream', ['odom', 'tracking', 'depth'])
def test_sensor_loss_stops_and_recovery_cannot_resume(stream):
    safety = flying()
    feed(safety, 102, skip=(stream,))
    assert decide(safety, 102, offboard=True, armed=True, landed=False) == 'stop'
    original = safety.fault
    feed(safety, 102.1)
    assert decide(safety, 102.1, offboard=True, armed=True, landed=False) == 'stop'
    assert safety.fault == original


def test_duplicate_samples_cannot_extend_liveness():
    safety = flying()
    for now in (101.4, 101.6, 101.9):
        feed(safety, now, skip=('odom',))
        assert not safety.accept('odom', 101.3, now, now + 1000)
    assert decide(safety, 101.9, offboard=True, armed=True, landed=False) == 'stop'


@pytest.mark.parametrize('stamp', [0, math.nan, math.inf, 99, 100.1])
def test_invalid_delayed_and_future_timestamps_rejected(stamp):
    safety = InputSafety()
    assert not safety.accept('depth', stamp, 100, 1100)
    assert 'depth' in safety.errors


def test_stopped_ros_clock_still_expires_by_monotonic_time():
    safety = flying()
    assert safety.stream_error('odom', 101.3, 1102) == 'odom: stale receipt or clock discontinuity'


def test_bad_sample_between_ticks_is_latched():
    safety = flying()
    safety.reject('tracking', 'tracking failed')
    feed(safety, 101.4)
    assert decide(safety, 101.4, offboard=True, armed=True) == 'stop'
    assert safety.fault == 'tracking failed'


def test_authority_loss_is_silent_and_cannot_resume():
    safety = flying()
    feed(safety, 101.4)
    assert decide(safety, 101.4, armed=True, landed=False) == 'silent'
    feed(safety, 101.5)
    assert decide(safety, 101.5, offboard=True, armed=True, landed=False) == 'stop'


def test_stale_authority_yields_even_when_sensors_are_fresh():
    safety = flying()
    feed(safety, 102.4, skip=('state',))
    assert decide(safety, 102.4, offboard=True, armed=True) == 'silent'
    assert safety.fault


def test_starting_node_on_armed_aircraft_never_starts_mission():
    safety = InputSafety()
    for now in (100, 100.2, 100.4, 100.6, 100.8, 101, 101.2):
        feed(safety, now)
        assert decide(safety, now, armed=True) == 'silent'
    feed(safety, 101.3)
    assert decide(safety, 101.3, armed=True, offboard=True) == 'stop'


def test_fault_does_not_clear_on_disarm():
    safety = flying()
    safety.reject('odom', 'invalid pose')
    feed(safety, 101.4)
    assert decide(safety, 101.4) == 'silent'
    assert safety.fault == 'invalid pose'


def test_operator_hold_is_not_overridden_by_zero_twist():
    safety = flying()
    feed(safety, 101.4)
    assert decide(safety, 101.4, armed=True, offboard=True, frozen=True) == 'silent'
    assert 'hold requested' in safety.fault


@pytest.mark.parametrize('changes', [dict(odom_max_age_s=0), dict(depth_max_age_s=math.nan),
                                     dict(min_corridor_coverage=0), dict(min_corridor_rays=0)])
def test_invalid_guard_configuration_fails_at_startup(changes):
    with pytest.raises(ValueError):
        SafetyParams(**changes)


def frame(depth):
    return DepthFrame(depth, (200, 200, 160, 120),
                      np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]), np.zeros(3), np.eye(3), stride=4)


@pytest.mark.parametrize('velocity', [(0, 0.2), (-0.2, 0)])
def test_sideways_and_reverse_motion_without_view_are_blocked(velocity):
    error = corridor_error(frame(np.full((240, 320), 2.0)), *velocity,
                           half_width=.36, band=.25, at_m=1.25, params=SafetyParams())
    assert error and 'unobserved' in error


def test_global_validity_does_not_hide_blank_motion_corridor():
    depth = np.full((240, 320), 2.0)
    depth[75:165, 95:225] = np.nan
    view = frame(depth)
    assert view.valid_frac > .8
    assert corridor_error(view, .2, 0, half_width=.36, band=.25, at_m=1.25, params=SafetyParams())
    assert corridor_error(frame(np.full_like(depth, 2)), .2, 0, half_width=.36, band=.25,
                          at_m=1.25, params=SafetyParams()) is None
