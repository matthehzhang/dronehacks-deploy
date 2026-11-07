"""Independent raw PX4 source and receipt clocks; no ROS transport required."""
import math

import pytest

from kestrel_nav.livenav.land_safety import Px4LandSafety
from kestrel_nav.livenav.safety import SafetyParams


def reference(g, t):
    return g.observe_reference(t, t+90, t+1000)


def land(g, t, landed=True):
    return g.observe_land(t, landed, t+90, t+1000)


def qualified():
    g = Px4LandSafety(SafetyParams())
    for t in (10., 10.2):
        assert reference(g, t) and land(g, t)
    assert g.error(100.2, 1010.2) is None
    return g


def test_two_advancing_samples_required_and_duplicate_cannot_change_flag():
    g = Px4LandSafety(SafetyParams())
    assert reference(g, 10.) and land(g, 10.)
    assert 'advancing' in g.error(100., 1010.)
    assert not g.observe_land(10., False, 100.1, 1010.1)
    assert g.landed is True and g.sample[2] == 1010.
    assert reference(g, 10.2) and land(g, 10.2, False)
    assert g.error(100.2, 1010.2) is None and g.landed is False


def test_slow_heartbeat_allowed_but_duplicate_does_not_extend_deadline():
    g = qualified()
    for t in (10.5, 10.9, 11.3, 11.5):
        reference(g, t)
    assert land(g, 11.5) and g.error(101.5, 1011.5) is None
    for t in (12., 12.5, 13., 13.5, 14., 14.2):
        reference(g, t)
        assert not g.observe_land(11.5, True, t+90, t+1000)
    assert 'stale' in g.error(104.2, 1014.2)


@pytest.mark.parametrize('stamp', [11., 15.1])
def test_fresh_callback_cannot_hide_delayed_or_future_capture(stamp):
    g = qualified()
    reference(g, 15.)
    assert not g.observe_land(stamp, True, 105., 1015.)
    assert 'source timestamp' in g.error(105., 1015.)


def test_delayed_reference_cannot_rebase_source_clock():
    g = qualified()
    assert not g.observe_reference(10.3, 101., 1011.)
    assert g.offset == pytest.approx(90.)
    assert 'delayed capture' in g.error(101., 1011.)


@pytest.mark.parametrize('stream', ['reference', 'sample'])
def test_clock_rewind_requires_restart(stream):
    g = qualified()
    if stream == 'reference':
        g.observe_reference(10.1, 100.3, 1010.3)
    else:
        g.observe_land(10.1, True, 100.3, 1010.3)
    reference(g, 10.4)
    land(g, 10.4)
    assert 'rewind' in g.error(100.4, 1010.4)


def test_source_clock_jump_cannot_make_future_samples_fresh():
    g = qualified()
    assert not g.observe_reference(20., 100.3, 1010.3)
    assert 'jumped' in g.error(100.3, 1010.3)


def test_frozen_ros_clock_does_not_freeze_receipt_expiry():
    g = qualified()
    assert 'reference' in g.error(100.2, 1011.)


def test_land_receipt_expires_independently_of_capture_age():
    g = qualified()
    # ROS paused while monotonic advanced; only the reference is freshened.
    g.observe_reference(10.3, 100.3, 1013.)
    assert 'receipt' in g.error(100.3, 1013.)


def test_missing_reference_and_invalid_flag_fail_closed():
    g = Px4LandSafety(SafetyParams())
    assert not land(g, 10.)
    assert g.landed is None
    reference(g, 10.2)
    assert not g.observe_land(10.2, 1, 100.2, 1010.2)
    assert 'invalid landed' in g.error(100.2, 1010.2)


@pytest.mark.parametrize('name', ['land_max_age_s', 'px4_reference_max_age_s'])
@pytest.mark.parametrize('value', [0., -1., math.nan, math.inf])
def test_invalid_deadlines_rejected(name, value):
    with pytest.raises(ValueError):
        SafetyParams(**{name: value})
