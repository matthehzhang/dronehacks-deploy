"""hop_plan geometry. No ROS needed: ``python3 -m pytest -q test/``."""
import math

import pytest

from cage_test.hop_plan import (
    along_heading,
    build_legs,
    carrot,
    leg_length,
    yaw_from_xyzw,
)


def yaw_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


@pytest.mark.parametrize('yaw', [0.0, 0.5, math.pi / 2, -2.0, math.pi])
def test_yaw_from_xyzw_round_trip(yaw):
    got = yaw_from_xyzw(*yaw_quat(yaw))
    assert math.isclose(math.cos(got), math.cos(yaw), abs_tol=1e-12)
    assert math.isclose(math.sin(got), math.sin(yaw), abs_tol=1e-12)


def test_yaw_invariant_under_negated_quaternion():
    q = yaw_quat(1.2)
    assert math.isclose(yaw_from_xyzw(*q), yaw_from_xyzw(*(-v for v in q)))


def test_legs_up_forward_back_down():
    legs = build_legs((1.0, 2.0, 0.1), 0.0, 1.0, 1.5, 0.5, 3.0)
    assert [leg.name for leg in legs] == ['up', 'forward', 'back', 'down']
    up, fwd, back, down = legs
    assert up.start == (1.0, 2.0, 0.1)
    assert up.end == pytest.approx((1.0, 2.0, 1.1))
    assert fwd.start == up.end
    assert fwd.end == pytest.approx((2.5, 2.0, 1.1))
    assert back.start == fwd.end and back.end == up.end
    assert down.start == up.end
    assert down.end == pytest.approx((1.0, 2.0, 0.6))
    assert all(leg.hold_s == 3.0 for leg in legs)


def test_forward_follows_the_heading_not_odom_x():
    # Nose turned to ENU +y (90 deg left of odom +x).
    fwd = build_legs((0.0, 0.0, 0.0), math.pi / 2, 1.0, 2.0, 0.5, 0.0)[1]
    assert fwd.end == pytest.approx((0.0, 2.0, 1.0), abs=1e-12)
    # Nose turned 180 deg.
    fwd = build_legs((0.0, 0.0, 0.0), math.pi, 1.0, 2.0, 0.5, 0.0)[1]
    assert fwd.end == pytest.approx((-2.0, 0.0, 1.0), abs=1e-12)


def test_forward_and_back_stay_level():
    for leg in build_legs((0.0, 0.0, 0.0), 0.7, 1.2, 1.0, 0.4, 0.0)[1:3]:
        assert leg.start[2] == leg.end[2] == pytest.approx(1.2)
        assert leg_length(leg) == pytest.approx(1.0)


def test_carrot_moves_along_and_stops_at_the_end():
    leg = build_legs((0.0, 0.0, 0.0), 0.0, 1.0, 2.0, 0.5, 0.0)[1]
    assert carrot(leg, 0.0) == pytest.approx(leg.start)
    assert carrot(leg, 0.5) == pytest.approx((0.5, 0.0, 1.0))
    assert carrot(leg, 2.0) == pytest.approx(leg.end)
    assert carrot(leg, 50.0) == pytest.approx(leg.end)
    assert carrot(leg, -1.0) == pytest.approx(leg.start)


def test_carrot_on_a_zero_length_leg_is_its_end():
    legs = build_legs((0.0, 0.0, 0.0), 0.0, 1.0, 0.0, 0.5, 0.0)
    assert leg_length(legs[1]) == 0.0
    assert carrot(legs[1], 0.0) == pytest.approx(legs[1].end)


def test_descent_carrot_never_goes_below_low_height():
    down = build_legs((0.0, 0.0, 0.0), 0.0, 1.0, 1.0, 0.5, 0.0)[3]
    for progress in (0.0, 0.25, 0.5, 0.75, 10.0):
        assert carrot(down, progress)[2] >= 0.5 - 1e-12


def test_along_heading_resolves_into_forward_left_up():
    # Heading +y: moving +y is forward, moving +x is to the right.
    assert along_heading((1.0, 3.0, 0.5), (1.0, 2.0, 0.0), math.pi / 2) == \
        pytest.approx((1.0, 0.0, 0.5))
    assert along_heading((2.0, 2.0, 0.0), (1.0, 2.0, 0.0), math.pi / 2) == \
        pytest.approx((0.0, -1.0, 0.0))


def test_forward_leg_end_is_pure_forward_for_any_heading():
    for yaw in (0.0, 1.0, -2.5, math.pi):
        start = (0.4, -0.7, 0.0)
        fwd = build_legs(start, yaw, 1.0, 1.5, 0.5, 0.0)[1]
        assert along_heading(fwd.end, start, yaw) == pytest.approx((1.5, 0.0, 1.0))
