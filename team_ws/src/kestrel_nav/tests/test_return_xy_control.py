"""Return-only horizontal gain/cap; preserve outbound, vertical and yaw commands."""
import math
from types import SimpleNamespace as NS

import pytest

from refnav import skybeta as S
from refnav.core import Vehicle


def publish(stage, phase, sp, velocity=None, land=False):
    mission = NS(stage=stage, phase=phase, targets=[], home=None)
    node = NS(mission=mission, get_logger=lambda: NS(info=lambda _: None, warning=lambda _: None))
    adapter = S.SkybetaOutput.__new__(S.SkybetaOutput)
    adapter.node, adapter.land_xy = node, None
    adapter._send = lambda *values: values
    v = Vehicle(pos=(0., 0., -1.), heading=0., armed=True, landed=False, vel=(0., 0., 0.))
    out = NS(commands=['land'] if land else [], stream=True, setpoint=sp,
             velocity=velocity, yawspeed=.1)
    if land:
        adapter._plan_landing(0., v)
        adapter.land_xy = sp[:2]
        adapter.return_control_active = stage == 'return'
        adapter.touch_and_go_active = stage == 'return'
    return adapter.publish(.1, out, v)


@pytest.mark.parametrize('phase', ['climb', 'turn', 'check', 'go'])
def test_only_return_halves_position_correction(phase):
    sp = (.2, .1, -1.2, .2)
    outbound = publish('outbound', phase, sp)
    returning = publish('return', phase, sp)
    assert outbound[:2] == pytest.approx((.1, .2))
    assert returning[:2] == pytest.approx((.05, .1))
    assert returning[2:] == outbound[2:]


def test_return_cap_applies_to_total_horizontal_vector_including_feedforward():
    sp = (3., 4., -1.2, .2)
    outbound = publish('outbound', 'go', sp, velocity=(.2, .1))
    returning = publish('return', 'go', sp, velocity=(.2, .1))
    assert math.hypot(*outbound[:2]) == pytest.approx(.6)
    assert math.hypot(*returning[:2]) == pytest.approx(.3)
    assert returning[2:] == outbound[2:]


def test_return_preserves_planned_feedforward_at_zero_position_error():
    assert publish('return', 'go', (0., 0., -1., 0.), velocity=(.3, 0.))[:2] == pytest.approx((0., .3))


def test_pad_landing_keeps_outbound_controller_and_final_landing_uses_return_limits():
    sp = (.2, .1, -1., 0.)
    outbound = publish('outbound', 'land', sp, land=True)
    returning = publish('return', 'land', sp, land=True)
    assert outbound[:2] == pytest.approx((.1, .2))
    assert returning[:2] == pytest.approx((.05, .1))
    assert outbound[2] == returning[2] == -S.LAND_VZ


def test_default_conversion_keeps_existing_outbound_gain_and_cap():
    result = S.twist_from_setpoint((.2, .1, -1., 0.), None, None, (0., 0., -1.), 0.)
    assert result[:2] == pytest.approx((.1, .2))
    assert S.KP_XY == 1. and S.V_XY_MAX == .6
