"""group119 (vendored) on skybeta: the twist conversion and planning from the venue launch mark."""
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from refnav.skybeta import KP_XY, V_XY_MAX, YAWRATE_MAX, twist_from_setpoint  # noqa: E402


def test_ned_setpoint_becomes_enu_twist():
    # 0.1 m north of the estimate, feed-forward 0.15 m/s north, 0.2 m below (NED d larger), facing north
    vx_e, vy_n, vz_up, wz = twist_from_setpoint((0.1, 0.0, 0.2, 0.0), (0.15, 0.0), None, (0.0, 0.0, 0.0), 0.0)
    assert vx_e == pytest.approx(0.0) and vy_n == pytest.approx(0.15 + KP_XY * 0.1)
    assert vz_up < 0                                     # setpoint below -> descend (ENU z up)
    assert wz == pytest.approx(0.0)


def test_yaw_rate_sign_and_clamps():
    # target yaw 30 deg clockwise (NED +) -> ENU yaw rate negative (clockwise seen from above)
    *_, wz = twist_from_setpoint((0, 0, 0, math.radians(30)), None, None, (0, 0, 0), 0.0)
    assert wz < 0 and abs(wz) <= YAWRATE_MAX + 1e-9
    vx_e, vy_n, _, _ = twist_from_setpoint((5.0, 5.0, 0, 0), None, None, (0, 0, 0), 0.0)
    assert math.hypot(vx_e, vy_n) == pytest.approx(V_XY_MAX)


def test_group119_plans_from_the_venue_launch_mark():
    import fly_course
    p, route, legs, doc = fly_course.plan_now((-2.515, 0.914), 90.0)
    assert len(route) >= 15 and p['min_clearance_m'] >= 0.41
