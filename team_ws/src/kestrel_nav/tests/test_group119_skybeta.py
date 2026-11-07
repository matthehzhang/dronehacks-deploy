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


def test_landing_descent_triggers_px4_land_detector():
    from refnav.skybeta import LAND_VZ
    assert LAND_VZ >= 1.1 * 0.25          # MulticopterLandDetector in_descend, fleet LNDMC_Z_VEL_MAX 0.25


def test_pad_landing_centres_footprint_and_faces_a_pad_side():
    import math
    from types import SimpleNamespace as NS
    from refnav import skybeta as sk
    sent = []
    node = NS(mission=NS(targets=[(1.0, 1.0), (5.0, -2.0)], home=(0.0, 0.0, 0.0, math.radians(3.0))),
              get_logger=lambda: NS(info=lambda s: None))
    out = sk.SkybetaOutput.__new__(sk.SkybetaOutput)
    out.node, out.land_xy, out.land_yaw = node, None, None
    out._send = lambda *a: sent.append(a) or a
    v = NS(pos=(5.06, -2.03, -0.7), heading=math.radians(131.0), armed=True, vel=(0.0, 0.0, 0.0))
    out.publish(0.0, NS(commands=['land'], stream=False, setpoint=None), v)
    # nearest pad side to 131 deg relative to home 3 deg is 3 + 90*1 = 93 deg... or 183: |131-93|=38 < |131-183|=52
    assert math.degrees(out.land_yaw) == pytest.approx(93.0)
    c, s = math.cos(out.land_yaw), math.sin(out.land_yaw)
    assert out.land_xy == pytest.approx((5.0 + 0.035 * c, -2.0 + 0.035 * s))
    assert sent[-1][2] == 0.0                       # not descending until centred and square


def test_path_leg_turn_without_a_path_keeps_heading():
    """Gazebo e313281: the first maze path arrived after the leg-1 turn, which faced the stop bearing (163 deg NED)
    into a divider; 7733cd0's blocking wait deadlocked. path_turn_hold: no turn toward the stop, go waits."""
    from refnav import core as C
    p = C.Params(route=[(5.0, -1.0)], track_mode="path", path_turn_hold=True, legs=[{"mode": "cruise"}])
    m = C.Mission(p)
    m.phase, m.t_phase, m.leg, m.sp = "turn", 0.0, 0, [0.0, 0.0, -0.7, 0.0]
    m.targets, m.leg_heading = [(-4.8, 1.5)], 2.84
    v = C.Vehicle(pos=(0.0, 0.0, -0.7), heading=0.0, pos_ok=True, armed=True, nav_state=C.NAV_OFFBOARD, landed=False)
    m.t_last = 0.0
    m.health.bad_for = lambda now: 0.0                        # supervision inputs healthy (not under test)
    m.depth_t, m._depth_silent_s = 1.0, (lambda now: 0.0)
    m.step(1.0, v)
    assert abs(m.sp[3]) < 1e-9 and m.leg_heading == 0.0       # heading kept, no swing toward the stop
    assert m.phase in ("turn", "check")
