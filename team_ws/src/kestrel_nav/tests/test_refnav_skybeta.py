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


def test_group154_plans_new_course_from_the_declared_launch_mark():
    import fly_course
    from coursenav.course import load_course
    start = load_course()['start']
    p, route, legs, doc = fly_course.plan_now(start['xy'], start['yaw_deg'])
    assert len(route) == len(legs) and route
    assert p['min_clearance_m'] >= doc['vehicle']['radius_m'] + doc['vehicle']['margin_m']
    assert [s['name'] for s in doc['sequence']] == ['tunnel_entry', 'tunnel_exit', 'tunnel_clearance', 'chicane', 'pad']


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


def test_landing_descent_continues_past_sixty_seconds_until_disarmed():
    from types import SimpleNamespace as NS
    from refnav import skybeta as sk, core as C

    sent, warnings = [], []
    m = C.Mission(C.Params(land_timeout_s=0.0))
    m.phase, m.t_phase, m.land_reason = 'land', 0.0, 'route complete'
    m.sp, m.home = [0, 0, -1.3, 0], (0, 0, 0, 0)
    adapter = sk.SkybetaOutput.__new__(sk.SkybetaOutput)
    adapter.node = NS(mission=m, get_logger=lambda: NS(info=lambda _: None, warning=warnings.append))
    adapter.land_xy, adapter.land_yaw = None, None
    adapter._send = lambda *args: sent.append(args) or args
    v = C.Vehicle(pos=(0, 0, -1.3), heading=0, armed=True, nav_state=C.NAV_OFFBOARD,
                  pos_ok=True, vel=(0, 0, 0), landed=False)
    for t in (0.0, 10.0, 61.0, 90.0):
        out = m.step(t, v)
        assert m.phase == 'land' and m.result is None
        adapter.publish(t, out, v)
    assert sent[-1][2] == pytest.approx(-sk.LAND_VZ)
    assert len(warnings) == 1 and 'DISARM NOT CONFIRMED' in warnings[0]
    # Touchdown alone must not stop the descent command while PX4 is still armed.
    v.landed = True
    out = m.step(91.0, v)
    adapter.publish(91.0, out, v)
    assert m.phase == 'land' and sent[-1][2] == pytest.approx(-sk.LAND_VZ)
    # Same disarm acknowledgement used in RefNav._tick for the skybeta adapter.
    v.armed = False
    m.land_auto_seen = True
    out = m.step(92.0, v)
    assert m.phase == 'done' and m.result == 'complete'
    assert any('PX4 reports DISARMED' in event for event in out.events)
    count = len(sent)
    assert adapter.publish(92.0, out, v) is None
    assert len(sent) == count


def test_drone_launch_waits_for_disarm_instead_of_using_the_landing_timeout():
    text = (ROOT / 'launch/refnav_skybeta.launch.py').read_text()
    assert "'land_timeout_s:=0.0'" in text


def disarming_adapter():
    from types import SimpleNamespace as NS
    from refnav import skybeta as sk

    class Command(NS):
        VEHICLE_CMD_COMPONENT_ARM_DISARM = 400

    sent, warnings = [], []
    node = NS(mission=NS(targets=[], home=(0, 0, 0, 0)), px4_system_id=20,
              _land_received_ros_s=None, _sys_us=lambda: 123456789,
              get_logger=lambda: NS(info=lambda _: None, warning=warnings.append))
    adapter = sk.SkybetaOutput.__new__(sk.SkybetaOutput)
    adapter.node, adapter.VehicleCommand = node, Command
    adapter.land_xy, adapter.land_yaw, adapter.disarm_sent_t = None, None, None
    adapter.command_pub = NS(publish=sent.append)
    adapter._send = lambda *args: args
    v = NS(pos=(0, 0, -.01), heading=0, vel=(0, 0, 0), armed=True, landed=False)
    return adapter, node, v, sent, warnings


def test_explicit_normal_disarm_after_touchdown_retries_until_px4_disarmed():
    from types import SimpleNamespace as NS

    adapter, node, v, sent, warnings = disarming_adapter()
    out = NS(commands=['land'], stream=False, setpoint=None)
    adapter.publish(0., out, v)
    assert sent == []  # low estimated height does not disarm an airborne vehicle
    v.landed, node._land_received_ros_s = True, 1.
    adapter.publish(1., out, v)
    assert len(sent) == 1
    command = sent[0]
    assert command.command == 400 and command.param1 == 0 and command.param2 == 0
    assert command.timestamp == 123456789
    assert command.target_system == command.source_system == 20
    assert command.target_component == command.source_component == 1
    assert command.from_external is True
    assert 'DISARM COMMAND SENT' in warnings[-1]
    adapter.publish(1.5, out, v)
    assert len(sent) == 1
    node._land_received_ros_s = 2.
    adapter.publish(2., out, v)
    assert len(sent) == 2
    v.armed = False
    assert adapter.publish(2.1, out, v) is None
    assert len(sent) == 2


@pytest.mark.parametrize('landed, age', [(False, 0.), (None, 0.), (True, 1.1), (True, -.1)])
def test_disarm_not_sent_for_missing_false_stale_or_future_touchdown(landed, age):
    from types import SimpleNamespace as NS

    adapter, node, v, sent, _ = disarming_adapter()
    out = NS(commands=['land'], stream=False, setpoint=None)
    adapter.publish(0., out, v)
    v.landed, node._land_received_ros_s = landed, 2. - age
    adapter.publish(2., out, v)
    assert sent == []


def test_no_disarm_before_landing_or_after_bounce():
    from types import SimpleNamespace as NS

    adapter, node, v, sent, _ = disarming_adapter()
    v.landed, node._land_received_ros_s = True, 0.
    adapter.publish(0., NS(commands=[], stream=False, setpoint=None), v)
    assert sent == []
    adapter.publish(.1, NS(commands=['land'], stream=False, setpoint=None), v)
    assert len(sent) == 1
    v.landed, node._land_received_ros_s = False, 2.
    adapter.publish(2., NS(commands=[], stream=False, setpoint=None), v)
    assert len(sent) == 1


def test_disarm_publisher_uses_resolved_px4_command_topic(monkeypatch):
    import types
    from types import SimpleNamespace as NS
    from refnav import skybeta as sk

    geometry = types.ModuleType('geometry_msgs.msg')
    geometry.TwistStamped = NS
    px4 = types.ModuleType('px4_msgs.msg')
    px4.VehicleCommand = type('VehicleCommand', (), {'VEHICLE_CMD_COMPONENT_ARM_DISARM': 400})
    topics = types.ModuleType('skybeta_flight_interface.topics')
    topics.resolve_endpoints = lambda: {'vehicle_command': '/fmu/in/vehicle_command_v7'}
    for package, child in [('geometry_msgs', geometry), ('px4_msgs', px4), ('skybeta_flight_interface', topics)]:
        monkeypatch.setitem(sys.modules, package, types.ModuleType(package))
        monkeypatch.setitem(sys.modules, child.__name__, child)
    bindings = []
    node = NS(create_publisher=lambda cls, topic, qos: bindings.append((cls, topic, qos)) or NS(publish=lambda _: None))
    adapter = sk.SkybetaOutput(node)
    assert adapter.VehicleCommand is px4.VehicleCommand
    assert (px4.VehicleCommand, '/fmu/in/vehicle_command_v7', 10) in bindings
