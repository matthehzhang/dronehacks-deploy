"""Return transitions, including differential comparison with the outbound-only controller.

The kinematic plant exercises message/state ordering; it is not a flight model.
"""
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from refnav import core as C, skybeta as S
from refnav.skybeta_return import SkybetaReturnMission, return_arc

ROOT = Path(__file__).resolve().parents[1]


def params():
    doc = json.loads((ROOT / 'course/course_v3.json').read_text())
    settings = {k: v for block in (doc['refnav_params'], doc['navigation']['refnav'])
                for k, v in block.items() if not k.startswith('_')}
    settings.update(track_mode='pose', height_m=.7, landed_z_m=.1, land_timeout_s=0.,
                    accept_armed_start=True, health_quiet_s=.2, arrive_m=.1,
                    first_turn_gate_s=0., blend_turn_deg=0.)
    legs = [dict(name='window_pre', hold=True),
            dict(name='window_post', hold=True, align=dict(group='window', target_z_m=1.4,
                 dist_m=.8, width_m=.9, dz_lo=-.4, dz_hi=.4))]
    return C.Params(route=[(2, 0), (2, 1)], legs=legs, **settings)


def feed(m, t):
    rotation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    assert m.on_depth(t, np.full((64, 64), np.nan, np.float32), (64, 64, 32, 32),
                      rotation, (0, 0, 0), (1, 0, 0, 0), now=t)
    m.on_health(t, True)
    if (m.p.track_mode == 'path' and m.leg == 0 and m.targets
            and getattr(m, 'accepts_outbound_paths', True)):
        assert m.on_path(t, 0, [(0, 0), (.5, 0), (1.5, 0), (2, 0)], [0., .3, .3, 0.], pid=1)


def adapter(m):
    commands, messages = [], []
    class Command(NS):
        VEHICLE_CMD_COMPONENT_ARM_DISARM = 400
        VEHICLE_CMD_DO_SET_MODE = 176
    output = S.SkybetaOutput.__new__(S.SkybetaOutput)
    output.node = NS(mission=m, _land_received_ros_s=0., _sys_us=lambda: 123456,
                     get_logger=lambda: NS(info=messages.append, warning=messages.append))
    output.VehicleCommand = Command
    output.command_pub = NS(publish=commands.append)
    output.land_xy = output.land_yaw = None
    output._send = lambda *values: values
    return output, commands, messages


def finished(m):
    m.launch_pose = (0., 0., 0., 0.)
    m.home = m.launch_pose
    m.sp, m.targets = [2., 1., -.2, 0.], [(2., 1.)]
    m.phase, m.t_phase = 'land', 0.
    m.land_reason, m.land_auto_seen = 'route complete', True
    v = C.Vehicle(pos=(2., 1., -.2), heading=.3, vel=(0, 0, 0), pos_ok=True,
                  armed=False, landed=True, nav_state=2)
    return v


@pytest.mark.parametrize('reason,result,landed,armed', [
    ('route complete', 'complete', True, False),
    ('pilot requested', 'complete', True, False),
    ('route complete', 'blocked', True, False),
    ('route complete', 'complete', False, False),
    ('route complete', 'complete', True, True),
])
def test_return_requires_successful_outbound_landing_and_disarm(reason, result, landed, armed):
    m = SkybetaReturnMission(params(), dict(height_above_floor_m=2.3))
    v = finished(m)
    m.land_reason, v.landed, v.armed = reason, landed, armed
    if result != 'complete':
        m.phase, m.result = 'done', result
    out = m.step(.1, v)
    expected = reason == 'route complete' and result == 'complete' and landed and not armed
    assert m.return_started == expected
    if expected:
        assert out.phase == 'arm'
        assert m.home == (*v.pos, v.heading)
        assert m.targets == [(0., 0.)]
        assert m.home[2] - m.p.height_m == pytest.approx(-2.2)  # 2.3 above original floor
        assert m.p.height_m == pytest.approx(2.0)  # raised pad, not a 2.3 m relative climb


@pytest.mark.parametrize('spec', [{}, None, {'height_above_floor_m': 'invalid'},
                                {'height_above_floor_m': float('nan')}])
def test_broken_optional_return_config_leaves_completed_outbound_run_alone(spec):
    m = SkybetaReturnMission(params(), spec)
    assert m.status(0)['mission_stage'] == 'outbound'
    v = finished(m)
    before = (m.p, tuple(m.sp), tuple(m.targets))
    out = m.step(.1, v)
    assert out.phase == 'done' and m.result == 'complete' and not m.return_started
    assert before == (m.p, tuple(m.sp), tuple(m.targets))
    assert out.commands == []
    assert any('RETURN SKIPPED' in e for e in out.events)


def test_return_climb_never_advances_xy_until_actual_height_settles():
    m = SkybetaReturnMission(params(), dict(height_above_floor_m=2.3))
    feed(m, .01)
    v = finished(m)
    m.step(.1, v)
    m.t_cmd = .25  # model an arm request before delivering its acknowledgement
    v.armed, v.nav_state = True, C.NAV_OFFBOARD
    feed(m, .3)
    m.step(.3, v)
    xy = tuple(m.sp[:2])
    # Commanded altitude reaches its target, but the measured pose deliberately stays low.
    for t in np.arange(.4, 12., .1):
        feed(m, float(t))
        out = m.step(float(t), v)
        assert out.phase == 'climb'
        assert out.setpoint[:2] == xy and out.velocity in (None, (0., 0.))
        assert m.leg_start is None
    v.pos = (*xy, -2.2)
    for t in (12., 12.4, 12.9):
        feed(m, t)
        assert m.step(t, v).phase == 'climb'
    # A height excursion restarts the one-second hold.
    v.pos = (*xy, -2.1)
    feed(m, 13.)
    assert m.step(13., v).phase == 'climb'
    v.pos = (*xy, -2.2)
    for t in (13.1, 13.6, 14.):
        feed(m, t)
        assert m.step(t, v).phase == 'climb'
    feed(m, 14.2)
    assert m.step(14.2, v).phase == 'turn'
    assert tuple(m.sp[:2]) == xy


@pytest.mark.parametrize('mode', ['pose', 'path'])
def test_full_outbound_is_identical_then_return_rearms_climbs_lands_and_disarms(mode):
    configuration = params()
    configuration.track_mode = mode
    configuration.arrive_m = .15  # deployed path profile; capture_m is .10 m
    baseline = C.Mission(copy.deepcopy(configuration))
    m = SkybetaReturnMission(configuration, dict(height_above_floor_m=2.3))
    normal, normal_commands, _ = adapter(baseline)
    returning, commands, messages = adapter(m)
    v = C.Vehicle(pos=(0., 0., 0.), heading=0., vel=(0, 0, 0), pos_ok=True,
                  armed=False, landed=True, nav_state=2)
    first_arm = False
    compared = 0
    pad_disarm = return_arm = return_done = False
    previous_stage = 'outbound'
    climb_xy = None
    arc_start = None
    max_offset = 0.
    events = []
    for i in range(1, 2601):
        t, dt = i / 10, .1
        if m.stage == 'outbound' and m.phase == 'arm' and not first_arm:
            v.armed, v.nav_state, first_arm = True, C.NAV_OFFBOARD, True
        for current, output in [(m, returning), (baseline, normal)]:
            feed(current, t)
            output.node._land_received_ros_s = t
            if output.land_xy is not None and v.armed is False and current.phase == 'land':
                current.land_auto_seen = True  # same acknowledgement as RefNav._tick
        expected = baseline.step(t, copy.deepcopy(v)) if previous_stage == 'outbound' else None
        out = m.step(t, v)
        events.extend(out.events)
        if expected is not None:
            if expected.phase != 'done':
                assert asdict(out) == asdict(expected)
                assert m.targets == baseline.targets
                assert m.stage == 'outbound'
                compared += 1
                expected_twist = normal.publish(t, expected, v)
                twist = returning.publish(t, out, v)
                assert twist == expected_twist
                assert [vars(c) for c in commands] == [vars(c) for c in normal_commands]
            else:
                assert expected.phase == 'done' and baseline.result == 'complete'
                assert v.armed is False and v.landed is True
                assert m.stage == 'return' and out.phase == 'arm'
                pad_disarm = True
                climb_xy = tuple(v.pos[:2])
                arc_start = np.array(v.pos[:2])
                twist = returning.publish(t, out, v)
                assert returning.land_xy is None  # first landing no longer owns the output
        else:
            twist = returning.publish(t, out, v)
        previous_stage = m.stage
        if m.stage == 'return' and out.phase in ('arm', 'climb'):
            assert tuple(out.setpoint[:2]) == climb_xy
            assert out.velocity in (None, (0., 0.))
        if m.stage == 'return' and out.phase == 'go':
            u = -arc_start/np.linalg.norm(arc_start)
            normal_axis = np.array([-u[1], u[0]])
            max_offset = max(max_offset, abs(float((np.array(v.pos[:2])-arc_start)@normal_axis)))
        for command in commands:
            if getattr(command, 'processed', False):
                continue
            command.processed = True
            if command.command == 176:
                assert pad_disarm
                v.nav_state = C.NAV_OFFBOARD
            elif command.param1 == 1:
                assert pad_disarm and v.landed is True
                v.armed, return_arm = True, True
            else:
                assert v.landed is True and command.param2 == 0
                v.armed, v.nav_state = False, 2
        # Expected command records must share the same test-only bookkeeping field.
        for command in normal_commands:
            command.processed = True
        if m.stage == 'finished':
            return_done = True
            break
        ground = -.2 if math.dist(v.pos[:2], (2., 1.)) < .6 else 0.
        if v.armed and twist is not None:
            ve, vn, vz_up, yaw_enu = twist
            p = (v.pos[0] + vn*dt, v.pos[1] + ve*dt, min(ground, v.pos[2] - vz_up*dt))
            v.vel = (vn, ve, (p[2] - v.pos[2])/dt)
            v.pos = p
            v.heading = S._wrap(v.heading - yaw_enu*dt)
            v.landed = abs(p[2] - ground) < 1e-8
        else:
            v.vel = (0., 0., 0.)
    assert compared > 200
    assert pad_disarm and return_arm and return_done, (m.stage, m.phase, v.pos, m.sp, m.leg, m.counts)
    assert m.result == 'complete' and v.armed is False and v.landed is True
    assert math.dist(v.pos[:2], (0., 0.)) < .15
    assert sum(c.command == 400 and c.param1 == 0 for c in commands) >= 2
    assert any('RETURN HEIGHT READY' in e for e in events)
    assert any('RETURN COMPLETE' in e for e in events)
    assert max_offset > .45


def test_outbound_arm_commands_never_enable_return_control():
    m = SkybetaReturnMission(params(), dict(height_above_floor_m=2.3))
    output, commands, _ = adapter(m)
    v = C.Vehicle(pos=(0, 0, 0), heading=0, armed=False, landed=True)
    output.publish(.1, C.Output('arm', commands=['offboard', 'arm']), v)
    assert commands == [] and output.land_xy is None


@pytest.mark.parametrize('arm_acknowledged', [False, True])
def test_missing_arm_or_offboard_acknowledgement_cannot_start_return_motion(arm_acknowledged):
    m = SkybetaReturnMission(params(), dict(height_above_floor_m=2.3))
    feed(m, .01)
    v = finished(m)
    m.step(.1, v)
    xy = tuple(v.pos[:2])
    saw_request = False
    for t in np.arange(.3, 30., .1):
        t = float(t)
        feed(m, t)
        out = m.step(t, v)
        saw_request |= 'arm' in out.commands
        if arm_acknowledged and 'arm' in out.commands:
            v.armed = True  # mode deliberately stays outside Offboard
        assert out.phase == 'arm'
        assert out.setpoint[:2] == xy and out.setpoint[2] == pytest.approx(-.2)
        assert out.velocity in (None, (0., 0.))
        assert m.outbound_complete
    assert saw_request


def test_return_ignores_late_outbound_path_and_correction_messages():
    m = SkybetaReturnMission(params(), dict(height_above_floor_m=2.3))
    v = finished(m)
    m.step(.1, v)
    targets = list(m.targets)
    assert m.on_path(.2, 0, [(99, 99), (100, 100)], [.3, 0]) is False
    m.on_correction(.2, 99, 99, .01)
    assert m.targets == targets and m.corr is None


def test_outbound_navigator_stops_planning_after_return_starts():
    import ast

    path = ROOT / 'coursenav/nav_node.py'
    tree = ast.parse(path.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NavNode')
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == '_tick')
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), 'exec'), namespace)
    # No ROS globals, nav objects or map access are needed once return owns control.
    assert namespace['_tick'](NS(drift=object(), status={'return_started': True})) is None


@pytest.mark.parametrize('value', [None, 'invalid', float('nan'), -1.])
def test_return_height_validation_is_deferred_until_after_outbound_completion(value):
    import fly_course

    spec = fly_course.skybeta_return_spec({'return': {'cruise_z_m': value}})
    m = SkybetaReturnMission(params(), spec)
    assert m.phase == 'wait' and m.stage == 'outbound'
    v = finished(m)
    out = m.step(.1, v)
    assert out.phase == 'done' and m.result == 'complete'
    assert m.return_started is False


@pytest.mark.parametrize('side', [-1, 1])
def test_return_arc_is_continuously_curved_with_exact_two_foot_offset(side):
    a, b = np.array([2., 1.]), np.array([0., 0.])
    points, speeds = return_arc(a, b, side=side)
    assert points[0] == pytest.approx(a) and points[-1] == pytest.approx(b)
    u = (b-a)/np.linalg.norm(b-a)
    normal = np.array([-u[1], u[0]])
    offsets = (points-a)@normal
    assert offsets[len(points)//2] == pytest.approx(side*.6096)
    assert np.max(np.abs(offsets)) == pytest.approx(.6096)
    assert speeds[0] == speeds[-1] == 0 and speeds.max() <= .3
    d = np.diff(points, axis=0)
    cross = d[:-1, 0]*d[1:, 1]-d[:-1, 1]*d[1:, 0]
    assert np.all(cross*side < 0)


def test_deployed_return_configuration_targets_two_point_three_metres():
    import fly_course
    from coursenav.course import load_course

    doc = load_course()
    assert fly_course.skybeta_return_spec(doc)['height_above_floor_m'] == 2.3
    assert doc['cruise_z_m'] == .8
