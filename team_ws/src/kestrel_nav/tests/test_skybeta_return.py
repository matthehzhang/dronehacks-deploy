"""Touch-and-go state transitions and curved return; no arm or mode commands.

The kinematic plant verifies state/message ordering, not real touchdown or PX4's
takeoff-ramp/automatic-disarm timing on the aircraft.
"""
import ast
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


def params(mode='pose'):
    doc = json.loads((ROOT / 'course/course_v3.json').read_text())
    settings = {k: v for block in (doc['refnav_params'], doc['navigation']['refnav'])
                for k, v in block.items() if not k.startswith('_')}
    settings.update(track_mode=mode, height_m=.7, landed_z_m=.1, land_timeout_s=0.,
                    accept_armed_start=True, health_quiet_s=.2, arrive_m=.15,
                    first_turn_gate_s=0., blend_turn_deg=0.)
    legs = [dict(name='window_pre', hold=True),
            dict(name='window_post', hold=True, align=dict(group='window', target_z_m=1.4,
                 dist_m=.8, width_m=.9, dz_lo=-.4, dz_hi=.4))]
    return C.Params(route=[(2, 0), (2, 1)], legs=legs, **settings)


def feed(m, v, t):
    rotation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    assert m.on_depth(t, np.full((64, 64), np.nan, np.float32), (64, 64, 32, 32),
                      rotation, (0, 0, 0), (1, 0, 0, 0), now=t)
    m.on_health(t, True)
    if isinstance(m, SkybetaReturnMission):
        m.on_landed(t, v.landed, t)
    if (m.p.track_mode == 'path' and m.leg == 0 and m.targets
            and getattr(m, 'accepts_outbound_paths', True)):
        assert m.on_path(t, 0, [(0, 0), (.5, 0), (1.5, 0), (2, 0)], [0., .3, .3, 0.], pid=1)


def adapter(m):
    commands = []
    class Command(NS):
        VEHICLE_CMD_COMPONENT_ARM_DISARM = 400  # deliberately no arm/mode helper constant
    out = S.SkybetaOutput.__new__(S.SkybetaOutput)
    out.node = NS(mission=m, _land_received_ros_s=0., _sys_us=lambda: 123456,
                  get_logger=lambda: NS(info=lambda _: None, warning=lambda _: None))
    out.VehicleCommand = Command
    out.command_pub = NS(publish=commands.append)
    out.land_xy = out.land_yaw = None
    out._send = lambda *values: values
    return out, commands


def pad_descent(spec=None):
    m = SkybetaReturnMission(params(), spec if spec is not None else dict(height_above_floor_m=2.3))
    m.launch_pose = m.home = (0., 0., 0., 0.)
    m.sp, m.targets = [2., 1., -.2, 0.], [(2., 1.)]
    m.phase, m.t_phase, m.land_reason = 'land', 0., 'route complete'
    v = C.Vehicle(pos=(2., 1., -.2), heading=0., vel=(0, 0, 0), pos_ok=True,
                  armed=True, landed=False, nav_state=C.NAV_OFFBOARD)
    feed(m, v, .1)
    m.on_pad_descent(.2)
    return m, v


def contact(m, v, now=.4):
    v.landed = True
    feed(m, v, now)
    return m.step(now, v)


@pytest.mark.parametrize('side', [-1, 1])
def test_circular_arc_has_exact_endpoints_and_two_foot_midpoint_offset(side):
    a, b = np.array([2., 1.]), np.array([0., 0.])
    points, speeds = return_arc(a, b, side=side)
    assert points[0] == pytest.approx(a) and points[-1] == pytest.approx(b)
    u = (b-a)/np.linalg.norm(b-a)
    normal = np.array([-u[1], u[0]])
    offsets = (points-a)@normal
    assert offsets[len(points)//2] == pytest.approx(side*.6096)
    assert np.max(np.abs(offsets)) == pytest.approx(.6096)
    assert np.max(np.linalg.norm(np.diff(points, axis=0), axis=1)) <= .025
    assert speeds[0] == speeds[-1] == 0 and speeds.max() <= .3
    d = np.diff(points, axis=0)
    cross = d[:-1, 0]*d[1:, 1]-d[:-1, 1]*d[1:, 0]
    assert np.all(cross*side < 0)  # continuously curved, not a two-segment detour


def test_low_height_or_elapsed_descent_is_not_a_touch():
    m, v = pad_descent()
    for t in np.arange(.3, 15., .1):
        feed(m, v, float(t))
        assert m.step(float(t), v).phase == 'land'
    assert not m.return_started and not m.pad_touched


@pytest.mark.parametrize('case', ['before_descent', 'duplicate', 'stale', 'false_again'])
def test_old_or_unconfirmed_ground_reports_cannot_start_touch_and_go(case):
    m, v = pad_descent()
    if case == 'before_descent':
        m.pad_descent_t = None
        m.on_landed(.3, True, .3)
        m.on_pad_descent(.4)
        m.on_landed(.5, True, .5)
        now = .5
    elif case == 'duplicate':
        m.on_landed(.3, True, .1)  # same capture as earlier false report
        now = .3
    elif case == 'stale':
        m.on_landed(.3, True, .3)
        now = 2.
    else:
        m.on_landed(.3, True, .3)
        m.on_landed(.4, False, .4)
        now = .4
    v.landed = True
    m.step(now, v)
    assert not m.return_started


def test_confirmed_touch_starts_climb_while_armed_without_an_arm_phase():
    m, v = pad_descent()
    out = contact(m, v)
    assert m.return_started and m.pad_touched and v.armed is True
    assert m.phase == out.phase == 'climb' and out.commands == []
    assert m.home[2]-m.p.height_m == pytest.approx(-2.2)
    assert m.p.height_m == pytest.approx(2.0)  # raised pad, absolute 2.3 m floor reference
    assert m.targets == [(0., 0.)]


def test_no_horizontal_advance_until_actual_climb_height_settles():
    m, v = pad_descent()
    contact(m, v)
    xy = tuple(v.pos[:2])
    for t in np.arange(.5, 12., .1):
        feed(m, v, float(t))
        out = m.step(float(t), v)
        assert out.phase == 'climb'
        assert out.setpoint[:2] == xy and out.velocity in (None, (0., 0.))
        assert m.leg_start is None
    v.pos = (*xy, -2.2)
    for t in (12., 12.5, 12.9):
        feed(m, v, t)
        assert m.step(t, v).phase == 'climb'
    v.pos = (*xy, -2.1)
    feed(m, v, 13.)
    assert m.step(13., v).phase == 'climb'
    v.pos = (*xy, -2.2)
    for t in (13.1, 13.6, 14.):
        feed(m, v, t)
        assert m.step(t, v).phase == 'climb'
    feed(m, v, 14.2)
    assert m.step(14.2, v).phase == 'turn'
    assert m.sp[:2] == list(xy)


@pytest.mark.parametrize('when', ['on_pad', 'during_climb'])
def test_auto_disarm_stops_return_and_never_rearms(when):
    m, v = pad_descent()
    if when == 'during_climb':
        contact(m, v)
    v.armed, v.landed = False, True
    if when == 'on_pad':
        m.land_auto_seen = True
    out = m.step(.5, v)
    assert out.phase == 'done' and m.stage == 'finished'
    output, commands = adapter(m)
    output.publish(.5, out, v)
    assert commands == [] and not out.stream


@pytest.mark.parametrize('spec', [{}, {'height_above_floor_m': 'invalid'},
                                {'height_above_floor_m': float('nan')}])
def test_invalid_optional_return_finishes_normal_pad_landing_and_disarm(spec):
    m, v = pad_descent(spec)
    out = contact(m, v)
    assert not m.return_started and m.stage == 'finish_pad'
    assert any('RETURN SKIPPED' in e for e in out.events)
    output, commands = adapter(m)
    output.land_xy, output.land_yaw, output.land_t0, output.descending = tuple(v.pos[:2]), 0., 0., True
    output.node._land_received_ros_s = .4
    output.publish(.4, out, v)
    assert len(commands) == 1 and commands[0].param1 == commands[0].param2 == 0


def test_abort_landing_does_not_start_a_touch_and_go():
    m, v = pad_descent()
    m.land_reason = 'pilot requested'
    contact(m, v)
    assert not m.return_started and not m.touch_and_go_pending


def test_abort_queued_at_touchdown_cannot_be_overwritten_by_the_handoff():
    m, v = pad_descent()
    v.landed = True
    feed(m, v, .4)
    m.request_land('operator abort at touchdown')
    out = m.step(.4, v)
    assert not m.return_started and not m.touch_and_go_pending
    assert m.land_request == 'operator abort at touchdown'
    output, commands = adapter(m)
    output.land_xy, output.land_yaw, output.land_t0, output.descending = tuple(v.pos[:2]), 0., 0., True
    output.node._land_received_ros_s = .4
    output.publish(.4, out, v)
    assert len(commands) == 1 and commands[0].param1 == 0


@pytest.mark.parametrize('mode', ['pose', 'path'])
def test_outbound_matches_until_contact_then_curved_return_and_only_final_disarm(mode):
    baseline = C.Mission(params(mode))
    m = SkybetaReturnMission(params(mode), dict(height_above_floor_m=2.3))
    normal, _ = adapter(baseline)
    output, commands = adapter(m)
    v = C.Vehicle(pos=(0., 0., 0.), heading=0., vel=(0, 0, 0), pos_ok=True,
                  armed=False, landed=True, nav_state=2)
    first_arm = False
    previous_stage = 'outbound'
    compared = 0
    curve_start = None
    largest_offset = 0.
    events = []
    for i in range(1, 3001):
        t, dt = i/10, .1
        if m.stage == 'outbound' and m.phase == 'arm' and not first_arm:
            v.armed, v.nav_state, first_arm = True, C.NAV_OFFBOARD, True  # pilot's only arming
        for current, published in ((m, output), (baseline, normal)):
            feed(current, v, t)
            published.node._land_received_ros_s = t
            if published.land_xy is not None and v.armed is False and current.phase == 'land':
                current.land_auto_seen = True
        expected = baseline.step(t, copy.deepcopy(v)) if previous_stage == 'outbound' else None
        out = m.step(t, v)
        events.extend(out.events)
        if expected is not None and not m.return_started:
            assert asdict(out) == asdict(expected)
            assert m.targets == baseline.targets
            twist = output.publish(t, out, v)
            assert twist == normal.publish(t, expected, v)
            assert commands == []
            compared += 1
        else:
            if previous_stage == 'outbound':
                assert v.landed is True and v.armed is True and out.phase == 'climb'
                assert expected.phase == 'land'
                curve_start = np.array(v.pos[:2])
            twist = output.publish(t, out, v)
        previous_stage = m.stage
        if m.stage == 'return' and out.phase == 'climb':
            assert out.setpoint[:2] == pytest.approx(curve_start)
            assert out.velocity in (None, (0., 0.))
        if m.stage == 'return' and m.phase == 'go':
            chord = -curve_start/np.linalg.norm(curve_start)
            normal_axis = np.array([-chord[1], chord[0]])
            largest_offset = max(largest_offset, abs(float((np.array(v.pos[:2])-curve_start)@normal_axis)))
        for command in commands:
            if getattr(command, 'processed', False):
                continue
            assert m.phase == 'land' and m.return_started
            assert command.command == 400 and command.param1 == command.param2 == 0
            assert v.landed is True
            command.processed = True
            v.armed, v.nav_state = False, 2
        if m.stage == 'finished':
            break
        ground = -.2 if math.dist(v.pos[:2], (2., 1.)) < .6 else 0.
        if v.armed and twist is not None:
            ve, vn, vz_up, yaw_enu = twist
            p = (v.pos[0]+vn*dt, v.pos[1]+ve*dt, min(ground, v.pos[2]-vz_up*dt))
            v.vel = (vn, ve, (p[2]-v.pos[2])/dt)
            v.pos, v.heading = p, S._wrap(v.heading-yaw_enu*dt)
            v.landed = abs(p[2]-ground) < 1e-8
        else:
            v.vel = (0., 0., 0.)
    assert compared > 200 and m.return_started and m.pad_touched
    assert m.stage == 'finished' and m.result == 'complete'
    assert v.armed is False and v.landed is True and math.dist(v.pos[:2], (0., 0.)) < .15
    assert len(commands) == 1 and largest_offset > .45
    assert any('PAD TOUCH CONFIRMED' in e for e in events)
    assert any('RETURN COMPLETE' in e for e in events)


def test_late_outbound_messages_cannot_replace_return_curve():
    m, v = pad_descent()
    contact(m, v)
    points = m.path['pts'].copy()
    assert m.on_path(.5, 0, [(99, 99), (100, 100)], [.3, 0]) is False
    m.on_correction(.5, 99, 99, .01)
    np.testing.assert_array_equal(m.path['pts'], points)
    assert m.corr is None


def test_adapter_contains_no_return_arm_or_mode_command_path():
    tree = ast.parse((ROOT / 'refnav/skybeta.py').read_text())
    constants = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
    assert 'VEHICLE_CMD_DO_SET_MODE' not in constants
    assert not any(n.name == '_vehicle_command' for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))


def test_outbound_navigator_stops_after_touch_and_go():
    path = ROOT / 'coursenav/nav_node.py'
    tree = ast.parse(path.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NavNode')
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == '_tick')
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), 'exec'), namespace)
    assert namespace['_tick'](NS(drift=object(), status={'return_started': True})) is None


def test_deployed_return_configuration_targets_two_point_three_metres():
    import fly_course
    from coursenav.course import load_course

    doc = load_course()
    assert fly_course.skybeta_return_spec(doc)['height_above_floor_m'] == 2.3
    assert doc['cruise_z_m'] == .8
