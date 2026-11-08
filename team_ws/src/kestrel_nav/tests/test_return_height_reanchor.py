"""Pad touchdown re-anchors PX4 Z without inheriting VIO or changing XY/yaw."""
from dataclasses import replace

import pytest

from refnav import core as C
from refnav.skybeta import twist_from_setpoint
from refnav.skybeta_return import SkybetaReturnMission
from refnav.window_vio_height import WindowVioHeight


@pytest.mark.parametrize('vio_error_m', [-1., 1., 100.])
@pytest.mark.parametrize('return_height_m', [2.1, 2.3])
def test_return_climb_ignores_wrong_vio_throughout_takeoff(vio_error_m, return_height_m):
    """Real mission/height/twist loop with a kinematic plant, not PX4 SITL.

    PX4 has a constant initial Z bias; VIO is wrong already on the pad and
    continues to drift throughout takeoff. Only the selected PX4 feedback
    should determine climb commands and the height-settle transition.
    """
    h = WindowVioHeight(lambda _: None)
    m = SkybetaReturnMission(C.Params(route=[(3, 0)], landed_z_m=.1, health_quiet_s=0.),
                             dict(height_above_floor_m=return_height_m))
    m.launch_pose = m.home = (0., 0., 0., 0.)
    raw = C.Vehicle(pos=(3., 0., -.45), heading=0., vel=(0, 0, 0),
                    landed=True, armed=True, pos_ok=True, nav_state=C.NAV_OFFBOARD)
    h.active, h.offset = True, 0.
    h.observe(1., 1., vio_error_m)
    wrong_vio = h.vehicle(1., m, raw, 1.)
    m.return_height_feedback = h.begin_return
    if hasattr(m, '_begin_touch_and_go'):
        m._begin_touch_and_go(1., wrong_vio)
    else:
        m._begin_return(1., wrong_vio, [])
        m.t_cmd = 1.  # normal rearm request already sent; arm ordering has separate coverage
    height, maximum, dt = .1, .1, .05
    for i in range(1, 601):
        t = 1 + i * dt
        raw = replace(raw, pos=(3., 0., -.45 - (height - .1)), landed=height < .12)
        h.observe(t, t, height - .1 + vio_error_m + .03 * (t - 1))
        corrected = h.vehicle(t, m, raw, t)
        assert corrected.pos[2] == pytest.approx(.1 - height)
        assert h.source == 'px4_reanchored'
        m.on_health(t, True)
        m.depth_t = m.corr_t = t
        out = m.step(t, corrected)
        assert out.commands == [] and m.land_reason is None
        command_up = twist_from_setpoint(out.setpoint, out.velocity, out.yawspeed,
                                          corrected.pos, raw.heading)[2]
        if m.phase not in ('arm', 'climb'):
            assert m.phase == 'turn', m.why
            assert height == pytest.approx(return_height_m, abs=.05)
            assert command_up <= .05
            break
        height += command_up * dt
        maximum = max(maximum, height)
    else:
        pytest.fail('return never completed its normal climb/settle with bad VIO')
    assert maximum <= return_height_m + .001


def test_return_uses_launch_height_and_frozen_px4_offset_not_vio():
    warnings = []
    h = WindowVioHeight(warnings.append)
    m = SkybetaReturnMission(C.Params(route=[(3, 0)], landed_z_m=.1),
                             dict(height_above_floor_m=2.1))
    m.launch_pose = m.home = (0., 0., 5., 0.)
    raw = C.Vehicle(pos=(3., 0., 4.55), heading=.2, vel=(0, 0, 0),
                    landed=True, armed=False, pos_ok=True)
    h.vehicle(1., m, raw, 1.)  # actual raw PX4 sample, not corrected VIO
    m.return_height_feedback = h.begin_return
    vio = replace(raw, pos=(3., 0., 4.8))
    if hasattr(m, '_begin_touch_and_go'):
        out = m._begin_touch_and_go(1., vio)
    else:
        out = m._begin_return(1., vio, [])
    corrected = h.vehicle(1., m, raw, 1.)
    assert corrected.pos == (3., 0., 5.)
    assert out.setpoint[:3] == corrected.pos
    assert twist_from_setpoint(out.setpoint, None, None, corrected.pos, raw.heading)[2] == 0
    assert m.home[2] - m.p.height_m == pytest.approx(3.)  # floor=5.1, target height=2.1
    assert h.return_offset == pytest.approx(.45)
    assert not h.active and h.source == 'px4_reanchored'
    # At target height, biased PX4 has moved by 2 m; an unrelated VIO error
    # or a terrain-relative bottom-distance change must not affect the command.
    h.observe(2., 2., 100.)
    airborne = replace(raw, pos=(3., 0., 2.55), landed=False, armed=True, rng=.3)
    corrected = h.vehicle(2., m, airborne, 2.)
    assert corrected.pos[2] == pytest.approx(3.)
    target = (*corrected.pos[:2], m.home[2] - m.p.height_m, raw.heading)
    assert twist_from_setpoint(target, None, None, corrected.pos, raw.heading)[2] == pytest.approx(0)
    assert raw.pos[2] == 4.55
    assert len(warnings) == 1 and 'RETURN HEIGHT REANCHORED' in warnings[0]


def test_handoff_only_runs_when_existing_mission_starts_return():
    h = WindowVioHeight(lambda _: None)
    m = SkybetaReturnMission(C.Params(route=[(3, 0)], legs=[{}, {'align': {'group': 'window'}}]),
                             dict(height_above_floor_m=2.1))
    m.return_height_feedback = h.begin_return
    raw = C.Vehicle(pos=(3., 0., -.45), vel=(0, 0, 0), pos_ok=True, landed=True)
    # Simply seeing landed=True while still outbound does not switch sources.
    h.vehicle(1., m, raw, 1.)
    assert h.return_offset is None
    assert h.source == 'px4'


def test_node_return_transition_publishes_in_reanchored_frame():
    import ast
    from pathlib import Path
    from types import SimpleNamespace as NS

    path = Path(__file__).resolve().parents[1] / 'refnav/node.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RefNav')
    tick = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_tick')
    bare = ast.ClassDef(name='RefNav', bases=[], keywords=[], body=[tick], decorator_list=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[bare], type_ignores=[])), str(path), 'exec'), namespace)
    node = namespace['RefNav']()
    m = SkybetaReturnMission(C.Params(route=[(3, 0)], landed_z_m=.1),
                             dict(height_above_floor_m=2.1))
    m.launch_pose = m.home = (0., 0., 0., 0.)
    m.phase, m.t_phase, m.land_reason, m.land_auto_seen = 'land', 0., 'route complete', True
    m.sp, m.targets = [3., 0., -1.3, 0.], [(3., 0.)]
    raw = C.Vehicle(pos=(3., 0., -.45), heading=0., vel=(0, 0, 0),
                    landed=True, armed=False, pos_ok=True, nav_state=C.NAV_OFFBOARD)
    if hasattr(m, 'on_pad_descent'):
        m.on_pad_descent(.1)
        m.on_landed(.2, False, .2)
        m.on_landed(.3, True, .3)
        raw.armed = True
    h = WindowVioHeight(lambda _: None)
    h.active, h.offset = True, 0.
    h.observe(1., 1., .1)  # VIO says -0.10, PX4 says -0.45; new anchor is 0
    m.return_height_feedback = h.begin_return
    node.mission, node.v, node.window_vio_height = m, raw, h
    node._now = lambda: 1.
    node.return_enabled = node.diagnostic_enabled = False
    node._lpos_received_ros_s, node._lpos_timestamp, node.done_at = 1., 123, None
    node.get_logger = lambda: NS(info=lambda _: None, warning=lambda _: None)
    node._publish_status = lambda: None
    sent = []
    node.sky = NS(land_xy=None, publish=lambda now, out, vehicle: sent.append(
        twist_from_setpoint(out.setpoint, None, None, vehicle.pos, vehicle.heading)))
    node.trace = NS(record=lambda *args, **kwargs: None)
    node._tick()
    assert m.return_started
    assert h.source == 'px4_reanchored'
    assert sent[0][2] == pytest.approx(0)
    assert m.sp[2] == pytest.approx(0)
