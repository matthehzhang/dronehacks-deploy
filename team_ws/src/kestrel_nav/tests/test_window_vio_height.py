"""Window-only feedback handoff, including the recorded phantom-climb failure."""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from refnav.core import Vehicle
from refnav.skybeta import twist_from_setpoint
from refnav.window_vio_height import WindowVioHeight


def setup():
    warnings = []
    h = WindowVioHeight(warnings.append)
    m = NS(p=NS(legs=[{}, {'align': {'group': 'window'}}, {'name': 'pad'}]), leg=0)
    v = Vehicle(pos=(1, 2, -.7), vel=(0, 0, 0), pos_ok=True, landed=False, heading=0)
    # VIO up=2.0 corresponds to PX4 down=-0.7 (independent origins).
    for i in range(21):
        t = 1 + i * .1
        h.observe(t, t, 2.0)
        assert h.vehicle(t, m, v, t) is v
    return h, m, v, warnings


def test_switch_preserves_target_twist_and_xy_then_rejects_phantom_climb():
    h, m, v, warnings = setup()
    sp = (1.1, 2.1, -1.3, .1)
    m.leg = 1
    corrected = h.vehicle(3, m, v, 3)
    assert corrected.pos == pytest.approx(v.pos)
    assert twist_from_setpoint(sp, None, None, corrected.pos, v.heading) == pytest.approx(
        twist_from_setpoint(sp, None, None, v.pos, v.heading))
    # Actual/VIO height is on target; PX4 reports a phantom 0.25 m climb.
    h.observe(3.1, 3.1, 2.6)
    raw = replace(v, pos=(1, 2, -1.55))
    corrected = h.vehicle(3.1, m, raw, 3.1)
    assert corrected.pos == pytest.approx((1, 2, -1.3))
    assert raw.pos[2] == -1.55  # raw PX4 report is retained for evidence
    assert twist_from_setpoint(sp, None, None, corrected.pos, v.heading)[2] == pytest.approx(0)
    assert twist_from_setpoint(sp, None, None, raw.pos, v.heading)[2] < -.2
    assert sum('HEIGHT ACTIVE' in w for w in warnings) == 1


def test_retains_frozen_calibration_through_pad_even_when_px4_is_wrong():
    h, m, v, _ = setup()
    offset = h.offset
    m.leg = 1
    h.vehicle(3, m, v, 3)
    m.leg = 2
    h.observe(3.1, 3.1, 2.6)
    corrected = h.vehicle(3.1, m, replace(v, pos=(1, 2, -1.8)), 3.1)
    assert corrected.pos[2] == pytest.approx(-1.3)
    assert h.offset == offset


def test_no_vio_or_no_calibration_does_not_block_mission():
    h, m, v, _ = setup()
    warnings = []
    h = WindowVioHeight(warnings.append)
    m.leg = 1
    for t in (3, 4, 5):
        assert h.vehicle(t, m, v, t) is v
    assert len(warnings) == 1 and 'UNAVAILABLE' in warnings[0]
    assert not h.active


def test_dropout_bridges_px4_changes_without_absolute_height_jump_and_resumes():
    h, m, v, warnings = setup()
    m.leg = 1
    h.vehicle(3, m, v, 3)
    raw = replace(v, pos=(1, 2, -1.55))
    h.observe(3.1, 3.1, 2.6)
    h.vehicle(3.1, m, raw, 3.1)
    # A 0.43 s gap (seen on hardware) is tolerated without a fallback.
    assert h.vehicle(3.53, m, raw, 3.53).pos[2] == pytest.approx(-1.3)
    h.observe(4.2, 3.1, 2.6)  # frozen source stamp cannot refresh freshness
    corrected = h.vehicle(4.2, m, replace(raw, pos=(1, 2, -1.57)), 4.2)
    assert corrected.pos[2] == pytest.approx(-1.32)
    assert h.source == 'px4_delta'
    h.observe(4.3, 4.3, 2.6)
    assert h.vehicle(4.3, m, raw, 4.3).pos[2] == pytest.approx(-1.3)
    assert any('STALE' in w for w in warnings) and any('RESUMED' in w for w in warnings)


def test_calibration_averages_only_recent_steady_samples_and_ignores_nan():
    h, m, v, _ = setup()
    before = h.offset
    h.observe(3.1, 3.1, float('nan'))
    assert h.vio[1] == 3
    h.observe(4, 4, 5)
    h.vehicle(4, m, replace(v, vel=(.5, 0, 0)), 4)
    assert h.offset == before
    h.observe(8, 8, 2.1)
    h.vehicle(8, m, v, 8)
    assert h.offset == pytest.approx(1.4)
    assert len(h.samples) == 1


def test_routes_without_window_keep_px4_feedback():
    h, m, v, _ = setup()
    m.p.legs = [{}, {}]
    assert h.vehicle(3, m, v, 3) is v
    assert not h.active


def test_real_route_selects_crossing_not_tunnel_or_clearance_stop():
    import fly_course
    from coursenav.course import load_course
    start = load_course()['start']
    _, _, legs, _ = fly_course.plan_now(start['xy'], start['yaw_deg'])
    windows = [i for i, leg in enumerate(legs) if leg.get('align', {}).get('group') == 'window']
    assert len(windows) == 1
    crossing = windows[0]
    assert legs[crossing]['name'] == 'window_post'
    assert legs[crossing - 1]['name'] == 'window_pre'


def test_node_passes_same_vio_height_to_mission_and_twist_output():
    """Exercise the production ROS callback/tick methods without a DDS runtime."""
    import ast
    from pathlib import Path
    from refnav import core as C

    path = Path(__file__).resolve().parents[1] / 'refnav/node.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RefNav')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in ('_tick', '_on_window_vio')]
    bare = ast.ClassDef(name='RefNav', bases=[], keywords=[], body=methods, decorator_list=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[bare], type_ignores=[])), str(path), 'exec'), namespace)
    node = namespace['RefNav']()
    h, m, v, _ = setup()
    m.leg = 1
    h.vehicle(3, m, v, 3)
    node.window_vio_height = h
    node._now = lambda: 3.1
    node._on_window_vio(NS(header=NS(stamp=NS(sec=3, nanosec=100000000)),
                           pose=NS(pose=NS(position=NS(z=2.6)))))
    node.v = replace(v, pos=(1, 2, -1.55))
    received = []
    def step(now, vehicle):
        received.append(vehicle)
        return C.Output('go', stream=True, setpoint=(1.1, 2, -1.3, 0))
    m.step = step
    m.phase, m.path, m.path_s, m.path_stop_done = 'go', None, 0, False
    node.mission = m
    node.return_enabled = node.diagnostic_enabled = False
    node._lpos_received_ros_s, node._lpos_timestamp = 3.1, 123
    node.done_at = None
    published = []
    node.sky = NS(land_xy=None, publish=lambda now, out, vehicle: published.append(
        twist_from_setpoint(out.setpoint, None, None, vehicle.pos, vehicle.heading)))
    trace = []
    node.trace = NS(record=lambda *args, **kwargs: trace.append(kwargs))
    node._tick()
    assert received[0].pos[2] == pytest.approx(-1.3)
    assert published[0][2] == pytest.approx(0)
    assert published[0][1] == pytest.approx(.1)  # horizontal progress continues
    assert node.v.pos[2] == -1.55
    assert trace[0]['window_vio_height']['source'] == 'vio'
