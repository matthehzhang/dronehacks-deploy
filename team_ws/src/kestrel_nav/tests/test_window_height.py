"""Depth height estimates and the measured-height interlock before a raised window crossing."""
from types import SimpleNamespace

import numpy as np
import pytest

from refnav import core as C, gate as G


META = dict(group="window", dist_m=0.8, width_m=0.9, dz_lo=-0.45, dz_hi=0.45)


def window_points(dz, edges=("sill", "header"), floor=False):
    patches = []
    for side in (-1, 1):
        r, d = np.meshgrid(side * np.linspace(0.45, 0.57, 9), np.linspace(-0.2, 0.2, 9))
        patches.append(np.column_stack((np.full(r.size, 0.8), r.ravel(), d.ravel())))
    for edge in edges:
        down = np.linspace(0.45, 0.55, 9) if edge == "sill" else np.linspace(-0.55, -0.45, 9)
        r, d = np.meshgrid(np.linspace(-0.3, 0.3, 11), down - dz)
        patches.append(np.column_stack((np.full(r.size, 0.8), r.ravel(), d.ravel())))
    if floor:
        f, r = np.meshgrid(np.linspace(0.73, 0.87, 11), np.linspace(-0.3, 0.3, 11))
        patches.append(np.column_stack((f.ravel(), r.ravel(), np.full(r.size, 0.7))))
    return np.vstack(patches)


@pytest.mark.parametrize("dz", [-0.25, 0.0, 0.25])
@pytest.mark.parametrize("edges", [("sill",), ("header",), ("sill", "header")])
def test_depth_centres_window_with_one_or_both_edges(dz, edges):
    est = G.find_opening(window_points(dz, edges), META)
    assert est is not None
    assert est["dz_m"] == pytest.approx(dz, abs=0.02)


@pytest.mark.parametrize("floor", [False, True])
def test_missing_edges_and_floor_do_not_invent_height(floor):
    est = G.find_opening(window_points(0, (), floor=floor), META)
    assert est is not None and est["dz_m"] is None


def test_conflicting_window_height_is_rejected():
    points = window_points(0)
    points[points[:, 2] < -0.3, 2] -= 0.25
    assert G.find_opening(points, META)["dz_m"] is None


def test_height_needs_multiple_consistent_frames():
    base = dict(lateral_m=0, heading_rad=0, width_m=0.9, range_m=0.8)
    for heights in [(0.25, None, None), (0.25, -0.25, 0.25)]:
        assert G.combine([dict(base, dz_m=d) for d in heights])["dz_m"] is None
    assert G.combine([dict(base, dz_m=d) for d in (0.24, 0.26, None)])["dz_m"] == pytest.approx(0.25)


def mission(group="window"):
    meta = dict(META, group=group)
    m = C.Mission(C.Params(route=[(1, 0)], legs=[dict(align=meta)]))
    m.leg = 0
    m.phase = "align"
    m.t_phase = 0
    m.sp = [0, 0, -0.8, 0]
    m.targets = [(1, 0)]
    m.leg_start = (0, 0)
    m.leg_dir = (1, 0)
    m.leg_heading = 0
    return m


@pytest.mark.parametrize("dz, edges", [(0.25, ("sill",)), (-0.25, ("header",)), (0, ())])
def test_alignment_uses_existing_frames_without_extra_wait(dz, edges):
    m = mission()
    m.frame = SimpleNamespace(points=lambda *_: window_points(dz, edges))
    for t in (0.1, 0.2, 0.3):
        m.depth_t = t
        out = C.Output(phase=m.phase)
        m._align(t, out, C.Vehicle(heading=0))
    assert m.phase == "check"
    assert m.recover is None
    assert m.sp[2] == pytest.approx(-0.8 - dz, abs=0.02)
    if not edges:
        assert any("height unavailable (unchanged)" in event for event in out.events)


@pytest.mark.parametrize("group, limit", [("window", 0.30), ("tunnel", 0.15)])
@pytest.mark.parametrize("sign", [-1, 1])
def test_window_limit_does_not_change_other_gate_limits(group, limit, sign):
    m = mission(group)
    m._apply_alignment(0.3, C.Output(phase="align"), dict(lateral_m=0, dz_m=sign * 0.5))
    assert m.sp[2] == pytest.approx(-0.8 - sign * limit)


def profiled_mission():
    m = mission()
    m.p.legs[0]['align'].update(target_z_m=1.4, dz_lo=-.4, dz_hi=.4)
    m.p.align_window_max_dz_m = .1
    m.p.track_mode = 'path'
    m.home = (0, 0, 0, 0)
    m.sp[2] = -.7  # landed origin is 0.10 m above floor: body is at 0.80 m
    m.leg_len = 1.0
    m.offboard_seen = True
    m.phase = 'turn'
    m.t_last = 0
    v = C.Vehicle(pos=(0, 0, -.7), vel=(0, 0, 0), heading=0, pos_ok=True,
                  nav_state=C.NAV_OFFBOARD, armed=True, landed=False)
    return m, v


def tick(m, v, t):
    m.on_health(t, True)
    m.depth_t = m.corr_t = t
    # Isolate the height interlock from the separately tested corridor obstacle policy.
    m._clearance = lambda _: float('inf')
    return m.step(t, v)


def test_nominal_height_waits_for_measured_pose_not_just_command():
    m, v = profiled_mission()
    before = tuple(m.sp[:2])
    for t in np.arange(.1, 12, .1):
        out = tick(m, v, float(t))
        assert out.phase == 'gate_height'
        assert out.setpoint[:2] == before
        assert out.velocity == (0, 0)
        assert out.accel == (0, 0)
        assert m.s == 0 and m.path_s == 0
        assert out.commands == []
    assert m.sp[2] == pytest.approx(-1.3)
    assert m.land_reason is None  # elapsed time never authorizes crossing or adds a landing


def test_settle_timer_restarts_when_height_leaves_tolerance():
    m, v = profiled_mission()
    m.sp[2] = -1.3
    v.pos = (0, 0, -1.3)
    tick(m, v, .1)
    for t in (.2, .4, .8):
        assert tick(m, v, t).phase == 'gate_height'
    v.pos = (0, 0, -1.2)
    tick(m, v, .9)
    v.pos = (0, 0, -1.3)
    for t in (1., 1.4, 1.9):
        assert tick(m, v, t).phase == 'gate_height'
    assert tick(m, v, 2.01).phase == 'align'
    assert m.sp[:2] == [0, 0]


@pytest.mark.parametrize('dz', [None, .5, -.5])
def test_camera_trim_has_its_own_height_hold_before_crossing(dz):
    m, v = profiled_mission()
    m.phase = 'align'
    m.sp[2] = -1.3
    v.pos = (0, 0, -1.3)
    estimate = dict(lateral_m=.1, heading_rad=0, width_m=.9, range_m=.8, dz_m=dz)
    m.frame = object()
    m._measure_gate = lambda _: dict(estimate)
    for t in (.1, .2, .3):
        out = tick(m, v, t)
    assert out.phase == 'gate_height'
    target = -1.3 - (0 if dz is None else np.clip(dz, -.1, .1))
    assert m.gate_height_target == pytest.approx(target)
    assert m.sp[:2] == [0, 0]  # lateral target moved, actual XY setpoint is still held
    assert m.targets[0] != (1, 0)
    v.pos = (0, 0, target)
    for t in np.arange(.4, 2.5, .1):
        tick(m, v, float(t))
        if m.phase == 'check':
            break
    assert m.phase == 'check'
    out = tick(m, v, 2.8)
    assert out.phase == 'go'
    assert m.sp[2] == pytest.approx(target)


def test_floor_reference_uses_takeoff_position_and_configured_landed_height():
    m, v = profiled_mission()
    m.home = (0, 0, 5, 0)
    m.p.landed_z_m = .18
    tick(m, v, .1)
    assert m.gate_height_target == pytest.approx(5 + .18 - 1.4)


def test_retains_window_height_on_following_pad_leg():
    m, v = profiled_mission()
    m.p.legs.append(dict(name='pad', hold=True))
    m.targets.append((2, 0))
    m.sp[2] = -1.38  # nominal plus accepted camera trim
    m.leg = 1
    m._next_leg(.1, C.Output('hold'), v)
    assert m.sp[2] == pytest.approx(-1.38)
    assert not m._window_height_required()


def test_repeated_alignments_cannot_accumulate_more_than_ten_cm_of_trim():
    m, _ = profiled_mission()
    m.sp[2] = -1.3
    for _ in range(3):
        m._apply_alignment(.1, C.Output('align'), dict(lateral_m=0, dz_m=.1))
    assert m.sp[2] == pytest.approx(-1.4)


@pytest.mark.parametrize('error', [-.03, .03])
def test_camera_measurement_accounts_for_actual_height_error(error):
    m, v = profiled_mission()
    m.sp[2] = -1.3
    v.pos = (0, 0, -1.3 - error)
    # Shift the synthetic opening edges so its centre is -error relative to the body.
    meta = m.p.legs[0]['align']
    meta['dz_lo'], meta['dz_hi'] = -.45, .45
    m.frame = SimpleNamespace(points=lambda *_: window_points(-error))
    est = m._measure_gate(v)
    assert est is not None
    assert est['dz_m'] == pytest.approx(0, abs=.02)


def test_missing_velocity_does_not_add_a_new_blocker_after_height_settles():
    m, v = profiled_mission()
    m.sp[2] = -1.3
    v.pos, v.vel = (0, 0, -1.3), None
    for t in np.arange(.1, 1.5, .1):
        tick(m, v, float(t))
    assert m.phase == 'align'


def test_no_continuous_height_veto_after_crossing_has_started():
    m, v = profiled_mission()
    m.phase = 'go'
    m.sp[2] = -1.3
    calls = []
    m._go = lambda *_: calls.append('advance')
    tick(m, v, .1)  # height differs: crossing keeps the original flight controller behavior
    assert m.phase == 'go' and calls == ['advance']


@pytest.mark.parametrize('partial_frames', [0, 1, 2])
def test_window_detection_timeout_discards_partial_offsets_and_continues(partial_frames):
    m, v = profiled_mission()
    m.phase = 'align'
    m.sp[2] = -1.3
    v.pos = (0, 0, -1.3)
    m.p.blocked_hold_s = 4
    m.align_hist = [dict(lateral_m=.2, dz_m=.1)] * partial_frames
    before = list(m.targets)
    out = tick(m, v, m.p.align_timeout_s + .1)
    assert out.phase == 'check'
    assert m.targets == before and m.sp == [0, 0, -1.3, 0]
    assert m.recover is None and m.land_reason is None and out.commands == []
    assert m.align_hist == []
    assert m.align_result['fallback'] is True
    assert any('WINDOW ALIGNMENT FALLBACK' in event for event in out.events)
    out = tick(m, v, m.p.align_timeout_s + .6)
    assert out.phase == 'go'


@pytest.mark.parametrize('distance, proceeds', [(np.nan, True), (.34, False)])
def test_real_blank_depth_falls_back_and_crosses_but_a_measured_wall_still_stops(distance, proceeds):
    import json
    from pathlib import Path

    m, v = profiled_mission()
    doc = json.loads((Path(__file__).resolve().parents[1] / 'course/course_v3.json').read_text())
    for settings in (doc['refnav_params'], doc['navigation']['refnav']):
        for key, value in settings.items():
            if not key.startswith('_'):
                setattr(m.p, key, value)
    m.phase = 'align'
    m.sp[2] = -1.3
    v.pos = (0, 0, -1.3)
    rotation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    image = np.full((160, 160), distance, dtype=np.float32)
    advanced = False
    events = []
    for t in np.arange(.1, 6, .1):
        t = float(t)
        assert m.on_depth(t, image, (160, 160, 80, 80), rotation, (0, 0, 0), (1, 0, 0, 0), now=t)
        m.on_health(t, True)
        out = m.step(t, v)
        events += out.events
        advanced |= m.v_cmd > 0
        assert out.commands == [] and m.land_reason is None
        assert m.sp[2] == pytest.approx(-1.3)
    assert any('WINDOW ALIGNMENT FALLBACK' in event for event in events)
    assert advanced == proceeds
    assert m.phase == ('go' if proceeds else 'blocked')


def test_tunnel_detection_timeout_keeps_existing_recovery_behavior():
    m = mission('tunnel')
    m.p.blocked_hold_s = 4
    v = C.Vehicle(pos=(0, 0, -.8), heading=0)
    out = C.Output('align')
    m._align(m.p.align_timeout_s + .1, out, v)
    assert m.phase == 'blocked'
    assert m.recover['kind'] == 'align'
    assert not any('WINDOW ALIGNMENT FALLBACK' in event for event in out.events)
