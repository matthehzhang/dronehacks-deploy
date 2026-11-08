"""Unobserved corridors explore continuously; movement still needs certified free space."""
from dataclasses import replace
import math
import numpy as np
import pytest
from refnav import core as C
from test_no_route_recovery import cruise
from test_recovery_controller import step
from test_refnav_blocked import STRAIGHT, speeds
from test_recovery_creep import give_hop, hop_pts


def exploring(phase='go', **kwargs):
    m, v = cruise(recovery_creep=True, **kwargs)
    m.phase = phase
    m.corr_t = None
    out = step(m, v, 10.1)
    assert m.recover['cause'] == 'unobserved'
    return m, v, out


@pytest.mark.parametrize('phase', ['check', 'go'])
def test_missing_corridor_starts_scanning_and_requests_certified_creep(phase):
    m, v, out = exploring(phase)
    assert out.phase == 'recovery_scan'
    assert out.setpoint[:3] == v.pos
    assert out.velocity == (0., 0.) and out.accel == (0., 0.)
    assert not out.commands and m.leg_blocks == 0
    assert m.replan_request['creep'] is True
    assert m.replan_request['observed_only'] is True
    assert m.replan_request['require_detour'] is False
    assert m.recover['old_path_s'] == .8


def test_no_reply_continues_multiple_exploration_rounds_without_landing():
    m, v, _ = exploring()
    phases, yaws, ids = set(), [], []
    for t in np.arange(10.15, 130., .05):
        v = replace(v, pos=tuple(m.sp[:3]), heading=m.sp[3])
        out = step(m, v, float(t))
        phases.add(out.phase); yaws.append(out.setpoint[3]); ids.append(m.replan_request['id'])
        assert out.phase not in ('land', 'done') and 'land' not in out.commands
        assert m.replan_request['require_detour'] is False
        assert out.setpoint[0] >= .5
        assert abs(out.setpoint[1]) < 1e-9
    assert 'recovery_scan' in phases
    assert max(yaws) - min(yaws) > math.radians(40)
    assert m.recover['exploration_round'] >= 4 and max(ids) > 5
    assert ids == sorted(ids)
    assert m.result is None


def test_full_certified_route_can_resume_after_multiple_rounds():
    m, v, _ = exploring()
    old_id = m.replan_request['id']
    m._finish_recovery(31., C.Output('recovery_scan'), v, 'round exhausted')
    rid = m.replan_request['id']
    assert rid > old_id
    for request, certified in [(old_id, True), (rid, False)]:
        assert not m.on_path(31.1, 0, STRAIGHT, speeds(STRAIGHT), pid=2,
                             request_id=request, recovery_certified=certified)
    assert m.on_path(31.1, 0, STRAIGHT, speeds(STRAIGHT), pid=2,
                     request_id=rid, recovery_certified=True)
    out = step(m, v, 31.15)
    assert out.phase == 'turn' and m.recover is None


def test_stalled_hops_begin_another_exploration_round_with_new_request_ids():
    m, v, _ = exploring()
    m.recover['stalls'] = m.p.creep_max_stalls - 1
    old_id = m.replan_request['id']
    m._end_hop(10.2, C.Output('creep'), v, 'corridor unobserved', False)
    assert m.recover['exploration_round'] == 2
    assert m.replan_request['id'] > old_id and m.replan_request['creep']
    assert m.result is None and m.phase == 'blocked'
    assert step(m, v, 10.25).phase == 'recovery_scan'


def test_new_exploration_round_can_accept_another_certified_hop():
    m, v, _ = exploring()
    m._finish_recovery(31., C.Output('recovery_scan'), v, 'round exhausted')
    give_hop(m, 31.1, hop_pts(v.pos[:2], (1.4, 0.)))
    assert step(m, v, 31.15).phase == 'creep'
    assert m.recover['cause'] == 'unobserved'


@pytest.mark.parametrize('supervision', ['depth', 'health', 'position', 'pilot'])
def test_fault_and_pilot_supervision_still_preempt_exploration(supervision):
    m, v, _ = exploring()
    m.health.on_msg(40., True); m.depth_t = 40.
    if supervision == 'depth': m.depth_t = None
    elif supervision == 'health': m.health = C.Health(0., .1)
    elif supervision == 'position': v = replace(v, pos_ok=False)
    else: v = replace(v, nav_state=2)
    out = m.step(40., v)
    assert out.phase == ('done' if supervision == 'pilot' else 'land')
    serial = m.replan_serial
    next_out = step(m, v, 40.1)
    assert next_out.phase == out.phase and m.replan_serial == serial
    if supervision == 'pilot':
        assert not out.stream and not out.commands
    else:
        assert 'land' in out.commands


def test_fresh_corridor_or_disabled_recovery_does_not_trigger_exploration():
    for enabled in (False, True):
        m, v = cruise()
        m.p.recovery_replan = enabled
        m.corr_t = 10.09 if enabled else None
        out = step(m, v, 10.1)
        assert m.recover is None
        assert out.phase not in ('land', 'recovery_scan')


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1., 0., 31.])
def test_invalid_exploration_delay_is_rejected(value):
    with pytest.raises(ValueError, match='corridor_explore_after_s'):
        C.Mission(C.Params(recovery_replan=True, corridor_explore_after_s=value))


def test_exhausted_hop_never_becomes_a_missing_route_reference():
    m, v, _ = exploring()
    m.recover['old_path'] = None
    m.recover['pid'] = None
    assert give_hop(m, 10.2, hop_pts(v.pos[:2], (1.4, 0.)))
    m._finish_recovery(31., C.Output('creep'), v, 'round exhausted')
    assert m.recover['old_path'] is None
    assert m.replan_request['reference_pid'] is None
