"""A disappeared navigator route must reach bounded recovery, not a corridor timeout."""
import json
from dataclasses import replace
from types import SimpleNamespace as NS
import numpy as np
import pytest
from refnav import core as C
from test_refnav_blocked import STRAIGHT, dense, fly_path, params, speeds
from test_recovery_controller import blocked, node_callback, step


def cruise(**kwargs):
    m,v=blocked(**kwargs)
    path=m.recover['old_path']
    m.recover=None;m.replan_request=None;m.replan_serial=0
    m.phase='go';m.path=path;m.path_s=.8;m.plan_reference=(path,.8)
    m.last_v=v;m.t_check=0.;m.corr_t=0.
    return m,v


def fail(m,t=10.01,reason='no collision-free path'):
    assert m.on_plan_result(t,0,None,'failed',reason)


def test_node_failure_queues_recovery_then_uses_current_measured_pose(node_callback):
    node=node_callback;m,v=cruise();node.mission=m
    old=m.path
    node._on_path(NS(data=json.dumps(dict(leg=0,request_id=None,plan_state='failed',reason='map says no route'))))
    assert m.path is None and m.recover is None and m.plan_failure is not None
    # Arrived vehicle telemetry, not the callback's older last_v, anchors the hold.
    v=replace(v,pos=(1.03,.02,-1.))
    out=step(m,v,10.1)
    assert out.phase=='blocked' and out.setpoint[:2]==v.pos[:2]
    assert out.velocity==(0.,0.) and out.accel==(0.,0.)
    assert m.recover['old_path'] is old and m.recover['old_path_s']==.8
    assert m.recover['pid']==1 and m.recover['cause']=='no_route'
    assert m.replan_request=={'id':1,'leg':0,'observed_only':True,
                             'reference_pid':1,'reference_progress_s':.8,'require_detour':False}
    assert m.status(10.1)['path_s'] is None and m.status(10.1)['path_pid'] is None
    assert m.counts['plan_stops']==1 and m.leg_blocks==1
    assert 'corridor' not in m.why


def test_multiple_pending_failed_replies_preserve_original_reference_and_reason():
    m,v=cruise();old=m.path
    assert m.status(10.)['path_s']==.8 and m.status(10.)['path_pid']==1
    assert m.on_plan_result(10.01,0,None,'pending')
    fail(m,10.02,'original planning failure')
    assert m.on_plan_result(10.03,0,None,'pending')
    fail(m,10.04,'awaiting bounded planner retry')
    assert not m.on_path(10.05,0,STRAIGHT,speeds(STRAIGHT),pid=9)
    step(m,v,10.1)
    assert m.recover['old_path'] is old and m.recover['pid']==1
    assert m.recover['old_path_s']==.8 and 'original planning failure' in m.recover['why']
    t0=m.recover['t0']
    assert not m.on_plan_result(10.2,0,None,'failed','late normal response')
    assert m.recover['t0']==t0 and m.recover['requests']==1
    m._request_recovery_plan(10.3)
    assert m.replan_request['require_detour'] is False


def test_no_reply_runs_scan_and_lands_for_no_route_not_fresh_depth_timeout():
    m,v=cruise();fail(m)
    phases=[];requests=[]
    for t in np.arange(10.1,31.,.05):
        v=replace(v,pos=tuple(m.sp[:3]),heading=m.sp[3])
        out=step(m,v,float(t))
        phases.append(out.phase)
        if m.replan_request:requests.append(m.replan_request['id'])
        if out.phase=='land':break
    assert 'blocked' in phases and 'recovery_scan' in phases and 'backoff' in phases
    assert m.phase=='land' and 'navigator has no route' in m.land_reason
    assert 'corridor' not in m.land_reason and 'depth silent' not in m.land_reason
    assert 2<=max(requests)<=5 and t-10.1<=20.05
    assert m.recover is None and m.replan_request is None


def test_no_route_scan_still_has_hard_timeout_when_vehicle_cannot_settle():
    m,v=cruise();fail(m);step(m,v,10.1)
    for t in np.arange(10.2,30.3,.1):
        out=step(m,replace(v,vel=(.2,0.,0.)),float(t))
        if out.phase=='land':break
    assert m.phase=='land' and 'navigator has no route' in m.land_reason
    assert 'bounded recovery timed out' in m.land_reason
    assert m.replan_serial==1


@pytest.mark.parametrize('phase',['wait','arm','climb','hover','hold','land','done'])
def test_failed_plan_never_starts_recovery_on_ground_or_inactive_leg_phase(phase):
    m,v=cruise();m.phase=phase;fail(m)
    assert m.plan_failure is None and m.recover is None and m.replan_request is None


def test_no_previous_valid_path_is_not_claimed_as_a_recovery_reference():
    m,v=cruise();m.path=None;m.plan_reference=None;fail(m)
    assert m.plan_failure is None


def test_no_route_counts_against_existing_per_leg_block_limit():
    m,v=cruise(blocked_max_per_leg=1);m.leg_blocks=1;fail(m)
    out=step(m,v,10.1)
    assert out.phase=='land' and 'blocked 2 times' in m.land_reason
    assert m.replan_request is None and m.recover is None


@pytest.mark.parametrize('supervision',['px4','depth','health','position'])
def test_supervision_takes_priority_over_queued_no_route_recovery(supervision):
    m,v=cruise();fail(m)
    if supervision=='px4':v=replace(v,nav_state=2)
    elif supervision=='depth':m.depth_t=None
    elif supervision=='health':m.health=C.Health(0.,.1)
    else:v=replace(v,pos_ok=False)
    if supervision!='health':m.health.on_msg(10.1,True)
    out=m.step(10.1,v)
    assert out.phase==('done' if supervision=='px4' else 'land')
    assert m.recover is None and m.replan_request is None and m.plan_failure is None
    if supervision=='px4':assert not out.stream and not out.commands


def test_no_route_rejects_stale_and_uncertified_reply_then_accepts_matching_detour():
    m,v=cruise();fail(m);step(m,v,10.1)
    points,stops=dense([(1.,0.),(1.,1.),(4.,1.)]);rid=m.replan_request['id']
    for request,cert in ((rid-1,True),(rid,False),(None,True)):
        assert not m.on_path(10.2,0,points,speeds(points,stops),stops,pid=2,
                             request_id=request,recovery_certified=cert)
    assert m.on_path(10.2,0,points,speeds(points,stops),stops,pid=2,
                     request_id=rid,recovery_certified=True)
    out=step(m,v,10.25)
    assert out.phase=='turn' and m.counts['replan_resumes']==1


def test_failed_normal_plan_certified_recovery_turns_checks_and_completes():
    state={}
    def tick(t,m):
        if not m.targets:return
        if m.phase=='go' and m.path_s>.2 and 'failed' not in state:
            m.on_plan_result(t,0,None,'failed','test obstruction closes original route')
            state['failed']=t
        if m.recover is not None and 'points' not in state:
            n,e=m.last_v.pos[:2]
            points,stops=dense([(n,e),(n,1.3),(4.,1.3)])
            state.update(points=points,stops=stops)
        if 'points' in state:
            req=m.replan_request
            m.on_path(t,0,state['points'],speeds(state['points'],state['stops']),state['stops'],pid=2,
                      request_id=None if req is None else req['id'],recovery_certified=req is not None)
        elif 'failed' not in state:m.on_path(t,0,STRAIGHT,speeds(STRAIGHT),pid=1)
    m,px4,log=fly_path(params(recovery_replan=True,recovery_scan_deg=25.),lambda t:[],tick)
    assert m.result=='complete',(m.result,m.why)
    phases=[phase for t,phase,*_ in log if t>=state['failed']]
    assert phases.index('blocked')<phases.index('turn')<phases.index('check')<phases.index('go')
    assert m.counts['plan_stops']==1 and m.counts['replan_resumes']==1
    assert abs(px4.pos[1]-1.3)<.2
