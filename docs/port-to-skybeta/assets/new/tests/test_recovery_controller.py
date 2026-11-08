"""Recovery protocol and stationary scan safety through real controller callbacks."""
import importlib
import json
import math
import sys
import types
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest
from refnav import core as C
from test_refnav_blocked import STRAIGHT, WALL, dense, fly_path, params, speeds


def blocked(**kwargs):
    m = C.Mission(params(recovery_replan=True, recovery_scan_deg=25., **kwargs))
    m.phase='go';m.targets=[(4.,0.)];m.home=(0.,0.,0.,0.)
    m.sp=[1.,0.,-1.,0.];m.leg_start=(0.,0.);m.leg_dir=(1.,0.);m.leg_len=4.
    m.offboard_seen=True;m.t_last=10.;m.t_started=0.;m.t_phase=9.
    v=C.Vehicle(pos=(1.,0.,-1.),heading=0.,pos_ok=True,armed=True,
                nav_state=C.NAV_OFFBOARD,vel=(0.,0.,0.),landed=False)
    m.on_path(10.,0,STRAIGHT,speeds(STRAIGHT),pid=1);m.path_s=1.
    m.depth_t=10.;m.depth_pts=np.array([[2.,0.,0.]])
    m.health.on_msg(10.,True)
    m._start_recover(10.,C.Output('go'),'blocked','obstacle',v)
    return m,v


def step(m,v,t,depth=True):
    m.health.on_msg(t,True)
    if depth:m.depth_t=t
    return m.step(t,v)


def scanning(**kwargs):
    m,v=blocked(**kwargs)
    step(m,v,14.1)
    assert m.phase=='recovery_scan'
    return m,v


def test_block_requests_immediately_and_invalidates_old_command_path():
    m,v=blocked()
    assert m.replan_request=={'id':1,'leg':0,'observed_only':True,
                             'reference_pid':1,'reference_progress_s':1.,'require_detour':True}
    assert m.path is None and m.recover['old_path']['pid']==1
    assert m.status(10.)['recovery_request']==m.replan_request
    out=step(m,v,10.1)
    assert out.velocity==(0.,0.) and out.accel==(0.,0.)
    assert out.setpoint[:3]==v.pos


def test_stale_uncertified_and_failed_replies_cannot_resume():
    m,v=blocked();rid=m.replan_request['id']
    assert not m.on_path(10.1,0,STRAIGHT,speeds(STRAIGHT),pid=2,request_id=rid-1,recovery_certified=True)
    assert not m.on_path(10.1,0,STRAIGHT,speeds(STRAIGHT),pid=2,request_id=rid,recovery_certified=False)
    assert m.path is None
    assert m.on_path(10.1,0,STRAIGHT,speeds(STRAIGHT),pid=2,request_id=rid,recovery_certified=True)
    assert not m.on_plan_result(10.2,0,rid-1,'failed','old') and m.path is not None
    assert m.on_plan_result(10.2,0,rid,'failed','unknown corridor')
    assert m.path is None
    assert step(m,v,10.3).phase=='blocked'
    assert m.on_plan_result(10.4,0,rid,'pending') and m.path is None


def test_finished_recovery_replies_cannot_replace_current_normal_path():
    m,v=blocked();rid=m.replan_request['id']
    m.recover=None;m.replan_request=None
    m.on_path(10.1,0,STRAIGHT,speeds(STRAIGHT),pid=3)
    assert not m.on_path(10.2,0,STRAIGHT,speeds(STRAIGHT),pid=2,request_id=rid,recovery_certified=True)
    assert m.path['pid']==3


def test_certified_detour_turns_checks_then_flies():
    detour={}
    def tick(t,m):
        if not m.targets:return
        if m.recover is not None and 'pts' not in detour:
            n,e=m.last_v.pos[:2]
            pts,stops=dense([(n,e),(n,1.3),(4.,1.3)])
            detour.update(pts=pts,stops=stops,t=t)
        if 'pts' in detour:
            req=m.replan_request
            m.on_path(t,0,detour['pts'],speeds(detour['pts'],detour['stops']),detour['stops'],pid=2,
                      request_id=None if req is None else req['id'],recovery_certified=req is not None)
        else:m.on_path(t,0,STRAIGHT,speeds(STRAIGHT),pid=1)
    m,px4,log=fly_path(params(recovery_replan=True,recovery_scan_deg=25.),lambda t:[WALL],tick)
    assert m.result=='complete',m.result
    phases=[p for t,p,*_ in log if t>=detour['t']]
    assert phases.index('turn')<phases.index('check')<phases.index('go')
    assert m.counts['replan_resumes']==1
    assert abs(px4.pos[1]-1.3)<.2


def test_scan_keeps_position_zero_translation_feedforward_and_bounded_yaw():
    m,v=scanning(yaw_accel_dps2=60.)
    anchor=tuple(m.sp[:3]);last_yaw=m.sp[3];requests=[m.replan_request['id']]
    for t in np.arange(14.15,22.,.05):
        v=replace(v,heading=m.sp[3])
        out=step(m,v,float(t))
        if out.phase!='recovery_scan':break
        assert out.setpoint[:3]==anchor
        assert out.velocity==(0.,0.) and out.accel==(0.,0.)
        assert abs(out.setpoint[3])<=math.radians(25.)+1e-9
        assert abs(C._wrap(out.setpoint[3]-last_yaw))<=math.radians(20.)*.05+1e-9
        assert out.yawspeed is None or abs(out.yawspeed)<=math.radians(20.)+1e-9
        last_yaw=out.setpoint[3];requests.append(m.replan_request['id'])
    assert max(requests)>=3 and max(requests)<=4
    assert m.phase=='backoff'


@pytest.mark.parametrize('field,value',[
    ('vel',None),('vel',(float('nan'),0.,0.)),('vel',(.081,0.,0.)),('vel',(0.,0.,.081)),
    ('heading',None),('heading',float('nan')),('pos',(float('nan'),0.,-1.)),
    ('pos',(1.11,0.,-1.))])
def test_scan_does_not_turn_or_advance_request_with_unsettled_invalid_pose(field,value):
    m,v=scanning();v=replace(v,**{field:value});old=tuple(m.sp);rid=m.replan_request['id']
    out=step(m,v,14.2)
    assert tuple(m.sp)==old
    assert m.replan_request['id']==rid
    assert out.velocity==(0.,0.) and out.accel==(0.,0.)


def test_scan_dwell_requires_an_additional_usable_depth_frame():
    m,v=scanning(depth_stale_s=2.)
    m.sp[3]=m.recover['scan_yaws'][0];v=replace(v,heading=m.sp[3])
    step(m,v,14.2)
    rid=m.replan_request['id']
    step(m,v,14.8,depth=False)
    assert m.replan_request['id']==rid  # recent but unchanged frame is insufficient
    step(m,v,14.85)
    assert m.replan_request['id']==rid+1


def test_request_count_cannot_renew_whole_recovery_budget():
    m,v=blocked()
    for t in np.linspace(10.1,15.,100):m._request_recovery_plan(float(t))
    assert m.recover['requests']==5 and m.replan_request['id']==5
    assert m.recover['t0']==10.


def test_close_obstacle_abandons_scan_without_yawing():
    m,v=scanning();m.depth_pts=np.array([[.3,0.,0.]])
    yaw=m.sp[3];out=step(m,v,14.2)
    assert out.phase=='backoff' and m.sp[3]==yaw


def test_scan_depth_loss_and_timeout_never_extend_recovery_budget():
    m,v=scanning();yaw=m.sp[3]
    out=step(m,v,15.2,depth=False)
    assert out.phase=='recovery_scan' and m.sp[3]==yaw
    out=step(m,v,20.,depth=False)
    assert out.phase=='land' and 'depth silent' in m.land_reason
    m,v=scanning(recovery_timeout_s=6.)
    for t in (14.2,14.8,15.4):
        m._request_recovery_plan(t)
        step(m,v,t)
    out=step(m,v,16.01)
    assert out.phase=='land' and 'timed out' in m.land_reason
    assert m.recover is None and m.replan_request is None


def test_scan_respects_px4_takeover_without_land_or_setpoint_stream():
    m,v=scanning();out=step(m,replace(v,nav_state=2),14.2)
    assert out.phase=='done' and m.result.startswith('yielded: PX4 left Offboard')
    assert not out.stream and out.setpoint is None and not out.commands


@pytest.fixture
def node_callback(monkeypatch):
    modules={'rclpy':{},'rclpy.node':{'Node':object},
             'rclpy.clock':{'Clock':object,'ClockType':object},
             'rclpy.qos':{'QoSProfile':lambda **kw:kw,'ReliabilityPolicy':NS(BEST_EFFORT=1),
                          'DurabilityPolicy':NS(TRANSIENT_LOCAL=1),'HistoryPolicy':NS(KEEP_LAST=1),
                          'qos_profile_sensor_data':None},
             'sensor_msgs.msg':{'CameraInfo':object,'Image':object},
             'std_msgs.msg':{'Bool':object,'String':NS}}
    for name,attrs in modules.items():
        module=types.ModuleType(name);vars(module).update(attrs);monkeypatch.setitem(sys.modules,name,module)
    monkeypatch.delitem(sys.modules,'refnav.node',raising=False)
    cls=importlib.import_module('refnav.node').RefNav
    n=object.__new__(cls);n.mission,_=blocked();n.mission.targets.append((5.,0.))
    n._now=lambda:10.1;n.trace=NS(record=lambda *a,**kw:None)
    return n


def test_node_rejects_stale_or_uncertified_targets_and_malformed_payload(node_callback):
    n=node_callback;m=n.mission;rid=m.replan_request['id'];original=list(m.targets)
    for reply in ({'request_id':rid-1,'recovery_certified':True},
                  {'request_id':rid,'recovery_certified':False}):
        n._on_path(NS(data=json.dumps(dict(reply,leg=0,plan_state='ready',pid=2,
                   pts_ne=STRAIGHT.tolist(),v=speeds(STRAIGHT).tolist(),targets_ne={'1':[99.,99.]}))))
        assert m.targets==original and m.path is None
    for payload in ('[]','null','{"leg": "bad", "plan_state": "failed"}'):
        n._on_path(NS(data=payload))
        assert m.targets==original


@pytest.mark.parametrize('name,value', [
    ('recovery_timeout_s', float('nan')), ('recovery_timeout_s', float('inf')),
    ('recovery_timeout_s', 0.), ('recovery_timeout_s', 31.),
    ('recovery_scan_deg', 46.), ('recovery_scan_rate_dps', 21.),
    ('recovery_scan_dwell_s', float('nan'))])
def test_invalid_recovery_configuration_cannot_start(name, value):
    with pytest.raises(ValueError, match=name):
        C.Mission(params(recovery_replan=True, **{name:value}))


def test_failed_normal_plan_brakes_at_measured_pose_not_old_carrot():
    m,v=blocked()
    m.recover=None;m.replan_request=None;m.last_v=v
    m.sp[:2]=[1.3,.2]
    m.on_path(10.1,0,STRAIGHT,speeds(STRAIGHT),pid=3)
    assert m.on_plan_result(10.2,0,None,'failed','no route')
    assert m.path is None and tuple(m.sp[:2])==v.pos[:2]
