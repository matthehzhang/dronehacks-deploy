"""Execute real NavNode callbacks/tick with ROS message/clock stand-ins."""
import importlib
import json
import sys
import types
from types import SimpleNamespace as NS
import numpy as np
import pytest
from coursenav.planner import PlanError
from test_observed_recovery import scene, observed


@pytest.fixture
def node(monkeypatch):
    modules = {
        'rclpy': {}, 'rclpy.node': {'Node': object},
        'rclpy.executors': {'ExternalShutdownException': RuntimeError},
        'rclpy.qos': {'QoSProfile': lambda **kw: kw, 'ReliabilityPolicy': NS(RELIABLE=1),
                      'DurabilityPolicy': NS(TRANSIENT_LOCAL=1), 'HistoryPolicy': NS(KEEP_LAST=1)},
        'nav_msgs.msg': {'OccupancyGrid': object}, 'std_msgs.msg': {'String': NS}}
    for name, attrs in modules.items():
        mod = types.ModuleType(name);vars(mod).update(attrs);monkeypatch.setitem(sys.modules,name,mod)
    monkeypatch.delitem(sys.modules,'coursenav.nav_node',raising=False)
    cls = importlib.import_module('coursenav.nav_node').NavNode
    n = object.__new__(cls)
    n.nav, ref = scene();n.obst=n.nav.prior_obst;n.surf_dist=n.obst.distance()
    n.reference_path,n.reference_leg=ref,0
    n.recovery_reference=n.recovery_prefix=n.recovery=None
    n.recovery_last_id=-1;n.recovery_last_attempt=None
    n.recovery_min_interval_s=.5;n.recovery_grid_max_age_s=2.
    n.local_pts=None;n.local_explain_m=.15;n.prior_state='unknown';n.observed_free=None
    n.nav_legs=[{'path':True,'goals':[[6.,2.5]],'stop':[6.,2.5]}]
    n.nav_leg=0;n.pid=4;n.cache={0:('must not use',)};n.decl=0.;n.normal_last_attempt=None
    n.drift={'delta':[0,0,0],'nominal':[1.,2.5,0.],'fixture':[0,0,0]}
    n.status={'leg':0,'phase':'blocked'}
    n.stats=dict(replans=0,plans=0,plan_errors=0,local_cells=0)
    n.now=10.;n.get_clock=lambda:NS(now=lambda:NS(nanoseconds=int(n.now*1e9)))
    n.get_logger=lambda:NS(info=lambda *a:None,warning=lambda *a:None)
    n.messages=[];n.pub=NS(publish=lambda m:n.messages.append(json.loads(m.data)))
    return n


def request(n, rid, phase='blocked'):
    n._on_status(NS(data=json.dumps({'leg':0,'phase':phase,'stop_m':.95,
        'recovery_request':{'id':rid,'leg':0,'observed_only':True}})))


def test_explicit_request_invalidates_old_path_and_cached_plan(node):
    request(node,1)
    assert node.nav.path is None and not node.cache
    node._tick()
    out=node.messages[-1]
    assert out['request_id']==1 and out['plan_state']=='failed'
    assert out['recovery_certified'] is False and 'pts_ne' not in out
    assert 'fresh' in out['reason']
    errors=node.stats['plan_errors'];node._tick()
    assert node.stats['plan_errors']==errors  # no implicit retry on repeated id


def test_new_request_retries_bounded_and_stale_ids_are_ignored(node,monkeypatch):
    request(node,4);node._tick()
    request(node,5,'recovery_scan');node._tick()
    assert node.messages[-1]['request_id']==5 and node.messages[-1]['plan_state']=='pending'
    request(node,3)
    assert node.status['recovery_request']['id']==5
    node.now+=.6;node._tick()
    assert node.messages[-1]['plan_state']=='failed'
    assert node.stats['plan_errors']==2


def test_fresh_certified_detour_ready_then_stale_grid_revokes_it(node):
    mask=np.zeros(node.obst.occ.shape,bool);mask[42:48,46:54]=True
    node.nav.set_local(mask);node.observed_free=observed()
    request(node,1);node._tick();out=node.messages[-1]
    assert out['plan_state']=='ready' and out['recovery_certified'] is True
    assert out['pid']==5 and out['certified_prefix_m']>=1 and len(out['pts_ne'])>2
    node.now=12.01;node._tick();out=node.messages[-1]
    assert out['plan_state']=='failed' and out['recovery_certified'] is False
    assert 'pts_ne' not in out and node.nav.path is None


def test_regular_failed_replan_does_not_publish_old_path(node,monkeypatch):
    node.cache={};node.status={'leg':0,'phase':'go'}
    monkeypatch.setattr(node.nav,'needs_replan',lambda *a:True)
    def fail(*a,**kw):raise PlanError('no route')
    monkeypatch.setattr(node.nav,'plan',fail)
    node._tick()
    assert node.nav.path is None and 'pts_ne' not in node.messages[-1]
    assert node.messages[-1]['plan_state']=='failed' and node.messages[-1]['request_id'] is None
    assert node.reference_path is not None  # retained only as non-publishable recovery reference


def test_unknown_prior_cannot_explain_away_a_local_obstacle(node):
    node.local_pts=np.array([[1.,2.5]])
    node.surf_dist[:]=0.
    node._tick()
    assert node.stats['local_cells']==1
    node.prior_state='placed';node._tick()
    assert node.stats['local_cells']==0


def test_local_grid_frame_order_and_stamp_are_checked(node):
    def msg(stamp=10,frame='coursenav_nominal',qz=0):
        return NS(header=NS(stamp=NS(sec=stamp,nanosec=0),frame_id=frame),
                  info=NS(origin=NS(position=NS(x=0,y=0),orientation=NS(x=0,y=0,z=qz,w=1)),
                          resolution=.05,width=40,height=40),data=[12]*1600)
    node._on_local(msg())
    assert node.observed_free.stamp==10
    node._on_local(msg(9))
    assert node.observed_free.stamp==10
    node._on_local(msg(frame='course'))
    assert node.observed_free is None
    node._on_local(msg(qz=float('nan')))
    assert node.observed_free is None


def test_replayed_finished_request_never_reenters_normal_planning(node):
    request(node,8);node._tick()
    node._on_status(NS(data=json.dumps({'leg':0,'phase':'go'})))
    assert node.recovery is None
    request(node,8,'blocked')
    assert node.status['phase']=='go' and node.recovery is None


def test_grid_expiring_during_planning_cannot_certify_reply(node, monkeypatch):
    mask=np.zeros(node.obst.occ.shape,bool);mask[42:48,46:54]=True
    node.nav.set_local(mask);node.observed_free=observed()
    plan=node.nav.plan_recovery
    def slow_plan(*args, **kwargs):
        result=plan(*args, **kwargs)
        node.now=12.01
        return result
    monkeypatch.setattr(node.nav,'plan_recovery',slow_plan)
    request(node,1);node._tick()
    out=node.messages[-1]
    assert out['plan_state']=='failed' and 'expired' in out['reason']
    assert not out['recovery_certified'] and 'pts_ne' not in out


def test_straight_leg_still_publishes_target_corrections_without_failure(node):
    node.nav_legs[0]['path']=False
    node._tick()
    assert 'plan_state' not in node.messages[-1]
    assert 'targets_ne' in node.messages[-1]


def test_failed_normal_plan_retries_after_cooldown_and_recovers(node,monkeypatch):
    node.cache={};node.status={'leg':0,'phase':'go'}
    calls=[];real_plan=node.nav.plan
    def temporary_failure(*args,**kwargs):
        calls.append(node.now)
        if len(calls)==1:raise PlanError('temporary obstruction')
        return real_plan(*args,**kwargs)
    monkeypatch.setattr(node.nav,'needs_replan',lambda *a:True)
    monkeypatch.setattr(node.nav,'plan',temporary_failure)
    node._tick()
    assert node.nav.path is None and node.messages[-1]['plan_state']=='failed'
    node.now+=.2;node._tick()
    assert len(calls)==1 and 'pts_ne' not in node.messages[-1]
    node.now+=node.nav.p.replan_every_s;node._tick()
    assert len(calls)==2 and node.messages[-1]['plan_state']=='ready'
    assert node.nav.path is not None and node.messages[-1]['pts_ne']
