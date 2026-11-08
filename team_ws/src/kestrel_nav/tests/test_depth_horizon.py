"""Regression contracts for the hardware zero-speed deadlock and depth horizons."""
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from coursenav.depth_planner import DepthPlanner
from coursenav.planner import PlanError
from coursenav.smooth import SmoothParams
from coursenav.localgrid import LocalGrid, LocalParams
from coursenav.scan import Scan
from refnav.core import Mission, Params, Vehicle, Output


def planner():
    return DepthPlanner(.4,SmoothParams(need_m=.4,reach_m=.28,tube_response_s=.5,center_stops=False))


def test_horizon_extends_to_goal_as_new_depth_reveals_space():
    n=planner()
    prob=np.full((120,100),-1)
    prob[:65,:]=10
    n.update((-1.,-2.5),.05,prob)
    p=n.plan((0.,0.),(4.,0.))
    assert not n.terminal and p.pts[-1,0]<2.25
    assert np.interp(.1,p.s,p.v)>0 and n.safe(np.array((0.,0.)))
    prob[:,:]=10
    n.update((-1.,-2.5),.05,prob)
    p=n.plan((.5,0.),(4.,0.))
    assert n.terminal and np.allclose(p.pts[-1],(4.,0.))


def test_recorded_hardware_start_moves_when_depth_certifies_space():
    # This exact estimate produced the old prior's 23 cm zero-speed prefix.
    n=planner();n.update((-5.,-2.),.05,np.full((120,120),10))
    p=n.plan((-2.4067,.7239),(-2.5,1.35))
    assert n.terminal and np.interp(.1,p.s,p.v)>0
    assert np.allclose(p.pts[0],(-2.4067,.7239))


def test_translating_vio_frame_preserves_route_and_speed():
    a,b=planner(),planner();offset=np.array([3.,-7.])
    prob=np.full((120,100),10)
    a.update((-1.,-2.5),.05,prob)
    b.update(np.array([-1.,-2.5])+offset,.05,prob)
    pa=a.plan((0.,0.),(4.,0.));pb=b.plan(offset,np.array([4.,0.])+offset)
    assert np.allclose(pa.pts,pb.pts-offset) and np.allclose(pa.v,pb.v)


def test_new_wall_with_open_end_invalidates_and_detours_live_route():
    n=planner();prob=np.full((120,100),10)
    n.update((-1.,-2.5),.05,prob);n.plan((0.,0.),(4.,0.))
    prob[55:58,30:70]=95
    n.update((-1.,-2.5),.05,prob)
    assert not n.safe(np.array((0.,0.)))
    p=n.plan((0.,0.),(4.,0.))
    assert n.terminal and max(abs(p.pts[:,1]))>1.
    assert n._safe_points(p.pts)


def test_no_snapping_out_of_unknown_or_measured_obstacle_buffer():
    n=planner();prob=np.full((80,80),-1)
    n.update((-2.,-2.),.05,prob)
    with pytest.raises(PlanError,match='start_footprint_not_observed_clear'):
        n.plan((0.,0.),(1.,0.))
    prob[:,:]=10;prob[43:46,38:43]=95
    n.update((-2.,-2.),.05,prob)
    with pytest.raises(PlanError,match='start_footprint_not_observed_clear'):
        n.plan((0.,0.),(1.,0.))
    assert n.path is None


def test_already_at_waypoint_provides_terminal_capture_instead_of_search():
    n=planner();n.update((-2.,-2.),.05,np.full((100,100),10))
    p=n.plan((0.,0.),(0.,0.))
    assert n.terminal and len(p.pts)==2 and p.length==0


def test_pilot_volume_is_opt_in_fixed_and_obstacles_override_it():
    prob=np.full((100,100),-1)
    n=DepthPlanner(.4,SmoothParams(need_m=.4),launch_xy=(0.,0.),launch_radius=1.2)
    n.update((-2.5,-2.5),.05,prob)
    n.plan((0.,0.),(.5,0.))
    assert n.terminal
    prob[53:56,48:53]=95
    n.update((-2.5,-2.5),.05,prob)
    with pytest.raises(PlanError,match='start_footprint_not_observed_clear'):
        n.plan((0.,0.),(.5,0.))


def test_valid_far_return_carves_horizon_without_inventing_wall_at_range_limit():
    g=LocalGrid(LocalParams(min_cells=1))
    g.integrate(1.,(0.,0.,0.),Scan(np.array([0.]),np.array([8.]),np.array([.05])))
    assert g.stats()['known']>40
    assert not g.occupied().any()
    assert np.any(g.L<0)


def mission():
    m=Mission(Params(track_mode='path',local_horizon=True))
    m.phase='go';m.leg=0;m.targets=[(4.,0.)];m.sp=[0.,0.,-.7,0.]
    m.home=[0.,0.,0.,0.];m.leg_dir=(1.,0.);m.leg_len=4.;m.leg_heading=0.
    m.t_started=0.;m.t_phase=0.;m.t_last=1.;m.t_check=1.
    m.health.bad_for=lambda now:0.;m.depth_t=2.;m.offboard_seen=False
    m.last_v=Vehicle(pos=(0.,0.,-.7),heading=0.,pos_ok=True,vel=(0.,0.,0.))
    return m


def test_no_depth_route_searches_without_landing_or_translation():
    m=mission();o=m.step(2.,m.last_v)
    assert m.phase=='search' and m.v_cmd==0
    assert m.sp[:2]==[0.,0.] and not o.commands


def test_frontier_arrival_does_not_complete_mission_waypoint():
    m=mission();m.p.arrive_settle_s=0
    m.on_path(2.,0,[(0.,0.),(.5,0.)],[.2,0.],pid=1,terminal=False)
    m.last_v.pos=(.5,0.,-.7)
    m._clearance=lambda v:math.inf;m.corridor_fresh=lambda now:True
    m._go_path(2.,.05,Output('go'),m.last_v)
    assert m.phase=='search' and m.legs_done==0 and m.path is None
    assert not m.on_path(2.1,0,[(0.,0.),(.5,0.)],[.2,0.],pid=1,terminal=False)
    assert m.path is None # repeated messages cannot resurrect a consumed horizon


def test_new_horizon_leaves_search_faces_path_before_translation():
    m=mission();m.step(2.,m.last_v)
    m.on_path(2.1,0,[(0.,0.),(0.,1.)],[0.,.2],pid=2,terminal=False)
    o=m.step(2.1,m.last_v)
    assert m.phase=='turn' and m.v_cmd==0 and not o.commands


def test_waypoint_intent_never_loads_origin_geometry(monkeypatch):
    from coursenav.course import load_course, PROJECT
    from coursenav.waypoint_intent import plan_intent
    d=load_course(PROJECT/'course/course_depth_waypoints.json')
    p,route,legs,_=plan_intent(d,(-2.515,.914),90.)
    assert len(route)==6 and all(not L.get('align') and not L.get('tunnel') for L in legs)
    assert p['clearance_pending'] and legs[-1]['hold'] and p['waypoints'][-1]['land']


def test_ros_depth_navigator_and_localizer_start_without_origin_map(tmp_path,monkeypatch):
    import json
    rclpy=pytest.importorskip('rclpy')
    from coursenav.course import PROJECT,load_course
    from coursenav.nav_node import NavNode
    from coursenav.loc_node import LocNode
    course=PROJECT/'course/course_depth_waypoints.json'
    monkeypatch.setenv('KESTREL_COURSE_FILE',str(course))
    doc=load_course(course)
    route=tmp_path/'route.json'
    route.write_text(json.dumps(dict(nav_legs=[dict(path=True,stop=[1.,0.],goals=[[1.,0.]])],
        start_xy=[0.,0.],params={},navigation=doc['navigation'])))
    rclpy.init(args=['--ros-args','-p',f'route_file:={route}','-p',f'log_dir:={tmp_path}'])
    nav=loc=None
    try:
        nav=NavNode();loc=LocNode()
        assert nav.depth_only and loc.depth_only and loc.live is None
        records=[];nav.pub=NS(publish=lambda m:records.append(json.loads(m.data)))
        nav.status=dict(leg=0,phase='go',v_cmd=0.)
        nav.drift=dict(delta=[99.,99.,1.],nominal=[0.,0.,0.],fixture=[0.,0.,0.])
        nav.local_fields=((-2.,-2.),.05,np.full((100,100),10))
        nav.local_stamp=nav.get_clock().now().nanoseconds*1e-9;nav.local_error=''
        nav._tick()
        assert records[-1]['valid'] and records[-1]['terminal']
        assert np.allclose(records[-1]['pts_course'][0],[0.,0.]) # PF delta ignored
        nav.local_fields[2][40,40]=95
        nav.local_stamp=nav.get_clock().now().nanoseconds*1e-9
        nav._tick()
        assert not records[-1]['valid'] # unsafe path withdrawn without throttle delay
    finally:
        if nav is not None:nav.destroy_node()
        if loc is not None:loc.destroy_node()
        rclpy.shutdown()


def test_depth_mode_cli_dry_run_never_starts_flight_or_loads_map(tmp_path,monkeypatch,capsys):
    import fly_course
    from coursenav.course import PROJECT
    monkeypatch.setenv('KESTREL_COURSE_FILE',str(PROJECT/'course/course_depth_waypoints.json'))
    monkeypatch.setattr(fly_course,'run_record_dir',lambda *a:tmp_path)
    monkeypatch.setattr(fly_course,'start_navigator',lambda *a:pytest.fail('flight started on dry run'))
    assert fly_course.main(['--dry-run','--skip-preflight','--no-own-localizer',
                           '--start-from-mark','--no-return','--landed-z','0.1'])==0
    assert 'live depth pending' in capsys.readouterr().out


def test_package_contains_only_horizon_navigation():
    from coursenav.course import PROJECT
    for name in ('navigator.py','objmap.py','pf.py','occgrid.py','return_plan.py','offline_sim.py'):
        assert not (PROJECT/'coursenav'/name).exists()
    assert sorted(x.name for x in (PROJECT/'launch').glob('*.py'))==['refnav_skybeta.launch.py']
    assert not (PROJECT/'maps/course_origin_5cm.npz').exists()
    import ast
    from refnav import core
    tree=ast.parse((PROJECT/'refnav/core.py').read_text())
    mission=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Mission')
    methods={n.name:n for n in mission.body if isinstance(n,ast.FunctionDef)}
    assert '_go_pose' not in methods and '_advance' not in methods
    assert '_go_path' in methods and '_go' not in methods
    assert core.Mission(core.Params())._path_leg()


def test_unknown_footprint_band_is_not_published_as_an_obstacle():
    from coursenav.nav_node import NavNode
    from types import SimpleNamespace as NS
    n=planner();prob=np.full((100,100),-1);prob[20:80,20:80]=10
    n.update((-2.5,-2.5),.05,prob)
    published=[]
    fake=NS(depth_nav=n,pub_grid=NS(publish=published.append),
            get_clock=lambda:NS(now=lambda:NS(to_msg=lambda:None)))
    NavNode._publish_grid(fake)
    cells=np.array(published[-1].data)
    assert not np.any(cells==100) # no measured obstacles anywhere
    assert np.any(cells==-1) and np.any(cells==0)


def test_follower_has_no_legacy_live_depth_stops():
    # Motion is limited only by the planner (measured walls + room boundary).
    for name in ('_clearance','_blocks','_geometry_cap','_obstacle','_retreat','_stuck_landing','corridor_fresh'):
        assert not hasattr(Mission, name)
    assert not hasattr(Params(), 'corridor_wait_land_s') and not hasattr(Params(), 'safe_land')
