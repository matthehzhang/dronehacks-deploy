"""Real-bag regression: poor prior fit must not disable relative depth mapping."""
import math
from types import SimpleNamespace as NS
import numpy as np
import pytest
from coursenav.localgrid import LocalGrid, LocalParams, course_mask
from coursenav.scan import Scan
from coursenav.course import Grid2D
from coursenav.navigator import Navigator


def scan_wall(x=1.):
    yy=np.linspace(-.5,.5,41)
    return Scan(np.arctan2(yy,x),np.hypot(x,yy),np.full(len(yy),.05))


def test_local_grid_builds_without_any_pf_covariance_or_match_score():
    g=LocalGrid()
    for t in (1.,1.1,1.2,1.3):g.integrate(t,(0.,0.,0.),scan_wall())
    assert g.stats()['known']>0 and g.stats()['occupied']>0
    assert np.allclose(g.obstacle_points()[:,0],1.,atol=.06)
    assert np.any(g.grid_fields()[-1] == -1)  # unobserved remains unknown


def test_local_hits_reach_planner_without_a_confident_prior_live_grid():
    prior=Grid2D(np.zeros((100,100),bool),np.array((-2.5,-2.5)),.05)
    nav=Navigator(prior,prior,np.full(prior.occ.shape,5.),.414082845,live=None)
    nav.path=NS(pts=np.array([[-1.,0.],[0.,0.],[1.,0.]]),s=np.array([0.,1.,2.]),clearance=np.full(3,5.))
    # New obstacle on route, despite the other live map having no cells.
    mask=course_mask([[0.,0.]],prior)
    nav.set_local(mask)
    assert nav.map_changed() and not nav.path_safe()
    assert nav.fuse()[0].occ[tuple(prior.ij((0.,0.)))]
    old=mask.copy();mask[:]=False;mask[tuple(prior.ij((0.,1.)))]=True
    assert old.sum()==mask.sum()
    nav.set_local(mask)
    assert nav.map_changed() and nav.path_safe()  # equal count, changed position


def test_prior_is_retained_and_near_prior_local_obstacle_is_not_filtered():
    grid=Grid2D(np.zeros((60,60),bool),np.array((-1.5,-1.5)),.05)
    grid.occ[tuple(grid.ij((.5,0.)))]=True
    nav=Navigator(grid,grid,np.full(grid.occ.shape,5.),.414082845,live=None)
    nav.set_local(course_mask([[.4,0.]],grid))
    assert nav.fuse()[0].occ.sum()==2


def test_ros_local_integrator_ignores_bad_pf_fit_but_rejects_stale_inputs():
    pytest.importorskip('rclpy')
    from coursenav.loc_node import LocNode
    import time
    yy=np.repeat(np.linspace(-.4,.4,40),5)
    points=np.column_stack([np.ones(len(yy)),yy,np.zeros(len(yy))])
    node=NS(local=LocalGrid(),nominal=np.array((0.,0.,0.)),local_stamp=None,local_height=None,
            local_error='',att_received=10.,height=(time.monotonic(),.8),vio_ok=True,
            get_clock=lambda:NS(now=lambda:NS(nanoseconds=10_100_000_000)))
    # There deliberately is no PF/covariance field on this fixture.
    assert LocNode._integrate_local(node,10.,np.zeros(3),points,.8)
    assert node.local.stats()['known']>0 and not node.local_error
    assert not LocNode._integrate_local(node,10.,np.zeros(3),points,.8)  # duplicate
    node.att_received=9.
    assert not LocNode._integrate_local(node,10.05,np.zeros(3),points,.8)
    node.att_received=10.;node.vio_ok=False
    assert not LocNode._integrate_local(node,10.05,np.zeros(3),points,.8)


def test_ground_returns_are_not_inserted_as_takeoff_walls():
    pytest.importorskip('rclpy')
    from coursenav.loc_node import LocNode
    import time
    yy=np.repeat(np.linspace(-.4,.4,40),5)
    floor=np.column_stack([np.ones(len(yy)),yy,np.full(len(yy),-.1)])
    node=NS(local=LocalGrid(),nominal=np.zeros(3),local_stamp=None,local_height=None,
            local_error='',att_received=10.,height=(time.monotonic(),.1),vio_ok=True,
            get_clock=lambda:NS(now=lambda:NS(nanoseconds=10_100_000_000)))
    assert LocNode._integrate_local(node,10.,np.zeros(3),floor,.1)
    assert node.local.stats()['occupied']==0 and node.local.stats()['known']==0


def test_nav_adapter_refuses_stale_and_wrong_frame_local_grids():
    pytest.importorskip('rclpy')
    from coursenav.nav_node import NavNode
    node=NS(local_pts=np.zeros((0,2)),local_stamp=None,local_error='',
            get_clock=lambda:NS(now=lambda:NS(nanoseconds=10_100_000_000)))
    msg=NS(header=NS(stamp=NS(sec=10,nanosec=0),frame_id='coursenav_nominal'),
           info=NS(origin=NS(position=NS(x=0.,y=0.)),resolution=.05,width=2,height=2),data=[-1,0,100,0])
    NavNode._on_local(node,msg)
    assert len(node.local_pts)==1 and node.local_stamp==10.
    msg.header.frame_id='course'
    NavNode._on_local(node,msg)
    assert node.local_error and node.local_stamp==10.
    msg.header.frame_id='coursenav_nominal';msg.header.stamp.sec=9
    NavNode._on_local(node,msg)
    assert node.local_error and node.local_stamp==10.


def test_real_ros_localizer_constructs_and_publishes_separate_unknown_local_grid(tmp_path):
    rclpy=pytest.importorskip('rclpy')
    from coursenav.loc_node import LocNode
    owns=not rclpy.ok()
    if owns:rclpy.init(args=[])
    node=None
    try:
        # Constructor uses the real active map, publisher and message types.
        node=LocNode.__new__(LocNode)
        LocNode.__init__(node)
        messages=[]
        node.pub_local=NS(publish=messages.append)
        node.local.integrate(10.,(0.,0.,0.),scan_wall())
        node.local_stamp=10.
        node._publish_local()
        assert messages[-1].header.frame_id=='coursenav_nominal'
        assert messages[-1].header.stamp.sec==10
        assert -1 in messages[-1].data
        assert node.pf.p.min_fit_frac==.35
    finally:
        if node is not None:node.destroy_node()
        if owns:rclpy.shutdown()


def test_obstacle_behind_progress_does_not_discard_safe_remaining_route():
    prior=Grid2D(np.zeros((100,100),bool),np.array((-2.5,-2.5)),.05)
    nav=Navigator(prior,prior,np.full(prior.occ.shape,5.),.414082845,live=None)
    nav.path=NS(pts=np.array([[-1.,0.],[0.,0.],[1.,0.]]),s=np.array([0.,1.,2.]),clearance=np.full(3,5.))
    nav.set_local(course_mask([[-1.,0.]],prior))
    assert nav.path_safe([0.,0.])
    nav.set_local(course_mask([[1.,0.]],prior))
    assert not nav.path_safe([0.,0.])


def test_real_navigator_does_not_fuse_legacy_map(tmp_path):
    import json
    rclpy=pytest.importorskip('rclpy')
    from coursenav.nav_node import NavNode
    route=tmp_path/'route.json';route.write_text(json.dumps({'nav_legs':[],'params':{}}))
    owns=not rclpy.ok()
    if owns:rclpy.init(args=['--ros-args','-p',f'route_file:={route}'])
    node=None
    try:
        node=NavNode()
        assert node.nav.live is None
        before=node.nav.fuse()[1].occ.copy()
        node._on_occ(NS(data=[100]*4))
        assert np.array_equal(before,node.nav.fuse()[1].occ)
        xy=node.nav.prior_obst.xy(np.argwhere(~before)[0])
        node.nav.set_local(course_mask([xy],node.nav.prior_obst))
        assert not np.array_equal(before,node.nav.fuse()[1].occ)
    finally:
        if node is not None:node.destroy_node()
        if owns:rclpy.shutdown()
