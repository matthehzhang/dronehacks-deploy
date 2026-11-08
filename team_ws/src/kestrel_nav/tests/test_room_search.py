"""Room boundary as a hard non-depth constraint, and waypoint-directed search."""
import heapq
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from coursenav.depth_planner import DepthPlanner, grid_distances
from coursenav.planner import PlanError
from coursenav.room import RoomBoundary
from coursenav.smooth import SmoothParams
from refnav.core import Mission, Params, Vehicle, _wrap

NEED = .4


def planner(room=None, **kw):
    return DepthPlanner(NEED, SmoothParams(need_m=NEED, reach_m=.28, tube_response_s=.5, center_stops=False),
                        room=room, **kw)


def rect(x0, y0, x1, y1):
    return RoomBoundary([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def test_room_clearance_sign_and_published_course_file():
    r = rect(-1, -1, 1, 1)
    assert np.allclose(r.clearance(np.array([[0., 0.], [.9, 0.], [1.5, 0.]])), [1., .1, -.5])
    from coursenav.course import PROJECT, load_course, resolve
    doc = load_course(PROJECT/'course/course_depth_waypoints.json')
    room = RoomBoundary.load(resolve(doc, 'room_boundary_file'))
    assert np.allclose(np.abs(room.poly), 3.048)
    need = doc['vehicle']['radius_m']+doc['vehicle']['margin_m']
    # Every waypoint and the declared launch must satisfy the hard boundary.
    import json
    intent = json.loads(resolve(doc, 'waypoint_file').read_text())
    for xy in [doc['start']['xy']]+[w['xy'] for w in intent['waypoints']]:
        assert room.clearance(np.array(xy)) >= need+.05, xy


def test_depth_seeing_through_nets_cannot_route_outside_room():
    # Depth reports free space far beyond the (invisible) net at y=1.5.
    room = rect(-2., -1.5, 2., 1.5)
    prob = np.full((100, 100), 10)          # origin (-2.5,-2.5), 5 m square, all observed free
    prob[49:51, :68] = 95                    # measured wall x=0 from y=-2.5 up to y=0.9
    n = planner(room)
    n.update((-2.5, -2.5), .05, prob)
    # Without the room the 0.6 m gap plus free space beyond y=1.5 would be a route.
    free_world = planner()
    free_world.update((-2.5, -2.5), .05, prob)
    free_world.plan((-1., 0.), (1., 0.))
    assert free_world.terminal and free_world.path.pts[:, 1].max() > 1.5-NEED
    try:
        p = n.plan((-1., 0.), (1., 0.))
    except PlanError:
        return
    assert not n.terminal
    assert np.all(room.clearance(p.pts) >= NEED-1e-6)


def test_room_overrides_launch_disc_and_start_violation_is_explicit():
    room = rect(-2., -2., 2., 2.)
    n = planner(room, launch_xy=(1.8, 0.), launch_radius=1.)
    n.update((-2.5, -2.5), .05, np.full((100, 100), -1))
    with pytest.raises(PlanError, match='start_violates_room_boundary_clearance'):
        n.plan((1.8, 0.), (0., 0.))
    with pytest.raises(PlanError, match='waypoint_violates_room_boundary'):
        n.plan((1.4, 0.), (1.9, 0.))
    p = n.plan((1.4, 0.), (1.4, .15))
    assert n.terminal and np.all(room.clearance(p.pts) >= NEED-1e-6)


def test_room_keepout_is_published_distinct_from_measured_obstacles():
    from coursenav.nav_node import NavNode
    n = planner(rect(-2., -2., 2., 2.))
    n.update((-2.5, -2.5), .05, np.full((100, 100), 10))
    published = []
    fake = NS(depth_nav=n, pub_grid=NS(publish=published.append),
              get_clock=lambda: NS(now=lambda: NS(to_msg=lambda: None)))
    NavNode._publish_grid(fake)
    cells = np.array(published[-1].data)
    assert np.any(cells == 99) and not np.any(cells == 100)
    prob = np.full((100, 100), 10)
    prob[50, 50] = 95
    n.update((-2.5, -2.5), .05, prob)
    NavNode._publish_grid(fake)
    cells = np.array(published[-1].data)
    assert np.any(cells == 99) and np.any(cells == 100)


def test_observation_pose_advances_toward_waypoint():
    n = planner()
    prob = np.full((120, 100), -1)
    prob[:60, :] = 10                     # observed free up to x = 2
    n.update((-1., -2.5), .05, prob)
    p = n.plan((0., 0.), (4., 0.))
    assert not n.terminal
    assert n.diag['progress_m'] >= .15
    assert np.linalg.norm(p.pts[-1]-(4., 0.)) < 4.-.15


def test_detour_around_measured_wall_counts_as_progress_but_unknown_is_never_flown():
    # Wall ahead with an observed opening to the north; straight-line progress
    # would refuse the detour, the optimistic cost-to-go accepts it.
    n = planner()
    prob = np.full((140, 140), -1)
    prob[:, :] = 10
    prob[70:, :] = -1                     # x >= 1.5 unobserved
    prob[50:53, :95] = 95                  # wall at x~0.6, y < 1.25
    n.update((-2., -3.5), .05, prob)
    p = n.plan((0., -1.), (3., -1.))
    assert not n.terminal and n.diag['progress_m'] > 0
    assert p.pts[:, 1].max() > 1.2         # goes around the wall's open end
    assert n._safe_points(p.pts)           # only certified cells


def test_frontier_outside_room_is_not_a_search_target():
    room = rect(-2., -2., 2., 2.)
    n = planner(room)
    prob = np.full((120, 120), -1)         # origin (-3,-3)
    prob[10:110, 10:110] = 10              # the whole room observed free; outside unknown
    n.update((-3., -3.), .05, prob)
    edge = n._frontier_distance()
    inside = n._room_cells > NEED
    assert np.all(edge[inside] > NEED+n.frontier_band) or not np.isfinite(edge[inside]).any()


def test_csgraph_reachability_matches_python_dijkstra():
    rng = np.random.default_rng(3)
    occ = rng.random((40, 37)) < .3
    occ[5, 5] = False
    d = grid_distances(~occ, .05, [((5, 5), 0.)])
    ref = np.full(occ.shape, math.inf)
    ref[5, 5] = 0.
    q = [(0., 5, 5)]
    while q:
        c, i, j = heapq.heappop(q)
        if c > ref[i, j]:
            continue
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            a, b = i+di, j+dj
            if not (0 <= a < 40 and 0 <= b < 37) or occ[a, b] or (di and dj and (occ[i, b] or occ[a, j])):
                continue
            nc = c+.05*math.hypot(di, dj)
            if nc < ref[a, b]:
                ref[a, b] = nc
                heapq.heappush(q, (nc, a, b))
    assert np.array_equal(np.isfinite(d), np.isfinite(ref))
    assert np.allclose(d[np.isfinite(d)], ref[np.isfinite(ref)], atol=1e-6)


def sweeping_mission(heading, waypoint_ne):
    m = Mission(Params(track_mode='path', local_horizon=True, corridor_search=True))
    m.phase = 'go'; m.leg = 0; m.targets = [(9., 9.)]; m.waypoints_ne = [waypoint_ne]
    m.sp = [0., 0., -.7, heading]; m.home = [0., 0., 0., 0.]
    m.leg_dir = (1., 0.); m.leg_len = 4.; m.leg_heading = heading
    m.t_started = 0.; m.t_phase = 0.; m.t_last = 1.; m.t_check = 1.
    m.health.bad_for = lambda now: 0.; m.offboard_seen = False
    m.last_v = Vehicle(pos=(0., 0., -.7), heading=heading, pos_ok=True, vel=(0., 0., 0.))
    return m


@pytest.mark.parametrize('heading', [0., math.pi, -2.5])
def test_search_is_bounded_sweep_centred_on_waypoint_bearing(heading):
    m = sweeping_mission(heading, (0., 4.))   # waypoint due east: bearing +90 deg in NED
    t, dwell, entered = 2., set(), False
    for _ in range(4000):
        t += .05
        m.depth_t = t
        v = Vehicle(pos=(0., 0., -.7), heading=m.sp[3], pos_ok=True, vel=(0., 0., 0.))
        m.step(t, v)
        assert m.phase == 'search' and m.v_cmd == 0 and m.sp[:2] == [0., 0.]
        assert math.isclose(m.search_center, math.pi/2)
        in_sector = abs(_wrap(m.sp[3]-math.pi/2)) <= math.radians(60.)+1e-6
        assert in_sector or not entered     # once inside the sector it never leaves
        entered |= in_sector
        if m.search_settled is not None:
            dwell.add(round(math.degrees(_wrap(m.search_target-math.pi/2))))
    assert dwell == {0, 30, 60, -30, -60}
    assert m.search_cycles >= 2           # repeats the sector, not a full revolution


def test_waypoint_goal_ne_feeds_sweep_not_horizon_endpoint():
    m = sweeping_mission(0., (4., 0.))
    m.on_path(2., 0, [(0., 0.), (0., -.5)], [.2, 0.], pid=1, terminal=False)
    assert m.targets[0] == (0., -.5) and m.waypoints_ne[0] == (4., 0.)
    m.path = None
    m.depth_t = 2.
    m.step(2., m.last_v)
    assert m.phase == 'search' and math.isclose(m.search_center, 0.)
