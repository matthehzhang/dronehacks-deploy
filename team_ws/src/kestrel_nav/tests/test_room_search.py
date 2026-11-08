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
    assert not n.terminal and n.diag['commit'] == 'guide_prefix'
    assert np.linalg.norm(p.pts[-1]-(4., 0.)) < 4.-1.          # commits most of the seen space
    assert p.v[-1] == 0 and n._safe_points(p.pts)               # backup stop inside certified space
    assert np.interp(.1, p.s, p.v) > 0


def test_guide_prefix_extends_as_space_is_revealed():
    n = planner()
    prob = np.full((120, 100), -1)
    prob[:40, :] = 10                     # seen to x = 1
    n.update((-1., -2.5), .05, prob)
    first = n.plan((0., 0.), (4., 0.)).pts[-1, 0]
    prob[:80, :] = 10                     # camera sees 2 m further along the guide
    n.update((-1., -2.5), .05, prob)
    second = n.plan((.3, 0.), (4., 0.)).pts[-1, 0]
    assert second > first+1.5


def test_mixed_or_decayed_evidence_still_certified_but_return_dominated_is_not():
    n = planner()
    prob = np.full((100, 100), 10)
    prob[45:55, 45:55] = 50               # decayed / speck-filtered cells around the aircraft
    n.update((-2.5, -2.5), .05, prob)
    assert n.plan((0., 0.), (1., 0.)) is not None and n.terminal
    prob[49:52, 52:55] = 80               # several unconfirmed returns 0.15 m beside it
    n.update((-2.5, -2.5), .05, prob)
    with pytest.raises(PlanError, match='start_footprint_not_observed_clear'):
        n.plan((0., 0.), (1., 0.))


def test_body_disc_is_certified_but_does_not_override_measured_obstacles():
    n = planner(body_radius=.3)
    prob = np.full((100, 100), -1)
    n.update((-2.5, -2.5), .05, prob, position=(0., 0.))
    assert n.free[50, 50] and not n.free[50, 60]
    prob[50, 52] = 95
    n.update((-2.5, -2.5), .05, prob, position=(0., 0.))
    assert not n.free[50, 52]


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
    assert not n.terminal
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
    m = Mission(Params(track_mode='path', local_horizon=True))
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


def test_new_obstacle_in_long_free_cell_is_occupied_after_three_returns():
    from coursenav.localgrid import LocalGrid, LocalParams
    from coursenav.scan import Scan
    g = LocalGrid(LocalParams(min_cells=1))
    far = Scan(np.array([0.]), np.array([2.5]), np.array([.05]))
    for k in range(20):
        g.integrate(float(k)*.1, (0., 0., 0.), far)       # ray at 1 m observed free many times
    near = Scan(np.array([0.]), np.array([1.0]), np.array([.05]))
    for k in range(3):
        g.integrate(2.+k*.1, (0., 0., 0.), near)
    assert g.occupied().any()


def test_committed_route_survives_coverage_flicker_but_not_a_new_obstacle():
    n = planner(body_radius=.29)
    prob = np.full((100, 100), -1)
    prob[50:, :] = 10                       # everything ahead (x >= 0) observed; behind unknown
    n.update((-2.5, -2.5), .05, prob, position=(.05, 0.))
    p = n.plan((.05, 0.), (2., 0.))
    assert n.terminal and np.interp(.1, p.s, p.v) > 0   # relief: departs although behind is unseen
    prob[50:53, :] = -1                     # coverage beside the aircraft lost
    n.update((-2.5, -2.5), .05, prob, position=(.05, 0.))
    assert n.safe(np.array((.05, 0.)))      # committed route kept
    prob[78:81, 48:53] = 95                 # a measured obstacle appears on the route
    n.update((-2.5, -2.5), .05, prob, position=(.05, 0.))
    assert not n.safe(np.array((.05, 0.)))


def window_points(dist, bottom_rel, top_rel, lateral=.3, step=.03):
    """Synthetic level-frame points (x fwd, y right, z down) of a wall at dist with an opening."""
    ys = np.arange(-lateral, lateral+1e-9, step)
    zs = np.arange(-1.0, 1.0, step)                       # up from -1 to +1 relative
    pts = [(dist, y, -z) for y in ys for z in zs if z <= bottom_rel or z >= top_rel]
    pts += [(dist+2.5, y, -z) for y in ys for z in zs if bottom_rel < z < top_rel]   # seen through
    return np.array(pts)


@pytest.mark.parametrize('bottom,top', [(.30, 1.10), (.55, 1.35), (.15, .80)])
def test_vertical_target_is_midpoint_of_any_measured_opening(bottom, top):
    from refnav.vertical import opening_target
    h = .8
    t, info = opening_target(window_points(1.0, bottom-h, top-h), h, .8, .37, .24, .35, 1.5)
    assert info['kind'] == 'opening' and abs(t-(bottom+top)/2) < .05


def test_vertical_solid_wall_open_space_and_one_sided_faces_keep_cruise():
    from refnav.vertical import opening_target
    wall = window_points(1.0, 0., 0.)                      # no gap
    assert opening_target(wall, .8, .8, .37, .24, .35, 1.5)[0] is None
    floor = np.array([(x, 0., .8) for x in np.arange(.4, 2.4, .05)])   # floor only, 0.8 m below
    assert opening_target(floor, .8, .8, .37, .24, .35, 1.5)[0] is None
    # a wall cut off by the image's top edge: surface just below, nothing above (run 5 ratchet)
    cut = np.array([(1.1, y, -z) for y in np.arange(-.3, .3, .03) for z in np.arange(-.6, -.1, .03)])
    assert opening_target(cut, .8, .8, .37, .24, .35, 1.5)[0] is None
    # a band with edges above and below but nothing visible through it is not an opening
    blind = np.array([p for p in window_points(1.0, -.3, .4) if p[0] < 1.5])
    t, info = opening_target(blind, .8, .8, .37, .24, .35, 1.5)
    assert t is None and info['kind'] == 'not_an_opening'


def test_vertical_hold_until_past_the_opening_then_cruise():
    from refnav.vertical import VerticalCentring
    vc = VerticalCentring(.8, .37, .24, rate_mps=10.)
    vc.observe(0., window_points(1.2, .55-.8, 1.35-.8), .8, (0., 0.), 0.)
    assert abs(vc.step(.1, .1, .8, (0., 0.)) - .95) < .05
    vc.observe(1., np.zeros((0, 3)), .95, (.8, 0.), 0.)        # edges out of view up close
    assert abs(vc.step(1.1, .1, .95, (.8, 0.)) - .95) < .05
    assert abs(vc.step(3., 2., .95, (1.8, 0.)) - .8) < .05     # 0.4 m past: cruise


def test_altitude_is_centred_but_rate_limited_in_flight():
    m = sweeping_mission(0., (4., 0.))
    from refnav.vertical import VerticalCentring
    m.vc = VerticalCentring(.8, .37, .24, rate_mps=.25)
    m.home = [0., 0., 0., 0.]
    m.depth_t = 2.
    m.vc.observe(2., window_points(1.0, .55-.7, 1.35-.7), .7, (0., 0.), 0.)
    m.step(2.0, m.last_v)
    m.depth_t = 2.4
    m.step(2.4, m.last_v)
    assert .7 < -m.sp[2] <= .7+.25*.5+1e-6    # two 0.25 s ticks toward 0.95 at <= 0.25 m/s


def test_follower_flies_a_route_closed_loop_through_step():
    """Kinematic closed loop: step() commands, a point-mass follows them; the route is completed."""
    from refnav.core import Mission, Params, Vehicle
    m = Mission(Params(track_mode='path', local_horizon=True, speed_mps=.5, accel_mps2=.5, brake_mps2=.4,
                       arrive_m=.15, hold_s=.25))
    m.phase = 'turn'; m.leg = 0; m.targets = [(2., 1.)]; m.waypoints_ne = [(2., 1.)]
    m.sp = [0., 0., -.7, 0.]; m.home = [0., 0., 0., 0.]; m.leg_heading = 0.
    m.t_started = 0.; m.t_phase = 0.; m.t_last = 0.
    m.health.bad_for = lambda now: 0.; m.offboard_seen = False
    pos, hdg, t = np.array([0., 0.]), 0., 0.
    pts = [(0., 0.), (1., 0.), (2., 1.)]
    s = [0., 1., 1.+math.sqrt(2)]
    v = [0., .4, 0.]
    dense = np.array([np.interp(np.linspace(0, s[-1], 40), s, np.array(pts)[:, k]) for k in (0, 1)]).T
    vd = np.interp(np.linspace(0, s[-1], 40), s, v); vd[1:-1] = np.maximum(vd[1:-1], .2)
    phases = set()
    for k in range(1200):
        t += .05
        m.depth_t = t
        m.on_path(t, 0, dense, vd, pid=1, terminal=True)
        veh = Vehicle(pos=(pos[0], pos[1], -.7), heading=hdg, pos_ok=True, vel=(0., 0., 0.))
        o = m.step(t, veh)
        phases.add(m.phase)
        hdg = m.sp[3]
        if m.v_vec is not None and m.phase == 'go':
            pos = pos + np.array(m.v_vec)*.05 + .3*(np.array(m.sp[:2])-pos)*.05
        if m.phase in ('land', 'done'):
            break
    assert 'go' in phases and m.legs_done == 1 and m.phase in ('land', 'done')
    assert np.linalg.norm(pos-(2., 1.)) < .2
