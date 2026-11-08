"""Recovery creep (recovery_creep): short certified observed-free hops, flown slowly, then look again.

Sensing side: dense FREE-ONLY rays, a separate free-carving range and the airframe's own (swept) disc give the
local grid the evidence a hop needs, without inventing obstacles or free space. Planning side: a hop is a short
path entirely inside certified space that gets closer to the leg goal along the fused map. Control side: refnav
flies it at creep speed with the live-depth stop still on, never treats it as the leg, and is bounded by hops,
stalls and an absolute time budget."""
import json
import math
from dataclasses import replace

import numpy as np
import pytest

from coursenav.course import Grid2D
from coursenav.localgrid import LocalGrid, LocalParams
from coursenav.observed_free import ObservedFree
from coursenav.planner import PlanError
from coursenav.scan import Scan, local_scan_from_points, with_dense_free_rays
from refnav import core as C
from test_nav_recovery_protocol import node, request  # noqa: F401  (fixture)
from test_observed_recovery import scene
from test_recovery_controller import step
from test_refnav_blocked import STRAIGHT, WALL, dense, fly_path, params, speeds


# ------------------------------------------------------------------------------------------------ sensing
def wall_points(dist, half_deg=37.9, n=400, gap=None):
    """A flat surface `dist` m ahead across the field of view, at body height (+-0.2 m)."""
    b = np.radians(np.linspace(-half_deg, half_deg, n))
    if gap is not None:
        b = b[(np.degrees(b) < gap[0]) | (np.degrees(b) > gap[1])]
    z = np.linspace(-0.2, 0.2, 9)
    B, Z = np.meshgrid(b, z)
    r = dist / np.cos(B)
    return np.stack([r * np.cos(B), r * np.sin(B), Z], -1).reshape(-1, 3)


def integrate(g, pts, n=4, dense_bins=0, pose=(0., 0., 0.), t0=0.):
    t = t0
    for _ in range(n):
        s = local_scan_from_points(pts, .8, .55, 1.05, obstacle_range_m=g.p.max_range_m,
                                   camera_origin_xy=np.array([.1, 0.]))
        if dense_bins:
            s = with_dense_free_rays(s, pts, .8, .55, 1.05, n_bins=dense_bins, obstacle_range_m=g.p.max_range_m,
                                     camera_origin_xy=np.array([.1, 0.]))
        g.integrate(t, pose, s)
        t += .25
    return t


def test_dense_free_rays_fill_the_gaps_but_never_mark_obstacles():
    sparse, rich = LocalGrid(), LocalGrid()
    integrate(sparse, wall_points(2.5))
    t = integrate(rich, wall_points(2.5), dense_bins=160)

    def certifiable(g):     # cells a recovery path may use: clearance >= the vehicle buffer (0.414 m)
        o, res, w, h, data = g.grid_fields(now=t)
        return (ObservedFree.from_grid(o, res, w, h, data, t).clearance >= .414 + .05 * math.sqrt(2) / 2).sum()
    assert certifiable(rich) > certifiable(sparse) * 1.5
    # obstacle marking is exactly the coarse scan's: dense rays add no occupied cell
    assert np.array_equal(rich.occupied(), sparse.occupied())
    assert (rich.L > 0).sum() == (sparse.L > 0).sum()


def test_dense_bin_with_too_few_near_points_gives_no_ray_and_vetoes_far_carving():
    far = wall_points(6.0)
    lone = np.array([[1.0, 0.001, 0.0], [1.0, 0.002, 0.0]])        # 2 near points: too few for a return
    pts = np.vstack([far, lone])
    s = with_dense_free_rays(Scan.empty(), pts, .8, .55, 1.05, n_bins=160, camera_origin_xy=np.array([.1, 0.]))
    edges = np.linspace(-math.radians(38), math.radians(38), 161)
    k = int(np.digitize(math.atan2(.0015, 1.0), edges) - 1)        # the lone points' bin
    in_bin = (s.bearing >= edges[k]) & (s.bearing < edges[k + 1])
    assert not in_bin.any()                                         # no near ray, and the far ray is vetoed
    assert ((s.bearing >= edges[k - 1]) & (s.bearing < edges[k])).any()   # the neighbour still carves
    assert s.free_only.all()


def test_free_only_flag_must_match_the_beams():
    g = LocalGrid()
    bad = Scan(np.zeros(2), np.ones(2), np.ones(2) * .05, None, np.zeros(3, bool))
    with pytest.raises(ValueError):
        g.integrate(0., (0., 0., 0.), bad)


def test_free_range_carves_further_without_marking_beyond_the_obstacle_range():
    short, long_ = LocalGrid(), LocalGrid(LocalParams(free_range_m=4.0))
    integrate(short, wall_points(6.0), dense_bins=160)
    integrate(long_, wall_points(6.0), dense_bins=160)
    reach = lambda g: np.hypot(*(g.cell_centres(np.argwhere((g.L < 0) & (g.seen > 0))) ).T).max()
    assert reach(short) <= 3.0 + 1e-6 < reach(long_) <= 4.0 + 1e-6
    assert not (long_.L > 0).any()                                  # a 6 m wall marks nothing


def test_own_disc_is_published_free_for_the_swept_age_then_expires():
    g = LocalGrid(LocalParams(body_radius_m=.29, swept_age_s=20.))
    g.integrate(0., (1., 1., 0.), Scan.empty())
    centre = g.ij([1., 1.])[0]
    prob = lambda now: np.asarray(g.grid_fields(now=now)[4], np.int16).reshape(g.n, g.n).T
    assert prob(5.)[tuple(centre)] == 0
    edge = g.ij([1.32, 1.])[0]                                      # outside the 0.29 m disc: still unknown
    assert prob(5.)[tuple(edge)] == -1
    assert prob(20.5)[tuple(centre)] == -1                          # swept evidence expired


def test_confident_obstacle_and_later_hits_override_swept_free():
    g = LocalGrid(LocalParams(body_radius_m=.29, swept_age_s=20.))
    g.integrate(0., (1., 1., 0.), Scan.empty())
    i, j = g.ij([1., 1.])[0]
    g.L[i, j] = g.p.occ_thresh + .5                                 # confident obstacle where we were: wins
    assert np.asarray(g.grid_fields(now=1.)[4], np.int16).reshape(g.n, g.n).T[i, j] != 0
    g = LocalGrid(LocalParams(body_radius_m=.29, swept_age_s=20.))
    g.integrate(0., (1., 1., 0.), Scan.empty())
    # a later return lands on the swept cell (seen from 1 m west): the swept evidence is revoked
    g.integrate(1., (0., 1., 0.), Scan(np.array([0.]), np.array([1.]), np.array([.05])))
    i, j = g.ij([1., 1.])[0]
    assert not math.isfinite(g.last_body[i, j])


def test_library_defaults_keep_the_previous_behaviour():
    p = LocalParams()
    assert p.free_range_m == 0 and p.body_radius_m == 0 and p.swept_age_s == 0
    g = LocalGrid()
    g.integrate(0., (1., 1., 0.), Scan.empty())
    assert not np.isfinite(g.last_body).any()


# ------------------------------------------------------------------------------------------------ planning
def strip(lo_i=8, hi_i=45, lo_j=30, hi_j=70, value=12):
    prob = np.full((160, 100), -1)
    prob[lo_i:hi_i, lo_j:hi_j] = value
    return ObservedFree.from_grid([0, 0], .05, 160, 100, prob.T.ravel(), 10.)


def test_hop_is_short_certified_and_makes_progress():
    nav, _ = scene()
    g = strip()
    path, before, after = nav.plan_creep_hop([1., 2.5], [6., 2.5], g, lambda p: p)
    assert np.linalg.norm(path.pts[-1] - path.pts[0]) <= nav.p.hop_max_m + 1e-6
    assert g.certifies_path(path.pts, lambda p: p, nav.need)
    assert before - after >= nav.p.hop_min_progress_m
    assert nav.plan_mode == "creep" and path.v[-1] == 0.


def test_hop_never_starts_or_ends_in_unknown_space():
    nav, _ = scene()
    with pytest.raises(PlanError, match="start footprint"):
        nav.plan_creep_hop([1., 2.5], [6., 2.5], strip(lo_i=15), lambda p: p)   # free starts 0.25 m ahead
    nav, _ = scene()
    with pytest.raises(PlanError):
        # only the aircraft's own surroundings are free: nowhere certified to go
        nav.plan_creep_hop([1., 2.5], [6., 2.5], strip(lo_i=8, hi_i=27, lo_j=38, hi_j=63), lambda p: p)


def test_hop_must_get_closer_to_the_goal_along_the_map():
    nav, _ = scene()
    # free space only BEHIND the aircraft (away from the goal)
    with pytest.raises(PlanError, match="closer"):
        nav.plan_creep_hop([2., 2.5], [6., 2.5], strip(lo_i=15, hi_i=50), lambda p: p)


def test_hop_cannot_cross_a_prior_wall_even_if_observed_free():
    nav, _ = scene()
    nav.prior_blocked.occ[25:27, :] = True                          # a mapped wall 0.25 m ahead, full width
    nav._fused = (nav.prior_obst, nav.prior_blocked, nav._effective(nav.prior_clear, nav.prior_blocked))
    nav._fused_signature = nav._signature()
    with pytest.raises(PlanError):
        nav.plan_creep_hop([1., 2.5], [6., 2.5], strip(), lambda p: p)


def test_cost_to_go_goes_around_walls():
    nav, _ = scene()
    blocked = Grid2D(nav.prior_blocked.occ.copy(), nav.prior_blocked.origin, nav.prior_blocked.res)
    blocked.occ[60:62, 0:80] = True                                 # wall with a gap at the far side
    cost = nav.cost_to_go([6., 2.5], blocked)
    c = cost[tuple(blocked.ij([1., 2.5]))]
    assert math.isfinite(c) and c > 5.0 + .5                        # longer than the 5 m straight line


# ------------------------------------------------------------------------------------------------ nav node
def creep_request(n, rid, hops=0, creep=True):
    n._on_status(type("M", (), {"data": json.dumps({"leg": 0, "phase": "blocked", "stop_m": .95,
        "recovery_request": {"id": rid, "leg": 0, "observed_only": True, "creep": creep, "creep_hops": hops}})})())


def test_node_answers_a_creep_request_with_a_hop_only_when_enabled(node):
    node.observed_free = strip()
    creep_request(node, 1)
    node._tick()
    out = node.messages[-1]
    assert out["plan_state"] == "failed"                            # recovery_creep off: unchanged behaviour
    node.recovery_creep = True
    creep_request(node, 2)
    node.now = 11.
    node.observed_free = strip()
    node.observed_free.stamp = 11.
    node._tick()
    out = node.messages[-1]
    assert out["plan_state"] == "ready" and out["recovery_hop"] is True and out["recovery_certified"] is True
    assert out["pts_ne"] and out["hop_progress_m"] >= .1
    assert node.pid not in getattr(node, "reference_history", {})   # a hop never becomes a reference route
    assert node.reference_path is not node.nav.path


def test_node_does_not_hop_when_refnav_did_not_ask(node):
    node.recovery_creep = True
    node.observed_free = strip()
    creep_request(node, 1, creep=False)
    node._tick()
    assert node.messages[-1]["plan_state"] == "failed"


def test_hop_in_flight_survives_map_changes_but_not_lost_certification(node):
    node.recovery_creep = True
    node.observed_free = strip()
    creep_request(node, 1)
    node._tick()
    assert node.messages[-1]["plan_state"] == "ready"
    node.nav.map_changed = lambda: True                             # the camera saw something new elsewhere
    node._tick()
    assert node.messages[-1]["plan_state"] == "ready"
    g = strip(hi_i=12)                                              # the hop's space is no longer observed
    g.stamp = 10.
    node.observed_free = g
    node._tick()
    assert node.messages[-1]["plan_state"] == "failed"
    assert "creep hop lost" in node.messages[-1]["reason"]


# ------------------------------------------------------------------------------------------------ refnav
CREEP = dict(recovery_replan=True, recovery_scan_deg=25., recovery_creep=True)


def blocked_creep(**kw):
    m = C.Mission(params(**{**CREEP, **kw}))
    m.phase = 'go'; m.targets = [(4., 0.)]; m.home = (0., 0., 0., 0.)
    m.sp = [1., 0., -1., 0.]; m.leg_start = (0., 0.); m.leg_dir = (1., 0.); m.leg_len = 4.
    m.offboard_seen = True; m.t_last = 10.; m.t_started = 0.; m.t_phase = 9.
    v = C.Vehicle(pos=(1., 0., -1.), heading=0., pos_ok=True, armed=True,
                  nav_state=C.NAV_OFFBOARD, vel=(0., 0., 0.), landed=False)
    m.on_path(10., 0, STRAIGHT, speeds(STRAIGHT), pid=1); m.path_s = 1.
    m.depth_t = 10.; m.depth_pts = np.zeros((0, 3))
    m.health.on_msg(10., True)
    m._start_recover(10., C.Output('go'), 'blocked', 'obstacle', v)
    return m, v


def hop_pts(a, b):
    pts, _ = dense([a, b], .05)
    return pts


def give_hop(m, t, pts, pid=7):
    rid = m.replan_request['id']
    return m.on_path(t, 0, pts, speeds(pts), pid=pid, request_id=rid, recovery_certified=True, hop=True)


def test_creep_requests_say_so_and_hops_never_change_the_leg():
    m, v = blocked_creep()
    assert m.replan_request['creep'] is True and m.replan_request['creep_hops'] == 0
    assert give_hop(m, 10.1, hop_pts((1., 0.), (1.3, .35)))
    assert m.targets == [(4., 0.)] and m.leg_len == 4.
    assert m.plan_reference[0]['pid'] == 1


def test_hop_rejected_without_recovery_creep_and_outside_recovery():
    m = C.Mission(params(recovery_replan=True, recovery_scan_deg=25.))
    m.targets = [(4., 0.)]
    assert not m.on_path(10., 0, STRAIGHT, speeds(STRAIGHT), pid=2, hop=True)
    m, v = blocked_creep()
    m.p.recovery_creep = False
    assert not give_hop(m, 10.1, hop_pts((1., 0.), (1.3, .35)))


def fly_hop(m, v, t0, until, track=True):
    """Step the mission with a vehicle that follows its setpoint (heading too) exactly."""
    t = t0
    hop = m.path
    while t < until:
        t = round(t + .05, 3)
        if m.path is None and hop is not None and m.phase == 'creep':
            pass
        if m.recover is not None and hop is not None and m.phase == 'creep':
            give_hop(m, t, hop['pts'], pid=hop['pid'])                 # the navigator keeps it fresh
        out = step(m, v, t)
        if track and out.setpoint is not None:
            v = replace(v, pos=tuple(out.setpoint[:3]), heading=out.setpoint[3])
        m.corr_t = t
        if m.phase != 'creep':
            break
    return t, v, out


def test_a_hop_turns_in_place_then_creeps_no_faster_than_creep_speed_then_looks_again():
    m, v = blocked_creep()
    end = (1.3, .4)
    give_hop(m, 10.1, hop_pts((1., 0.), end))
    out = step(m, v, 10.15)
    assert m.phase == 'creep' and m.recover['hop']['stage'] == 'turn'
    assert out.setpoint[:2] == (1., 0.)                              # no translation while turning
    t, v, out = 10.15, v, out
    last = np.array(v.pos[:2])
    hop = m.path
    while m.phase == 'creep' and t < 25.:
        t = round(t + .05, 3)
        give_hop(m, t, hop['pts'], pid=hop['pid'])
        m.depth_t = t
        m.clr_hist = [(t, math.inf)] * 3; m.corr_t = t               # live depth: clear ahead
        out = m.step(t, v)
        m.health.on_msg(t, True)
        new = np.array(out.setpoint[:2])
        assert np.linalg.norm(new - last) <= m.p.creep_speed_mps * .05 + 1e-6 + (1e-9 if m.phase == 'creep' else 1.)
        v = replace(v, pos=tuple(out.setpoint[:3]), heading=out.setpoint[3])
        last = new
    assert m.phase == 'blocked' and m.recover['hops'] == 1 and m.recover['stalls'] == 0
    assert math.dist(v.pos[:2], end) < .15
    assert m.counts['creep_hops'] == 1
    # the hop's request is consumed: the navigator re-publishing it cannot restart it
    assert not give_hop(m, t + .05, hop['pts'], pid=hop['pid'])
    # dwell, then ONE new request that reports the hop count
    rid = m.replan_request['id']
    for k in range(1, 30):
        step(m, v, t + .05 * k)
    assert m.replan_request['id'] == rid + 1 and m.replan_request['creep_hops'] == 1


def test_obstacle_on_a_hop_stops_it_and_two_stalls_land():
    m, v = blocked_creep()
    for n in range(2):
        hop = hop_pts(v.pos[:2], (v.pos[0] + .3, v.pos[1] + .3))
        if n:
            for k in range(1, 40):                                    # dwell -> new request
                step(m, v, 10. + 3 * n + .05 * k)
        give_hop(m, 10. + 3 * n + 2.1, hop, pid=10 + n)
        t = 10. + 3 * n + 2.15
        step(m, v, t)
        assert m.phase == 'creep'
        m.recover['hop']['stage'] = 'move'
        m.depth_t = t + .05; m.clr_hist = [(t, .3)] * 3; m.corr_t = t + .05   # obstacle 0.3 m ahead
        give_hop(m, t + .05, hop, pid=10 + n)
        step(m, v, t + .05)
    assert m.phase == 'land' and 'creep made no progress' in m.why
    assert m.counts['creep_stalls'] == 2


def test_withdrawn_or_stale_hop_ends_without_progress():
    m, v = blocked_creep()
    give_hop(m, 10.1, hop_pts((1., 0.), (1.3, .4)))
    step(m, v, 10.15)
    assert m.on_plan_result(10.2, 0, m.replan_request['id'], 'failed', 'creep hop lost fresh observed-free certification')
    step(m, v, 10.25)
    assert m.phase == 'blocked' and m.recover['stalls'] == 1


def test_unobserved_corridor_never_advances_and_ends_the_hop():
    m, v = blocked_creep()
    give_hop(m, 10.1, hop_pts((1., 0.), (1.3, .4)))
    step(m, v, 10.15)
    m.recover['hop']['stage'] = 'move'
    hop = m.path
    t, sp0 = 10.15, tuple(m.sp[:2])
    while m.phase == 'creep' and t < 14.:
        t = round(t + .05, 3)
        give_hop(m, t, hop['pts'], pid=hop['pid'])
        m.corr_t = None                                              # nothing observed ahead
        step(m, v, t)
        if m.phase == 'creep':
            assert tuple(m.sp[:2]) == sp0
    assert m.phase == 'blocked' and m.recover['stalls'] == 1


def test_absolute_budget_lands_however_much_progress():
    m, v = blocked_creep(creep_max_s=12.)
    m.recover['deadline'] = 1e9                                      # as if every hop had made progress
    out = step(m, v, 22.1)
    assert m.phase == 'land' and 'timed out' in m.why


def test_after_the_last_hop_only_full_detours_are_requested():
    m, v = blocked_creep(creep_max_hops=1)
    m.recover['hops'] = 1
    m._request_recovery_plan(11.)
    assert m.replan_request['creep'] is False


def test_creep_bounds_are_validated():
    with pytest.raises(ValueError):
        C.Mission(params(**{**CREEP, 'creep_speed_mps': 1.0}))
    with pytest.raises(ValueError):
        C.Mission(params(**{**CREEP, 'creep_max_s': 300.}))


def test_creep_end_to_end_around_an_unmapped_wall():
    """Closed loop (DynamicPX4 + rendered depth): a wall blocks the path; the 'navigator' answers the first
    two creep requests with 0.35 m hops sideways-forward and the third with a certified full detour."""
    state = {"hops": 0, "detour": None}

    def tick(t, m):
        if not m.targets:
            return
        req = m.replan_request
        if m.recover is not None and req is not None:
            n, e = m.last_v.pos[:2]
            if req.get("creep") and state["hops"] < 2 and m.phase in ("blocked", "recovery_scan"):
                if state.get("rid") != req["id"]:
                    state.update(rid=req["id"], hop=hop_pts((n, e), (n + .1, e + .33)), hops=state["hops"] + 1)
                give_hop(m, t, state["hop"], pid=100 + state["hops"])
            elif m.phase == "creep" and state.get("hop") is not None:
                give_hop(m, t, state["hop"], pid=100 + state["hops"])
            elif state["hops"] >= 2 and m.phase in ("blocked", "recovery_scan"):
                if state["detour"] is None:
                    pts, stops = dense([(n, e), (n, 1.3), (4., 1.3)])
                    state["detour"] = (pts, stops)
                pts, stops = state["detour"]
                m.on_path(t, 0, pts, speeds(pts, stops), stops, pid=200, request_id=req["id"], recovery_certified=True)
        elif m.recover is None and state["detour"] is not None:
            pts, stops = state["detour"]
            m.on_path(t, 0, pts, speeds(pts, stops), stops, pid=200)
        elif m.recover is None:
            m.on_path(t, 0, STRAIGHT, speeds(STRAIGHT), pid=1)
    m, px4, log = fly_path(params(**CREEP), lambda t: [WALL], tick, seconds=160.)
    assert m.counts["creep_hops"] == 2, (m.counts, m.result)
    assert m.result == "complete", m.result
    phases = [p for _, p, *_ in log]
    assert "creep" in phases
    # never closer than the prop tips (0.294 m) to the wall (n = 2.5, e in [-0.6, 0.6]) at any instant
    gap = min(math.hypot(max(abs(pos[0] - 2.5), 0.), max(abs(pos[1]) - .6, 0.)) for *_, pos in log)
    assert gap > .294, gap
    # within each hop, once the in-place turn has settled (the aircraft arrives still braking from cruise),
    # the aircraft moves no faster than the creep speed (+ the PX4 model's tracking slack)
    speeds_in_hops, run = [], []
    for (_, ph, _, _, a), (_, ph2, _, _, b) in zip(log[:-1], log[1:]):
        if ph == ph2 == "creep":
            run.append(math.dist(a[:2], b[:2]) / .05)
        else:
            speeds_in_hops += run[30:]
            run = []
    assert speeds_in_hops and max(speeds_in_hops) <= C.Params().creep_speed_mps + .05, max(speeds_in_hops)
