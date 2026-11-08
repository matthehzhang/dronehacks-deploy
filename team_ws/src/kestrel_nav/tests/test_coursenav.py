"""coursenav: map, planner, scan and particle filter on the published course (no ROS, no simulator).

Ported from the team workspace (AMMistry18/workspace tests/test_coursenav.py). Dropped here: the tilt study,
the offline closed-loop flights and the gate-alignment tests, which drive the old refnav controller that
skybeta replaces with skybeta_flight_interface; and the generality test (tools/generality.py not ported)."""
import math
from pathlib import Path

import numpy as np
import pytest

from kestrel_nav.coursenav import objmap, planner
from kestrel_nav.coursenav.course import Grid2D, load_course as _load_course, plan_grids, resolve, scan_grid
from kestrel_nav.coursenav.objmap import CourseMap
from kestrel_nav.coursenav.pf import CorrectionPF, PFParams, compose, inverse
from kestrel_nav.coursenav.scan import Scan, depth_points_level, level_from_px4_q, level_from_rp, scan_from_points


def load_course(*, derive=True):
    """Historical drawing regression; active map has dedicated test_new_course coverage."""
    from kestrel_nav.coursenav.course import PROJECT
    return _load_course(PROJECT / 'course/course_v3_legacy.json', derive=derive)


ROOT = Path(__file__).resolve().parent.parent
MARKINGS = objmap.FLOOR_MARKING_PREFIXES


@pytest.fixture(scope="module")
def course():
    doc = load_course()
    path = resolve(doc, "map")
    if not path.exists():
        cm = objmap.build(objmap.load_obj(resolve(doc, "obj")))
        cm.save(path)
    cm = CourseMap.load(path)
    obst, blocked, clearance = plan_grids(cm, doc)
    sg = scan_grid(cm)
    return doc, cm, obst, blocked, clearance, sg, sg.distance()


# ---------------------------------------------------------------- map
def test_obj_triangles_are_sampled_densely():
    tri = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], float)
    p = objmap.sample_triangles(tri, 0.05)
    assert p[:, 0].max() == pytest.approx(1.0) and p[:, 1].max() == pytest.approx(1.0)
    assert np.all(p.sum(axis=1) <= 1.0 + 1e-9)


def test_map_matches_published_geometry(course):
    doc, cm, obst, blocked, clearance, *_ = course
    # rear barrier south face is y=0.406, top 1.727; launch mark centre is free
    assert obst.lookup([[-2.0, 0.41]])[0]                                   # face (surfaces only; interior hollow)
    assert not obst.lookup([doc["start"]["xy"]])[0]
    assert cm.band(1.9, 2.5)[tuple(cm.xy_to_ij([-2.0, 0.41]))] == False  # noqa: E712  barrier is lower
    # window opening is open at cruise height and closed below the sill
    w = doc["chicane_options"]["window"]["center"]
    assert not cm.band(0.55, 1.05)[tuple(cm.xy_to_ij(w))]
    assert cm.band(0.28, 0.33, exclude_prefixes=MARKINGS)[tuple(cm.xy_to_ij(w))]   # sill top face


def test_vertical_ray_hits_floor_and_markings_can_be_ignored(course):
    cm = course[1]
    r = cm.raycast([[0.0, -1.5, 1.0]], [[0, 0, -1]], 3.0)
    assert r[0] == pytest.approx(1.0, abs=0.06)
    r2 = cm.raycast([[-2.5146, 0.9144, 1.0]], [[0, 0, -1]], 3.0, ignore_prefixes=("Launch mark",))
    assert r2[0] == pytest.approx(1.0, abs=0.06)


# ---------------------------------------------------------------- planner
@pytest.mark.parametrize("chicane", ["window"])   # gate_b: refused at 0.40 m (gate rule); see the gate_b tests
def test_full_course_plan_keeps_clearance(course, chicane):
    doc, cm, obst, blocked, clearance, *_ = course
    p = planner.plan(doc, obst, blocked, clearance, chicane=chicane, cmap=cm)
    names = [w["name"] for w in p["waypoints"]]
    for goal in ("start", "maze_exit", "tunnel_entry", "tunnel_exit", chicane, "pad"):
        assert goal in names
    assert p["min_clearance_m"] >= p["required_clearance_m"]
    assert names.index("tunnel_exit") == names.index("tunnel_entry") + 1    # straight tunnel run
    legs = [math.dist(p["waypoints"][i]["xy"], p["waypoints"][i + 1]["xy"]) for i in range(len(names) - 1)]
    assert min(legs) > 0.15


def test_course_policy_keeps_mapped_walls_outside_refnavs_corridor(course):
    """Planned clearance (vehicle radius + margin) must exceed refnav's corridor half-width, and the
    plan band must equal refnav's band, or a planned leg past a corner is called blocked."""
    doc = course[0]
    rp = doc["refnav_params"]
    assert doc["vehicle"]["radius_m"] + doc["vehicle"]["margin_m"] >= rp["corridor_half_m"] + 0.05
    assert rp["band_m"] == doc["plan_band_m"]
    assert rp["corridor_half_m"] >= doc["vehicle"]["radius_m"]


def test_gate_b_crossed_along_the_wall_normal_cannot_meet_the_course_clearance(course):
    """Crossing along the C2 wall's normal points the exit at the barrier (0.35 m left; needs 0.40):
    with the direction search switched off, Gate B cannot be planned."""
    import copy
    doc, cm, obst, blocked, clearance, *_ = course
    d = copy.deepcopy(doc)
    d["chicane_options"]["gate_b"]["cross_search_deg"] = 0.0
    with pytest.raises(planner.PlanError):
        planner.plan(d, obst, blocked, clearance, chicane="gate_b")


def test_gate_b_crossing_direction_is_searched_not_typed_in(course):
    """The planner searches +-30 deg about the gap's perpendicular and finds a crossing with the
    course clearance (flights D/E flew about this direction, then typed in as 'due west')."""
    doc, cm, obst, blocked, clearance, *_ = course
    assert "cross_deg" not in doc["chicane_options"]["gate_b"]
    n, info = planner.gate_crossing(doc, "gate_b", obst, clearance)
    assert info["feasible"] and abs(info["chosen_deg"]) <= 30.0 and info["chosen_deg"] != 0.0
    # wrapped: -180 and +180 are the same heading (ported from workspace 11c3531)
    assert abs(abs(math.degrees(math.atan2(n[1], n[0]))) - 180.0) <= 25.0


def test_the_chicane_is_chosen_by_score_and_the_window_wins(course):
    doc, cm, obst, blocked, clearance, *_ = course
    raw = load_course(derive=False)
    assert "chicane" not in raw                                   # no fixed choice in the course file
    res = planner.plan_options(doc, obst, blocked, clearance, cmap=cm)
    assert set(res["scores"]) == {"window"}                       # gate_b is refused at 0.40 m (gate rule)
    assert res["chicane"] == "window"
    assert "reachable only through gate_b" in res["reason"]
    assert res["scores"]["window"]["points"] == 5
    assert "window" in res["reason"] and "gate_b" in res["reason"]
    p = planner.plan(doc, obst, blocked, clearance, cmap=cm)        # chicane None: the same choice
    assert p["chicane"] == "window" and p["scores"]


def test_no_leg_ends_facing_a_solid_surface_inside_the_blind_range(course):
    """Flight E's stop: the Gate B approach turned to face the barrier at 0.47 m. The chosen plan
    has no leg end facing a solid surface within face_min_m, nor one inside the turn's arc."""
    doc, cm, obst, blocked, clearance, *_ = course
    p = planner.plan(doc, obst, blocked, clearance, chicane="window", cmap=cm)
    assert p["leg_end_violations"] == []
    relaxed = doc["leg_policy"]["face_relaxed_m"]
    for w in p["waypoints"][:-1]:
        assert w.get("facing_m") is None or w["facing_m"] >= relaxed, w
    for name in ("window_pre", "tunnel_entry"):                 # the gate approaches meet the full rule
        w = next(w for w in p["waypoints"] if w["name"] == name)
        assert w.get("facing_m") is None or w["facing_m"] >= doc["leg_policy"]["face_min_m"]


def test_a_via_point_facing_a_wall_is_moved_back_along_its_leg(course):
    doc, cm, obst, blocked, clearance, *_ = course
    solid = planner.solid_band(cm, doc)
    # a corner right at a maze wall: the via point faces the divider 0.3 m away
    wps = [{"name": "start", "xy": [-2.5, 1.2]}, {"name": "via1", "xy": [-2.5, 2.3]}, {"name": "goal", "xy": [-1.4, 2.6]}]
    before = planner.facing_distance(solid, wps[1]["xy"], math.atan2(0.3, 1.1))
    out, moved = planner.fix_leg_ends([dict(w) for w in wps], obst, blocked, clearance, 0.40, doc["leg_policy"], solid)
    if before < doc["leg_policy"]["face_min_m"]:
        assert moved == ["via1"] and out[1]["end_ok"]
        assert out[1]["xy"][1] < wps[1]["xy"][1]                  # moved back along the incoming (northward) leg
    else:
        assert moved == []


def test_alignment_geometry_is_measured_on_the_map(course):
    doc, cm, obst, blocked, clearance, *_ = course
    p = planner.plan(doc, obst, blocked, clearance, chicane="window", cmap=cm)
    w = next(w for w in p["waypoints"] if w["name"] == "window")
    a = w["align"]
    assert a["width_m"] == pytest.approx(0.914, abs=0.08)         # the 36 in window, voxel-quantised
    assert a["dz_lo"] == pytest.approx(0.305 - 0.8, abs=0.01) and a["dz_hi"] == pytest.approx(1.219 - 0.8, abs=0.01)
    assert a["dist_m"] == pytest.approx(0.8) and a["visible"]
    t = next(w for w in p["waypoints"] if w["name"] == "tunnel_entry")
    assert t["align"]["group"] == "tunnel" and t["align"]["visible"]
    assert 1.1 < t["align"]["width_m"] < 58 * 0.0254 + 0.05        # the oval's width across the flight band


def test_the_tunnel_leg_is_a_passage_by_geometry_and_a_maze_leg_is_not(course):
    doc, cm, obst, blocked, clearance, *_ = course
    p = planner.plan(doc, obst, blocked, clearance, chicane="window", cmap=cm)
    ex = next(w for w in p["waypoints"] if w["name"] == "tunnel_exit")
    assert ex["passage"]["is_passage"] and ex["passage"]["map_clear"] and ex["passage"]["enclosed_frac"] >= 0.4
    z = doc["cruise_z_m"]
    assert planner.passage_fraction(cm, p["waypoints"][1]["xy"], p["waypoints"][2]["xy"], z) < 0.1


def test_leg_metadata_marks_gates_and_passages_by_name_only(course):
    doc, cm, obst, blocked, clearance, *_ = course
    p = planner.plan(doc, obst, blocked, clearance, chicane="window", cmap=cm)
    legs = planner.to_legs(p["waypoints"])
    assert len(legs) == len(planner.flight_waypoints(p["waypoints"])) - 1 == len(p["waypoints"]) - 2
    assert not any(l["name"] == "window" for l in legs)          # the centre is crossed, not stopped at
    win = next(l for l in legs if l["name"] == "window_post")
    assert win["hold"] and win["gate_group"] == "window" and win["align"]["group"] == "window"
    assert win["crosses"] == "window"
    tun = next(l for l in legs if l["name"] == "tunnel_exit")
    assert tun["tunnel"]["map_clear"] and tun["align"]["group"] == "tunnel"
    assert not any("xy" in l for l in legs)
    assert not any(l.get("hold") for l in legs if l["name"].startswith("via"))


def test_planner_refuses_blocked_goal(course):
    doc, cm, obst, blocked, clearance, *_ = course
    with pytest.raises(planner.PlanError):
        planner.astar(blocked, clearance, doc["start"]["xy"], [-2.0, 0.45])   # inside the barrier


def test_route_conversion_is_forward_left():
    wps = [{"xy": [0, 0]}, {"xy": [0, 2]}, {"xy": [-1, 2]}]
    r = planner.to_route(wps, (0, 0), math.pi / 2)                           # facing +y (north)
    assert r[0] == pytest.approx((2.0, 0.0)) and r[1] == pytest.approx((2.0, 1.0))


# ---------------------------------------------------------------- scan
def test_level_rotations_agree():
    # PX4 FRD->NED for a body pitched nose-up 10 deg, yawed 30 deg
    pitch, yaw = math.radians(10), math.radians(30)
    cy, sy, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    q = (cy * cp, -sy * sp, cy * sp, sy * cp)                                # q_yaw * q_pitch
    # nose-up in FRD is +pitch; in FLU the same physical attitude is pitch about +y_FLU = -y_FRD
    np.testing.assert_allclose(level_from_px4_q(q), level_from_rp(0.0, -pitch), atol=1e-9)


def test_depth_scan_sees_a_wall_ahead():
    K = (421.0, 421.0, 319.5, 199.5)
    R_bc = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)               # optical -> FLU body
    depth = np.full((400, 640), 2.0, np.float32)                              # fronto-parallel wall at 2 m
    pts = depth_points_level(depth, K, R_bc, [0.1, 0, 0], np.eye(3))
    s = scan_from_points(pts, height_m=0.8)
    assert len(s) > 30
    straight = s.range[np.argmin(np.abs(s.bearing))]
    assert straight == pytest.approx(2.1, abs=0.03)


# ---------------------------------------------------------------- particle filter
def synth_scan(cm, pose, rng, z=0.8, n=48, hfov=math.radians(72)):
    """Forward depth beams only (the airframe's own sensor)."""
    b = np.linspace(-hfov / 2, hfov / 2, n)
    a = pose[2] + b
    o = np.c_[np.full(len(b), pose[0]), np.full(len(b), pose[1]), np.full(len(b), z)]
    r = cm.raycast(o, np.c_[np.cos(a), np.sin(a), np.zeros(len(b))], 4.0, ignore_prefixes=MARKINGS)
    ok = np.isfinite(r) & (r > 0.4)
    r = r[ok] + rng.normal(0, 0.03, ok.sum())
    return Scan(b[ok], r, 0.05 + 0.02 * r ** 2)


def test_compose_inverse_roundtrip():
    p = np.array([1.0, -2.0, 0.7])
    np.testing.assert_allclose(compose(p, inverse(p)), [0, 0, 0], atol=1e-12)


def test_pf_finds_the_spawn_offset(course):
    """The sim spawns 0.20 m north of the launch-mark prior; the scan match must find it."""
    doc, cm, *_, sg, dist = course
    truth = np.array([-2.5146, 1.1144, math.pi / 2])
    rng = np.random.default_rng(3)
    pf = CorrectionPF(sg, dist, PFParams(), seed=3)
    odom = np.zeros(3)
    pf.init((*doc["start"]["xy"], math.radians(doc["start"]["yaw_deg"])), 0.3, math.radians(8), odom)
    for _ in range(30):
        pf.predict(odom, 0.07)
        pf.update(odom, synth_scan(cm, truth, rng))
    m, _ = pf.estimate(odom)
    assert math.hypot(m[0] - truth[0], m[1] - truth[1]) < 0.08
    assert abs(m[2] - truth[2]) < math.radians(4)


def test_pf_removes_vio_drift_through_the_maze(course):
    """Fly the planned maze legs with a VIO that has 5% scale error and 1 deg/m heading drift
    (deliberately worse than the measured 2 cm p95 short-hover VIO, to exercise the correction)."""
    doc, cm, obst, blocked, clearance, sg, dist = course
    p = planner.plan(doc, obst, blocked, clearance, chicane="window")
    wps = [np.array(w["xy"]) for w in p["waypoints"][: [w["name"] for w in p["waypoints"]].index("maze_exit") + 1]]
    pts = [wps[0]]
    for a, b in zip(wps[:-1], wps[1:]):
        n = max(1, int(math.dist(a, b) / 0.05))
        pts += [a + (b - a) * k / n for k in range(1, n + 1)]
    rng = np.random.default_rng(5)
    start_yaw = math.pi / 2
    pf = CorrectionPF(sg, dist, PFParams(), seed=5)
    truth_prev = np.array([*pts[0], start_yaw])
    odom = np.zeros(3)
    pf.init((*doc["start"]["xy"], start_yaw), 0.3, math.radians(8), odom)
    errs, vio_errs = [], []
    T_true = compose(truth_prev, inverse(odom))
    travelled = 0.0
    for i in range(1, len(pts)):
        d = pts[i] - pts[i - 1]
        yaw = math.atan2(d[1], d[0]) if np.linalg.norm(d) > 1e-6 else truth_prev[2]
        truth = np.array([*pts[i], yaw])
        step = compose(inverse(truth_prev), truth)                           # body-frame increment
        travelled += float(np.hypot(*step[:2]))
        step[:2] *= 1.05                                                     # VIO scale error
        step[2] += math.radians(1.0) * float(np.hypot(*step[:2]))            # VIO heading drift
        odom = compose(odom, step)
        truth_prev = truth
        pf.predict(odom, 0.07)
        if i % 2 == 0:                                                       # depth at ~half the step rate
            pf.update(odom, synth_scan(cm, truth, rng))
        m, _ = pf.estimate(odom)
        errs.append(math.hypot(m[0] - truth[0], m[1] - truth[1]))
        vio = compose(T_true, odom)
        vio_errs.append(math.hypot(vio[0] - truth[0], vio[1] - truth[1]))
    assert travelled > 5.0
    assert vio_errs[-1] > 0.2                                                # the raw VIO really has drifted
    assert np.percentile(errs[40:], 95) < 0.15                               # the course requirement
    assert errs[-1] < 0.10


# ---------------------------------------------------------------- offline tools (render, tilt, closed loop)
def test_fast_raycast_agrees_with_the_reference_march(course):
    cm = course[1]
    rng = np.random.default_rng(0)
    o = np.c_[rng.uniform(-2.8, 2.8, 800), rng.uniform(-2.8, 2.8, 800), rng.uniform(0.2, 2.5, 800)]
    d = rng.normal(size=(800, 3))
    a = cm.raycast(o, d, 6.0)
    b, _ = cm.raycast_fast(o, d, 6.0)
    assert np.array_equal(np.isfinite(a), np.isfinite(b))
    both = np.isfinite(a)
    assert np.mean(np.abs(a[both] - b[both]) <= 0.06) > 0.98             # corner clips may differ by a voxel


def test_rendered_depth_sees_the_barrier_behind_and_the_maze_ahead(course):
    from kestrel_nav.coursenav.render import Camera, DepthModel, DepthRenderer, body_pose_matrix
    cm = course[1]
    r = DepthRenderer(cm, Camera(), DepthModel(noise=False))
    R, t = body_pose_matrix(-2.5146, 1.1144, 0.8, -math.pi / 2)          # facing south: barrier face y 0.508
    img = r.render(R, t)
    assert img[200, 320] == pytest.approx(1.1144 - 0.508 - 0.116, abs=0.06)
    s = scan_from_points(depth_points_level(img, r.cam.K, r.cam.R_body_cam(), r.cam.t_body, np.eye(3)), 0.8)
    assert len(s) > 30


# ---------------------------------------------------------------- localizer tails (A6)
def test_dominant_cluster_beats_the_mean_of_a_split_cloud():
    from kestrel_nav.coursenav.pf import CorrectionPF, PFParams
    g = Grid2D(np.zeros((200, 200), bool), np.array([-5.0, -5.0]), 0.05)
    pf = CorrectionPF(g, np.full((200, 200), 1.0), PFParams(n=100, use_cluster=True), seed=1)
    pf.init((0.0, 0.0, 0.0), 0.0, 0.0, np.zeros(3))
    b = np.zeros((100, 3))
    b[:70, 0] = 1.0 + np.linspace(-0.03, 0.03, 70)              # 70 % of the weight at x = 1 ...
    b[70:, 0] = 3.0 + np.linspace(-0.03, 0.03, 30)              # ... 30 % at x = 3 (another wall explains it)
    pf.T = b
    m, cov = pf.estimate(np.zeros(3))
    assert m[0] == pytest.approx(1.0, abs=0.02) and pf.last_cluster_frac == pytest.approx(0.7, abs=0.02)
    pf.p.use_cluster = False
    m2, _ = pf.estimate(np.zeros(3))
    assert m2[0] == pytest.approx(1.6, abs=0.02)                  # the mean lands between the two walls


def test_a_scan_the_map_does_not_explain_is_skipped(course):
    doc, cm, *_, sg, dist = course
    pf = CorrectionPF(sg, dist, PFParams(), seed=3)
    truth = np.array([-2.5146, 1.1144, math.pi / 2])
    pf.init(tuple(truth), 0.05, math.radians(2), np.zeros(3))
    rng = np.random.default_rng(3)
    good = synth_scan(cm, truth, rng)
    assert pf.update(np.zeros(3), good) and pf.skipped == 0
    w_before = pf.w.copy()
    junk = Scan(good.bearing, np.full(len(good), 0.5), good.sigma)   # a 0.5 m bubble ahead: free space on the map
    assert not pf.update(np.zeros(3), junk) and pf.skipped == 1
    np.testing.assert_array_equal(pf.w, w_before)                 # the belief was not touched
    few = Scan(good.bearing[:5], good.range[:5], good.sigma[:5])
    assert not pf.update(np.zeros(3), few) and pf.skipped == 2


def test_beams_on_the_net_are_not_informative(course):
    doc, cm, *_, sg, dist = course
    net = cm.band(0.3, 1.6, only_prefixes=("Net",))
    pf = CorrectionPF(sg, dist, PFParams(), seed=3, uninformative=net)
    truth = np.array([-2.5146, 1.1144, math.pi / 2])             # facing the north net 1.9 m away
    pf.init(tuple(truth), 0.02, math.radians(1), np.zeros(3))
    rng = np.random.default_rng(3)
    scan = synth_scan(cm, truth, rng)
    inf = pf.informative(np.zeros(3), scan)
    assert inf.sum() < len(scan)                                   # the net beams are discounted ...
    assert inf.sum() >= 5                                          # ... the M1 divider's beams remain
    n, frac = pf.scan_fit(np.zeros(3), scan)                       # too few: judged on every known beam
    assert n >= 8 and frac > 0.5
    assert pf.update(np.zeros(3), scan)                            # the start scan is still used


def test_correction_is_limited_to_what_drift_allows():
    from kestrel_nav.coursenav.pf import CorrectionLimiter
    lim = CorrectionLimiter(rate_per_m=0.05, rate_per_s=0.01)
    first = lim.step((0.2, 0.0), 0.0, 0.0)
    assert first == pytest.approx([0.2, 0.0])                     # the start offset passes at once
    out = lim.step((0.5, 0.0), 1.0, 0.0)                           # a 0.3 m jump after 1 m: 5 cm of it
    assert out == pytest.approx([0.25, 0.0])
    out = lim.step((0.5, 0.0), 0.0, 2.0)                           # hovering 2 s: 2 cm more
    assert out == pytest.approx([0.27, 0.0])


def test_correction_is_projected_onto_the_measured_directions():
    from kestrel_nav.coursenav.pf import project_correction
    cov = np.diag([0.3 ** 2, 0.02 ** 2, 0.01])                     # a corridor along x: x unobservable
    out, kept, P = project_correction((0.2, 0.1), cov, 0.10)
    assert kept == 1 and out == pytest.approx([0.0, 0.1], abs=1e-9)
    assert P @ np.array([1.0, 1.0]) == pytest.approx([0.0, 1.0], abs=1e-9)
    out2, kept2, _ = project_correction((0.2, 0.1), np.diag([0.03 ** 2, 0.02 ** 2, 0.01]), 0.10)
    assert kept2 == 2 and out2 == pytest.approx([0.2, 0.1])


# ---------------------------------------------------------------- Track B2: staged cost-to-go fields
def test_navfield_builds_fast_and_descends_through_every_stage_to_the_pad(course):
    from kestrel_nav.coursenav.navfield import NavField, stages_from_course
    doc, cm, obst, blocked, clearance, *_ = course
    stages = stages_from_course(doc)
    assert [s.name for s in stages] == ["maze_exit", "tunnel_in", "tunnel_out", "chicane", "pad"]
    nf = NavField(blocked, clearance, stages)
    assert nf.build_s < 0.5                                       # 20 ms here; the Orin target is 100 ms
    start = doc["start"]["spawn_xy"]
    assert math.isfinite(nf.cost_to_go(start, 0))
    xy, st, crossed, pts = start, 0, [], []
    for _ in range(80):
        path, st, cr = nf.descend(xy, st, lookahead_m=1.0)
        pts.append(path)
        crossed += cr
        xy = path[-1]
        if st == len(stages) - 1 and nf.cost_to_go(xy, st) < 1e-6:
            break
    assert crossed == ["maze_exit", "tunnel_mouth", "tunnel_far", "window"], crossed   # the window's credit wins
    assert math.dist(xy, doc["sequence"][-1]["xy"]) < 0.35
    full = np.concatenate(pts)
    ij = obst.ij(full)
    assert clearance[ij[:, 0], ij[:, 1]].min() >= 0.40 - 1e-9


def test_navfield_takes_gate_b_when_the_window_pays_nothing(course):
    from kestrel_nav.coursenav.navfield import NavField, stages_from_course
    doc, cm, obst, blocked, clearance, *_ = course
    d = copy_doc = __import__("copy").deepcopy(doc)
    d["chicane_options"]["window"]["points"] = 0
    d["chicane_options"]["window"]["width_m"] = 0.3                # too narrow for the hop margins: unusable
    stages = stages_from_course(d)
    nf = NavField(blocked, clearance, stages)
    xy, st, crossed, pts = d["start"]["spawn_xy"], 0, [], []
    for _ in range(80):
        path, st, cr = nf.descend(xy, st, lookahead_m=1.0)
        pts.append(path)
        crossed += cr
        xy = path[-1]
        if st == len(stages) - 1 and nf.cost_to_go(xy, st) < 1e-6:
            break
    crossings = planner.gate_crossings(np.concatenate(pts),
                                      planner.gate_segment(d["chicane_options"]["gate_b"]), (-1, 0))
    assert [direction for _, direction in crossings] in ([], [1])
    # At 0.40 m the remaining gate cannot be reached on its declared pre side.
    assert not math.isfinite(nf.cost_to_go(d["start"]["spawn_xy"], 0))
    assert not crossed and not crossings



def test_gate_b_at_course_clearance_is_refused_not_crossed_twice(course):
    """At 0.40 m the only way to gate_b_pre is through Gate B itself: before the gate rule, A* flew
    through the gap the wrong way, turned in the pocket beyond it (Flight E landed there on silent
    depth, 2026-11-06) and crossed back. A waypoint-name count did not see it; the planner must refuse
    and say why."""
    doc, cm, obst, blocked, clearance, *_ = course
    with pytest.raises(planner.PlanError, match="reachable only through gate_b"):
        planner.plan(doc, obst, blocked, clearance, chicane="gate_b", cmap=cm)


def test_gate_b_crosses_once_where_the_pocket_has_another_way_in(course):
    """At 0.38 m (margin 0.10) a second way into the pocket opens: one crossing, in the chosen direction."""
    doc, cm, *_ = course
    d = dict(doc, vehicle=dict(doc["vehicle"], margin_m=0.10))
    obst, blocked, clearance = plan_grids(cm, d)
    p = planner.plan(d, obst, blocked, clearance, chicane="gate_b", cmap=cm)
    g = next(w for w in p["waypoints"] if w["name"] == "gate_b")
    cr = planner.gate_crossings([w["xy"] for w in p["waypoints"]], p["gate_lines"][0], g["normal"])
    assert [s for _, s in cr] == [1]
    assert p["min_clearance_m"] >= p["required_clearance_m"] - 1e-9


def test_window_plan_crosses_the_window_once(course):
    doc, cm, obst, blocked, clearance, *_ = course
    p = planner.plan(doc, obst, blocked, clearance, chicane="window", cmap=cm)
    g = next(w for w in p["waypoints"] if w["name"] == "window")
    cr = planner.gate_crossings([w["xy"] for w in p["waypoints"]], p["gate_lines"][0], g["normal"])
    assert [s for _, s in cr] == [1]


def test_gate_crossings_counts_passes_through_the_opening_only():
    line, n = [[0.0, -0.5], [0.0, 0.5]], [-1.0, 0.0]          # opening on x = 0, cross westwards
    assert planner.gate_crossings([[1, 0], [-1, 0]], line, n) == [(0, 1)]
    assert planner.gate_crossings([[-1, 0], [1, 0]], line, n) == [(0, -1)]
    through_and_back = [[-1, 0], [1, 0.1], [1, -0.1], [-1, 0]]
    assert [s for _, s in planner.gate_crossings(through_and_back, line, n)] == [-1, 1]
    around_the_end = [[1, 0], [1, 2], [-1, 2], [-1, 0]]       # passes beyond y = 0.5: not through the gate
    assert planner.gate_crossings(around_the_end, line, n) == []
    on_the_line = [[1, 0], [0, 0], [-1, 0]]                    # pre -> centre -> post legs: one crossing
    assert planner.gate_crossings(on_the_line, line, n) == [(1, 1)]
