"""livenav unit tests: pure Python, no ROS. The closed-loop course flight is in test_livenav_sim.py."""
import math

import numpy as np
import pytest

from kestrel_nav.coursenav.scan import Scan
from kestrel_nav.livenav.follower import FollowParams, Follower, path_obstacle, path_progress, point_at
from kestrel_nav.livenav.mission import LiveMission, MissionParams, Observation
from kestrel_nav.livenav.navgrid import NavGrid, NavParams
from kestrel_nav.livenav.pad import PadDetector
from kestrel_nav.livenav.padsim import PadFloor
from kestrel_nav.livenav.replanner import goals_from_plan


@pytest.fixture(scope="module")
def mission():
    return LiveMission(MissionParams(landed_z_m=0.067, use_pf=False))


# ---------------------------------------------------------------------------- grid
def test_fence_clips_returns_beyond_the_net(mission):
    g = mission.grid
    pose = (2.6, 0.0, 0.0)                                   # 0.45 m from the east net, facing it
    scan = Scan(np.array([0.0, 0.3]), np.array([3.0, 3.0]), np.array([0.05, 0.05]))
    clipped = g.fence_clip(pose, scan)
    assert clipped.range[0] == pytest.approx(3.0493 - 2.6, abs=1e-6)
    assert clipped.range[1] < 0.6                            # oblique ray also stops at the net plane


def test_live_obstacle_blocks_and_prior_walls_do_not_thicken(mission):
    g = NavGrid(mission.cmap, mission.doc, NavParams())
    before = g.blocked.occ.sum()
    # an unmapped pillar 1 m ahead in the open south area, seen 5 times
    pose = (1.3, -2.3, math.pi / 2)
    scan = Scan(np.linspace(-0.12, 0.12, 7), np.full(7, 1.0), np.full(7, 0.03))   # a 0.24 m wide face
    for k in range(5):
        g.integrate(pose, scan, now=k * 1.0)
    g.rebuild(force=True)
    assert g.blocked.occ.sum() > before
    assert g.is_blocked((1.3, -1.3))
    # returns on a mapped wall (the barrier north face at y=0.508) are explained by the prior
    g2 = NavGrid(mission.cmap, mission.doc, NavParams())
    pose2 = (-0.6, 1.5, -math.pi / 2)                        # facing south at the barrier, 1 m away
    scan2 = Scan(np.linspace(-0.12, 0.12, 7), np.full(7, 0.95), np.full(7, 0.03))   # 5 cm short of the wall
    for k in range(5):
        g2.integrate(pose2, scan2, now=k * 1.0)
    g2.rebuild(force=True)
    assert g2.live.landmarks().sum() == 0


# ---------------------------------------------------------------------------- follower
def test_path_obstacle_only_counts_returns_in_the_corridor():
    path = np.array([[0.0, 0.0], [2.0, 0.0]])
    _, s_here, _, cum, seg = path_progress(path, np.array([0.0, 0.0]))
    pose = (0.0, 0.0, 0.0)
    ahead = Scan(np.array([0.0]), np.array([1.0]), np.array([0.03]))
    beside = Scan(np.array([math.pi / 2]), np.array([0.5]), np.array([0.03]))
    assert path_obstacle(ahead, pose, path, cum, seg, s_here, 0.36, 2.0) == pytest.approx(1.0, abs=0.02)
    assert path_obstacle(beside, pose, path, cum, seg, s_here, 0.36, 2.0) == math.inf


def test_follower_slows_for_a_return_and_steers_to_the_carrot():
    f = Follower(FollowParams(v_cruise=0.4))
    path = np.array([[0.0, 0.0], [3.0, 0.0]])
    clear = Scan(np.array([0.0]), np.array([4.0]), np.array([0.03]))
    c1 = f.step(0.0, (0.0, 0.0, 0.0), path, None, clear, 1.0, brake_end=False)
    for k in range(1, 30):
        c1 = f.step(k * 0.1, (0.0, 0.0, 0.0), path, None, clear, 1.0, brake_end=False)
    assert c1.vx == pytest.approx(0.4, abs=0.02) and abs(c1.vy) < 1e-6
    near = Scan(np.array([0.0]), np.array([0.5]), np.array([0.03]))
    f2 = Follower(FollowParams(v_cruise=0.4))
    for k in range(30):
        c2 = f2.step(k * 0.1, (0.0, 0.0, 0.0), path, None, near, 1.0, brake_end=False)
    assert c2.speed < 0.02 and "scan" in c2.limiter


def test_corner_chord_measure_ignores_cell_zigzag():
    f = Follower()
    zig = np.array([[i * 0.05, (i % 2) * 0.05] for i in range(40)])
    _, s_here, _, cum, seg = path_progress(zig, zig[0])
    assert f._turn_ahead(zig, cum, seg, s_here, 1.0) < math.radians(15)
    corner = np.array([[0, 0], [1, 0], [1, 1.0]])
    _, s_here, _, cum, seg = path_progress(corner, corner[0])
    assert f._turn_ahead(corner, cum, seg, s_here, 1.5) > math.radians(70)


# ---------------------------------------------------------------------------- replanner
def test_goals_and_gate_legs(mission):
    goals = goals_from_plan(mission.plan)
    names = [g.name for g in goals]
    assert names[-1] == "pad" and "tunnel_exit" in names
    post = next(g for g in goals if g.crosses)
    assert post.straight and post.is_gate_leg
    assert not next(g for g in goals if g.name == "tunnel_exit").straight   # the tube is A*'d


def test_replanner_routes_around_a_live_pillar(mission):
    m = LiveMission(MissionParams(landed_z_m=0.067, use_pf=False))
    rp = m.replanner
    rp.k = [g.name for g in m.goals].index("via14")                   # tunnel exit -> open south area
    start = (2.3, -0.85)
    p0 = rp.plan(start, 0.0)
    assert p0 is not None and not rp.blocked
    mid = (p0[0] + p0[1]) / 2                                      # middle of the first leg, not a vertex
    pose = (start[0], start[1], math.atan2(mid[1] - start[1], mid[0] - start[0]))
    d = math.dist(start, mid)
    scan = Scan(np.zeros(6), np.full(6, d), np.full(6, 0.03))
    for k in range(6):
        m.grid.integrate(pose, scan, now=k * 1.0)
    m.grid.rebuild(force=True)
    p1 = rp.plan(start, 10.0)
    assert p1 is not None and not rp.blocked
    assert all(math.dist(pt, mid) > 0.3 for pt in p1[1:-1]) or len(p1) > len(p0)


# ---------------------------------------------------------------------------- pad detector
@pytest.mark.parametrize("dist", [2.0, 3.0, 4.0])
def test_pad_detector_on_synthetic_frames(dist):
    K = (421.016, 421.016, 319.248, 200.659)
    floor = PadFloor((-2.042, -0.792))
    det = PadDetector(K, (400, 640))
    yaw = math.radians(90.0)
    cam_xy = floor.pad_xy - dist * np.array([math.cos(yaw), math.sin(yaw)])
    img = floor.render(K, (400, 640), cam_xy, 0.829, yaw)
    r = det.detect(img, 0.829)
    assert r is not None
    assert r[0] == pytest.approx(dist, abs=0.05) and abs(r[1]) < 0.05


def test_pad_detector_rejects_plain_floor():
    K = (421.016, 421.016, 319.248, 200.659)
    det = PadDetector(K, (400, 640))
    img = np.full((400, 640), 230, np.uint8)
    assert det.detect(img, 0.829) is None


# ---------------------------------------------------------------------------- mission frames
def test_course_odom_round_trip(mission):
    m = mission
    m.set_transform((0.3, -0.2), math.radians(100.0))
    sx, sy, syaw = m.plan["start_course"]
    x, y, yaw = m.odom_to_course((0.3, -0.2), math.radians(100.0))
    assert (x, y) == pytest.approx((sx, sy), abs=1e-9) and yaw == pytest.approx(syaw, abs=1e-9)
    vx, vy = m.course_vec_to_odom(1.0, 0.0)
    assert math.hypot(vx, vy) == pytest.approx(1.0)


def test_mission_waits_then_climbs(mission):
    m = LiveMission(MissionParams(landed_z_m=0.067, use_pf=False))
    c = m.step(Observation(now=0.0, odom_xyz=(0, 0, 0), odom_yaw=0.0, offboard_active=False))
    assert c.state == "WAIT" and c.vz == 0.0
    c = m.step(Observation(now=0.1, odom_xyz=(0, 0, 0), odom_yaw=0.0, offboard_active=True))
    assert c.state == "CLIMB"
    c = m.step(Observation(now=0.2, odom_xyz=(0, 0, 0), odom_yaw=0.0, offboard_active=True))
    assert c.vz > 0.2
    c = m.step(Observation(now=0.3, odom_xyz=(0, 0, 0), odom_yaw=0.0, offboard_active=False))
    assert c.state == "PAUSED" and c.vx == 0.0 and c.vz == 0.0
