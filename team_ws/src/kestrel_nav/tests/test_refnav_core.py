"""refnav.core + refnav.depth, without ROS: a fake PX4 (kinematic, obeys Offboard/arm/land) and a
ray-cast depth camera in a world of walls. Each safety behaviour in core.py's docstring is pinned
here, and the same flight on two different clocks (simulation seconds, Unix seconds) must make
the same decisions."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refnav import core as C  # noqa: E402
from refnav import depth as D  # noqa: E402

W, H, FX = 160, 100, 106.0                  # a small 74 deg HFOV camera
FLOOR_D = 0.18                              # the floor below the body origin of a landed aircraft (as in the simulator)
K = (FX, FX, W / 2, H / 2)
R_BODY_CAM = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])   # FLU <- optical, forward camera
T_BODY_CAM = np.array([0.16, 0.0, 0.0])


def q_yaw(yaw):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def render_depth(pos_ned, yaw, walls):
    """Depth (m, NaN = no return) seen by the forward camera of a level aircraft at pos_ned/yaw.
    walls: [(axis, value, lo, hi)] vertical planes, axis 0 = n = value, axis 1 = e = value,
    extending lo..hi along the other horizontal axis and from the floor (d = 0) up to d = -2.5."""
    vs, us = np.mgrid[0:H, 0:W].astype(np.float64)
    d_opt = np.stack([(us - K[2]) / K[0], (vs - K[3]) / K[1], np.ones_like(us)], -1)
    d_flu = d_opt @ R_BODY_CAM.T
    d_frd = d_flu * np.array([1.0, -1.0, -1.0])
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    d_ned = d_frd @ Rz.T
    o = np.asarray(pos_ned) + Rz @ (T_BODY_CAM * np.array([1.0, -1.0, -1.0]))
    best = np.full((H, W), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = (FLOOR_D - o[2]) / d_ned[..., 2]               # the floor, 0.18 m below the body when landed
        best = np.where((tf > 0), np.minimum(best, tf), best)
        for axis, val, lo, hi in walls:
            t = (val - o[axis]) / d_ned[..., axis]
            p = o + d_ned * t[..., None]
            other = p[..., 1 - axis]
            hit = (t > 0) & (other >= lo) & (other <= hi) & (p[..., 2] <= 0) & (p[..., 2] >= -2.5)
            best = np.where(hit, np.minimum(best, t), best)
    best[~np.isfinite(best)] = np.nan
    return best.astype(np.float32)


class FakePX4:
    """Obeys the mission the way PX4 does at this level of detail: Offboard only after setpoints
    have been streaming, arm, track the setpoint at <= 1 m/s and 60 deg/s, AUTO_LAND descends
    and disarms on the floor."""

    def __init__(self, pos=(0.0, 0.0, 0.0), yaw=0.0):
        self.pos = np.array(pos, float)
        self.yaw = yaw
        self.armed, self.nav, self.landed = False, 4, True
        self.stream_since = None

    def apply(self, t, out, dt):
        if out.stream:
            self.stream_since = t if self.stream_since is None else self.stream_since
        else:
            self.stream_since = None
        for c in out.commands:
            if c == "offboard" and self.stream_since is not None and t - self.stream_since >= 0.5:
                self.nav = C.NAV_OFFBOARD
            elif c == "arm":
                self.armed = True
            elif c == "land":
                self.nav = C.NAV_AUTO_LAND
        if self.armed and self.nav == C.NAV_OFFBOARD and out.setpoint is not None:
            n, e, d, yaw = out.setpoint
            step = np.array([n, e, d]) - self.pos
            dist = np.linalg.norm(step)
            self.pos += step * min(1.0, 1.0 * dt / dist) if dist > 1e-9 else 0
            err = (yaw - self.yaw + math.pi) % (2 * math.pi) - math.pi
            self.yaw += max(-math.radians(60) * dt, min(math.radians(60) * dt, err))
            self.landed = self.pos[2] > -0.05
        elif self.nav == C.NAV_AUTO_LAND and self.armed:
            self.pos[2] = min(0.0, self.pos[2] + 0.7 * dt)
            if self.pos[2] >= 0.0:
                self.landed, self.armed = True, False

    def vehicle(self):
        return C.Vehicle(pos=tuple(self.pos), heading=self.yaw, pos_ok=True, q=q_yaw(self.yaw),
                         armed=self.armed, nav_state=self.nav, landed=self.landed)


ROOM = [(0, 15.0, -15.0, 15.0), (0, -15.0, -15.0, 15.0), (1, 15.0, -15.0, 15.0), (1, -15.0, -15.0, 15.0)]


def fly(params, walls=(), t0=0.0, seconds=120.0, dt=0.05, health=lambda t: True, depth_on=lambda t: True,
        nav_override=None, yaw0=0.0, phantom=lambda t: False, depth_fn=None, on_tick=None, px4=None):
    """depth_fn(t, img) -> img replaces what the camera delivers (blank, sparse, ...);
    on_tick(t, mission) runs before each step (outside requests)."""
    walls = list(walls) + ROOM                         # indoors: a 30 x 30 m room around everything
    if params.depth_border_px == C.Params().depth_border_px:
        params.depth_border_px = 3                     # the real camera's 24 px border, scaled to this 160 x 100 one
    m = C.Mission(params)
    px4 = FakePX4(yaw=yaw0) if px4 is None else px4
    log = []
    t = 0.0
    next_depth = 0.0
    while t < seconds and m.phase != "done":
        now = t0 + t
        if health(t) is not None:
            m.on_health(now, health(t))
        if t >= next_depth and depth_on(t):
            img = render_depth(px4.pos, px4.yaw, walls)
            if phantom(t):
                img[6:40, 56:104] = 0.6                  # a stereo false match: 0.6 m, near the top, ONE frame
            if depth_fn is not None:
                img = depth_fn(t, img)
            m.on_depth(now, img, K, R_BODY_CAM, T_BODY_CAM, q_yaw(px4.yaw), now=now)
            next_depth = t + 1 / 7.5
        v = px4.vehicle()
        if nav_override is not None and nav_override(t) is not None:
            px4.nav = nav_override(t)
            v.nav_state = px4.nav
        if on_tick is not None:
            on_tick(t, m)
        out = m.step(now, v)
        log.append((t, out.phase, out.stream, tuple(out.commands), px4.pos.copy()))
        px4.apply(now, out, dt)
        t += dt
    return m, px4, log


ROUTE = [(3.0, 0.0), (3.0, 3.0)]            # forward 3 m, then 3 m to the left


def test_nominal_route_completes_and_stops_streaming_once_px4_lands():
    m, px4, log = fly(C.Params(route=ROUTE))
    assert m.result == "complete", (m.result, m.why)
    assert m.legs_done == 2
    phases = [p for _, p, *_ in log]
    assert phases.index("arm") < phases.index("climb") < phases.index("go") < phases.index("land")
    # the takeoff heading is north (yaw 0): forward 3 m = n 3; left 3 m = e -3
    assert np.allclose(m.targets[1], (3.0, -3.0), atol=1e-9)
    land_auto = [i for i, (_, p, *_r) in enumerate(log) if p == "land" and px4.nav == C.NAV_AUTO_LAND]
    first_auto = next(i for i, (t, p, s, c, pos) in enumerate(log) if p == "land" and not s)
    assert all(not s for _, p, s, *_ in log[first_auto:])            # never streams again
    assert not px4.armed and px4.landed


def test_route_is_in_the_takeoff_frame():
    m, px4, log = fly(C.Params(route=[(2.0, 0.0)]), yaw0=math.radians(90))
    assert m.result == "complete"
    assert np.allclose(m.targets[0], (0.0, 2.0), atol=1e-9)          # facing east: forward = +e


def test_no_health_never_arms():
    m, px4, log = fly(C.Params(route=ROUTE, wait_s=20.0), health=lambda t: None)
    assert m.result.startswith("refused") and not px4.armed
    assert all("arm" not in c for *_x, c, _p in log)


def test_unhealthy_estimator_in_the_air_lands():
    m, px4, log = fly(C.Params(route=ROUTE), health=lambda t: t < 12.0)
    assert m.result.startswith("landed: estimator"), m.result
    assert m.legs_done < 2 and not px4.armed


def test_health_gap_restarts_the_quiet_window():
    h = C.Health(quiet_s=3.0, stale_s=1.0)
    for k in range(40):
        h.on_msg(k * 0.1, True)
    assert h.arm_ok(3.95)[0]
    h.on_msg(6.0, True)                                  # 2 s silent, then healthy again
    assert not h.arm_ok(6.05)[0]


def test_no_translation_without_live_depth_and_landing_when_it_stays_silent():
    # depth stops at t = 14 s (mid-route): the carrot freezes, then the aircraft lands
    m, px4, log = fly(C.Params(route=[(6.0, 0.0)], speed_mps=0.5), depth_on=lambda t: t < 14.0)
    assert m.result.startswith("landed: depth silent"), m.result
    pos_at_cut = [pos for t, p, s, c, pos in log if abs(t - 14.6) < 0.026][0]
    before_land = [pos for t, p, s, c, pos in log if p == "go" and t > 15.0]
    assert before_land and max(abs(q[0] - pos_at_cut[0]) for q in before_land) < 0.3
    assert m.counts["depth_holds"] > 0


def test_obstacle_ahead_stops_and_lands_where_it_is():
    wall = [(0, 4.5, -5.0, 5.0)]                          # a wall across the route at n = 4.5
    m, px4, log = fly(C.Params(route=[(8.0, 0.0)]), walls=wall)
    assert m.result.startswith("landed: route blocked"), m.result
    assert px4.pos[0] < 4.5 - 0.6                         # stopped short of the wall
    assert m.counts["obstacle_stops"] == 1


def test_a_wall_beside_the_route_is_not_an_obstacle():
    walls = [(1, 1.0, -5.0, 10.0), (1, -1.0, -5.0, 10.0)]   # a 2 m wide corridor along the route
    m, px4, log = fly(C.Params(route=[(5.0, 0.0)]), walls=walls)
    assert m.result == "complete", m.result


def test_pilot_takeover_is_never_fought():
    # 13 s in, someone switches PX4 to Position mode (nav_state 2)
    m, px4, log = fly(C.Params(route=ROUTE), nav_override=lambda t: 2 if t >= 13.0 else None)
    assert m.result.startswith("yielded"), m.result
    after = [(s, c) for t, p, s, c, pos in log if t > 13.0]
    assert all(not s and not c for s, c in after)


def test_same_decisions_on_any_clock():
    # a tick that is exact in binary (1/32 s) and thresholds off its grid: at Unix-time magnitudes
    # a 50 ms tick is not representable, and a decision that lands EXACTLY on a threshold may
    # round either way - float rounding of the inputs, not a dependence on the clock
    prm = dict(route=ROUTE, health_quiet_s=3.013, stream_lead_s=1.013, hold_s=1.013, cmd_resend_s=1.013)
    a = fly(C.Params(**prm), t0=0.0, dt=1 / 32)
    b = fly(C.Params(**prm), t0=1.76e9, dt=1 / 32)       # Unix time, as on hardware
    pa = [(round(t, 3), p, c) for t, p, s, c, pos in a[2]]
    pb = [(round(t, 3), p, c) for t, p, s, c, pos in b[2]]
    assert pa == pb and a[0].result == b[0].result == "complete"


# ------------------------------------------------------------------ depth.py
def test_clearance_levels_a_tilted_aircraft_and_ignores_the_floor():
    walls = [(0, 3.0, -5.0, 5.0)]
    pos = (0.0, 0.0, -1.2)
    img = render_depth(pos, 0.0, walls)
    for pitch_deg in (0.0, 6.0):                          # same image, attitude claims a nose-up tilt
        q = (math.cos(math.radians(pitch_deg) / 2), 0.0, math.sin(math.radians(pitch_deg) / 2), 0.0)
        pts = D.level_points(img, K, R_BODY_CAM, T_BODY_CAM, D.level_rotation(q), stride=2)
        clr = D.clearance(pts, 0.0, 0.6, 0.4)
        if pitch_deg == 0.0:
            assert clr == pytest.approx(3.0, abs=0.05)
    floor_only = render_depth(pos, 0.0, [])
    pts = D.level_points(floor_only, K, R_BODY_CAM, T_BODY_CAM, D.level_rotation((1.0, 0, 0, 0)), stride=2)
    assert math.isinf(D.clearance(pts, 0.0, 0.6, 0.4))


def test_bearing_selects_the_corridor():
    pts = np.array([[2.0, 1.5, 0.0]] * 5)                 # 2 m ahead, 1.5 m to the right
    assert math.isinf(D.clearance(pts, 0.0, 0.6, 0.4))
    assert D.clearance(pts, math.atan2(1.5, 2.0), 0.6, 0.4) == pytest.approx(2.5)


def test_decode_depth_encodings():
    mm = np.array([[0, 1000], [2500, 65535]], np.uint16)
    a = D.decode_depth("16UC1", 2, 2, 4, mm.tobytes())
    assert np.isnan(a[0, 0]) and a[0, 1] == pytest.approx(1.0) and a[1, 0] == pytest.approx(2.5)
    m = np.array([[np.nan, 1.5], [0.0, np.inf]], np.float32)
    b = D.decode_depth("32FC1", 2, 2, 8, m.tobytes())
    assert b[0, 1] == pytest.approx(1.5) and np.isnan(b[0, 0]) and np.isnan(b[1, 0]) and np.isnan(b[1, 1])
    with pytest.raises(ValueError):
        D.decode_depth("rgb8", 1, 1, 3, b"\0\0\0")


def test_a_one_frame_false_match_does_not_stop_the_route():
    hits = []

    def phantom(t):                                   # one frame every ~2 s while flying
        on = 12.0 < t and int(t * 7.5) % 15 == 0
        if on:
            hits.append(t)
        return on
    prm = dict(route=[(4.0, 0.0), (4.0, 3.0)], depth_border_px=2, min_obstacle_px=1200)   # 24 sampled points: big enough
    m, px4, log = fly(C.Params(**prm), phantom=phantom)
    assert hits and m.result == "complete", m.result
    assert m.counts["unconfirmed_obstacles"] > 0       # it saw them, and did not believe a single frame
    one = fly(C.Params(**dict(prm, confirm_frames=1)), phantom=phantom)[0]
    assert one.result.startswith("landed: route blocked")   # without confirmation the same flight stops


def test_border_crop_removes_edge_rows():
    img = np.full((H, W), np.nan, np.float32)
    img[0:3, 60:100] = 0.6
    q = (1.0, 0.0, 0.0, 0.0)
    keep = D.level_points(img, K, R_BODY_CAM, T_BODY_CAM, D.level_rotation(q), stride=1, border_px=0)
    crop = D.level_points(img, K, R_BODY_CAM, T_BODY_CAM, D.level_rotation(q), stride=1, border_px=4)
    assert len(keep) == 120 and len(crop) == 0


def test_sparse_points_are_not_an_obstacle_but_a_pole_is():
    """min_obstacle_px: a few scattered depth returns do not stop the aircraft; a 5 cm pole does."""
    q = (1.0, 0.0, 0.0, 0.0)
    m = C.Mission(C.Params(route=[(3.0, 0.0)]))
    m.leg_heading = 0.0
    img = np.full((800, 1280), 5.0, np.float32)           # the real camera's size, a wall 5 m ahead
    Kr = (842.0, 842.0, 640.0, 400.0)
    img[300:308, 600:616] = 0.63                           # 8 x 16 px of junk, persistent
    v = C.Vehicle(heading=0.0)
    for t in (1.0, 1.1):                                   # the mission evaluates every new frame
        m.on_depth(t, img, Kr, R_BODY_CAM, T_BODY_CAM, q)
        c = m._clearance(v)
    assert c == pytest.approx(5.16, abs=0.05)             # the wall, not the junk at 0.63 m
    pole = np.full((800, 1280), 5.0, np.float32)
    w = int(0.05 / 3.0 * 842) + 1                          # a 5 cm pole 3 m ahead
    pole[100:700, 640 - w // 2:640 + w // 2 + 8] = 3.0
    for t in (1.2, 1.3):
        m.on_depth(t, pole, Kr, R_BODY_CAM, T_BODY_CAM, q)
        c = m._clearance(v)
    assert c == pytest.approx(3.16, abs=0.05)


def test_a_px4_failsafe_landing_mid_route_is_never_streamed_over():
    """PX4 switching itself to AUTO_LAND during the route (a failsafe) is a takeover like a pilot's:
    the mission stops streaming at once (PX4's land detector reads the setpoint topic)."""
    m, px4, log = fly(C.Params(route=ROUTE), nav_override=lambda t: C.NAV_AUTO_LAND if t >= 13.0 else None)
    assert m.result.startswith("yielded"), m.result
    assert all(not s and not c for t, p, s, c, pos in log if t > 13.0)



# ------------------------------------------------- unusable depth is never permission to move
def _advance(log, t_from):
    """How far the aircraft moved forward (north) after t_from."""
    p = [pos for t, ph, st, c, pos in log if t >= t_from]
    return max(q[0] for q in p) - p[0][0] if p else 0.0


@pytest.mark.parametrize("kind", ["blank", "sparse", "future", "stale"])
def test_unusable_depth_never_moves_the_aircraft_and_ends_in_a_landing(kind):
    """From t = 11 s (the route has just started) every depth frame is unusable: blank (no
    return anywhere), sparse (5 % of pixels return), stamped 1 s in the future, or stamped 1 s
    in the past. The aircraft must not move along the route, and must land when no usable depth
    has arrived for depth_lost_land_s."""
    rng = np.random.default_rng(0)

    def bad(t, img):
        if t < 11.0:
            return img
        if kind == "blank":
            return np.full_like(img, np.nan)
        if kind == "sparse":
            out = np.full_like(img, np.nan)
            m = rng.random(img.shape) < 0.05
            out[m] = img[m]
            return out
        return img
    m = C.Mission(C.Params(route=[(6.0, 0.0)], speed_mps=0.5, depth_border_px=3))
    px4 = FakePX4()
    log, t, next_depth = [], 0.0, 0.0
    while t < 40.0 and m.phase != "done":
        m.on_health(t, True)
        if t >= next_depth:
            img = bad(t, render_depth(px4.pos, px4.yaw, ROOM))
            stamp = t
            if t >= 11.0 and kind == "future":
                stamp = t + 1.0
            if t >= 11.0 and kind == "stale":
                stamp = t - 1.0
            m.on_depth(stamp, img, K, R_BODY_CAM, T_BODY_CAM, q_yaw(px4.yaw), now=t)
            next_depth = t + 1 / 7.5
        out = m.step(t, px4.vehicle())
        log.append((t, out.phase, out.stream, tuple(out.commands), px4.pos.copy()))
        px4.apply(t, out, 0.05)
        t += 0.05
    t_go = next(t for t, ph, *_ in log if ph == "go")
    assert t_go < 11.0, t_go                              # it was flying the route when depth went bad
    pos_at_11 = next(pos for t, *_r, pos in log if t >= 11.0)
    ahead = max(pos[0] for t, *_r, pos in log if t >= 11.0) - pos_at_11[0]
    # what was already commanded (one carrot lead at most) may complete; nothing beyond it
    assert ahead < m.p.lead_max_m + 0.05, ahead
    assert m.result.startswith("landed: depth silent"), m.result
    key = {"blank": "depth_unusable", "sparse": "depth_unusable", "future": "depth_future", "stale": "depth_stale"}[kind]
    assert m.counts[key] > 0


def test_a_blank_frame_does_not_refresh_depth_and_says_why():
    m = C.Mission(C.Params(route=[(3.0, 0.0)], depth_border_px=3))
    ok = m.on_depth(1.0, render_depth(np.array([0, 0, -1.2]), 0.0, ROOM), K, R_BODY_CAM, T_BODY_CAM, q_yaw(0.0), now=1.0)
    assert ok and m.depth_t == 1.0
    assert not m.on_depth(1.5, np.full((H, W), np.nan, np.float32), K, R_BODY_CAM, T_BODY_CAM, q_yaw(0.0), now=1.5)
    assert m.depth_t == 1.0 and m.counts["depth_unusable"] == 1


def test_a_corridor_out_of_view_is_unknown_not_free():
    """A good frame looking AWAY from the leg keeps the camera alive but is no evidence for the leg."""
    m = C.Mission(C.Params(route=[(3.0, 0.0)], depth_border_px=3))
    m.leg_heading = math.radians(90)                      # the leg is to the right; the camera looks ahead
    v = C.Vehicle(heading=0.0)
    for k in range(3):
        m.on_depth(1.0 + k * 0.1, render_depth(np.array([0, 0, -1.2]), 0.0, ROOM), K, R_BODY_CAM, T_BODY_CAM,
                   q_yaw(0.0), now=1.0 + k * 0.1)
        assert m._clearance(v) is None
    assert m.depth_fresh(1.25) and not m.corridor_fresh(1.25)
    assert m.counts["corridor_unobserved"] == 3


def test_coverage_counts_far_returns_and_ignores_missing_ones():
    q = q_yaw(0.0)
    near_wall = np.full((H, W), 9.0, np.float32)          # everything 9 m away: beyond 6 m, still a return
    f = D.DepthFrame(near_wall, K, R_BODY_CAM, T_BODY_CAM, D.level_rotation(q), stride=4, border_px=3)
    cov, n = f.corridor_coverage(0.0, 0.6, 0.4, 1.9)
    assert n > 10 and cov == pytest.approx(1.0)
    assert len(f.points(0.3, 6.0)) == 0                   # ... and no obstacle
    holed = near_wall.copy()
    holed[30:70, 60:100] = np.nan                         # the middle of the image returned nothing
    cov2, _ = D.DepthFrame(holed, K, R_BODY_CAM, T_BODY_CAM, D.level_rotation(q), stride=4, border_px=3) \
        .corridor_coverage(0.0, 0.6, 0.4, 1.9)
    assert cov2 < 0.6


# ---- the interactive runtime's takeoff/land example and landing requests (2026-10-17) ----------------
def test_takeoff_hover_land_with_an_empty_route():
    m, px4, log = fly(C.Params(route=[], height_m=1.2, hover_s=4.0))
    assert m.result == "complete", (m.result, m.why)
    hover = [t for t, p, *_ in log if p == "hover"]
    assert hover and hover[-1] - hover[0] >= 4.0 - 0.1               # held the whole hover_s
    assert max(-pos[2] for *_, pos in log) == pytest.approx(1.2, abs=0.2)
    assert not px4.armed and px4.landed


def test_hover_zero_is_the_old_behaviour():
    m, px4, log = fly(C.Params(route=ROUTE))
    assert "hover" not in {p for _, p, *_ in log} and m.result == "complete"


def test_a_landing_request_in_the_air_lands_through_the_mission():
    asked = []

    def ask(t, m):
        if m.phase == "hover" and not asked:
            asked.append(t)
            m.request_land("operator")
    m, px4, log = fly(C.Params(route=[], hover_s=30.0), on_tick=ask)
    assert asked and m.result == "landed: landing requested: operator", m.result
    land = [t for t, p, *_ in log if p == "land"]
    assert land[0] - asked[0] < 0.2                                   # at once, not after the hover
    assert not px4.armed and px4.landed


def test_a_landing_request_before_arming_never_arms():
    m, px4, log = fly(C.Params(route=[(3.0, 0.0)]), on_tick=lambda t, m: m.request_land("stop") if t > 0.5 else None)
    assert m.result.startswith("refused: landing requested: stop"), m.result
    assert not any("arm" in c for *_, c, _ in log) and not px4.armed


def _sky(t, img):
    """The field case (acc2 run_001): the camera works but sees mostly sky - depth on ~15 % of the pixels."""
    img = img.copy()
    img[: int(H * 0.85)] = np.nan
    return img


def test_too_little_depth_never_arms_and_says_how_much_there_was():
    """The wait names the measured fraction and the floor - from the start, and in the final refusal."""
    seen = []
    m, px4, log = fly(C.Params(route=[], wait_s=20.0), depth_fn=_sky, seconds=30.0,
                      on_tick=lambda t, m: seen.append(m.why) if 5.0 < t < 5.1 else None)
    assert not px4.armed
    assert "depth on " in seen[0] and "this mission needs 30% (min_valid_frac)" in seen[0], seen
    assert m.result.startswith("refused: waiting for live depth: the latest frame was refused: depth on"), m.result
    st = m.status(30.0)
    assert st["depth_valid_frac"] < 0.3 and "min_valid_frac" in st["depth_reject"]
    m2, px4b, _ = fly(C.Params(route=[], min_valid_frac=0.05), depth_fn=_sky, seconds=60.0)
    assert m2.result == "complete" and m2.depth_reject is None, (m2.result, m2.why)


def test_no_depth_at_all_is_said_as_such():
    m, px4, log = fly(C.Params(route=[], wait_s=5.0), depth_on=lambda t: False, seconds=10.0)
    assert m.result == "refused: waiting for live depth: no depth frame has arrived after 5 s", m.result


# ---- course corridors (coursenav POC): opt-in leg_end_margin_m -----------------------------------
MAZE = [(1, 0.5, -1.0, 4.0), (1, -0.5, -1.0, 4.0), (0, 3.0, -1.0, 1.0)]   # 1 m corridor, end wall at n = 3


def test_maze_corridor_leg_ending_before_a_wall_completes_with_the_margin():
    prm = C.Params(route=[(2.5, 0.0)], corridor_half_m=0.35, leg_end_margin_m=0.25, speed_mps=0.4, arrive_m=0.15)
    m, px4, log = fly(prm, walls=MAZE)
    assert m.result == "complete", (m.result, m.why)
    assert px4.pos[0] < 3.0 - 0.4                                       # never closer to the end wall than planned


def test_without_the_margin_the_same_maze_leg_is_called_blocked():
    """Why the opt-in exists: the end wall is closer than stop_m before the leg is done."""
    prm = C.Params(route=[(2.5, 0.0)], corridor_half_m=0.35, speed_mps=0.4, arrive_m=0.15)
    m, px4, log = fly(prm, walls=MAZE)
    assert m.result.startswith("landed: route blocked"), m.result


def test_with_the_margin_an_obstacle_on_the_leg_still_stops_it():
    wall = [(0, 2.0, -5.0, 5.0)]                                         # across the leg, 2 m ahead of a 4 m leg
    prm = C.Params(route=[(4.0, 0.0)], corridor_half_m=0.35, leg_end_margin_m=0.25)
    m, px4, log = fly(prm, walls=wall)
    assert m.result.startswith("landed: route blocked"), m.result
    assert px4.pos[0] < 2.0 - 0.5


# ---- map corrections (coursenav POC): opt-in, bounded, applied at leg starts -----------------------
def _fly_with_correction(prm, corr):
    """corr(t) -> (dn, de, sigma) or None, fed as a fresh correction each tick."""
    def tick(t, m):
        c = corr(t)
        if c is not None:
            m.on_correction(t, *c)
    return fly(prm, on_tick=tick)


def test_corrections_are_off_by_default():
    m, px4, log = _fly_with_correction(C.Params(route=[(2.0, 0.0), (2.0, 2.0)]), lambda t: (0.0, -0.3, 0.02))
    assert m.result == "complete" and m.counts["corrections"] == 0
    assert np.allclose(m.targets[1], (2.0, -2.0), atol=1e-9)


def test_a_correction_moves_only_the_remaining_targets_in_bounded_steps():
    prm = C.Params(route=[(2.0, 0.0), (2.0, 2.0), (4.0, 2.0)], correction_max_m=0.4)
    m, px4, log = _fly_with_correction(prm, lambda t: (0.0, -0.3, 0.02))
    assert m.result == "complete"
    assert m.counts["corrections"] == 2                                   # 0.15 + 0.15 = 0.30 m
    assert np.allclose(m.targets[0], (2.0, 0.0 - 0.15), atol=1e-9)        # leg 1 got the first step only
    assert np.allclose(m.targets[2], (4.0, -2.0 - 0.30), atol=1e-9)


def test_uncertain_or_stale_corrections_are_ignored():
    prm = C.Params(route=[(2.0, 0.0), (2.0, 2.0)], correction_max_m=0.4)
    m, *_ = _fly_with_correction(prm, lambda t: (0.0, -0.3, 0.5))           # sigma 0.5 m
    assert m.counts["corrections"] == 0
    m2 = C.Mission(prm)
    m2.on_correction(0.0, 0.0, -0.3, 0.02)
    m2.targets, m2.leg, m2.sp = [(2.0, 0.0)], 0, [0.0, 0.0, -1.0, 0.0]
    m2._apply_correction(5.0, C.Output("hold"))                            # 5 s old
    assert m2.counts["corrections"] == 0


def test_a_correction_is_capped_in_total():
    prm = C.Params(route=[(1.0, 0.0)] * 1 + [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (3.0, 2.0)], correction_max_m=0.2)
    m, *_ = _fly_with_correction(prm, lambda t: (0.0, -1.0, 0.02))
    assert 0.15 - 1e-9 <= math.hypot(*m.corr_applied) <= 0.2 + 1e-9     # never past the cap (deadband may leave < 0.06)


# ---- insufficient corridor coverage alone has no landing deadline -------------------------------
def _no_return_ahead(t, img):
    """Depth sees the floor but nothing at body height ahead (e.g. through a net): the corridor
    1.5 m ahead is never observed."""
    out = img.copy()
    out[: H // 2 + 8, :] = np.nan
    return out


def test_an_unobservable_corridor_hovers_forever_by_default():
    m, px4, log = fly(C.Params(route=[(3.0, 0.0)], stop_m=0.9, lead_max_m=0.6), depth_fn=_no_return_ahead,
                      seconds=60.0)
    assert m.phase == "check" and m.result is None                       # the original rule: wait
    assert m.counts["corridor_unobserved"] > 0 and px4.armed


@pytest.mark.parametrize("creep", [False, True])
def test_legacy_corridor_timeout_cannot_land_for_unobserved_space(creep):
    m, px4, log = fly(C.Params(route=[(3.0, 0.0)], stop_m=0.9, lead_max_m=0.6,
                              corridor_wait_land_s=10.0, recovery_replan=True, recovery_creep=creep),
                      depth_fn=_no_return_ahead, seconds=60.0)
    assert m.phase == "check" and m.result is None
    assert m.counts["corridor_unobserved"] > 0 and px4.armed
    assert m.legs_done == 0


def test_the_corridor_wait_does_not_fire_on_a_normal_route():
    m, px4, log = fly(C.Params(route=ROUTE, corridor_wait_land_s=2.0))
    assert m.result == "complete", (m.result, m.why)

