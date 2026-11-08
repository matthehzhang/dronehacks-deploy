"""Competition-profile options of refnav.core (all off by default; tests/test_refnav_core.py pins the
defaults). Each option: the transient case recovers, the persistent case still stops or lands.
The flight B / D / E stops of 2026-11-06 are recreated at the end."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refnav import core as C                                   # noqa: E402
from test_refnav_core import (FLOOR_D, H, K, R_BODY_CAM, ROOM, T_BODY_CAM, W, FakePX4,   # noqa: E402
                              fly, q_yaw, render_depth)

PROFILE = dict(speed_mps=0.3, lead_max_m=0.25, arrive_m=0.15, hold_s=0.3, yaw_rate_dps=30.0,
               accel_mps2=0.5, blend_turn_deg=30.0, blocked_hold_s=4.0, blocked_backoff_m=0.2, blocked_recheck_s=2.0,
               depth_lost_pause_static=True, depth_lost_land_s=5.0, stop_per_mps=1.0, stop_base_m=0.45, stop_min_m=0.55,
               min_valid_frac_static=0.15, min_corridor_coverage=0.45, confirm_frames=3, corridor_half_m=0.32,
               band_m=0.25, leg_end_margin_m=0.25, corridor_wait_land_s=10.0)


def profile(**kw):
    return C.Params(**dict(PROFILE, **kw))


# ---------------------------------------------------------------- A2: velocity profile
def test_velocity_profile_ramps_and_brakes_to_rest_at_a_leg_end():
    vel, phases = [], []
    m, px4, log = fly(profile(route=[(3.0, 0.0), (3.0, 3.0)]),
                      on_tick=lambda t, mm: phases.append((t, mm.phase, mm.v_cmd, mm.s, mm.leg_len)))
    assert m.result == "complete", (m.result, m.why)
    go = [(t, v, s, L) for t, ph, v, s, L in phases if ph == "go"]
    vmax = max(v for _, v, _, _ in go)
    assert 0.28 <= vmax <= 0.3 + 1e-9                        # cruises at the profile speed, never above
    # braking: inside the braking distance the commanded speed obeys v <= sqrt(2 a remaining) (+ one tick)
    for t, v, s, L in go:
        rem = L - s + v * 0.05                                # remaining distance when the speed was chosen
        assert v <= math.sqrt(2 * 0.5 * rem) + 1e-6, (t, v, rem)
    # the speed ramps up from rest: the first go tick is slow
    assert go[0][1] < 0.1
    # 90 deg corner: a full stop, so the aircraft arrives at rest
    end_leg1 = [v for t, ph, v, s, L in phases if ph == "hold"]
    assert end_leg1 and all(v == 0.0 for v in end_leg1)


def test_velocity_feed_forward_is_commanded_only_with_the_profile():
    outs = []

    def tick(t, mm):
        pass
    m, px4, log = fly(profile(route=[(2.0, 0.0)]), on_tick=tick)
    assert m.result == "complete"
    m2 = C.Mission(profile(route=[(2.0, 0.0)]))
    px = FakePX4()
    seen_vel = False
    t, nd = 0.0, 0.0
    while t < 40 and m2.phase != "done":
        m2.on_health(t, True)
        if t >= nd:
            m2.on_depth(t, render_depth(px.pos, px.yaw, ROOM), K, R_BODY_CAM, T_BODY_CAM, q_yaw(px.yaw), now=t)
            nd = t + 1 / 7.5
        out = m2.step(t, px.vehicle())
        if out.phase == "go" and out.velocity is not None and math.hypot(*out.velocity) > 0.2:
            seen_vel = True
        px.apply(t, out, 0.05)
        t += 0.05
    assert seen_vel
    m3, _, _ = fly(C.Params(route=[(2.0, 0.0)]))
    assert m3.result == "complete"


# ---------------------------------------------------------------- A2: blended turns
def test_a_small_corner_is_flown_without_stopping_and_a_big_one_still_stops():
    # 20 deg corner: forward 2 m, then 2 m at 20 deg left
    a = math.radians(20)
    route = [(2.0, 0.0), (2.0 + 2.0 * math.cos(a), 2.0 * math.sin(a))]
    m, px4, log = fly(profile(route=route))
    assert m.result == "complete", (m.result, m.why)
    assert m.counts["blended_turns"] == 1 and m.legs_done == 2
    phases = [p for _, p, *_ in log]
    holds = sum(1 for a, b in zip(phases, phases[1:]) if b == "hold" and a != "hold")
    assert holds == 1                                           # only the final leg holds
    m2, px4b, log2 = fly(profile(route=[(2.0, 0.0), (2.0, 2.0)]))   # 90 deg: stop, turn, check
    assert m2.result == "complete" and m2.counts["blended_turns"] == 0
    assert "turn" in [p for _, p, *_ in log2][[p for _, p, *_ in log2].index("go"):]


def test_gate_legs_never_blend():
    a = math.radians(20)
    route = [(2.0, 0.0), (2.0 + 2.0 * math.cos(a), 2.0 * math.sin(a))]
    legs = [{"name": "pre", "mode": "cruise"}, {"name": "gate", "mode": "gate", "hold": True}]
    m, px4, log = fly(profile(route=route, legs=legs))
    assert m.result == "complete" and m.counts["blended_turns"] == 0


def test_blending_never_moves_into_an_unobserved_corridor():
    """At a blended corner the new leg's corridor must be observed before the carrot advances."""
    a = math.radians(25)
    route = [(2.0, 0.0), (2.0 + 2.0 * math.cos(a), 2.0 * math.sin(a))]
    moved_before_evidence = []

    def tick(t, mm):
        if mm.phase == "go" and mm.leg == 1 and mm.s > 0 and mm.corr_t is None:
            moved_before_evidence.append(t)
    m, px4, log = fly(profile(route=route), on_tick=tick)
    assert m.result == "complete" and not moved_before_evidence


# ---------------------------------------------------------------- A3a: hold and recheck
def _wall_until(t_off, wall):
    def walls_fn(t):
        return [wall] if t < t_off else []
    return walls_fn


def fly_dynamic(params, walls_fn, seconds=120.0, min_range=None, on_tick=None):
    """Like fly(), with walls that may change over time (walls_fn(t, pos_ned)) and an optional camera
    minimum range."""
    if params.depth_border_px == C.Params().depth_border_px:
        params.depth_border_px = 3
    m = C.Mission(params)
    px4 = FakePX4()
    log, t, nd = [], 0.0, 0.0
    while t < seconds and m.phase != "done":
        m.on_health(t, True)
        if t >= nd:
            img = render_depth(px4.pos, px4.yaw, list(walls_fn(t, px4.pos)) + ROOM)
            if min_range is not None:
                img[img < min_range] = np.nan
            m.on_depth(t, img, K, R_BODY_CAM, T_BODY_CAM, q_yaw(px4.yaw), now=t)
            nd = t + 1 / 7.5
        if on_tick is not None:
            on_tick(t, m)
        out = m.step(t, px4.vehicle())
        log.append((t, out.phase, out.stream, tuple(out.commands), px4.pos.copy()))
        px4.apply(t, out, 0.05)
        t += 0.05
    return m, px4, log


def test_a_transient_obstacle_is_waited_out_and_the_route_resumes():
    """A wall appears across the route 2 m ahead at t = 12 s and vanishes 1.5 s later (a person,
    stereo noise, the net): the aircraft stops, hovers, and resumes without landing."""
    wall = (0, 2.5, -5.0, 5.0)
    seen = []

    def walls(t, pos):
        if not seen and pos[0] >= 1.9:                       # appears 0.6 m ahead (inside the 0.75 m stop)
            seen.append(t)
        return [wall] if seen and t < seen[0] + 1.5 else []
    m, px4, log = fly_dynamic(profile(route=[(4.0, 0.0)]), walls)
    assert m.result == "complete", (m.result, m.why)
    assert m.counts["obstacle_stops"] == 1 and m.counts["recoveries"] == 1
    assert "blocked" in [p for _, p, *_ in log] and "backoff" not in [p for _, p, *_ in log]


def test_a_persistent_obstacle_backs_off_rechecks_and_lands():
    wall = (0, 2.5, -5.0, 5.0)
    m, px4, log = fly_dynamic(profile(route=[(4.0, 0.0)]), lambda t, pos: [wall])
    assert m.result.startswith("landed: route blocked") and "recheck" in m.result, m.result
    phases = [p for _, p, *_ in log]
    assert "blocked" in phases and "backoff" in phases and "recheck" in phases
    n_at_block = next(pos[0] for t, p, s, c, pos in log if p == "blocked")
    n_at_land = next(pos[0] for t, p, s, c, pos in log if p == "land")
    assert 0.1 < n_at_block - n_at_land < 0.35                 # backed off about 0.2 m along the leg
    assert px4.pos[0] < 2.5 - 0.5 and not px4.armed
    t_block = next(t for t, p, *_ in log if p == "blocked")
    t_back = next(t for t, p, *_ in log if p == "backoff")
    assert 4.0 <= t_back - t_block < 4.3


def test_an_obstacle_that_clears_during_the_recheck_resumes():
    wall = (0, 2.5, -5.0, 5.0)
    appear = 12.0
    gone = []

    def walls(t, pos):
        return [wall] if (t >= appear and not gone) else []
    seen = {}

    def tick(t, mm):
        seen.setdefault(mm.phase, t)
        if mm.phase == "recheck" and not gone:
            gone.append(t)                                   # the obstacle leaves while rechecking
    m, px4, log = fly_dynamic(profile(route=[(4.0, 0.0)]), walls, on_tick=tick)
    assert m.result == "complete", (m.result, m.why)
    assert m.counts["recoveries"] == 1 and "recheck" in seen


def test_hold_and_recheck_is_off_by_default():
    m, px4, log = fly(C.Params(route=[(8.0, 0.0)]), walls=[(0, 4.5, -5.0, 5.0)])
    assert m.result.startswith("landed: route blocked") and "blocked" not in [p for _, p, *_ in log]


# ---------------------------------------------------------------- A3b: depth silence pauses while static
def _silent_while(phases_off, params, route=((2.0, 0.0), (2.0, 2.0)), seconds=90.0):
    """Depth stops while the mission is in one of phases_off (the camera 'sees' nothing there)."""
    holder = {}

    def tick(t, mm):
        holder["m"] = mm

    def depth_on(t):
        mm = holder.get("m")
        return mm is None or mm.phase not in phases_off
    return fly(params, depth_on=depth_on, on_tick=tick, seconds=seconds)


def test_depth_silence_during_an_in_place_turn_does_not_land():
    """A 90 deg turn at 30 deg/s takes 3 s: with the pause the silence is not counted and the route
    completes; without it (the original rule, 3 s limit) the same flight lands."""
    m, px4, log = _silent_while(("turn",), profile(route=[(2.0, 0.0), (2.0, 2.0)], yaw_rate_dps=20.0,
                                                   depth_lost_land_s=3.0))
    assert m.result == "complete", (m.result, m.why)
    m2, *_ = _silent_while(("turn",), C.Params(route=[(2.0, 0.0), (2.0, 2.0)], yaw_rate_dps=20.0))
    assert m2.result.startswith("landed: depth silent"), m2.result


def test_silence_while_moving_still_stops_translation_and_lands():
    m, px4, log = fly(profile(route=[(6.0, 0.0)]), depth_on=lambda t: t < 16.0)
    assert m.result.startswith("landed: depth silent"), m.result
    t_land = next(t for t, p, *_ in log if p == "land")
    assert 4.8 <= t_land - 16.0 < 5.3                        # depth_lost_land_s = 5 s after the last frame
    pos_at_cut = [pos for t, p, s, c, pos in log if abs(t - 16.5) < 0.026][0]
    moved = max(pos[0] for t, p, s, c, pos in log if t > 17.0) - pos_at_cut[0]
    assert moved < 0.3                                        # translation stopped at once (one lead at most)


def test_silence_that_persists_after_the_turn_still_lands():
    m, px4, log = _silent_while(("turn", "check", "go", "hold"), profile(route=[(2.0, 0.0), (2.0, 2.0)]))
    assert m.result.startswith("landed: depth silent"), m.result


# ---------------------------------------------------------------- A3d: stop distance follows speed
def test_stop_distance_scales_with_the_planned_speed():
    m = C.Mission(profile(route=[(3.0, 0.0)]))
    m.leg_len, m.leg_v_end = 3.0, 0.0
    assert m._stop_m(0.0) == pytest.approx(0.75)              # cruising at 0.3 m/s
    assert m._stop_m(3.0) == pytest.approx(0.55)              # at rest at the end: the floor
    assert C.Mission(C.Params(route=[(3.0, 0.0)]))._stop_m(0.0) == 0.9


def test_a_wall_just_past_the_leg_end_stops_a_fixed_0_9_m_rule_but_not_the_scaled_one():
    wall = [(0, 3.62, -5.0, 5.0)]                              # 0.62 m past the end of a 3 m leg
    base = dict(corridor_half_m=0.32, band_m=0.25, speed_mps=0.3, lead_max_m=0.25, arrive_m=0.15, accel_mps2=0.5)
    m_old, *_ = fly(C.Params(route=[(3.0, 0.0)], **base), walls=wall)
    assert m_old.result.startswith("landed: route blocked"), m_old.result
    m_new, px4, _ = fly(C.Params(route=[(3.0, 0.0)], stop_per_mps=1.0, **base), walls=wall)
    assert m_new.result == "complete", (m_new.result, m_new.why)
    assert px4.pos[0] < 3.62 - 0.5


def test_the_scaled_stop_still_stops_for_a_wall_on_the_leg():
    m, px4, _ = fly(profile(route=[(4.0, 0.0)], blocked_hold_s=0.0), walls=[(0, 2.0, -5.0, 5.0)])
    assert m.result.startswith("landed: route blocked"), m.result
    assert px4.pos[0] < 2.0 - 0.5


# ---------------------------------------------------------------- A3e: valid-pixel floor while static
def _sparse(frac):
    img = render_depth(np.array([0.0, 0.0, -1.2]), 0.0, [(0, 3.0, -5.0, 5.0)] + ROOM)   # a wall fills the view
    out = np.full_like(img, np.nan)
    rng = np.random.default_rng(1)
    keep = rng.random(img.shape) < frac
    out[keep] = img[keep]
    return out


def test_a_sparse_frame_counts_while_static_but_not_while_moving():
    m = C.Mission(profile(route=[(3.0, 0.0)], depth_border_px=3))
    m.phase = "check"
    assert m.on_depth(1.0, _sparse(0.2), K, R_BODY_CAM, T_BODY_CAM, q_yaw(0.0), now=1.0)
    m.phase = "go"
    assert not m.on_depth(1.2, _sparse(0.2), K, R_BODY_CAM, T_BODY_CAM, q_yaw(0.0), now=1.2)
    assert "min_valid_frac" in m.depth_reject[1] and "static" not in m.depth_reject[1]
    m.phase = "check"
    assert not m.on_depth(1.4, _sparse(0.05), K, R_BODY_CAM, T_BODY_CAM, q_yaw(0.0), now=1.4)   # below both floors
    m0 = C.Mission(C.Params(route=[(3.0, 0.0)], depth_border_px=3))
    m0.phase = "check"
    assert not m0.on_depth(1.0, _sparse(0.2), K, R_BODY_CAM, T_BODY_CAM, q_yaw(0.0), now=1.0)   # default: one floor


# ---------------------------------------------------------------- A3c: three confirming frames
def test_three_frame_confirmation_rejects_two_frame_phantoms():
    hits = []

    def phantom(t):                                       # two consecutive frames every ~2 s
        on = 12.0 < t and int(t * 7.5) % 15 in (0, 1)
        if on:
            hits.append(t)
        return on
    prm = dict(route=[(4.0, 0.0), (4.0, 3.0)], depth_border_px=2, min_obstacle_px=1200, confirm_frames=3)
    m, px4, log = fly(C.Params(**prm), phantom=phantom)
    assert hits and m.result == "complete", m.result
    two = fly(C.Params(**dict(prm, confirm_frames=2)), phantom=phantom)[0]
    assert two.result.startswith("landed: route blocked")


# ---------------------------------------------------------------- A7: event start on an armed vehicle
def test_an_armed_landed_vehicle_is_accepted_at_start_only_with_the_option():
    def armed_fly(params):
        if params.depth_border_px == C.Params().depth_border_px:
            params.depth_border_px = 3
        m = C.Mission(params)
        px4 = FakePX4()
        px4.armed = True                                      # the team armed it on the mark (rule 7.4)
        log, t, nd = [], 0.0, 0.0
        while t < 60 and m.phase != "done":
            m.on_health(t, True)
            if t >= nd:
                m.on_depth(t, render_depth(px4.pos, px4.yaw, ROOM), K, R_BODY_CAM, T_BODY_CAM, q_yaw(px4.yaw), now=t)
                nd = t + 1 / 7.5
            out = m.step(t, px4.vehicle())
            log.append((t, out.phase, tuple(out.commands)))
            px4.apply(t, out, 0.05)
            t += 0.05
        return m, px4, log
    m, px4, log = armed_fly(profile(route=[(2.0, 0.0)], accept_armed_start=True))
    assert m.result == "complete", (m.result, m.why)
    assert not any("arm" in c for _, _, c in log)              # never re-sends an arm command
    assert any("offboard" in c for _, _, c in log)
    m2, *_ = armed_fly(profile(route=[(2.0, 0.0)]))
    assert m2.result.startswith("yielded: vehicle armed before"), m2.result


def test_an_armed_vehicle_in_the_air_is_still_refused_with_the_option():
    """accept_armed_start only accepts a LANDED armed vehicle: an airborne one is someone else's flight."""
    m = C.Mission(profile(route=[(2.0, 0.0)], accept_armed_start=True))
    v = C.Vehicle(pos=(0, 0, -1.0), heading=0.0, pos_ok=True, q=q_yaw(0.0), armed=True, nav_state=2, landed=False)
    out = m.step(0.0, v)
    assert m.result.startswith("yielded"), m.result


# ---------------------------------------------------------------- recreations of the 2026-11-06 stops
def test_flight_b_recreated_a_corner_0_33_m_beside_the_leg():
    """Flight B: a maze corner 0.32 m beside the leg line blocked leg 1 with a +-0.35 m corridor.
    The course profile's corridor is +-0.32 m with 0.40 m planned clearance: the corner is outside."""
    corner = [(1, 0.33, 1.0, 1.4)]                            # a wall edge 0.33 m to the right, 1-1.4 m ahead
    old, *_ = fly(C.Params(route=[(3.0, 0.0)], corridor_half_m=0.35, band_m=0.4, speed_mps=0.3), walls=corner)
    assert old.result.startswith("landed: route blocked"), old.result
    new, *_ = fly(profile(route=[(3.0, 0.0)]), walls=corner)
    assert new.result == "complete", (new.result, new.why)


def test_flight_d_recreated_drift_puts_a_real_wall_in_the_corridor():
    """Flight D: VIO drift of 0.23 m put the C1 wall end inside the corridor on leg 13. A real wall
    stays: the profile holds, backs off, rechecks and then lands. Still a correct stop."""
    wall = (1, 0.25, 1.2, 2.2)                               # a wall edge well inside the corridor, ahead
    m, px4, log = fly_dynamic(profile(route=[(3.0, 0.0)]), lambda t, pos: [wall])
    assert m.result.startswith("landed: route blocked") and "recheck" in m.result, m.result
    assert not px4.armed and px4.pos[0] < 1.2 - 0.4


def test_flight_e_recreated_turning_to_face_a_wall_inside_the_camera_blind_range():
    """Flight E: after arriving, the turn put the barrier 0.47 m ahead (0.35 m from the camera: inside
    its 0.4 m minimum range). Depth went to 5 % valid and refnav landed on 'depth silent' 3 s later.
    With the profile the silence is not counted during the turn, but the check after it still has
    no usable depth: it lands after depth_lost_land_s (5 s). The fix for E is the planner's leg-end
    rule (no leg may end facing a surface closer than 0.8 m), not refnav."""
    walls = [(1, -0.47, -1.0, 3.0)]                           # a wall 0.47 m to the LEFT, along the first leg
    m, px4, log = fly_dynamic(profile(route=[(2.0, 0.0), (2.0, 2.0)]), lambda t, pos: walls, min_range=0.4)
    assert m.legs_done == 1
    assert m.result.startswith("landed: depth silent"), m.result
    t_turn = next(t for t, p, *_ in log if p == "turn" and t > 5.0)
    t_land = next(t for t, p, *_ in log if p == "land")
    assert t_land - t_turn > 5.0                               # the turn itself (3 s) did not count


# ---------------------------------------------------------------- A4: gate alignment in the mission
WIN_W = 0.914


def _window_walls(e_centre, n_plane=3.0):
    """A wall across the route at n_plane with an opening WIN_W wide centred at e_centre."""
    return [(0, n_plane, -6.0, e_centre - WIN_W / 2), (0, n_plane, e_centre + WIN_W / 2, 6.0)]


def _gate_route(pre=2.2, plane=3.0, post=3.5):
    route = [(pre, 0.0), (plane, 0.0), (post, 0.0), (post + 1.5, 0.0)]
    legs = [{"name": "window_pre", "mode": "gate", "hold": True, "gate_group": "window"},
            {"name": "window", "mode": "gate", "hold": True, "gate_group": "window",
             "align": {"group": "window", "dist_m": plane - pre, "width_m": WIN_W}},
            {"name": "window_post", "mode": "gate", "hold": True, "gate_group": "window"},
            {"name": "after", "mode": "cruise"}]
    return route, legs


@pytest.mark.parametrize("e_true", [0.0, 0.18, -0.18])
def test_the_window_crossing_is_recentred_on_the_opening_measured_in_depth(e_true):
    """The route believes the opening is straight ahead (e = 0); in truth it is e_true to the side
    (a global position error). The alignment hover measures it and the crossing passes through the
    opening's centre, never the frame."""
    route, legs = _gate_route()
    walls = _window_walls(e_true)
    crossing = []

    def tick(t, mm):
        pass
    m, px4, log = fly_dynamic(profile(route=route, legs=legs), lambda t, pos: walls, seconds=150.0,
                              on_tick=lambda t, mm: crossing.append(px4_pos_holder[0].copy()) if mm.phase == "go" else None)
    assert m.result == "complete", (m.result, m.why)
    assert m.counts["alignments"] == 1 and m.last_align is not None
    assert m.last_align["lateral_m"] == pytest.approx(-e_true, abs=0.05)        # e+ is to the RIGHT (NED, heading 0)
    # the window centre target moved to the true opening
    assert m.targets[1][1] == pytest.approx(e_true, abs=0.05)
    # the aircraft crossed the plane inside the opening, near its centre
    at_plane = [pos for t, p, s, c, pos in log if abs(pos[0] - 3.0) < 0.05]
    assert at_plane and all(abs(q[1] - e_true) < 0.12 for q in at_plane), [q[1] for q in at_plane]


px4_pos_holder = [np.zeros(3)]


def test_a_missing_opening_holds_then_lands_and_never_crosses():
    route, legs = _gate_route()
    walls = [(0, 3.0, -6.0, 6.0)]                                # a blank wall where the window should be
    m, px4, log = fly_dynamic(profile(route=route, legs=legs), lambda t, pos: walls, seconds=150.0)
    assert m.result.startswith("landed: gate not found"), m.result
    assert m.counts["alignments"] == 0 and px4.pos[0] < 3.0 - 0.5
    assert "align" in [p for _, p, *_ in log] and "blocked" in [p for _, p, *_ in log]


def test_gate_alignment_is_inert_without_leg_metadata():
    route, _ = _gate_route()
    m, px4, log = fly_dynamic(profile(route=route), lambda t, pos: _window_walls(0.0), seconds=150.0)
    assert m.result == "complete" and "align" not in [p for _, p, *_ in log]


# ---------------------------------------------------------------- A5d: tunnel mode
TUBE_HALF = 0.45                 # narrow enough for the test harness's small camera to see both rims


def _tube(e_half=TUBE_HALF, n0=3.2, n1=4.7):
    """A straight passage along the route: two walls at +-e_half from n0 to n1 (its mouth at n0)."""
    return [(1, e_half, n0, n1), (1, -e_half, n0, n1)]


def _tunnel_route(with_align=True, with_tunnel=True, map_clear=True):
    route = [(2.0, 0.0), (5.5, 0.0), (7.0, 0.0)]
    passage = {"name": "tunnel_exit", "mode": "tunnel", "hold": True, "gate_group": "tunnel"}
    if with_align:
        passage["align"] = {"group": "tunnel", "dist_m": 1.2, "width_m": 2 * TUBE_HALF}
    if with_tunnel:
        passage["tunnel"] = {"length_m": 3.5, "map_clear": map_clear, "enclosed_frac": 0.43}
    legs = [{"name": "tunnel_entry", "mode": "align", "hold": True}, passage, {"name": "after", "mode": "cruise"}]
    return route, legs


def _dark_between(n_lo, n_hi):
    """Darkness: no depth at all while the aircraft is between n_lo and n_hi (inside the tube)."""
    def dark(t, img, pos):
        return np.full_like(img, np.nan) if n_lo <= pos[0] <= n_hi else img
    return dark


def fly_dark(params, walls_fn, dark_fn, seconds=150.0, health=lambda t, pos: True):
    if params.depth_border_px == C.Params().depth_border_px:
        params.depth_border_px = 3
    m = C.Mission(params)
    px4 = FakePX4()
    log, t, nd, dark_advance = [], 0.0, 0.0, []
    while t < seconds and m.phase != "done":
        m.on_health(t, health(t, px4.pos))
        if t >= nd:
            img = dark_fn(t, render_depth(px4.pos, px4.yaw, list(walls_fn(t, px4.pos)) + ROOM), px4.pos)
            m.on_depth(t, img, K, R_BODY_CAM, T_BODY_CAM, q_yaw(px4.yaw), now=t)
            nd = t + 1 / 7.5
        s_before = m.s
        out = m.step(t, px4.vehicle())
        if out.phase == "go" and m.s > s_before and not m.depth_fresh(t):
            dark_advance.append((t, px4.pos[0]))
        log.append((t, out.phase, out.stream, tuple(out.commands), px4.pos.copy()))
        px4.apply(t, out, 0.05)
        t += 0.05
    return m, px4, log, dark_advance


TUNNEL = dict(tunnel_mode=True, tunnel_speed_mps=0.3, tunnel_max_leg_m=4.0, tunnel_align_max_age_s=6.0)


def test_tunnel_mode_advances_through_a_dark_passage_only_when_every_condition_holds():
    route, legs = _tunnel_route()
    m, px4, log, dark = fly_dark(profile(route=route, legs=legs, **TUNNEL), lambda t, pos: _tube(),
                                 _dark_between(3.1, 4.8))
    assert m.result == "complete", (m.result, m.why)
    assert m.counts["alignments"] == 1 and m.counts["tunnel_ticks"] > 0 and dark
    assert min(n for _, n in dark) >= 3.0 and max(n for _, n in dark) <= 4.9      # advanced only inside
    assert not px4.armed and px4.pos[0] > 6.5


def test_tunnel_mode_is_off_by_default_so_darkness_lands_safely():
    route, legs = _tunnel_route()
    m, px4, log, dark = fly_dark(profile(route=route, legs=legs), lambda t, pos: _tube(), _dark_between(3.1, 4.8))
    assert m.result.startswith("landed: depth silent"), m.result
    assert not dark and m.counts["tunnel_ticks"] == 0


def test_tunnel_mode_refuses_a_leg_the_planner_did_not_mark_as_a_passage():
    route, legs = _tunnel_route(with_tunnel=False)
    m, px4, log, dark = fly_dark(profile(route=route, legs=legs, **TUNNEL), lambda t, pos: _tube(),
                                 _dark_between(3.1, 4.8))
    assert m.result.startswith("landed: depth silent") and not dark, m.result


def test_tunnel_mode_refuses_without_a_fresh_alignment_at_the_mouth():
    route, legs = _tunnel_route(with_align=False)
    m, px4, log, dark = fly_dark(profile(route=route, legs=legs, **TUNNEL), lambda t, pos: _tube(),
                                 _dark_between(3.1, 4.8))
    assert m.result.startswith("landed: depth silent") and not dark, m.result
    route, legs = _tunnel_route()
    stale = dict(TUNNEL, tunnel_align_max_age_s=0.3)              # the mouth is ~3 s past the alignment
    m2, px4b, log2, dark2 = fly_dark(profile(route=route, legs=legs, **stale), lambda t, pos: _tube(),
                                     _dark_between(3.1, 4.8))
    assert m2.result.startswith("landed: depth silent") and not dark2, m2.result


def test_tunnel_mode_refuses_when_the_mouth_is_not_the_expected_opening():
    """The walls are 0.4 m apart where a 0.9 m bore was expected: alignment fails, no crossing."""
    route, legs = _tunnel_route()
    m, px4, log, dark = fly_dark(profile(route=route, legs=legs, **TUNNEL), lambda t, pos: _tube(e_half=0.2),
                                 _dark_between(3.1, 4.8))
    assert m.result.startswith("landed: gate not found"), m.result
    assert not dark and px4.pos[0] < 3.2 - 0.5


def test_tunnel_mode_refuses_when_the_map_is_not_clear():
    route, legs = _tunnel_route(map_clear=False)
    m, px4, log, dark = fly_dark(profile(route=route, legs=legs, **TUNNEL), lambda t, pos: _tube(),
                                 _dark_between(3.1, 4.8))
    assert m.result.startswith("landed: depth silent") and not dark, m.result


def test_a_real_return_inside_the_passage_still_stops():
    """Something in the tube (a wall across it at n = 4.1) comes into view once the dark part is
    behind: the normal obstacle rule applies, hold-and-recheck, then a safe landing."""
    route, legs = _tunnel_route()
    walls = _tube() + [(0, 4.3, -0.45, 0.45)]
    m, px4, log, dark = fly_dark(profile(route=route, legs=legs, **TUNNEL), lambda t, pos: walls,
                                 _dark_between(3.1, 3.6))
    assert m.result.startswith("landed: route blocked"), m.result
    assert dark and px4.pos[0] < 4.3 - 0.4 and not px4.armed


def test_an_unhealthy_estimator_inside_the_passage_lands():
    route, legs = _tunnel_route()
    m, px4, log, dark = fly_dark(profile(route=route, legs=legs, **TUNNEL), lambda t, pos: _tube(),
                                 _dark_between(3.1, 4.8), health=lambda t, pos: pos[0] < 3.4)
    assert m.result.startswith("landed: estimator"), m.result
    assert dark and px4.pos[0] < 4.8 and not px4.armed


def test_tunnel_mode_logs_entry_and_exit():
    route, legs = _tunnel_route()
    events = []
    m = C.Mission(profile(route=route, legs=legs, **TUNNEL))
    px4 = FakePX4()
    t, nd = 0.0, 0.0
    dark = _dark_between(3.1, 4.8)
    while t < 150 and m.phase != "done":
        m.on_health(t, True)
        if t >= nd:
            m.on_depth(t, dark(t, render_depth(px4.pos, px4.yaw, _tube() + ROOM), px4.pos), K, R_BODY_CAM, T_BODY_CAM,
                       q_yaw(px4.yaw), now=t)
            nd = t + 1 / 7.5
        out = m.step(t, px4.vehicle())
        events += out.events
        px4.apply(t, out, 0.05)
        t += 0.05
    assert m.result == "complete"
    assert any("tunnel mode: entered" in e for e in events) and any("tunnel mode: left" in e for e in events)
