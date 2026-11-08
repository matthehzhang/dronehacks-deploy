"""Vertical centring and the altitude search against windows seen through the real depth intrinsics.

Rays are cast through host_depth's 512x320 image (f 305.3 px, the border and stride refnav uses) from a
level camera 0.046 m above the body origin, at a wall with a rectangular opening and a far wall behind it.
The P10W case is the sim layout whose window (0.914..1.626 m) sits wholly above the 0.8 m cruise height.
"""
import math

import numpy as np
import pytest

from refnav.core import Mission, Params, Vehicle, _wrap
from refnav.vertical import VerticalCentring, opening_target

F, W, H, BORDER, STRIDE, CAM_UP = 305.3, 512, 320, 24, 8, .046


def scene(dist, sill, header, body_h, offset=0., width=.9144, far=3., keep=1., seed=0, holes=None):
    """Level-frame points (x fwd, y right, z down) from the body origin at body_h above the floor.
    holes: [(sill, header)] for several openings in the same wall (default: the one given)."""
    holes = holes or [(sill, header)]
    v, u = np.mgrid[BORDER:H-BORDER:STRIDE, BORDER:W-BORDER:STRIDE]
    ry, rz = (u.ravel()-W/2)/F, (v.ravel()-H/2)/F          # right, down per metre forward
    cam_h = body_h + CAM_UP
    pts = []
    for y, z in zip(ry, rz):
        hit_h = cam_h - z*dist
        y_at = y*dist
        in_hole = any(s < hit_h < h for s, h in holes) and abs(y_at - offset) < width/2
        d = dist if not in_hole else dist + far
        h_d = cam_h - z*d
        if h_d < 0:                                          # floor first
            d = cam_h/z
        pts.append((d, y*d, -(cam_h - z*d) + body_h))       # z down relative to the body origin
    pts = np.array(pts)
    if keep < 1:
        pts = pts[np.random.default_rng(seed).random(len(pts)) < keep]
    return pts


def target(pts, h):
    return opening_target(pts, h, .8, .37, .24, .35, 1.5)


def test_raised_window_found_from_range_and_climbed_to():
    t, info = target(scene(2.0, .9144, 1.6256, .81), .81)
    assert info['kind'] == 'opening' and abs(t - (.9144 + 1.6256)/2) < .06


@pytest.mark.parametrize('offset', [-.15, .15])
def test_raised_window_off_centre_line(offset):
    t, info = target(scene(2.0, .9144, 1.6256, .81, offset=offset), .81)
    assert info['kind'] == 'opening' and abs(t - 1.27) < .06


def test_raised_window_with_sparse_sim_depth():
    t, info = target(scene(2.0, .9144, 1.6256, .81, keep=.27), .81)
    assert info['kind'] == 'opening' and abs(t - 1.27) < .08


def test_raised_window_up_close_is_a_face_not_an_opening():
    t, info = target(scene(.5, .9144, 1.6256, .81), .81)     # the opening is out of the camera's view
    assert t is None and info['kind'] == 'solid_face'


def test_original_window_still_centres_around_cruise():
    t, info = target(scene(1.5, .3048, 1.2192, .81), .81)
    assert info['kind'] == 'opening' and abs(t - (.3048 + 1.2192)/2) < .06


def test_two_openings_nearest_wins():
    t, _ = target(scene(2.0, 0, 0, .81, holes=[(.2, .7), (1.0, 1.55)]), .81)
    assert t is not None and abs(t - .45) < .06              # nearer to 0.81 than 1.275


def blocked_mission():
    """Hovering at cruise in front of a raised-window wall 0.5 m ahead, no route (search)."""
    m = Mission(Params(track_mode='path', local_horizon=True, height_m=.8, alt_rate_mps=10.))
    m.phase = 'go'; m.leg = 0; m.targets = [(9., 0.)]; m.waypoints_ne = [(4., 0.)]
    m.sp = [0., 0., -.8, 0.]; m.home = [0., 0., 0., 0.]
    m.leg_dir = (1., 0.); m.leg_len = 4.; m.leg_heading = 0.
    m.t_started = 0.; m.t_phase = 0.; m.t_last = 1.
    m.health.bad_for = lambda now: 0.; m.offboard_seen = False
    m.vc = VerticalCentring(.8, .37, .24, rate_mps=10.)
    m.last_v = Vehicle(pos=(0., 0., -.8), heading=0., pos_ok=True, vel=(0., 0., 0.))
    return m


def fly(m, t, alt_of_route=None, steps=3000, dt=.05):
    """Step the mission; the aircraft follows the setpoint exactly. A route appears once the aircraft is
    at a height inside alt_of_route (the navigator's band is the current height +/- 0.25 m)."""
    for _ in range(steps):
        t += dt
        h = -m.sp[2]
        m.depth_t = t
        m.depth_pts = scene(.5, .9144, 1.6256, h)
        v = Vehicle(pos=(m.sp[0], m.sp[1], -h), heading=m.sp[3], pos_ok=True, vel=(0., 0., 0.))
        m.last_v = v
        if alt_of_route and alt_of_route[0] <= h <= alt_of_route[1] and m.phase == 'search':
            m.on_path(t, 0, [(0., 0.), (2., 0.)], [.3, 0.], pid=1)
        m.step(t, v)
        if m.phase != 'search':
            return t
    return t


def test_blocked_face_triggers_altitude_ladder_and_holds_the_level_found():
    m = blocked_mission()
    t = fly(m, 2., steps=1)
    assert m.phase == 'search'
    seen = set()
    for _ in range(6000):                                    # no route at any height: ladder then cruise
        t += .05
        h = -m.sp[2]
        m.depth_t = t
        m.depth_pts = scene(.5, .9144, 1.6256, h)
        v = Vehicle(pos=(0., 0., -h), heading=m.sp[3], pos_ok=True, vel=(0., 0., 0.))
        m.step(t, v)
        seen.add(round(h, 2))
        assert m.phase == 'search' and m.sp[:2] == [0., 0.]
        assert .35 - 1e-6 <= h <= 1.5 + 1e-6
    assert max(seen) >= 1.25 and min(seen) < .8              # climbed, later tried below cruise too
    # a route appears at 1.3 m: the search ends there and that height is held through the face
    m = blocked_mission()
    t = fly(m, 2., alt_of_route=(1.2, 1.4))
    assert m.phase == 'turn'
    h = -m.sp[2]
    assert 1.2 <= h <= 1.4
    for k in range(20):                                       # moving forward, edges out of view
        t += .1
        v = Vehicle(pos=(.05*k, 0., -h), heading=0., pos_ok=True, vel=(.3, 0., 0.))
        m.depth_pts = np.zeros((0, 3))
        m.step(t, v)
        assert abs(-m.sp[2] - h) < 1e-6                       # not dragged back to cruise into the sill
    t += .1
    m.step(t, Vehicle(pos=(1.6, 0., -h), heading=0., pos_ok=True, vel=(.3, 0., 0.)))
    assert -m.sp[2] < h                                       # past the face: back toward cruise


def test_no_face_ahead_no_altitude_search():
    m = blocked_mission()
    t = 2.
    for _ in range(3000):
        t += .05
        m.depth_t = t
        m.depth_pts = np.zeros((0, 3))                       # open space: only headings are searched
        v = Vehicle(pos=(0., 0., -.8), heading=m.sp[3], pos_ok=True, vel=(0., 0., 0.))
        m.step(t, v)
        assert abs(-m.sp[2] - .8) < 1e-6


def test_search_keeps_turning_without_usable_depth():
    """No usable depth at any heading: the sweep still steps through its headings (a yaw in place needs no
    depth) instead of waiting at the first one for ever; course params: 45 deg steps, right first, <= 90."""
    m = Mission(Params(track_mode='path', local_horizon=True, search_step_deg=45., search_sweep_deg=90.,
                       search_blind_s=2.))
    m.phase = 'go'; m.leg = 0; m.targets = [(9., 0.)]; m.waypoints_ne = [(4., 0.)]
    m.sp = [0., 0., -.8, 0.]; m.home = [0., 0., 0., 0.]
    m.leg_dir = (1., 0.); m.leg_len = 4.; m.leg_heading = 0.
    m.t_started = 0.; m.t_phase = 0.; m.t_last = 1.
    m.health.bad_for = lambda now: 0.; m.offboard_seen = False
    m.last_v = Vehicle(pos=(0., 0., -.8), heading=0., pos_ok=True, vel=(0., 0., 0.))
    t, dwell = 2., []
    for _ in range(1200):                                   # 60 s, depth never fresh
        t += .05
        v = Vehicle(pos=(0., 0., -.8), heading=m.sp[3], pos_ok=True, vel=(0., 0., 0.))
        m.step(t, v)
        assert m.phase == 'search' and m.sp[:2] == [0., 0.]
        assert abs(_wrap(m.sp[3])) <= math.radians(90.) + 1e-6
        if not dwell or dwell[-1] != m.search_target:
            dwell.append(m.search_target)
    first = [round(math.degrees(_wrap(a))) for a in dwell[:5]]
    assert first == [0, 45, 90, -45, -90]                   # right (+, NED) first
    assert len(dwell) > 5                                    # and it keeps cycling


def test_course_file_uses_sparse_depth_and_45_degree_look_around():
    import json
    from pathlib import Path
    doc = json.loads((Path(__file__).resolve().parents[1]/'course/course_depth_waypoints.json').read_text())
    rp = doc['refnav_params']
    assert rp['min_valid_frac_static'] <= .05 and rp['min_valid_frac'] <= .1
    assert rp['search_step_deg'] == 45. and rp['search_sweep_deg'] == 90.


@pytest.mark.parametrize('floor', [.10, .18])
def test_altitude_search_goes_straight_to_window_height_above_the_floor(floor):
    """Blocked at the window: the first height tried is 1.35 m above the FLOOR (window centre ~55 in),
    whatever the landed body height; not a small step up that leaves the sill in the navigator's band."""
    m = blocked_mission()
    m.p.floor_below_home_m = floor
    t, first = 2., None
    for _ in range(3000):
        t += .05
        h = -m.sp[2]
        m.depth_t = t
        m.depth_pts = scene(.5, .9144, 1.6256, h + floor - .1)
        m.step(t, Vehicle(pos=(0., 0., -h), heading=m.sp[3], pos_ok=True, vel=(0., 0., 0.)))
        if m.vc.override is not None:
            first = m.vc.override
            break
    assert first is not None and abs(first + floor - 1.35) < 1e-9
