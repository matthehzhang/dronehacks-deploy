"""The recovery sweep faces the route the navigator may rejoin, not just the nose (recovery_scan_route_m).

DIVERSE01 (range_ray_fix sim): the blocked route ran 36-73 deg right of the nose from the rejoin
lookahead on, at or past the 38 deg edge of the field of view; a +/-25 deg sweep about the nose
certified no rejoin, five requests failed and the aircraft landed.
"""
import math
from dataclasses import replace

import numpy as np
import pytest
from refnav import core as C
from test_refnav_blocked import dense, params, speeds
from test_recovery_controller import step


def blocked_on(corners, heading=0., pos=(1., 0.), s0=1., **kwargs):
    pts, stops = dense(corners)
    m = C.Mission(params(recovery_replan=True, recovery_scan_deg=25., **kwargs))
    m.phase = 'go'; m.targets = [tuple(corners[-1])]; m.home = (0., 0., 0., 0.)
    m.sp = [pos[0], pos[1], -1., heading]; m.leg_start = (0., 0.); m.leg_dir = (1., 0.); m.leg_len = 4.
    m.offboard_seen = True; m.t_last = 10.; m.t_started = 0.; m.t_phase = 9.
    v = C.Vehicle(pos=(pos[0], pos[1], -1.), heading=heading, pos_ok=True, armed=True,
                  nav_state=C.NAV_OFFBOARD, vel=(0., 0., 0.), landed=False)
    m.on_path(10., 0, pts, speeds(pts, stops), stops, pid=1); m.path_s = s0
    m.depth_t = 10.; m.depth_pts = np.array([[2., 0., 0.]])
    m.health.on_msg(10., True)
    m._start_recover(10., C.Output('go'), 'blocked', 'obstacle', v)
    return m, v


def sweep(m, v):
    step(m, v, 14.1)
    assert m.phase == 'recovery_scan'
    return [math.degrees(y) for y in m.recover['scan_yaws']]


RIGHT_TURN = [(0., 0.), (1.2, 0.), (2.7, 1.5), (2.7, 3.5)]   # turns ~45 deg right just past the block


def test_default_is_unchanged_sweep_about_the_heading_at_the_block():
    m, v = blocked_on(RIGHT_TURN)
    assert sweep(m, v) == pytest.approx([-25., 25., 0.])
    assert m.status(14.1)['recover']['scan_centre_deg'] == 0.


def test_sweep_centres_on_route_bearing_and_ends_facing_it():
    m, v = blocked_on(RIGHT_TURN, recovery_scan_route_m=1.5)
    yaws = sweep(m, v)
    centre = yaws[-1]
    # the rejoin window [s0+1 m, s0+2.5 m] lies 30-60 deg right of the aircraft
    assert 30. < centre < 60.
    assert yaws == pytest.approx([centre - 25., centre + 25., centre])
    assert m.status(14.1)['recover']['scan_centre_deg'] == pytest.approx(centre, abs=.1)
    # the far edge of the sweep now covers the whole window plus the camera's 38 deg half field of view
    pts, _ = dense(RIGHT_TURN)
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    window = pts[(s >= 2.) & (s <= 3.5)] - [1., 0.]
    bearings = np.degrees(np.arctan2(window[:, 1], window[:, 0]))
    assert np.all(np.abs(bearings - centre) < 25. + 38.)


def test_straight_route_keeps_the_nose():
    m, v = blocked_on([(0., 0.), (4., 0.)], recovery_scan_route_m=1.5)
    assert sweep(m, v) == pytest.approx([-25., 25., 0.], abs=1e-6)


def test_left_turn_is_mirrored():
    left = [(n, -e) for n, e in RIGHT_TURN]
    m, v = blocked_on(left, recovery_scan_route_m=1.5)
    assert -60. < sweep(m, v)[-1] < -30.


def test_turn_is_bounded():
    behind = [(0., 0.), (1.05, 0.), (1.05, 0.4), (-2., 0.4)]   # the route doubles back behind the aircraft
    m, v = blocked_on(behind, recovery_scan_route_m=1.5, recovery_scan_route_max_deg=60.)
    assert abs(sweep(m, v)[-1]) <= 60. + 1e-6


def test_heading_wraps_through_north():
    # facing 170 deg (south-ish), route bends further right through 180
    corners = [(1., 0.), (0., .18), (-.2, .2), (-1.7, -1.3)]
    pts, _ = dense(corners)
    m, v = blocked_on(corners, heading=math.radians(170.), pos=(0., .18), s0=1.02, recovery_scan_route_m=1.5)
    centre = sweep(m, v)[-1]
    off = (centre - 170. + 180.) % 360. - 180.
    assert 20. < off < 70.
    assert -180. <= centre <= 180.


def test_after_a_hop_the_window_starts_from_the_current_position_not_the_block():
    # Blocked at s0=1 on a route that runs north, then turns 90 deg right at n=3.
    corners = [(0., 0.), (3., 0.), (3., 3.)]
    m, v = blocked_on(corners, recovery_scan_route_m=1.5)
    at_block = m._route_scan_centre(m.recover, v)
    # a hop carried the aircraft to (2.6, 0.3), a step ahead and beside the route, still facing north
    hopped = replace(v, pos=(2.6, .3, -1.))
    m.recover.update(pos=hopped.pos[:2], yaw=0.)
    after = m._route_scan_centre(m.recover, hopped)
    # at the block the window [2, 3.5] m is mostly straight ahead; after the hop, with the window starting
    # from the projected progress (s=2.6) it lies round the corner, to the right
    assert abs(math.degrees(at_block)) < 25.
    assert math.degrees(after) > 50.
    # and the reference the navigator gets is still the original one
    assert m.recover['old_path_s'] == 1.


def test_window_never_starts_behind_the_block():
    m, v = blocked_on(RIGHT_TURN, recovery_scan_route_m=1.5)
    behind = replace(v, pos=(.3, 0., -1.))        # drifted back toward the leg start (s ~ 0.3 < s0 = 1)
    got = m._route_scan_centre(m.recover, behind)
    # expected: the window stays [s0 + 1, s0 + 2.5] = [2, 3.5], bearings taken from the drifted position
    pts, _ = dense(RIGHT_TURN)
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    q = np.linspace(2., 3.5, 9)
    d = np.stack([np.interp(q, s, pts[:, 0]) - .3, np.interp(q, s, pts[:, 1])], axis=1)
    d /= np.linalg.norm(d, axis=1)[:, None]
    want = math.atan2(d[:, 1].mean(), d[:, 0].mean())
    assert got == pytest.approx(want, abs=1e-9)


def test_no_reference_route_falls_back_to_heading():
    m, v = blocked_on(RIGHT_TURN, recovery_scan_route_m=1.5)
    m.recover['old_path'] = None
    assert sweep(m, v) == pytest.approx([-25., 25., 0.])


def test_route_scan_param_is_bounded():
    with pytest.raises(ValueError, match='recovery_scan_route_m'):
        C.Mission(params(recovery_replan=True, recovery_scan_route_m=3.5))
    with pytest.raises(ValueError, match='recovery_scan_route_max_deg'):
        C.Mission(params(recovery_replan=True, recovery_scan_route_max_deg=float('nan')))


def test_last_request_is_sent_facing_the_route():
    m, v = blocked_on(RIGHT_TURN, recovery_scan_route_m=1.5)
    sweep(m, v)
    centre = m.recover['scan_yaws'][-1]
    sent = []
    for t in np.arange(14.15, 40., .05):
        v = replace(v, pos=tuple(m.sp[:3]), heading=m.sp[3])
        rid = None if m.replan_request is None else m.replan_request['id']
        out = step(m, v, float(t))
        if m.replan_request is not None and m.replan_request['id'] != rid:
            sent.append((m.replan_request['id'], v.heading))
        if out.phase != 'recovery_scan':
            break
    assert sent, 'the sweep sent no requests'
    assert abs(C._wrap(sent[-1][1] - centre)) < math.radians(3.5)
