"""Search look-around: sparse depth is used, and a heading with no usable depth is passed over."""
import math

from refnav.core import Mission, Params, Vehicle, _wrap


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
