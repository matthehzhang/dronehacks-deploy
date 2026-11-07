"""The safe course policy (livenav.safe) and its node hooks. Offline only; nothing here has flown."""
import math

import numpy as np
import pytest

from kestrel_nav.livenav import safe as safe_mod
from kestrel_nav.livenav.mission import MissionParams
from kestrel_nav.livenav.offline_sim import OfflineFlight, SimConfig
from kestrel_nav.livenav.opening import combine, find_opening_flu, to_leg_frame
from kestrel_nav.livenav.safe import REACH_M, SafeMission, SafeParams, prop_reach_m, stop_distance
from kestrel_nav.livenav.safety import InputSafety


def test_prop_reach_is_the_seven_inch_fleet_geometry():
    # rear rotor (-0.15, 0.14): hypot 0.2052 + 7 in / 2 (0.0889) = 0.2941
    assert prop_reach_m() == pytest.approx(0.294083, abs=1e-6)
    assert SafeParams().corridor_half_m > REACH_M
    assert REACH_M > 0.28            # the old 0.28 m planner radius under-states the 7-inch airframe


def wall_with_opening(dist, width, centre_left, rng, n=4000):
    """Level FLU points of a wall `dist` ahead with an opening of `width` centred `centre_left` left."""
    y = rng.uniform(-1.2, 1.2, n)
    z = rng.uniform(-0.3, 0.3, n)
    keep = np.abs(y - centre_left) > width / 2
    x = dist + rng.normal(0, 0.005, n)
    return np.column_stack([x, y, z])[keep]


@pytest.mark.parametrize('offset', [-0.12, -0.03, 0.0, 0.05, 0.11])
def test_opening_centre_is_measured_from_the_jambs(offset):
    rng = np.random.default_rng(1)
    meta = {'dist_m': 0.8, 'width_m': 0.87}
    ests = [find_opening_flu(wall_with_opening(0.8, 0.87, offset, rng), meta) for _ in range(5)]
    m = combine(ests)
    assert m['two_edge_frames'] == 5
    assert m['lateral_m'] == pytest.approx(offset, abs=0.02)
    assert m['width_m'] == pytest.approx(0.87, abs=0.04)


def test_leg_frame_rotation_keeps_a_centred_opening_centred():
    est = {'lateral_m': 0.0, 'range_m': 0.8, 'heading_rad': 0.0}
    assert to_leg_frame(est, 0.0)['lateral_m'] == pytest.approx(0.0)
    # body yawed 5 deg left of the leg: the leg (and the centred opening on it) lies 5 deg to the
    # RIGHT of the body axis, bearing +5 deg = wrap(body yaw - leg heading) as safe.py computes it
    rotated = to_leg_frame({'lateral_m': -0.8 * math.tan(math.radians(5)), 'range_m': 0.8, 'heading_rad': 0.0},
                           math.radians(5))
    assert abs(rotated['lateral_m']) < 0.01


def test_stopping_budget_grows_with_speed():
    p = SafeParams()
    assert stop_distance(0.0, p) == 0.0
    assert stop_distance(0.30, p) > stop_distance(0.20, p) > stop_distance(0.12, p)
    # at the cap speed the vehicle stops inside what a 2 m depth view shows
    assert p.reach_m + p.front_buffer_m + stop_distance(p.v_cap, p) < 2.0


def fly(slip=0.0, seed=0, **params):
    m = SafeMission(MissionParams(landed_z_m=0.067, time_limit_s=175.0), safe_params=SafeParams(**params))
    trace = []
    step = m.step

    def recording(obs):
        cmd = step(obs)
        trace.append((m.state, math.hypot(cmd.vx, cmd.vy), cmd.yaw_rate))
        return cmd
    m.step = recording
    fl = OfflineFlight(m, SimConfig(seed=seed, spawn=tuple(m.doc['start']['xy']), prop_radius_m=REACH_M, yaw_slip_s=slip,
                                    max_time_s=185.0, pad_style='square', pad_truth_offset=(0.06, -0.04), pad_yaw_deg=20.0))
    res = fl.run()
    res.land_pose = getattr(fl, 'land_pose', None)
    return m, res, trace


@pytest.fixture(scope='module')
def slip_flight():
    return fly(slip=1.0)


def test_course_completes_without_contact_under_run010_yaw_slip(slip_flight):
    m, res, _ = slip_flight
    assert not res.collided, res.result
    assert res.result == 'landed', (res.result, res.states[-3:])
    assert res.min_clearance_m >= REACH_M + 0.03
    assert res.t_end < 175.0
    # venue pad (0.64 m blue square, 6 cm / 20 deg off the course file): all four prop discs inside it
    from kestrel_nav.livenav.square_pad import footprint_inside
    x, y, yaw, pad = res.land_pose
    inside, margin = footprint_inside(x, y, yaw, pad, 0.64, math.radians(20.0))
    assert inside, (margin, m.pad_source)
    assert m.pad_source.startswith('detected')


def test_never_translates_while_turning(slip_flight):
    m, _, trace = slip_flight
    sp = m.sp
    turning = [(v, w) for s, v, w in trace if s in ('FLY', 'PAD') and abs(w) > sp.heading_hold_rate + 1e-9]
    assert turning, 'the course has corners: some in-place turns must have happened'
    assert max(v for v, _ in turning) <= sp.creep_mps + 1e-9


def test_both_crossings_were_centred_by_measurement(slip_flight):
    m, _, _ = slip_flight
    measured = [e for e in m.events if e[1] == 'ALIGN' and e[2] == 'measured']
    centred = [e for e in m.events if e[3] == 'centred on the opening']
    assert len(centred) == 2                                 # tunnel mouth and window
    assert all(abs(e[3]['correction_m']) <= m.sp.align_max_shift_m for e in measured)


def test_unmeasurable_opening_is_refused_not_crossed(monkeypatch):
    monkeypatch.setattr(safe_mod, 'find_opening_flu', lambda pts, meta, **kw: None)
    m, res, _ = fly(seed=1)
    assert not res.collided
    assert m.state == 'HOLD' and m.gate_refused
    assert m.replanner.goal.name == 'tunnel_exit'          # held before the first crossing leg


@pytest.mark.parametrize('waived, action', [(frozenset(), 'stop'), (frozenset({'depth'}), 'run')])
def test_depth_waiver_on_descent_keeps_every_other_stream_required(waived, action):
    s = InputSafety()
    for stream in ('state', 'odom', 'tracking'):
        assert s.accept(stream, 9.95, 10.0, 10.0)
    s.active = True                                          # in flight, depth missing
    s.waived = waived
    assert s.evaluate(10.0, 10.0, offboard=True, armed=True, landed=False, link=True,
                      position_valid=True) == action
    s2 = InputSafety()
    for stream in ('state', 'tracking'):
        s2.accept(stream, 9.95, 10.0, 10.0)
    s2.active, s2.waived = True, frozenset({'depth'})
    assert s2.evaluate(10.0, 10.0, offboard=True, armed=True, landed=False, link=True,
                       position_valid=True) == 'stop'       # odometry is never waived


def test_waived_stream_reject_does_not_trip_the_fault():
    s = InputSafety()
    s.active, s.waived = True, frozenset({'depth'})
    s.reject('depth', 'depth: valid fraction 0.05 below 0.30')     # close to the floor on descent
    assert s.fault is None
    s.reject('odom', 'odom: stale')
    assert s.fault == 'odom: stale'


def scan_of(ranges_ahead):
    from kestrel_nav.coursenav.scan import Scan
    r = np.asarray(ranges_ahead, float)
    return Scan(bearing=np.zeros(len(r)), range=r, sigma=np.full(len(r), 0.02))


@pytest.fixture(scope='module')
def gate_mission():
    return SafeMission(MissionParams(landed_z_m=0.067))


def test_gate_leg_stops_for_an_obstacle_and_remembers_it_inside_stereo_range(gate_mission):
    from kestrel_nav.livenav.mission import Command, Observation
    m = gate_mission
    m.T = (0.0, 0.0, 0.0)
    m.vel_course = np.zeros(2)
    xy = np.array([0.0, 0.0])
    m.seg = dict(a=xy.copy(), b=np.array([2.0, 0.0]), dir=np.array([1.0, 0.0]), heading=0.0, v=0.12,
                 gate='window_post', line_pt=xy.copy(), ahead_mem=None)
    m._phase('GO', 10.0)
    m.align_meta = {}                                 # no re-centring in this unit test
    obs = Observation(now=10.0, odom_xyz=(0, 0, 0.7), odom_yaw=0.0, offboard_active=True)
    # 0.6 m ahead leaves 0.21 m of room: it may still creep (it stops 0.394 m short)
    m.scan_nav = scan_of([0.6])
    cmd = m._drive(10.0, obs, (0.0, 0.0, 0.0), Command())
    assert m.phase == 'GO' and 0 < math.hypot(cmd.vx, cmd.vy) <= 0.12 + 0.06
    # 0.45 m ahead: room 0.056 m < min segment -> stop
    m.scan_nav = scan_of([0.45])
    cmd = m._drive(10.0, obs, (0.0, 0.0, 0.0), Command())
    assert m.phase == 'BRAKE' and math.hypot(cmd.vx, cmd.vy) == 0.0
    # same segment, the return now inside stereo minimum range (dropped from the scan): still a stop
    m.seg = dict(a=xy.copy(), b=np.array([2.0, 0.0]), dir=np.array([1.0, 0.0]), heading=0.0, v=0.12,
                 gate='window_post', line_pt=xy.copy(), ahead_mem=(10.0, 0.55))
    m._phase('GO', 10.5)
    m.scan_nav = scan_of([])
    cmd = m._drive(10.5, obs, (0.1, 0.0, 0.0), Command())
    assert m.phase == 'BRAKE' and math.hypot(cmd.vx, cmd.vy) == 0.0
