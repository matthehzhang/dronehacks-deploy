"""tune_plan geometry and analysis. No ROS needed: ``python3 -m pytest -q test/``."""
import math

import pytest

from cage_test import analysis
from cage_test.hop_plan import build_legs
from cage_test.tune_plan import (
    Segment,
    build_plan,
    distance_to_path,
    parse_list,
    plan_duration,
    validate,
)

START = (0.3, -0.2, 0.1)


def test_hop_sequence_matches_the_original_hop():
    old = build_legs(START, 0.4, 1.0, 1.0, 0.5, 3.0)
    new = build_plan(START, 0.4, {}, 'hop')
    assert [s.name for s in new] == [leg.name for leg in old]
    for a, b in zip(new, old):
        assert a.start == pytest.approx(b.start) and a.end == pytest.approx(b.end)
        assert a.hold_s == b.hold_s and a.kind == 'move'


def test_every_pattern_starts_and_ends_at_home():
    segs = build_plan(START, 1.0, {}, 'hover,steps,speed,yaw,vertical,square')
    home = segs[0].end
    assert segs[-1].start == pytest.approx(home)
    for prev, nxt in zip(segs, segs[1:]):
        assert nxt.start == pytest.approx(prev.end), (prev.name, nxt.name)


def test_speed_sweep_order_and_distance():
    segs = build_plan(START, 0.0, {'speed_list_mps': '0.3,0.6', 'speed_distance_m': 1.5,
                                   'speed_axes': 'forward,left'}, 'speed')
    sweep = [s for s in segs if s.pattern == 'speed']
    assert [s.name for s in sweep] == ['forward @0.30', 'back @0.30', 'forward @0.60',
                                      'back @0.60', 'left @0.30', 'right @0.30',
                                      'left @0.60', 'right @0.60']
    assert sweep[0].end == pytest.approx((1.8, -0.2, 1.1))
    assert sweep[4].end == pytest.approx((0.3, 1.3, 1.1))


def test_yaw_sweep_turns_out_and_back_at_each_rate():
    segs = [s for s in build_plan(START, 0.0, {'yaw_rates_dps': '30,90'}, 'yaw')
            if s.kind == 'yaw']
    assert [round(math.degrees(s.yaw_rate)) for s in segs] == [30, 30, 90, 90]
    assert sum(s.yaw_delta for s in segs) == pytest.approx(0.0)
    assert all(s.start == s.end for s in segs)


def test_steps_are_steps_around_home():
    segs = [s for s in build_plan(START, 0.0, {}, 'steps') if s.pattern == 'steps']
    assert all(s.kind == 'step' for s in segs) and len(segs) == 8
    up = next(s for s in segs if s.name.startswith('step up') and 'return' not in s.name)
    assert up.end[2] - up.start[2] == pytest.approx(0.3)


@pytest.mark.parametrize('cfg,seq', [
    ({}, 'loop'), ({}, ''), ({'yaw_rates_dps': '200'}, 'yaw'),
    ({'speed_list_mps': '0'}, 'speed'), ({'vertical_span_m': 0.6}, 'vertical'),
    ({'step_z_m': 0.6}, 'steps'), ({'low_height_m': 1.2}, 'hop'),
])
def test_bad_plans_are_refused(cfg, seq):
    with pytest.raises(ValueError):
        validate(cfg, seq)


def test_parse_list_accepts_numbers_and_strings():
    assert parse_list('0.25, 0.5,1') == [0.25, 0.5, 1.0]
    assert parse_list('60') == [60.0]
    assert parse_list([1, 2]) == [1.0, 2.0]


def test_distance_to_path_ignores_lag_counts_sideways_and_overshoot():
    seg = Segment('s', 'p', 'move', (0, 0, 1), (1, 0, 1), 0.5, 0, 0, 0)
    assert distance_to_path((0.2, 0.0, 1.0), seg) == pytest.approx(0.0)
    assert distance_to_path((0.5, 0.3, 1.0), seg) == pytest.approx(0.3)
    assert distance_to_path((1.4, 0.0, 1.0), seg) == pytest.approx(0.4)


def test_plan_duration_counts_ramps_and_holds():
    segs = build_plan((0, 0, 0), 0.0, {'hold_seconds': 1.0, 'speed_mps': 0.5}, 'hop')
    # up 1 m, fwd 1 m, back 1 m, down 0.5 m at 0.5 m/s + 4 holds of 1 s
    assert plan_duration(segs) == pytest.approx(3.5 / 0.5 + 4.0)


def test_step_response_numbers():
    seg = Segment('step', 'steps', 'step', (0, 0, 1), (0.5, 0, 1), None, 5.0, 0, 0)
    rows = []
    for i in range(80):          # first order, tau 0.5 s, plus 10 % overshoot bump
        t = i * 0.05
        x = 0.5 * (1 - math.exp(-t / 0.5)) + 0.08 * math.exp(-((t - 1.5) ** 2) / 0.05)
        rows.append({'t': t, 'seg': 0, 'phase': 'hold', 'vio_x': x, 'vio_y': 0.0,
                     'vio_z': 1.0, 'cmd_x': 0.5, 'cmd_y': 0.0, 'cmd_z': 1.0})
    s = analysis.summarize([seg], rows)[0]
    assert s['rise_time_s'] == pytest.approx(0.5 * math.log(9), abs=0.06)
    assert 5.0 < s["overshoot_pct"] < 20.0
    assert s['settling_time_s'] is not None and s['steady_state_error_m'] < 0.01
