"""unobserved_explore: a leg whose corridor no depth frame observes is explored (sweep, back-off along the path
already flown, recheck) and resumed as soon as it is observed, instead of landing after corridor_wait_land_s.
Bounded per exploration and per leg; never translates forward into unobserved space."""
import math

import numpy as np
import pytest

from refnav import core as C
from test_refnav_core import FX, H, W, FakePX4, fly, _no_return_ahead


EXPLORE = dict(unobserved_explore=True, unobserved_trigger_s=2.0, corridor_wait_land_s=5.0)


def base(**kw):
    return C.Params(route=[(3.0, 0.0)], stop_m=0.9, lead_max_m=0.6, **kw)


def go_time(prm):
    """When the clean route first enters go (to schedule a dropout inside the leg)."""
    _, _, log = fly(base(**prm))
    return next(t for t, ph, *_ in log if ph == "go")


def dropout(t0, t1):
    def fn(t, img):
        return _no_return_ahead(t, img) if t0 <= t < t1 else img
    return fn


def test_a_temporary_blind_spell_is_explored_and_resumed_instead_of_landing():
    tg = go_time(EXPLORE)
    blind = dropout(tg + 0.5, tg + 6.5)                 # 6 s blind mid-leg: longer than corridor_wait_land_s
    m, px4, log = fly(base(corridor_wait_land_s=5.0), depth_fn=blind, seconds=90.0)
    assert m.result.startswith("landed: corridor of leg 1 not observed"), m.result     # without: lands
    m, px4, log = fly(base(**EXPLORE), depth_fn=blind, seconds=90.0)
    assert m.result == "complete", (m.result, m.why)
    assert m.counts["explores"] >= 1 and m.counts["explore_resumes"] >= 1
    phases = [ph for _, ph, *_ in log]
    assert "explore_scan" in phases


def test_exploring_never_translates_forward_into_the_unobserved_corridor():
    tg = go_time(EXPLORE)
    m, px4, log = fly(base(**EXPLORE), depth_fn=dropout(tg + 0.5, 1e9), seconds=90.0)
    rows = [(t, ph, pos) for t, ph, _, _, pos in log]
    start = next(i for i, (_, ph, _) in enumerate(rows) if ph == "explore_scan")
    n0 = rows[start][2][0]
    explore = [pos for _, ph, pos in rows[start:] if ph in C.EXPLORE]
    assert max(p[0] for p in explore) <= n0 + 0.05      # forward (n) never beyond where exploring began
    assert min(p[0] for p in explore) < n0 - 0.15       # ... it backed off along the flown leg


def test_a_corridor_never_observed_lands_after_a_bounded_exploration():
    m, px4, log = fly(base(**EXPLORE), depth_fn=_no_return_ahead, seconds=120.0)
    assert m.result.startswith("landed: corridor of leg 1 not observed"), m.result
    assert m.counts["explores"] == 1 and m.counts["explore_resumes"] == 0
    t_explore = [t for t, ph, *_ in log if ph in C.EXPLORE]
    assert t_explore[-1] - t_explore[0] <= m.p.unobserved_budget_s + 0.1
    assert not px4.armed


def test_a_permanent_blind_patch_ahead_is_bounded_per_leg():
    """Nothing returns within +-13 deg of the camera axis (a textureless panel face): sweeping 20 deg observes the
    leg's corridor, go faces it again and is blind again. After unobserved_max_per_leg explorations the original
    corridor-wait rule lands."""
    cols = np.abs(np.degrees(np.arctan((np.arange(W) - W / 2) / FX))) < 13.0

    def patch(t, img):
        out = img.copy()
        out[: H // 2 + 8, cols] = np.nan
        return out
    # the course's corridor profile (course_new_origin.json): +-0.32 m wide, +-0.25 m band, 45 % coverage, so the
    # window 1.5 m ahead spans ~+-12 deg and the patch hides it until the camera turns away
    prm = base(**EXPLORE, corridor_half_m=0.32, band_m=0.25, min_corridor_coverage=0.45)
    m, px4, log = fly(prm, depth_fn=patch, seconds=180.0)
    assert m.counts["explores"] == m.p.unobserved_max_per_leg
    assert m.counts["explore_resumes"] >= 1
    assert m.result.startswith("landed: corridor of leg 1 not observed"), m.result


def test_back_off_creeps_at_most_015_mps():
    tg = go_time(EXPLORE)
    m, px4, log = fly(base(**EXPLORE), depth_fn=dropout(tg + 0.5, 1e9), seconds=90.0)
    back = [pos for _, ph, _, _, pos in log if ph == "explore_back"]
    v = [math.dist(a[:2], b[:2]) / .05 for a, b in zip(back[:-1], back[1:])]
    assert v and max(v) <= .15 + .05


def test_off_by_default_and_bounds_checked():
    assert C.Params().unobserved_explore is False
    with pytest.raises(ValueError):
        C.Mission(base(unobserved_explore=True, unobserved_budget_s=600.))
    with pytest.raises(ValueError):
        C.Mission(base(unobserved_explore=True, unobserved_backoff_m=3.))


def test_path_legs_explore_back_along_the_navigator_path_and_complete():
    from test_refnav_blocked import STRAIGHT, fly_path, params, publish_straight, speeds
    prm = params(**EXPLORE)
    _, _, clean = fly_path(params(**EXPLORE), lambda t: [], publish_straight)
    tg = next(t for t, ph, *_ in clean if ph == "go")
    blind = dropout(tg + 1.0, tg + 11.5)    # outlasts the sweep: the back-off runs, then the view returns
    from test_refnav_core import render_depth, ROOM
    from coursenav.px4_model import DynamicPX4
    px4 = DynamicPX4()

    def depth_fn(t, img):
        return blind(t, render_depth(px4.pos, px4.yaw, ROOM))
    m, px4, log = fly(prm, depth_fn=depth_fn, on_tick=publish_straight, seconds=120.0, px4=px4)
    assert m.result == "complete", (m.result, m.why)
    assert m.counts["explores"] >= 1 and m.counts["explore_resumes"] >= 1
    back = [pos for _, ph, _, _, pos in log if ph == "explore_back"]
    assert back and abs(max(abs(p[1]) for p in back)) < 0.1        # stayed on the path while backing off
