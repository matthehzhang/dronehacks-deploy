"""Exercise the deployed waypoint profile with blank depth and measured walls."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from refnav import core as C


@pytest.fixture(params=["course/course_v3.json", "kestrel_nav/data/course/course_v3.json"])
def params(request):
    doc = json.loads((ROOT / request.param).read_text())
    settings = {k: v for k, v in doc["refnav_params"].items() if not k.startswith("_")}
    settings.update({k: v for k, v in doc["navigation"]["refnav"].items() if not k.startswith("_")})
    return C.Params(route=[(5.0, 0.0)], **settings)


def checking(params, mode="carrot"):
    params.track_mode = mode
    m = C.Mission(params)
    m.phase, m.t_phase, m.t_check, m.t_started, m.t_last = "check", 0.0, 0.0, 0.0, 0.0
    m.home, m.sp = (0.0, 0.0, 0.0, 0.0), [0.0, 0.0, -0.7, 0.0]
    m.targets, m.leg_start, m.leg_dir, m.leg_len = [(5.0, 0.0)], (0.0, 0.0), (1.0, 0.0), 5.0
    m.offboard_seen = True
    v = C.Vehicle(pos=(0.0, 0.0, -0.7), heading=0.0, pos_ok=True,
                  armed=True, nav_state=C.NAV_OFFBOARD, landed=False, vel=(0.0, 0.0, 0.0))
    return m, v


def feed(m, t, distance=np.nan):
    # Optical (right, down, forward) -> body FLU (forward, left, up).
    rotation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    image = np.full((160, 160), distance, dtype=np.float32)
    assert m.on_depth(t, image, (160, 160, 80, 80), rotation, (0, 0, 0), (1, 0, 0, 0), now=t)
    m.on_health(t, True)
    if m.p.track_mode == "path":
        assert m.on_path(t, 0, [(0, 0), (2.5, 0), (5, 0)], [.3, .3, 0])


@pytest.mark.parametrize("mode", ["carrot", "pose", "path"])
@pytest.mark.parametrize("distance", [np.nan, 0.5])
def test_blank_or_half_metre_wall_allows_motion_past_old_timeout(params, mode, distance):
    m, v = checking(params, mode)
    advanced = False
    for t in np.arange(.1, 12.1, .1):
        feed(m, float(t), distance)
        m.step(float(t), v)
        advanced |= m.v_cmd > 0
    assert m.phase == "go"
    assert m.land_reason is None
    assert advanced
    assert m.counts["depth_holds"] == 0
    assert m._stop_m() == pytest.approx(.35)
    assert not m._blocks(.35, 5.0)
    assert m._blocks(.349, 5.0)


def test_close_measured_wall_still_stops(params):
    m, v = checking(params)
    for t in [.1, .2, .3]:
        feed(m, t, .34)
        m.step(t, v)
    assert m.phase == "blocked"
    assert "obstacle 0.34 m" in m.why


def test_camera_silence_still_lands_with_prominent_reason(params):
    m, v = checking(params)
    for t in [.1, .2, .3]:
        feed(m, t)
        m.step(t, v)
    assert m.phase == "go"
    m.on_health(6.0, True)
    out = m.step(6.0, v)
    assert m.phase == "land"
    assert "!!!!!!!!!! AUTO LAND: depth silent for more than 5 s !!!!!!!!!!" in out.events


@pytest.mark.parametrize("cause", ["position", "health", "complete", "requested", "takeover"])
def test_landing_and_takeover_banners(params, cause):
    m, v = checking(params)
    feed(m, .1)
    if cause == "position":
        v.pos_ok = False
        out = m.step(.1, v)
        expected = "AUTO LAND: PX4 position or heading invalid"
    elif cause == "health":
        m.health.bad_for = lambda now: 2.0
        out = m.step(.1, v)
        expected = "AUTO LAND: estimator unhealthy or silent for 2.0 s"
    elif cause == "takeover":
        v.nav_state = C.NAV_AUTO_LAND
        out = m.step(.1, v)
        expected = "CONTROL YIELDED: PX4 left Offboard"
        assert "external landing reason unavailable" in out.events[-1]
    elif cause == "requested":
        m.request_land("pilot requested")
        out = m.step(.1, v)
        expected = "AUTO LAND: landing requested: pilot requested"
    else:
        out = C.Output(phase=m.phase)
        m._land(.1, out, "route complete")
        expected = "AUTO LAND: route complete"
    assert any(e.startswith("!!!!!!!!!! " + expected) and e.endswith(" !!!!!!!!!!") for e in out.events)
