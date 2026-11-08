"""Window height correction from depth, without an extra hold/retry state."""
from types import SimpleNamespace

import numpy as np
import pytest

from refnav import core as C, gate as G


META = dict(group="window", dist_m=0.8, width_m=0.9, dz_lo=-0.45, dz_hi=0.45)


def window_points(dz, edges=("sill", "header"), floor=False):
    patches = []
    for side in (-1, 1):
        r, d = np.meshgrid(side * np.linspace(0.45, 0.57, 9), np.linspace(-0.2, 0.2, 9))
        patches.append(np.column_stack((np.full(r.size, 0.8), r.ravel(), d.ravel())))
    for edge in edges:
        down = np.linspace(0.45, 0.55, 9) if edge == "sill" else np.linspace(-0.55, -0.45, 9)
        r, d = np.meshgrid(np.linspace(-0.3, 0.3, 11), down - dz)
        patches.append(np.column_stack((np.full(r.size, 0.8), r.ravel(), d.ravel())))
    if floor:
        f, r = np.meshgrid(np.linspace(0.73, 0.87, 11), np.linspace(-0.3, 0.3, 11))
        patches.append(np.column_stack((f.ravel(), r.ravel(), np.full(r.size, 0.7))))
    return np.vstack(patches)


@pytest.mark.parametrize("dz", [-0.25, 0.0, 0.25])
@pytest.mark.parametrize("edges", [("sill",), ("header",), ("sill", "header")])
def test_depth_centres_window_with_one_or_both_edges(dz, edges):
    est = G.find_opening(window_points(dz, edges), META)
    assert est is not None
    assert est["dz_m"] == pytest.approx(dz, abs=0.02)


@pytest.mark.parametrize("floor", [False, True])
def test_missing_edges_and_floor_do_not_invent_height(floor):
    est = G.find_opening(window_points(0, (), floor=floor), META)
    assert est is not None and est["dz_m"] is None


def test_conflicting_window_height_is_rejected():
    points = window_points(0)
    points[points[:, 2] < -0.3, 2] -= 0.25
    assert G.find_opening(points, META)["dz_m"] is None


def test_height_needs_multiple_consistent_frames():
    base = dict(lateral_m=0, heading_rad=0, width_m=0.9, range_m=0.8)
    for heights in [(0.25, None, None), (0.25, -0.25, 0.25)]:
        assert G.combine([dict(base, dz_m=d) for d in heights])["dz_m"] is None
    assert G.combine([dict(base, dz_m=d) for d in (0.24, 0.26, None)])["dz_m"] == pytest.approx(0.25)


def mission(group="window"):
    meta = dict(META, group=group)
    m = C.Mission(C.Params(route=[(1, 0)], legs=[dict(align=meta)]))
    m.leg = 0
    m.phase = "align"
    m.t_phase = 0
    m.sp = [0, 0, -0.8, 0]
    m.targets = [(1, 0)]
    m.leg_start = (0, 0)
    m.leg_dir = (1, 0)
    m.leg_heading = 0
    return m


@pytest.mark.parametrize("dz, edges", [(0.25, ("sill",)), (-0.25, ("header",)), (0, ())])
def test_alignment_uses_existing_frames_without_extra_wait(dz, edges):
    m = mission()
    m.frame = SimpleNamespace(points=lambda *_: window_points(dz, edges))
    for t in (0.1, 0.2, 0.3):
        m.depth_t = t
        out = C.Output(phase=m.phase)
        m._align(t, out, C.Vehicle(heading=0))
    assert m.phase == "check"
    assert m.recover is None
    assert m.sp[2] == pytest.approx(-0.8 - dz, abs=0.02)
    if not edges:
        assert any("height unavailable (unchanged)" in event for event in out.events)


@pytest.mark.parametrize("group, limit", [("window", 0.30), ("tunnel", 0.15)])
@pytest.mark.parametrize("sign", [-1, 1])
def test_window_limit_does_not_change_other_gate_limits(group, limit, sign):
    m = mission(group)
    m._apply_alignment(0.3, C.Output(phase="align"), dict(lateral_m=0, dz_m=sign * 0.5))
    assert m.sp[2] == pytest.approx(-0.8 - sign * limit)
