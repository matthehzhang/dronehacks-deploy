"""Goals derived from the published OBJ (coursenav.semantics), rule 9.5.

The reference numbers below are the course drawing's published dimensions (inches converted to
metres), used only to check the measurement. Flight code never sees them."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from kestrel_nav.coursenav.course import PROJECT, load_course, resolve
from kestrel_nav.coursenav.objmap import CourseMap
from kestrel_nav.coursenav.semantics import Geometry, SemanticsError, resolve_course, tube_axis, wall_opening

RAW = PROJECT / "course" / "course_v3.json"
IN = 0.0254


@pytest.fixture(scope="module")
def doc():
    return load_course()


def test_the_course_file_stores_no_goal_coordinates():
    def walk(v, path=""):
        if isinstance(v, dict):
            for k, x in v.items():
                assert k not in ("xy", "center", "normal", "target_xy"), f"coordinate stored at {path}/{k}"
                walk(x, f"{path}/{k}")
        elif isinstance(v, list):
            for i, x in enumerate(v):
                walk(x, f"{path}[{i}]")
    walk(json.loads(RAW.read_text()))


def test_start_is_the_launch_mark_heading_along_the_guideline(doc):
    assert doc["start"]["xy"] == pytest.approx([-99 * IN, 36 * IN], abs=0.005)
    assert doc["start"]["yaw_deg"] == pytest.approx(90.0, abs=0.5)


def test_gate_a_goal_is_past_the_guideline_end(doc):
    d = doc["derived"]["maze_exit"]
    end, goal = np.array(d["line_end_xy"]), np.array(d["xy"])
    assert end == pytest.approx([0.33, 2.585], abs=0.02)
    assert np.dot(goal - end, d["direction"]) == pytest.approx(0.40, abs=0.01)   # whole drone past


def test_tunnel_runs_along_the_tube_axis(doc):
    t = doc["derived"]["tunnel"]
    assert t["length_m"] == pytest.approx(60 * IN, abs=0.01)
    assert t["entry"][0] == pytest.approx(t["exit"][0], abs=0.005)                # straight along y
    assert t["entry"][1] > t["exit"][1]                                            # entered from the maze side


def test_window_is_the_opening_in_c2a(doc):
    w = doc["chicane_options"]["window"]
    assert w["width_m"] == pytest.approx(36 * IN, abs=0.01)
    assert w["z_m"] == pytest.approx([12 * IN, 48 * IN], abs=0.01)
    pad = np.array(doc["sequence"][-1]["xy"])
    assert np.dot(w["normal"], pad - np.array(w["center"])) > 0                    # crossed towards the pad


def test_gate_b_is_the_passable_gap_past_c2b(doc):
    g = doc["chicane_options"]["gate_b"]
    assert g["width_m"] >= 0.6
    assert g["center"] == pytest.approx([0.33, -0.07], abs=0.03)


def test_pad_centre_and_radius(doc):
    pad = doc["sequence"][-1]
    assert pad["land"] and pad["radius_m"] == pytest.approx(30 * IN, abs=0.005)


def test_goals_follow_the_geometry_when_a_part_moves(doc):
    """Move the pad and the C2a wall in the OBJ: the derived goals move with them."""
    geo = Geometry(resolve(doc, "obj"))
    shift = np.array([0.30, -0.20, 0.0])
    for name in list(geo.groups):
        if name.startswith(("Landing pad", "C2a")):
            V, F = geo.groups[name]
            geo.groups[name] = (V + shift, F)
    raw = load_course(derive=False)
    moved = resolve_course(raw, CourseMap.load(resolve(raw, "map")), geo)
    assert np.subtract(moved["sequence"][-1]["xy"], doc["sequence"][-1]["xy"]) == pytest.approx(shift[:2], abs=0.005)
    assert np.subtract(moved["chicane_options"]["window"]["center"],
                       doc["chicane_options"]["window"]["center"]) == pytest.approx(shift[:2], abs=0.01)


def test_tube_axis_from_normals_is_not_fooled_by_a_square_tube():
    """A 1 x 1 m square tube along x is as wide as long: the axis must still be x."""
    class G:
        def triangles(self, _):
            tris = []
            for y0, z0, y1, z1 in ((0, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 1), (0, 1, 0, 0)):
                a, b, c, d = (0, y0, z0), (1, y0, z0), (1, y1, z1), (0, y1, z1)
                tris += [(a, b, c), (a, c, d)]
            return np.array(tris, float)

        def points(self, _):
            return self.triangles(None).reshape(-1, 3)
    entry, exit_, info = tube_axis(G(), {"group": "x", "approach_m": 0.5}, np.array([-5.0, 0.5]))
    assert entry == pytest.approx([-0.5, 0.5]) and exit_ == pytest.approx([1.5, 0.5])


def test_a_wall_without_an_opening_is_an_error(doc):
    geo = Geometry(resolve(doc, "obj"))
    with pytest.raises(SemanticsError):
        wall_opening(geo, {"group": "M1"}, 0.8)


def test_the_enclosure_is_measured_from_the_net_and_the_spawn_offset_is_a_sim_profile(doc):
    raw = json.loads(RAW.read_text())
    assert "lo" not in raw["enclosure"] and "fixture_offset_m" not in raw["start"]
    assert doc["enclosure"]["lo"] == pytest.approx([-120 * IN, -120 * IN], abs=0.02)   # the 20 x 20 ft net
    assert doc["enclosure"]["hi"] == pytest.approx([120 * IN, 120 * IN], abs=0.02)
    assert doc["start"]["spawn_xy"] == pytest.approx([-99 * IN, 36 * IN + 0.2], abs=0.005)
    raw.pop("sim_profile")
    comp = resolve_course(raw, CourseMap.load(resolve(raw, "map")), Geometry(resolve(raw, "obj")))
    assert "spawn_xy" not in comp["start"]                        # competition: the start is the mark
