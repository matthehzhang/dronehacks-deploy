"""The team's reconstructed origin map (course/course_new_origin.json, new_course_origin_map.zip), now the stack's
default course: it loads, its map is the published-OBJ build, the window route plans from the launch mark
(which sits 1.4 cm inside the planning buffer), and the navigator can fly out of the buffer."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from coursenav import course as CO                                  # noqa: E402
from coursenav import planner                                       # noqa: E402
from coursenav.course import Grid2D, load_course, plan_grids, resolve  # noqa: E402
from coursenav.navigator import Navigator                           # noqa: E402
from coursenav.navplan import smooth_params_for                     # noqa: E402
from coursenav.objmap import CourseMap                              # noqa: E402

NEW = ROOT / "course" / "course_new_origin.json"


@pytest.fixture(scope="module")
def new():
    doc = load_course(NEW)
    cm = CourseMap.load(resolve(doc, "map"))
    obst, blocked, clearance = plan_grids(cm, doc)
    return doc, cm, obst, blocked, clearance


def test_the_flight_stack_default_is_the_new_course_and_the_tests_pin_v3(monkeypatch):
    monkeypatch.delenv("COURSENAV_COURSE", raising=False)
    assert CO.default_course_path() == NEW
    monkeypatch.setenv("COURSENAV_COURSE", "course/course_v3.json")
    assert CO.default_course_path() == ROOT / "course" / "course_v3.json"


def test_the_map_is_the_published_obj_build_of_the_new_obj(new):
    doc, cm, *_ = new
    import hashlib
    assert cm.meta["source"] == "published OBJ"
    assert cm.meta["obj_sha256"] == hashlib.sha256(resolve(doc, "obj").read_bytes()).hexdigest()
    assert {"M1 maze divider", "M2 maze divider", "M3 maze divider", "C2a chicane wall (window)"} <= set(cm.groups)


def test_goals_start_at_the_tunnel_and_the_drone_starts_on_the_mark(new):
    raw = json.loads(NEW.read_text())                                   # the file, before goal expansion
    assert [s["name"] for s in raw["sequence"]] == ["tunnel", "chicane", "pad"]
    assert raw["start"]["xy"] == [-2.515, 0.914] and raw["sim_profile"]["fixture_offset_m"] == [0.0, 0.0]


def test_the_window_route_plans_from_the_launch_mark_with_a_bounded_escape(new):
    doc, cm, obst, blocked, clearance = new
    res = planner.plan_options(doc, obst, blocked, clearance, start_xy=doc["start"]["xy"], cmap=cm)
    assert res["chicane"] == "window"
    wps = res["plan"]["waypoints"]
    assert wps[1]["name"] == "via_start_escape"
    assert np.linalg.norm(np.subtract(wps[1]["xy"], wps[0]["xy"])) <= 0.15
    names = [w["name"] for w in wps]
    assert names.index("tunnel_entry") < names.index("tunnel_exit") < names.index("window") < names.index("pad")
    need = res["plan"]["required_clearance_m"]
    assert min(res["plan"]["segment_clearance_m"][1:]) >= need - 1e-6     # everything after the escape step


def test_start_escape_never_moves_closer_to_anything_and_is_bounded():
    occ = np.zeros((60, 60), bool)
    occ[:, 20] = True                                                     # a wall along x at y = 1.0
    obst = Grid2D(occ, np.zeros(2), 0.05)
    clr = obst.distance()
    blocked = Grid2D(clr < 0.414, obst.origin, obst.res)
    start = np.array([1.5, 1.4])                                          # 0.375 m from the wall: inside
    q = planner.start_escape(start, blocked, obst, clr, 0.15)
    assert q is not None and q[1] > start[1] and np.linalg.norm(q - start) <= 0.15
    assert planner.start_escape(np.array([1.5, 2.2]), blocked, obst, clr, 0.15) is None   # already free
    with pytest.raises(planner.PlanError):
        planner.start_escape(np.array([1.5, 1.2]), blocked, obst, clr, 0.05)              # too deep for 5 cm


def test_the_navigator_path_out_of_the_buffer_has_an_executable_speed(new):
    doc, cm, obst, blocked, clearance = new
    need = doc["vehicle"]["radius_m"] + doc["vehicle"]["margin_m"]
    navcfg = doc["navigation"]
    ep = dict(doc["refnav_params"])
    ep.update({k: v for k, v in navcfg.get("refnav", {}).items() if not k.startswith("_")})
    nav = Navigator(obst, blocked, clearance, need, smooth_params_for(navcfg, ep, need))
    P = nav.plan(np.array(doc["start"]["xy"]), [np.array(doc["derived"]["tunnel"]["entry"])])
    assert P.v[0] == 0.0
    s0 = nav._free_start(np.array(doc["start"]["xy"]), *nav.fuse()[1:])
    esc = float(np.linalg.norm(s0 - doc["start"]["xy"]))
    assert 0 < esc <= nav.p.escape_m
    inside = P.v[(P.s > 1e-9) & (P.s <= esc + 1e-9)]                    # the escape step, start excluded
    assert len(inside) and np.all(inside > 0) and np.all(inside <= nav.p.escape_creep_mps + 1e-9)
