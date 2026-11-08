"""Changed-layout tests using real depth-ray integration and A*."""
import math

import numpy as np
import pytest

from kestrel_nav.coursenav.objmap import CourseMap
from kestrel_nav.coursenav.scan import Scan
from kestrel_nav.livenav.navgrid import NavGrid, NavParams
from kestrel_nav.livenav.replanner import Goal, Replanner


@pytest.mark.parametrize("side", [-1, 1])
def test_observed_opening_replaces_old_wall_and_expires(side):
    res = .05
    shape = (80, 80, 6)
    labels = np.zeros(shape, np.uint8)
    labels[40, :, :] = 1
    labels[:, (0, -1), :] = 2
    cmap = CourseMap(labels > 0, np.array([-2., -2., .55]), res, labels, ["maze", "fence"], {})
    doc = {"cruise_z_m": .7, "vehicle": {"radius_m": .22, "margin_m": .08},
           "adaptive_obstacle_groups": ["maze"]}
    g = NavGrid(cmap, doc, NavParams(adaptive=True))
    start = (-.65, 0.)
    goal = (1., side * .85)
    rp = Replanner(g, [Goal({"name": "checkpoint", "xy": goal})])
    assert rp.plan(start, 0.) is None

    # The new layout has an opening on either side. Returns beyond the old wall
    # establish the gap; outside the gap the real wall still produces hits.
    yaw = side * .65
    angles = np.linspace(-.66, .66, 48)
    y_at_wall = .65 * np.tan(yaw + angles)
    through = abs(y_at_wall - side * .82) < .50
    ranges = np.where(through, 2.5, .65 / np.cos(yaw + angles))
    scan = Scan(angles, ranges, np.full(len(angles), .03))
    for t in (.2, .4, .6):
        g.integrate((*start, yaw), scan, t)
    g.rebuild(.6, force=True)
    assert rp.plan(start, .6) is None
    for t in (.8, 1., 1.2, 1.4):
        g.integrate((*start, yaw), scan, t)
    g.rebuild(1.4, force=True)
    path = rp.plan(start, 1.4)
    assert path is not None and not rp.blocked
    crossings = [a[1] + (b[1] - a[1]) * (-a[0]) / (b[0] - a[0])
                 for a, b in zip(path, path[1:]) if a[0] < 0 <= b[0]]
    assert len(crossings) == 1 and abs(crossings[0] - side * .82) < .25
    assert g.combined.occ[:, 0].all()
    assert g.is_blocked((.025, -side * .82), hard=True)

    g.rebuild(12.)
    assert rp.plan(start, 12.) is None


def test_bootstrap_never_routes_through_the_prior_maze(monkeypatch):
    from kestrel_nav.coursenav import planner
    from kestrel_nav.livenav.safe import SafeMission, SafeParams

    original = planner.astar
    start_calls = []

    def record_start(grid, clear, start, goal, *args, **kwargs):
        start_calls.append(np.asarray(start))
        return original(grid, clear, start, goal, *args, **kwargs)

    monkeypatch.setattr(planner, "astar", record_start)
    mission = SafeMission(nav_params=NavParams(adaptive=True), safe_params=SafeParams(live_grid=True))
    assert mission.goals[0].name == "maze_exit"
    assert start_calls and all(math.dist(p, mission.plan["start_course"][:2]) > .3 for p in start_calls)
    assert {"tunnel_exit", "window_post"} <= mission.align_meta.keys()
    assert mission.replanner.path is None
