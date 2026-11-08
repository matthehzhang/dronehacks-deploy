"""The course as BUILT differs from the published map (2026-11-07, skybeta-14: the maze was simplified on
the day; the localizer never matched a scan, the navigator kept routing through the old maze, and refnav
held at a wall that was not on the map for 45 s). Nothing here knows the new layout: the navigator must
learn it from depth like any other difference.

Pipeline exercised (no ROS): depth rays from a simulated as-built world -> coursenav.occgrid.LiveGrid
(the localizer's ungated navigation map) -> its OccupancyGrid encoding -> coursenav.nav_node.LiveLandmarks
-> coursenav.navigator.Navigator (fuse, needs_replan, path_blocked_ahead, plan) on the real course map.
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("scipy")


from coursenav.course import load_course, plan_grids, resolve, scan_grid, see_through_grid  # noqa: E402
from coursenav.navigator import Navigator  # noqa: E402
from coursenav.navplan import smooth_params_for  # noqa: E402
from coursenav.objmap import CourseMap  # noqa: E402
from coursenav.occgrid import LiveGrid, OccParams  # noqa: E402
from coursenav.scan import Scan  # noqa: E402
from coursenav.livemap import LiveLandmarks  # noqa: E402
from coursenav.planner import PlanError  # noqa: E402
from refnav import core as C  # noqa: E402

START = (-2.5146, 0.9144)                  # the venue launch mark (course_v3.json start)
NEW_WALL = ((-2.95, 2.2), (-2.25, 2.2))    # across the old route's first corner (where skybeta-14 held)
REMOVED = (-2.25, 0.05, 0.65, 2.45)        # x0, x1, y0, y1: the old maze dividers in here are gone


@pytest.fixture(scope="module")
def course():
    doc = load_course()
    cmap = CourseMap.load(resolve(doc, "map"))
    obst, blocked, clear = plan_grids(cmap, doc)
    need = float(doc["vehicle"]["radius_m"] + doc["vehicle"]["margin_m"])
    from coursenav.navplan import stops_and_legs
    from coursenav import planner
    p = planner.plan(doc, obst, blocked, clear, start_xy=(-2.525, 0.975), chicane="window", cmap=cmap)
    _, _, nav_legs = stops_and_legs(p, START, math.radians(90.0))
    navcfg = {k: v for k, v in doc["navigation"].items() if not k.startswith("_")}
    params = dict(doc["refnav_params"], **navcfg.get("refnav", {}))
    goals = [np.array(g, float) for g in nav_legs[0]["goals"]]
    return SimpleNamespace(doc=doc, cmap=cmap, obst=obst, blocked=blocked, clear=clear, need=need,
                           sg=scan_grid(cmap), st=see_through_grid(cmap), goals=goals,
                           gate_lines=p.get("gate_lines", []), sp=smooth_params_for(navcfg, params, need))


def as_built(c, removed=None, new_wall=None):
    """The scan-band surfaces of the course as built: the published ones, less the removed box's
    (never a net/window/tunnel), plus a new wall."""
    truth = c.sg.occ.copy()
    if removed:
        X, Y = np.meshgrid(np.arange(truth.shape[0]), np.arange(truth.shape[1]), indexing="ij")
        cx, cy = c.sg.origin[0] + (X + 0.5) * c.sg.res, c.sg.origin[1] + (Y + 0.5) * c.sg.res
        x0, x1, y0, y1 = removed
        truth[(cx > x0) & (cx < x1) & (cy > y0) & (cy < y1) & ~c.st] = False
    if new_wall:
        for u in np.linspace(0, 1, 60):
            ij = c.sg.ij(np.array(new_wall[0]) * (1 - u) + np.array(new_wall[1]) * u)
            truth[ij[0], ij[1]] = True
    return truth


def fly(c, truth, t_max=60.0):
    """Kinematic closed loop: depth rays from the as-built world -> the localizer's navigation map
    (ungated LiveGrid) -> LiveLandmarks -> Navigator re-plans; the aircraft follows the path at 0.25 m/s,
    turns toward it at 30 deg/s, and holds (refnav's rule) while an as-built surface in view is within
    0.32 m of the path in the next 0.95 m. -> dict of what happened."""
    from scipy import ndimage
    td = ndimage.distance_transform_edt(~truth) * c.sg.res
    tclr = lambda q: td[tuple(c.sg.ij(q))]          # noqa: E731

    def scan_from(pose, fov=0.7, n=60, rmax=4.0):
        b, r = np.linspace(-fov, fov, n), np.full(n, np.nan)
        for k, bb in enumerate(b):
            a = pose[2] + bb
            for d in np.arange(0.05, rmax, 0.02):
                ij = c.sg.ij(np.array([pose[0] + d * math.cos(a), pose[1] + d * math.sin(a)]))
                if not c.sg.inside(ij):
                    break
                if truth[ij[0], ij[1]]:
                    r[k] = d
                    break
        ok = np.isfinite(r)
        return Scan(b[ok], r[ok], np.full(ok.sum(), 0.05))

    grid = LiveGrid(c.sg, OccParams(), see_through=c.st)
    lm = LiveLandmarks(c.sg, see_through=c.st, plan_obst=c.obst.occ)
    nav = Navigator(c.obst, c.blocked, c.clear, c.need, c.sp, live=lm, gate_lines=c.gate_lines)

    def grid_msg():
        prob = np.rint(grid.probability() * 100).astype(np.int8)
        prob[grid.seen == 0] = -1
        nx, ny = grid.L.shape
        return SimpleNamespace(info=SimpleNamespace(width=nx, height=ny), data=prob.T.ravel().tolist())

    pos, yaw = np.array(START, float), math.pi / 2
    nav.plan(pos, c.goals, t=0.0)
    res = dict(replans=0, optimistic=0, plan_errors=0, held_s=0.0, min_true_clearance=math.inf, reached=False)
    t, dt, v = 0.0, 0.1, 0.25
    while t < t_max:
        t += dt
        if int(round(t / dt)) % 2 == 0:
            pose = np.array([pos[0], pos[1], yaw])
            grid.integrate(pose, scan_from(pose))
            lm.on_grid(grid_msg())
        P, prog = nav.path, nav.progress(pos)
        j = min(int(np.searchsorted(P.s, prog + 0.35)), len(P.pts) - 1)
        d = P.pts[j] - pos
        err = 0.0
        if np.hypot(*d) > 1e-6:
            err = (math.atan2(d[1], d[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
            yaw += max(-math.radians(30) * dt, min(math.radians(30) * dt, err))

        def in_view(q):
            return abs((math.atan2(q[1] - pos[1], q[0] - pos[0]) - yaw + math.pi) % (2 * math.pi) - math.pi) < 0.7
        ahead = P.pts[(P.s >= prog) & (P.s <= prog + 0.95)]
        held = any(tclr(q) < 0.32 and in_view(q) for q in ahead[1:] if np.hypot(*(q - pos)) > 0.05) \
            or abs(err) > math.radians(20)
        want = nav.needs_replan(t, prog) or (held and (nav.t_plan is None or t - nav.t_plan >= 1.0)
                                             and nav.path_blocked_ahead(prog))
        if want:
            try:
                nav.plan(pos, c.goals, t=t)
                res["replans"] += 1
                res["optimistic"] += int(nav.optimistic)
                continue
            except PlanError:
                nav.t_plan = t
                res["plan_errors"] += 1
        if held:
            res["held_s"] += dt
            continue
        i = min(int(np.searchsorted(P.s, prog + v * dt)), len(P.pts) - 1)
        pos = P.pts[i].copy()
        res["min_true_clearance"] = min(res["min_true_clearance"], tclr(pos))
        if np.hypot(*(pos - c.goals[-1])) < 0.1:
            res["reached"] = True
            break
    res["t"] = t
    return res


def test_unchanged_course_flies_as_before(course):
    r = fly(course, as_built(course))
    assert r["reached"] and r["replans"] == 0 and r["optimistic"] == 0 and r["plan_errors"] == 0


def test_changed_maze_is_learned_from_depth_and_flown(course):
    """The skybeta-14 case: a wall where the published map has the first corner, the old dividers gone."""
    r = fly(course, as_built(course, REMOVED, NEW_WALL), t_max=90.0)
    assert r["reached"], r
    assert r["optimistic"] >= 1                      # it first assumed the unconfirmed dividers away ...
    assert r["replans"] > r["optimistic"]            # ... then re-planned once depth confirmed the way
    assert r["min_true_clearance"] >= course.doc["vehicle"]["radius_m"] + 0.1, r


def test_dead_end_never_wanders_and_reports_no_path(course):
    """A new wall and nothing removed: no way through. It may try the unconfirmed openings it can see,
    but holds (refnav lands after blocked_max_repeats) instead of moving toward a surface."""
    r = fly(course, as_built(course, None, NEW_WALL), t_max=30.0)
    assert not r["reached"] and r["plan_errors"] > 0
    assert r["min_true_clearance"] >= course.doc["vehicle"]["radius_m"] + 0.1, r


def test_nets_are_never_assumed_away(course):
    lm = LiveLandmarks(course.sg, see_through=course.st, plan_obst=course.obst.occ)
    nx, ny = course.sg.occ.shape
    everything_free = SimpleNamespace(info=SimpleNamespace(width=nx, height=ny), data=[0] * (nx * ny))
    lm.on_grid(everything_free)
    net = course.st & course.obst.occ
    assert net.any()
    assert not (lm.cleared() & net).any() and not (lm.unconfirmed() & net).any()
    assert (lm.cleared() & course.obst.occ & ~course.st).any()       # solid surfaces seen through do clear


def test_a_surface_with_returns_is_not_cleared(course):
    lm = LiveLandmarks(course.sg, see_through=course.st, plan_obst=course.obst.occ)
    nx, ny = course.sg.occ.shape
    prob = np.where(course.sg.occ, 95, 0).astype(int)                # every published surface returns
    lm.on_grid(SimpleNamespace(info=SimpleNamespace(width=nx, height=ny), data=prob.T.ravel().tolist()))
    assert not lm.cleared().any() and not lm.landmarks().any() and not lm.unconfirmed().any()


# ---------------------------------------------------------------- refnav
def _path_mission(**kw):
    p = C.Params(route=[(1.0, 0.0), (3.0, 0.0)], blocked_hold_s=4.0, track_mode="path", **kw)
    m = C.Mission(p)
    m.phase, m.leg, m.targets = "go", 0, [(3.0, 0.0), (3.0, 3.0)]
    # 2 m north to a stop-and-turn corner (index 40), then 1 m east
    pts = [(k * 0.05, 0.0) for k in range(41)] + [(2.0, 0.05 * k) for k in range(1, 21)]
    assert m.on_path(10.0, 0, pts, [0.3] * len(pts), stops=[40], pid=7)
    return m, pts


def test_block_is_judged_against_the_next_corner_not_the_leg_end():
    m, _ = _path_mission()
    v = C.Vehicle(pos=(1.0, 0.0, -0.8), heading=0.0, pos_ok=True)
    remaining, s_now = m._path_remaining(10.1, v)
    assert remaining == pytest.approx(1.0) and s_now == pytest.approx(1.0)


def test_repeated_blocks_on_one_path_land():
    m, pts = _path_mission(blocked_max_repeats=3)
    v = C.Vehicle(pos=(1.0, 0.0, -0.8), heading=0.0, pos_ok=True)
    out = C.Output(phase="go")
    for k in range(3):
        m._start_recover(11.0 + k, out, "blocked", "route blocked", v)
        assert m.phase == "blocked"
        m.phase = "go"
    m._start_recover(14.0, out, "blocked", "route blocked", v)
    assert m.phase == "land" and "blocked 4 times" in out.events[-1]


def test_a_new_path_restarts_the_block_count():
    m, pts = _path_mission(blocked_max_repeats=1)
    v = C.Vehicle(pos=(1.0, 0.0, -0.8), heading=0.0, pos_ok=True)
    out = C.Output(phase="go")
    m._start_recover(11.0, out, "blocked", "route blocked", v)
    m.phase = "go"
    assert m.on_path(12.0, 0, pts, [0.3] * len(pts), stops=[40], pid=8)    # the navigator re-planned
    m._start_recover(12.1, out, "blocked", "route blocked", v)
    assert m.phase == "blocked"


def test_unlimited_by_default():
    m, _ = _path_mission()
    v = C.Vehicle(pos=(1.0, 0.0, -0.8), heading=0.0, pos_ok=True)
    out = C.Output(phase="go")
    for k in range(10):
        m._start_recover(11.0 + k, out, "blocked", "route blocked", v)
        assert m.phase == "blocked"
        m.phase = "go"
