"""Directed openings are graph barriers except for their own collision-checked hop."""
import copy
import math

import numpy as np

from kestrel_nav.coursenav import planner
from kestrel_nav.coursenav.course import Grid2D, load_course, plan_grids, resolve
from kestrel_nav.coursenav.navfield import Gate, NavField, Stage, stages_from_course
from kestrel_nav.coursenav.objmap import CourseMap


def descend_all(nf, start):
    xy, stage, points, crossed = start, 0, [start], []
    for _ in range(100):
        path, stage, gates = nf.descend(xy, stage, lookahead_m=1)
        points.extend(path[1:])
        crossed.extend(gates)
        if np.allclose(xy, path[-1]):
            break
        xy = path[-1]
    return np.asarray(points), crossed


def test_zero_reward_window_is_used_when_gate_b_is_unreachable():
    doc = copy.deepcopy(load_course())
    doc['chicane_options']['window']['points'] = 0
    cm = CourseMap.load(resolve(doc, 'map'))
    _, blocked, clearance = plan_grids(cm, doc)
    nf = NavField(blocked, clearance, stages_from_course(doc))
    path, crossed = descend_all(nf, doc['start']['spawn_xy'])
    assert crossed == ['maze_exit', 'tunnel_mouth', 'tunnel_far', 'window']
    for g in nf.gates:
        crossings = planner.gate_crossings(path, (g.p, g.q), g.normal)
        assert [d for _, d in crossings] == ([1] if g.name in crossed else [])
    assert planner.gate_crossings(path, planner.gate_segment(doc['chicane_options']['gate_b']), (-1, 0)) == []


def test_gate_hop_checks_its_interior_for_obstacles():
    grid = Grid2D(np.zeros((80, 80), bool), np.zeros(2), .05)
    grid.occ[40, :] = True  # hop endpoints are free, but its middle is a solid wall
    gate = Gate('opening', (2, .5), (2, 3.5), (1, 0), margin_m=.4)
    nf = NavField(grid, grid.distance(), [Stage('cross', [gate]), Stage('end', pad_xy=(3, 2))])
    assert not math.isfinite(nf.cost_to_go((1, 2), 0))
    path, crossed = descend_all(nf, (1, 2))
    assert len(path) == 1 and crossed == []


def test_wrong_side_start_cannot_cross_back_or_snap_through_gate():
    grid = Grid2D(np.zeros((80, 80), bool), np.zeros(2), .05)
    gate = Gate('opening', (2, 0), (2, 4), (1, 0), margin_m=.4)
    nf = NavField(grid, np.full(grid.occ.shape, 10.),
                  [Stage('cross', [gate]), Stage('end', pad_xy=(3, 2))])
    path, crossed = descend_all(nf, (3, 1))
    assert len(path) == 1 and not crossed
    path, crossed = descend_all(nf, (1, 2))
    assert crossed == ['opening']
    assert [d for _, d in planner.gate_crossings(path, (gate.p, gate.q), gate.normal)] == [1]
