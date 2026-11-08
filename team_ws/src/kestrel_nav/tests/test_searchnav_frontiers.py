"""Exploration ranks useful, reachable views without seeing through walls."""
import math

import numpy as np

from kestrel_nav.searchnav.grid import DepthGrid, GridParams, FREE, OCCUPIED, UNKNOWN


def scene(*, wall=False, doorway=False, symmetric=False):
    grid = DepthGrid(GridParams(extent_m=8., resolution_m=.1))
    grid.reset((0., 0.), 1.)
    xx = grid.origin[0] + (np.arange(grid.n) + .5) * grid.p.resolution_m
    yy = grid.origin[1] + (np.arange(grid.n) + .5) * grid.p.resolution_m
    free = np.ones(grid.states.shape, bool)
    if symmetric:
        free[:, :, np.abs(xx) > 2.] = False
    else:
        free[:, :, xx > 1.6] = False
    occupied = np.zeros_like(free)
    if wall:
        occupied[:, :, np.abs(xx - 1.4) < .06] = True
        if doorway:
            occupied[:, np.abs(yy) < .8, :] = False
    grid.apply_observation(free, occupied, 1.)
    return grid


def test_unknown_behind_solid_wall_is_not_an_observation_goal():
    grid = scene(wall=True)
    before = grid.states.copy()
    assert grid.frontiers((0., 0.), heading=0.) == []
    assert grid.frontier_summary['candidates'] > 0
    assert grid.frontier_summary['occluded'] == grid.frontier_summary['evaluated'] > 0
    assert np.array_equal(grid.states, before)  # Visibility must not clear unknown.


def test_doorway_exposes_frontier_and_every_selected_goal_has_a_safe_route():
    grid = scene(wall=True, doorway=True)
    choices = grid.frontiers((0., 0.), heading=0.)
    assert choices
    for (_, goal), detail in zip(choices, grid.frontier_details):
        path = grid.astar((0., 0.), goal)
        assert path is not None and grid.route_safe(path)
        assert detail['visible_unknown_rays'] > 0
        assert detail['extra_clearance_m'] > 0
        assert detail['travel_m'] > .35


def test_equally_useful_frontiers_prefer_less_turning():
    grid = scene(symmetric=True)
    east = grid.frontiers((0., 0.), heading=0.)
    west = grid.frontiers((0., 0.), heading=math.pi)
    assert east and west
    assert east[0][1][0] > 0 and west[0][1][0] < 0


def test_visited_views_are_excluded_and_unknown_start_has_no_frontier():
    grid = scene(symmetric=True)
    first = grid.frontiers((0., 0.))[0][1]
    choices = grid.frontiers((0., 0.), visited=[first])
    assert all(np.linalg.norm(point-first) >= .45 for _, point in choices)
    empty = DepthGrid(GridParams(extent_m=4.))
    empty.reset((0., 0.), 1.)
    assert empty.frontiers((0., 0.)) == []
    assert 'unknown' in empty.frontier_summary['reason']
