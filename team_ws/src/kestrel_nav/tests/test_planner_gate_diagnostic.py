"""A blocked approach must name the opening crossed by the fallback path."""
import numpy as np
import pytest

from kestrel_nav.coursenav.course import Grid2D
from kestrel_nav.coursenav import planner


@pytest.mark.parametrize("goal_x, expected", [(4, "second"), (7, "second, first")])
def test_gate_failure_names_crossed_line_not_next_goal(goal_x, expected):
    grid = Grid2D(np.zeros((160, 100), bool), np.zeros(2), .05)
    clearance = np.full(grid.occ.shape, 10.)
    doc = dict(vehicle=dict(radius_m=.28, margin_m=.12), start=dict(xy=[1, 2]))
    goals = [dict(name='approach_first', xy=[goal_x, 2]),
             dict(name='first', xy=[6, 2], gate_line=[[6, 0], [6, 5]], normal=[1, 0]),
             dict(name='second', xy=[3, 2], gate_line=[[3, 0], [3, 5]], normal=[1, 0])]
    with pytest.raises(planner.PlanError) as error:
        planner.plan(doc, grid, grid, clearance, goals=goals)
    message = str(error.value)
    assert f'reachable only through {expected}:' in message
    assert 'through first' not in message


def test_gate_exclusion_without_a_crossing_does_not_invent_one():
    grid = Grid2D(np.zeros((160, 100), bool), np.zeros(2), .05)
    clearance = np.full(grid.occ.shape, 10.)
    doc = dict(vehicle=dict(radius_m=.28, margin_m=.12), start=dict(xy=[1, 2]))
    goals = [dict(name='approach', xy=[2.975, 2]),
             dict(name='opening', xy=[3, 2], gate_line=[[3, 0], [3, 5]], normal=[1, 0])]
    with pytest.raises(planner.PlanError, match='no gate centreline crossing'):
        planner.plan(doc, grid, grid, clearance, goals=goals)
