"""A failed replan must withdraw the previously published route."""
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip('rclpy')
from coursenav.nav_node import NavNode
from coursenav.planner import PlanError


def test_failed_replan_does_not_republish_previous_points():
    published = []
    def fail(*args, **kwargs):raise PlanError('observed obstruction')
    import numpy as np
    nav = NS(path=object(), p=NS(replan_every_s=1.), needs_replan=lambda *a:True, plan=fail,
             local_obstacles=np.zeros((2,2),bool),map_changed=lambda:False)
    node = NS(drift={'delta':[0.,0.,0.], 'nominal':[0.,0.,0.], 'fixture':[0.,0.,0.]},
              status={'leg':0,'phase':'go','v_cmd':0.}, nav=nav, decl=0.,
              nav_leg=0, pid=3, cache={}, last_attempt=-math.inf, search_replanned=False,
              local_stamp=None,local_error='',local_required=False,
              get_clock=lambda:NS(now=lambda:NS(nanoseconds=10_000_000_000)),
              stats={'plan_errors':0}, nav_legs=[{'path':True,'stop':[2.,0.],'goals':[[2.,0.]]}],
              get_logger=lambda:NS(warning=lambda *a:None),
              pub=NS(publish=lambda m:published.append(json.loads(m.data))))
    NavNode._tick(node)
    assert nav.path is None
    assert published[-1]['valid'] is False
    assert not published[-1].get('pts_ne')
    # Absence of a path must itself request another bounded A* attempt. The
    # map-change predicate can already be consumed by the previous attempt.
    nav.needs_replan=lambda *a:False
    node.last_attempt=-math.inf
    NavNode._tick(node)
    assert node.stats['plan_errors']==2 and not published[-1].get('pts_ne')


def test_passed_intermediate_goal_never_returns_when_aircraft_moves_away():
    from coursenav.nav_node import advance_goals
    import numpy as np
    goals=[[0.,0.],[2.,0.],[4.,0.]]
    i=advance_goals(goals,np.array([0.,0.]))
    assert i==1
    assert advance_goals(goals,np.array([1.,0.]),i)==1
    i=advance_goals(goals,np.array([2.,0.]),i)
    assert i==2
    assert advance_goals(goals,np.array([0.,0.]),i)==2
