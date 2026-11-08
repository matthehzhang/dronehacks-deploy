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
    nav = NS(path=object(), p=NS(replan_every_s=1.), needs_replan=lambda *a:True, plan=fail)
    node = NS(drift={'delta':[0.,0.,0.], 'nominal':[0.,0.,0.], 'fixture':[0.,0.,0.]},
              status={'leg':0,'phase':'go','v_cmd':0.}, nav=nav, decl=0.,
              nav_leg=0, pid=3, cache={}, last_attempt=-math.inf, search_replanned=False,
              stats={'plan_errors':0}, nav_legs=[{'path':True,'stop':[2.,0.],'goals':[[2.,0.]]}],
              get_logger=lambda:NS(warning=lambda *a:None),
              pub=NS(publish=lambda m:published.append(json.loads(m.data))))
    NavNode._tick(node)
    assert nav.path is None
    assert published[-1]['valid'] is False
    assert not published[-1].get('pts_ne')
