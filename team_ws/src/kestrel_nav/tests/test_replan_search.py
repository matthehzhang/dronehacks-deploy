import math
from refnav.core import Mission, Params, Vehicle

def flying():
    m=Mission(Params(track_mode="path",corridor_search=True,corridor_wait_land_s=1.))
    m.phase="go";m.leg=0;m.targets=[(2.,0.)];m.sp=[0.,0.,-.7,0.]
    m.home=[0.,0.,0.,0.];m.leg_dir=(1.,0.);m.leg_len=2.;m.leg_heading=0.
    m.t_started=0.;m.t_phase=0.;m.t_last=1.;m.t_check=0.
    m.last_v=Vehicle(pos=(0.,0.,-.7),heading=0.,pos_ok=True,vel=(0.,0.,0.))
    m.health.bad_for=lambda now:0.
    m.depth_t=2.;m.offboard_seen=False
    return m

def test_changed_path_faces_new_direction_before_translation():
    m=flying();m.t_check=2.
    assert m.on_path(2.,0,[(0.,0.),(0.,2.)],[0.,.3],pid=2)
    o=m.step(2.,m.last_v)
    assert m.phase=="turn"
    assert m.sp[:2]==[0.,0.]
    assert m.v_cmd==0 and not o.commands
    assert any("new A* route" in e for e in o.events)

def test_unobserved_corridor_searches_instead_of_landing():
    m=flying();m.path=None;m.corr_t=None
    o=m.step(2.,m.last_v)
    assert m.phase=="search" and m.sp[:2]==[0.,0.]
    assert m.v_cmd==0 and "land" not in o.commands
    assert m.sp[3]>0


def test_repeated_invalid_paths_do_not_move_climb_or_hover_anchor():
    import json
    from types import SimpleNamespace as NS
    import pytest
    pytest.importorskip('rclpy')
    from refnav.node import RefNav
    m=flying();m.phase='climb';m.sp=[0.,0.,-.7,0.]
    m.last_v.pos=(.2,.3,-.7);m.path=None
    n=NS(mission=m,trace=NS(record=lambda *a,**k:None),_now=lambda:2.)
    invalid=NS(data=json.dumps(dict(leg=0,valid=False)))
    RefNav._on_path(n,invalid)
    assert m.sp[:2]==[0.,0.]
    m.phase='go';m.path={'pid':1}
    RefNav._on_path(n,invalid)
    assert m.sp[:2]==[.2,.3]
    m.last_v.pos=(.4,.5,-.7)
    RefNav._on_path(n,invalid)
    assert m.sp[:2]==[.2,.3]
