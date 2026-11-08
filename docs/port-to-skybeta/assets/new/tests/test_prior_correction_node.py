"""Exercise trust transitions and evidence lifetime through real NavNode callbacks."""
import json
from types import SimpleNamespace as NS
import numpy as np
import pytest
from test_nav_recovery_protocol import node, request
from test_observed_recovery import observed
from test_prior_correction import wall_scene, all_observed


def message(**fields):
    return NS(data=json.dumps(fields))


def grid_message(*, frame='coursenav_nominal', data=None, qz=0.):
    return NS(header=NS(stamp=NS(sec=10,nanosec=0),frame_id=frame),
              info=NS(origin=NS(position=NS(x=0.,y=0.),orientation=NS(x=0.,y=0.,z=qz,w=1.)),
                      resolution=.1,width=80,height=50),
              data=[12]*4000 if data is None else data)


def install_wall_scene(n):
    n.nav=wall_scene();n.obst=n.nav.prior_obst;n.surf_dist=n.obst.distance()
    n.reference_path=None;n.reference_leg=0;n.cache={}
    n.observed_clear=None;n.local_pts=None
    n.nav_legs=[{'path':True,'goals':[[6.,2.5]],'stop':[6.,2.5]}]
    n.status={'leg':0,'phase':'go'}


def test_prior_restore_invalidates_local_only_path_and_cached_plan(node):
    install_wall_scene(node)
    node._on_prior(message(state='dropped'))
    node.observed_free=all_observed(node.nav)
    node._tick()
    assert node.messages[-1]['plan_state']=='ready'
    assert node.messages[-1]['plan_mode']=='local_only'
    assert node.nav.path_certificate is not None
    node.cache[0]=('must never reuse',)
    node._on_prior(message(state='placed'))
    assert node.nav.path is None and node.nav.path_certificate is None and not node.cache
    node._tick()
    assert node.messages[-1]['plan_state']=='failed'
    assert 'pts_ne' not in node.messages[-1]


def test_recovery_certificate_expires_after_resume_in_normal_go_phase(node):
    mask=np.zeros(node.obst.occ.shape,bool);mask[42:48,46:54]=True
    node.nav.set_local(mask);node.observed_free=observed()
    request(node,1);node._tick()
    assert node.messages[-1]['plan_state']=='ready' and node.messages[-1]['recovery_certified']
    pid=node.messages[-1]['pid']
    node._on_status(message(leg=0,phase='go',path_pid=pid,path_s=0.))
    assert node.recovery is None
    node.now=10.2;node._tick()
    assert node.messages[-1]['plan_state']=='ready' and 'pts_ne' in node.messages[-1]
    node.now=12.01;node._tick()
    out=node.messages[-1]
    assert out['plan_state']=='failed' and 'expired' in out['reason']
    assert out['request_id'] is None and 'pts_ne' not in out and node.nav.path is None


@pytest.mark.parametrize('bad',[
    {'frame':'course'}, {'qz':float('nan')}, {'data':[12]},
])
def test_malformed_grid_revokes_both_certification_and_pending_clearing(node,bad):
    install_wall_scene(node);node._on_prior(message(state='dropped'))
    node._on_local(grid_message())
    assert node.observed_free is not None and node.observed_clear is not None
    node._on_local(grid_message(**bad))
    assert node.observed_free is None and node.observed_clear is None
    node._tick()
    assert not node.nav.cleared.any()
    assert node.messages[-1]['plan_state']=='failed' and 'pts_ne' not in node.messages[-1]


def test_unknown_prior_does_not_allow_live_free_cells_to_erase_wall(node):
    install_wall_scene(node);node._on_local(grid_message())
    node._on_prior(message(state='unknown'));node._tick()
    assert not node.nav.cleared.any()
    assert node.messages[-1]['plan_state']=='failed'
    node._on_prior(message(state='dropped'));node._tick()
    assert node.nav.cleared[35,25]
    assert node.messages[-1]['plan_state']=='ready'


def test_prior_drop_preserves_nonpublishable_preplan_for_first_failure(node):
    reference=node.reference_path
    node.preplan_references={0:(reference,node.nav.goals)}
    node.reference_path=None;node.reference_leg=None
    node._on_prior(message(state='dropped'))
    assert not node.cache and node.nav.path is None
    request(node,1)
    assert node.recovery_reference is reference and node.recovery_progress==0.
    assert node.nav.path is None  # reference alone can never authorize a command
