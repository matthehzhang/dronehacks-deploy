"""Changing the flight-height slab cannot silently reuse an old 2-D path."""
from types import SimpleNamespace

import numpy as np
import pytest

from kestrel_nav.searchnav.grid import DepthGrid, GridParams, UNKNOWN, OCCUPIED, FREE
from kestrel_nav.searchnav.mission import SearchMission, Observation
from kestrel_nav.searchnav.opening import ClearanceCertificate


def setup():
    grid=DepthGrid(GridParams(extent_m=6.))
    grid.reset((0.,0.),1.)
    grid.apply_observation(np.ones_like(grid.states,bool),np.zeros_like(grid.states,bool),1.)
    m=SearchMission(grid);m.started=1.;m.hold_z=1.
    return m


def obs(t=2., **change):
    opening=SimpleNamespace(center_world=(2.,0.,1.3),normal_world=(1.,0.,0.),
        right_world=(0.,-1.,0.),up_world=(0.,0.,1.),body_fits=True,stamp=t)
    values=dict(now=t,position=(0.,.2,1.),yaw=0.,ready=True,armed=True,
                offboard=True,landed=False,opening=opening)
    values.update(change)
    return Observation(**values)


def cert(start,end,radius,half_height,now):
    return ClearanceCertificate(True,now,start,end,radius,half_height,'test measured volume')


def test_centering_requires_3d_certificate_and_never_blindly_traverses():
    m=setup();m.begin_scan(obs(),'test')
    assert m.tick(obs(2.1)).phase=='center_opening'
    blocked=m.tick(obs(2.2))
    assert blocked.velocity==(0.,0.,0.,0.)
    assert m.centering_status['requires_3d_clearance']
    allowed=m.tick(obs(2.3,certify_sweep=cert))
    assert allowed.velocity[0]==0. and allowed.velocity[1]<0. and allowed.velocity[2]>0.
    assert np.linalg.norm(allowed.velocity[:3])<=.15+1e-9


def test_centered_height_rebuild_holds_until_adapter_finishes_and_discards_old_path():
    m=setup();m.begin_scan(obs(),'test');m.tick(obs(2.1))
    result=m.tick(obs(2.2,position=(0.,0.,1.3),certify_sweep=cert))
    assert result.phase=='rebuild_height' and not m.route
    assert m.hold_z==pytest.approx(1.3) and m.grid.height==pytest.approx(1.3)
    assert np.all(m.grid.states==UNKNOWN)
    waiting=m.tick(obs(2.3,position=(0.,0.,1.3),slab_ready=False))
    assert waiting.phase=='rebuild_height' and waiting.velocity==(0.,0.,0.,0.)
    assert m.tick(obs(2.4,position=(0.,0.,1.3),slab_ready=True)).phase=='scan'
    assert not m.route  # No speculative motion before fresh-map A*.


def test_missing_clearance_is_bounded_and_authority_loss_still_yields():
    m=setup();m.begin_scan(obs(),'test');m.tick(obs(2.1));m.tick(obs(2.2))
    assert m.tick(obs(4.3)).phase=='rebuild_height'
    assert m.hold_z==1.  # Hold reached height; don't command a guessed correction.
    m=setup();m.begin_scan(obs(),'test');m.tick(obs(2.1))
    lost=m.tick(obs(2.2,offboard=False,certify_sweep=cert))
    assert lost.phase=='yield' and lost.velocity is None


def test_old_height_replay_and_duplicate_frames_cannot_clear_new_obstacle():
    m=setup();g=m.grid
    hit=np.zeros_like(g.states,bool);i,j=g.cell((.5,0.));hit[:,i,j]=True
    g.apply_observation(np.zeros_like(hit),hit,3.)
    for old in (1.5,2.,2.5):g.apply_observation(hit,np.zeros_like(hit),old)
    assert np.all(g.states[hit]==OCCUPIED) and np.all(g.clear_count[hit]==0)
    g.apply_observation(hit,np.zeros_like(hit),4.)
    for _ in range(4):g.apply_observation(hit,np.zeros_like(hit),4.)
    assert np.all(g.states[hit]==OCCUPIED) and np.all(g.clear_count[hit]==1)
    g.apply_observation(hit,np.zeros_like(hit),4.1)
    assert np.all(g.states[hit]==OCCUPIED)
    g.apply_observation(hit,np.zeros_like(hit),4.2)
    assert np.all(g.states[hit]==FREE)
    g.apply_observation(np.zeros_like(hit),hit,3.5)
    assert np.all(g.states[hit]==FREE)  # Older frame may not resurrect stale hits either.
