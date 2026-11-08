"""Behavioral boundaries for correcting a contradicted prior and certified fallback."""
import numpy as np
import pytest
from coursenav.course import Grid2D
from coursenav.navigator import Navigator
from coursenav.observed_free import ObservedFree
from coursenav.planner import PlanError
from coursenav.smooth import SmoothParams


def wall_scene(wall=True):
    occ=np.zeros((80,50),bool);occ[[0,-1],:]=True;occ[:,[0,-1]]=True
    boundary=occ.copy()
    if wall:occ[35,:]=True
    grid=Grid2D(occ,np.zeros(2),.1);clear=grid.distance()
    nav=Navigator(grid,Grid2D(clear<.2,grid.origin,grid.res),clear,.2,
                  SmoothParams(need_m=.2,cruise_mps=.3),protect=boundary,fixed_blocked=boundary)
    return nav


def all_observed(nav,value=12,stamp=10.):
    shape=nav.prior_obst.occ.shape
    return ObservedFree.from_grid([0,0],.1,*shape,np.full(shape,value).T.ravel(),stamp)


def test_clearing_requires_dropped_and_keeps_protected_fixed_confirmed_cells():
    nav=wall_scene();nav.protect[35,5]=True;nav.fixed[35,10]=True
    free=np.ones(nav.prior_obst.occ.shape,bool)
    assert nav.set_free(free)==0 and not nav.cleared.any()
    nav.set_prior_dropped(True)
    local=np.zeros_like(free);local[35,20]=True;nav.set_local(local)
    assert nav.set_free(free)>0
    assert not nav.cleared[35,5] and not nav.cleared[35,10] and not nav.cleared[35,20]
    assert nav.cleared[35,30]
    nav.set_local(None);nav.set_free(free)
    assert nav.confirmed[35,20] and not nav.cleared[35,20]
    assert nav.local_only_map()[0].occ[35,20]
    nav.set_prior_dropped(False)
    assert not nav.cleared.any() and not nav.confirmed.any()
    assert nav.fuse()[0].occ[35,30]


def test_equal_cell_counts_at_different_locations_refresh_fused_geometry():
    nav=wall_scene(False);a=np.zeros(nav.prior_obst.occ.shape,bool);b=a.copy()
    a[20,10]=True;b[50,35]=True
    nav.set_local(a);assert nav.fuse()[0].occ[20,10]
    nav.set_local(b);fused=nav.fuse()[0].occ
    assert not fused[20,10] and fused[50,35]


@pytest.mark.parametrize('value',[None,-1,50,100])
def test_local_only_fallback_cannot_cross_unknown_without_observations(value):
    nav=wall_scene();nav.set_prior_dropped(True)
    obs=None if value is None else all_observed(nav,value)
    with pytest.raises(PlanError):
        nav.plan([1.,2.5],[[6.,2.5]],observed=obs,to_nominal=lambda p:p)
    assert nav.path is None


def test_fully_observed_fallback_crosses_false_wall_with_nonzero_speed():
    nav=wall_scene();nav.set_prior_dropped(True);obs=all_observed(nav)
    path=nav.plan([1.,2.5],[[6.,2.5]],observed=obs,to_nominal=lambda p:p)
    assert nav.plan_mode=='local_only'
    assert np.any(path.pts[:,0]<3.5) and np.any(path.pts[:,0]>3.6)
    assert obs.certifies_path(path.pts,lambda p:p,.2)
    assert np.max(path.v)>.1
    assert nav.path_certificate is not None
    nav.set_prior_dropped(False)
    assert nav.path is None and nav.path_certificate is None
    with pytest.raises(PlanError):nav.plan([1.,2.5],[[6.,2.5]],observed=obs,to_nominal=lambda p:p)


def test_protected_wall_cannot_be_bypassed_even_when_every_cell_observed():
    nav=wall_scene();nav.protect[35,:]=True;nav.set_prior_dropped(True)
    with pytest.raises(PlanError):
        nav.plan([1.,2.5],[[6.,2.5]],observed=all_observed(nav),to_nominal=lambda p:p)


def test_unchanged_fused_field_keeps_exact_clearance_and_unrelated_change_does_not_replan():
    nav=wall_scene(False)
    exact=nav.prior_clear+.003
    nav.prior_clear=exact
    result=nav.fuse()[2]
    np.testing.assert_array_equal(result[~nav.prior_blocked.occ],exact[~nav.prior_blocked.occ])
    nav.plan([1.,2.5],[[6.,2.5]],t=0.)
    local=np.zeros(nav.prior_obst.occ.shape,bool);local[20,5]=True;nav.set_local(local)
    assert not nav.needs_replan(2.,0.)
