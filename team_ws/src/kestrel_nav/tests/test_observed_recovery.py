"""Recovery is a certified local detour, never permission to erase prior geometry."""
import math
import numpy as np
import pytest
from coursenav.course import Grid2D
from coursenav.drift import DriftEKF
from coursenav.navigator import Navigator
from coursenav.observed_free import ObservedFree
from coursenav.planner import PlanError
from coursenav.smooth import SmoothParams


def scene():
    occ = np.zeros((160, 100), bool)
    occ[[0, -1], :] = True
    occ[:, [0, -1]] = True
    grid = Grid2D(occ, np.zeros(2), .05)
    clear = grid.distance()
    nav = Navigator(grid, Grid2D(clear < .25, grid.origin, grid.res), clear, .25,
                    SmoothParams(need_m=.25, cruise_mps=.3))
    ref = nav.plan([1., 2.5], [[6., 2.5]])
    return nav, ref


def observed(block=True, unknown=False, stamp=10.):
    prob = np.full((160, 100), 12)
    prob[85:, :] = -1  # end of original leg is UNOBSERVED, suffix must remain exact
    if block:
        prob[42:48, 46:54] = -1 if unknown else 100
    return ObservedFree.from_grid([0, 0], .05, 160, 100, prob.T.ravel(), stamp)


def test_certification_never_treats_unknown_or_ambiguous_as_free():
    for value in [-1, 50, 88, 100]:
        g = ObservedFree.from_grid([0, 0], .05, 40, 40, np.full(1600, value), 10.)
        assert not g.certifies_path([[.8, 1.], [1.2, 1.]], lambda p:p, .25)
    g = ObservedFree.from_grid([0, 0], .05, 40, 40, np.full(1600, 12), 10.)
    assert g.certifies_path([[.8, 1.], [1.2, 1.]], lambda p:p, .25)
    assert not g.certifies_path([[.01, 1.], [.2, 1.]], lambda p:p, .25)
    assert not g.fresh(12.01, 2.) and not g.fresh(9.99, 2.)


def test_nominal_course_transform_and_swept_footprint():
    g = observed(block=True)
    ekf = DriftEKF.from_state([3., -2., math.pi/2], [1., 2.5, 0.])
    nominal = np.array([[1., 1.], [3., 1.]])
    course = ekf.to_course(nominal, [1., 2.5, 0.])
    assert g.certifies_path(course, lambda p:ekf.to_nominal(p, [1., 2.5, 0.]), .25)
    assert not g.certifies_path([[1., 2.3], [3., 2.3]], lambda p:p, .25)  # footprint hits obstacle


def test_safe_observed_detour_keeps_exact_unobserved_reference_suffix():
    nav, ref = scene()
    mask = np.zeros(nav.prior_obst.occ.shape, bool);mask[42:48,46:54] = True
    nav.set_local(mask)
    path, prefix, join = nav.plan_recovery([1., 2.5], ref, observed(), lambda p:p)
    assert observed().certifies_path(prefix, lambda p:p, nav.need)
    assert np.array_equal(path.pts[len(prefix)-1:], ref.pts[join:])
    assert len(prefix)-1 in path.stops and path.v[len(prefix)-1] == 0.
    assert path.pts[-1].tolist() == [6., 2.5]
    assert np.max(abs(prefix[:,1]-2.5)) > .3


def test_unknown_corridor_and_prior_wall_cannot_be_cleared_by_free_evidence():
    nav, ref = scene()
    nav.prior_blocked.occ[35:65,:] = True
    nav._fused = (nav.prior_obst, nav.prior_blocked, nav._effective(nav.prior_clear,nav.prior_blocked))
    with pytest.raises(PlanError):
        nav.plan_recovery([1., 2.5], ref, observed(block=False), lambda p:p)
    nav, ref = scene()
    g = ObservedFree.from_grid([0,0],.05,160,100,np.full(16000,-1),10.)
    with pytest.raises(PlanError):
        nav.plan_recovery([1.,2.5],ref,g,lambda p:p)


def test_new_pid_on_same_geometry_is_not_a_recovery():
    nav, ref = scene()
    with pytest.raises(PlanError, match='does not leave'):
        nav.plan_recovery([1.,2.5],ref,observed(block=False),lambda p:p)
