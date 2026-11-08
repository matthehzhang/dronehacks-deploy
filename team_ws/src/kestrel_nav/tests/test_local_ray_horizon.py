"""Finite distant returns support bounded free space, never an artificial horizon wall."""
import numpy as np
import pytest
from coursenav.localgrid import LocalGrid, LocalParams, free_points_from_grid
from coursenav.scan import Scan, scan_from_points, local_scan_from_points


def ray(distance, sigma=.05, bearing=0., origin=True):
    origin_xy = np.zeros(2) if origin is True else origin
    return Scan(np.array([bearing]), np.array([distance]), np.array([sigma]),
                free_origin_xy=origin_xy)


def grid():
    return LocalGrid(LocalParams(max_range_m=3., decay_s=0., min_cells=1))


def value(g, xy):
    return g.L[tuple(g.ij(xy)[0])]


def seen(g, xy):
    return g.seen[tuple(g.ij(xy)[0])]


@pytest.mark.parametrize('distance', [5., 8.])
def test_far_valid_return_carves_inside_horizon_without_phantom_endpoint(distance):
    g=grid()
    for _ in range(3):
        assert g.integrate(0., (0.,0.,0.), ray(distance, .05+.02*distance**2))
    assert value(g, [1.,0.])<0 and value(g, [2.8,0.])<0
    assert not (g.L>0).any()  # No obstacle at the clipped 3 m endpoint.
    assert seen(g, [3.1,0.])==0 and seen(g, [-.2,0.])==0
    assert seen(g, [1.,.5])==0  # A ray is not a filled field-of-view wedge.
    pts=free_points_from_grid(*g.grid_fields(now=0.))
    assert len(pts)>0 and np.max(pts[:,0])<3.


@pytest.mark.parametrize('distance', [np.nan,np.inf,-np.inf,0.,-1.])
def test_missing_or_invalid_return_never_marks_free_or_occupied(distance):
    g=grid()
    assert not g.integrate(0., (0.,0.,0.), ray(distance))
    assert not g.seen.any() and not g.L.any()


@pytest.mark.parametrize('sigma', [np.nan,np.inf,-1.])
def test_far_return_requires_valid_uncertainty(sigma):
    g=grid()
    assert not g.integrate(0., (0.,0.,0.), ray(5., sigma))
    assert not g.seen.any()


def test_uncertain_far_return_cannot_certify_horizon():
    g=grid()
    g.integrate(0., (0.,0.,0.), ray(5., 10.))
    assert not g.seen.any()


def test_nearby_endpoint_remains_an_obstacle_and_space_behind_it_unknown():
    g=grid()
    for _ in range(3):g.integrate(0., (0.,0.,0.), ray(1., .07))
    assert g.occupied()[tuple(g.ij([1.,0.])[0])]
    assert value(g, [.5,0.])<0
    assert seen(g, [1.2,0.])==0


def test_endpoint_at_horizon_is_real_but_beyond_horizon_is_not_clipped_to_a_hit():
    exact=grid();beyond=grid()
    exact.integrate(0., (0.,0.,0.), ray(3., .01))
    beyond.integrate(0., (0.,0.,0.), ray(3.01, .01))
    assert value(exact, [3.,0.])>0
    assert not (beyond.L>0).any()


def test_far_ray_free_evidence_still_expires_and_nonreturns_do_not_refresh_it():
    g=grid()
    for _ in range(3):g.integrate(0., (0.,0.,0.), ray(8., 1.33))
    assert len(free_points_from_grid(*g.grid_fields(now=8.)))>0
    g.integrate(8.1, (0.,0.,0.), ray(np.inf))
    assert len(free_points_from_grid(*g.grid_fields(now=8.1)))==0


def test_same_scan_near_hit_wins_over_a_far_ray_through_its_cell():
    g=grid();s=Scan(np.array([0.,0.]),np.array([1.,8.]),np.array([.07,1.33]), free_origin_xy=np.zeros(2))
    g.integrate(0., (0.,0.,0.), s)
    assert value(g,[1.,0.])>0 and not np.isfinite(g.last_free[tuple(g.ij([1.,0.])[0])])


def test_far_return_without_camera_origin_fails_closed():
    g=grid()
    assert not g.integrate(0., (0.,0.,0.), ray(8.,1.33,origin=None))
    assert not g.seen.any()


def test_uncertainty_margin_shortens_far_ray_before_horizon():
    g=grid()
    g.integrate(0., (0.,0.,0.), ray(5.,1.2))
    assert seen(g,[.8,0.])>0
    assert seen(g,[1.6,0.])==0  # 5 - 3*1.2 - cell margin is less than 1.6 m.
    assert not (g.L>0).any()


def test_offset_camera_does_not_clear_unmeasured_body_to_camera_segment():
    g=grid()
    origin=np.array([.2,.4]);endpoint=np.array([8.,.4])
    g.integrate(0.,(0.,0.,0.),ray(np.linalg.norm(endpoint),.05,
                                np.arctan2(endpoint[1],endpoint[0]),origin))
    assert seen(g,[.5,.4])>0 and seen(g,[2.5,.4])>0
    assert seen(g,[.1,.4])==0 and seen(g,[.5,0.])==0
    assert seen(g,[3.1,.4])==0 and not (g.L>0).any()


def test_offset_origin_rotates_with_body_and_horizon_stays_body_centred():
    g=grid()
    g.integrate(0.,(1.,-1.,np.pi/2),ray(8.,.05,origin=np.array([.2,0.])))
    assert seen(g,[1.,-.5])>0 and seen(g,[1.,1.8])>0
    assert seen(g,[1.,-1.1])==0 and seen(g,[1.,2.1])==0


def points(ranges, bearing=0., z=0.):
    ranges=np.asarray(ranges,float)
    return np.column_stack([ranges*np.cos(bearing),ranges*np.sin(bearing),np.full(len(ranges),z)])


def extract(pts, origin=None):
    return local_scan_from_points(pts,.8,.3,1.6,obstacle_range_m=3.,ray_range_m=12.,
                                  camera_origin_xy=np.zeros(2) if origin is None else origin,
                                  n_bins=1,hfov_rad=1.,min_pts=3)


@pytest.mark.parametrize('far_range',[5.,8.])
def test_scan_retains_far_valid_returns_and_actual_sampled_bearing(far_range):
    origin=np.array([.1,.04]);pts=points([far_range,far_range+.05,far_range+.1],bearing=.08)
    scan=extract(pts,origin)
    assert len(scan)==1 and scan.range[0]>3.
    assert np.isclose(scan.bearing[0],.08)  # not the angular-bin centre, which is zero
    np.testing.assert_allclose(scan.free_origin_xy,origin)


def test_dense_far_points_cannot_move_existing_near_return_quantile():
    near=points(np.linspace(.6,1.5,10),bearing=.04)
    far=points(np.full(1000,8.),bearing=.1)
    legacy=scan_from_points(near,.8,.3,1.6,rmax=3.,n_bins=1,hfov_rad=1.,min_pts=3)
    combined=extract(np.vstack([near,far]))
    np.testing.assert_array_equal(combined.range,legacy.range)
    np.testing.assert_array_equal(combined.bearing,legacy.bearing)
    np.testing.assert_array_equal(combined.sigma,legacy.sigma)


@pytest.mark.parametrize('near_range',[.1,.3,.6,3.])
def test_even_one_finite_near_point_vetoes_far_free_ray_in_same_bin(near_range):
    scan=extract(np.vstack([points([near_range]),points(np.full(100,8.))]))
    assert len(scan)==0  # insufficient near support is unknown, not a far free-space ray


def test_near_points_outside_height_band_do_not_veto_valid_far_ray():
    scan=extract(np.vstack([points([.2],z=2.),points([5.,5.1,5.2])]))
    assert len(scan)==1 and scan.range[0]>3.


@pytest.mark.parametrize('pts',[
    np.zeros((5,3)), np.full((5,3),np.nan),
    np.array([[np.inf,0.,0.]]*5), points([13.,14.,15.]),
])
def test_scan_missing_invalid_or_beyond_sensor_range_is_unknown(pts):
    assert len(extract(pts))==0
