"""Free observations age independently of occupancy decay and ROS publication."""
import numpy as np
from coursenav.localgrid import LocalGrid, LocalParams, occupied_points_from_grid
from coursenav.scan import Scan


def ray(distance=2.):
    return Scan(np.array([0.]),np.array([distance]),np.array([.01]))


def free_grid(t=0.):
    grid=LocalGrid(LocalParams(min_cells=1))
    for _ in range(5):grid.integrate(t,[0.,0.,0.],ray())
    return grid


def value(grid,point,now=None):
    ij=grid.ij(point)[0]
    data=grid.grid_fields(now=now)[-1].reshape(grid.n,grid.n).T
    return int(data[tuple(ij)])


def test_zero_time_free_observation_is_valid_and_republish_does_not_refresh():
    grid=free_grid(0.)
    assert 0<=value(grid,[1.,0.],now=0.)<=35
    assert 0<=value(grid,[1.,0.],now=8.)<=35
    before=grid.last_free.copy()
    assert value(grid,[1.,0.],now=8.01)==-1
    assert value(grid,[1.,0.],now=9.)==-1
    assert np.array_equal(before,grid.last_free)
    assert value(grid,[1.,0.])>=0  # optional age gate preserves legacy callers
    assert value(grid,[1.,0.],now=-.01)==-1  # clock must not precede evidence


def test_empty_or_out_of_range_scan_cannot_renew_free_evidence():
    grid=free_grid(0.)
    before=grid.last_free.copy()
    assert not grid.integrate(7.,[0.,0.,0.],Scan.empty())
    assert not grid.integrate(8.1,[0.,0.,0.],ray(4.))
    assert np.array_equal(before,grid.last_free)
    assert value(grid,[1.,0.],now=8.1)==-1
    grid.integrate(8.2,[0.,0.,0.],ray())
    assert 0<=value(grid,[1.,0.],now=8.2)<=35


def test_hit_revokes_free_even_while_cumulative_logodds_remain_negative():
    grid=free_grid(2.)
    grid.integrate(3.,[0.,0.,0.],ray(1.))
    ij=tuple(grid.ij([1.,0.])[0])
    assert grid.L[ij]<0 and not np.isfinite(grid.last_free[ij])
    assert value(grid,[1.,0.],now=3.)==-1


def test_recentre_keeps_world_timestamps_and_new_cells_are_never_free():
    grid=free_grid(0.)
    point=np.array([1.,0.]);old=tuple(grid.ij(point)[0])
    stamp=grid.last_free[old]
    grid._centre_on([2.,1.])
    moved=tuple(grid.ij(point)[0])
    assert moved!=old and grid.last_free[moved]==stamp==0.
    assert 0<=value(grid,point,now=7.)<=35
    assert value(grid,point,now=8.1)==-1
    assert np.all(np.isneginf(grid.last_free[grid.seen==0]))
    grid._centre_on([100.,100.])
    assert np.all(np.isneginf(grid.last_free))


def test_expiring_free_cells_never_changes_occupied_or_clump_semantics():
    grid=free_grid(1.)
    # Include an isolated occupied speck and an ambiguous positive cell.
    grid.p.min_cells=3
    grid.L[10,10]=3.;grid.seen[10,10]=5
    grid.L[15,15]=.8;grid.seen[15,15]=5
    grid.L[20:23,20]=3.;grid.seen[20:23,20]=5
    legacy=grid.grid_fields();aged=grid.grid_fields(now=50.)
    assert np.array_equal(occupied_points_from_grid(*legacy),occupied_points_from_grid(*aged))
    old=legacy[-1].reshape(grid.n,grid.n).T
    new=aged[-1].reshape(grid.n,grid.n).T
    assert np.array_equal(old[grid.L>=0],new[grid.L>=0])
    assert new[10,10]==50 and new[15,15]>50
    assert np.all(new[20:23,20]>=88)
