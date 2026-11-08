"""Only explicit fresh free evidence can cross the OccupancyGrid reader boundary."""
import numpy as np
import pytest
from coursenav.localgrid import LocalGrid, LocalParams, free_points_from_grid
from coursenav.scan import Scan

def test_reader_layout_threshold_and_invalid_cells():
    data=[-1,0,23,24,np.nan,np.inf,-np.inf,101,-2,22.5,100,50]
    pts=free_points_from_grid([10,20],.5,4,3,data)
    np.testing.assert_allclose(pts,[[10.75,20.25],[11.25,20.25]])
    assert free_points_from_grid([0,0],1,2,1,[-1,50]).shape==(0,2)

@pytest.mark.parametrize('kwargs',[
 {'origin':[np.nan,0]}, {'origin':[0,0,0]}, {'res':0}, {'res':np.inf},
 {'width':0}, {'height':1.5}, {'data':[0,0]}, {'free_prob':-1},
 {'free_prob':50}, {'free_prob':np.nan},
])
def test_reader_rejects_invalid_metadata(kwargs):
    args=dict(origin=[0,0],res=.1,width=1,height=1,data=[0]);args.update(kwargs)
    with pytest.raises(ValueError):free_points_from_grid(**args)

def test_expiry_not_refreshed_by_publication_or_empty_scan():
    grid=LocalGrid(LocalParams(decay_s=0,min_cells=1))
    scan=Scan(np.array([0.]),np.array([2.]),np.array([.01]))
    for _ in range(3):grid.integrate(0.,(0.,0.,0.),scan)
    assert len(free_points_from_grid(*grid.grid_fields(now=8.)))>0
    assert len(free_points_from_grid(*grid.grid_fields(now=8.001)))==0
    grid.integrate(9.,(0.,0.,0.),None)
    assert len(free_points_from_grid(*grid.grid_fields(now=9.)))==0

def test_free_probability_is_weighted_confidence_not_net_hit_count():
    grid=LocalGrid(LocalParams(decay_s=0,min_cells=1))
    grid.integrate(0.,(0.,0.,0.),Scan(np.array([0.]),np.array([1.]),np.array([.01])))
    for t in range(1,5):grid.integrate(float(t),(0.,0.,0.),Scan(np.array([0.]),np.array([2.]),np.array([.01])))
    ij=tuple(grid.ij([1.,0.])[0])
    assert np.isclose(grid.L[ij],-.75)
    pts=free_points_from_grid(*grid.grid_fields(now=4.))
    assert not np.any(np.all(np.isclose(pts,grid.cell_centres(ij)),axis=1))
