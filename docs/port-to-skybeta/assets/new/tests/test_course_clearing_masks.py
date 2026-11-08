"""Clearing must preserve geometry that a depth ray cannot disprove."""
from pathlib import Path
import numpy as np
import pytest
from scipy import ndimage
from coursenav.course import enclosure_mask, keepout_mask, load_course, plan_grids, protected_mask, resolve, see_through_grid
from coursenav.objmap import CourseMap
ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('course', ['course_new_origin.json', 'course_v3.json'])
def test_real_course_protection_preserves_original_planning(course):
    doc = load_course(ROOT/'course'/course)
    cmap = CourseMap.load(resolve(doc, 'map'))
    before = cmap.occupied.copy()
    protected = protected_mask(cmap, doc)
    see = see_through_grid(cmap)
    assert see.any() and protected[see].all()
    for prefixes in [('Net',), ('Tube',), ('C2a',)]:
        cells = cmap.band(.3, 1.6, only_prefixes=prefixes) & see
        assert cells.any(), (course, prefixes)
        assert protected[cells].all()
    obstacle, blocked, dist = plan_grids(cmap, doc, surface=False)
    original = dist < doc['vehicle']['radius_m'] + doc['vehicle']['margin_m']
    for ko in doc.get('keepout', []):
        cells = cmap.band(ko['zlo'], ko['zhi'], only_prefixes=ko['groups'])
        inflated = ndimage.distance_transform_edt(~cells)*cmap.res < ko['radius_m']
        original |= inflated
        assert protected[cells | inflated].all()
    assert np.array_equal(blocked.occ, original)
    assert np.array_equal(cmap.occupied, before)
    enc = doc.get('enclosure')
    if enc:
        xy = obstacle.xy(np.stack(np.meshgrid(np.arange(protected.shape[0]), np.arange(protected.shape[1]), indexing='ij'), -1))
        outside = np.any((xy < enc['lo']) | (xy > enc['hi']), axis=-1)
        assert outside.any() and protected[outside].all()
    assert (obstacle.occ & ~protected).any()

def test_zero_radius_keepout_still_protects_actual_surface():
    occ=np.zeros((7,7,4),bool);labels=np.zeros(occ.shape,np.int16)
    occ[3,3,:]=True;labels[3,3,:]=1
    cmap=CourseMap(occ,np.zeros(3),.5,labels,['Fixed fixture'])
    doc={'keepout':[{'groups':['Fixed'],'zlo':0.,'zhi':1.6,'radius_m':0.}]}
    assert not keepout_mask(cmap,doc).any()
    assert protected_mask(cmap,doc)[3,3]

def test_enclosure_thin_surface_and_net_protected_but_solid_wall_clearable():
    occ=np.zeros((8,8,8),bool);labels=np.zeros(occ.shape,np.int16)
    occ[2,2,:]=True;labels[2,2,:]=1
    occ[3,3,2]=True;labels[3,3,2]=2
    occ[4,4,:]=True;labels[4,4,:]=3
    cmap=CourseMap(occ,np.zeros(3),.25,labels,['Net wall','Tube roof','Solid wall'])
    mask=protected_mask(cmap,{'enclosure':{'lo':[.5,.5],'hi':[1.5,1.5]}})
    assert mask[2,2] and mask[3,3] and mask[0,0]
    assert not mask[4,4]


def test_boundary_crossing_cell_is_fixed_even_when_its_centre_is_inside():
    occ=np.zeros((4,4,2),bool)
    cmap=CourseMap(occ,np.zeros(3),1.,np.zeros(occ.shape,np.int16),[])
    mask=enclosure_mask(cmap,{'enclosure':{'lo':[.2,.2],'hi':[3.8,3.8]}})
    assert mask[0,1] and mask[3,1] and mask[1,0] and mask[1,3]
    assert not mask[1:3,1:3].any()
    assert not occ.any()  # no occupied net voxel needed for boundary protection
