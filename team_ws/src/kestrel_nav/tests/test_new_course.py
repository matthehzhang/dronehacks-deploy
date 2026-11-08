"""The active navigation/localization prior is the downloaded entrance revision."""
import copy
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NAMESPACES = ['coursenav']
if (ROOT / 'kestrel_nav/data').is_dir():
    NAMESPACES.append('kestrel_nav.coursenav')


@pytest.fixture(scope='module', params=NAMESPACES)
def active(request):
    course = importlib.import_module(request.param + '.course')
    maps = importlib.import_module(request.param + '.objmap')
    planner = importlib.import_module(request.param + '.planner')
    doc = course.load_course()
    cmap = maps.CourseMap.load(course.resolve(doc, 'map'))
    return course, planner, doc, cmap


def test_prior_provenance_and_goal_resolution(active):
    course, _, doc, cmap = active
    obj = course.resolve(doc, 'obj')
    assert obj.name == 'new_course_origin.obj'
    assert course.resolve(doc, 'map').name == 'course_origin_5cm.npz'
    manifest = json.loads((obj.parent / 'new_course_manifest.json').read_text())
    assert hashlib.sha256(obj.read_bytes()).hexdigest() == cmap.meta['obj_sha256'] == manifest['obj_sha256']
    assert cmap.meta['scenario'] == 'topview_entrance'
    assert cmap.meta['official_published'] is False
    assert doc['start']['xy'] == manifest['start']['xy']
    assert doc['start']['yaw_deg'] == manifest['start']['yaw_deg']
    assert [s['name'] for s in doc['sequence']] == ['tunnel_entry', 'tunnel_exit', 'chicane', 'pad']
    assert 'maze_exit' not in doc['derived']
    assert doc['vehicle']['radius_m'] + doc['vehicle']['margin_m'] == pytest.approx(.4140828452868319)


def test_planner_and_localizer_see_revised_m1(active):
    course, _, doc, cmap = active
    # M1 now meets the north/west boundary; the old interior upright is gone.
    scan = course.scan_grid(cmap)
    north = cmap.band(.3, 1.6, only_prefixes=('M1',))
    xy = scan.xy(np.argwhere(north))
    assert xy[:, 1].min() > 2.85
    # Net labels own the overlapping boundary voxels; check the OBJ endpoint too.
    assert xy[:, 0].min() < -2.95
    maps = importlib.import_module(course.__package__ + '.objmap')
    vertices, faces = maps.load_obj(course.resolve(doc, 'obj'))['M1 maze divider']
    vertices = vertices[np.unique(faces)]
    assert vertices[:, 0].min() == pytest.approx(-3.048, abs=1e-5)
    assert vertices[:, 1].max() == pytest.approx(3.048, abs=1e-5)
    assert not north[tuple(scan.ij((-1.5, 1.5)))]
    assert np.all(scan.occ[north])


def test_new_route_preserves_full_clearance(active):
    course, planner, doc, cmap = active
    obst, blocked, clearance = course.plan_grids(cmap, doc)
    assert course.surface_clearance(doc, cmap, .55, 1.05) is not None
    free = blocked.xy(np.argwhere(~blocked.occ))
    start = free[np.argmin(np.linalg.norm(free - doc['start']['xy'], axis=1))]
    assert np.linalg.norm(start - doc['start']['xy']) <= .3
    plan = planner.plan(doc, obst, blocked, clearance, start_xy=start, chicane='window', cmap=cmap)
    assert plan['min_clearance_m'] >= plan['required_clearance_m']
    names = [w['name'] for w in plan['waypoints']]
    assert 'maze_exit' not in names
    assert names.index('tunnel_exit') == names.index('tunnel_entry') + 1
    assert names[-1] == 'pad'
    for a, b in zip(plan['waypoints'], plan['waypoints'][1:]):
        assert planner.segment_clearance(obst, clearance, a['xy'], b['xy']) >= plan['required_clearance_m'] - 1e-9


def test_surface_field_rejects_mismatched_or_modified_map(active):
    course, _, doc, cmap = active
    changed = copy.copy(cmap)
    changed.meta = dict(cmap.meta, obj_sha256='wrong')
    assert course.surface_clearance(doc, changed, .55, 1.05) is None
    changed.meta = dict(cmap.meta, world='modified geometry')
    assert course.surface_clearance(doc, changed, .55, 1.05) is None


def test_packaged_assets_match_source():
    packaged = ROOT / 'kestrel_nav/data'
    if not packaged.is_dir():
        return
    for name in ('course/course_v3.json', 'course/new_course_origin.obj',
                 'course/new_course_origin.mtl', 'maps/course_origin_5cm.npz'):
        assert (ROOT / name).read_bytes() == (packaged / name).read_bytes()
