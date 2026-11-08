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
    # Deployment starts 8 inches forward of the original imported mark.
    assert doc['start']['xy'] == pytest.approx([
        manifest['start']['xy'][0], manifest['start']['xy'][1] + 8 * .0254])
    assert doc['start']['yaw_deg'] == manifest['start']['yaw_deg']
    assert [s['name'] for s in doc['sequence']] == ['tunnel_entry', 'tunnel_exit', 'tunnel_clearance', 'chicane', 'pad']
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


def test_tunnel_clearance_is_a_required_stop_before_window(active):
    from coursenav.navplan import stops_and_legs, smooth_params_for
    from coursenav.navigator import Navigator
    from refnav import core

    course, planner, doc, cmap = active
    grids = course.plan_grids(cmap, doc)
    plan = planner.plan(doc, *grids, start_xy=doc['start']['xy'], chicane='window', cmap=cmap)
    route, legs, nav_legs = stops_and_legs(plan, doc['start']['xy'], np.deg2rad(doc['start']['yaw_deg']))
    assert [leg['name'] for leg in nav_legs] == [
        'tunnel_entry', 'tunnel_exit', 'tunnel_clearance', 'window_pre', 'window_post', 'pad']
    stop = nav_legs[2]
    assert stop['stop'] == pytest.approx([2.05, -1.85])
    assert stop['path'] and legs[2]['hold']
    assert 'align' not in legs[2] and 'tunnel' not in legs[2]
    assert np.asarray(stop['goals'][-1]) == pytest.approx(stop['stop'])

    mission = core.Mission(core.Params(route=route, legs=legs, track_mode='path', blend_turn_deg=30))
    mission.targets = [(0, 0)] * len(route)
    assert mission._stops_at_end(2)

    need = doc['vehicle']['radius_m'] + doc['vehicle']['margin_m']
    nav = Navigator(*grids, need, smooth_params_for(doc['navigation'], doc['refnav_params'], need),
                    gate_lines=plan.get('gate_lines'))
    # Initial planning and a replan must both end at the clearance stop, not the window.
    start = np.asarray(nav_legs[1]['stop'])
    for offset in [(0, 0), (.02, -.04)]:
        path = nav.plan(start + offset, [np.asarray(g) for g in stop['goals']])
        assert path.pts[-1] == pytest.approx(stop['stop'])
        assert path.v[-1] == pytest.approx(0)
    # Smoothing for the following leg starts at this stop rather than cutting across it.
    path = nav.plan(np.asarray(stop['stop']), [np.asarray(g) for g in nav_legs[3]['goals']])
    assert path.pts[0] == pytest.approx(stop['stop'])


def test_packaged_assets_match_source():
    packaged = ROOT / 'kestrel_nav/data'
    if not packaged.is_dir():
        return
    for name in ('course/course_v3.json', 'course/new_course_origin.obj',
                 'course/new_course_origin.mtl', 'maps/course_origin_5cm.npz'):
        assert (ROOT / name).read_bytes() == (packaged / name).read_bytes()
