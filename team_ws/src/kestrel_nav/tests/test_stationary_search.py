"""Recovery must observe before moving, and retain airborne supervision."""
import math
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refnav import core as C
from coursenav.live_pose import arena_grid, vio_pose
from coursenav.navigator import Navigator


def searching():
    m = C.Mission(C.Params(track_mode='path', search_enabled=True, health_quiet_s=0., yaw_accel_dps2=60.))
    m.sp = [1., 2., -.7, 0.]
    m.targets = [(4., 2.)]
    m.leg_start, m.leg_dir, m.leg_len = (1., 2.), (1., 0.), 3.
    v = C.Vehicle(pos=(1., 2., -.7), heading=0., pos_ok=True, armed=True, nav_state=C.NAV_OFFBOARD, landed=False)
    m._start_search(0., C.Output('go'), v, 'blocked')
    return m, v


def test_full_sweep_holds_position_caps_yaw_and_never_resumes_old_path():
    m, v = searching()
    m.path = {'pid':None, 't':0., 'leg':0}
    for k in range(300):
        t = k*.1
        m.depth_t = t
        m.on_health(t, True)
        before = m.sp[3]
        m._search(t, .1, C.Output('search'), v)
        assert m.sp[:3] == [1., 2., -.7]
        assert abs(C._wrap(m.sp[3]-before)) <= math.radians(20)*.1+1e-9
        v.heading = m.sp[3]
        m.path['t'] = t
    assert m.search['index'] == 4
    assert m.phase == 'search'
    # Only a different plan can leave search, through a turn and the normal corridor check.
    m.path = None
    m.on_path(30., 0, [[1.,2.],[4.,2.]], [0.,0.], [], pid=42)
    m.depth_t = 30.
    m.on_health(30., True)
    m._search(30., .1, C.Output('search'), v)
    assert m.phase == 'turn'


def test_search_pauses_yaw_with_stale_depth_and_times_out():
    m, v = searching()
    initial = m.sp.copy()
    m.on_health(1., True)
    m._search(1., .1, C.Output('search'), v)
    assert m.sp == initial
    m._search(46., .1, C.Output('search'), v)
    assert m.phase == 'land'


def test_search_yields_to_px4_and_still_lands_on_bad_health():
    m, v = searching()
    m.offboard_seen = True
    v.nav_state = 3
    out = m.step(.1, v)
    assert m.phase == 'done' and not out.stream
    m, v = searching()
    m.on_health(0., False)
    m.depth_t = 5.
    m.on_health(5., False)
    m.step(5., v)
    assert m.phase == 'land'


def test_missing_path_starts_bounded_search_for_timed_follower():
    m, v = searching()
    m.phase, m.search = 'go', None
    m.p.path_timed = True
    for t in (0., 3.):
        m._go_timed(t, .1, C.Output('go'), v)
    assert m.phase == 'search'
    m.search_attempts[0] = m.p.search_max_per_leg
    m._start_search(4., C.Output('search'), v, 'blocked')
    assert m.phase == 'land'


def test_vio_pose_retains_covariance_without_a_wall_fix():
    cov = np.diag([.0004,.0009,.001,.001,.001,.0001])
    mean, out = vio_pose([1.,2.,math.pi/2], [2.,0.,.1], cov)
    assert mean == pytest.approx([1.,4.,math.pi/2+.1])
    assert np.diag(out) == pytest.approx([.0009,.0004,.0001])
    with pytest.raises(ValueError):
        vio_pose([0.,0.,0.], [0.,0.,0.], np.zeros((6,6)))


def test_equal_count_moving_obstacle_invalidates_map_cache():
    g = arena_grid(fence=True)
    class Live:
        prior = g
        cells = np.zeros(g.occ.shape, bool)
        def landmarks(self):return self.cells
    live = Live()
    nav = Navigator(g, g, g.distance(), .414, live=live)
    live.cells[40,40] = True
    nav.fuse()
    live.cells[40,40], live.cells[45,45] = False, True
    assert nav.map_changed()
    assert nav.fuse()[0].occ[45,45] and not nav.fuse()[0].occ[40,40]
