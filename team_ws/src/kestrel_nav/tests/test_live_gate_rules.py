"""Gate constraints at the hardware planner/follower boundary; no ROS or vehicle."""
import numpy as np
import pytest
from kestrel_nav.coursenav.course import PROJECT as COURSE_ROOT
LEGACY_COURSE = COURSE_ROOT / "course/course_v3_legacy.json"


from kestrel_nav.coursenav import planner
from kestrel_nav.coursenav.course import Grid2D
from kestrel_nav.livenav.follower import Follower
from kestrel_nav.livenav.mission import LiveMission, MissionParams
from kestrel_nav.livenav.replanner import Goal, Replanner
from kestrel_nav.coursenav.scan import Scan


class OpenGrid:
    def __init__(self):
        self.blocked = Grid2D(np.zeros((80, 80), bool), np.zeros(2), .05)
        self.blocked_hard = Grid2D(self.blocked.occ.copy(), np.zeros(2), .05)
        self.clearance = np.full((80, 80), 10.)

    def carved(self, xy):
        return self.blocked

    def nearest_free(self, xy, **kwargs):
        return np.asarray(xy, float)


def make_replanner(goals=None, full=True):
    grid = OpenGrid()
    line = [[2, 0], [2, 4]] if full else [[2, 1], [2, 3]]
    rules = {'opening': {'line': line, 'normal': [1, 0]}}
    goals = goals or [Goal(dict(name='pre', xy=[1, 2], mode='gate'))]
    return Replanner(grid, goals, gate_rules=rules)


def test_live_mission_receives_actual_selected_gate_geometry():
    mission = LiveMission(course_file=LEGACY_COURSE, params=MissionParams(use_pf=False))
    assert set(mission.replanner.gate_rules) == {'window'}
    assert mission.replanner.gate_rules['window']['line'] == mission.plan['gate_lines'][0]
    pre = next(g.xy for g in mission.goals if g.name == 'window_pre')
    post = next(g.xy for g in mission.goals if g.name == 'window_post')
    mission.replanner.k = next(i for i,g in enumerate(mission.goals) if g.name == 'window_pre')
    assert mission.replanner.chord_blocked(pre, post, True)
    mission.replanner.k += 1
    assert not mission.replanner.chord_blocked(pre, post, True)
    assert mission.replanner.chord_blocked(post, pre, True)


@pytest.mark.parametrize('soft_failure', [False, True])
def test_wrong_side_cannot_reach_pre_even_through_hard_fallback(soft_failure):
    rp = make_replanner()
    if soft_failure:
        rp.g.blocked.occ[50, :] = True
    assert rp.plan([3, 2], 0) is None and rp.blocked


def test_shortcut_and_retained_path_keep_gate_closed():
    rp = make_replanner(full=False)
    path = rp.plan([3, 2], 0)
    assert path is not None and len(path) > 2
    assert planner.gate_crossings(path, rp.gate_rules['opening']['line'], [1, 0]) == []
    assert not rp.path_blocked()
    rp.path = np.array([[3, 2], [1, 2]])
    assert rp.path_blocked()
    assert rp.chord_blocked([3, 2], [1, 2], True)


def test_nearest_free_connector_cannot_jump_gate(monkeypatch):
    rp = make_replanner()
    monkeypatch.setattr(rp.g, 'nearest_free', lambda xy, **kw: np.array([1., 2.]))
    assert rp.plan([3, 2], 0) is None and rp.blocked


def test_continued_via_extension_keeps_gate_closed():
    goals = [Goal(dict(name='via1', xy=[3, 2])), Goal(dict(name='via2', xy=[1, 2]))]
    rp = make_replanner(goals, full=False)
    path = rp.plan([3, 1.5], 0)
    assert path is not None and rp.continues
    assert planner.gate_crossings(path, rp.gate_rules['opening']['line'], [1, 0]) == []


def test_only_selected_forward_crossing_is_open_and_replan_never_backtracks():
    goals = [Goal(dict(name='pre', xy=[1, 2], mode='gate')),
             Goal(dict(name='post', xy=[3, 2], mode='gate', crosses='opening'))]
    rp = make_replanner(goals)
    rp.k = 1
    assert rp.plan([1, 2], 0) is not None
    assert not rp.path_blocked()
    # On the post side, obstruction ahead must not send us pre-side and back.
    rp.g.blocked_hard.occ[60, 30] = True
    assert rp.plan([3, 1], 1) is None and rp.blocked


def test_minimum_lookahead_cannot_command_through_a_blocked_chord():
    f = Follower()
    f.v_prev[:] = [.3, 0]
    command = f.step(1, (0, 0, 0), np.array([[0, 0], [2, 0]]), None, None, 1,
                     chord_blocked=lambda a,b,hard: True, gate_blocked=lambda a,b: True)
    assert command.vx == command.vy == command.speed == 0
    assert command.limiter == 'gate blocked'
    assert np.array_equal(f.v_prev, [0, 0])


def test_plane_origin_cannot_backtrack_to_pre_when_direct_crossing_blocked():
    goals = [Goal(dict(name='pre', xy=[1, 2], mode='gate')),
             Goal(dict(name='post', xy=[3, 2], mode='gate', crosses='opening'))]
    rp = make_replanner(goals)
    rp.k = 1
    rp.g.blocked_hard.occ[48:53, 37:40] = True
    assert rp.plan([2, 1.8], 0) is None
    assert rp.gate_blocked([2, 1.8], [1, 2])


@pytest.mark.parametrize('crossing_active', [True, False])
def test_hold_backoff_cannot_cross_gate_backwards(crossing_active):
    rp = make_replanner([Goal(dict(name='post', xy=[3, 2], mode='gate',
                                   crosses='opening' if crossing_active else None))])
    f = Follower()
    near = Scan(np.array([0.]), np.array([.3]), np.array([.03]))
    cmd = f.step(1, (2.1, 2, 0), None, rp.goal, near, 1, hold=True,
                 gate_blocked=rp.gate_blocked)
    assert cmd.vx == cmd.vy == 0 and cmd.limiter == 'gate blocked'


def test_gate_guard_does_not_change_escape_from_physical_inflation():
    rp = make_replanner([Goal(dict(name='goal', xy=[1, 3]))])
    rp.g.blocked_hard.occ[20, 40] = True
    # Existing nearest-free recovery may start within a physical inflation cell;
    # the newly added final gate guard must not reinterpret that as a gate crossing.
    assert not rp.gate_blocked([1.025, 2.025], [1.025, 2.5])
