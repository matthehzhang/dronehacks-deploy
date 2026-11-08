"""refnav blocked recovery on navigator path legs (bag 20261107-201744: 18 blocks on leg 1, the drone hovered
without turning until the pilot took over).

* recovery judges the corridor with the same along-path check as go, and only when every confirm frame is clear;
* a path re-planned while blocked is followed: turn to it, check it, go;
* blocked_max_per_leg lands instead of looping blocked -> recovered -> blocked for ever."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coursenav.px4_model import DynamicPX4                     # noqa: E402
from refnav import core as C                                   # noqa: E402
from test_refnav_core import ROOM, fly                         # noqa: E402
from test_refnav_profile import PROFILE                        # noqa: E402

WALL = (0, 2.5, -0.6, 0.6)          # a 1.2 m wide obstacle across the path at n = 2.5 (not on any prior map)


def dense(corners, step=0.05):
    """A polyline through `corners` (NED n, e) sampled every `step`; -> (pts, indices of interior corners)."""
    pts, stops = [np.asarray(corners[0], float)], []
    for a, b in zip(corners[:-1], corners[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        k = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        pts += [a + (b - a) * i / k for i in range(1, k + 1)]
        stops.append(len(pts) - 1)
    return np.array(pts), stops[:-1]


def speeds(pts, stops=(), v=0.3, a=0.4):
    """A rest-to-rest profile between stops (the navigator's profile comes to 0 at every stop corner)."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    knots = [0.0] + [s[i] for i in stops] + [s[-1]]
    d = np.min([np.abs(s - k) for k in knots], axis=0)
    return np.minimum(v, np.sqrt(2 * a * d))


def params(**kw):
    return C.Params(**dict(PROFILE, route=[(4.0, 0.0)], legs=[{"name": "p"}], track_mode="path", **kw))


def fly_path(prm, walls_fn, on_tick=None, seconds=120.0):
    """fly() with path legs: walls_fn(t) -> walls; on_tick(t, m) publishes the path."""
    # fly() renders static walls; re-render with the walls of this instant
    from test_refnav_core import render_depth

    def dyn(t, img):
        return render_depth(px4_ref[0].pos, px4_ref[0].yaw, list(walls_fn(t)) + ROOM)

    px4 = DynamicPX4()      # path mode moves on velocity feed-forward: FakePX4 tracks positions only
    px4_ref = [px4]
    return fly(prm, depth_fn=dyn, on_tick=on_tick, seconds=seconds, px4=px4)


STRAIGHT, _ = dense([(0.0, 0.0), (4.0, 0.0)])


def publish_straight(t, m):
    if m.targets is not None:
        m.on_path(t, 0, STRAIGHT, speeds(STRAIGHT), [], pid=1)


def test_a_flickering_obstacle_lands_after_the_per_leg_limit_instead_of_looping():
    """Present 3 s, gone 1 s, again and again: each gap 'recovers' and the next approach blocks again.
    Without a limit that loops until something else ends the flight; with blocked_max_per_leg 2 the third
    block lands."""
    def walls(t):
        return [WALL] if (t % 4.0) < 3.0 else []
    m, _, log = fly_path(params(blocked_max_per_leg=2), walls, publish_straight, seconds=150)
    assert m.result.startswith("landed: route blocked") and "blocked 3 times" in m.result, m.result
    assert m.counts["obstacle_stops"] == 3

    m0, _, _ = fly_path(params(), walls, publish_straight, seconds=150)
    assert m0.counts["obstacle_stops"] > 3                   # the old behaviour: block after block


def test_a_persistent_obstacle_on_a_path_leg_still_holds_backs_off_and_lands():
    m, px4, log = fly_path(params(), lambda t: [WALL], publish_straight)
    assert m.result.startswith("landed: route blocked") and "recheck" in m.result, m.result
    assert m.counts["recoveries"] == 0 and m.counts["obstacle_stops"] == 1
    assert px4.pos[0] < 2.5 - 0.5


def test_recovery_needs_every_confirm_frame_clear():
    """One clear frame in the middle of a persistent obstacle is not a recovery (the old rule took the
    LARGEST clearance of the confirm frames, so a single dropout resumed straight into the wall)."""
    blink = []

    def walls(t):
        # one frame-length dropout every 1.5 s once blocked
        return [] if any(abs(t - b) < 0.07 for b in blink) else [WALL]

    def tick(t, m):
        publish_straight(t, m)
        if m.phase in ("blocked", "recheck") and (not blink or t - blink[-1] > 1.5):
            blink.append(t + 0.2)
    m, _, _ = fly_path(params(), walls, tick)
    assert m.counts["recoveries"] == 0 and m.counts["obstacle_stops"] == 1, (m.counts, m.result)
    assert m.result.startswith("landed: route blocked")


def test_a_path_replanned_while_blocked_is_turned_to_and_flown():
    """The navigator publishes a detour (new pid) while refnav holds in front of the obstacle: refnav turns
    to the new path, checks it and flies it to the end instead of waiting for the old corridor to clear."""
    detour = {}

    def tick(t, m):
        if m.targets is None:
            return
        if m.phase == "blocked" and "pts" not in detour:
            n, e = m.last_v.pos[0], m.last_v.pos[1]
            pts, stops = dense([(n, e), (n, 1.3), (4.0, 1.3)])
            detour.update(pts=pts, stops=stops, t=t)
        if "pts" in detour:
            m.on_path(t, 0, detour["pts"], speeds(detour["pts"], detour["stops"]), detour["stops"], pid=2)
        else:
            publish_straight(t, m)
    m, px4, log = fly_path(params(), lambda t: [WALL], tick)
    assert m.result == "complete", (m.result, m.why)
    assert m.counts["replan_resumes"] == 1 and m.counts["obstacle_stops"] == 1
    phases = [p for t, p, *_ in log if t >= detour["t"]]
    assert phases.index("turn") < phases.index("check") < phases.index("go")
    assert "backoff" not in phases
    assert abs(px4.pos[0] - 4.0) < 0.2 and abs(px4.pos[1] - 1.3) < 0.2


def test_a_replan_into_the_same_obstacle_counts_as_another_block():
    """A 'detour' that still faces the obstacle is checked like any leg: it blocks again (and counts)."""
    replanned = []

    def tick(t, m):
        if m.targets is None:
            return
        if m.phase == "blocked" and not replanned:
            replanned.append(t)
        pid = 2 if replanned else 1
        m.on_path(t, 0, STRAIGHT, speeds(STRAIGHT), [], pid=pid)
    m, _, _ = fly_path(params(blocked_max_per_leg=1), lambda t: [WALL], tick)
    assert m.counts["replan_resumes"] == 1
    assert m.result.startswith("landed: route blocked") and "blocked 2 times" in m.result, m.result
