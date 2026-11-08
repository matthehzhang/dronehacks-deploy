"""Fresh 3-D swept-depth evidence, transient-hit retention and unknown blocking."""
from dataclasses import replace
import time

import numpy as np
import pytest

from kestrel_nav.searchnav.clearance import DepthClearanceHistory, ClearanceParams
from kestrel_nav.searchnav.grid import DepthGrid, GridParams, FREE

R = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
K = (180., 180., 159.5, 119.5)
REMOTE_CAMERA = (-2., 0., 1.)
RADIUS, HALF_HEIGHT = .294082845+.12, .4+.12
START, END = (0., 0., 1.), (0., .08, 1.06)


def frame(history, stamp, depth=6., camera=REMOTE_CAMERA):
    d = np.full((240, 320), depth, float) if np.isscalar(depth) else depth
    assert history.integrate(d, K, R, camera, stamp, stamp+.01)


def cert(history, now):
    return history.certify(START, END, RADIUS, HALF_HEIGHT, now)


def free_history():
    history = DepthClearanceHistory()
    for t in (1., 1.1, 1.2):
        frame(history, t)
    assert cert(history, 1.3).safe, history.status()
    return history


def test_requires_three_distinct_finite_supported_captures_for_whole_sweep():
    history = DepthClearanceHistory()
    for t in (1., 1.1):
        frame(history, t)
        assert not cert(history, t+.02).safe
    frame(history, 1.2)
    result = cert(history, 1.3)
    assert result.safe and result.radius_m == RADIUS and result.half_height_m == HALF_HEIGHT
    assert history.status()['unknown'] == 0


def test_body_blind_space_is_unknown_and_never_cleared_from_fixture():
    history = DepthClearanceHistory()
    for t in (1., 1.1, 1.2):
        frame(history, t, camera=START)
    result = cert(history, 1.3)
    assert not result.safe and 'unknown' in result.reason
    assert history.status()['unknown'] > 0


def test_transient_hit_immediately_blocks_until_three_later_clear_captures():
    history = free_history()
    frame(history, 1.4, depth=2.)  # new wall through the proposed vehicle volume.
    assert not cert(history, 1.42).safe
    assert history.status()['occupied'] > 0
    for t in (1.5, 1.6):
        frame(history, t)
        assert not cert(history, t+.02).safe
        assert history.status()['occupied'] > 0
    frame(history, 1.7)
    assert cert(history, 1.72).safe


def test_intermediate_raw_hit_is_kept_even_without_intermediate_query():
    history = free_history()
    frame(history, 1.4, depth=2.)
    frame(history, 1.41)
    assert not cert(history, 1.43).safe
    assert history.status()['occupied'] > 0


def test_invalid_region_does_not_refresh_or_clear_volume_and_bad_frame_stops_certification():
    history = free_history()
    frame(history, 1.4, depth=2.)
    d = np.full((240, 320), 6.)
    d[50:190, 100:220] = np.nan
    for t in (1.5, 1.6, 1.7):
        frame(history, t, depth=d)
    assert not cert(history, 1.72).safe
    assert history.status()['occupied'] > 0
    assert not history.integrate(np.full((240, 320), np.nan), K, R, REMOTE_CAMERA, 1.8, 1.81)
    assert 'valid depth' in cert(history, 1.82).reason


def test_current_freshness_reset_envelope_and_lookahead_are_independent_guards():
    history = free_history()
    assert not cert(history, 2.).safe
    assert not history.certify(START, END, .2, HALF_HEIGHT, 1.3).safe
    assert not history.certify(START, (1., 0., 1.), RADIUS, HALF_HEIGHT, 1.3).safe
    history.reset()
    assert not history.frames and not history.cache and not cert(history, 1.3).safe
    with pytest.raises(ValueError):
        ClearanceParams(local_valid_fraction=.4)


def test_history_spans_yaw_scan_is_bounded_and_replay_rebuilds_only_supported_height():
    history = DepthClearanceHistory()
    for t in np.arange(1., 16.01, 1/15):
        frame(history, float(t))
    assert history.frames[-1]['stamp']-history.frames[0]['stamp'] > 14.
    assert history.status()['stored_bytes'] < 640*19200*4
    grid = DepthGrid(GridParams(extent_m=6.))
    grid.reset((0., 0.), 1.1)
    replayed = history.replay_into(grid, 16.)
    assert 3 < replayed <= 65
    assert np.any(grid.collapsed() == FREE)
    # Expired evidence cannot keep a sweep free even with a new camera looking
    # elsewhere; no free bubble is created around the vehicle's new pose.
    for t in (37., 37.1, 37.2):
        frame(history, t, camera=START)
    assert history.frames[0]['stamp'] >= 37.-20.
    assert not cert(history, 37.3).safe


def test_repeat_query_cache_rechecks_new_obstacles_without_replaying_every_capture():
    history = free_history()
    first = cert(history, 1.3)
    assert first.safe and history.cache
    assert cert(history, 1.31).safe
    frame(history, 1.4, depth=2.)
    assert not cert(history, 1.41).safe


def test_queued_replay_keeps_intermediate_hit_frames_and_four_hz_free_projections():
    history = DepthClearanceHistory()
    for t in (1., 1.08, 1.16, 1.24, 1.32, 1.40):
        frame(history, t, depth=2. if t == 1.16 else 6.)
    frames = list(history.replay_frames(1.42))
    assert [f['stamp'] for f in frames] == sorted(f['stamp'] for f in frames)
    assert len(frames) == 6
    transient = next(f for f in frames if f['stamp'] == 1.16)
    assert not transient['full'] and np.isclose(np.nanmedian(transient['depth']), 2.)
    full = [f['stamp'] for f in frames if f['full']]
    assert np.all(np.diff(full) >= .25)
    grid = DepthGrid(GridParams(extent_m=6.))
    grid.reset((0., 0.), 1.)
    assert history.replay_into(grid, 1.42) == len(full)
    assert np.any(grid.collapsed() == 100)  # two raw later-free hits do not erase a wall.
