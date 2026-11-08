"""Depth apertures cannot turn unknown pixels or a 2-D route into 3-D motion."""
from dataclasses import replace
import math

import numpy as np
import pytest

from kestrel_nav.searchnav.opening import (DepthOpeningDetector, OpeningTracker,
    OpeningParams, OpeningCenteringController, ClearanceCertificate)


R = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
K = (180., 180., 159.5, 119.5)
T = np.array([0., 0., 1.])


def scene(center_z=1.25, width=1.2, height=1.4, center_y=-.2, invalid_hole=False):
    v, u = np.mgrid[:240, :320]
    # World coordinates where each calibrated pixel meets the near frame x=2.
    y, z = -(u-K[2])/K[0]*2., T[2]-(v-K[3])/K[1]*2.
    hole = (np.abs(y-center_y) < width/2) & (np.abs(z-center_z) < height/2)
    depth = np.full((240, 320), 2.)
    depth[hole] = np.nan if invalid_hole else 4.
    return depth, hole


def measured(**kw):
    depth, _ = scene(**kw)
    detector = DepthOpeningDetector()
    candidates = detector.detect(depth, K, R, T, 1.)
    assert len(candidates) == 1, detector.diagnostics
    return candidates[0]


def test_detects_world_aperture_center_and_different_window_heights():
    low, high = measured(center_z=.8), measured(center_z=1.4)
    assert np.allclose(low.center_world, (2., -.2, .8), atol=.03)
    assert np.allclose(high.center_world, (2., -.2, 1.4), atol=.03)
    assert low.body_fits and high.body_fits
    assert 1.1 < low.width_m < 1.2 and 1.3 < low.height_m < 1.4
    assert min(low.side_support) >= .65 and low.behind_fraction >= .6


def test_frame_height_is_world_up_with_pitched_calibrated_camera():
    angle = math.radians(10.)
    c, s = math.cos(angle), math.sin(angle)
    pitched = R @ np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
    v, u = np.mgrid[:240, :320]
    rays = np.stack([(u-K[2])/K[0], (v-K[3])/K[1], np.ones(u.shape)], axis=-1)@pitched.T
    near = 2./rays[..., 0]
    xyz = T+rays*near[..., None]
    hole = (np.abs(xyz[..., 1]+.2) < .6) & (np.abs(xyz[..., 2]-1.25) < .7)
    depth = np.where(hole, 4./rays[..., 0], near)
    detector = DepthOpeningDetector()
    found = detector.detect(depth, K, pitched, T, 1.)
    assert len(found) == 1, detector.diagnostics
    assert np.allclose(found[0].center_world, (2., -.2, 1.25), atol=.035)
    assert found[0].height_m > 1.3 and found[0].body_fits


def test_invalid_no_return_is_not_an_opening_and_sparse_whole_frame_rejected():
    d, _ = scene(invalid_hole=True)
    detector = DepthOpeningDetector()
    assert detector.detect(d, K, R, T, 1.) == []
    d[:] = np.nan
    d[100:110, 100:110] = 4.
    assert detector.detect(d, K, R, T, 2.) == []
    assert detector.diagnostics['full_valid_fraction'] < .3


def test_cropped_frame_or_insufficient_space_behind_is_not_aperture():
    detector = DepthOpeningDetector()
    d, _ = scene(center_y=1.5)
    assert detector.detect(d, K, R, T, 1.) == []
    d, hole = scene()
    d[hole] = 2.3  # A recess does not prove enough depth for the vehicle.
    assert detector.detect(d, K, R, T, 2.) == []


def test_small_measured_opening_is_reported_but_never_confirmed_for_traversal():
    small = measured(width=.65)
    assert not small.body_fits
    tracker = OpeningTracker()
    for t in (1., 1.2, 1.4):
        assert tracker.update([replace(small, stamp=t)], t) is None
    assert 'vehicle envelope' in tracker.reason
    with pytest.raises(ValueError):
        OpeningParams(radius_m=.2)


def test_tracking_needs_distinct_stable_captures_and_rejects_ambiguity_or_staleness():
    opening = measured()
    tracker = OpeningTracker()
    assert tracker.update([opening], 1.) is None
    assert tracker.update([opening], 1.) is None
    assert tracker.update([replace(opening, stamp=1.2)], 1.2) is None
    assert tracker.update([replace(opening, stamp=1.41)], 1.41) is not None
    assert tracker.current(1.8) is not None
    assert tracker.current(2.) is None
    other = replace(opening, center_world=(2., -.7, 1.25), stamp=1.5)
    assert tracker.update([replace(opening, stamp=1.5), other], 1.5) is None
    assert tracker.reason == 'ambiguous apertures'


def certificate(start, end, radius, half_height, now):
    return ClearanceCertificate(True, now, start, end, radius, half_height)


def test_centering_requests_height_correction_but_without_3d_certificate_holds():
    opening = measured(center_z=1.4)
    controller = OpeningCenteringController()
    command = controller.command(opening, (0., 0., 1.), 0., (0., 0., 0.), 1.1, authority=True)
    assert command.phase == 'center'
    assert command.requested_velocity[2] > 0
    assert command.requested_velocity[1] < 0
    assert command.authorized_velocity == (0., 0., 0., 0.)
    assert command.requires_3d_clearance and 'unavailable' in command.reason


def test_stop_face_center_then_explicit_slow_traverse_each_with_matching_certificate():
    opening = measured()
    controller = OpeningCenteringController()
    stop = controller.command(opening, (0., 0., 1.), 0., (.1, 0., 0.), 1.1, authority=True,
                              certify_sweep=certificate)
    assert stop.phase == 'stop' and stop.authorized_velocity == (0., 0., 0., 0.)
    align = controller.command(opening, (0., 0., 1.), math.pi/2, (0., 0., 0.), 1.1,
                               authority=True, certify_sweep=certificate)
    assert align.phase == 'align' and align.authorized_velocity[3] < 0
    center = controller.command(opening, (0., 0., 1.), 0., (0., 0., 0.), 1.1, authority=True,
                                certify_sweep=certificate)
    assert center.phase == 'center'
    assert center.authorized_velocity == center.requested_velocity
    pos = (0., opening.center_world[1], opening.center_world[2])
    wait = controller.command(opening, pos, 0., (0., 0., 0.), 1.1, authority=True,
                              certify_sweep=certificate)
    assert wait.phase == 'centered' and wait.authorized_velocity == (0., 0., 0., 0.)
    move = controller.command(opening, pos, 0., (0., 0., 0.), 1.1, authority=True,
                              certify_sweep=certificate, traverse=True)
    assert move.phase == 'traverse' and np.allclose(move.authorized_velocity, (.2, 0., 0., 0.))


def test_centerer_continues_certified_motion_without_stop_go_oscillation_and_reset_stops():
    opening = measured()
    controller = OpeningCenteringController()
    first = controller.command(opening, (0., 0., 1.), 0., (0., 0., 0.), 1.1, authority=True,
                               certify_sweep=certificate)
    assert first.phase == 'center' and np.linalg.norm(first.authorized_velocity[:3]) > 0
    continuing = controller.command(opening, (0., -.01, 1.01), 0., (0., -.1, .1), 1.11,
                                    authority=True, certify_sweep=certificate)
    assert continuing.phase == 'center' and np.linalg.norm(continuing.authorized_velocity[:3]) > 0
    controller.reset()
    stopped = controller.command(opening, (0., -.01, 1.01), 0., (0., -.1, .1), 1.12,
                                 authority=True, certify_sweep=certificate)
    assert stopped.phase == 'stop' and stopped.authorized_velocity == (0., 0., 0., 0.)


@pytest.mark.parametrize('bad', ('old', 'small', 'mismatch', 'blocked', 'two_dimensional', 'malformed', 'infinite'))
def test_stale_small_mismatched_or_2d_clearance_never_authorizes_altitude(bad):
    opening = measured()
    def checker(start, end, radius, half_height, now):
        cert = certificate(start, end, radius, half_height, now)
        return {'old': replace(cert, stamp=now-1), 'small': replace(cert, radius_m=.1),
                'mismatch': replace(cert, end_world=(0., 0., 0.)),
                'blocked': replace(cert, safe=False, reason='unknown overhead'),
                'two_dimensional': True, 'malformed': replace(cert, start_world=(0.,)),
                'infinite': replace(cert, radius_m=math.inf)}[bad]
    command = OpeningCenteringController().command(opening, (0., 0., 1.), 0., (0., 0., 0.),
                                                   1.1, authority=True, certify_sweep=checker)
    assert command.requested_velocity[2] > 0
    assert command.authorized_velocity == (0., 0., 0., 0.)


def test_stale_opening_and_authority_loss_revoke_motion():
    opening = measured()
    controller = OpeningCenteringController()
    for kwargs in ({'now': 2., 'authority': True}, {'now': 1.1, 'authority': False}):
        cmd = controller.command(opening, (0., 0., 1.), 0., (0., 0., 0.),
                                 certify_sweep=certificate, **kwargs)
        assert cmd.phase == 'hold' and cmd.authorized_velocity == (0., 0., 0., 0.)
