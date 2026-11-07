"""Venue landing pad: blue rounded-square mat (colour frames), fusion and landing footprint. Offline only."""
import math
import os

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from kestrel_nav.livenav import square_pad as sq
from kestrel_nav.livenav.square_padsim import DRONE_HW, DRONE_K, R_FLU_OPT, pitch_rot, render


@pytest.mark.parametrize('pad_xy, yaw', [((2.4, -0.3), 0.0), ((2.6, 0.4), 0.2), ((2.0, 0.15), 0.35)])
@pytest.mark.parametrize('pitch_err_deg', [0.0, 3.0])
def test_bearing_is_accurate_even_with_a_wrong_mount_pitch(pad_xy, yaw, pitch_err_deg):
    img = render(DRONE_K, DRONE_HW, 0.83, pad_xy, pad_yaw=yaw)
    d = sq.detect(img, DRONE_K, pitch_rot(math.radians(pitch_err_deg)) @ R_FLU_OPT, np.zeros(3), 0.83)
    assert d is not None and 'centre_xy' in d
    p = np.asarray(pad_xy)
    u = p / np.linalg.norm(p)
    lateral = float((d['centre_xy'] - p) @ np.array([-u[1], u[0]]))
    assert abs(lateral) <= 0.02                               # SIGMA_LATERAL_M
    if d['floor_from'] == 'square':                           # pitch-immune range
        assert abs(float((d['centre_xy'] - p) @ u)) <= 0.10


def test_grey_image_is_not_detected():
    img = render(DRONE_K, DRONE_HW, 0.83, (2.4, 0.0))
    grey = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    assert sq.detect(grey, DRONE_K, R_FLU_OPT, np.zeros(3), 0.83) is None


def test_wrong_size_square_is_rejected():
    img = render(DRONE_K, DRONE_HW, 0.83, (2.6, 0.0), side=1.2)
    d = sq.detect(img, DRONE_K, R_FLU_OPT, np.zeros(3), 0.83)
    assert d is None or not d.get('size_ok') or d.get('partial')
    assert sq.fix_from_detection(d, (0.116, 0.037)) is None


def test_fusion_intersects_bearings_from_two_viewpoints():
    truth = np.array([1.0, 2.0])
    f = sq.PadFuser(truth + [0.3, -0.2], prior_sigma_m=0.4)
    for cam in (np.array([3.0, 2.0]), np.array([1.0, 4.5])):  # 90 deg apart
        u = (truth - cam) / np.linalg.norm(truth - cam)
        t = np.array([-u[1], u[0]])
        cov = 0.25 ** 2 * np.outer(u, u) + 0.02 ** 2 * np.outer(t, t)
        f.add(truth + 0.2 * u, cov)                            # range badly off, bearing exact
    assert np.linalg.norm(f.xy - truth) < 0.03


def test_footprint_rule():
    ok, m = sq.footprint_inside(0.035, 0.0, 0.0, (0.0, 0.0), 0.64)   # footprint centre on the pad centre
    assert ok and m == pytest.approx((0.64 - 0.498) / 2, abs=1e-3)
    assert not sq.footprint_inside(0.035, 0.08, 0.0, (0.0, 0.0), 0.64)[0]


PHOTOS = [os.path.expanduser(f'~/Downloads/IMG_959{k}.jpg') for k in (3, 4)]


@pytest.mark.skipif(not all(os.path.exists(p) for p in PHOTOS), reason='venue photos not on this machine')
@pytest.mark.parametrize('path', PHOTOS)
@pytest.mark.parametrize('gain', [0.4, 1.0, 1.3])
def test_venue_photos(path, gain):
    img = np.clip(cv2.imread(path).astype(float) * gain, 0, 255).astype(np.uint8)
    d = sq.detect(img)
    assert d is not None and not d['partial'] and d['solidity'] > 0.95
