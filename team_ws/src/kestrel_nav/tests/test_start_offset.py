"""start_forward_in: the drone was placed some inches forward of the launch mark on the guideline."""
import math

import pytest

from coursenav import course


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv('KESTREL_START_FORWARD_IN', raising=False)
    monkeypatch.delenv('KESTREL_START_FORWARD_M', raising=False)
    return monkeypatch


def test_no_offset_keeps_the_declared_mark(clean_env):
    doc = course.load_course()
    assert doc['start']['xy'] == [-2.515, 0.914] and 'forward_offset_m' not in doc['start']


def test_inches_move_the_start_along_its_heading(clean_env):
    clean_env.setenv('KESTREL_START_FORWARD_IN', '27.75')
    st = course.load_course()['start']
    assert st['declared_xy'] == [-2.515, 0.914]
    assert math.isclose(st['forward_offset_m'], 27.75 * .0254)
    assert math.isclose(st['xy'][0], -2.515, abs_tol=1e-9)            # yaw 90: straight along +y
    assert math.isclose(st['xy'][1], 0.914 + .70485, abs_tol=1e-9)
    assert st['yaw_deg'] == 90.0                                        # still facing the same way


def test_zero_and_conflicting_units(clean_env):
    clean_env.setenv('KESTREL_START_FORWARD_IN', '0.0')
    assert course.load_course()['start']['xy'] == [-2.515, 0.914]
    clean_env.setenv('KESTREL_START_FORWARD_M', '0.5')
    with pytest.raises(ValueError):
        course.load_course()


def test_flat_course_config_1p3_m_cruise_band_015_no_height_features(clean_env):
    """Whole course at 1.3 m with a +/-0.15 m obstacle band: a 1.073..1.727 m window (centre 1.4 m) and the
    0.914..1.626 m window both fit the band at cruise; no height changes in flight."""
    import re
    from pathlib import Path
    doc = course.load_course()
    assert doc['cruise_z_m'] == 1.3
    yaml = (Path(course.PROJECT) / 'onboard/config/common.yaml').read_text()
    band = float(re.search(r'band_half_m:\s*([0-9.]+)', yaml).group(1))
    assert band == .15
    for sill, header in ((1.0728, 1.7272), (.9144, 1.6256)):
        assert sill < 1.3 - band and 1.3 + band < header
