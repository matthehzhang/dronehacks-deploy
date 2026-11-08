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
