"""Live occupancy grid (coursenav.occgrid): filling, gating, landmarks, conflicts, rule 6.4 record."""
import math
from pathlib import Path

import numpy as np
import pytest

from kestrel_nav.coursenav.course import Grid2D
from kestrel_nav.coursenav.occgrid import LiveGrid, OccParams
from kestrel_nav.coursenav.pf import CorrectionPF
from kestrel_nav.coursenav.scan import Scan

ROOT = Path(__file__).resolve().parent.parent / "kestrel_nav"
RES = 0.05
CONFIDENT = np.diag([0.02 ** 2, 0.02 ** 2, math.radians(0.5) ** 2])


def prior(wall_x=None):
    occ = np.zeros((100, 100), bool)                       # x, y in [-2.5, 2.5)
    if wall_x is not None:
        occ[int((wall_x + 2.5) / RES), :] = True
    return Grid2D(occ, np.array([-2.5, -2.5]), RES)


def scan(bearings, ranges):
    b, r = np.asarray(bearings, float), np.asarray(ranges, float)
    return Scan(b, r, np.full(len(r), 0.05))


def cell(g, x, y):
    i, j = g.prior.ij((x, y))
    return g.L[i, j]


def test_one_beam_marks_free_space_then_the_return():
    g = LiveGrid(prior())
    assert g.integrate((0.0, 0.0, 0.0), scan([0.0], [1.0]), CONFIDENT)
    assert cell(g, 0.5, 0.0) == pytest.approx(g.p.l_free)
    assert cell(g, 1.0, 0.0) == pytest.approx(g.p.l_occ)
    assert cell(g, 1.5, 0.0) == 0.0                           # beyond the return: unknown
    assert cell(g, 0.0, 1.0) == 0.0                           # no beam there: unknown


def test_a_bearing_without_a_return_adds_nothing():
    g = LiveGrid(prior())
    g.integrate((0.0, 0.0, 0.0), Scan.empty(), CONFIDENT)
    assert not g.L.any()


def test_pose_heading_rotates_the_beams():
    g = LiveGrid(prior())
    g.integrate((0.0, 0.0, math.pi / 2), scan([0.0], [1.0]), CONFIDENT)
    assert cell(g, 0.0, 1.0) > 0 and cell(g, 1.0, 0.0) == 0.0


def test_uncertain_localizer_scans_are_gated_out():
    g = LiveGrid(prior())
    loose = np.diag([0.2 ** 2, 0.2 ** 2, math.radians(0.5) ** 2])
    assert not g.integrate((0.0, 0.0, 0.0), scan([0.0], [1.0]), loose)
    assert g.scans_gated == 1 and not g.L.any()


def test_each_cell_is_updated_once_per_scan_and_clamped():
    g = LiveGrid(prior())
    many = scan(np.linspace(-0.01, 0.01, 9), np.full(9, 1.0))  # 9 beams hitting the same cell
    g.integrate((0.0, 0.0, 0.0), many, CONFIDENT)
    assert cell(g, 1.0, 0.0) == pytest.approx(g.p.l_occ)
    for _ in range(50):
        g.integrate((0.0, 0.0, 0.0), many, CONFIDENT)
    assert cell(g, 1.0, 0.0) == pytest.approx(g.p.l_max)
    assert cell(g, 0.5, 0.0) == pytest.approx(g.p.l_min)


def wall_scan(x_wall, n=31):
    b = np.linspace(-0.5, 0.5, n)
    return scan(b, x_wall / np.cos(b))


def test_an_unmapped_wall_becomes_a_landmark_and_a_mapped_one_does_not():
    g = LiveGrid(prior(wall_x=1.0))
    for _ in range(4):
        g.integrate((0.0, 0.0, 0.0), wall_scan(1.0), CONFIDENT)     # mapped wall
        g.integrate((0.0, 0.0, math.pi), wall_scan(1.5), CONFIDENT)  # unmapped wall at x = -1.5
    lm = g.landmarks()
    assert g.occupied()[g.prior.ij((1.0, 0.0))[0], g.prior.ij((1.0, 0.0))[1]]
    assert not lm[:, :][g.prior.ij((1.0, 0.0))[0]].any()
    i, j = g.prior.ij((-1.5, 0.0))
    assert lm[i, j]
    f = g.landmark_field(force=True)
    assert f[i, j] == 0.0
    assert f[g.prior.ij((-1.0, 0.0))[0], j] == pytest.approx(0.5, abs=RES)


def test_a_single_speck_is_not_a_landmark():
    g = LiveGrid(prior())
    for _ in range(5):
        g.integrate((0.0, 0.0, 0.0), scan([0.0], [1.0]), CONFIDENT)
    assert g.occupied().sum() == 1 and not g.landmarks().any()
    assert g.landmark_field(force=True) is None


def test_seeing_through_a_mapped_wall_is_a_conflict():
    g = LiveGrid(prior(wall_x=1.0))
    for _ in range(4):
        g.integrate((0.0, 0.0, 0.0), wall_scan(2.0), CONFIDENT)
    assert g.conflicts().sum() > 10
    assert g.prior.occ.sum() == 100                          # the prior is never edited


def test_pf_field_swap_requires_the_same_grid():
    p = prior(wall_x=1.0)
    pf = CorrectionPF(p, p.distance())
    with pytest.raises(ValueError):
        pf.set_distance(np.zeros((10, 10)))
    pf.set_distance(np.zeros_like(pf.dist))


def test_record_is_labelled_and_never_loaded_by_flight_code(tmp_path):
    g = LiveGrid(prior(wall_x=1.0))
    g.integrate((0.0, 0.0, 0.0), wall_scan(1.0), CONFIDENT)
    g.save(tmp_path / "occ.npz", source="test")
    rec = LiveGrid.load_record(tmp_path / "occ.npz")
    assert "never load" in rec["meta"]["rule_6_4"]
    assert rec["logodds"].shape == (100, 100)
    flight = [*ROOT.rglob("*.py")]                                 # every module of this package
    for f in flight:
        text = f.read_text(errors="ignore")
        if f.name == "occgrid.py":
            continue
        assert "load_record" not in text and '"logodds"' not in text, f     # nothing reads a record back


# ---------------------------------------------------------------- against the real course map
def _course_map():
    from kestrel_nav.coursenav.course import load_course, resolve
    from kestrel_nav.coursenav.objmap import CourseMap
    return CourseMap.load(resolve(load_course(), "map"))


def test_see_through_covers_openings_but_not_full_walls():
    from kestrel_nav.coursenav.course import scan_grid, see_through_grid
    cm = _course_map()
    see, sg = see_through_grid(cm), scan_grid(cm)
    assert not (see & ~sg.occ).any()                                  # only ever marks OBJ surfaces
    for prefix, lo, hi in (("Net", 0.95, 1.0), ("Tube oval", 0.9, 1.0), ("M1", 0.0, 0.05), ("M2", 0.0, 0.05)):
        m = cm.band(0.3, 1.6, only_prefixes=(prefix,))
        frac = see[m].mean()
        assert lo <= frac <= hi, (prefix, frac)
    i, j = cm.xy_to_ij((-0.471, -2.272))                              # window centre: an opening
    assert see[i, j] or not sg.occ[i, j]


