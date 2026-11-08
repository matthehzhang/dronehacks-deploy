"""Prior-map certification requires informative, continuous evidence."""
import math

import pytest

from coursenav.priorstate import PriorMonitor, PriorParams


def feed(monitor, start, duration, fit, beams=8):
    for i in range(round(duration * 4) + 1):
        monitor.on_fit(start + i * .25, beams, fit)
    return monitor.state


def test_no_evidence_does_not_certify_the_declared_start():
    m = PriorMonitor()
    assert m.state == 'unknown'
    assert m.as_dict()['fit'] is None and m.as_dict()['t'] is None
    for t in range(20):
        assert m.on_fit(t, 7, 1.) == 'unknown'
    assert m.last_fit is None and m.t_last is None


def test_initial_good_evidence_certifies_only_after_restore_interval():
    m = PriorMonitor()
    assert feed(m, 0, 1.75, .35) == 'unknown'
    assert m.on_fit(2., 8, .35) == 'placed'
    assert m.changed_t == 2. and m.drops == 0


def test_initial_bad_evidence_drops_only_after_drop_interval():
    m = PriorMonitor()
    assert feed(m, 0, 2.75, .29) == 'unknown'
    assert m.on_fit(3., 8, .29) == 'dropped'
    assert m.changed_t == 3. and m.drops == 1


def test_hysteresis_thresholds_and_continuous_drop_restore():
    m = PriorMonitor()
    assert feed(m, 0, 2, .8) == 'placed'
    assert feed(m, 2.25, 4, .30) == 'placed'  # equality does not drop
    assert feed(m, 6.5, 2.75, .20) == 'placed'
    assert m.on_fit(9.5, 8, .20) == 'dropped'
    assert feed(m, 9.75, 4, .349) == 'dropped'
    assert feed(m, 14, 2, .35) == 'placed'
    assert m.drops == 1


@pytest.mark.parametrize('observation', [
    (1.75, 7, .8), (1.75, None, .8), (1.75, math.nan, .8),
    (1.75, 8.5, .8), (1.75, 8, math.nan), (1.75, 8, math.inf),
    (1.75, 8, -0.1), (1.75, 8, 1.1), (math.nan, 8, .8),
    (math.inf, 8, .8), (1.5, 8, .8), (1.25, 8, .8),
])
def test_unusable_observation_interrupts_certification_without_refreshing_fit(observation):
    m = PriorMonitor()
    feed(m, 0, 1.5, .8)
    previous = (m.t_last, m.last_fit)
    assert m.on_fit(*observation) == 'unknown'
    assert (m.t_last, m.last_fit) == previous
    assert m.since is None and m.last_rejection is not None
    assert feed(m, 2, 1.75, .8) == 'unknown'
    assert m.on_fit(4., 8, .8) == 'placed'


@pytest.mark.parametrize('initial,target,fit,duration', [
    ('unknown', 'placed', .8, 2.), ('unknown', 'dropped', .1, 3.),
    ('placed', 'dropped', .1, 3.), ('dropped', 'placed', .8, 2.),
])
def test_missing_interval_cannot_satisfy_continuous_evidence(initial, target, fit, duration):
    m = PriorMonitor()
    if initial == 'placed':
        feed(m, 0, 2, .8)
    elif initial == 'dropped':
        feed(m, 0, 3, .1)
    assert m.state == initial
    m.on_fit(5., 8, fit)
    assert m.on_fit(25., 8, fit) == initial
    assert feed(m, 25.25, duration-.5, fit) == initial
    assert m.on_fit(25.+duration, 8, fit) == target


def test_hysteresis_band_resets_unknown_candidate_in_both_directions():
    m = PriorMonitor()
    feed(m, 0, 1.75, .8)
    assert m.on_fit(2, 8, .32) == 'unknown'
    assert feed(m, 2.25, 1.75, .8) == 'unknown'
    assert m.on_fit(4.25, 8, .1) == 'unknown'
    assert feed(m, 4.5, 2.5, .1) == 'unknown'
    assert m.on_fit(7.25, 8, .1) == 'dropped'


def test_sparse_observation_advances_watermark_not_accepted_fit():
    m = PriorMonitor()
    m.on_fit(0, 8, .8)
    m.on_fit(1, 0, .8)
    assert m.on_fit(.5, 8, .8) == 'unknown'
    assert m.t_last == 0 and m.last_rejection == 'out_of_order'


def test_custom_fit_gate_is_used_for_initial_certification_and_restore():
    m = PriorMonitor(PriorParams(restore_fit=.42))
    assert feed(m, 0, 3, .4) == 'unknown'
    assert feed(m, 3.25, 2, .42) == 'placed'
    assert feed(m, 5.5, 3, .1) == 'dropped'
    assert feed(m, 8.75, 3, .4) == 'dropped'
    assert feed(m, 12, 2, .42) == 'placed'


@pytest.mark.parametrize('initial,target,fit,duration', [
    ('placed', 'dropped', .1, 3.), ('dropped', 'placed', .8, 2.),
])
def test_sparse_data_retains_last_verdict_but_interrupts_pending_transition(initial, target, fit, duration):
    m = PriorMonitor()
    feed(m, 0, 3, .8 if initial == 'placed' else .1)
    assert m.state == initial
    feed(m, 4, duration-.25, fit)
    before = m.t_last
    assert m.on_fit(4+duration, 0, fit) == initial
    assert m.t_last == before
    assert m.as_dict()['has_evidence'] is True
    assert m.as_dict()['last_rejection'] == 'too_few_beams'
    start = 4+duration+.25
    assert feed(m, start, duration-.25, fit) == initial
    assert m.on_fit(start+duration, 8, fit) == target
