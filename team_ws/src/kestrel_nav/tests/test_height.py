"""refnav.height.AnchoredHeight: constant height over a window sill; large reference jumps disregarded."""
import math

from refnav.height import AnchoredHeight


def run(h, ref_of_t, vio_of_t, t0=0., t1=60., dt=.05, vio_gaps=()):
    out = []
    t, vio_t = t0, None
    while t <= t1:
        v = vio_of_t(t)
        if not any(a <= t < b for a, b in vio_gaps):
            vio_t, vio = t, v
        out.append((t, h.update(t, ref_of_t(t), vio, vio_t)))
        t += dt
    return out


def test_sill_dip_then_persistent_px4_offset_keeps_height_constant():
    """Sim P09HF27 (f96ecce): flying at 1.27 m, PX4/rangefinder reads 0.27 m over the sill for ~1.5 s, then
    PX4's height stays 0.74 m too high for the rest of the flight. The anchored height must not move."""
    h = AnchoredHeight(tau_s=10., jump_m=.15)
    true = 1.27
    def ref(t):
        if t < 20: return true
        if t < 21.5: return .27
        return true + .74
    out = run(h, ref, lambda t: true, t1=60.)
    assert max(abs(z - true) for _, z in out) < .01
    assert h.rejected > 0


def test_slow_reference_drift_is_still_followed():
    """VIO's vertical drift (slow) is what the anchor is for: a 0.1 m drift over 60 s is tracked."""
    h = AnchoredHeight(tau_s=10., jump_m=.15)
    out = run(h, lambda t: 1.3, lambda t: 1.3 + .1 * t / 60., t1=90.)
    assert abs(out[-1][1] - 1.3) < .03


def test_vio_freeze_during_a_jump_holds_the_last_height():
    h = AnchoredHeight(tau_s=10., jump_m=.15)
    def ref(t): return 1.3 if t < 10.1 else .3            # jump inside the freeze
    out = run(h, ref, lambda t: 1.3, t1=12., vio_gaps=[(10., 10.5)])
    assert all(abs(z - 1.3) < .01 for t, z in out)
    assert h.held > 0


def test_vio_freeze_follows_small_consistent_reference_motion():
    h = AnchoredHeight(tau_s=10., jump_m=.15, vio_stale_s=.06)   # the gap rule from the first late sample
    def ref(t): return 1.3 + (.05 * (t - 10.) / .4 if 10. <= t <= 10.4 else (0.05 if t > 10.4 else 0.))
    def vio(t): return ref(t)
    out = run(h, ref, vio, t1=10.4, vio_gaps=[(10., 10.45)])
    assert abs(out[-1][1] - 1.35) < .01


def test_no_vio_yet_uses_the_reference():
    h = AnchoredHeight()
    assert h.update(0., 1.0, None, None) == 1.0
    assert math.isnan(h.update(0., None, None, None))
