"""Height for the altitude hold and the obstacle band: VIO height anchored slowly to a reference height,
with large reference jumps disregarded. PURE.

The reference is PX4's height (refnav) or the downward rangefinder (coursenav.loc_node). On skybeta PX4's height
reference IS that rangefinder (EKF2_HGT_REF 2), so over a window sill it reads the floor ~1 m higher, and after
the sill PX4's estimate stayed ~0.7 m off (sim P09HF27, f96ecce: the drone was driven down 0.5 m and clipped
the sill). VIO does not see the sill. So:

* VIO is the height; the reference only removes VIO's slow vertical drift, with time constant tau_s.
* A reference that suddenly disagrees with the anchored height by more than jump_m is disregarded (not
  followed, not averaged in) until it agrees again: the commanded height stays constant over a sill.
* While VIO is late (a freeze), the last height is held; the reference's own change is used only while it
  stays within jump_m of where it was at the last good sample.
Heights share one axis convention (both NED down, or both up); only differences are used.
"""
from __future__ import annotations

import math


class AnchoredHeight:
    def __init__(self, tau_s=10.0, jump_m=0.15, vio_stale_s=0.3):
        self.tau_s, self.jump_m, self.vio_stale_s = float(tau_s), float(jump_m), float(vio_stale_s)
        self.off = None          # reference - VIO, slowly corrected
        self.t = None            # time of the last anchoring step
        self.last = None         # (height out, reference) at the last good VIO sample
        self.rejected = 0        # reference samples disregarded as jumps
        self.held = 0            # outputs held through a VIO gap

    def update(self, now, ref, vio, vio_t):
        """-> height. ref: reference height (None / non-finite = none); vio: VIO height (None = none) received
        at vio_t."""
        ref_ok = ref is not None and math.isfinite(ref)
        fresh = vio is not None and vio_t is not None and math.isfinite(vio) and 0 <= now - vio_t <= self.vio_stale_s
        if not fresh:
            if self.last is None:
                return ref if ref_ok else math.nan
            out, ref0 = self.last
            self.held += 1
            if ref_ok and ref0 is not None and abs(ref - ref0) <= self.jump_m:
                return out + (ref - ref0)                # a small, consistent move: follow it
            return out                                   # a jump, or no reference: hold the height
        if ref_ok:
            meas = ref - vio
            if self.off is None:
                self.off = meas
            elif abs(meas - self.off) <= self.jump_m:
                dt = max(0.0, now - (self.t if self.t is not None else now))
                self.off += min(1.0, dt / max(self.tau_s, 1e-3)) * (meas - self.off)
            else:
                self.rejected += 1                       # a jump (sill, EKF height reset): disregard it
        self.t = now
        out = vio + (self.off if self.off is not None else 0.0)
        self.last = (out, ref if ref_ok else (self.last[1] if self.last else None))
        return out
