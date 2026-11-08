"""Is the prior map describing what the camera sees? The kill switch for the prior. PURE: no ROS.

The localizer's scan fit (coursenav.pf.scan_fit: the share of informative beams the best particle puts
within fit_tol_m of a mapped surface) IS the agreement measure WAG's prior grid uses (share of shared cells
that disagree). Measured from the launch mark on course v3: the view rendered from the map fits 100 %, a
0.2-0.3 m or 10 deg placement error ~ 47-50 %, depth 10 % short 55 %, random ranges 23 % (p90 33 %). The
real drone in bag 20261107-201744 sat at 10-20 % from the first scan on the ground: chance, i.e. the map
described nothing in view.

  placed   the prior is trusted (start: the declared start pose, until the scans say otherwise)
  dropped  fit below drop_fit for drop_s continuously (chance level: the map does not describe the view)
  placed   again after fit >= restore_fit (the localizer's own gate) for restore_s continuously

Scans with too few informative beams (n < min_beams: a view of nothing but the net) say nothing and change
nothing. Consumers: refnav (prior_required: no arming / land while dropped), the status page, the docs.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PriorParams:
    drop_fit: float = 0.30           # chance level from the launch mark is 0.23 (p90 0.33)
    drop_s: float = 3.0
    restore_fit: float = 0.35        # the localizer's own fit gate (pf min_fit_frac)
    restore_s: float = 2.0
    min_beams: int = 8


class PriorMonitor:
    def __init__(self, p: PriorParams = None):
        self.p = p or PriorParams()
        self.state = "placed"
        self.since = None            # when the current run of low (placed) or good (dropped) fits started
        self.last_fit = None
        self.t_last = None
        self.drops = 0
        self.changed_t = None

    def on_fit(self, t, n_beams, frac):
        """One scan's fit (n informative beams, fraction explained) at time t. -> the state."""
        self.t_last = t
        if n_beams is None or n_beams < self.p.min_beams:
            return self.state
        self.last_fit = float(frac)
        if self.state == "placed":
            if frac < self.p.drop_fit:
                self.since = t if self.since is None else self.since
                if t - self.since >= self.p.drop_s:
                    self.state, self.since, self.changed_t = "dropped", None, t
                    self.drops += 1
            else:
                self.since = None
        else:
            if frac >= self.p.restore_fit:
                self.since = t if self.since is None else self.since
                if t - self.since >= self.p.restore_s:
                    self.state, self.since, self.changed_t = "placed", None, t
            else:
                self.since = None
        return self.state

    def as_dict(self):
        return {"state": self.state, "fit": None if self.last_fit is None else round(self.last_fit, 3),
                "t": self.t_last, "drops": self.drops, "changed_t": self.changed_t,
                "drop_fit": self.p.drop_fit, "restore_fit": self.p.restore_fit}
