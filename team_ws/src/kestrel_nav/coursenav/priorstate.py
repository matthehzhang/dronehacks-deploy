"""Evidence-based prior-map status. PURE: no ROS.

A declared launch pose is a hypothesis, not evidence that the map is placed.
Start unknown; enough informative consecutive good fits certify placement, while
consecutive bad fits drop it. Previously placed/dropped states retain hysteresis.
Sparse, malformed, repeated, or out-of-order observations interrupt a pending
transition. Missing observations cannot count as continuous evidence either.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class PriorParams:
    drop_fit: float = 0.30
    drop_s: float = 3.0
    restore_fit: float = 0.35
    restore_s: float = 2.0
    min_beams: int = 8
    max_fit_gap_s: float = 0.5       # 4 Hz adaptive localizer; no evidence across longer gaps


class PriorMonitor:
    def __init__(self, p: PriorParams = None):
        self.p = p or PriorParams()
        self.state = "unknown"
        self.since = None
        self._candidate = None
        self.last_fit = None
        self.t_last = None         # last accepted informative observation, not a heartbeat
        self._latest_t = None      # timestamp watermark including unusable observations
        self.last_rejection = None
        self.drops = 0
        self.changed_t = None

    def _interrupt(self, reason):
        self.since, self._candidate = None, None
        self.last_rejection = reason
        return self.state

    def on_fit(self, t, n_beams, frac):
        """Consume a fit at capture time t; return unknown, placed, or dropped.

        Only finite, ordered observations with enough informative beams count.
        A rejected observation cannot refresh the last accepted fit or timestamp.
        """
        try:
            t = float(t)
        except (TypeError, ValueError, OverflowError):
            return self._interrupt("invalid_time")
        if not math.isfinite(t) or t < 0:
            return self._interrupt("invalid_time")
        if self._latest_t is not None and t <= self._latest_t:
            return self._interrupt("out_of_order")
        self._latest_t = t
        try:
            n, frac = float(n_beams), float(frac)
        except (TypeError, ValueError, OverflowError):
            return self._interrupt("invalid_fit")
        if not math.isfinite(n) or not n.is_integer() or n < 0 or not math.isfinite(frac) or not 0 <= frac <= 1:
            return self._interrupt("invalid_fit")
        if n < self.p.min_beams:
            return self._interrupt("too_few_beams")
        if self.t_last is not None and t - self.t_last > self.p.max_fit_gap_s + 1e-9:
            self.since, self._candidate = None, None
        self.last_fit, self.t_last, self.last_rejection = frac, t, None

        candidate = None
        if self.state != "dropped" and frac < self.p.drop_fit:
            candidate = "dropped"
        elif self.state != "placed" and frac >= self.p.restore_fit:
            candidate = "placed"
        if candidate is None:
            self.since, self._candidate = None, None
            return self.state
        if candidate != self._candidate:
            self.since, self._candidate = t, candidate
        duration = self.p.drop_s if candidate == "dropped" else self.p.restore_s
        if t - self.since >= duration:
            self.state, self.changed_t = candidate, t
            self.since, self._candidate = None, None
            if candidate == "dropped":
                self.drops += 1
        return self.state

    def as_dict(self):
        return {"state": self.state, "fit": None if self.last_fit is None else round(self.last_fit, 3),
                "t": self.t_last, "has_evidence": self.t_last is not None,
                "drops": self.drops, "changed_t": self.changed_t,
                "drop_fit": self.p.drop_fit, "restore_fit": self.p.restore_fit,
                "max_fit_gap_s": self.p.max_fit_gap_s, "last_rejection": self.last_rejection}
