"""Reject impossible VIO kinematics and require a quiet recovery interval."""
from __future__ import annotations

import math
from collections import deque


class VioHealth:
    def __init__(self, *, max_speed_mps=5.0, jump_m=1.0, window_s=10.0, max_jumps=1, quiet_s=5.0):
        self.max_speed = float(max_speed_mps)
        self.jump_m = float(jump_m)
        self.window_s = float(window_s)
        self.max_jumps = int(max_jumps)
        self.quiet_s = float(quiet_s)
        self._last = None               # (t, pos) of the last FINITE, advancing sample
        self._events = deque()          # times of insane samples
        self._last_bad_t = None
        self.reason = "no samples"
        self.n_bad = 0

    def update(self, t, pos):
        """-> (ok, reason). t in seconds (measurement time), pos a 3-tuple in any consistent frame."""
        bad = None
        finite = math.isfinite(t) and all(math.isfinite(float(c)) for c in pos)
        if not finite:
            bad = "non-finite sample"
        elif self._last is not None:
            t0, p0 = self._last
            dt = t - t0
            d = math.dist(p0, pos)
            if dt <= 0:
                bad = "measurement time did not advance" + (f" (moved {d:.2f} m)" if d > 0 else "")
            elif d > self.jump_m:
                bad = f"jump {d:.2f} m"
            elif d / dt > self.max_speed:
                bad = f"speed {d / dt:.1f} m/s"
        if finite and (self._last is None or t > self._last[0]):
            self._last = (t, tuple(float(c) for c in pos))
        tb = t if math.isfinite(t) else (self._last[0] if self._last else 0.0)
        if bad:
            self.n_bad += 1
            self._last_bad_t = tb
            self._events.append(tb)
        while self._events and tb - self._events[0] > self.window_s:
            self._events.popleft()
        if len(self._events) > self.max_jumps:
            self.reason = f"{len(self._events)} insane samples in {self.window_s:.0f} s (last: {bad or 'earlier'})"
            return False, self.reason
        if self._last_bad_t is not None and tb - self._last_bad_t < self.quiet_s:
            self.reason = f"settling after '{bad or 'insane sample'}' ({tb - self._last_bad_t:.1f}/{self.quiet_s:.0f} s quiet)"
            return False, self.reason
        self.reason = "ok"
        return True, self.reason
