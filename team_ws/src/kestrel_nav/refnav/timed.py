"""Timed trajectory along a navigator path: the reference is a function of TIME, not of the aircraft's pose.

The reactive path follower (core.Mission._go_path) recomputes its setpoint every tick from the latest pose, so
the whole pose -> setpoint -> PX4 chain sits inside its loop. Here the motion between two stops is planned once
as s(t), v(t), a(t) along the path and streamed by a clock: PX4 tracks position + velocity + acceleration
feed-forward with its own estimator at its own rate, and the pose only adjusts how fast the clock runs (live
caps, tube, falling behind) and when to re-plan.

TimedTrajectory: from (s0, v0) to rest at s_end under the navigator's speed profile (its caps from clearance,
curvature, the tube model and the stop approach), forward/backward passes with accel / brake, then the speed is
smoothed in TIME with a moving average of jerk_window_s: distance is preserved exactly (the average of v(t)
integrates to the same s), the end is at rest, the speed never goes negative, and acceleration becomes
continuous with |jerk| <= max|a| / jerk_window_s (an S-curve, the role Ruckig plays in an online generator).
"""
import math

import numpy as np


class TimedTrajectory:
    def __init__(self, s_grid, v_cap, s0, v0, s_end, accel, brake, jerk_window_s=0.25, ds=0.01, dt=0.02):
        s_grid = np.asarray(s_grid, float)
        v_cap = np.asarray(v_cap, float)
        self.s0, self.s_end = float(s0), float(s_end)
        L = max(self.s_end - self.s0, 0.0)
        if L < 1e-3:
            self.t = np.array([0.0, 1e-3])
            self.s = np.array([self.s0, self.s0])
            self.v = np.zeros(2)
            self.a = np.zeros(2)
            self.T = 0.0
            return
        n = max(2, int(math.ceil(L / ds)) + 1)
        S = np.linspace(self.s0, self.s_end, n)
        h = S[1] - S[0]
        cap = np.interp(S, s_grid, v_cap)
        # the profile is 0 at stops and at the path's start: from here on, only its caps ahead matter (the
        # start is ramped from v0 by accel below)
        cap[0] = max(cap[0], float(v0))
        cap = np.maximum(cap, 0.0)
        cap[-1] = 0.0
        v = cap.copy()
        v[0] = min(max(float(v0), 0.0), cap[0])
        for i in range(1, n):                                   # forward: what acceleration allows
            v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2.0 * accel * h))
        for i in range(n - 2, -1, -1):                          # backward: what braking allows
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * brake * h))
        vavg = np.maximum(0.5 * (v[1:] + v[:-1]), 1e-3)
        t_s = np.concatenate([[0.0], np.cumsum(h / vavg)])      # time at each arc-length sample
        # resample v on a uniform time grid, then smooth in time (moving average, zero after the end)
        T0 = float(t_s[-1])
        tg = np.arange(0.0, T0 + dt, dt)
        vt = np.interp(tg, t_s, v)
        k = max(1, int(round(jerk_window_s / dt)))
        if k > 1:
            pad_end = np.zeros(k)
            vt_p = np.concatenate([vt, pad_end])
            ker = np.ones(k) / k
            # causal-centred: keep the start speed (v0) at t = 0 by padding the front with v0
            front = np.full(k // 2, vt[0])
            vt_s = np.convolve(np.concatenate([front, vt_p]), ker, mode="valid")[:len(vt_p)]
            vt = vt_s
            tg = np.arange(len(vt)) * dt
        # integrate s(t) (trapezoid) and rescale so the end lands exactly on s_end
        st = np.concatenate([[0.0], np.cumsum(0.5 * (vt[1:] + vt[:-1]) * dt)])
        if st[-1] > 1e-9:
            scale = L / st[-1]
            st *= scale
            vt = vt * scale
        # trim the trailing rest
        moving = np.nonzero(vt > 1e-4)[0]
        last = min(len(vt) - 1, (moving[-1] + 2) if len(moving) else 1)
        self.t, self.s, self.v = tg[:last + 1], self.s0 + st[:last + 1], vt[:last + 1]
        self.s[-1] = self.s_end
        self.v[-1] = 0.0
        self.a = np.gradient(self.v, self.t) if len(self.t) > 2 else np.zeros(len(self.t))
        self.T = float(self.t[-1])

    def at(self, tau):
        """(s, v, a) of the reference at trajectory time tau (clamped to [0, T])."""
        tau = min(max(float(tau), 0.0), self.T)
        return (float(np.interp(tau, self.t, self.s)), float(np.interp(tau, self.t, self.v)),
                float(np.interp(tau, self.t, self.a)))

    def time_at_s(self, s):
        """Trajectory time at which the reference passes arc length s (for re-syncing)."""
        return float(np.interp(s, self.s, self.t))
