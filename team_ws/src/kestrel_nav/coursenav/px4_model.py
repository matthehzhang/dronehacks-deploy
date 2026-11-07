"""A dynamic stand-in for PX4's multicopter position control in Offboard. PURE (numpy).

The offline closed loop and the refnav tests used to move the vehicle straight to the setpoint at
<= 1 m/s (a kinematic follower). That can never overshoot, so it never showed the 0.18-0.36 m leg-end
overshoot of managed run 1 (docs/RUN_2026-11-07_MANAGED_1.md). This model has the structure of PX4's
cascade, with PX4's DEFAULT gains (the sim's parameter file is not readable here, so every number is a
GUESS until fitted to a bag; see fit notes below):

  position loop   v_sp = v_ff + MPC_XY_P * (p_sp - p)              (position NaN: velocity control)
                  |v_sp_xy| <= MPC_XY_VEL_MAX (1.5 m/s, docs/SIMULATOR_INTERNALS.md section 7)
  velocity loop   a_sp = a_ff + VEL_P * (v_sp - v) + VEL_I * integral - VEL_D * dv/dt
                  |a_sp_xy| <= g * tan(tilt_max)
  airframe        the achieved acceleration follows a_sp after a pure delay and a first-order lag
                  (attitude + motor response): a <- a + (a_sp(t - delay) - a) * dt / tau
  yaw             rate-limited towards the yaw setpoint

Fit notes: with these defaults refnav's competition carrot profile (feed-forward + 0.5 m/s2 carrot
braking) overshoots its leg ends by roughly 0.1-0.3 m here, the same size as managed run 1; the slow
D/E profile does not. Fit tau / delay / gains to a bag (PX4 setpoint vs local position) before trusting
finer numbers.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

G = 9.81


@dataclass
class PX4ModelParams:
    xy_p: float = 0.95              # MPC_XY_P (PX4 default)
    xy_vel_p: float = 1.8           # MPC_XY_VEL_P_ACC
    xy_vel_i: float = 0.4           # MPC_XY_VEL_I_ACC
    xy_vel_d: float = 0.2           # MPC_XY_VEL_D_ACC
    z_p: float = 1.0                # MPC_Z_P
    z_vel_p: float = 4.0            # MPC_Z_VEL_P_ACC
    xy_vel_max: float = 1.5         # MPC_XY_VEL_MAX (sim params file, SIMULATOR_INTERNALS.md)
    z_vel_max: float = 1.0
    tilt_max_deg: float = 45.0      # MPC_TILTMAX_AIR (PX4 default)
    tau_s: float = 0.15             # GUESS: attitude + thrust response of a 1.25 kg 7 in quad
    delay_s: float = 0.05           # GUESS: estimator + control + actuation delay
    yaw_rate_dps: float = 60.0


def _clip_norm(v, vmax):
    n = float(np.hypot(v[0], v[1]))
    if n > vmax > 0:
        v = v.copy()
        v[:2] *= vmax / n
    return v


class PX4Model:
    """State in PX4 local NED: pos, vel, acc (3-vectors), yaw (rad). step() integrates one tick."""

    def __init__(self, pos=(0.0, 0.0, 0.0), yaw=0.0, params: PX4ModelParams = None):
        self.p = params or PX4ModelParams()
        self.pos = np.array(pos, float)
        self.vel = np.zeros(3)
        self.acc = np.zeros(3)
        self.yaw = float(yaw)
        self.i_xy = np.zeros(2)
        self._v_err_prev = None
        self._queue = deque()           # (t_due, a_sp)

    def reset_integrators(self):
        self.i_xy[:] = 0.0
        self._v_err_prev = None
        self._queue.clear()

    def step(self, t, dt, position=None, velocity=None, accel=None, yaw=None):
        """position/velocity/accel: 3-vectors in NED (NaN components = not commanded), or None."""
        p = self.p
        pos_sp = np.full(3, np.nan) if position is None else np.asarray(position, float)
        v_ff = np.zeros(3) if velocity is None else np.nan_to_num(np.asarray(velocity, float))
        a_ff = np.zeros(3) if accel is None else np.nan_to_num(np.asarray(accel, float))
        v_sp = v_ff.copy()
        if np.all(np.isfinite(pos_sp[:2])):
            v_sp[:2] += p.xy_p * (pos_sp[:2] - self.pos[:2])
        if math.isfinite(pos_sp[2]):
            v_sp[2] += p.z_p * (pos_sp[2] - self.pos[2])
        v_sp = _clip_norm(v_sp, p.xy_vel_max)
        v_sp[2] = max(-p.z_vel_max, min(p.z_vel_max, v_sp[2]))
        v_err = v_sp - self.vel
        d_err = np.zeros(3) if self._v_err_prev is None or dt <= 0 else (v_err - self._v_err_prev) / dt
        self._v_err_prev = v_err
        self.i_xy += v_err[:2] * dt
        a_sp = a_ff.copy()
        a_sp[:2] += p.xy_vel_p * v_err[:2] + p.xy_vel_i * self.i_xy + p.xy_vel_d * d_err[:2]
        a_sp[2] += p.z_vel_p * v_err[2]
        a_sp = _clip_norm(a_sp, G * math.tan(math.radians(p.tilt_max_deg)))
        self._queue.append((t + p.delay_s, a_sp))
        a_due = None
        while self._queue and self._queue[0][0] <= t + 1e-9:
            a_due = self._queue.popleft()[1]
        if a_due is not None:
            self._a_cmd = a_due
        a_cmd = getattr(self, "_a_cmd", np.zeros(3))
        k = min(1.0, dt / max(p.tau_s, 1e-6))
        self.acc += (a_cmd - self.acc) * k
        self.vel += self.acc * dt
        self.pos += self.vel * dt
        if yaw is not None and math.isfinite(yaw):
            err = (yaw - self.yaw + math.pi) % (2 * math.pi) - math.pi
            step = math.radians(p.yaw_rate_dps) * dt
            self.yaw += max(-step, min(step, err))
        return self.pos


class DynamicPX4:
    """PX4 as refnav sees it, with PX4Model dynamics: Offboard only after setpoints have streamed
    0.5 s, arm, Offboard tracking of refnav's Output (position + optional velocity/acceleration
    feed-forward + yaw), AUTO_LAND descends and disarms on the floor. Same interface as
    tests/test_refnav_core.FakePX4 (apply / vehicle / pos / yaw)."""

    def __init__(self, pos=(0.0, 0.0, 0.0), yaw=0.0, params: PX4ModelParams = None):
        self.m = PX4Model(pos, yaw, params)
        self.armed, self.nav, self.landed = False, 4, True
        self.stream_since = None
        self.hold = None                    # position held when a mode has no setpoint

    @property
    def pos(self):
        return self.m.pos

    @property
    def yaw(self):
        return self.m.yaw

    @property
    def vel(self):
        return self.m.vel

    def apply(self, t, out, dt):
        from refnav import core as C
        if out.stream:
            self.stream_since = t if self.stream_since is None else self.stream_since
        else:
            self.stream_since = None
        for c in out.commands:
            if c == "offboard" and self.stream_since is not None and t - self.stream_since >= 0.5:
                self.nav = C.NAV_OFFBOARD
            elif c == "arm":
                self.armed = True
            elif c == "land":
                self.nav = C.NAV_AUTO_LAND
            elif c == "disarm" and self.landed:
                self.armed = False
        if not self.armed:
            self.m.vel[:] = 0.0
            self.m.acc[:] = 0.0
            self.m.reset_integrators()
            return
        if self.nav == C.NAV_OFFBOARD and out.setpoint is not None:
            n, e, d, yaw = out.setpoint
            vel = getattr(out, "velocity", None)
            acc = getattr(out, "accel", None)
            self.m.step(t, dt, position=(n, e, d),
                        velocity=None if vel is None else (vel[0], vel[1], 0.0),
                        accel=None if acc is None else (acc[0], acc[1], 0.0), yaw=yaw)
        elif self.nav == C.NAV_AUTO_LAND:
            self.hold = self.hold if self.hold is not None else self.m.pos[:2].copy()
            self.m.step(t, dt, position=(self.hold[0], self.hold[1], math.nan), velocity=(0.0, 0.0, 0.5))
        else:
            self.hold = self.hold if self.hold is not None else self.m.pos.copy()
            self.m.step(t, dt, position=self.hold)
        if self.m.pos[2] >= 0.0 and self.m.vel[2] >= 0.0:      # on the floor (NED: down is +)
            self.m.pos[2] = 0.0
            self.m.vel[:] = 0.0
            self.m.acc[:] = 0.0
            self.landed = True
            if self.nav == C.NAV_AUTO_LAND:
                self.armed = False
        else:
            self.landed = self.m.pos[2] > -0.05

    def vehicle(self):
        from refnav import core as C
        h = self.m.yaw
        return C.Vehicle(pos=tuple(self.m.pos), heading=h, pos_ok=True,
                         q=(math.cos(h / 2), 0.0, 0.0, math.sin(h / 2)), armed=self.armed, nav_state=self.nav,
                         landed=self.landed, vel=tuple(self.m.vel))
