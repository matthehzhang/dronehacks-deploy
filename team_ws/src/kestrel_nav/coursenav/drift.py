"""The odometry -> course correction as a filter state (loosely coupled map-based localization). PURE.

VIO (cuVSLAM -> EKF2) is locally excellent and drifts slowly; the localizer (coursenav.pf) matches depth
against the course map and says where the body is in the course frame, with a covariance. Rather than
rate-limit that answer by a fixed speed (CorrectionLimiter), keep the correction itself as the state of
a small EKF, as map-based VIO localization does (the odometry-to-map transform estimated online,
updated by map measurements; Zhang et al. 2022, maplab):

  state   delta = (dx, dy, dth): course body pose = (nominal.xy + (dx, dy), nominal.th + dth), where
          "nominal" is where the VIO/PX4 frame puts the body (fixture transform o odometry).
  predict on every odometry increment dp (nominal frame) and dt:
          dxy += (R(dth) - I) dp           a heading error turns new motion into position error
          P   = F P F' + Q,  Q = diag(q_xy^2 d + q_xy_t^2 dt, ..., q_th^2 d + q_th_t^2 dt)
  update  with the localizer's body pose: z = pf_body - nominal_body, R = its covariance (floored);
          rejected when the Mahalanobis distance exceeds the 99 % gate (chi2, 3 dof): a wrong lock or a
          see-through scan cannot pull the correction; after `reset_after` consecutive rejections the
          covariance is re-inflated so a persistent, consistent disagreement is eventually accepted.

The gain follows the evidence: along a corridor's axis the localizer's covariance is large, so the
correction there barely moves (what project_correction did by hand); across it, it follows the drift
as fast as the drift model allows. Course <-> nominal conversion of any nearby point uses the current
state anchored at the body, so a whole path converts consistently every tick (no target-only shifts).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CHI2_3DOF_99 = 11.345


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _rot(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])


@dataclass
class DriftParams:
    q_xy_per_m: float = 0.05        # position drift growth, 1 sigma per sqrt(m) travelled (VIO ~2 % / m)
    q_xy_per_s: float = 0.005       # ... and per sqrt(s) (hover creep)
    q_th_per_m: float = math.radians(0.6)   # heading drift per sqrt(m) (flight D: ~0.3 deg/m)
    q_th_per_s: float = math.radians(0.05)
    r_floor_xy: float = 0.03        # localizer covariance floor (its particles can be over-confident)
    r_floor_th: float = math.radians(1.0)
    gate: float = CHI2_3DOF_99
    reset_after: int = 15           # consecutive rejections before the covariance is re-inflated


class DriftEKF:
    def __init__(self, p: DriftParams = None):
        self.p = p or DriftParams()
        self.x = None               # (dx, dy, dth)
        self.P = None
        self.last_nominal = None    # body pose in the nominal frame at the last predict
        self.rejected = 0
        self.consecutive_rejects = 0
        self.accepted = 0
        self.resets = 0

    @property
    def ready(self):
        return self.x is not None

    def init(self, delta, cov, nominal_body):
        self.x = np.array(delta, float)
        self.x[2] = _wrap(self.x[2])
        self.P = np.array(cov, float) + np.diag([self.p.r_floor_xy ** 2] * 2 + [self.p.r_floor_th ** 2])
        self.last_nominal = np.array(nominal_body, float)

    def predict(self, nominal_body, dt):
        """Propagate with the body's motion in the nominal frame since the last call."""
        nb = np.array(nominal_body, float)
        if self.x is None:
            self.last_nominal = nb
            return
        dp = nb[:2] - self.last_nominal[:2]
        d = float(np.hypot(*dp))
        th = self.x[2]
        R = _rot(th)
        self.x[:2] += (R - np.eye(2)) @ dp
        F = np.eye(3)
        dR = np.array([[-math.sin(th), -math.cos(th)], [math.cos(th), -math.sin(th)]])
        F[:2, 2] = dR @ dp
        p = self.p
        dt = max(dt, 0.0)
        q_xy = p.q_xy_per_m ** 2 * d + p.q_xy_per_s ** 2 * dt
        q_th = p.q_th_per_m ** 2 * d + p.q_th_per_s ** 2 * dt
        self.P = F @ self.P @ F.T + np.diag([q_xy, q_xy, q_th])
        self.last_nominal = nb

    def update(self, pf_body, pf_cov, nominal_body):
        """Localizer measurement: its course body pose and covariance, at the current nominal body pose.
        -> True if accepted."""
        z = np.array([pf_body[0] - nominal_body[0], pf_body[1] - nominal_body[1],
                      _wrap(pf_body[2] - nominal_body[2])], float)
        Rm = np.array(pf_cov, float)[:3, :3].copy()
        Rm[0, 0] = max(Rm[0, 0], self.p.r_floor_xy ** 2)
        Rm[1, 1] = max(Rm[1, 1], self.p.r_floor_xy ** 2)
        Rm[2, 2] = max(Rm[2, 2], self.p.r_floor_th ** 2)
        if self.x is None:
            self.init(z, Rm, nominal_body)
            return True
        y = z - self.x
        y[2] = _wrap(y[2])
        S = self.P + Rm
        try:
            d2 = float(y @ np.linalg.solve(S, y))
        except np.linalg.LinAlgError:
            return False
        if d2 > self.p.gate:
            self.rejected += 1
            self.consecutive_rejects += 1
            if self.consecutive_rejects >= self.p.reset_after:
                self.P = self.P + Rm + np.outer(y, y)       # make room for a consistent disagreement
                self.consecutive_rejects = 0
                self.resets += 1
            return False
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = _wrap(self.x[2])
        I_K = np.eye(3) - K
        self.P = I_K @ self.P @ I_K.T + K @ Rm @ K.T          # Joseph form: stays symmetric positive
        self.accepted += 1
        self.consecutive_rejects = 0
        return True

    # ---------------------------------------------------------------- frame conversion near the body
    def body_course(self, nominal_body):
        nb = np.asarray(nominal_body, float)
        return np.array([nb[0] + self.x[0], nb[1] + self.x[1], _wrap(nb[2] + self.x[2])])

    def to_course(self, pts_nominal, nominal_body):
        """Nominal-frame points near the body -> course frame (rotation about the body)."""
        nb = np.asarray(nominal_body, float)
        bc = self.body_course(nb)
        q = np.atleast_2d(np.asarray(pts_nominal, float)) - nb[:2]
        return q @ _rot(self.x[2]).T + bc[:2]

    def to_nominal(self, pts_course, nominal_body):
        """Course-frame points -> nominal frame (inverse of to_course)."""
        nb = np.asarray(nominal_body, float)
        bc = self.body_course(nb)
        q = np.atleast_2d(np.asarray(pts_course, float)) - bc[:2]
        return q @ _rot(-self.x[2]).T + nb[:2]

    @staticmethod
    def from_state(delta, nominal_body):
        """A converter from a published state (another process's EKF): to_nominal / to_course only."""
        f = DriftEKF()
        f.x = np.array(delta, float)
        f.P = np.eye(3) * 1e-6
        f.last_nominal = np.array(nominal_body, float)
        return f

    def sigma_xy(self):
        return float(math.sqrt(max(self.P[0, 0], self.P[1, 1]))) if self.P is not None else float("inf")
