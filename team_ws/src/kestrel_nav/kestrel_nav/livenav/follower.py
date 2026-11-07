"""Carrot (pure-pursuit) velocity follower for a quadrotor on a 2D path. PURE (numpy).

Why velocity, not waypoint legs: the managed run of 2026-11-07 overshot every leg end by 0.2-0.36 m
and then lost VIO while turning in place beside a barrier. A carrot follower never commands a stop
at a corner; it slows into it, keeps moving through it and lets yaw catch up.

Outputs a course-frame horizontal velocity, a yaw-rate and the speed it decided on. The caller
converts to the odom frame and adds the vertical axis.

Speed schedule (the minimum of):
  * cruise (or gate speed on gate/tunnel legs),
  * braking to the goal:        v <= sqrt(2 a d_goal) + v_floor,
  * corners within look-ahead:  v <= cruise * (1 - 0.8 * turn/90deg),
  * map clearance at the body:  v ramps from v_floor at clear_slow to cruise at clear_full,
  * the depth scan straight ahead in the motion corridor: v ramps from 0 at stop_m to cruise at
    stop_m + reactive_ramp_m (the last-resort guard; depth is blind under ~0.4 m so stop_m > 0.4),
  * heading error > 45 deg: v <= max(v_floor, cruise * cos(err)) so the camera is not far from
    the direction of travel for long.
Acceleration is rate-limited. Yaw tracks the velocity direction (gate legs: the gate normal).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..coursenav.scan import Scan


@dataclass
class FollowParams:
    v_cruise: float = 0.4
    v_gate: float = 0.25
    v_floor: float = 0.08
    a_max: float = 0.5            # m/s^2 commanded accel/decel
    lookahead_m: float = 0.55
    corner_look_m: float = 1.0
    clear_slow_m: float = 0.30    # at/below this map clearance: creep (planned paths keep >= 0.40)
    clear_full_m: float = 0.50    # at/above this: full speed (maze corridors are ~1.0 m wide)
    lookahead_min_m: float = 0.25 # the carrot is pulled back until the chord to it is free
    stop_m: float = 0.55          # scan return closer than this straight ahead -> stop
    reactive_ramp_m: float = 0.7
    corridor_half_m: float = 0.36 # prop tip 0.28 + 0.08
    yaw_rate_max: float = math.radians(70)
    yaw_kp: float = 2.0
    yaw_near_surface_m: float = 0.42
    yaw_rate_near: float = math.radians(25)
    hold_backoff_mps: float = 0.12
    lag_s: float = 0.3            # velocity response lag compensated by steering from the predicted position


@dataclass
class Command:
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    speed: float = 0.0
    yaw_target: float = float("nan")
    limiter: str = ""              # what bounded the speed (for logs)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def path_progress(path, xy):
    """Index of the segment the point projects onto, the arc length to that projection and the
    total length."""
    seg_len = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    best, bi, bt = np.inf, 0, 0.0
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        ab = b - a
        L2 = float(ab @ ab)
        t = 0.0 if L2 == 0 else float(np.clip((xy - a) @ ab / L2, 0.0, 1.0))
        d = float(np.linalg.norm(xy - (a + t * ab)))
        if d < best:
            best, bi, bt = d, i, t
    return bi, cum[bi] + bt * seg_len[bi], cum[-1], cum, seg_len


def point_at(path, cum, seg_len, s):
    s = float(np.clip(s, 0.0, cum[-1]))
    i = int(np.searchsorted(cum, s, side="right") - 1)
    i = min(max(i, 0), len(path) - 2)
    t = 0.0 if seg_len[i] == 0 else (s - cum[i]) / seg_len[i]
    return path[i] + t * (path[i + 1] - path[i])


def path_obstacle(scan: Scan, pose, path, cum, seg_len, s_here, half_width_m, look_m):
    """Arc distance (from the vehicle, along the path) to the nearest scan return that lies within
    half_width_m of the upcoming path, looking look_m ahead. inf if the corridor is clear."""
    if scan is None or len(scan) == 0:
        return math.inf
    x, y, yaw = pose
    ang = yaw + scan.bearing
    ex = x + scan.range * np.cos(ang)
    ey = y + scan.range * np.sin(ang)
    best = math.inf
    for i in range(len(path) - 1):
        if cum[i + 1] < s_here or cum[i] > s_here + look_m or seg_len[i] < 1e-6:
            continue
        a, b = path[i], path[i + 1]
        ab = (b - a) / seg_len[i]
        t = (ex - a[0]) * ab[0] + (ey - a[1]) * ab[1]
        t = np.clip(t, 0.0, seg_len[i])
        px, py = a[0] + t * ab[0], a[1] + t * ab[1]
        d = np.hypot(ex - px, ey - py)
        s_obs = cum[i] + t - s_here
        sel = (d <= half_width_m) & (s_obs > -0.05) & (s_obs <= look_m)
        if sel.any():
            best = min(best, float(s_obs[sel].min()))
    return max(best, 0.0)


def scan_ahead(scan: Scan, direction_body_rad, half_width_m, rmax=4.0):
    """Nearest scan return inside a corridor of half-width around a body bearing. inf if none."""
    if scan is None or len(scan) == 0:
        return math.inf
    rel = scan.bearing - direction_body_rad
    along = scan.range * np.cos(rel)
    across = np.abs(scan.range * np.sin(rel))
    sel = (along > 0) & (across <= half_width_m) & (along <= rmax)
    return float(along[sel].min()) if sel.any() else math.inf


class Follower:
    def __init__(self, params: FollowParams = None):
        self.p = params or FollowParams()
        self.v_prev = np.zeros(2)
        self.t_prev = None

    def reset(self):
        self.v_prev = np.zeros(2)
        self.t_prev = None

    def step(self, now, pose, path, goal, scan: Scan, clearance_m, gate_leg=False, hold=False, brake_end=True,
             chord_blocked=None, vel=None) -> Command:
        """pose = (x, y, yaw) course frame; path (n,2) from the vehicle to the goal; scan in the
        level body frame; clearance_m = map clearance at the body. hold: command zero velocity
        (used while blocked), still yawing toward the goal and backing off if something is too close."""
        p = self.p
        dt = 0.0 if self.t_prev is None else max(1e-3, now - self.t_prev)
        self.t_prev = now
        xy = np.asarray(pose[:2], float)
        yaw = float(pose[2])
        # steer from where the vehicle will be when the command takes effect
        xy_pred = xy + (np.asarray(vel, float)[:2] * p.lag_s if vel is not None else 0.0)
        cmd = Command()
        if path is None or len(path) < 2:
            v_des = np.zeros(2)
            yaw_target = float("nan")
            cmd.limiter = "no path"
        else:
            path = np.asarray(path, float)
            i, s_here, s_total, cum, seg_len = path_progress(path, xy_pred)
            carrot = point_at(path, cum, seg_len, s_here + p.lookahead_m)
            if chord_blocked is not None:
                # pure pursuit cuts corners: shorten the look-ahead until the straight chord to the
                # carrot keeps the planning margin; if nothing does, accept the hard margin
                L = p.lookahead_m
                while L > p.lookahead_min_m and chord_blocked(xy_pred, carrot, False):
                    L -= 0.05
                    carrot = point_at(path, cum, seg_len, s_here + L)
                if chord_blocked(xy_pred, carrot, False):
                    L = p.lookahead_m
                    carrot = point_at(path, cum, seg_len, s_here + L)
                    while L > p.lookahead_min_m and chord_blocked(xy_pred, carrot, True):
                        L -= 0.05
                        carrot = point_at(path, cum, seg_len, s_here + L)
            d = carrot - xy_pred
            dist_c = float(np.linalg.norm(d))
            if dist_c < 1e-3:
                d = path[-1] - xy
                dist_c = float(np.linalg.norm(d))
            u = d / max(dist_c, 1e-6)
            d_goal = max(s_total - s_here, float(np.linalg.norm(path[-1] - xy_pred)))
            v_max = p.v_gate if gate_leg else p.v_cruise
            limiter = "cruise"
            v = v_max
            # braking to the end of the path (only when the path ends at a stop)
            if brake_end:
                vb = math.sqrt(2.0 * p.a_max * max(d_goal, 0.0)) + p.v_floor
                if vb < v:
                    v, limiter = vb, "goal"
            # corners within the look-ahead
            turn = self._turn_ahead(path, cum, seg_len, s_here, p.corner_look_m)
            vc = v_max * max(0.2, 1.0 - 0.8 * turn / (math.pi / 2))
            if vc < v:
                v, limiter = vc, "corner"
            # map clearance
            f = float(np.clip((clearance_m - p.clear_slow_m) / max(p.clear_full_m - p.clear_slow_m, 1e-6), 0.0, 1.0))
            vcl = p.v_floor + f * (v_max - p.v_floor)
            if vcl < v:
                v, limiter = vcl, "clearance"
            # reactive guard: scan returns inside the corridor of the UPCOMING path
            dir_body = wrap(math.atan2(u[1], u[0]) - yaw)
            ahead = path_obstacle(scan, pose, path, cum, seg_len, s_here, p.corridor_half_m, p.stop_m + p.reactive_ramp_m + 0.3)
            fr = float(np.clip((ahead - p.stop_m) / p.reactive_ramp_m, 0.0, 1.0))
            vr = fr * v_max
            if vr < v:
                v, limiter = vr, f"scan {ahead:.2f} m"
            # heading error
            err = abs(dir_body)
            if err > math.radians(45):
                vh = max(p.v_floor, v_max * max(math.cos(err), 0.0))
                if vh < v:
                    v, limiter = vh, "heading"
            v_des = u * v
            yaw_target = math.atan2(u[1], u[0])
            if gate_leg and len(path) >= 2:
                # through an opening the nose points along the leg, not at a short-term carrot
                leg = path[-1] - path[0]
                if np.linalg.norm(leg) > 0.05:
                    yaw_target = math.atan2(leg[1], leg[0])
            cmd.limiter = limiter
        if hold:
            v_des = np.zeros(2)
            cmd.limiter = "hold"
            # back off from anything inside the stop distance straight ahead
            ahead0 = scan_ahead(scan, 0.0, p.corridor_half_m)
            if ahead0 < p.stop_m:
                v_des = -np.array([math.cos(yaw), math.sin(yaw)]) * p.hold_backoff_mps
                cmd.limiter = f"hold, backing off ({ahead0:.2f} m ahead)"
        # acceleration limit
        if dt > 0:
            dv = v_des - self.v_prev
            n = float(np.linalg.norm(dv))
            if n > p.a_max * dt:
                v_des = self.v_prev + dv / n * p.a_max * dt
        self.v_prev = v_des
        cmd.vx, cmd.vy = float(v_des[0]), float(v_des[1])
        cmd.speed = float(np.linalg.norm(v_des))
        # yaw
        if hold and goal is not None:
            g = np.asarray(goal.xy, float) - xy
            if np.linalg.norm(g) > 0.3:
                yaw_target = math.atan2(g[1], g[0])
        if not math.isnan(yaw_target):
            e = wrap(yaw_target - yaw)
            rate_max = p.yaw_rate_near if (clearance_m < p.yaw_near_surface_m and cmd.speed < 0.1) else p.yaw_rate_max
            cmd.yaw_rate = float(np.clip(p.yaw_kp * e, -rate_max, rate_max))
            cmd.yaw_target = yaw_target
        return cmd

    @staticmethod
    def _turn_ahead(path, cum, seg_len, s_here, look_m, chord_m=0.3):
        """Heading change of the path over the next look_m, measured on chords of chord_m so that
        grid-cell zigzags do not read as corners: angle between the chord starting here and the
        most-turned later chord."""
        if cum[-1] - s_here < chord_m * 1.5:
            return 0.0
        a0 = point_at(path, cum, seg_len, s_here)
        a1 = point_at(path, cum, seg_len, s_here + chord_m)
        h0 = math.atan2(a1[1] - a0[1], a1[0] - a0[0])
        turn = 0.0
        s = s_here + chord_m
        while s + chord_m <= min(cum[-1], s_here + look_m + chord_m):
            b0 = point_at(path, cum, seg_len, s)
            b1 = point_at(path, cum, seg_len, s + chord_m)
            if math.dist(b0, b1) > 1e-3:
                turn = max(turn, abs(wrap(math.atan2(b1[1] - b0[1], b1[0] - b0[0]) - h0)))
            s += chord_m / 2
        return turn
