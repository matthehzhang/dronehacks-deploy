"""The live mission state machine. PURE: numbers in, a velocity command out. No ROS, no clocks.

    WAIT     until the flight interface reports Offboard (the pilot armed, then switched to Position)
    CLIMB    straight up to cruise height, no horizontal motion
    FLY      replanner + follower through the course goals; integrates every depth scan into the grid
    PAD      the last goal: the ring detector refines the pad centre; fly over it
    DESCEND  down to `descend_to_z` (the interface floor on the drone; LAND is a separate flag for a
             sim adapter), then hold
    HOLD     blocked / timed out: zero velocity, back off from anything inside the stop distance
    PAUSED   Offboard lost: the pilot has the vehicle; nothing is commanded
    FAULT    invalid/missing input: zero command, latched until process restart

Frames. odom = cuVSLAM's ENU, origin at the VIO start. course = the published OBJ frame. At the
moment Offboard starts the vehicle is assumed to be at the course start pose (launch mark, heading
along the guideline, or the declared start parameters); that fixes course->odom for the run.
Everything in here is computed in the course frame; `Command` velocities come back in odom ENU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..coursenav import planner
from ..coursenav.course import load_course, resolve, scan_field, scan_grid
from ..coursenav.objmap import CourseMap
from ..coursenav.pf import CorrectionPF, PFParams, compose, inverse
from ..coursenav.scan import Scan, scan_from_points
from .follower import FollowParams, Follower, wrap
from .navgrid import NavGrid, NavParams
from .replanner import ReplanParams, Replanner, goals_from_plan


@dataclass
class MissionParams:
    cruise_z_m: float = 0.8           # body height above the floor (course file default)
    landed_z_m: float = 0.10          # body origin above the floor when landed: PX4 dist_bottom on skybeta-14, 2026-11-07
    descend_to_z_odom: float = 0.3    # interface floor on the drone; the sim adapter may land instead
    climb_rate_mps: float = 0.35
    descend_rate_mps: float = 0.25
    z_kp: float = 1.2
    z_tol_m: float = 0.12
    pad_fix_max_age_s: float = 2.0
    pad_shift_max_m: float = 0.6      # ignore detections further than this from the map's pad centre
    pad_fix_min_range_m: float = 2.0  # the level camera sees the whole ring set only beyond ~1.9 m; closer
                                      # fixes are biased (0.27 m at 1.6 m on synthetic frames)
    pad_fix_max_range_m: float = 4.6
    pad_min_fixes: int = 3            # before the detections move the goal off the map/PF estimate
    pad_over_m: float = 0.12          # "over the pad" tolerance before descending
    time_limit_s: float = 170.0       # the run is 180 s; stop driving before the referee does
    hold_recheck_s: float = 2.0
    blocked_abort_s: float = 20.0     # blocked this long -> stay in HOLD (pilot lands)
    chicane: str = "auto"
    start_xy: tuple = None            # course frame; None = course-file prior (launch mark)
    start_yaw_deg: float = None
    # localizer (coursenav.pf): corrects course<-odom from the depth scan against the prior map
    use_pf: bool = True
    pf_particles: int = 400
    pf_sigma_xy_m: float = 0.10       # placement uncertainty on the start mark
    pf_sigma_yaw_deg: float = 4.0
    pf_max_sigma_m: float = 0.20      # ignore estimates less certain than this
    pf_alpha: float = 0.3             # low-pass on the transform update
    scan_band_lo: float = 0.3         # localizer scan band (the field grid is built with it)
    scan_band_hi: float = 1.6
    scan_rmax_m: float = 4.0
    depth_max_age_s: float = 0.5


@dataclass
class Observation:
    now: float
    odom_xyz: tuple                   # odom ENU position
    odom_yaw: float
    offboard_active: bool
    pts_level: np.ndarray = None      # (n,3) depth points in the LEVEL body frame (x fwd, y left, z up)
    height_m: float = 0.8             # body origin above the floor when the points were taken
    pts_t: float = -math.inf
    pad_fix: tuple = None             # (x_fwd, y_left) of the pad centre in the level body frame, from
                                      # the BODY origin (the sensor layer adds the camera offset)
    pad_fix_t: float = -math.inf


@dataclass
class Command:
    vx: float = 0.0                   # odom ENU, m/s
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0
    land: bool = False
    state: str = ""
    note: str = ""
    goal: str = ""
    path_course: np.ndarray = None
    pose_course: tuple = None
    speed: float = 0.0


class LiveMission:
    def __init__(self, params: MissionParams = None, nav_params: NavParams = None,
                 replan_params: ReplanParams = None, follow_params: FollowParams = None, course_file=None):
        self.p = params or MissionParams()
        self.doc = load_course(course_file) if course_file else load_course()
        self.cmap = CourseMap.load(resolve(self.doc, "map"))
        self.grid = NavGrid(self.cmap, self.doc, nav_params)
        self.follower = Follower(follow_params)
        self.replan_params = replan_params or ReplanParams()
        self.plan = self._plan_course()
        self.goals = goals_from_plan(self.plan)
        self.pad_xy_map = np.asarray(self.doc["sequence"][-1]["xy"], float)
        centres = [w for w in self.plan['waypoints'] if w.get('part') == 'centre']
        lines = self.plan['gate_lines']
        if len(centres) != len(lines):
            raise ValueError('plan gate geometry does not match crossing goals')
        gate_rules = {w['name']: {'line': line, 'normal': w['normal']}
                      for w, line in zip(centres, lines)}
        self.replanner = Replanner(self.grid, self.goals, self.replan_params, gate_rules=gate_rules)
        self.state = "WAIT"
        self.T = None                 # course->odom: odom = R(dyaw) course + t
        self.t_offboard = None
        self.t_state = None
        self.pad_est = None           # course-frame pad centre from detections (running mean)
        self.pad_n = 0
        self.hold_since = None
        self._last_odom = None        # (t, x, y) odom, for the velocity estimate
        self.vel_course = np.zeros(2)
        self._pts_t = -math.inf
        self.scan_nav = None          # flight-band scan (guard + live grid)
        self.scan_pf = None           # full-band scan (localizer)
        self.pf = None
        self.pf_updates = 0
        self.pf_sigma = math.nan
        self._pf_last_t = None
        if self.p.use_pf:
            field = scan_grid(self.cmap, self.p.scan_band_lo, self.p.scan_band_hi)
            net = self.cmap.band(self.p.scan_band_lo, self.p.scan_band_hi, only_prefixes=("Net",))
            self.pf = CorrectionPF(field, scan_field(field, self.doc), PFParams(n=self.p.pf_particles), uninformative=net)
        self.z_odom_cruise = self.p.cruise_z_m - self.p.landed_z_m
        self.log = []
        self.fault = None

    # ------------------------------------------------------------------ planning
    def _plan_course(self):
        doc, g = self.doc, self.grid
        if self.p.start_xy is None:
            sx, sy = doc["start"].get("spawn_xy", doc["start"]["xy"])
        else:
            sx, sy = self.p.start_xy
        if self.p.chicane == "auto":
            res = planner.plan_options(doc, g.obst, g.prior_blocked, g.prior_clearance, start_xy=(sx, sy), cmap=self.cmap)
            plan = res["plan"]
            plan["reason"] = res["reason"]
        else:
            plan = planner.plan(doc, g.obst, g.prior_blocked, g.prior_clearance, start_xy=(sx, sy),
                                chicane=self.p.chicane, cmap=self.cmap)
            plan["reason"] = "forced"
        plan["start_course"] = (float(sx), float(sy), math.radians(
            float(doc["start"]["yaw_deg"]) if self.p.start_yaw_deg is None else self.p.start_yaw_deg))
        return plan

    # ------------------------------------------------------------------ frames
    def set_transform(self, odom_xy, odom_yaw):
        sx, sy, syaw = self.plan["start_course"]
        dyaw = wrap(odom_yaw - syaw)
        c, s = math.cos(dyaw), math.sin(dyaw)
        self.T = (odom_xy[0] - (c * sx - s * sy), odom_xy[1] - (s * sx + c * sy), dyaw)

    def odom_to_course(self, xy, yaw):
        tx, ty, dyaw = self.T
        c, s = math.cos(-dyaw), math.sin(-dyaw)
        x, y = xy[0] - tx, xy[1] - ty
        return (c * x - s * y, s * x + c * y, wrap(yaw - dyaw))

    def course_vec_to_odom(self, vx, vy):
        dyaw = self.T[2]
        c, s = math.cos(dyaw), math.sin(dyaw)
        return c * vx - s * vy, s * vx + c * vy

    # ------------------------------------------------------------------ helpers
    def _set(self, state, now, note=""):
        if state != self.state:
            self.log.append((now, self.state, state, note))
        self.state = state
        self.t_state = now

    def _vz(self, z_now, z_target, rate):
        return float(np.clip(self.p.z_kp * (z_target - z_now), -rate, rate))

    def _localize(self, obs: Observation):
        """Particle filter over course<-odom from the full-band scan against the prior map."""
        odom = np.array([obs.odom_xyz[0], obs.odom_xyz[1], obs.odom_yaw])
        dt = 0.0 if self._pf_last_t is None else max(0.0, obs.now - self._pf_last_t)
        self._pf_last_t = obs.now
        self.pf.predict(odom, dt, True)
        if self.scan_pf is not None and len(self.scan_pf) and self.pf.update(odom, self.scan_pf):
            self.pf_updates += 1
        mean, cov = self.pf.estimate(odom)
        self.pf_sigma = math.sqrt(max(cov[0, 0], cov[1, 1], 0.0))
        if self.pf_sigma > self.p.pf_max_sigma_m:
            return
        T_co = compose(mean, inverse(odom))            # course <- odom
        T_oc = inverse(T_co)                           # odom <- course: what self.T stores
        a = self.p.pf_alpha
        tx, ty, dyaw = self.T
        self.T = (tx + a * (float(T_oc[0]) - tx), ty + a * (float(T_oc[1]) - ty),
                  wrap(dyaw + a * wrap(float(T_oc[2]) - dyaw)))

    def _update_velocity(self, now, pose_c):
        if self._last_odom is not None:
            dt = now - self._last_odom[0]
            if 0.02 <= dt <= 0.5:
                v = (np.asarray(pose_c[:2]) - self._last_odom[1]) / dt
                self.vel_course = 0.6 * self.vel_course + 0.4 * v
        self._last_odom = (now, np.asarray(pose_c[:2], float))

    def _update_pad(self, obs: Observation, pose_c):
        if obs.pad_fix is None or obs.now - obs.pad_fix_t > self.p.pad_fix_max_age_s:
            return
        x, y, yaw = pose_c
        fx, fy = obs.pad_fix
        if not (self.p.pad_fix_min_range_m <= math.hypot(fx, fy) <= self.p.pad_fix_max_range_m):
            return
        c, s = math.cos(yaw), math.sin(yaw)
        est = np.array([x + c * fx - s * fy, y + s * fx + c * fy])
        if np.linalg.norm(est - self.pad_xy_map) > self.p.pad_shift_max_m:
            return
        self.pad_n += 1
        a = 1.0 / min(self.pad_n, 10)
        self.pad_est = est if self.pad_est is None else self.pad_est + a * (est - self.pad_est)
        if self.pad_n >= self.p.pad_min_fixes:
            self.goals[-1].xy = self.pad_est.copy()

    # ------------------------------------------------------------------ the tick
    def abort(self, now, reason):
        if self.fault is None:
            self.fault = reason
            self._set('FAULT', now, reason)
            self.follower.reset()
        self.scan_nav = self.scan_pf = None
        return Command(state='FAULT', note=self.fault)

    def step(self, obs: Observation) -> Command:
        p = self.p
        now = obs.now
        if self.fault is not None:
            return Command(state='FAULT', note=self.fault)
        if not all(math.isfinite(v) for v in (*obs.odom_xyz, obs.odom_yaw, now)):
            return self.abort(now, 'non-finite mission observation')
        cmd = Command(state=self.state)
        z = obs.odom_xyz[2]
        if self.state not in ("WAIT", "PAUSED") and not obs.offboard_active:
            self._set("PAUSED", now, "Offboard lost")
            cmd.state = self.state
            return cmd
        if self.state == "WAIT":
            if obs.offboard_active:
                self.set_transform(obs.odom_xyz[:2], obs.odom_yaw)
                self.t_offboard = now
                self.follower.reset()
                if self.pf is not None:
                    sx, sy, syaw = self.plan["start_course"]
                    self.pf.init((sx, sy, syaw), self.p.pf_sigma_xy_m, math.radians(self.p.pf_sigma_yaw_deg),
                                 np.array([obs.odom_xyz[0], obs.odom_xyz[1], obs.odom_yaw]))
                    self._pf_last_t = now
                self._set("CLIMB", now)
            cmd.state = self.state
            return cmd
        if self.state == "PAUSED":
            return cmd
        scan_fresh = (obs.pts_level is not None and math.isfinite(obs.pts_t)
                      and -0.05 <= now - obs.pts_t <= p.depth_max_age_s)
        if not scan_fresh:
            self.scan_nav = self.scan_pf = None
        new_scan = scan_fresh and obs.pts_t > self._pts_t
        if new_scan:
            self._pts_t = obs.pts_t
            band = self.doc.get("plan_band_m", 0.25)
            # the nav scan is the planner's flight band exactly: below it are boulder tops (a keep-out
            # in the prior), above it the window lintel and the tube roof, which must not become walls
            self.scan_nav = scan_from_points(obs.pts_level, obs.height_m, zlo=self.p.cruise_z_m - band,
                                             zhi=self.p.cruise_z_m + band, rmax=self.p.scan_rmax_m)
            self.scan_pf = scan_from_points(obs.pts_level, obs.height_m, zlo=self.p.scan_band_lo,
                                            zhi=self.p.scan_band_hi, rmax=self.p.scan_rmax_m)
        if self.state in ('FLY', 'PAD', 'HOLD', 'DESCEND') and (self.scan_nav is None or not len(self.scan_nav)):
            return self.abort(now, 'missing, stale or empty navigation depth scan')
        if new_scan:
            if self.pf is not None and self.pf.ready and self.state in ("CLIMB", "FLY", "PAD", "HOLD", "DESCEND"):
                self._localize(obs)
        pose_c = self.odom_to_course(obs.odom_xyz[:2], obs.odom_yaw)
        cmd.pose_course = pose_c
        self._update_velocity(now, pose_c)
        # the live grid eats every scan, in every flying state
        if new_scan and self.scan_nav is not None and self.state in ("FLY", "PAD", "HOLD", "CLIMB"):
            self.grid.integrate(pose_c, self.scan_nav, now)
        if now - self.t_offboard > p.time_limit_s and self.state in ("FLY", "PAD", "HOLD"):
            self._set("HOLD", now, "time limit")
        if self.state == "CLIMB":
            cmd.vz = self._vz(z, self.z_odom_cruise, p.climb_rate_mps)
            if abs(z - self.z_odom_cruise) <= p.z_tol_m:
                self._set("FLY", now)
            cmd.state = self.state
            return cmd
        cmd.vz = self._vz(z, self.z_odom_cruise, p.climb_rate_mps)
        rp = self.replanner
        if self.state in ("FLY", "PAD"):
            if rp.done:
                self._set("DESCEND", now)
            else:
                goal = rp.goal
                if goal is rp.goals[-1] and self.state == "FLY":
                    self._set("PAD", now)
                if self.state == "PAD":
                    self._update_pad(obs, pose_c)
                if rp.arrived(pose_c[:2]):
                    if goal is rp.goals[-1]:
                        self._set("DESCEND", now)
                    else:
                        rp.advance()
                        goal = rp.goal
                if self.state != "DESCEND" and goal is not None:
                    path = rp.update(pose_c[:2], now)
                    clearance = self.grid.clearance_at(pose_c[:2])
                    if rp.blocked:
                        self.hold_since = self.hold_since or now
                        self._set("HOLD", now, rp.last_reason)
                    else:
                        self.hold_since = None
                        fc = self.follower.step(now, pose_c, path, goal, self.scan_nav, clearance, gate_leg=goal.is_gate_leg,
                                                brake_end=not rp.continues,
                                                chord_blocked=rp.chord_blocked,
                                                gate_blocked=rp.gate_blocked,
                                                vel=self.vel_course)
                        cmd.vx, cmd.vy = self.course_vec_to_odom(fc.vx, fc.vy)
                        cmd.yaw_rate, cmd.speed, cmd.note = fc.yaw_rate, fc.speed, fc.limiter
                        cmd.goal, cmd.path_course = goal.name, path
        if self.state == "HOLD":
            goal = rp.goal
            clearance = self.grid.clearance_at(pose_c[:2])
            fc = self.follower.step(now, pose_c, None, goal, self.scan_nav, clearance, hold=True,
                                    gate_blocked=rp.gate_blocked)
            cmd.vx, cmd.vy = self.course_vec_to_odom(fc.vx, fc.vy)
            cmd.yaw_rate, cmd.note = fc.yaw_rate, fc.limiter
            if goal is not None and self.hold_since is not None and now - self.t_state >= p.hold_recheck_s \
                    and now - self.hold_since < p.blocked_abort_s and now - self.t_offboard <= p.time_limit_s:
                path = rp.plan(pose_c[:2], now)
                if path is not None and not rp.blocked:
                    self._set("FLY", now, "path clear again")
        if self.state == "DESCEND":
            # hold the pad centre horizontally while descending
            goal_xy = self.goals[-1].xy
            dx, dy = goal_xy[0] - pose_c[0], goal_xy[1] - pose_c[1]
            vx_c, vy_c = float(np.clip(0.8 * dx, -0.2, 0.2)), float(np.clip(0.8 * dy, -0.2, 0.2))
            cmd.vx, cmd.vy = self.course_vec_to_odom(vx_c, vy_c)
            over = math.hypot(dx, dy) <= p.pad_over_m
            cmd.vz = self._vz(z, p.descend_to_z_odom, p.descend_rate_mps) if over else 0.0
            cmd.land = over and z <= p.descend_to_z_odom + 0.05
            cmd.note = "descending" if over else "centring over the pad"
            cmd.goal = "pad"
        cmd.state = self.state
        return cmd

    def status(self):
        return dict(state=self.state, goal=self.replanner.goal.name if self.replanner.goal else None,
                    goal_index=self.replanner.k, goals=len(self.goals), replans=self.replanner.replans,
                    pad_fixes=self.pad_n, live=self.grid.live_stats(), pf_updates=self.pf_updates,
                    pf_sigma=None if math.isnan(self.pf_sigma) else round(self.pf_sigma, 3),
                    T=None if self.T is None else [round(float(v), 3) for v in self.T], fault=self.fault)
