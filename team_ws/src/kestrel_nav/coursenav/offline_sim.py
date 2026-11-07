"""Offline closed-loop course flight: the REAL refnav mission + the REAL localizer PF, against a
kinematic vehicle, a drifting VIO and depth rendered from the published map. OFFLINE TESTING ONLY:
no ROS, no simulator, no PX4. It never commands anything.

What is real here: refnav.core.Mission (every decision), refnav.depth (clearance and corridor
checks), coursenav.scan + coursenav.pf (localizer), coursenav.planner (route from the localizer's
start estimate, as HANDOFF section 4.5 does it).

What is modelled (GUESSES, labelled):
  * the vehicle tracks PX4's setpoint in PX4's OWN estimate frame at <= 1 m/s, 60 deg/s (like
    tests/test_refnav_core.FakePX4); level attitude; no wind, no lag;
  * VIO: scale error, heading drift per metre and a random walk;
  * darkness (SimConfig.darkness "lit" | "dim" | "dark", A5b): the enclosed passage (cells with a
    roof above the flight band, from the map's geometry) is dark. "dark": surfaces inside it return
    no depth at all (the lit exit and the room beyond still do) and VIO loses visual tracking inside
    (vio_ok False: an extra dark_vio_walk random walk per sqrt metre, as an IMU-only pose would
    drift); "dim": 70 % of the returns inside drop out and the walk triples. GUESSES: the real
    tunnel's brightness is unmeasured in the sim and unknown at the venue (assume dark);
  * PX4's estimate = VIO rotated by the declared +Y heading (mode "vio", today's pipeline) or the
    localizer's corrected odometry (mode "coursenav", HANDOFF 4.6 redirect), with no EKF smoothing;
    mode "corrected" (what flight E flew): PX4 runs on raw VIO and the localizer's (course - VIO+fixture)
    offset reaches refnav as a bounded correction of its remaining targets (refnav correction_*);
  * depth: coursenav.render (stereo error, net model, 5 cm voxels);
  * the drone is a sphere of prop-tip radius for collision checks (pessimistic vertically);
  * optionally the WORLD differs from the published OBJ (SimConfig.unmapped, shift_group/shift_m):
    extra objects and a moved wall exist only in the rendered depth and the collision check; the
    localizer and planner still use the published map. This is how the live map's landmarks and
    conflicts are exercised offline.

Frames: course ENU (x east, y north, z up, floor = 0); PX4 NED (n = ENU y, e = ENU x, d = -up).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from refnav import core as C

from . import planner
from .course import see_through_grid, plan_grids, scan_grid
from .objmap import FLOOR_MARKING_PREFIXES, CourseMap
from .occgrid import LiveGrid, OccParams
from .pf import CorrectionLimiter, CorrectionPF, PFParams, compose, inverse, project_correction, wrap
from .render import Camera, DepthModel, DepthRenderer, body_pose_matrix
from .scan import depth_points_level, scan_from_points



def world_map(cmap: CourseMap, unmapped=(), shift_group="", shift_m=(0.0, 0.0)) -> CourseMap:
    """The simulated WORLD: the published map, optionally with a group moved and extra (unlabelled)
    objects added. Only the renderer and the collision check see it."""
    if not unmapped and not (shift_group and any(shift_m)):
        return cmap
    occ, lab = cmap.occupied.copy(), cmap.labels.copy()
    if shift_group and any(shift_m):
        m = cmap.group_mask([shift_group])
        di, dj = (int(round(v / cmap.res)) for v in shift_m)
        occ[m], lab[m] = False, 0
        moved = np.roll(np.roll(m, di, axis=0), dj, axis=1)
        occ |= moved
        lab[moved] = np.roll(np.roll(cmap.labels, di, axis=0), dj, axis=1)[moved]
    nx, ny, nz = cmap.shape
    X, Y = np.meshgrid(cmap.origin[0] + (np.arange(nx) + 0.5) * cmap.res,
                       cmap.origin[1] + (np.arange(ny) + 0.5) * cmap.res, indexing="ij")
    for x, y, r, h in unmapped:
        col = np.hypot(X - x, Y - y) <= r
        k1 = min(nz, cmap.z_index(h) + 1)
        occ[col, max(0, cmap.z_index(0.0)):k1] = True
    meta = dict(cmap.meta, world="published map + simulated as-built differences (offline only)")
    return CourseMap(occ, cmap.origin.copy(), cmap.res, lab, list(cmap.groups), meta)


def dark_region(cmap: CourseMap, z, roof_within_m=1.5):
    """2D cells of an enclosed passage: a mapped surface above the flight band (between z + 0.3 and
    z + roof_within_m), excluding the surfaces themselves. Pure geometry: the tube's footprint."""
    k0 = max(0, cmap.z_index(z + 0.3))
    k1 = min(cmap.shape[2] - 1, cmap.z_index(z + roof_within_m))
    roof = cmap.occupied[:, :, k0:k1 + 1].any(axis=2)
    return roof & ~cmap.band(z - 0.25, z + 0.25)


def route_params(doc):
    """refnav Params for course routes: the course file's refnav_params (shared with tools/plan_route.py)."""
    return {k: v for k, v in doc.get("refnav_params", {}).items() if not k.startswith("_")}


@dataclass
class SimConfig:
    mode: str = "vio"                       # "vio" | "coursenav" | "corrected"
    chicane: str = "auto"                   # "auto": the planner scores every option and chooses
    legs: int = 0                           # fly only the first N legs (0 = all, ending with the pad landing)
    spawn: tuple = (-2.5146, 1.1144)        # measured sim spawn (0.20 m north of the launch mark)
    spawn_yaw_deg: float = 90.0
    declared_heading_deg: float = 90.0      # ev_bridge's heading reference in mode "vio"
    landed_z: float = 0.18                  # body origin above the floor when landed (GUESS: refnav test fixture)
    cruise_z: float = None                  # body height above the floor in flight (default: course cruise_z_m)
    vio_scale_err: float = 0.02             # flight D (2026-11-06): 0.23 m by the tunnel exit (~10 m), 2.3 % of distance
    vio_yaw_drift_deg_per_m: float = 0.3    # ... modelled as 2 % scale + 0.3 deg/m heading drift (a little worse)
    vio_walk_m_per_sqrt_m: float = 0.01     # GUESS: unmodelled random motion per sqrt(metre)
    rangefinder_sigma: float = 0.02
    tilt_deg: float = 0.0
    depth: DepthModel = field(default_factory=DepthModel)
    depth_latency_s: float = 0.04
    dt: float = 0.05                        # refnav's 20 Hz tick
    depth_every: int = 2                    # depth (and localizer) at 10 Hz
    pre_arm_s: float = 4.0                  # landed localization before the route is planned
    max_time_s: float = 400.0
    prop_radius_m: float | None = None     # None: score with the course's resolved aircraft envelope
    particles: int = 600
    live_map: bool = True                   # fill a live occupancy grid (coursenav.occgrid) from the scans
    use_landmarks: bool = False             # ... and let the PF match live landmarks as well as the prior map
    unmapped: tuple = ()                    # extra world objects NOT in the map: ((x, y, radius, height), ...)
    shift_group: str = ""                   # move this OBJ group in the world only (as-built error) ...
    shift_m: tuple = (0.0, 0.0)             # ... by (dx, dy) metres
    darkness: str = "lit"                   # "lit" | "dim" | "dark": the enclosed passage (tunnel) is dark
    dark_vio_walk_m_per_sqrt_m: float = 0.08   # GUESS: IMU-only drift while vision is lost inside
    corr_rate_per_m: float = 0.05           # loc_node's correction rate limit (m per m travelled) ...
    corr_rate_per_s: float = 0.01           # ... and per second
    corr_project_sigma_m: float = 0.10      # loc_node: keep a correction only along directions with sigma <= this
    course_nav: bool = False                # nav="path": take the smoothing/refnav settings from the course's
                                            # navigation block, as fly_course.py does (else the older defaults)
    refnav_overrides: dict = None           # refnav Params to override the course profile (experiments)
    nav: str = "legs"                       # "legs": the planner's waypoint route (refnav legs + bounded
                                            # corrections); "path": stops only, cruise legs flown along the
                                            # navigator's smoothed paths on the fused map (coursenav.navigator),
                                            # correction by the drift EKF (coursenav.drift), path follower
    px4: str = "kinematic"                  # "kinematic": straight to the setpoint at <= 1 m/s (never overshoots);
                                            # "dynamic": coursenav.px4_model (PX4's cascade, lag, delay; overshoots
                                            # like managed run 1 with the carrot profile)
    px4_tau_s: float = 0.15                 # dynamic model: attitude/thrust lag (GUESS until fitted to a bag)
    px4_delay_s: float = 0.05               # dynamic model: control delay (GUESS)
    seed: int = 0


@dataclass
class SimResult:
    result: str
    t_end: float
    legs_done: int
    legs: int
    collided: bool
    min_clearance_m: float
    pf_err_p50: float
    pf_err_p95: float
    pf_err_max: float
    vio_err_final: float
    land_err_m: float
    start_est: tuple
    plan: dict
    log: dict
    tunnel_ticks: int = 0                  # refnav ticks that advanced without depth (tunnel mode)
    dark_ticks: int = 0                    # ticks the body spent inside the dark region


def _ned(xy):
    return (xy[1], xy[0])


def _enu(ne):
    return (ne[1], ne[0])


class OfflineFlight:
    def __init__(self, doc, cmap: CourseMap, cfg: SimConfig = None):
        self.doc, self.cmap, self.cfg = doc, cmap, cfg or SimConfig()
        c = self.cfg
        # Resolve once without mutating a caller's reusable SimConfig.
        self.prop_radius_m = float(doc['vehicle']['radius_m'] if c.prop_radius_m is None else c.prop_radius_m)
        if not math.isfinite(self.prop_radius_m) or self.prop_radius_m <= 0:
            raise ValueError('collision-scoring radius must be positive and finite')
        if c.mode not in ("vio", "coursenav", "corrected"):
            raise ValueError("mode must be vio, coursenav or corrected")
        self.rng = np.random.default_rng(c.seed)
        self.cruise_z = c.cruise_z if c.cruise_z is not None else doc["cruise_z_m"]
        self.cam = Camera(tilt_deg=c.tilt_deg)
        self.world = world_map(cmap, c.unmapped, c.shift_group, c.shift_m)
        self.renderer = DepthRenderer(self.world, self.cam, c.depth, seed=c.seed + 1)
        sg = scan_grid(cmap)
        net = cmap.band(0.3, 1.6, only_prefixes=("Net",))
        self.pf = CorrectionPF(sg, sg.distance(), PFParams(n=c.particles), seed=c.seed + 2, uninformative=net)
        self.limiter = CorrectionLimiter(c.corr_rate_per_m, c.corr_rate_per_s)
        self.last_corr_odom = None
        self.live = (LiveGrid(sg, OccParams(use_in_pf=c.use_landmarks), see_through=see_through_grid(cmap))
                     if (c.live_map or c.use_landmarks) else None)
        self.prior_dist = self.pf.dist
        self.t = 0.0
        self.obst, self.blocked, self.clearance = plan_grids(cmap, doc)
        if c.darkness not in ("lit", "dim", "dark"):
            raise ValueError("darkness must be lit, dim or dark")
        self.dark_mask = dark_region(self.world, self.cruise_z) if c.darkness != "lit" else None
        self.vio_lost = False
        self.dark_ticks = 0
        from scipy import ndimage
        solid = self.world.occupied & ~self.world.group_mask(FLOOR_MARKING_PREFIXES)
        self.solid_edt = ndimage.distance_transform_edt(~solid) * cmap.res
        # truth (course ENU) and VIO odom (cuVSLAM start frame, FLU)
        self.truth = np.array([c.spawn[0], c.spawn[1], math.radians(c.spawn_yaw_deg)])
        self.z = c.landed_z
        self.odom = np.zeros(3)
        self.M_vio = np.array([0.0, 0.0, math.radians(c.declared_heading_deg)])   # PX4 frame <- odom
        sxy = doc["start"].get("spawn_xy", doc["start"]["xy"])                   # the DECLARED start fixture
        self.nominal_T = np.array([sxy[0], sxy[1], math.radians(doc["start"].get("yaw_deg", 90.0))])
        self.T_mean = None
        self.last_cov = np.diag([0.09, 0.09, 0.02])
        self.px4 = dict(armed=False, nav=4, stream_since=None)
        self.px4m = None                    # SimConfig.px4 == "dynamic": coursenav.px4_model.PX4Model
        self.mission = None
        self.travelled = 0.0

    # --------------------------------------------------------------- frames
    def _M(self):
        """PX4-estimate frame (ENU-oriented) <- odom."""
        if self.cfg.mode == "coursenav" and self.T_mean is not None:
            return self.T_mean
        return self.M_vio

    def est(self):
        return compose(self._M(), self.odom)

    def vehicle(self):
        e = self.est()
        n, east = _ned(e[:2])
        heading = float(wrap(math.pi / 2 - e[2]))
        armed, nav = self.px4["armed"], self.px4["nav"]
        vel = tuple(float(c) for c in self.px4m.vel) if self.px4m is not None else None
        return C.Vehicle(pos=(n, east, -(self.z - self.cfg.landed_z)), heading=heading, pos_ok=True,
                         q=(math.cos(heading / 2), 0.0, 0.0, math.sin(heading / 2)), armed=armed, nav_state=nav,
                         landed=self.z <= self.cfg.landed_z + 1e-3, vel=vel)

    # --------------------------------------------------------------- sensing
    def in_dark(self, xy=None):
        if self.dark_mask is None:
            return False
        ij = self.cmap.xy_to_ij(self.truth[:2] if xy is None else xy)
        if np.any(ij < 0) or np.any(ij >= np.array(self.dark_mask.shape)):
            return False
        return bool(self.dark_mask[ij[0], ij[1]])

    def depth_frame(self):
        R_wb, t_wb = body_pose_matrix(self.truth[0], self.truth[1], self.z, self.truth[2])
        if self.dark_mask is None:
            return self.renderer.render(R_wb, t_wb)
        img, hits = self.renderer.render(R_wb, t_wb, with_hits=True)
        ok = (hits[..., 0] >= 0)
        inside = np.zeros(img.shape, bool)
        inside[ok] = self.dark_mask[hits[..., 0][ok], hits[..., 1][ok]]
        if self.cfg.darkness == "dark":
            img[inside] = np.nan                               # no light on the surfaces inside
        else:
            drop = inside & (self.rng.random(img.shape) < 0.7)
            img[drop] = np.nan
        return img

    def localize(self, img, dt):
        self.pf.predict(self.odom, dt, not self.vio_lost)
        h = self.z + self.rng.normal(0, self.cfg.rangefinder_sigma)
        pts = depth_points_level(img, self.cam.K, self.cam.R_body_cam(), self.cam.t_body, np.eye(3))
        scan = scan_from_points(pts, h)
        if len(scan):
            self.pf.update(self.odom, scan)
        mean, cov = self.pf.estimate(self.odom)
        self.T_mean = compose(mean, inverse(self.odom))
        self.last_cov = cov
        if self.live is not None:
            self.live.integrate(mean, scan, cov)
            if self.live.p.use_in_pf:
                f = self.live.landmark_field(now=self.t)
                self.pf.set_distance(self.prior_dist if f is None else f)
        return mean, len(scan)

    def clearance_now(self):
        ijk = np.floor((np.array([*self.truth[:2], self.z]) - self.cmap.origin) / self.cmap.res).astype(int)
        if np.any(ijk < 0) or np.any(ijk >= np.array(self.cmap.shape)):
            return 0.0
        return float(self.solid_edt[tuple(ijk)] - self.cmap.res * 0.5)

    # --------------------------------------------------------------- dynamics
    def _move(self, d_est_xy, d_yaw):
        """Move PX4's estimate by d_est_xy / d_yaw; the body really moves what VIO would have
        measured as that motion, divided by its scale error and minus its heading drift."""
        c = self.cfg
        th = self._M()[2]
        cm, sm = math.cos(-th), math.sin(-th)
        d_odom = np.array([cm * d_est_xy[0] - sm * d_est_xy[1], sm * d_est_xy[0] + cm * d_est_xy[1]])
        co, so = math.cos(-self.odom[2]), math.sin(-self.odom[2])
        body = np.array([co * d_odom[0] - so * d_odom[1], so * d_odom[0] + co * d_odom[1]]) / (1 + c.vio_scale_err)
        dist = float(np.hypot(*body))
        ct, st = math.cos(self.truth[2]), math.sin(self.truth[2])
        self.truth[0] += ct * body[0] - st * body[1]
        self.truth[1] += st * body[0] + ct * body[1]
        if dist > 0:
            self.truth[:2] += self.rng.normal(0, c.vio_walk_m_per_sqrt_m * math.sqrt(dist), 2)
        self.truth[2] = float(wrap(self.truth[2] + d_yaw - math.radians(c.vio_yaw_drift_deg_per_m) * dist))
        dark = self.in_dark()
        self.vio_lost = dark and c.darkness == "dark"
        if dark and dist > 0:                                  # vision lost or degraded: the pose drifts more
            walk = c.dark_vio_walk_m_per_sqrt_m * (1.0 if c.darkness == "dark" else 1.0 / 3.0)
            d_odom = d_odom + self.rng.normal(0, walk * math.sqrt(dist), 2)
        self.odom[:2] += d_odom
        self.odom[2] = float(wrap(self.odom[2] + d_yaw))
        self.travelled += dist

    def px4_step(self, t, out, dt):
        p = self.px4
        if out.stream:
            p["stream_since"] = t if p["stream_since"] is None else p["stream_since"]
        else:
            p["stream_since"] = None
        for cmd in out.commands:
            if cmd == "offboard" and p["stream_since"] is not None and t - p["stream_since"] >= 0.5:
                p["nav"] = C.NAV_OFFBOARD
            elif cmd == "arm":
                p["armed"] = True
            elif cmd == "land":
                p["nav"] = C.NAV_AUTO_LAND
            elif cmd == "disarm" and self.z <= self.cfg.landed_z + 1e-3:
                p["armed"] = False
        if not p["armed"]:
            return
        if p["nav"] == C.NAV_AUTO_LAND:
            self.z = max(self.cfg.landed_z, self.z - 0.5 * dt)
            if self.z <= self.cfg.landed_z + 1e-3:
                p["armed"] = False
            return
        if p["nav"] == C.NAV_OFFBOARD and out.setpoint is not None and self.cfg.px4 == "dynamic":
            self._px4_dynamic(t, out, dt)
        elif p["nav"] == C.NAV_OFFBOARD and out.setpoint is not None:
            n, e, d, yaw_ned = out.setpoint
            est = self.est()
            tgt = np.array(_enu((n, e)))
            step = tgt - est[:2]
            dist = float(np.hypot(*step))
            d_xy = step * min(1.0, 1.0 * dt / dist) if dist > 1e-9 else np.zeros(2)
            err = float(wrap((math.pi / 2 - yaw_ned) - est[2]))
            d_yaw = max(-math.radians(60) * dt, min(math.radians(60) * dt, err))
            self._move(d_xy, d_yaw)
            z_t = self.cfg.landed_z - d
            self.z += max(-1.0 * dt, min(1.0 * dt, z_t - self.z))

    def _px4_dynamic(self, t, out, dt):
        """PX4's position control (coursenav.px4_model) on PX4's OWN estimate: the model's position is
        re-synced to the estimate every tick (corrections may move the estimate frame), its velocity and
        acceleration state carry over; the resulting motion moves the body through _move."""
        from .px4_model import PX4Model, PX4ModelParams
        n, e, d, yaw_ned = out.setpoint
        est = self.est()
        n0, e0 = _ned(est[:2])
        d0 = -(self.z - self.cfg.landed_z)
        h0 = float(wrap(math.pi / 2 - est[2]))
        if self.px4m is None:
            self.px4m = PX4Model((n0, e0, d0), h0, PX4ModelParams(tau_s=self.cfg.px4_tau_s, delay_s=self.cfg.px4_delay_s))
        m = self.px4m
        m.pos[:] = (n0, e0, d0)
        m.yaw = h0
        vel = None if out.velocity is None else (out.velocity[0], out.velocity[1], 0.0)
        acc = None if getattr(out, "accel", None) is None else (out.accel[0], out.accel[1], math.nan)
        m.step(t, dt, position=(n, e, d), velocity=vel, accel=acc, yaw=yaw_ned)
        d_ne = m.pos[:2] - np.array([n0, e0])
        d_yaw_ned = float(wrap(m.yaw - h0))
        self._move(np.array(_enu(tuple(d_ne))), -d_yaw_ned)
        self.z = max(self.cfg.landed_z, self.cfg.landed_z - float(m.pos[2]))

    # --------------------------------------------------------------- path mode (navigator)
    def _path_setup(self, plan, est, prm):
        """Stops only (tunnel mouth/exit, the chicane's pre/post, the pad) as refnav's route; the
        cruise legs between them are navigator paths. Pass-through goals (the maze exit) stay goals."""
        from .drift import DriftEKF
        from .navigator import Navigator
        from .smooth import SmoothParams
        from .navplan import stops_and_legs
        route, legs, nav_legs = stops_and_legs(plan, est[:2], est[2])
        self.nav_legs = [dict(L, stop=np.array(L["stop"], float), goals=[np.array(g, float) for g in L["goals"]])
                         for L in nav_legs]
        prm.route, prm.legs = route, legs
        prm.track_mode = "path"
        prm.correction_max_m = 0.0                       # the drift EKF + whole-path conversion replace it
        need = float(self.doc["vehicle"]["radius_m"] + self.doc["vehicle"]["margin_m"])
        if self.cfg.course_nav:
            # exactly what fly_course.py --nav path flies: the course's navigation block (smoothing + refnav)
            from .navplan import smooth_params_for
            navcfg = {k: v for k, v in self.doc.get("navigation", {}).items() if not k.startswith("_")}
            for k, v in navcfg.get("refnav", {}).items():
                if not k.startswith("_") and hasattr(prm, k):
                    setattr(prm, k, v)
            for k, v in (self.cfg.refnav_overrides or {}).items():
                setattr(prm, k, v)
            sp = smooth_params_for({k: v for k, v in navcfg.items() if k not in ("mode", "refnav", "steady")},
                                   {"yaw_rate_dps": prm.yaw_rate_dps}, need)
        else:
            sp = SmoothParams(need_m=need, cruise_mps=prm.speed_mps,
                              accel_mps2=prm.accel_mps2 if prm.accel_mps2 > 0 else 0.5,
                              brake_mps2=prm.brake_mps2 if prm.brake_mps2 > 0 else 0.4,
                              yaw_rate_dps=prm.yaw_rate_dps)
        self.flow_sp = sp
        self.nav = Navigator(self.obst, self.blocked, self.clearance, need, sp, live=self.live,
                             gate_lines=plan.get("gate_lines"))
        self.ekf = DriftEKF()
        self.nav_leg, self.nav_pid, self.nav_progress_goal = None, 0, 0
        self.nav_stats = dict(plans=0, replans=0, plan_errors=0)
        return prm

    def _course_to_ned(self, pts, nominal_body):
        """Course-frame points -> PX4 local NED of this simulation, through the drift EKF (course ->
        nominal) and the fixed VIO frame relation (nominal = fixture o odom, PX4 = declared heading o odom)."""
        q = self.ekf.to_nominal(pts, nominal_body) if self.ekf.ready else np.atleast_2d(np.asarray(pts, float))
        odom = np.stack([compose(inverse(self.nominal_T), np.array([x, y, 0.0]))[:2] for x, y in q])
        est = np.stack([compose(self.M_vio, np.array([x, y, 0.0]))[:2] for x, y in odom])
        return np.stack([est[:, 1], est[:, 0]], axis=1)            # ENU -> NED

    def _path_tick(self, t, depth_tick, mean):
        from .planner import PlanError
        m = self.mission
        nb = compose(self.nominal_T, self.odom)
        self.ekf.predict(nb, self.cfg.dt)
        if depth_tick and mean is not None and self.pf.updates >= 30:
            self.ekf.update(mean, self.last_cov, nb)
        if m.targets is None:
            return
        believed = self.ekf.body_course(nb) if self.ekf.ready else nb
        for j in range(m.leg + 1, min(len(self.nav_legs), len(m.targets))):     # future stops follow the EKF
            n, e = self._course_to_ned([self.nav_legs[j]["stop"]], nb)[0]
            m.targets[j] = (float(n), float(e))
        if m.leg >= len(self.nav_legs) or not self.nav_legs[m.leg]["path"]:
            return
        if m.phase not in ("hold", "turn", "check", "go", "blocked", "backoff", "recheck"):
            return
        L = self.nav_legs[m.leg]
        speed = float(np.hypot(*self.px4m.vel[:2])) if self.px4m is not None else 0.0
        want = self.nav_leg != m.leg
        if not want and self.nav.path is not None:
            progress = m.path_s if m.path is not None and m.path.get("pid") == self.nav_pid else 0.0
            want = self.nav.needs_replan(t, progress)
        if want:
            goals = [g for g in L["goals"] if np.hypot(*(g - believed[:2])) > 0.3 or g is L["goals"][-1]]
            # drop pass-through goals already behind (closer to the start of the current path than us)
            if self.nav_leg == m.leg and self.nav.path is not None and len(goals) > 1:
                passed = []
                for g in goals[:-1]:
                    gi = int(np.argmin(np.linalg.norm(self.nav.path.pts - g, axis=1)))
                    if self.nav.path.s[gi] < (m.path_s if m.path is not None else 0.0):
                        passed.append(g)
                goals = [g for g in goals if not any(g is q for q in passed)]
            try:
                self.nav.plan(believed[:2], goals, t=t, v_start=speed if self.nav_leg == m.leg else 0.0)
                self.nav_pid += 1
                self.nav_stats["replans" if self.nav_leg == m.leg else "plans"] += 1
                self.nav_leg = m.leg
            except PlanError:
                self.nav_stats["plan_errors"] += 1
                return
        P = self.nav.path
        m.on_path(t, m.leg, self._course_to_ned(P.pts, nb), P.v, P.stops, pid=self.nav_pid,
                  tubes=None if P.tube is None else P.tube)

    # --------------------------------------------------------------- run
    def plan_from(self, est):
        """Plan from the localizer's start estimate (chicane "auto": every option scored, the best
        chosen, as fly_course.py does at start). -> (plan, route, legs metadata)."""
        c = self.cfg
        chicane = None if c.chicane in (None, "auto") else c.chicane
        p = planner.plan(self.doc, self.obst, self.blocked, self.clearance, start_xy=est[:2], chicane=chicane,
                         cmap=self.cmap)
        route = planner.to_route(p["waypoints"], est[:2], est[2])
        legs = planner.to_legs(p["waypoints"], self.obst, self.clearance, self.doc["vehicle"]["radius_m"])
        st = self.doc.get("navigation", {}).get("steady", {})
        if (c.refnav_overrides or {}).get("tube_gate") and st.get("tube_response_s") and st.get("leg_speed_cap_mps"):
            for L in legs:                                   # the same per-leg speeds fly_course --nav steady gives
                if L.get("tube_m"):
                    L["speed_mps"] = min(planner.leg_speed_mps(L["tube_m"], st["tube_response_s"], st["leg_speed_cap_mps"]),
                                         L.get("end_room_mps", st["leg_speed_cap_mps"]))
        if c.legs:
            route, legs = route[:c.legs], legs[:c.legs]
        return p, route, legs

    def run(self, progress=None):
        c = self.cfg
        t, k = 0.0, 0
        log = {k_: [] for k_ in ("t", "truth", "pf", "est", "phase", "z", "beams")}
        pf_err, min_clr, collided = [], math.inf, False
        self.pf.init((*self.doc["start"]["xy"], math.radians(self.doc["start"]["yaw_deg"])),
                     self.doc["start"]["sigma_xy_m"], math.radians(self.doc["start"]["sigma_yaw_deg"]), self.odom)
        last_loc_t, mean, plan, start_est = 0.0, None, None, None
        while t < c.max_time_s:
            self.t = t
            depth_tick = k % c.depth_every == 0
            img = None
            if depth_tick:
                img = self.depth_frame()
                mean, nb = self.localize(img, t - last_loc_t)
                last_loc_t = t
                pf_err.append((t, math.hypot(mean[0] - self.truth[0], mean[1] - self.truth[1])))
            if self.mission is None and t >= c.pre_arm_s and mean is not None:
                start_est = tuple(float(v) for v in mean)
                plan, route, legs = self.plan_from(mean)
                if progress and plan.get("choice_reason"):
                    progress(f"{t:7.2f}s chicane {plan['chicane']}: {plan['choice_reason']}")
                prm = C.Params(route=route, legs=legs, height_m=self.cruise_z - c.landed_z, hover_s=0.0,
                               **dict(route_params(self.doc), **(c.refnav_overrides or {})))
                if c.mode != "corrected":
                    prm.correction_max_m = 0.0                 # corrections only in the mode that flies them
                if c.nav == "path":
                    prm = self._path_setup(plan, mean, prm)
                self.mission = C.Mission(prm)
            if self.mission is not None:
                m = self.mission
                m.on_health(t, True)
                if c.nav == "path":
                    self._path_tick(t, depth_tick, mean)
                if c.mode == "corrected" and c.nav != "path" and mean is not None and self.pf.updates >= 30:
                    # loc_node's /coursenav/px4_correction: the course drift of the "VIO frame at the declared
                    # fixture" belief (not truth), as a NED shift of the remaining targets
                    nominal = compose(self.nominal_T, self.odom)
                    raw = np.array([mean[0] - nominal[0], mean[1] - nominal[1]])
                    _, kept, P = project_correction(raw, self.last_cov, c.corr_project_sigma_m)
                    travelled = 0.0 if self.last_corr_odom is None else math.hypot(*(self.odom[:2] - self.last_corr_odom[:2]))
                    self.last_corr_odom = self.odom.copy()
                    prev = self.limiter.out if self.limiter.out is not None else raw
                    lim = self.limiter.step(prev + P @ (raw - prev), travelled, c.dt)
                    sig = max(0.05, math.sqrt(max(self.last_cov[0, 0], self.last_cov[1, 1])))
                    m.on_correction(t, -lim[1], -lim[0], sig)
                v = self.vehicle()
                if img is not None:
                    m.on_depth(t - c.depth_latency_s, img, self.cam.K, self.cam.R_body_cam(),
                               np.asarray(self.cam.t_body), v.q, now=t)
                out = m.step(t, v)
                for e in out.events:
                    if progress:
                        progress(f"{t:7.2f}s {e}")
                self.px4_step(t, out, c.dt)
                if m.phase == "done":
                    break
            if self.in_dark():
                self.dark_ticks += 1
            if self.z > c.landed_z + 0.1:
                clr = self.clearance_now()
                min_clr = min(min_clr, clr)
                if clr < self.prop_radius_m and not collided:
                    collided = True
                    if progress:
                        progress(f"{t:7.2f}s COLLISION: clearance {clr:.2f} m at truth {np.round(self.truth, 2)}")
            if depth_tick:
                log["t"].append(t)
                log["truth"].append([*self.truth])
                log["pf"].append([*mean])
                log["est"].append([*self.est()])
                log["phase"].append(self.mission.phase if self.mission else "localize")
                log["z"].append(self.z)
                log["beams"].append(nb)
            t += c.dt
            k += 1
        m = self.mission
        if c.nav == "path" and plan is not None:
            plan["nav"] = dict(self.nav_stats, ekf_rejected=self.ekf.rejected, ekf_accepted=self.ekf.accepted,
                               ekf_resets=self.ekf.resets, path_waits=self.mission.counts.get("path_waits", 0)
                               if self.mission else 0)
        airborne_err = [e for (tt, e), ph in zip(pf_err, log["phase"]) if ph not in ("localize", "wait", "arm")]
        errs = np.array(airborne_err or [e for _, e in pf_err])
        vio = compose(compose(np.array([*c.spawn, math.radians(c.spawn_yaw_deg)]), inverse(np.zeros(3))), self.odom)
        pad = np.asarray(self.doc["sequence"][-1]["xy"])
        return SimResult(
            result=(m.result if m and m.result else f"timeout in phase {m.phase if m else 'localize'}"),
            t_end=round(t, 2), legs_done=m.legs_done if m else 0, legs=len(m.targets) if m else 0,
            collided=collided, min_clearance_m=round(min_clr, 3),
            pf_err_p50=round(float(np.percentile(errs, 50)), 3), pf_err_p95=round(float(np.percentile(errs, 95)), 3),
            pf_err_max=round(float(errs.max()), 3),
            vio_err_final=round(math.hypot(vio[0] - self.truth[0], vio[1] - self.truth[1]), 3),
            land_err_m=round(float(np.hypot(*(self.truth[:2] - pad))), 3),
            start_est=start_est, plan=plan, log={k_: np.asarray(v) for k_, v in log.items()},
            tunnel_ticks=int(m.counts.get("tunnel_ticks", 0)) if m else 0, dark_ticks=self.dark_ticks)


def draw(cmap, obst, blocked, res: SimResult, out_png, title=""):
    import cv2
    from .viz import TopView
    tv = TopView(obst.origin, obst.occ.shape, cmap.res, 110)
    img = tv.base(obst.occ, blocked.occ)
    if res.plan:
        xy = [w["xy"] for w in res.plan["waypoints"]]
        cv2.polylines(img, [np.array([tv.px(q) for q in xy], np.int32)], False, (180, 180, 180), 2, cv2.LINE_AA)
    for key, col in (("truth", (30, 30, 220)), ("pf", (30, 160, 30))):
        pts = res.log[key]
        if len(pts):
            cv2.polylines(img, [np.array([tv.px(q) for q in pts], np.int32)], False, col, 1, cv2.LINE_AA)
    tv.text(img, title, (10, 22), (0, 0, 0))
    tv.text(img, "grey=plan  red=truth  green=localizer", (10, 42), (60, 60, 60))
    tv.text(img, f"{res.result}; legs {res.legs_done}/{res.legs}; min clear {res.min_clearance_m:.2f} m; "
                 f"loc err p95 {res.pf_err_p95:.2f} m", (10, img.shape[0] - 10), (0, 0, 0))
    cv2.imwrite(str(out_png), img)
