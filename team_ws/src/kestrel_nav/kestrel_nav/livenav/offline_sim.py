"""Offline closed loop for livenav: the REAL LiveMission against a kinematic vehicle, a drifting VIO
and depth ray-cast from the published map. OFFLINE TESTING ONLY: no ROS, no PX4, nothing commanded.

What is modelled (GUESSES, labelled):
  * vehicle: first-order lag (tau_v) on the commanded velocity, yaw-rate lag (tau_yaw); level
    attitude; no wind.  PX4 overshoot on position setpoints is NOT relevant: livenav commands velocity.
  * VIO: scale error + heading drift per metre + a random walk (numbers from flight D, 2026-11-06).
  * depth: coursenav.render (stereo noise, net model, 5 cm voxels) at 15 Hz with latency.
  * the world may differ from the map (unmapped objects, a moved wall) for the live grid to find.
  * collision: the body is a disk of prop-tip radius checked against the WORLD obstacle band.

Run:  python3 -m kestrel_nav.livenav.offline_sim [--png out.png] [--unmapped x,y,r,h ...] [--v 0.5]
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..coursenav.objmap import CourseMap
from ..coursenav.render import Camera, DepthModel, DepthRenderer, body_pose_matrix
from ..coursenav.scan import depth_points_level
from .follower import FollowParams, wrap
from .mission import Command, LiveMission, MissionParams, Observation
from .navgrid import NavParams
from .pad import PadDetector
from .padsim import PadFloor
from .replanner import ReplanParams


def world_map(cmap: CourseMap, unmapped=(), shift_group="", shift_m=(0.0, 0.0)) -> CourseMap:
    """The simulated WORLD: the published map plus as-built differences only the renderer and the
    collision check see. unmapped: ((x, y, radius, height), ...) cylinders."""
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
    return CourseMap(occ, cmap.origin.copy(), cmap.res, lab, list(cmap.groups), dict(cmap.meta, world="offline"))


@dataclass
class SimConfig:
    spawn: tuple = (-2.5146, 1.1144)        # sim spawn: 0.20 m north of the launch mark
    spawn_yaw_deg: float = 90.0
    start_err_xy: tuple = (0.0, 0.0)        # where the vehicle REALLY is vs where the mission assumes
    start_err_yaw_deg: float = 0.0
    landed_z: float = 0.067                 # body origin above the floor when landed (sim airframe)
    tau_v: float = 0.45                     # velocity response lag (GUESS: PX4 MPC at 1 m/s^2)
    tau_yaw: float = 0.3
    vio_scale_err: float = 0.02
    vio_yaw_drift_deg_per_m: float = 0.3
    vio_walk_m_per_sqrt_m: float = 0.01
    depth_hz: float = 15.0
    depth_latency_s: float = 0.04
    tick_hz: float = 10.0
    max_time_s: float = 200.0
    prop_radius_m: float = 0.28
    unmapped: tuple = ()
    shift_group: str = ""
    shift_m: tuple = (0.0, 0.0)
    depth: DepthModel = field(default_factory=DepthModel)
    camera: Camera = field(default_factory=Camera)
    pad_camera: bool = True                 # synthesise mono frames of the pad and run the ring detector
    pad_hz: float = 4.0
    seed: int = 0


@dataclass
class SimResult:
    result: str
    t_end: float
    goals_done: int
    goals: int
    collided: bool
    min_clearance_m: float
    vio_err_final_m: float
    land_err_m: float
    replans: int
    path_len_m: float
    truth: np.ndarray
    log: list
    states: list
    pad_runs: int = 0
    pad_hits: int = 0
    pad_est_err_m: float = math.nan


class OfflineFlight:
    def __init__(self, mission: LiveMission, cfg: SimConfig = None):
        self.m = mission
        self.cfg = cfg or SimConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.world = world_map(mission.cmap, self.cfg.unmapped, self.cfg.shift_group, self.cfg.shift_m)
        z = mission.p.cruise_z_m
        band = mission.doc.get("plan_band_m", 0.25)
        self.world_band = self.world.band(z - band, z + band)
        from scipy import ndimage
        self.world_dist = ndimage.distance_transform_edt(~self.world_band) * self.world.res
        self.renderer = DepthRenderer(self.world, self.cfg.camera, self.cfg.depth, seed=self.cfg.seed)
        # truth state (course frame)
        sx, sy = self.cfg.spawn
        self.x = sx + self.cfg.start_err_xy[0]
        self.y = sy + self.cfg.start_err_xy[1]
        self.z = self.cfg.landed_z
        self.yaw = math.radians(self.cfg.spawn_yaw_deg + self.cfg.start_err_yaw_deg)
        self.v = np.zeros(2)
        self.yaw_rate = 0.0
        # VIO: starts at the truth pose (odom origin = power-on pose); the mission assumes odom
        # origin = course start pose, so the start error above is what the mission does not know.
        self.vio_xy = np.zeros(2)
        self.vio_yaw = 0.0
        self.vio_z = 0.0
        self.odom0 = (self.x, self.y, self.yaw)
        self.dist_flown = 0.0
        cam = self.cfg.camera
        self.pad_floor = PadFloor(np.asarray(mission.doc["sequence"][-1]["xy"], float))
        self.pad_det = PadDetector(cam.K, (cam.height, cam.width))
        self.pad_runs = 0
        self.pad_hits = 0

    # ------------------------------------------------------------------ models
    def clearance_now(self):
        ij = self.world.xy_to_ij((self.x, self.y))
        if not (0 <= ij[0] < self.world_dist.shape[0] and 0 <= ij[1] < self.world_dist.shape[1]):
            return 0.0
        return float(self.world_dist[ij[0], ij[1]])

    def depth_points(self):
        """(n,3) level-body points from a rendered depth frame at the truth pose, and the body height."""
        R_wb, t_wb = body_pose_matrix(self.x, self.y, self.z, self.yaw)
        img = self.renderer.render(R_wb, t_wb)
        cam = self.cfg.camera
        pts = depth_points_level(img, cam.K, cam.R_body_cam(), cam.t_body, np.eye(3), stride=4, border_px=16)
        return pts, self.z + self.rng.normal(0.0, 0.02)

    def move(self, cmd: Command, dt):
        cfg = self.cfg
        # commanded odom-frame velocity -> course frame (odom axes = body FLU at power-on)
        c, s = math.cos(self.odom0[2]), math.sin(self.odom0[2])
        v_cmd = np.array([c * cmd.vx - s * cmd.vy, s * cmd.vx + c * cmd.vy])
        self.v += (v_cmd - self.v) * min(1.0, dt / cfg.tau_v)
        self.yaw_rate += (cmd.yaw_rate - self.yaw_rate) * min(1.0, dt / cfg.tau_yaw)
        d = self.v * dt
        self.x += d[0]
        self.y += d[1]
        self.yaw = wrap(self.yaw + self.yaw_rate * dt)
        self.z = max(cfg.landed_z, self.z + cmd.vz * dt)
        step = float(np.linalg.norm(d))
        self.dist_flown += step
        # VIO drift: the measured displacement in the VIO frame, scaled and rotated by the drift
        c0, s0 = math.cos(-self.odom0[2]), math.sin(-self.odom0[2])
        d_odom = np.array([c0 * d[0] - s0 * d[1], s0 * d[0] + c0 * d[1]])
        drift_yaw = math.radians(cfg.vio_yaw_drift_deg_per_m) * self.dist_flown
        cd, sd = math.cos(drift_yaw), math.sin(drift_yaw)
        d_meas = (1.0 + cfg.vio_scale_err) * np.array([cd * d_odom[0] - sd * d_odom[1], sd * d_odom[0] + cd * d_odom[1]])
        d_meas += self.rng.normal(0, cfg.vio_walk_m_per_sqrt_m * math.sqrt(step), 2) if step > 0 else 0.0
        self.vio_xy += d_meas
        self.vio_yaw = wrap(self.yaw - self.odom0[2] + drift_yaw)
        self.vio_z = self.z - cfg.landed_z

    def truth_pose(self):
        return (self.x, self.y, self.yaw)

    def pad_fix(self):
        """Synthesise the left mono frame at the truth pose and run the ring detector -> body-frame fix."""
        cam = self.cfg.camera
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        cam_xy = (self.x + c * cam.t_body[0] - s * cam.t_body[1], self.y + s * cam.t_body[0] + c * cam.t_body[1])
        if math.dist(cam_xy, self.pad_floor.pad_xy) > 6.0:
            return None
        img = self.pad_floor.render(cam.K, (cam.height, cam.width), cam_xy, self.z + cam.t_body[2], self.yaw, rng=self.rng)
        self.pad_runs += 1
        det = self.pad_det.detect(img, self.z + cam.t_body[2])
        if det is None:
            return None
        self.pad_hits += 1
        return (det[0] + cam.t_body[0], det[1] + cam.t_body[1])

    # ------------------------------------------------------------------ run
    def run(self, progress=None) -> SimResult:
        cfg, m = self.cfg, self.m
        dt = 1.0 / cfg.tick_hz
        depth_period = 1.0 / cfg.depth_hz
        t = 0.0
        next_depth = 0.0
        next_pad = 0.0
        pad_fix, pad_t = None, -math.inf
        pts, h, pts_t = None, 0.8, -math.inf
        truth, log, states = [], [], []
        min_clear = math.inf
        collided = False
        offboard = False
        result = "timeout"
        land_err = math.nan
        while t <= cfg.max_time_s:
            if t >= 1.0:
                offboard = True
            if t >= next_depth:
                (pts, h), pts_t = self.depth_points(), t - cfg.depth_latency_s
                next_depth += depth_period
            if cfg.pad_camera and t >= next_pad and m.state in ("FLY", "PAD") and m.replanner.k >= len(m.goals) - 2:
                fix = self.pad_fix()
                if fix is not None:
                    pad_fix, pad_t = fix, t
                next_pad += 1.0 / cfg.pad_hz
            obs = Observation(now=t, odom_xyz=(float(self.vio_xy[0]), float(self.vio_xy[1]), self.vio_z),
                              odom_yaw=self.vio_yaw, offboard_active=offboard, pts_level=pts, height_m=h, pts_t=pts_t,
                              pad_fix=pad_fix, pad_fix_t=pad_t)
            cmd = m.step(obs)
            self.move(cmd, dt)
            clr = self.clearance_now()
            if m.state in ("FLY", "PAD", "HOLD", "DESCEND") and self.z > 0.3:
                min_clear = min(min_clear, clr)
            truth.append((t, self.x, self.y, self.z, self.yaw, m.state, cmd.speed, clr))
            if states and states[-1][1] != m.state or not states:
                states.append((t, m.state, cmd.note))
            if clr < cfg.prop_radius_m and self.z > 0.3:
                collided = True
                result = f"collision at ({self.x:.2f}, {self.y:.2f}) in {m.state} toward {cmd.goal}"
                break
            if cmd.land:
                result = "landed"
                pad = np.asarray(m.doc["sequence"][-1]["xy"], float)
                land_err = math.dist((self.x, self.y), pad)
                break
            if m.state == "PAUSED":
                result = "paused"
                break
            if progress and int(t * cfg.tick_hz) % int(cfg.tick_hz * 5) == 0:
                progress(t, m, cmd)
            t += dt
        vio_err = math.dist((self.x, self.y), self._vio_in_course())
        tr = np.asarray(truth, dtype=object)
        xy = np.array([(r[1], r[2]) for r in truth])
        path_len = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0
        pad_err = math.nan
        if m.pad_est is not None:
            pad_err = math.dist(m.pad_est, np.asarray(m.doc["sequence"][-1]["xy"], float))
        return SimResult(result=result, t_end=t, goals_done=m.replanner.k, goals=len(m.goals), collided=collided,
                         min_clearance_m=min_clear, vio_err_final_m=vio_err, land_err_m=land_err,
                         replans=m.replanner.replans, path_len_m=path_len, truth=tr, log=log, states=states,
                         pad_runs=self.pad_runs, pad_hits=self.pad_hits, pad_est_err_m=pad_err)

    def _vio_in_course(self):
        """Where the mission believes the vehicle is, in the course frame (for scoring only)."""
        if self.m.T is None:
            return (self.x, self.y)
        return self.m.odom_to_course(self.vio_xy, self.vio_yaw)[:2]


def draw(m: LiveMission, res: SimResult, out_png, world: CourseMap = None):
    import cv2
    from ..coursenav.viz import TopView
    g = m.grid
    tv = TopView(g.obst.origin, g.obst.occ.shape, g.obst.res)
    world_band = world.band(m.p.cruise_z_m - 0.25, m.p.cruise_z_m + 0.25) if world is not None else g.obst.occ
    img = tv.base(world_band, g.blocked.occ)
    live = g.live.occupied()
    for i, j in zip(*np.nonzero(live)):
        cv2.circle(img, tv.px(g.obst.xy(np.array([i, j]))), 2, (0, 0, 255), -1)
    for gl in m.goals:
        cv2.circle(img, tv.px(gl.xy), 4, (200, 0, 200), 1)
        tv.label(img, gl.name, gl.xy, (200, 0, 200))
    pts = [tv.px((r[1], r[2])) for r in res.truth]
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(img, a, b, (0, 140, 0), 2)
    tv.text(img, f"{res.result}  t={res.t_end:.0f}s  goals {res.goals_done}/{res.goals}  "
                 f"min clr {res.min_clearance_m:.2f} m  replans {res.replans}", (8, 18), (0, 0, 0))
    cv2.imwrite(out_png, img)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default="")
    ap.add_argument("--v", type=float, default=0.4)
    ap.add_argument("--unmapped", nargs="*", default=[], help="x,y,r,h cylinders not in the map")
    ap.add_argument("--start-err", nargs=3, type=float, default=(0.0, 0.0, 0.0), metavar=("DX", "DY", "DYAW_DEG"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chicane", default="auto")
    a = ap.parse_args(argv)
    unmapped = tuple(tuple(float(v) for v in s.split(",")) for s in a.unmapped)
    cfg = SimConfig(unmapped=unmapped, start_err_xy=(a.start_err[0], a.start_err[1]), start_err_yaw_deg=a.start_err[2],
                    seed=a.seed)
    t0 = time.time()
    m = LiveMission(MissionParams(landed_z_m=cfg.landed_z, chicane=a.chicane), follow_params=FollowParams(v_cruise=a.v))
    print(f"plan: {m.plan.get('chicane')} {len(m.goals)} goals, {m.plan['length_m']} m ({m.plan.get('reason', '')[:80]})  [{time.time() - t0:.1f}s]")
    f = OfflineFlight(m, cfg)

    def prog(t, m, cmd):
        print(f"  t={t:5.1f} {m.state:7s} goal={cmd.goal:14s} v={cmd.speed:.2f} {cmd.note}")
    res = f.run(progress=prog)
    print(f"RESULT {res.result}: t={res.t_end:.1f}s goals {res.goals_done}/{res.goals} min clearance {res.min_clearance_m:.2f} m "
          f"VIO err {res.vio_err_final_m:.2f} m land err {res.land_err_m:.2f} m replans {res.replans} path {res.path_len_m:.1f} m "
          f"pad detections {res.pad_hits}/{res.pad_runs} (est err {res.pad_est_err_m:.2f} m)")
    for t, s, note in res.states:
        print(f"    {t:6.1f} -> {s} {note}")
    if a.png:
        draw(m, res, a.png, f.world)
        print("wrote", a.png)
    return 0 if res.result == "landed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
