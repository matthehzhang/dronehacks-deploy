"""Synthetic OAK depth from the published course map. PURE (numpy). OFFLINE TESTING ONLY.

Used by the tilt study and the offline closed-loop simulation. It is a geometry model of the
airframe's own forward depth camera, not an extra sensor, and flight code never imports it.

Assumptions (all from HANDOFF.md section 3, or labelled as guesses):
  * 640x400, fx = fy = 421, principal point at the centre; min range 0.4 m.
  * Mount: base_link <- oak_left at (0.116, 0.037, 0.029), level, facing forward; `tilt_deg`
    pitches it nose-down for the tilt study.
  * Stereo error (GUESS): sigma = 0.005 + 0.004 z^2 (0.1 px disparity, 7.5 cm baseline); 2 % of
    pixels return nothing.
  * The net (GUESS, unmeasured): 2 mm strands at 51 mm pitch. A pixel whose ray reaches the net
    returns the net with probability net_return_p; otherwise it returns the room behind it
    (net_room_gap_m further on, not in the map) or, with net_dropout, nothing.
  * Voxels are 5 cm: ranges are voxel-boundary ranges (SIM_SENSORS.md has the same caveat).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .objmap import CourseMap

# REP-103 optical (x right, y down, z forward) -> body FLU, for a level forward camera
R_FLU_OPT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


@dataclass
class Camera:
    width: int = 640
    height: int = 400
    fx: float = 421.0
    fy: float = 421.0
    cx: float = 319.5
    cy: float = 199.5
    t_body: tuple = (0.116, 0.037, 0.029)    # base_link <- oak_left (live TF, 2026-11-05)
    tilt_deg: float = 0.0                    # nose-down pitch of the mount
    min_m: float = 0.4
    max_m: float = 9.0

    @property
    def K(self):
        return (self.fx, self.fy, self.cx, self.cy)

    def R_body_cam(self):
        """body FLU <- optical."""
        a = math.radians(self.tilt_deg)
        c, s = math.cos(a), math.sin(a)
        Ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])     # +a about +y_FLU = nose down
        return Ry @ R_FLU_OPT

    @property
    def hfov_deg(self):
        return math.degrees(2 * math.atan2(self.width / 2, self.fx))

    @property
    def vfov_deg(self):
        return math.degrees(2 * math.atan2(self.height / 2, self.fy))


@dataclass
class DepthModel:
    sigma0: float = 0.005
    sigma_k: float = 0.004
    dropout: float = 0.02
    net_return_p: float = 0.3
    net_room_gap_m: float = 2.0
    net_dropout: bool = False
    noise: bool = True


def body_pose_matrix(x, y, z, yaw, roll=0.0, pitch=0.0):
    """world ENU <- body FLU rotation and translation (yaw about z, then pitch, then roll)."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
    Ry = np.array([[cp, 0, sp], [0, 1.0, 0], [-sp, 0, cp]])
    Rx = np.array([[1.0, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx, np.array([x, y, z], float)


class DepthRenderer:
    """Renders on a stride grid (every `stride`-th pixel in u and v, starting at 0) and repeats
    each sample over its stride x stride block, so consumers that sample multiples of the stride
    (refnav: stride 8, border 24; coursenav.scan: stride 4, border 16) read exact ray casts."""

    def __init__(self, cmap: CourseMap, cam: Camera = None, model: DepthModel = None, stride=4, seed=0):
        self.cmap, self.cam, self.model, self.stride = cmap, cam or Camera(), model or DepthModel(), int(stride)
        self.rng = np.random.default_rng(seed)
        c = self.cam
        vs, us = np.mgrid[0:c.height:self.stride, 0:c.width:self.stride]
        self.grid_shape = vs.shape
        self.rays_opt = np.stack([(us.ravel() - c.cx) / c.fx, (vs.ravel() - c.cy) / c.fy,
                                  np.ones(us.size)], axis=1)          # optical, z = 1
        self.ray_norm = np.linalg.norm(self.rays_opt, axis=1)
        net_ids = [k + 1 for k, g in enumerate(cmap.groups) if g.startswith("Net")]
        self.net_label = np.isin(cmap.labels, net_ids)
        cmap.distance_field()

    def camera_world(self, R_wb, t_wb):
        Rbc = self.cam.R_body_cam()
        return R_wb @ Rbc, t_wb + R_wb @ np.asarray(self.cam.t_body, float)

    def render(self, R_wb, t_wb, with_net=False, with_hits=False):
        """-> depth (H, W) float32 metres along the optical axis, NaN = no return
        (and, with with_net, a bool (H, W) of pixels whose ray reached the net; with with_hits, the
        (H, W, 2) xy cell index each pixel's ray hit, -1 where none)."""
        c, m = self.cam, self.model
        R_wc, o = self.camera_world(R_wb, t_wb)
        dirs = self.rays_opt @ R_wc.T
        dist, ijk = self.cmap.raycast_fast(np.repeat(o[None], len(dirs), 0), dirs, c.max_m)
        hit = np.isfinite(dist)
        net = np.zeros(len(dist), bool)
        net[hit] = self.net_label[ijk[hit, 0], ijk[hit, 1], ijk[hit, 2]]
        z = dist / self.ray_norm                                       # range along the ray -> optical depth
        n = len(z)
        if net.any():
            through = net & (self.rng.random(n) >= m.net_return_p)
            z = np.where(through, np.nan if m.net_dropout else z + m.net_room_gap_m / self.ray_norm, z)
        if m.noise:
            z = z + self.rng.normal(0.0, 1.0, n) * (m.sigma0 + m.sigma_k * np.nan_to_num(z) ** 2)
            z[self.rng.random(n) < m.dropout] = np.nan
        z[~(z >= c.min_m)] = np.nan
        img = self._expand(z.reshape(self.grid_shape))
        extra = []
        if with_net:
            extra.append(self._expand(net.reshape(self.grid_shape)))
        if with_hits:
            ij = np.where(hit[:, None], ijk[:, :2], -1).reshape(*self.grid_shape, 2)
            s_ = self.stride
            extra.append(np.repeat(np.repeat(ij, s_, axis=0), s_, axis=1)[:c.height, :c.width])
        return (img, *extra) if extra else img

    def _expand(self, a):
        s, c = self.stride, self.cam
        return np.repeat(np.repeat(a, s, axis=0), s, axis=1)[:c.height, :c.width].astype(
            np.float32 if a.dtype != bool else bool)
