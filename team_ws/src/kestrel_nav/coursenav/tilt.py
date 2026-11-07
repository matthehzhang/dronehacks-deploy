"""OAK down-tilt study: geometry only (no texture, lighting or stereo matching). PURE + OFFLINE.

For each candidate nose-down tilt, fly the camera along a planned path (heading = leg direction,
as refnav flies it) through the published map and score what the flight code would get:

  scan_beams    bins of the localizer's horizontal scan (coursenav.scan, 48 bins, 0.3-1.6 m band)
                that return: more = better map matching.
  corridor_ok   refnav's own gate for moving: of the rays crossing the leg corridor 1.5 m ahead
                (stop_m + lead_max_m; +-corridor_half_m, +-band_m), >= 60 % return depth. If the
                corridor leaves the image, refnav's carrot never advances.
  swept_vis     fraction of the drone's swept cross-section 1 m ahead (+-0.28 m wide, -0.05 to
                +0.25 m tall) inside the image: can it see what it is about to fly into?
  floor_blind_m horizontal distance ahead before the bottom of the image meets the floor.
  pad           on the final leg: the closest distance at which the pad centre is still in view.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from refnav import depth as D

from .render import Camera, DepthModel, DepthRenderer, body_pose_matrix
from .scan import depth_points_level, scan_from_points

LEVEL_FRD_FLU = D.level_rotation((1.0, 0.0, 0.0, 0.0))      # level FRD <- body FLU, level attitude


@dataclass
class RefnavGate:
    corridor_half_m: float = 0.35         # course route files
    band_m: float = 0.4
    at_m: float = 1.5                     # stop_m 0.9 + lead_max_m 0.6
    min_coverage: float = 0.6
    stride: int = 8
    border_px: int = 24


def sample_path(waypoints, step=0.25):
    """[(x, y)] waypoints -> (n, 3) samples (x, y, heading of their leg) and the leg index."""
    out, legs = [], []
    for k in range(len(waypoints) - 1):
        a, b = np.asarray(waypoints[k], float), np.asarray(waypoints[k + 1], float)
        L = float(np.linalg.norm(b - a))
        if L < 1e-6:
            continue
        yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        for s in np.arange(0.0, L, step):
            p = a + (b - a) * s / L
            out.append((p[0], p[1], yaw))
            legs.append(k)
    return np.asarray(out), np.asarray(legs)


def swept_visibility(cam: Camera, ahead=1.0, half_w=0.28, zlo=-0.05, zhi=0.25):
    """Fraction of a grid on the swept cross-section `ahead` m in front of the body inside the image."""
    ys, zs = np.meshgrid(np.linspace(-half_w, half_w, 15), np.linspace(zlo, zhi, 9))
    p_body = np.stack([np.full(ys.size, ahead), ys.ravel(), zs.ravel()], axis=1)
    p_cam = (p_body - np.asarray(cam.t_body)) @ cam.R_body_cam()          # optical
    z = p_cam[:, 2]
    u = cam.fx * p_cam[:, 0] / z + cam.cx
    v = cam.fy * p_cam[:, 1] / z + cam.cy
    inside = (z > cam.min_m) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    top = inside.reshape(zs.shape)[-1]
    return float(inside.mean()), bool(top.all())


def floor_blind(cam: Camera, body_z):
    """Horizontal distance (from the body) at which the image's bottom edge meets the floor."""
    down = math.radians(cam.tilt_deg + cam.vfov_deg / 2)
    if down >= math.pi / 2:
        return 0.0
    hc = body_z + cam.t_body[2]
    return cam.t_body[0] + hc / math.tan(down) if down > 0 else math.inf


def pad_closest_visible(renderer: DepthRenderer, approach_start, pad_xy, body_z, step=0.05):
    """Fly straight from approach_start to the pad centre; -> the smallest distance (m) at which the
    pad centre is in the image and unoccluded (inf: never seen)."""
    cam = renderer.cam
    a, b = np.asarray(approach_start, float), np.asarray(pad_xy, float)
    yaw = math.atan2(b[1] - a[1], b[0] - a[0])
    L = float(np.linalg.norm(b - a))
    best = math.inf
    target = np.array([b[0], b[1], 0.0])
    for s in np.arange(0.0, L, step):
        xy = a + (b - a) * s / L
        R_wb, t_wb = body_pose_matrix(xy[0], xy[1], body_z, yaw)
        R_wc, o = renderer.camera_world(R_wb, t_wb)
        p_cam = R_wc.T @ (target - o)
        if p_cam[2] <= cam.min_m:
            continue
        u, v = cam.fx * p_cam[0] / p_cam[2] + cam.cx, cam.fy * p_cam[1] / p_cam[2] + cam.cy
        if not (0 <= u < cam.width and 0 <= v < cam.height):
            continue
        rng = float(np.linalg.norm(target - o))
        d, _ = renderer.cmap.raycast_fast(o[None], (target - o)[None], rng + 0.2)
        if np.isfinite(d[0]) and d[0] >= rng - 0.1:
            best = min(best, float(np.linalg.norm(b - xy)))
    return best


def study(cmap, waypoints, tilts, body_z, pad_xy=None, step=0.25, gate=RefnavGate(), seed=0, n_bins=48):
    """-> list of per-tilt dicts (see the module docstring)."""
    samples, _ = sample_path(waypoints, step)
    rows = []
    for tilt in tilts:
        cam = Camera(tilt_deg=float(tilt))
        r = DepthRenderer(cmap, cam, DepthModel(noise=False), seed=seed)
        beams, cov_ok, cov = [], [], []
        for x, y, yaw in samples:
            R_wb, t_wb = body_pose_matrix(x, y, body_z, yaw)
            img = r.render(R_wb, t_wb)
            pts = depth_points_level(img, cam.K, cam.R_body_cam(), cam.t_body, np.eye(3))
            beams.append(len(scan_from_points(pts, body_z, n_bins=n_bins)))
            f = D.DepthFrame(img, cam.K, cam.R_body_cam(), cam.t_body, LEVEL_FRD_FLU,
                             stride=gate.stride, border_px=gate.border_px)
            c, n = f.corridor_coverage(0.0, gate.corridor_half_m, gate.band_m, gate.at_m)
            cov.append(c if n else 0.0)
            cov_ok.append(n > 0 and c >= gate.min_coverage)
        sv, top = swept_visibility(cam)
        row = {"tilt_deg": float(tilt),
               "scan_beams_mean": round(float(np.mean(beams)), 1),
               "scan_beams_p10": round(float(np.percentile(beams, 10)), 1),
               "scan_beams_min": int(np.min(beams)),
               "corridor_ok_frac": round(float(np.mean(cov_ok)), 3),
               "corridor_cov_mean": round(float(np.mean(cov)), 3),
               "swept_vis_1m": round(sv, 3), "swept_top_1m_visible": top,
               "floor_blind_m": round(floor_blind(cam, body_z), 2)}
        if pad_xy is not None and len(waypoints) >= 2:
            row["pad_closest_visible_m"] = round(pad_closest_visible(r, waypoints[-2], pad_xy, body_z), 2)
        rows.append(row)
    return rows
