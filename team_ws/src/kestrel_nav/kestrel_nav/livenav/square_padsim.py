"""Synthetic colour frames of the venue's square pad for tests and the offline sim. PURE (numpy)."""
import math

import numpy as np

R_FLU_OPT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])   # body FLU <- optical
DRONE_K = np.array([[637.9, 0, 529.4], [0, 637.9, 338.3], [0, 0, 1.0]])         # skybeta-14 left mono
DRONE_HW = (640, 1024)


def pitch_rot(pitch_down_rad):
    c, s = math.cos(pitch_down_rad), math.sin(pitch_down_rad)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])        # rotate FLU about +y: nose down


def render(K, hw, cam_h, pad_xy, side=0.64, pad_yaw=0.0, R_level_cam=None, corner_r=0.06, h_mark=True,
           floor_bgr=(70, 85, 190), pad_bgr=(240, 168, 60), seed=0):
    """BGR image from a camera `cam_h` above the floor; pad centre at level-body (x fwd, y left)."""
    R = R_FLU_OPT if R_level_cam is None else R_level_cam
    h, w = hw
    vs, us = np.mgrid[0:h, 0:w]
    rays = np.stack([us, vs, np.ones_like(us)], -1).reshape(-1, 3).astype(float) @ np.linalg.inv(K).T @ R.T
    down = -rays[:, 2]
    s = np.where(down > 1e-6, cam_h / np.maximum(down, 1e-9), np.inf)
    gx, gy = rays[:, 0] * s - pad_xy[0], rays[:, 1] * s - pad_xy[1]
    c, sn = math.cos(-pad_yaw), math.sin(-pad_yaw)
    with np.errstate(invalid="ignore"):
        px, py = c * gx - sn * gy, sn * gx + c * gy
    a = side / 2 - corner_r
    dx, dy = np.maximum(np.abs(px) - a, 0), np.maximum(np.abs(py) - a, 0)
    inside = (np.hypot(dx, dy) <= corner_r) & np.isfinite(s)
    rng = np.random.default_rng(seed)
    img = np.empty((h * w, 3))
    img[:] = floor_bgr
    img[inside] = pad_bgr
    if h_mark:                                       # tan H, ~0.25 of the side
        u = side * 0.25
        H = inside & (((np.abs(px) < u / 2) & (np.abs(py) < u * 0.12)) |
                      ((np.abs(np.abs(py) - u * 0.45) < u * 0.09) & (np.abs(px) < u / 2)))
        img[H] = (180, 186, 195)
    img += rng.normal(0, 6, img.shape)
    img[~np.isfinite(s) | (down <= 1e-6)] = (200, 200, 200)
    return np.clip(img, 0, 255).astype(np.uint8).reshape(h, w, 3)
