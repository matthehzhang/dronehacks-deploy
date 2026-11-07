"""Synthetic camera frames of the landing pad for OFFLINE tests of the ring detector. PURE.

The floor pattern comes from the published OBJ / MTL (measured 2026-11-07): floor grey 0.918, white
5 ft pad (0.980, r 0.762), three dark rings (0.122) at r 0.202-0.230, 0.479-0.511, 0.735-0.764, a dark
cross, and a grey approach ring (0.451) at r 0.975-1.006. The planar floor is pushed through the exact
ground-plane homography of a pinhole camera, then blurred and noised. Nothing above the floor is
rendered (a flat grey "wall" fills the view above the horizon). This is a PROXY for the simulator's
rendering, not a substitute for a real frame.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .pad import R_FLU_OPT, homography_floor_to_image

RINGS = [(0.202, 0.230), (0.479, 0.511), (0.735, 0.764)]
PAD_R = 0.762


class PadFloor:
    def __init__(self, pad_xy, res=0.005, half=1.6):
        self.pad_xy = np.asarray(pad_xy, float)
        self.res, self.half = res, half
        n = int(2 * half / res)
        img = np.full((n, n), 0.918, np.float32)
        c = (n // 2, n // 2)
        px = lambda r: int(round(r / res))    # noqa: E731
        cv2.circle(img, c, px(1.006), 0.451, -1)
        cv2.circle(img, c, px(0.975), 0.918, -1)
        cv2.circle(img, c, px(PAD_R), 0.980, -1)
        for lo, hi in RINGS:
            cv2.circle(img, c, px(hi), 0.122, -1)
            cv2.circle(img, c, px(lo), 0.980, -1)
        cv2.rectangle(img, (c[0] - px(0.102), c[1] - px(0.018)), (c[0] + px(0.102), c[1] + px(0.018)), 0.122, -1)
        cv2.rectangle(img, (c[0] - px(0.018), c[1] - px(0.102)), (c[0] + px(0.018), c[1] + px(0.102)), 0.122, -1)
        self.tex = img

    def render(self, K, image_hw, cam_xy, cam_h, yaw, blur=1.2, noise=0.01, rng=None):
        """mono8 frame from a level camera at (cam_xy, cam_h) looking along `yaw` (world frame z up)."""
        fx, fy, cx, cy = K
        Kmat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        H, W = image_hw
        c, s = math.cos(yaw), math.sin(yaw)
        R_wb = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        R_wc = R_wb @ R_FLU_OPT
        t_wc = np.array([cam_xy[0], cam_xy[1], cam_h])
        H_wi = homography_floor_to_image(R_wc, t_wc, Kmat)
        T = np.array([[self.res, 0, self.pad_xy[0] - self.half], [0, -self.res, self.pad_xy[1] + self.half], [0, 0, 1.0]])
        img = cv2.warpPerspective(self.tex, H_wi @ T, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0.918)
        img[: int(cy) + 1, :] = 0.75
        img = cv2.GaussianBlur(img, (0, 0), blur)
        rng = rng or np.random.default_rng(0)
        img = np.clip(img + rng.normal(0, noise, img.shape), 0, 1)
        return (img * 255).astype(np.uint8)
