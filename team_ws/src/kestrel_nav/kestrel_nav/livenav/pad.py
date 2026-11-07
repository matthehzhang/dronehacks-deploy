"""Landing-pad detector: the pad's painted rings seen by the forward camera. PURE (numpy + OpenCV).

The pad (published OBJ, measured 2026-11-07): white 5 ft disk, three dark rings 3 cm wide at
radii 0.216, 0.495 and 0.750 m, a dark cross at the centre; a grey approach ring at 0.99 m.

A level camera at 0.8 m sees the floor only beyond ~1.7 m, and a 3 cm line 3 m away is one pixel
tall, so ellipse fitting in the image is fragile. Instead the image is rectified onto the ground
plane with the known camera height and attitude (inverse perspective mapping), where the rings are
circles of KNOWN radius, and a matched filter with those radii scores candidate centres over the
part of each circle that lies on visible floor. On synthesised frames (tests) the centre error is
under 1 cm from 2.0 to 4.5 m; inside 1.7 m the pad is below the bottom of the frame.

Output: pad centre in the LEVEL BODY frame (x forward, y left) from the camera's ground point.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

RING_R = (0.216, 0.495, 0.750)
R_FLU_OPT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])    # optical -> FLU


@dataclass
class PadParams:
    res: float = 0.01                 # top-down metres per pixel
    x_range: tuple = (0.8, 5.0)       # forward extent of the rectified view
    y_half: float = 2.5
    min_score: float = 0.12
    min_valid_frac: float = 0.35
    ring_on: tuple = (0.495, 0.750)
    ring_off: tuple = (0.40, 0.62, 0.86)
    hough_param2: int = 14
    max_candidates: int = 8


def homography_floor_to_image(R_wc, t_wc, K):
    """H with image_px ~ H @ [X, Y, 1] for floor points Z = 0 (world frame with z up)."""
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc
    M = np.column_stack([R_cw[:, 0], R_cw[:, 1], t_cw])
    return K @ M


class PadDetector:
    def __init__(self, K, image_hw, params: PadParams = None):
        """K = (fx, fy, cx, cy); image_hw = (height, width) of the mono/RGB image used."""
        self.p = params or PadParams()
        fx, fy, cx, cy = K
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        self.hw = tuple(image_hw)
        self.ang = np.linspace(0, 2 * math.pi, 180, endpoint=False)

    # ------------------------------------------------------------------ geometry
    def _rectify(self, gray, cam_h, R_level_cam):
        """gray -> top-down image in the LEVEL frame whose origin is the camera's ground point.
        R_level_cam: level FLU <- camera optical (includes the mount tilt and the vehicle's roll/pitch)."""
        p = self.p
        R_wc = R_level_cam
        t_wc = np.array([0.0, 0.0, cam_h])
        H_wi = homography_floor_to_image(R_wc, t_wc, self.K)
        nx = int((p.x_range[1] - p.x_range[0]) / p.res)
        ny = int(2 * p.y_half / p.res)
        T = np.array([[0, -p.res, p.x_range[1]], [-p.res, 0, p.y_half], [0, 0, 1.0]])   # [j, i, 1] -> [x, y, 1]
        Hti = H_wi @ T
        top = cv2.warpPerspective(gray, Hti, (ny, nx), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        # pixels above the horizon are not floor: the horizon row for a pitched camera is where the
        # ray is horizontal; take the rows whose floor intersection is behind the camera as invalid.
        ones = np.full(self.hw, 255, np.uint8)
        valid = cv2.warpPerspective(ones, Hti, (ny, nx), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        # rays that hit the floor behind the camera or at infinity map to huge |x|: already outside the box
        xs = p.x_range[1] - np.arange(nx) * p.res
        valid &= (xs > 0.3)[:, None]

        def to_xy(j, i):
            return p.x_range[1] - i * p.res, p.y_half - j * p.res
        return top, valid, to_xy

    def _score(self, inv, valid, cj, ci):
        p = self.p
        res = p.res

        def sample(r_m):
            jj = np.rint(cj + (r_m / res) * np.cos(self.ang)).astype(int)
            ii = np.rint(ci + (r_m / res) * np.sin(self.ang)).astype(int)
            ok = (jj >= 0) & (jj < inv.shape[1]) & (ii >= 0) & (ii < inv.shape[0])
            ok[ok] &= valid[ii[ok], jj[ok]]
            if ok.sum() < p.min_valid_frac * len(self.ang):
                return None
            return inv[ii[ok], jj[ok]].mean()
        on = [sample(r) for r in p.ring_on]
        off = [sample(r) for r in p.ring_off]
        if any(v is None for v in on + off):
            return -1.0
        return float((np.mean(on) - np.mean(off)) / 255.0)

    # ------------------------------------------------------------------ detection
    def detect(self, gray, cam_h, R_level_cam=None):
        """-> (x_fwd, y_left, score) of the pad centre relative to the camera's ground point in the
        level frame, or None. gray: mono8 image (bgr is converted)."""
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        if R_level_cam is None:
            R_level_cam = R_FLU_OPT
        p = self.p
        top, valid, to_xy = self._rectify(gray, cam_h, R_level_cam)
        blur = cv2.GaussianBlur(top, (0, 0), 1.0)
        inv = (255 - blur).astype(np.float32)
        dark = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 25)
        dark[~valid] = 0
        cands = []
        for r_m in p.ring_on:
            r_px = r_m / p.res
            circles = cv2.HoughCircles(dark, cv2.HOUGH_GRADIENT, dp=1.5, minDist=r_px * 0.5, param1=80,
                                       param2=p.hough_param2, minRadius=int(r_px * 0.85), maxRadius=int(r_px * 1.15))
            if circles is not None:
                cands += [(float(c[0]), float(c[1])) for c in circles[0][: p.max_candidates]]
        best = None
        for cj0, ci0 in cands:
            bj, bi, bs = cj0, ci0, -1.0
            for step, span in ((4, 16), (1, 4)):
                cj_c, ci_c = bj, bi
                for dj in range(-span, span + 1, step):
                    for di in range(-span, span + 1, step):
                        sc = self._score(inv, valid, cj_c + dj, ci_c + di)
                        if sc > bs:
                            bj, bi, bs = cj_c + dj, ci_c + di, sc
            if best is None or bs > best[2]:
                best = (*to_xy(bj, bi), bs)
        if best is None or best[2] < p.min_score:
            return None
        return best


def level_cam_rotation(R_body_cam, roll=0.0, pitch=0.0):
    """Level FLU <- camera optical, for a camera mounted with body-FLU <- optical rotation
    R_body_cam on a vehicle at the given roll and pitch (yaw removed)."""
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Ry @ Rx @ np.asarray(R_body_cam, float)
