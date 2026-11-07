"""Top-down course images with OpenCV (shared by the plan, scan and localizer tools)."""
from __future__ import annotations

import cv2
import numpy as np


class TopView:
    def __init__(self, origin_xy, shape_ij, res, px_per_m=110):
        self.o = np.asarray(origin_xy, float)
        self.w_m = shape_ij[0] * res
        self.h_m = shape_ij[1] * res
        self.s = px_per_m
        self.res = res
        self.W, self.H = int(self.w_m * px_per_m), int(self.h_m * px_per_m)

    def px(self, xy):
        x, y = xy[0], xy[1]
        return (int(round((x - self.o[0]) * self.s)), int(round(self.H - (y - self.o[1]) * self.s)))

    def base(self, occ, inflated=None):
        img = np.full((self.H, self.W, 3), 255, np.uint8)

        def paint(mask, col):
            m = cv2.resize(np.flipud(mask.T).astype(np.uint8), (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            img[m > 0] = col

        if inflated is not None:
            paint(inflated, (225, 225, 225))
        paint(occ, (60, 60, 60))
        for v in np.arange(np.ceil(self.o[0]), self.o[0] + self.w_m, 1.0):
            cv2.line(img, self.px((v, self.o[1])), self.px((v, self.o[1] + self.h_m)), (235, 235, 235), 1)
        for v in np.arange(np.ceil(self.o[1]), self.o[1] + self.h_m, 1.0):
            cv2.line(img, self.px((self.o[0], v)), self.px((self.o[0] + self.w_m, v)), (235, 235, 235), 1)
        return img

    def label(self, img, text, xy, col):
        p = self.px(xy)
        cv2.putText(img, text, (p[0] + 4, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

    @staticmethod
    def text(img, text, org, col):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
