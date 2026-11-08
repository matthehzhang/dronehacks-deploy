"""Overall room (enclosure) limits as a hard planning constraint. PURE: no ROS.

Nets may be invisible to the depth camera, so the enclosure boundary cannot
come from measured depth. It is declared separately, in the course frame, as
one simple polygon: the inside faces of the perimeter nets. Only the outer
enclosure is represented. No internal obstacle, wall, buffer or passage from
the origin map participates; those come from measured depth alone.

The planner treats the polygon edge like a surface: the aircraft footprint
plus margin must stay inside, independently of what depth reports there.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


class RoomBoundary:
    def __init__(self, polygon, source=''):
        poly = np.asarray(polygon, float)
        if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3 or not np.isfinite(poly).all():
            raise ValueError('room boundary must be >= 3 finite XY vertices')
        if np.allclose(poly[0], poly[-1]):
            poly = poly[:-1]
        area = 0.5*float(np.sum(poly[:, 0]*np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1)*poly[:, 1]))
        if abs(area) < 1.:
            raise ValueError('room boundary encloses less than 1 m^2')
        self.poly = poly
        self.source = str(source)

    @classmethod
    def load(cls, path):
        doc = json.loads(Path(path).read_text())
        return cls(doc['polygon_xy'], doc.get('source', ''))

    def contains(self, xy):
        """Even-odd point-in-polygon for (..., 2) points; boundary points count as inside."""
        p = np.asarray(xy, float)
        x, y = p[..., 0], p[..., 1]
        inside = np.zeros(p.shape[:-1], bool)
        a, b = self.poly, np.roll(self.poly, -1, axis=0)
        for (x0, y0), (x1, y1) in zip(a, b):
            crosses = (y0 > y) != (y1 > y)
            with np.errstate(divide='ignore', invalid='ignore'):
                xc = x0 + (y - y0)*(x1 - x0)/(y1 - y0)
            inside ^= crosses & (x < xc)
        return inside

    def clearance(self, xy):
        """Signed distance (m) to the polygon edge: positive inside, negative outside."""
        p = np.asarray(xy, float)
        d = np.full(p.shape[:-1], math.inf)
        a, b = self.poly, np.roll(self.poly, -1, axis=0)
        for p0, p1 in zip(a, b):
            e = p1 - p0
            t = np.clip(((p - p0) @ e)/float(e @ e), 0., 1.)
            q = p0 + t[..., None]*e
            d = np.minimum(d, np.linalg.norm(p - q, axis=-1))
        return np.where(self.contains(p), d, -d)

    def as_dict(self):
        return dict(polygon_xy=self.poly.round(4).tolist(), source=self.source)
