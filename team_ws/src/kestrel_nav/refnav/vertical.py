"""Vertical centring from live depth. PURE (numpy only); no course geometry.

Openings (windows, gaps under bars, passages over sills) are measured from the
depth points in the corridor ahead, relative to the aircraft, so neither their
height nor the height estimate's absolute error matters. In each forward slab
(near to far), surfaces above and below the aircraft that intrude into the
corridor are found; the floor is excluded. The first slab that constrains the
vertical band decides the target:

  above and below, gap >= 2*clearance -> the gap's midpoint
  only below                          -> at least clearance above it
  only above                          -> at least clearance below it
  gap < 2*clearance (a solid face)    -> not an opening; ignored here (the planner
                                         treats it as a wall)

Points are in the level heading frame relative to the body origin:
x forward, y right, z down (refnav.depth.DepthFrame.points).
"""
from __future__ import annotations

import numpy as np


def opening_target(pts, height, cruise, lateral_m, clearance_m, alt_min, alt_max,
                   x_min=0.3, x_max=2.5, slab_m=0.15, min_pts=6, floor_band_m=0.12):
    """-> (target height above the floor, info dict) or (None, info) when nothing constrains."""
    info = {}
    if pts is None or len(pts) == 0 or not np.isfinite(height):
        return None, info
    p = np.asarray(pts, float)
    x, y, up = p[:, 0], p[:, 1], -p[:, 2]
    keep = ((x > x_min) & (x < x_max) & (np.abs(y) <= lateral_m)
            & (up > -(height - floor_band_m))            # the floor is not an obstacle
            & (up < alt_max + 0.5 - height))             # far overhead (roof, nets) is irrelevant
    x, up = x[keep], up[keep]
    if len(x) < min_pts:
        return None, info
    order = np.argsort(x)
    x, up = x[order], up[order]
    edges = np.arange(x_min, x_max + slab_m, slab_m)
    idx = np.digitize(x, edges) - 1
    for k in range(len(edges) - 1):
        z = up[idx == k]
        if len(z) < min_pts:
            continue
        above, below = z[z > 0], z[z < 0]
        top = float(np.quantile(above, .1)) if len(above) >= 3 else None
        bot = float(np.quantile(below, .9)) if len(below) >= 3 else None
        if top is None and bot is None:
            continue
        info = dict(slab_x_m=round(float(edges[k] + slab_m / 2), 2),
                    top_rel_m=None if top is None else round(top, 3),
                    bottom_rel_m=None if bot is None else round(bot, 3))
        if top is not None and bot is not None:
            if top - bot < 2 * clearance_m:
                info['kind'] = 'solid_face'
                continue                                  # a wall at this height: the planner's business
            rel = (top + bot) / 2
            info['kind'] = 'opening'
        elif bot is not None:
            rel = max(cruise - height, bot + clearance_m)
            info['kind'] = 'below_only'
        else:
            rel = min(cruise - height, top - clearance_m)
            info['kind'] = 'above_only'
        target = float(np.clip(height + rel, alt_min, alt_max))
        info['target_m'] = round(target, 3)
        return target, info
    return None, info


class VerticalCentring:
    """Target height for the altitude hold: cruise unless depth shows an opening ahead.

    The target found for an opening is held until the aircraft is hold_m past it
    (its edges leave the camera's view up close), then cruise resumes. The output
    is rate limited."""

    def __init__(self, cruise, lateral_m, clearance_m, alt_min=.35, alt_max=1.5,
                 rate_mps=.25, hold_m=.4, hold_s=20.):
        self.cruise, self.lateral_m, self.clearance_m = float(cruise), float(lateral_m), float(clearance_m)
        self.alt_min, self.alt_max = float(alt_min), float(alt_max)
        self.rate, self.hold_m, self.hold_s = float(rate_mps), float(hold_m), float(hold_s)
        self.target = float(cruise)
        self.out = None
        self.held = None          # (target, opening NE point, heading, time seen)
        self.info = {}

    def observe(self, now, pts, height, pos_ne, heading):
        t, info = opening_target(pts, height, self.cruise, self.lateral_m, self.clearance_m,
                                 self.alt_min, self.alt_max)
        self.info = info
        if t is None:
            return
        if (self.held is not None and self.held[4] == 'opening' and info.get('kind') != 'opening'
                and self._past(pos_ne) <= self.hold_m):
            return        # one edge out of view up close: keep the full opening already measured
        d = info['slab_x_m']
        at = (pos_ne[0] + d * np.cos(heading), pos_ne[1] + d * np.sin(heading))
        self.held = (t, at, heading, now, info.get('kind'))

    def _past(self, pos_ne):
        """Signed distance of the aircraft past the held opening along its approach heading."""
        _, at, hdg, _, _ = self.held
        return (pos_ne[0] - at[0]) * np.cos(hdg) + (pos_ne[1] - at[1]) * np.sin(hdg)

    def step(self, now, dt, height, pos_ne):
        target = self.cruise
        if self.held is not None:
            t, seen = self.held[0], self.held[3]
            if self._past(pos_ne) <= self.hold_m and now - seen <= self.hold_s:
                target = t
            else:
                self.held = None
        self.target = target
        if self.out is None:
            self.out = float(height)
        step = self.rate * max(dt, 0.)
        self.out += float(np.clip(target - self.out, -step, step))
        return self.out
