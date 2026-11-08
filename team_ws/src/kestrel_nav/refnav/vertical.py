"""Vertical centring from live depth. PURE (numpy only); no course geometry.

Only a confirmed OPENING moves the altitude target: in one forward slab of the
corridor ahead there is a surface above the aircraft and one below it (floor
excluded), the gap between them fits the airframe with vertical_clearance_m at
each edge, and depth returns are seen THROUGH the gap from beyond the slab
(proof it is a hole, not a wall). The target is the gap's midpoint; otherwise
cruise height. One-sided readings are ignored: with a narrow vertical field of
view a wall cut off by the image edge looks exactly like a sill or a bar, and
anything inside the aircraft's own height band is already a 2-D wall for the
planner. No opening height is configured anywhere.

Points are in the level heading frame relative to the body origin:
x forward, y right, z down (refnav.depth.DepthFrame.points).
"""
from __future__ import annotations

import numpy as np


def opening_target(pts, height, cruise, lateral_m, clearance_m, alt_min, alt_max,
                   x_min=0.3, x_max=2.5, slab_m=0.15, min_pts=6, floor_band_m=0.12,
                   max_gap_m=1.3, behind_m=0.3, min_through=15):
    """-> (target height above the floor, info dict) or (None, info) when no opening is confirmed."""
    info = {}
    if pts is None or len(pts) == 0 or not np.isfinite(height):
        return None, info
    p = np.asarray(pts, float)
    x, y, up = p[:, 0], p[:, 1], -p[:, 2]
    keep = ((x > x_min) & (np.abs(y) <= lateral_m)
            & (up > -(height - floor_band_m))            # the floor is not an obstacle
            & (up < alt_max + 0.5 - height))             # far overhead (roof, nets) is irrelevant
    x, up = x[keep], up[keep]
    near = x < x_max
    if near.sum() < min_pts:
        return None, info
    edges = np.arange(x_min, x_max + slab_m, slab_m)
    idx = np.digitize(x, edges) - 1
    for k in range(len(edges) - 1):
        z = up[(idx == k) & near]
        if len(z) < min_pts:
            continue
        above, below = z[z > 0], z[z < 0]
        top = float(np.quantile(above, .1)) if len(above) >= 3 else None
        bot = float(np.quantile(below, .9)) if len(below) >= 3 else None
        if top is None or bot is None:
            continue                                      # one-sided: never moves the altitude
        slab_x = float(edges[k] + slab_m / 2)
        gap = top - bot
        through = int(((x > slab_x + behind_m) & (up > bot) & (up < top)).sum())
        info = dict(slab_x_m=round(slab_x, 2), top_rel_m=round(top, 3), bottom_rel_m=round(bot, 3),
                    gap_m=round(gap, 3), through=through)
        if gap < 2 * clearance_m:
            info['kind'] = 'solid_face'
            return None, info                             # a wall here: the planner's business
        if gap > max_gap_m or through < min_through:
            info['kind'] = 'not_an_opening'
            return None, info
        target = float(np.clip(height + (top + bot) / 2, alt_min, alt_max))
        info.update(kind='opening', target_m=round(target, 3))
        return target, info
    return None, info


class VerticalCentring:
    """Target height for the altitude hold: cruise unless depth shows an opening ahead.

    The target found for an opening is held until the aircraft is hold_m past it
    (its edges leave the camera's view up close), then cruise resumes. The output
    is rate limited."""

    def __init__(self, cruise, lateral_m, clearance_m, alt_min=.35, alt_max=1.5,
                 rate_mps=.25, hold_m=.4, hold_s=6.):
        self.cruise, self.lateral_m, self.clearance_m = float(cruise), float(lateral_m), float(clearance_m)
        self.alt_min, self.alt_max = float(alt_min), float(alt_max)
        self.rate, self.hold_m, self.hold_s = float(rate_mps), float(hold_m), float(hold_s)
        self.target = float(cruise)
        self.base = float(cruise)
        self.probe_offsets = (.25, .45, -.2, 0.)   # altitudes tried when a full yaw sweep finds no route
        self.probes = 0
        self.out = None
        self.held = None          # (target, opening NE point, heading, time seen)
        self.info = {}

    def probe_next(self):
        """A full yaw sweep found no route at this height: try the next probe altitude.

        The new height becomes the cruise height (kept once a route is found there)."""
        self.probes += 1
        off = self.probe_offsets[(self.probes - 1) % len(self.probe_offsets)]
        self.cruise = float(np.clip(self.base + off, self.alt_min, self.alt_max))
        self.held = None
        return self.cruise

    def observe(self, now, pts, height, pos_ne, heading):
        t, info = opening_target(pts, height, self.cruise, self.lateral_m, self.clearance_m,
                                 self.alt_min, self.alt_max)
        self.info = info
        if t is None:
            return
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
