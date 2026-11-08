"""Vertical centring from live depth. PURE (numpy only); no course geometry.

Only a confirmed OPENING moves the altitude target: in one forward slab of the
corridor ahead the face's height profile has a free interval with a measured
surface above it and one below it (floor excluded), the interval fits the
airframe with vertical_clearance_m at each edge inside alt_min..alt_max, and
depth returns are seen THROUGH it from beyond the slab (proof it is a hole,
not a wall). The interval need not contain the aircraft's own height: a window
whose sill is above cruise is found from range and climbed to. Of several, the
one nearest the current height wins. The target is its midpoint; otherwise
cruise height. One-sided readings are ignored: with a narrow vertical field of
view a wall cut off by the image edge looks exactly like a sill or a bar, and
an interval is only free between two surfaces actually measured in that slab
(nothing between them can be out of view). No opening height is configured.

Points are in the level heading frame relative to the body origin:
x forward, y right, z down (refnav.depth.DepthFrame.points).
"""
from __future__ import annotations

import numpy as np


def _free_intervals(z, res, min_bin_pts, occ_frac):
    """Free height intervals of one slab: maximal runs of empty bins with an occupied bin on each side.
    A bin is occupied when it holds at least occ_frac of a full-width bin's points (the 75th percentile of
    the non-empty bins): the sliver of wall beside an opening that the corridor clips when the aircraft
    is a little off its centre line does not close it. Seeing through is still required by the caller.
    -> [(bottom, top)] in relative metres: the highest point below the run, the lowest point above it."""
    lo = np.floor(z.min() / res) * res
    bins = np.floor((z - lo) / res).astype(int)
    counts = np.bincount(bins)
    full = np.percentile(counts[counts > 0], 75)
    occ = counts >= max(min_bin_pts, occ_frac * full)
    filled = np.flatnonzero(occ)
    out = []
    for a, b in zip(filled[:-1], filled[1:]):
        if b - a > 1:                                    # empty bins a+1 .. b-1, bounded by a and b
            out.append((float(z[bins == a].max()), float(z[bins == b].min())))
    return out


def opening_target(pts, height, cruise, lateral_m, clearance_m, alt_min, alt_max,
                   x_min=0.3, x_max=2.5, slab_m=0.15, min_pts=6, floor_band_m=0.12,
                   max_gap_m=1.3, behind_m=0.3, min_through=15, res_m=0.05, min_bin_pts=2, occ_frac=0.25):
    """-> (target height above the floor, info dict) or (None, info) when no opening is confirmed."""
    info = {}
    if pts is None or len(pts) == 0 or not np.isfinite(height):
        return None, info
    p = np.asarray(pts, float)
    x, y, up = p[:, 0], p[:, 1], -p[:, 2]
    keep = ((x > x_min) & (np.abs(y) <= lateral_m)
            & (up > -(height - floor_band_m))            # the floor is not an obstacle
            & (up < alt_max + 0.5 - height))             # far overhead (roof, nets) is irrelevant
    beyond = x > x_min                                   # every return, for the see-through test
    fx, fy, fup = x[beyond], y[beyond], up[beyond]
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
        slab_x = float(edges[k] + slab_m / 2)
        gaps = _free_intervals(z, res_m, min_bin_pts, occ_frac)
        if not gaps:
            if z.min() < 0 < z.max():
                info = dict(slab_x_m=round(slab_x, 2), top_rel_m=round(float(z[z > 0].min()), 3),
                            bottom_rel_m=round(float(z[z < 0].max()), 3), gap_m=0.0, through=0,
                            kind='solid_face', gaps=0)
                return None, info                         # a wall across our height: the planner's business
            continue                                      # one-sided: never moves the altitude
        best, rejected = None, None
        # seen through: a return well beyond the face whose ray crosses the face's plane inside the gap
        # and the corridor (the ray, not the far point, must be in the corridor: a far wall seen through
        # a 0.9 m window subtends few pixels inside a 0.74 m corridor at its own distance)
        far = fx > slab_x + behind_m
        sc = slab_x / fx[far]
        ry, rup = fy[far] * sc, fup[far] * sc
        for bot, top in gaps:
            gap = top - bot
            through = int(((np.abs(ry) <= lateral_m) & (rup > bot) & (rup < top)).sum())
            target = float(np.clip(height + (top + bot) / 2, alt_min, alt_max))
            fits = (target - clearance_m >= height + bot - 1e-9 and target + clearance_m <= height + top + 1e-9)
            cand = dict(slab_x_m=round(slab_x, 2), top_rel_m=round(top, 3), bottom_rel_m=round(bot, 3),
                        gap_m=round(gap, 3), through=through, gaps=len(gaps))
            if gap < 2 * clearance_m or not fits:
                cand['kind'] = 'solid_face'
            elif gap > max_gap_m or through < min_through:
                cand['kind'] = 'not_an_opening'
            else:
                cand.update(kind='opening', target_m=round(target, 3))
                if best is None or abs(target - height) < abs(best[0] - height):
                    best = (target, cand)
                continue
            # report the most informative rejection: a wider gap over a narrower one
            if rejected is None or gap > rejected['gap_m']:
                rejected = cand
        if best is not None:
            return best
        return None, rejected
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
        self.out = None
        self.held = None          # (target, opening NE point, heading, time seen, kind, hold_s)
        self.override = None      # altitude commanded by the mission's altitude search (wins over all)
        self.info = {}

    def observe(self, now, pts, height, pos_ne, heading):
        t, info = opening_target(pts, height, self.cruise, self.lateral_m, self.clearance_m,
                                 self.alt_min, self.alt_max)
        self.info = info
        if t is None:
            return
        d = info['slab_x_m']
        at = (pos_ne[0] + d * np.cos(heading), pos_ne[1] + d * np.sin(heading))
        self.held = (t, at, heading, now, info.get('kind'), self.hold_s)

    def hold_level(self, target, at_ne, heading, now, hold_s, kind='altitude_search'):
        """Keep `target` until the aircraft is hold_m past at_ne along heading (or hold_s has passed):
        a height found by searching, not by seeing both edges, must survive the edges leaving the view."""
        self.held = (float(np.clip(target, self.alt_min, self.alt_max)), tuple(at_ne), heading, now, kind,
                     float(hold_s))

    def _past(self, pos_ne):
        """Signed distance of the aircraft past the held opening along its approach heading."""
        _, at, hdg = self.held[:3]
        return (pos_ne[0] - at[0]) * np.cos(hdg) + (pos_ne[1] - at[1]) * np.sin(hdg)

    def step(self, now, dt, height, pos_ne):
        target = self.cruise
        if self.override is not None:
            target = float(np.clip(self.override, self.alt_min, self.alt_max))
        elif self.held is not None:
            t, seen, hold_s = self.held[0], self.held[3], self.held[5]
            if self._past(pos_ne) <= self.hold_m and now - seen <= hold_s:
                target = t
            else:
                self.held = None
        self.target = target
        if self.out is None:
            self.out = float(height)
        step = self.rate * max(dt, 0.)
        self.out += float(np.clip(target - self.out, -step, step))
        return self.out
