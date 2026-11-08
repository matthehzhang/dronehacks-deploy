"""The navigator's view of the live occupancy grid (coursenav.loc_node /coursenav/occupancy). PURE (numpy +
scipy): used by coursenav.nav_node and testable without ROS."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .course import Grid2D


class LiveLandmarks:
    """The localizer's published occupancy grid as the navigator's live map:
      landmarks(): confident surfaces more than novel_min_m from any prior surface, in clumps of at
                   least min_cells (coursenav.occgrid rules): obstacles the published map lacks;
      cleared():   prior surface cells the live map has confidently seen THROUGH (p <= free_prob), with
                   no confident live surface within keep_m, that are not see-through by design (nets,
                   window, tunnel: see_through) and come in clumps of at least min_cells: surfaces the
                   published map has that the built course does not (a wall moved or removed)."""

    def __init__(self, prior: Grid2D, occ_prob=0.88, novel_min_m=0.15, min_cells=4, free_prob=0.23,
                 keep_m=0.10, see_through=None, plan_obst=None):
        self.prior = prior
        self.prior_dist = prior.distance()
        self.occ_prob, self.novel_min_m, self.min_cells = occ_prob, novel_min_m, min_cells
        self.free_prob, self.keep_m = free_prob, keep_m
        self.see_through = np.zeros(prior.occ.shape, bool) if see_through is None else np.asarray(see_through, bool)
        # the cells removable from the PLANNING obstacle grid: its surfaces (default: the scan band's)
        self.plan_obst = prior.occ if plan_obst is None else np.asarray(plan_obst, bool)
        self.occupied = np.zeros(prior.occ.shape, bool)
        self.free = np.zeros(prior.occ.shape, bool)
        self.known = np.zeros(prior.occ.shape, bool)
        self.likely_occ = np.zeros(prior.occ.shape, bool)

    def on_grid(self, m):
        nx, ny = self.prior.occ.shape
        if m.info.width != nx or m.info.height != ny:
            return False
        prob = np.asarray(m.data, np.int8).reshape(ny, nx).T          # row-major, x fastest; -1 unknown
        self.occupied = prob >= int(round(self.occ_prob * 100))
        self.free = (prob >= 0) & (prob <= int(round(self.free_prob * 100)))
        self.known = prob >= 0
        self.likely_occ = prob >= 50
        return True

    def _clumps(self, mask):
        if not mask.any() or self.min_cells <= 1:
            return mask
        lab, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
        sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, bool)
        keep[1:] = sizes >= self.min_cells
        return keep[lab]

    def landmarks(self):
        return self._clumps(self.occupied & (self.prior_dist > self.novel_min_m))

    def cleared(self):
        if not self.free.any():
            return np.zeros(self.plan_obst.shape, bool)
        r = max(1, int(round(self.keep_m / self.prior.res)))
        near_occ = ndimage.binary_dilation(self.occupied, structure=np.ones((2 * r + 1, 2 * r + 1), bool))
        return self._clumps(self.plan_obst & self.free & ~self.see_through & ~near_occ)

    def evidence_of_change(self):
        """Has this run seen the course differ from the published map (a removed or a new surface)?"""
        return bool(self.cleared().any() or self.landmarks().any())

    def unconfirmed(self):
        """Prior surface cells this run has NOT confirmed: no reading at all, or readings that never made
        them more likely occupied than not (within keep_m), excluding the surfaces a beam passes by design
        (nets, window, tunnel), which are never assumed away."""
        r = max(1, int(round(self.keep_m / self.prior.res)))
        near = ndimage.binary_dilation(self.likely_occ, structure=np.ones((2 * r + 1, 2 * r + 1), bool))
        return self.plan_obst & ~near & ~self.see_through
