"""The navigator: paths between the mission's stops, planned on the FUSED map and re-planned in flight.

    goals      from the course semantics (rule 9.5): the stops refnav must hold at (tunnel mouth, the
               chicane's pre/post points, the pad) and the pass-through goals between them (the maze
               exit line);
    map        prior (the published map, the "origin map": an optimisation, not the truth) + the live
               occupancy grid built from depth this run (coursenav.occgrid). Live surfaces the prior does
               not explain (landmarks) are added as obstacles, and prior surfaces the live map has
               confidently seen THROUGH (live.cleared(): never a see-through surface such as a net, the
               window or the tunnel, never a keep-out) are removed, so a course built differently from
               the published layout (a wall moved or taken out) is flown as it IS, not as published;
    buffer     every obstacle is inflated by the vehicle radius + margin (`need`): A* runs on cells with
               clearance >= need, preferring more (corridor centring);
    path       A* -> line-of-sight pruning -> map-bounded corner arcs -> time-optimal speed profile
               (coursenav.smooth);
    re-plan    when a live obstacle appears within `need` of the remaining path, or every
               `replan_every_s` while the live map grows, from where the aircraft IS (believed course
               pose) at its current speed;
    gates      the plan's gate openings (plan["gate_lines"]) are walls here too: a gate is crossed only
               by refnav's straight pre -> centre -> post legs, so a re-plan can never route through the
               gap and back (Flight E, 2026-11-06).

Everything here is in the course frame; the caller converts to the flight frame every tick.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .course import Grid2D
from .planner import PlanError, astar, with_walls
from .smooth import Path, SmoothParams, build


@dataclass
class NavParams:
    replan_every_s: float = 1.0      # re-plan at most this often when the live map changed
    prefer_m: float = 0.6            # A*: clearance it prefers (corridor centring)
    crowd_weight: float = 4.0
    escape_m: float = 0.6            # a start inside the buffer (drifted close to a wall): step out first
    optimistic_unknown: bool = True  # no path on the fused map AND this run has already seen the course differ
                                     # from the published map: plan again treating prior surfaces the live
                                     # map has never observed as open (never nets/window/tunnel, never
                                     # keep-outs). The aircraft flies toward them only behind refnav's depth
                                     # check; what it sees there re-plans the path (explore-and-verify).
    ahead_skip_m: float = 0.2        # path checks start this far ahead of the aircraft's progress
    assumed_free_cost: float = 8.0   # optimistic plans: extra cost per metre through space that is free only
                                     # because an unconfirmed prior surface was assumed away (A* prefers
                                     # confirmed-free space and still finds the way when there is no other)


class Navigator:
    def __init__(self, obst: Grid2D, blocked: Grid2D, clearance, need_m, smooth: SmoothParams = None,
                 live=None, params: NavParams = None, gate_lines=None):
        blocked = with_walls(blocked, gate_lines)
        self.prior_obst, self.prior_blocked, self.prior_clear = obst, blocked, clearance
        self.need = float(need_m)
        self.sp = smooth or SmoothParams(need_m=need_m)
        self.sp.need_m = self.need
        self.live = live if (live is not None and live.prior.occ.shape == obst.occ.shape) else None
        self.p = params or NavParams()
        self._fused = (obst, blocked, self._effective(clearance, blocked))
        # keep-outs and gate walls: blocked for reasons other than a surface within `need` in this band.
        # They stay blocked whatever the live map says (live clearing only ever removes a SURFACE).
        self.keepout = blocked.occ & ~(np.asarray(clearance) < self.need)
        self._sig = (0, 0, 0)
        self.t_plan = None
        self.path: Path | None = None
        self.goals = None
        self.replans = 0
        self.optimistic = False       # the current path assumes unobserved prior surfaces are gone

    # ------------------------------------------------------------------ map
    @staticmethod
    def _effective(clear, blocked):
        """Clearance as the path checks see it: 0 inside any blocked cell. `blocked` also holds the
        keep-outs (e.g. boulders below the flight band, which the band's clearance does not see), so
        pruning shortcuts and corner arcs can never enter one."""
        eff = np.array(clear, float, copy=True)
        eff[blocked.occ] = 0.0
        return eff

    def _signature(self):
        lm = self.live.landmarks()
        cl = self.live.cleared() if hasattr(self.live, "cleared") else None
        kn = getattr(self.live, "known", None)
        n_known = 0 if kn is None else int((kn & self.prior_obst.occ).sum())   # prior surfaces observed so far
        return lm, cl, (int(lm.sum()), 0 if cl is None else int(cl.sum()), n_known)

    def fuse(self):
        """Prior - live-cleared surfaces + live landmarks -> (obstacle grid, blocked grid, clearance).
        Cached until either set changes."""
        if self.live is None:
            return self._fused
        lm, cl, sig = self._signature()
        if sig == self._sig:
            return self._fused
        self._sig = sig
        occ = self.prior_obst.occ.copy()
        if cl is not None:
            occ &= ~cl
        occ |= lm
        if sig[1] == 0:
            # nothing removed: keep the prior's exact surface clearance, only add the landmarks' distance
            clear = np.minimum(self.prior_clear, ndimage.distance_transform_edt(~lm) * self.prior_obst.res) \
                if sig[0] else np.asarray(self.prior_clear, float)
            blocked_occ = self.prior_blocked.occ | (clear < self.need)
        else:
            clear = ndimage.distance_transform_edt(~occ) * self.prior_obst.res
            blocked_occ = self.keepout | (clear < self.need)
        blocked = Grid2D(blocked_occ, self.prior_obst.origin, self.prior_obst.res)
        self._fused = (Grid2D(occ, self.prior_obst.origin, self.prior_obst.res), blocked, self._effective(clear, blocked))
        return self._fused

    def map_changed(self):
        return self.live is not None and self._signature()[2] != self._sig

    # ------------------------------------------------------------------ planning
    def _free_start(self, start, blocked, clear):
        """The aircraft may have drifted inside the buffer: start from the nearest cell outside it."""
        ij = blocked.ij(start)
        if blocked.inside(ij) and not blocked.occ[tuple(ij)]:
            return np.asarray(start, float)
        r = int(math.ceil(self.p.escape_m / blocked.res))
        i0, j0 = ij
        best, bd = None, math.inf
        for i in range(max(0, i0 - r), min(blocked.occ.shape[0], i0 + r + 1)):
            for j in range(max(0, j0 - r), min(blocked.occ.shape[1], j0 + r + 1)):
                if not blocked.occ[i, j]:
                    d = math.hypot(i - i0, j - j0)
                    if d < bd:
                        best, bd = (i, j), d
        if best is None:
            raise PlanError(f"start {tuple(np.round(start, 2))} is inside the obstacle buffer with no way out")
        return blocked.xy(np.array(best))

    def plan(self, start_xy, goals, t=0.0, v_start=0.0, keep_sharp_last=False):
        """Path from start_xy through goals[:-1] (pass-through) to goals[-1] (a stop). -> smooth.Path.
        On the fused map; failing that, and only once the course has been seen to differ from the published
        map (optimistic_unknown), on the fused map with the never-observed prior surfaces opened."""
        try:
            path = self._plan_on(self.fuse(), start_xy, goals, v_start)
            optimistic = False
        except PlanError as e:
            if not (self.p.optimistic_unknown and self.live is not None and hasattr(self.live, "unconfirmed")
                    and self.live.evidence_of_change()):
                raise
            try:
                path = self._plan_on(self._optimistic_grids(), start_xy, goals, v_start)
            except PlanError:
                raise e from None
            optimistic = True
        self.path, self.goals, self.t_plan = path, [np.asarray(g, float) for g in goals], t
        self.optimistic = optimistic
        self.replans += 1
        return path

    def _optimistic_grids(self):
        """The fused map with every unconfirmed prior surface opened (+ the A* penalty for crossing where one
        was). Keep-outs and gate walls stay."""
        obst, _, fused_clear = self.fuse()
        opened = obst.occ & self.live.unconfirmed()
        occ = obst.occ & ~opened
        clear = ndimage.distance_transform_edt(~occ) * self.prior_obst.res
        blocked = Grid2D(self.keepout | (clear < self.need), self.prior_obst.origin, self.prior_obst.res)
        # assumed free: cells that are free now only because an opened surface is gone
        assumed = (np.asarray(fused_clear) < self.need) & ~blocked.occ
        penalty = np.where(assumed, self.p.assumed_free_cost, 0.0)
        return (Grid2D(occ, self.prior_obst.origin, self.prior_obst.res), blocked,
                self._effective(clear, blocked), penalty)

    def _plan_on(self, grids, start_xy, goals, v_start):
        obst, blocked, clear = grids[:3]
        penalty = grids[3] if len(grids) > 3 else None
        pts = [np.asarray(start_xy, float)]
        s0 = self._free_start(start_xy, blocked, clear)
        if np.linalg.norm(s0 - pts[0]) > 1e-9:
            pts.append(s0)
        poly, keep = [pts[0]] if len(pts) == 1 else [pts[0], pts[1]], []
        cur = poly[-1]
        if len(pts) == 2 and np.linalg.norm(s0 - pts[0]) < 0.15:
            # kestral_nav934am (skybeta): a few-cm hop out of the margin (the venue mark is 1.4 cm inside it) set
            # the path's first heading, which refnav turns to face: 65-99 deg from the mark depending on cm of
            # position (Gazebo e313281: turned 73 deg into M1). Route from s0 without a vertex there.
            poly, cur = [pts[0]], np.asarray(s0, float)
        for g in goals:
            seg = astar(blocked, clear, cur, g, prefer_m=self.p.prefer_m, weight=self.p.crowd_weight, penalty=penalty)
            poly.extend(list(seg[1:]))
            keep.append(len(poly) - 1)
            cur = np.asarray(g, float)
        poly = np.asarray(poly, float)
        if len(poly) > 2:
            # kestral_nav934am: the first segment must be long enough to give the leg's heading (merge vertices within 0.30 m: offline the first heading then stays within 83-99 deg of the 86 deg corridor)
            near = [i for i in range(1, len(poly) - 1)
                    if np.linalg.norm(poly[i] - poly[0]) < 0.30 and i not in keep]
            if near:
                drop = set(near)
                idx = [i for i in range(len(poly)) if i not in drop]
                keep = [idx.index(k) for k in keep if k in idx]
                poly = poly[idx]
        return build(poly, obst, clear, self.sp, keep=keep, v_start=v_start, v_end=0.0)

    def progress(self, pos_xy):
        """Arc length of the current path's point nearest to pos_xy (course frame)."""
        if self.path is None or len(self.path.pts) == 0:
            return 0.0
        d = np.hypot(*(self.path.pts - np.asarray(pos_xy, float)[None, :2]).T)
        return float(self.path.s[int(np.argmin(d))])

    def needs_replan(self, t, progress_s):
        """A live obstacle within the buffer of the path still AHEAD of progress_s (what is behind the
        aircraft does not matter), checked whenever the live map changed, at most every replan_every_s."""
        if self.path is None or self.live is None:
            return False
        if self.t_plan is not None and t - self.t_plan < self.p.replan_every_s:
            return False
        if not self.map_changed():
            return False
        return self.path_blocked_ahead(progress_s)

    def path_blocked_ahead(self, progress_s):
        """Is the path ahead of progress_s inside the buffer on the map it was planned for (the fused map, or
        for an optimistic path the fused map with the still-unobserved prior surfaces open)?"""
        if self.path is None:
            return False
        _, _, clear = (self._optimistic_grids() if self.optimistic else self.fuse())[:3]
        # from a little ahead of the aircraft: where it IS may be inside the buffer (it drifted, or the plan
        # stepped it out of the margin: _free_start), which is not an obstacle on the way
        ahead = self.path.pts[self.path.s >= progress_s + self.p.ahead_skip_m]
        if len(ahead) == 0:
            return False
        ij = self.prior_obst.ij(ahead)
        ok = self.prior_obst.inside(ij)
        c = clear[ij[ok, 0], ij[ok, 1]]
        return bool(len(c) and c.min() < self.need)
