"""The navigator: paths between the mission's stops, planned on the FUSED map and re-planned in flight.

    goals      from the course semantics (rule 9.5): the stops refnav must hold at (tunnel mouth, the
               chicane's pre/post points, the pad) and the pass-through goals between them (the maze
               exit line);
    map        prior (the published map, the "origin map": an optimisation, not the truth) + the live
               occupancy grid built from depth this run (coursenav.occgrid). Live surfaces the prior does
               not explain (landmarks) are added as obstacles; a prior wall is never removed in this
               version (safe: at worst a longer path);
    buffer     every obstacle is inflated by the vehicle radius + margin (`need`): A* runs on cells with
               clearance >= need, preferring more (corridor centring);
    path       A* -> line-of-sight pruning -> map-bounded corner arcs -> time-optimal speed profile
               (coursenav.smooth);
    local      obstacles from the LOCAL grid (coursenav.localgrid: odometry frame, never gated on the
               localizer), converted to the course frame by the caller with the same transform as the
               aircraft's believed pose (set_local). ADD-ONLY like the landmarks: live data never removes a
               prior wall, net or keep-out (depth sees through the nets);
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
from .smooth import Path, SmoothParams, build, speed_profile, segment_min_clearance, stop_approach_caps


@dataclass
class NavParams:
    replan_every_s: float = 1.0      # re-plan at most this often when the live map changed
    prefer_m: float = 0.6            # A*: clearance it prefers (corridor centring)
    crowd_weight: float = 4.0
    escape_m: float = 0.6            # a start inside the buffer (drifted close to a wall): step out first
    escape_creep_mps: float = 0.1    # ... at this speed (the profile is 0 inside the buffer)


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
        self._lm_count = 0
        self.local = None                # course-grid bool mask of local obstacles (set_local)
        self._local_sig = None           # signature of the local mask the fused map was built with
        self.t_plan = None
        self.path: Path | None = None
        self.goals = None
        self.replans = 0

    # ------------------------------------------------------------------ map
    @staticmethod
    def _effective(clear, blocked):
        """Clearance as the path checks see it: 0 inside any blocked cell. `blocked` also holds the
        keep-outs (e.g. boulders below the flight band, which the band's clearance does not see), so
        pruning shortcuts and corner arcs can never enter one."""
        eff = np.array(clear, float, copy=True)
        eff[blocked.occ] = 0.0
        return eff

    def set_local(self, mask):
        """Local-grid obstacles as a bool mask on the prior grid (None clears them). Add-only."""
        if mask is not None:
            mask = np.asarray(mask, bool)
            if mask.shape != self.prior_obst.occ.shape:
                raise ValueError("local mask must share the prior grid")
            if not mask.any():
                mask = None
        self.local = mask

    @staticmethod
    def _sig(mask):
        return None if mask is None else hash(np.packbits(mask).tobytes())

    def _landmarks(self):
        return None if self.live is None else self.live.landmarks()

    def fuse(self):
        """Prior + live landmarks + local obstacles -> (obstacle grid, blocked grid, clearance). Cached until
        they change."""
        if self.live is None and self.local is None and self._local_sig is None:
            return self._fused
        lm = self._landmarks()
        n = 0 if lm is None else int(lm.sum())
        sig = self._sig(self.local)
        if n == self._lm_count and sig == self._local_sig:
            return self._fused
        self._lm_count, self._local_sig = n, sig
        occ = self.prior_obst.occ.copy()
        if lm is not None:
            occ |= lm
        if self.local is not None:
            occ |= self.local
        clear = ndimage.distance_transform_edt(~occ) * self.prior_obst.res
        blocked = Grid2D(self.prior_blocked.occ | (clear < self.need), self.prior_obst.origin, self.prior_obst.res)
        self._fused = (Grid2D(occ, self.prior_obst.origin, self.prior_obst.res), blocked, self._effective(clear, blocked))
        return self._fused

    def map_changed(self):
        lm = self._landmarks()
        return ((lm is not None and int(lm.sum()) != self._lm_count)
                or self._sig(self.local) != self._local_sig)

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

    def plan(self, start_xy, goals, t=0.0, v_start=0.0, keep_sharp_last=False, observed_free=None):
        """Path from start_xy through goals[:-1] (pass-through) to goals[-1] (a stop). -> smooth.Path."""
        obst, blocked, clear = self.fuse()
        if observed_free is not None:
            allowed = np.asarray(observed_free, bool)
            if allowed.shape != blocked.occ.shape:
                raise PlanError("observed-free mask must share the planning grid")
            blocked = Grid2D(blocked.occ | ~allowed, blocked.origin, blocked.res)
            clear = self._effective(clear, blocked)
            start_cell = blocked.ij(np.asarray(start_xy, float))
            if not blocked.inside(start_cell) or blocked.occ[tuple(start_cell)]:
                # Recovery never invokes the usual buffer escape through unknown.
                raise PlanError("recovery start footprint is not certified free")
        pts = [np.asarray(start_xy, float)]
        s0 = self._free_start(start_xy, blocked, clear)
        if np.linalg.norm(s0 - pts[0]) > 1e-9:
            pts.append(s0)
        poly, keep = [pts[0]] if len(pts) == 1 else [pts[0], pts[1]], []
        cur = poly[-1]
        for g in goals:
            seg = astar(blocked, clear, cur, g, prefer_m=self.p.prefer_m, weight=self.p.crowd_weight)
            poly.extend(list(seg[1:]))
            keep.append(len(poly) - 1)
            cur = np.asarray(g, float)
        poly = np.asarray(poly, float)
        path = build(poly, obst, clear, self.sp, keep=keep, v_start=v_start, v_end=0.0)
        if np.linalg.norm(s0 - pts[0]) > 1e-9 and len(path.v) > 1:
            # the start was inside the buffer: the profile is 0 there (no tube), which no executor can fly.
            # Creep out (escape_creep_mps, never above what the profile allows just beyond the escape point)
            esc = float(np.linalg.norm(s0 - pts[0]))
            beyond = path.v[path.s > esc + 1e-9]
            creep = min(self.p.escape_creep_mps, float(beyond[0]) if len(beyond) and beyond[0] > 0 else self.p.escape_creep_mps)
            inside = (path.s > 1e-9) & (path.s <= esc + 1e-9)
            path.v[inside] = np.maximum(path.v[inside], creep)
        self.path, self.goals, self.t_plan = path, [np.asarray(g, float) for g in goals], t
        self.replans += 1
        return path

    def plan_recovery(self, start_xy, reference, observed, to_nominal, t=0., lookahead_m=1.):
        """Observed-free detour to an unchanged reference suffix; never clear prior walls.

        Only the changed prefix must be fully observed. The suffix remains the exact
        old route, subject to the original map and refnav's ordinary depth checks.
        """
        if reference is None or len(reference.pts) < 2:
            raise PlanError("no reference path for a bounded recovery detour")
        start_xy = np.asarray(start_xy, float)
        nearest = int(np.argmin(np.linalg.norm(reference.pts-start_xy, axis=1)))
        allowed = observed.allowed_course_cells(self.prior_obst, to_nominal, self.need)
        ij = self.prior_obst.ij(reference.pts)
        inside = self.prior_obst.inside(ij)
        certified = np.zeros(len(ij), bool)
        certified[inside] = allowed[ij[inside, 0], ij[inside, 1]]
        candidates = np.flatnonzero(certified & (reference.s >= reference.s[nearest] + max(1., lookahead_m)))
        if not len(candidates):
            raise PlanError("no observed-free rejoin beyond recovery lookahead")
        # Bound planning work per explicit request, including a distant visible rejoin.
        candidates = candidates[np.unique(np.linspace(0, len(candidates)-1, min(8, len(candidates))).astype(int))]
        obst, _, clear = self.fuse()
        old_goals = [] if self.goals is None else [np.array(g, float) for g in self.goals]
        last_reason = "no certified detour"
        for join in candidates:
            goals = [g for g in old_goals
                     if nearest < int(np.argmin(np.linalg.norm(reference.pts-g, axis=1))) < join]
            goals.append(reference.pts[join])
            try:
                prefix = self.plan(start_xy, goals, t=t, v_start=0., observed_free=allowed)
                if not observed.certifies_path(prefix.pts, to_nominal, self.need):
                    raise PlanError("detour swept footprint is not observed free")
                # New PID alone is not progress: a real detour must depart from the
                # reference segments, rather than simply densify the same corridor.
                a, b = reference.pts[:-1], reference.pts[1:]
                ab = b-a
                q = prefix.pts[:, None, :]-a[None, :, :]
                u = np.clip(np.sum(q*ab, axis=2)/np.maximum(np.sum(ab*ab, axis=1), 1e-12), 0., 1.)
                distance = np.linalg.norm(q-u[:, :, None]*ab, axis=2).min(axis=1)
                if float(distance.max()) < max(.10, 2*self.prior_obst.res):
                    raise PlanError("replan does not leave the blocked reference corridor")
                suffix = reference.pts[join:]
                if any(segment_min_clearance(obst, clear, a, b) < self.need
                       for a, b in zip(suffix[:-1], suffix[1:])):
                    raise PlanError("unchanged suffix is blocked by the fused map")
                k = len(prefix.pts)-1
                pts = np.concatenate((prefix.pts, reference.pts[join+1:]))
                arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
                curvature = np.r_[prefix.kappa, reference.kappa[join+1:]]
                heading = np.r_[prefix.heading, reference.heading[join+1:]]
                if join < len(reference.pts)-1:
                    heading[k] = reference.heading[join]
                clearance = np.r_[prefix.clearance, reference.clearance[join+1:]]
                # Recheck/reduce the suffix's clearance with current fused obstacles.
                cells = obst.ij(pts);valid = obst.inside(cells)
                if not valid.all():
                    raise PlanError("recovery path leaves the planning grid")
                clearance = np.minimum(clearance, clear[cells[:, 0], cells[:, 1]])
                stops = sorted(set(prefix.stops + [k] + [k+s-join for s in reference.stops if s > join]))
                path = Path(pts, arc, curvature, heading, np.zeros(len(pts)), clearance, stops)
                speed_profile(path, self.sp, v_start=0., v_end=0., stop_caps=stop_approach_caps(path, obst, self.sp))
                self.path, self.goals, self.t_plan = path, old_goals, t
                return path, prefix.pts.copy(), int(join)
            except PlanError as exc:
                last_reason = str(exc)
                self.path = None
                self.goals = old_goals
        raise PlanError(last_reason)

    def needs_replan(self, t, progress_s):
        """A live obstacle within the buffer of the path still ahead, or the periodic re-plan is due."""
        if self.path is None or (self.live is None and self.local is None and self._local_sig is None):
            return False
        if self.t_plan is not None and t - self.t_plan < self.p.replan_every_s:
            return False
        if not self.map_changed():
            return False
        _, _, clear = self.fuse()
        ahead = self.path.pts[self.path.s >= progress_s]
        if len(ahead) == 0:
            return False
        ij = self.prior_obst.ij(ahead)
        ok = self.prior_obst.inside(ij)
        c = clear[ij[ok, 0], ij[ok, 1]]
        return bool(len(c) and c.min() < self.need)
