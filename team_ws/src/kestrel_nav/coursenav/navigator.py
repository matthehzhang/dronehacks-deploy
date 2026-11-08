"""The navigator: paths between the mission's stops, planned on the FUSED map and re-planned in flight.

    goals      from the course semantics (rule 9.5): the stops refnav must hold at (tunnel mouth, the
               chicane's pre/post points, the pad) and the pass-through goals between them (the maze
               exit line);
    map        prior (the published map, the "origin map": an optimisation, not the truth) + the live
               occupancy grid built from depth this run (coursenav.occgrid). Live surfaces the prior does
               not explain (landmarks) are added as obstacles; prior walls may be cleared only by recent confident pass-through evidence while dropped;
               protected and camera-confirmed prior cells remain;
    buffer     every obstacle is inflated by the vehicle radius + margin (`need`): A* runs on cells with
               clearance >= need, preferring more (corridor centring);
    path       A* -> line-of-sight pruning -> map-bounded corner arcs -> time-optimal speed profile
               (coursenav.smooth);
    local      obstacles from the LOCAL grid (coursenav.localgrid: odometry frame, never gated on the
               localizer), converted to the course frame by the caller with the same transform as the
               aircraft's believed pose (set_local). Nets, keep-outs, gate walls and enclosure bounds persist.
               A local-only fallback requires an observed-free swept footprint, never unknown space;
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
    clear_prior: bool = True
    fallback_local_only: bool = True
    confirm_m: float = 0.15
    escape_creep_mps: float = 0.1    # ... at this speed (the profile is 0 inside the buffer)


class Navigator:
    def __init__(self, obst: Grid2D, blocked: Grid2D, clearance, need_m, smooth: SmoothParams = None,
                 live=None, params: NavParams = None, gate_lines=None, protect=None, fixed_blocked=None):
        blocked = with_walls(blocked, gate_lines)
        self.protect = None if protect is None else np.asarray(protect, bool).copy()
        self.fixed = None if fixed_blocked is None else with_walls(
            Grid2D(np.asarray(fixed_blocked, bool).copy(), obst.origin, obst.res), gate_lines).occ
        for mask in (self.protect, self.fixed):
            if mask is not None and mask.shape != obst.occ.shape:
                raise ValueError("prior protection must share the planning grid")
        self.prior_dropped = False
        self.cleared = np.zeros(obst.occ.shape, bool)
        self.confirmed = np.zeros(obst.occ.shape, bool)
        self.plan_mode = None
        self.path_certificate = None
        self._fused_signature = None
        self._plan_signature = None
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
        if self.prior_dropped and mask is not None:
            near = ndimage.distance_transform_edt(~mask) * self.prior_obst.res <= self.p.confirm_m + 1e-9
            self.confirmed |= self.prior_obst.occ & near
            self.cleared &= ~self.confirmed

    def can_correct_prior(self):
        return self.protect is not None and self.fixed is not None

    def set_prior_dropped(self, dropped):
        dropped = bool(dropped)
        if dropped == self.prior_dropped:
            return False
        self.prior_dropped = dropped
        if not dropped:
            self.cleared[:] = False
            self.confirmed[:] = False
        # A local-only/cleared path never survives a trust-mode transition.
        self.path = None
        self.path_certificate = None
        return True

    def set_free(self, mask):
        if not (self.prior_dropped and self.p.clear_prior and self.can_correct_prior()) or mask is None:
            return 0
        mask = np.asarray(mask, bool)
        if mask.shape != self.prior_obst.occ.shape:
            raise ValueError("free mask must share the planning grid")
        new = mask & self.prior_obst.occ & ~self.protect & ~self.fixed & ~self.confirmed & ~self.cleared
        self.cleared |= new
        return int(new.sum())

    @staticmethod
    def _sig(mask):
        return None if mask is None else hash(np.packbits(mask).tobytes())

    def _landmarks(self):
        return None if self.live is None else self.live.landmarks()

    def _signature(self):
        return (self.prior_dropped, self._sig(self.local), self._sig(self._landmarks()),
                self._sig(self.cleared), self._sig(self.confirmed))

    def _build(self, prior_occ, rebuild_buffer=False):
        occ = prior_occ.copy()
        lm = self._landmarks()
        if lm is not None:
            occ |= lm
        if self.local is not None:
            occ |= self.local
        clear = ndimage.distance_transform_edt(~occ) * self.prior_obst.res
        base = self.fixed if rebuild_buffer else self.prior_blocked.occ
        blocked = Grid2D(base | (clear < self.need), self.prior_obst.origin, self.prior_obst.res)
        return Grid2D(occ, blocked.origin, blocked.res), blocked, self._effective(clear, blocked)

    def fuse(self):
        signature = self._signature()
        if signature != self._fused_signature:
            lm = self._landmarks()
            if not self.cleared.any() and self.local is None and (lm is None or not lm.any()):
                self._fused = (self.prior_obst, self.prior_blocked,
                               self._effective(self.prior_clear, self.prior_blocked))
            else:
                self._fused = self._build(self.prior_obst.occ & ~self.cleared,
                                          rebuild_buffer=bool(self.cleared.any()))
            self._fused_signature = signature
        return self._fused

    def local_only_map(self):
        if not (self.prior_dropped and self.can_correct_prior()):
            raise PlanError("local-only planning requires a dropped prior and protection masks")
        return self._build(self.prior_obst.occ & (self.protect | self.confirmed), rebuild_buffer=True)

    def map_changed(self):
        return self._signature() != self._plan_signature

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

    def plan(self, start_xy, goals, t=0.0, v_start=0.0, keep_sharp_last=False,
             observed_free=None, observed=None, to_nominal=None):
        """Try corrected prior first; local-only fallback requires a certified whole path."""
        mode = "fused"
        try:
            path = self._plan_on(self.fuse(), start_xy, goals, v_start, observed_free)
        except PlanError:
            if not (self.prior_dropped and self.p.fallback_local_only and self.can_correct_prior()
                    and observed is not None and to_nominal is not None):
                raise
            allowed = observed.allowed_course_cells(self.prior_obst, to_nominal, self.need)
            path = self._plan_on(self.local_only_map(), start_xy, goals, v_start, allowed)
            if not observed.certifies_path(path.pts, to_nominal, self.need):
                raise PlanError("local-only swept footprint is not observed free")
            mode = "local_only"
        self.path, self.goals, self.t_plan = path, [np.asarray(g, float) for g in goals], t
        self.plan_mode = mode
        self.path_certificate = path.pts.copy() if mode == "local_only" else None
        self._plan_signature = self._signature()
        self.replans += 1
        return path

    def _plan_on(self, maps, start_xy, goals, v_start=0., observed_free=None):
        obst, blocked, clear = maps
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
        return path

    def plan_recovery(self, start_xy, reference, observed, to_nominal, t=0., lookahead_m=1., progress_s=None, require_detour=True):
        """Observed-free detour to an unchanged reference suffix; never clear prior walls.

        Only the changed prefix must be fully observed. The suffix remains the exact
        old route, subject to the original map and refnav's ordinary depth checks.
        """
        if reference is None or len(reference.pts) < 2:
            raise PlanError("no reference path for a bounded recovery detour")
        start_xy = np.asarray(start_xy, float)
        nearest = (int(np.searchsorted(reference.s, max(0., progress_s), side="left"))
                   if progress_s is not None else int(np.argmin(np.linalg.norm(reference.pts-start_xy, axis=1))))
        nearest = min(nearest, len(reference.pts)-1)
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
                prefix = self.plan(start_xy, goals, t=t, v_start=0., observed_free=allowed,
                                   observed=observed, to_nominal=to_nominal)
                prefix_mode = self.plan_mode
                prefix_obst = self.local_only_map()[0] if prefix_mode == "local_only" else obst
                if not observed.certifies_path(prefix.pts, to_nominal, self.need):
                    raise PlanError("detour swept footprint is not observed free")
                # New PID alone is not progress: a real detour must depart from the
                # reference segments, rather than simply densify the same corridor.
                a, b = reference.pts[:-1], reference.pts[1:]
                ab = b-a
                q = prefix.pts[:, None, :]-a[None, :, :]
                u = np.clip(np.sum(q*ab, axis=2)/np.maximum(np.sum(ab*ab, axis=1), 1e-12), 0., 1.)
                distance = np.linalg.norm(q-u[:, :, None]*ab, axis=2).min(axis=1)
                if require_detour and float(distance.max()) < max(.10, 2*self.prior_obst.res):
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
                # The certified local-only prefix must not inherit the removed prior
                # wall's zero clearance; only the unchanged suffix uses fused clearance.
                clearance[k:] = np.minimum(clearance[k:], clear[cells[k:, 0], cells[k:, 1]])
                stops = sorted(set(prefix.stops + [k] + [k+s-join for s in reference.stops if s > join]))
                path = Path(pts, arc, curvature, heading, np.zeros(len(pts)), clearance, stops)
                prefix_caps = stop_approach_caps(path, prefix_obst, self.sp)
                suffix_caps = stop_approach_caps(path, obst, self.sp)
                caps = dict(prefix_caps)
                caps.update({i: v for i, v in suffix_caps.items() if i >= k})
                speed_profile(path, self.sp, v_start=0., v_end=0., stop_caps=caps)
                self.path, self.goals, self.t_plan = path, old_goals, t
                self.plan_mode = prefix_mode
                self.path_certificate = prefix.pts.copy()
                self._plan_signature = self._signature()
                return path, prefix.pts.copy(), int(join)
            except PlanError as exc:
                last_reason = str(exc)
                self.path = None
                self.goals = old_goals
        raise PlanError(last_reason)

    def needs_replan(self, t, progress_s):
        """Trust/clearing changes invalidate plans; ordinary obstacles replan only the affected path."""
        if self.path is None:
            return False
        if not self.map_changed():
            return False
        if self.t_plan is not None and t - self.t_plan < self.p.replan_every_s:
            return False
        current = self._signature()
        if (self._plan_signature is None or current[0] != self._plan_signature[0]
                or current[3:] != self._plan_signature[3:]):
            return True
        _, _, clear = self.local_only_map() if self.plan_mode == "local_only" else self.fuse()
        ahead = self.path.pts[self.path.s >= progress_s]
        if not len(ahead):
            return False
        cells = self.prior_obst.ij(ahead)
        inside = self.prior_obst.inside(cells)
        return bool(not inside.all() or np.any(clear[cells[inside, 0], cells[inside, 1]] < self.need))
