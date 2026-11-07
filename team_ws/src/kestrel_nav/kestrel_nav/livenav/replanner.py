"""Ordered course goals -> an A* path on the live blocked grid, replanned as the grid changes. PURE.

Goals come from coursenav.planner (the published-OBJ plan: maze exit, tunnel entry/exit, chicane
pre/post points, pad). Only the GOAL SEQUENCE is reused; the route between goals is searched again
here, on the prior + live grid, every time the live grid blocks the current path (or on a timer).

Gate legs ("straight": tunnel and window crossings) are flown as a straight line from the pre point
and are never re-routed: if the live grid blocks a crossing, the replanner reports blocked and the
mission holds at the pre point instead of improvising through a 0.9 m opening.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..coursenav import planner
from ..coursenav.planner import PlanError
from ..coursenav.course import Grid2D
from .navgrid import NavGrid


@dataclass
class ReplanParams:
    replan_every_s: float = 1.0
    deviation_m: float = 0.45         # replan when the vehicle is this far off its path
    prefer_m: float = 0.6             # A* crowding cost: prefer cells with at least this clearance
    weight: float = 4.0
    simplify_step: float = 0.025
    arrive_m: float = 0.20            # goal reached
    arrive_gate_m: float = 0.15       # tighter for gate pre/post points


class Goal:
    def __init__(self, w):
        self.name = w["name"]
        self.xy = np.asarray(w["xy"], float)
        self.mode = w.get("mode", "cruise")
        self.crosses = w.get("crosses")
        self.gate = w.get("gate")
        self.land = bool(w.get("land"))
        self.passage = bool(w.get("passage", {}).get("is_passage"))
        # a gate crossing (pre -> post through the opening) is flown as ONE straight line: the inflated
        # grid cannot represent a 0.9 m window, and the planner already verified the straight segment's
        # clearance on the raw obstacle band. The tunnel (passage) is wide enough for A*.
        self.straight = bool(w.get("straight")) or self.crosses is not None
        n = w.get("normal") or (w.get("align") or {}).get("normal")
        self.normal = np.asarray(n, float) if n is not None else None

    @property
    def is_gate_leg(self):
        return self.straight or self.mode in ("gate", "tunnel") or self.crosses is not None

    def __repr__(self):
        return f"Goal({self.name}, {self.xy.round(2).tolist()}, {self.mode}{', straight' if self.straight else ''})"


def goals_from_plan(plan_result):
    """coursenav plan -> [Goal], skipping the take-off waypoint."""
    wps = planner.flight_waypoints(plan_result["waypoints"])
    return [Goal(w) for w in wps[1:]]


class Replanner:
    def __init__(self, grid: NavGrid, goals, params: ReplanParams = None, gate_rules=None):
        self.g = grid
        self.goals = list(goals)
        self.p = params or ReplanParams()
        self.gate_rules = gate_rules or {}
        empty = Grid2D(np.zeros_like(grid.blocked.occ), grid.blocked.origin, grid.blocked.res)
        # Static exclusion masks, applied after carving and to the hard fallback too.
        self.gate_masks = {
            allowed: planner.with_walls(empty, [rule['line'] for name, rule in self.gate_rules.items()
                                                if name != allowed]).occ
            for allowed in [None, *self.gate_rules]
        }
        self.k = 0
        self.path = None                  # (n,2) course frame: vehicle -> goal (-> next goal if the goal is a plain via)
        self.goal_s = None                # arc length of the current goal along self.path
        self.path_t = -math.inf
        self.blocked = False
        self.last_reason = ""
        self.replans = 0

    def _policy_grid(self, grid, crossing=None):
        return Grid2D(grid.occ | self.gate_masks[crossing], grid.origin, grid.res)

    def _segment_blocked(self, a, b, crossing=None):
        return not planner.segment_free(self._policy_grid(self.g.blocked_hard, crossing),
                                        a, b, step=self.p.simplify_step)

    def _path_allowed(self, path, crossing=None):
        if any(self._segment_blocked(a, b, crossing) for a, b in zip(path[:-1], path[1:])):
            return False
        return self._gate_path_allowed(path, crossing)

    def _gate_path_allowed(self, path, crossing=None):
        mask = Grid2D(self.gate_masks[crossing], self.g.blocked.origin, self.g.blocked.res)
        if any(not planner.segment_free(mask, a, b, step=self.p.simplify_step)
               for a, b in zip(path[:-1], path[1:])):
            return False
        for name, rule in self.gate_rules.items():
            directions = [d for _, d in planner.gate_crossings(path, rule['line'], rule['normal'])]
            if directions and (name != crossing or directions != [1]):
                return False
            # gate_crossings has no previous side when the path begins on the plane.
            # Reject departing that plane backwards into the pre side as well.
            a, b = np.asarray(rule['line'], float)
            tangent = b - a
            normal = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
            if normal @ np.asarray(rule['normal']) < 0:
                normal = -normal
            along = float((path[0] - a) @ tangent / (tangent @ tangent))
            if abs(float((path[0] - a) @ normal)) <= 1e-6 and 0 <= along <= 1:
                sides = (np.asarray(path) - a) @ normal
                nonzero = sides[np.abs(sides) > 1e-6]
                if len(nonzero) and nonzero[0] < 0:
                    return False
        return True

    def gate_blocked(self, a, b):
        """Gate-only motion guard, including HOLD/backoff; no change to inflation recovery."""
        crossing = self.goal.crosses if self.goal is not None and self.goal.straight else None
        return not self._gate_path_allowed(np.asarray([a, b]),
                                          crossing if crossing in self.gate_rules else None)

    def chord_blocked(self, a, b, hard=False):
        """Follower shortcuts obey the same gate policy as its route."""
        crossing = self.goal.crosses if self.goal is not None and self.goal.straight else None
        crossing = crossing if crossing in self.gate_rules else None
        grid = self.g.blocked_hard if hard else self.g.blocked
        if not planner.segment_free(self._policy_grid(grid, crossing), a, b,
                                    step=self.p.simplify_step):
            return True
        return not self._path_allowed(np.asarray([a, b]), crossing)

    # ------------------------------------------------------------------ goals
    @property
    def goal(self):
        return self.goals[self.k] if self.k < len(self.goals) else None

    @property
    def done(self):
        return self.k >= len(self.goals)

    def arrived(self, xy) -> bool:
        g = self.goal
        if g is None:
            return True
        tol = self.p.arrive_gate_m if g.is_gate_leg else self.p.arrive_m
        if math.dist(xy, g.xy) <= tol:
            return True
        # a plain via is also "reached" once the vehicle's projection on the path has passed it
        if self.continues and self.path is not None and self.goal_s is not None:
            from .follower import path_progress
            _, s_here, _, _, _ = path_progress(self.path, np.asarray(xy, float))
            return s_here >= self.goal_s - tol and math.dist(xy, g.xy) <= 0.6
        return False

    @property
    def continues(self):
        """The path runs on past the current goal: it is a plain cruise via, not a gate/tunnel point,
        not the pad, and the next goal is not a gate leg (gate legs start exactly at their pre point)."""
        g = self.goal
        nxt = self.goals[self.k + 1] if self.k + 1 < len(self.goals) else None
        return g is not None and nxt is not None and not g.is_gate_leg and not g.land and not nxt.is_gate_leg

    def advance(self):
        self.k += 1
        self.path = None
        self.blocked = False

    # ------------------------------------------------------------------ planning
    def _simplify(self, pts):
        """Greedy line-of-sight shortcut on the live blocked grid."""
        if len(pts) <= 2:
            return pts
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and self._segment_blocked(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        return np.asarray(out)

    def plan(self, xy, now):
        """Compute a fresh path from xy to the current goal. Sets self.blocked on failure."""
        g = self.goal
        xy = np.asarray(xy, float)
        self.path_t = now
        self.replans += 1
        if g is None:
            self.path = None
            return None
        if g.is_gate_leg and g.straight:
            # straight from here if that line is clear; otherwise back to the leg's start (the pre
            # point) and through from there; a crossing the live grid blocks either way is a HOLD
            pre = self.goals[self.k - 1].xy if self.k > 0 else xy
            crossing = g.crosses if g.crosses in self.gate_rules else None
            before_plane = True
            if crossing is not None:
                rule = self.gate_rules[crossing]
                a, b = np.asarray(rule['line'], float)
                tangent = b - a
                normal = np.array([-tangent[1], tangent[0]])
                normal /= np.linalg.norm(normal)
                if normal @ np.asarray(rule['normal']) < 0:
                    normal = -normal
                before_plane = float((xy - a) @ normal) < -1e-6
            if self._path_allowed(np.vstack([xy, g.xy]), crossing):
                self.path = np.vstack([xy, g.xy])
            elif before_plane and self._path_allowed(np.vstack([xy, pre, g.xy]), crossing):
                self.path = np.vstack([xy, pre, g.xy])
            else:
                self.path = None
            self.blocked = self.path is None
            self.last_reason = "gate crossing blocked by the live grid or gate rule" if self.blocked else ""
            self.goal_s = None if self.path is None else float(np.linalg.norm(np.diff(self.path, axis=0), axis=1).sum())
            return self.path
        start = self.g.nearest_free(xy, hard=True)
        if start is None:
            self.blocked, self.last_reason, self.path = True, "vehicle cell blocked, no free cell within reach", None
            return None
        seg = self._astar(start, g.xy, g.name, carve=True)
        if seg is None:
            return None
        pts = np.vstack([xy, seg]) if math.dist(xy, seg[0]) > 1e-6 else seg
        path = self._simplify(pts)
        self.goal_s = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        if self.continues:
            nxt = self.goals[self.k + 1]
            ext = self._astar(g.xy, nxt.xy, nxt.name)
            if ext is not None and len(ext) > 1:
                path = np.vstack([path, self._simplify(ext)[1:]])
        if not self._gate_path_allowed(path):
            self.blocked, self.last_reason, self.path = True, "path violates gate rule", None
            return None
        self.path = path
        self.blocked = False
        self.last_reason = ""
        return self.path

    def _astar(self, a_xy, b_xy, name, carve=False):
        goal_xy = self.g.nearest_free(b_xy, max_m=0.35)
        if goal_xy is None:
            self.blocked, self.last_reason, self.path = True, f"goal {name} is inside a live obstacle", None
            return None
        grid = self.g.blocked
        if carve:
            # the vehicle IS at a_xy, so that spot is free at least to the hard margin: let the search
            # start from inside the planning margin instead of jumping to the nearest soft-free cell
            grid = self.g.carved(a_xy)
        try:
            return planner.astar(self._policy_grid(grid), self.g.clearance, a_xy, goal_xy,
                                 prefer_m=self.p.prefer_m, weight=self.p.weight)
        except PlanError:
            pass
        # too tight at full margin: retry at the hard margin (the follower's corridor still guards it)
        start = self.g.nearest_free(a_xy, hard=True)
        goal_xy = self.g.nearest_free(b_xy, max_m=0.35, hard=True)
        if start is None or goal_xy is None:
            self.blocked, self.last_reason, self.path = True, f"no free cell near start/goal {name}", None
            return None
        try:
            return planner.astar(self._policy_grid(self.g.blocked_hard), self.g.clearance, start, goal_xy,
                                 prefer_m=self.p.prefer_m, weight=self.p.weight)
        except PlanError as e:
            self.blocked, self.last_reason, self.path = True, str(e), None
            return None

    def path_blocked(self):
        """Validity uses the HARD inflation: a path planned at full margin is not thrown away because a
        live return lands a few centimetres inside the planning margin."""
        if self.path is None:
            return True
        crossing = self.goal.crosses if self.goal is not None and self.goal.straight else None
        return not self._path_allowed(self.path, crossing if crossing in self.gate_rules else None)

    def deviation(self, xy):
        if self.path is None or len(self.path) < 2:
            return math.inf
        return float(min(_point_segment_dist(xy, a, b) for a, b in zip(self.path[:-1], self.path[1:])))

    def update(self, xy, now):
        """Keep a valid path from xy to the goal; replan when needed. Returns the path (or None)."""
        if self.goal is None:
            return None
        if self.path is None or now - self.path_t >= self.p.replan_every_s \
                or self.deviation(xy) > self.p.deviation_m or self.path_blocked():
            return self.plan(xy, now)
        return self.path


def _point_segment_dist(p, a, b):
    p, a, b = np.asarray(p, float), np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    L2 = float(ab @ ab)
    t = 0.0 if L2 == 0 else float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))
