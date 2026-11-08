"""Path smoothing and the time-optimal speed profile. PURE (numpy).

A* on the grid gives a polyline of cell centres; flying it as-is means a stop and an in-place turn
at every corner. This module turns it into a path the follower can fly continuously:

  1. prune      line-of-sight shortcuts (A* post-smoothing, cf. Theta*): drop every vertex the
                previous kept vertex can see with the full clearance `need` (prop tip + margin);
  2. corners    each remaining corner becomes a circular arc tangent to both segments. The radius is
                the LARGEST in [r_min, r_max] whose arc keeps the full clearance on the clearance map
                and whose tangent points fit on the two segments (at most half of each, so
                neighbouring arcs never overlap). The deviation from the corner is therefore bounded
                by the map, not by a tuning knob: smoothing can never cut a wall. A corner where even
                r_min does not fit stays sharp and is flagged as a STOP (stop, turn in place, go);
  3. profile    the time-optimal speed along the path for a point mass (the special case of TOPP /
                TOPP-RA with decoupled limits): v(s) <= cruise, <= sqrt(a_lat / kappa) on arcs,
                <= yaw_rate / kappa (the camera faces the direction of travel, so the yaw rate is
                v * kappa), <= a clearance-regulated cap near obstacles (regulated pure pursuit), 0 at
                stops; then a forward pass (acceleration) and a backward pass (braking). Fastest
                profile under those limits, with no overshoot by construction.

Frames: any metric 2D frame (the course frame here). Units: m, s, rad.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SmoothParams:
    need_m: float = 0.40            # required clearance: vehicle radius + margin (course vehicle block)
    prefer_extra_m: float = 0.08    # shortcuts and arcs first try for need + this (where the space allows):
                                    # any-angle shortcuts otherwise skim the buffer's edge even in a roomy
                                    # corridor, spending the whole margin on the geometry
    r_min_m: float = 0.15           # smallest arc radius before a corner is flown as a stop-and-turn
    r_max_m: float = 1.5            # largest arc radius tried
    ds_m: float = 0.05              # densification step of the output path
    cruise_mps: float = 0.6
    accel_mps2: float = 0.5         # along-path acceleration
    brake_mps2: float = 0.4         # along-path braking (kept below accel: PX4 lags when braking)
    lat_accel_mps2: float = 0.5     # centripetal limit on arcs
    yaw_rate_dps: float = 45.0      # heading follows the path tangent: v * kappa <= this
    slow_clear_m: float = 0.30      # within need + this of an obstacle, the cap falls ...
    slow_min_mps: float = 0.25      # ... linearly towards this at exactly `need`
    stop_turn_deg: float = 100.0    # corners sharper than this are always stop-and-turn
    center_stops: bool = False      # move each stop-and-turn vertex to the highest-clearance point of its corner
                                    # region (center_stop_vertices), so the in-place turn has the most room
    center_face_min_m: float = 0.8  # ... never facing the next leg's first surface closer than this (or than the
                                    # original vertex did): leg_policy face_min_m, the depth camera rule
    arc_min_tube_m: float = 0.0     # > 0 (with reach_m): a corner is flown as an arc only if every point of the
                                    # arc keeps a tube (clearance - reach) / 2 of at least this; else stop-and-turn.
                                    # An arc yaws while moving, which adds cross-track error the tube must hold
    crab_max_deg: float = 0.0       # > 0: CRAB. Consecutive segments whose directions all lie within +- this of
                                    # one heading form a run flown at that FIXED heading (the aircraft slides
                                    # sideways by rolling, it does not yaw); only run boundaries are in-place
                                    # turns. Corners inside a run are arcs without yaw (or yawless stops when no
                                    # arc fits). Bounded by the depth camera: the travel direction plus the
                                    # corridor must stay in view (refnav checks the corridor along the travel)
    # flow profile (flow-corners branch): room-based speed and stop approach, from the cloud runs' data
    reach_m: float = 0.0            # > 0: rotor reach; tube = (clearance - reach) / 2 per point (planner.leg_tube_m)
    tube_response_s: float = 0.0    # > 0: v <= tube / this (cross-track p90 = 0.44 s * v, steady run015)
    overshoot_a: float = 0.0        # stop overshoot fit a + b * v (runs 015+016) + margin: the approach to a
    overshoot_b: float = 0.0        # stop is slowed so the overshoot plus the reach fits the room beyond it
    overshoot_margin_m: float = 0.0
    stop_approach_m: float = 1.0    # how far before a stop that approach cap applies


@dataclass
class Path:
    pts: np.ndarray                 # (N, 2) points every ~ds_m
    s: np.ndarray                   # (N,) arc length from the start
    kappa: np.ndarray               # (N,) curvature (1/m), 0 on straight parts
    heading: np.ndarray             # (N,) tangent direction (rad, atan2(dy, dx))
    v: np.ndarray                   # (N,) speed profile (m/s), 0 at the end
    clearance: np.ndarray           # (N,) obstacle clearance along the path (m)
    stops: list = field(default_factory=list)   # indices of sharp corners flown as stop-and-turn
    tube: np.ndarray = None         # (N,) cross-track tube (m) for refnav's gate, when reach_m > 0
    stop_vertices: list = field(default_factory=list)   # the input polyline's vertices that became stops
    yaw: np.ndarray = None          # (N,) CRAB: the fixed heading to hold at each point (rad); None = the tangent
    turn_stops: list = field(default_factory=list)      # CRAB: the stops where the heading changes (the rest
                                                        # of `stops` are yawless: brake, go on sideways)

    @property
    def length(self):
        return float(self.s[-1]) if len(self.s) else 0.0

    def duration(self):
        v = np.maximum(0.5 * (self.v[1:] + self.v[:-1]), 1e-3)
        return float(np.sum(np.diff(self.s) / v))


def _clear_at(grid, clearance, pts):
    """Clearance (m) at points; 0 off the map."""
    pts = np.atleast_2d(pts)
    ij = grid.ij(pts)
    inside = grid.inside(ij)
    out = np.zeros(len(pts))
    if np.any(inside):
        out[inside] = clearance[ij[inside, 0], ij[inside, 1]]
    return out


def segment_min_clearance(grid, clearance, p, q, step=0.02):
    n = max(2, int(math.ceil(math.dist(p, q) / step)) + 1)
    return float(_clear_at(grid, clearance, np.linspace(p, q, n)).min())


def prune(poly, grid, clearance, need_m, keep=(), prefer_m=None):
    """Greedy line-of-sight shortcut: from each kept vertex jump to the furthest later vertex visible
    with clearance >= need_m. With prefer_m (> need_m), a jump that also keeps prefer_m is taken instead
    when it is a real shortcut (at least 0.4 m, or half the need_m jump): roomy where the space allows,
    never a staircase of grid steps where it does not. Vertices in `keep` (indices) are never removed."""
    poly = np.asarray(poly, float)
    if len(poly) <= 2:
        return poly.copy()
    keep = set(int(k) for k in keep) | {0, len(poly) - 1}
    out = [0]
    i = 0

    def furthest(i, lim, need):
        j = lim
        while j > i + 1 and segment_min_clearance(grid, clearance, poly[i], poly[j]) < need:
            j -= 1
        return j
    while i < len(poly) - 1:
        lim = min(k for k in keep if k > i)
        j = furthest(i, lim, need_m)
        if prefer_m is not None and prefer_m > need_m and j > i + 1:
            jp = furthest(i, j, prefer_m)
            d_need, d_pref = math.dist(poly[i], poly[j]), math.dist(poly[i], poly[jp])
            if jp > i + 1 and (d_pref >= 0.4 or d_pref >= 0.5 * d_need):
                j = jp
        out.append(j)
        i = j
    return poly[out]


def _arc_points(c, r, a0, a1, ccw, ds):
    sweep = (a1 - a0) % (2 * math.pi) if ccw else -((a0 - a1) % (2 * math.pi))
    n = max(2, int(math.ceil(abs(sweep) * r / ds)) + 1)
    ang = a0 + np.linspace(0.0, sweep, n)
    return np.stack([c[0] + r * np.cos(ang), c[1] + r * np.sin(ang)], axis=1)


def _corner_arc(p0, p1, p2, r, ds):
    """Arc of radius r tangent to p0->p1 and p1->p2 at the corner p1. -> (points, t_len, turn) or None."""
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    w = (p2 - p1) / np.linalg.norm(p2 - p1)
    turn = math.atan2(u[0] * w[1] - u[1] * w[0], float(np.dot(u, w)))    # signed heading change
    if abs(turn) < 1e-6:
        return None
    t = r * math.tan(abs(turn) / 2.0)
    a = p1 - u * t                                                       # tangent point on the first segment
    b = p1 + w * t                                                       # ... on the second
    left = np.array([-u[1], u[0]])
    ccw = turn > 0
    c = a + (left if ccw else -left) * r
    a0 = math.atan2(a[1] - c[1], a[0] - c[0])
    a1 = math.atan2(b[1] - c[1], b[0] - c[0])
    return _arc_points(c, r, a0, a1, ccw, ds), t, turn


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def crab_runs(poly, crab_max_deg, boundary_cost, forced=()):
    """CRAB: partition the polyline's segments into runs, each flown at one heading that keeps every segment's
    direction within +-crab_max_deg of the nose (the aircraft slides sideways; it does not yaw). Dynamic
    programming over contiguous partitions: minimise the sum of boundary_cost[v] over the vertices v where a
    run ends (a corner the aircraft must stop at costs a stop; one it can fly as an arc costs little), and
    then the number of runs. A vertex in `forced` (a gate's pre point: stop and align) always ends a run.
    Headings: the first run takes the middle of its feasible interval; each later run keeps the previous
    heading when it is feasible, else moves to the nearest edge of its interval (the least turning).
    -> (run id per segment, heading per run (rad))."""
    poly = np.asarray(poly, float)
    d = np.diff(poly, axis=0)
    h = np.arctan2(d[:, 1], d[:, 0])
    m = len(h)
    lim = math.radians(2.0 * crab_max_deg)
    best, prev, span = [math.inf] * (m + 1), [-1] * (m + 1), [None] * (m + 1)
    best[0] = 0.0
    for j in range(1, m + 1):                       # a run of segments i .. j-1
        ref, lo, hi = h[j - 1], 0.0, 0.0
        for i in range(j - 1, -1, -1):
            off = _wrap(h[i] - ref)
            lo, hi = min(lo, off), max(hi, off)
            if hi - lo > lim or (i + 1 <= j - 1 and (i + 1) in forced):
                break
            c = best[i] + (boundary_cost.get(i, 1.0) if i > 0 else 0.0) + 1e-3
            if c < best[j]:
                best[j], prev[j], span[j] = c, i, (ref, lo, hi)
    runs, j = [], m
    while j > 0:
        runs.append((prev[j], j, span[j]))
        j = prev[j]
    runs.reverse()
    run_of, heads, cur = np.zeros(m, int), [], None
    half = math.radians(crab_max_deg)
    for r, (i, j, (ref, lo, hi)) in enumerate(runs):
        run_of[i:j] = r
        centre, w = ref + 0.5 * (lo + hi), max(half - 0.5 * (hi - lo), 0.0)   # feasible: centre +- w
        if cur is None:
            cur = centre
        else:
            cur = centre + max(-w, min(w, _wrap(cur - centre)))
        heads.append(_wrap(cur))
    return run_of, heads


def _fit_arc(poly, k, seg_len, turn, grid, clearance, p):
    """The largest corner arc at vertex k (radius r_max_m down to r_min_m) whose chords keep need_m (the
    roomier need_m + prefer_extra_m first). -> (points, tangent length, radius, signed turn) or None."""
    t_max = 0.5 * min(seg_len[k - 1], seg_len[k])
    r0 = min(p.r_max_m, t_max / math.tan(turn / 2.0))
    for need in (p.need_m + max(p.prefer_extra_m, 0.0), p.need_m):     # roomy first, the buffer at least
        r = r0
        while r >= p.r_min_m - 1e-9:
            arc = _corner_arc(poly[k - 1], poly[k], poly[k + 1], r, p.ds_m)
            # the arc is flown as chords between its points: certify the chords, not only the points
            # (a chord can cut a blocked cell's corner between two clear points)
            if arc is not None and min(segment_min_clearance(grid, clearance, a, b)
                                       for a, b in zip(arc[0][:-1], arc[0][1:])) >= need:
                return (arc[0], arc[1], r, arc[2])
            r *= 0.8
    return None


def _turn_at(poly, k):
    u = poly[k] - poly[k - 1]
    w = poly[k + 1] - poly[k]
    return abs(math.atan2(u[0] * w[1] - u[1] * w[0], float(np.dot(u, w))))


def _corners(poly, seg_len, grid, clearance, p, keep_sharp):
    """The heading-follows-the-tangent decision per vertex: {k: arc} flown while yawing, and the set of
    stop-and-turn vertices."""
    n = len(poly)
    arcs, stops = {}, set()
    for k in range(1, n - 1):
        turn = _turn_at(poly, k)
        if turn < math.radians(1.0):
            continue
        if k in keep_sharp or turn > math.radians(p.stop_turn_deg):
            stops.add(k)
            continue
        found = _fit_arc(poly, k, seg_len, turn, grid, clearance, p)
        if found is not None and p.arc_min_tube_m > 0 and p.reach_m > 0:
            arc_tube = (_clear_at(grid, clearance, found[0]).min() - p.reach_m) / 2.0
            if arc_tube < p.arc_min_tube_m:
                found = None                              # too narrow to yaw while moving: stop and turn
        if found is None:
            stops.add(k)                                  # no arc fits with full clearance: stop and turn
        else:
            arcs[k] = found
    # adjacent arcs count as one bend (GROUP119_NEXT item 2): two arcs whose tangent points meet (no
    # straight between them) and that together turn more than stop_turn_deg are flown as one sharp bend,
    # so both become stop-and-turns (a merged 75 deg arc slipped past the per-corner 60 deg rule)
    for k in sorted(arcs):
        if k + 1 in arcs and k in arcs:
            touching = arcs[k][1] + arcs[k + 1][1] >= seg_len[k] - 0.05
            same_way = arcs[k][3] * arcs[k + 1][3] > 0
            if touching and same_way and abs(arcs[k][3]) + abs(arcs[k + 1][3]) > math.radians(p.stop_turn_deg):
                for j in (k, k + 1):
                    arcs.pop(j, None)
                    stops.add(j)
    return arcs, stops


def smooth(poly, grid, clearance, p: SmoothParams = None, keep_sharp=()):
    """Polyline (after prune) -> dense points with curvature and stop flags. Corners in keep_sharp
    (vertex indices, e.g. a gate's pre point where the aircraft must stop and align) stay sharp.
    CRAB (crab_max_deg > 0): runs of segments flown at a fixed heading (crab_runs). Inside a run every
    corner is an arc without yaw (r_min_m and the clearance only: the yaw rules do not apply) or, when
    none fits, a yawless stop. Between runs the corner keeps what the tangent-following rules decide: an arc
    (the heading turns from one run's to the next while moving, as before) or a stop-and-turn."""
    p = p or SmoothParams()
    poly = np.asarray(poly, float)
    n = len(poly)
    if n < 2:
        raise ValueError("a path needs at least two points")
    seg_len = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    keep_sharp = set(keep_sharp)
    arcs, stops = _corners(poly, seg_len, grid, clearance, p, keep_sharp)
    crab = p.crab_max_deg > 0
    flow = set()                                          # CRAB: run boundaries flown as arcs (yaw while moving)
    turn_vertices = set(stops)
    if crab:
        cost = {k: (1.0 if k in stops else 0.05 if k in arcs else 1.0) for k in range(1, n - 1)}
        run_of, run_head = crab_runs(poly, p.crab_max_deg, cost, forced=keep_sharp)
        t_arcs = arcs
        arcs, stops, turn_vertices = {}, set(), set()
        for k in range(1, n - 1):
            if run_of[k - 1] != run_of[k]:                # a boundary: the tangent-following decision
                if k in t_arcs:
                    arcs[k] = t_arcs[k]
                    flow.add(k)
                else:
                    stops.add(k)
                    turn_vertices.add(k)
                continue
            turn = _turn_at(poly, k)                      # inside a run: no yaw
            if turn < math.radians(1.0):
                continue
            found = _fit_arc(poly, k, seg_len, turn, grid, clearance, p)
            if found is None:
                stops.add(k)                              # yawless stop: brake, go on sideways
            else:
                arcs[k] = found
    # assemble: straight pieces between tangent points, arcs at corners
    pts, kap, stop_idx, yaw, turn_idx = [poly[0][None, :]], [np.zeros(1)], [], [], []
    if crab:
        yaw.append(np.full(1, run_head[run_of[0]]))
    for k in range(1, n):
        start = poly[k - 1] + (poly[k] - poly[k - 1]) / seg_len[k - 1] * (arcs[k - 1][1] if (k - 1) in arcs else 0.0)
        end = poly[k] - (poly[k] - poly[k - 1]) / seg_len[k - 1] * (arcs[k][1] if k in arcs else 0.0)
        L = float(np.linalg.norm(end - start))
        m = max(1, int(math.ceil(L / p.ds_m)))
        line = np.linspace(start, end, m + 1)[1:]
        pts.append(line)
        kap.append(np.zeros(len(line)))
        if crab:
            yaw.append(np.full(len(line), run_head[run_of[k - 1]]))
        if k in arcs:
            arc_pts, _, r, turn = arcs[k]
            pts.append(arc_pts[1:])
            kap.append(np.full(len(arc_pts) - 1, math.copysign(1.0 / r, turn)))
            if crab:
                a0 = run_head[run_of[k - 1]]
                if k in flow:                             # the heading turns to the next run's along the arc
                    dy = _wrap(run_head[run_of[k]] - a0)
                    f = np.arange(1, len(arc_pts)) / max(len(arc_pts) - 1, 1)
                    yaw.append(np.array([_wrap(a0 + dy * x) for x in f]))
                else:
                    yaw.append(np.full(len(arc_pts) - 1, a0))
        if k in stops:
            stop_idx.append(sum(len(a) for a in pts) - 1)
            if k in turn_vertices:
                turn_idx.append(stop_idx[-1])
    P = np.concatenate(pts)
    K = np.concatenate(kap)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-9])
    idx_map = np.cumsum(keep) - 1
    P, K = P[keep], K[keep]
    Y = np.concatenate(yaw)[keep] if crab else None
    stop_idx = sorted({int(idx_map[i]) for i in stop_idx})
    turn_idx = sorted({int(idx_map[i]) for i in turn_idx})
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    d = np.gradient(P, axis=0)
    heading = np.arctan2(d[:, 1], d[:, 0])
    clr = _clear_at(grid, clearance, P)
    return Path(P, s, K, heading, np.zeros(len(P)), clr, stop_idx, stop_vertices=sorted(stops), yaw=Y,
                turn_stops=turn_idx)


def facing_room(grid, p, heading, max_m=3.0, step=None):
    """Distance from p along heading to the first obstacle cell of grid (inf beyond max_m), less half a
    cell diagonal (the raster's worst case)."""
    step = step or grid.res / 2.0
    d = np.array([math.cos(heading), math.sin(heading)])
    for k in range(1, int(max_m / step) + 1):
        q = np.asarray(p, float) + d * k * step
        ij = grid.ij(q)
        if not grid.inside(ij) or grid.occ[tuple(ij)]:
            return k * step - grid.res * math.sqrt(2.0) / 2.0
    return math.inf


def stop_approach_caps(path: Path, grid, p: SmoothParams):
    """{stop index: fastest approach whose predicted overshoot (a + b v + margin) plus the reach fits the
    room ahead} for every interior stop and the end (planner.end_room_speed_mps, applied to a path)."""
    if p.overshoot_b <= 0 or p.reach_m <= 0:
        return {}
    caps = {}
    for i in list(path.stops) + [len(path.pts) - 1]:
        if i <= 0:
            continue
        j = max(0, i - 3)
        h = math.atan2(path.pts[i][1] - path.pts[j][1], path.pts[i][0] - path.pts[j][0])
        room = facing_room(grid, path.pts[i], h)
        if math.isfinite(room):
            caps[i] = max(0.05, (room - p.reach_m - p.overshoot_margin_m - p.overshoot_a) / p.overshoot_b)
    return caps


def speed_profile(path: Path, p: SmoothParams = None, v_start=0.0, v_end=0.0, stop_caps=None):
    """Time-optimal speeds along `path` under the limits in SmoothParams (forward + backward pass)."""
    p = p or SmoothParams()
    n = len(path.s)
    cap = np.full(n, p.cruise_mps)
    k = np.abs(path.kappa)
    curved = k > 1e-9
    cap[curved] = np.minimum(cap[curved], np.sqrt(p.lat_accel_mps2 / k[curved]))
    if path.yaw is None:
        cap[curved] = np.minimum(cap[curved], math.radians(p.yaw_rate_dps) / k[curved])
    else:                                                   # CRAB: only where the held heading itself turns
        dyaw = np.abs(np.angle(np.exp(1j * np.diff(path.yaw)))) / np.maximum(np.diff(path.s), 1e-6)
        rate = np.concatenate([dyaw, [0.0]])
        turning = rate > 1e-6
        cap[turning] = np.minimum(cap[turning], math.radians(p.yaw_rate_dps) / rate[turning])
    if p.slow_clear_m > 0:
        f = np.clip((path.clearance - p.need_m) / p.slow_clear_m, 0.0, 1.0)
        cap = np.minimum(cap, p.slow_min_mps + (p.cruise_mps - p.slow_min_mps) * f)
    if p.reach_m > 0:
        path.tube = np.maximum((path.clearance - p.reach_m) / 2.0, 0.0)
        if p.tube_response_s > 0:
            cap = np.minimum(cap, path.tube / p.tube_response_s)
    for i in path.stops:
        cap[i] = 0.0
    if stop_caps:
        for i, vmax in stop_caps.items():
            near = (path.s <= path.s[i]) & (path.s >= path.s[i] - p.stop_approach_m)
            cap[near] = np.minimum(cap[near], vmax)
    cap[0] = min(cap[0], v_start)
    cap[-1] = min(cap[-1], v_end)
    v = cap.copy()
    ds = np.diff(path.s)
    for i in range(1, n):                                   # forward: what acceleration allows
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2.0 * p.accel_mps2 * ds[i - 1]))
    for i in range(n - 2, -1, -1):                          # backward: what braking allows
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * p.brake_mps2 * ds[i]))
    path.v = v
    return path


def center_stop_vertices(poly, stop_vertices, grid, clearance, p: SmoothParams, fixed=()):
    """Move each stop vertex (not an endpoint, not in `fixed`) to the point of greatest clearance in its
    corner region: within half the shorter adjacent segment (so it stays the same corner). A candidate
    must keep both adjacent segments at the full requirement (need_m) and no worse than before, and must
    face the outgoing leg's first surface no closer than min(center_face_min_m, the original vertex's
    facing room). Pure geometry from the clearance field; no positions are stored.
    -> (new polyline, number of vertices moved)."""
    poly = np.array(poly, float)
    moved = 0
    for k in sorted(stop_vertices):
        if k <= 0 or k >= len(poly) - 1 or k in fixed:
            continue
        a, v0, b = poly[k - 1], poly[k], poly[k + 1]
        radius = 0.5 * min(np.linalg.norm(v0 - a), np.linalg.norm(b - v0))
        if radius < grid.res:
            continue
        c0 = float(_clear_at(grid, clearance, v0)[0])
        seg_need = max(p.need_m, min(segment_min_clearance(grid, clearance, a, v0),
                                     segment_min_clearance(grid, clearance, v0, b)))
        face0 = facing_room(grid, v0, math.atan2(b[1] - v0[1], b[0] - v0[0]))
        face_need = min(p.center_face_min_m, face0)
        i0, j0 = grid.ij(v0)
        r = int(math.ceil(radius / grid.res))
        best, best_c = None, c0 + 1e-6
        for i in range(i0 - r, i0 + r + 1):
            for j in range(j0 - r, j0 + r + 1):
                q = grid.xy(np.array([i, j]))
                if np.linalg.norm(q - v0) > radius or not grid.inside(np.array([i, j])):
                    continue
                cq = float(clearance[i, j])
                if cq <= best_c:
                    continue
                if (segment_min_clearance(grid, clearance, a, q) < seg_need
                        or segment_min_clearance(grid, clearance, q, b) < seg_need):
                    continue
                if facing_room(grid, q, math.atan2(b[1] - q[1], b[0] - q[0])) < face_need:
                    continue
                best, best_c = q, cq
        if best is not None:
            poly[k] = best
            moved += 1
    return poly, moved


def build(poly, grid, clearance, p: SmoothParams = None, keep=(), keep_sharp=(), v_start=0.0, v_end=0.0):
    """A* polyline -> pruned, corner-smoothed, speed-profiled Path. `keep`: vertex indices of the
    input polyline that must stay (goals); `keep_sharp`: input indices that must stay sharp stops."""
    p = p or SmoothParams()
    poly = np.asarray(poly, float)
    keep = set(keep) | set(keep_sharp)
    pruned = prune(poly, grid, clearance, p.need_m, keep=keep, prefer_m=p.need_m + max(p.prefer_extra_m, 0.0))
    # map keep_sharp onto the pruned polyline (kept vertices survive pruning by construction)
    sharp = [int(np.argmin(np.linalg.norm(pruned - poly[i], axis=1))) for i in keep_sharp]
    path = smooth(pruned, grid, clearance, p, keep_sharp=sharp)
    if p.center_stops and path.stop_vertices:
        fixed = {int(np.argmin(np.linalg.norm(pruned - poly[i], axis=1))) for i in keep}
        centred, n = center_stop_vertices(pruned, path.stop_vertices, grid, clearance, p, fixed=fixed)
        if n:
            path = smooth(centred, grid, clearance, p, keep_sharp=sharp)
    return speed_profile(path, p, v_start=v_start, v_end=v_end, stop_caps=stop_approach_caps(path, grid, p))
