"""Course planner: A* with a corridor-centring cost through an ordered list of goals. PURE.

Goals come from the course file (rule 9.5): checkpoints, a straight tunnel axis and a chicane gate
(window or Gate B), each expanded to pre/centre/post points along its crossing direction.

What this module decides, all from the published map and the course file's policy numbers:
  * the chicane option: every option is planned and scored (checkpoint points, estimated flight
    time, clearance and leg-end penalties); the best wins and the scores are returned (plan_options);
  * a gate's crossing direction: searched within +-cross_search_deg of the gap's perpendicular for
    the direction with the most clearance whose approach does not end facing a surface (no typed-in
    angle);
  * leg ends: no leg may end where a mapped surface lies within face_min_m along the next heading
    (the depth camera's blind range: flight E, 2026-11-06) or within arc_min_m anywhere in the arc the
    in-place turn sweeps; via points that break this are moved back along their incoming leg;
  * gate openings are walls to A*: a gate is crossed only by its own pre -> centre -> post legs, and
    the route must cross it exactly once in the chosen direction, or the option is refused (a route
    that flies through the gap and back, as Flight E did at Gate B, is never produced);
  * nearly collinear legs are merged and short legs avoided;
  * per-leg metadata for refnav: gate alignment geometry (how far ahead the opening's plane is, how
    wide it is, its sill/header relative to the body) and passage legs (an enclosed straight
    passage detected from the map's geometry: a roof above and walls on both sides).
"""
from __future__ import annotations

import heapq
import math

import numpy as np

from .course import Grid2D

SQ2 = math.sqrt(2.0)
NBRS = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2)]
CAMERA_FORWARD_M = 0.116            # the depth camera sits this far ahead of the body origin (TF, 2026-11-05)
CAMERA_HALF_FOV_DEG = 35.0          # usable horizontal half field of view after refnav's border crop


class PlanError(RuntimeError):
    pass


# ---------------------------------------------------------------------------- grid search
def astar(blocked: Grid2D, clearance, a_xy, b_xy, prefer_m=0.6, weight=4.0):
    """Shortest path on 8-connected cells, each step costing length * (1 + weight * crowding),
    crowding = how far inside prefer_m of an obstacle the cell is (0..1). -> (n,2) xy."""
    occ = blocked.occ
    nx, ny = occ.shape
    a, b = tuple(blocked.ij(a_xy)), tuple(blocked.ij(b_xy))
    for name, c in (("start", a), ("goal", b)):
        if not blocked.inside(np.array(c)) or occ[c]:
            raise PlanError(f"{name} {a_xy if name == 'start' else b_xy} is blocked or off the map")
    crowd = np.clip((prefer_m - clearance) / prefer_m, 0.0, 1.0)
    g = np.full(occ.shape, np.inf)
    parent = {}
    g[a] = 0.0
    h = lambda c: math.hypot(c[0] - b[0], c[1] - b[1])   # noqa: E731  admissible (cost >= length)
    pq = [(h(a), a)]
    done = np.zeros(occ.shape, bool)
    while pq:
        _, c = heapq.heappop(pq)
        if done[c]:
            continue
        if c == b:
            break
        done[c] = True
        for dx, dy, step in NBRS:
            n = (c[0] + dx, c[1] + dy)
            if not (0 <= n[0] < nx and 0 <= n[1] < ny) or occ[n] or done[n]:
                continue
            if dx and dy and (occ[c[0] + dx, c[1]] or occ[c[0], c[1] + dy]):
                continue                                   # no corner cutting
            cost = g[c] + step * (1.0 + weight * 0.5 * (crowd[c] + crowd[n]))
            if cost < g[n]:
                g[n] = cost
                parent[n] = c
                heapq.heappush(pq, (cost + h(n), n))
    if not np.isfinite(g[b]):
        raise PlanError(f"no path from {a_xy} to {b_xy}")
    cells = [b]
    while cells[-1] != a:
        cells.append(parent[cells[-1]])
    pts = blocked.xy(np.array(cells[::-1]))
    pts[0], pts[-1] = a_xy, b_xy
    return pts


def segment_clearance(clear_grid: Grid2D, clearance, p, q, step=0.02):
    """Minimum obstacle clearance (m) sampled along p->q."""
    n = max(2, int(math.ceil(math.dist(p, q) / step)) + 1)
    pts = np.linspace(p, q, n)
    ij = clear_grid.ij(pts)
    if not np.all(clear_grid.inside(ij)):
        return 0.0
    return float(clearance[ij[:, 0], ij[:, 1]].min())


def segment_free(blocked: Grid2D, p, q, step=0.02):
    """No blocked cell (inflated obstacle or keep-out, e.g. the boulders below the flight band) along p->q."""
    n = max(2, int(math.ceil(math.dist(p, q) / step)) + 1)
    ij = blocked.ij(np.linspace(p, q, n))
    if not np.all(blocked.inside(ij)):
        return False
    return not blocked.occ[ij[:, 0], ij[:, 1]].any()


def facing_distance(obst: Grid2D, p, heading_rad, max_m=3.0, step=None):
    """Distance from p along a heading to the first mapped surface in the obstacle band (inf: none
    within max_m). What the depth camera will face when the aircraft points that way."""
    step = step or obst.res * 0.25
    c, s = math.cos(heading_rad), math.sin(heading_rad)
    for t in np.arange(step, max_m, step):
        ij = obst.ij((p[0] + t * c, p[1] + t * s))
        if not obst.inside(ij):
            return math.inf
        if obst.occ[ij[0], ij[1]]:
            return float(t)
    return math.inf


def corridor_distance(obst: Grid2D, p, heading_rad, half_width_m, max_m=3.0, rmin=0.0):
    """Distance from p along a heading to the nearest mapped cell inside a corridor +-half_width_m
    wide (inf: none within max_m). What refnav's corridor check sees along that heading."""
    occ = getattr(obst, "_occ_xy", None)
    if occ is None:
        occ = obst.xy(np.argwhere(obst.occ))
        obst._occ_xy = occ
    if len(occ) == 0:
        return math.inf
    u = np.array([math.cos(heading_rad), math.sin(heading_rad)])
    rel = occ - np.asarray(p, float)
    along = rel @ u
    across = rel @ np.array([-u[1], u[0]])
    sel = (along >= rmin) & (along <= max_m) & (np.abs(across) < half_width_m)
    return float(along[sel].min()) if sel.any() else math.inf


def arc_min_distance(obst: Grid2D, p, h_from, h_to, step_deg=5.0, max_m=3.0):
    """Minimum facing distance over every heading of the in-place turn from h_from to h_to."""
    d = _wrap(h_to - h_from)
    n = max(1, int(math.ceil(abs(math.degrees(d)) / step_deg)))
    return min(facing_distance(obst, p, h_from + d * k / n, max_m) for k in range(n + 1))


def opening_edges(obst: Grid2D, centre, direction, slab_m=0.2, max_m=2.0):
    """The opening around centre in the plane perpendicular to direction: distance from centre to the
    nearest mapped cell on the left and on the right within +-slab_m of that plane (m; inf: none
    within max_m). Works for jambs, a wall END beside a gap and a tube's rim alike."""
    occ_xy = obst.xy(np.argwhere(obst.occ))
    if len(occ_xy) == 0:
        return math.inf, math.inf
    u = _unit(direction)
    left_dir = np.array([-u[1], u[0]])
    rel = occ_xy - np.asarray(centre, float)
    along, lat = rel @ u, rel @ left_dir
    in_slab = np.abs(along) <= slab_m
    left = lat[in_slab & (lat > 0) & (lat <= max_m)]
    right = -lat[in_slab & (lat < 0) & (lat >= -max_m)]
    return (float(left.min()) if len(left) else math.inf), (float(right.min()) if len(right) else math.inf)


def free_width(obst: Grid2D, centre, direction, max_m=2.0):
    """How far free space extends from centre to the left and to the right of a direction (m)."""
    h = math.atan2(direction[1], direction[0])
    left = facing_distance(obst, centre, h + math.pi / 2, max_m)
    right = facing_distance(obst, centre, h - math.pi / 2, max_m)
    return left, right


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------- gate openings as walls
def gate_segment(gate):
    """The gate's opening as a 2D line segment (a, b), from the resolved chicane option.
    Gate B (a gap past a wall end): from the wall end through the centre to the far side.
    Window (an opening in a wall): centre +- width/2 along the wall. The chosen crossing direction may
    differ from the opening's own normal; the opening stays where it is."""
    c = np.asarray(gate["center"], float)
    if "wall_end_xy" in gate:
        a = np.asarray(gate["wall_end_xy"], float)
        return a, 2.0 * c - a
    n = _unit(gate["normal"])
    t = np.array([-n[1], n[0]])
    h = 0.5 * float(gate["width_m"])
    return c - t * h, c + t * h


def gate_crossings(xy, line, cross_normal):
    """Times the polyline `xy` passes through the gate segment `line` = (a, b).
    -> list of (leg index, +1 crossing in the declared direction / -1 against it). A pass that goes
    around the end of the segment is not a crossing."""
    a, b = (np.asarray(p, float) for p in line)
    u = b - a
    L = float(np.linalg.norm(u))
    u = u / L
    nl = np.array([-u[1], u[0]])
    if float(nl @ np.asarray(cross_normal, float)) < 0:
        nl = -nl                                     # +side = the side the declared crossing ends on
    out = []
    pts = [np.asarray(p, float) for p in xy]
    side_prev = 0.0
    for k in range(len(pts) - 1):
        p, q = pts[k], pts[k + 1]
        sp, sq = float((p - a) @ nl), float((q - a) @ nl)
        if abs(sp) > 1e-9:
            side_prev = sp
        if abs(sq) < 1e-9 or side_prev == 0.0 or np.sign(sq) == np.sign(side_prev):
            continue
        # the side changed on this leg (possibly via a point exactly on the line): where?
        f = sp / (sp - sq) if abs(sp - sq) > 1e-12 and abs(sp) > 1e-9 else 0.0
        hit = p + f * (q - p)
        along = float((hit - a) @ u)
        if 0.0 <= along <= L:
            out.append((k, 1 if sq > 0 else -1))
        side_prev = sq
    return out


def with_walls(blocked: Grid2D, lines):
    """A copy of `blocked` with every line segment drawn in as a wall (dilated one cell). Used with the
    gate openings so A* (the planner's and the navigator's) reaches a gate only by its own
    pre -> centre -> post legs, never by flying through it on the way."""
    if not lines:
        return blocked
    occ = blocked.occ.copy()
    for a, b in lines:
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(2, int(math.ceil(np.linalg.norm(b - a) / (blocked.res / 3))) + 1)
        ij = blocked.ij(np.linspace(a, b, n))
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                c = ij + np.array([di, dj])
                c = c[blocked.inside(c)]
                occ[c[:, 0], c[:, 1]] = True
    return Grid2D(occ, blocked.origin, blocked.res)


# ---------------------------------------------------------------------------- goals and gates
def gate_crossing(doc, key, obst, clearance, search_deg=None, step_deg=5.0):
    """Choose a gate's crossing direction: within +-search_deg of the gap's perpendicular (the derived
    normal), the direction whose straight pre->post crossing has the most clearance, whose pre point
    does not face a surface closer than face_min_m through the gap, nearest the perpendicular on ties.
    -> (unit normal, info dict)."""
    gate = doc["chicane_options"][key]
    c, n0 = np.asarray(gate["center"], float), _unit(gate["normal"])
    pre_d, post_d = float(gate["depth_m"]), float(gate.get("post_m", gate["depth_m"]))
    v = doc["vehicle"]
    min_clear = v["radius_m"] + v["margin_m"]
    pol = doc.get("leg_policy", {})
    face_min = float(pol.get("face_min_m", 0.8))
    search = float(gate.get("cross_search_deg", 30.0) if search_deg is None else search_deg)
    base = math.atan2(n0[1], n0[0])
    angles = [0.0] if search <= 0 else sorted({round(a, 6) for a in np.arange(-search, search + 1e-9, step_deg)} | {0.0},
                                              key=abs)
    cands = []
    for a in angles:
        h = base + math.radians(a)
        n = np.array([math.cos(h), math.sin(h)])
        pre, post = c - n * pre_d, c + n * post_d
        clr = segment_clearance(obst, clearance, pre, post)
        face = facing_distance(obst, pre, h)
        cands.append(dict(deg=a, normal=[round(float(x), 4) for x in n], clearance_m=round(clr, 3),
                          facing_m=round(face, 2) if math.isfinite(face) else None,
                          ok=bool(clr >= min_clear and face >= face_min)))
    ok = [k for k in cands if k["ok"]]
    if ok:
        best = max(ok, key=lambda k: (round(k["clearance_m"], 2), -abs(k["deg"])))
    else:                                                      # nothing meets the policy: the least bad, flagged
        best = max(cands, key=lambda k: (k["clearance_m"], -abs(k["deg"])))
    return np.asarray(best["normal"], float), dict(chosen_deg=best["deg"], feasible=best["ok"], candidates=cands,
                                                   searched_deg=search)


def expand_goals(doc, chicane, obst=None, clearance=None):
    """Course-file sequence -> ordered [{"name", "xy", "mode", "straight", ...}] for one chicane option.
    With the grids, each gate's crossing direction is searched (gate_crossing)."""
    pts = []
    info = {}
    for g in doc["sequence"]:
        if "choose" in g:
            key = chicane
            if key is None:
                raise PlanError("expand_goals needs a chicane option (plan_options chooses one)")
            gate = doc["chicane_options"][key]
            c, d = np.asarray(gate["center"], float), float(gate["depth_m"])
            post_d = float(gate.get("post_m", d))
            if obst is not None and clearance is not None:
                n, ginfo = gate_crossing(doc, key, obst, clearance)
                info[key] = ginfo
            else:
                n = _unit(gate["normal"])
            pts.append({"name": f"{key}_pre", "xy": list(c - n * d), "mode": "gate", "gate": key, "part": "pre"})
            a, b = gate_segment(gate)
            pts.append({"name": f"{key}", "xy": list(c), "mode": "gate", "straight": True, "gate": key, "part": "centre",
                        "normal": [float(v) for v in n], "pre_m": d,
                        "gate_line": [[float(v) for v in a], [float(v) for v in b]]})
            pts.append({"name": f"{key}_post", "xy": list(c + n * post_d), "mode": "gate", "straight": True,
                        "gate": key, "part": "post"})
        else:
            pts.append({"name": g["name"], "xy": list(g["xy"]), "mode": g.get("mode", "cruise"),
                        "straight": bool(g.get("straight")), "land": bool(g.get("land"))})
    return pts, info


# ---------------------------------------------------------------------------- the plan
def plan(doc, obst: Grid2D, blocked: Grid2D, clearance, start_xy=None, chicane=None, goals=None, cmap=None):
    """-> dict(waypoints=[{name, xy, mode, clearance_m, facing_m, ...}], length_m, min_clearance_m, ...).
    chicane None: every option is planned and the best is chosen (plan_options); the result carries
    "chicane" and "scores"."""
    if chicane is None and goals is None:
        return plan_options(doc, obst, blocked, clearance, start_xy=start_xy, cmap=cmap)["plan"]
    v = doc["vehicle"]
    min_clear = v["radius_m"] + v["margin_m"]
    start = list(start_xy if start_xy is not None else doc["start"]["xy"])
    gate_info = {}
    if goals is None:
        goals, gate_info = expand_goals(doc, chicane, obst, clearance)
    gate_lines = [g["gate_line"] for g in goals if "gate_line" in g]
    gated = with_walls(blocked, gate_lines)          # A* never flies through a gate on the way to it
    path = [np.asarray(start, float)]
    names = [{"name": "start", "mode": "takeoff"}]
    keep = [0]
    for i, g in enumerate(goals):
        p = path[-1]
        q = np.asarray(g["xy"], float)
        if g.get("straight"):
            c = segment_clearance(obst, clearance, p, q)
            if c < min_clear:
                raise PlanError(f"straight segment to {g['name']} has only {c:.2f} m clearance (< {min_clear:.2f})")
            seg = [q]
        else:
            try:
                seg = list(astar(gated, clearance, p, q))[1:]
            except PlanError:
                if gated is blocked:
                    raise
                ungated = astar(blocked, clearance, p, q)  # raises if physically unreachable too
                crossed = []
                for h in goals:
                    if "gate_line" in h:
                        hits = gate_crossings(ungated, h["gate_line"], h["normal"])
                        if hits:
                            crossed.append((hits[0][0], h["name"]))
                if crossed:
                    gate = ", ".join(name for _, name in sorted(crossed))
                    raise PlanError(f"{g['name']} is reachable only through {gate}: the ungated path crosses "
                                    f"those opening lines outside their crossing legs. No gated approach at "
                                    f"{min_clear:.2f} m clearance") from None
                # Dilated exclusion cells can block a goal/approach even when
                # the fallback never crosses an opening's centreline.
                raise PlanError(f"{g['name']} has no gated approach at {min_clear:.2f} m clearance: "
                                "gate exclusion cells block it, but the ungated path has no gate "
                                "centreline crossing") from None
        for s in seg[:-1]:
            path.append(np.asarray(s, float))
            names.append(None)
        path.append(q)
        names.append({k: g[k] for k in ("name", "mode", "land", "gate", "part", "normal", "pre_m") if k in g})
        keep.append(len(path) - 1)
    pol = doc.get("leg_policy", {})
    solid = solid_band(cmap, doc) if cmap is not None else obst
    # keep every goal; shortcut between them with a little extra clearance where it exists
    out_idx = _simplify_idx(path, obst, clearance, min_clear, keep, extra=doc.get("prefer_extra_clearance_m", 0.05),
                            min_leg=float(pol.get("min_leg_m", 0.2)), collinear_deg=float(pol.get("collinear_deg", 0.0)),
                            blocked=blocked, keep_path_clearance_m=float(pol.get("keep_path_clearance_m", 0.0)))
    wps = []
    for k in out_idx:
        meta = names[k] or {"name": f"via{len(wps)}", "mode": "cruise"}
        wps.append(dict(meta, xy=[float(c) for c in path[k]]))
    wps, moved = fix_leg_ends(wps, obst, blocked, clearance, min_clear, pol, solid)
    xy = [w["xy"] for w in wps]
    for g in goals:                                  # shortcuts and moved leg ends must not cut a gate either
        if "gate_line" in g:
            cr = gate_crossings(xy, g["gate_line"], g["normal"])
            if [s for _, s in cr] != [1]:
                raise PlanError(f"route crosses {g['name']} {len(cr)} time(s) {[s for _, s in cr]}; "
                                f"needs exactly once, in the chosen direction")
    annotate(wps, obst, clearance, doc, cmap, solid)
    segs = [segment_clearance(obst, clearance, wps[i]["xy"], wps[i + 1]["xy"]) for i in range(len(wps) - 1)]
    length = sum(math.dist(wps[i]["xy"], wps[i + 1]["xy"]) for i in range(len(wps) - 1))
    for w in wps:
        w["xy"] = [round(float(c), 4) for c in w["xy"]]
    out = {"waypoints": wps, "length_m": round(length, 3), "min_clearance_m": round(min(segs), 3),
           "segment_clearance_m": [round(c, 3) for c in segs], "required_clearance_m": min_clear,
           "chicane": chicane, "gate_crossings": gate_info, "gate_lines": gate_lines, "moved_leg_ends": moved,
           "leg_end_violations": [w["name"] for w in wps[1:-1] if not w.get("end_ok", True)],
           "leg_end_relaxed": [w["name"] for w in wps[1:-1] if w.get("end_relaxed")]}
    out["time_estimate_s"] = round(estimate_time(wps, doc), 1)
    rp = {k: v for k, v in doc.get("refnav_params", {}).items() if not k.startswith("_")}
    # refnav's own blocking rule replayed on the map (+ 3 cm for cell rounding): any hit is a predicted false stop
    out["predicted_false_blocks"] = refnav_block_check(wps, obst, float(rp.get("corridor_half_m", 0.32)) + 0.03,
                                                       effective_stop_m(rp), float(rp.get("leg_end_margin_m", 0.25)))
    return out


def _seg_ok(grid, clearance, blocked, p, q, need):
    """A shortcut segment keeps `need` clearance in the flight band AND crosses no keep-out."""
    if segment_clearance(grid, clearance, p, q) < need:
        return False
    return blocked is None or segment_free(blocked, p, q)


def _clear_pts(grid, clearance, pts):
    ij = grid.ij(np.asarray(pts, float))
    ok = grid.inside(ij)
    out = np.zeros(len(ij))
    out[ok] = clearance[ij[ok, 0], ij[ok, 1]]
    return out


def _furthest(path, grid, clearance, i, need, keep, blocked=None, need_ik=None):
    j = i + 1
    for k in range(i + 1, len(path)):
        n_k = need if need_ik is None else need_ik(i, k, need)
        if k > i + 1 and not _seg_ok(grid, clearance, blocked, path[i], path[k], n_k):
            break
        j = k
        if k in keep:
            break
    return j


def _simplify_idx(path, grid, clearance, min_clear, keep, extra=0.05, min_leg=0.2, collinear_deg=0.0, blocked=None,
                  keep_path_clearance_m=0.0):
    """Greedy line-of-sight shortcutting. Prefers segments with min_clear + extra; where the
    passage cannot give that (a gate), accepts min_clear rather than emitting cell-sized hops.
    A shortcut never crosses a blocked cell (keep-outs such as the boulders, which lie below the
    flight band and so are invisible to the clearance field: found offline 2026-11-06, a planned
    leg crossed boulder 3). Then drops non-goal points that would make a leg shorter than min_leg,
    and (collinear_deg > 0) non-goal points whose heading change is smaller than that, where the
    merged leg keeps clearance.

    keep_path_clearance_m > 0: a shortcut must also keep the clearance of the A* sub-path it
    replaces, up to this cap (the A* prefer_m: beyond it A* itself does not centre), less half a
    cell (the A* path is resolved only to its cells). Greedy shortcutting otherwise pulls every leg
    to min_clear + extra wherever that is shorter, even in a corridor that offers more (window plan,
    2026-11-07: a 0.400 m maze segment where 0.45 m was available)."""
    keep = set(keep) | {0, len(path) - 1}
    if keep_path_clearance_m > 0:
        c_path = np.minimum(_clear_pts(grid, clearance, path), keep_path_clearance_m) - grid.res / 2

        def need_ik(i, k, base):
            return max(base, float(c_path[min(i, k):max(i, k) + 1].min()))
    else:
        def need_ik(i, k, base):
            return base
    out = [0]
    i = 0
    while i < len(path) - 1:
        j = _furthest(path, grid, clearance, i, min_clear + extra, keep, blocked, need_ik)
        if j == i + 1 and j not in keep:
            j = _furthest(path, grid, clearance, i, min_clear, keep, blocked, need_ik)
        if j == i + 1 and j not in keep and keep_path_clearance_m > 0:
            j = _furthest(path, grid, clearance, i, min_clear, keep, blocked)   # no cell-sized hops
        out.append(j)
        i = j
    changed = True
    while changed:
        changed = False
        for n in range(1, len(out) - 1):
            a, b, c = out[n - 1], out[n], out[n + 1]
            if b in keep:
                continue
            short = min(math.dist(path[a], path[b]), math.dist(path[b], path[c])) < min_leg
            h1 = math.atan2(path[b][1] - path[a][1], path[b][0] - path[a][0])
            h2 = math.atan2(path[c][1] - path[b][1], path[c][0] - path[b][0])
            straightish = collinear_deg > 0 and abs(math.degrees(_wrap(h2 - h1))) < collinear_deg
            if (short or straightish) and _seg_ok(grid, clearance, blocked, path[a], path[c], need_ik(a, c, min_clear)):
                del out[n]
                changed = True
                break
    return out


def solid_band(cmap, doc):
    """The obstacle band without the nets: what the depth camera faces as a dense surface. The nets
    are 92 % open (returns unmeasured), so they do not count for the leg-end facing rules."""
    from .objmap import FLOOR_MARKING_PREFIXES
    z, band = doc["cruise_z_m"], doc.get("plan_band_m", 0.25)
    return Grid2D(cmap.band(z - band, z + band, exclude_prefixes=tuple(FLOOR_MARKING_PREFIXES) + ("Net",)),
                  cmap.origin[:2].copy(), cmap.res)


def fix_leg_ends(wps, obst, blocked, clearance, min_clear, pol, solid=None):
    """Apply the leg-end rule: at every interior waypoint the next heading must face no mapped
    surface within face_min_m, and the in-place turn's arc none within arc_min_m. A via point that
    breaks it is moved back along its incoming leg (5 cm steps, up to move_max_m) to the first
    position that satisfies it while both adjacent legs keep min_clear. Goal points are not moved;
    they are flagged (end_ok False) for the chicane scorer. -> (waypoints, [names moved])."""
    face_min = float(pol.get("face_min_m", 0.8))
    face_relaxed = float(pol.get("face_relaxed_m", 0.6))
    arc_min = float(pol.get("arc_min_m", 0.6))
    move_max = float(pol.get("move_max_m", 0.5))
    overrun = float(pol.get("overrun_min_m", 0.4))
    solid = obst if solid is None else solid
    moved = []
    for k in range(1, len(wps) - 1):
        a, b, c = np.asarray(wps[k - 1]["xy"]), np.asarray(wps[k]["xy"]), np.asarray(wps[k + 1]["xy"])
        if _end_ok(solid, a, b, c, face_min, arc_min, overrun):
            wps[k]["end_ok"] = True
            continue
        if not wps[k]["name"].startswith("via"):
            wps[k]["end_ok"] = _end_ok(solid, a, b, c, face_relaxed, arc_min, overrun)
            wps[k]["end_relaxed"] = wps[k]["end_ok"]
            continue
        # move the via point back along its incoming leg: first for the full rule, then (a maze corner
        # cannot always give 0.8 m) for the relaxed facing distance, which still clears the camera's
        # blind range (0.4 m + its 0.12 m offset)
        best, relaxed = None, False
        u = _unit(b - a)
        for want, rlx in ((face_min, False), (face_relaxed, True)):
            if _end_ok(solid, a, b, c, want, arc_min, overrun):
                best, relaxed = b, rlx
                break
            for d in np.arange(0.05, move_max + 1e-9, 0.05):
                if d >= math.dist(a, b) - 0.1:
                    break
                q = b - u * d
                ij = blocked.ij(q)
                if not blocked.inside(ij) or blocked.occ[ij[0], ij[1]]:
                    continue
                if not _seg_ok(obst, clearance, blocked, q, c, min_clear) or not _seg_ok(obst, clearance, blocked, a, q, min_clear):
                    continue
                if _end_ok(solid, a, q, c, want, arc_min, overrun):
                    best, relaxed = q, rlx
                    break
            if best is not None:
                break
        if best is not None:
            if not np.allclose(best, b):
                wps[k]["xy"] = [float(v) for v in best]
                moved.append(wps[k]["name"])
            wps[k]["end_ok"] = True
            if relaxed:
                wps[k]["end_relaxed"] = True
        else:
            wps[k]["end_ok"] = False
    return wps, moved


CORRIDOR_HALF_M = 0.35              # refnav's corridor half-width (0.32) + 3 cm of map-cell rounding


def _end_ok(obst, a, b, c, face_min, arc_min, overrun_min=0.0):
    """The leg-end rule at waypoint b (a -> b -> c): nothing within face_min along the next heading
    (over refnav's corridor width), nothing within arc_min anywhere in the turn's arc, and nothing
    within overrun_min beyond b along the incoming heading (over the corridor width: refnav's
    leg_end_margin rule), all on the obstacle band."""
    h_in = math.atan2(b[1] - a[1], b[0] - a[0])
    h_out = math.atan2(c[1] - b[1], c[0] - b[0])
    return (corridor_distance(obst, b, h_out, CORRIDOR_HALF_M) >= face_min
            and arc_min_distance(obst, b, h_in, h_out) >= arc_min
            and corridor_distance(obst, b, h_in, CORRIDOR_HALF_M) >= overrun_min)


def passage_fraction(cmap, a, b, z, roof_within_m=1.5, wall_within_m=1.0, step=0.05):
    """Fraction of the points along a->b that lie inside an enclosed passage: a mapped surface above
    (between z + 0.3 and z + roof_within_m) AND surfaces within wall_within_m on both sides. Pure
    geometry: no group names."""
    if cmap is None:
        return 0.0
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    if L < 1e-6:
        return 0.0
    u = (b - a) / L
    nrm = np.array([-u[1], u[0]])
    k0 = max(0, cmap.z_index(z + 0.3))
    k1 = min(cmap.shape[2] - 1, cmap.z_index(z + roof_within_m))
    band = cmap.band(z - 0.25, z + 0.25)
    obst = Grid2D(band, cmap.origin[:2].copy(), cmap.res)
    n_enclosed = 0
    ts = np.arange(0.0, L + 1e-9, step)
    for t in ts:
        p = a + u * t
        ij = cmap.xy_to_ij(p)
        if np.any(ij < 0) or np.any(ij >= np.array(cmap.shape[:2])):
            continue
        roof = bool(cmap.occupied[ij[0], ij[1], k0:k1 + 1].any())
        if not roof:
            continue
        h = math.atan2(nrm[1], nrm[0])
        if facing_distance(obst, p, h, wall_within_m) < wall_within_m and \
                facing_distance(obst, p, h + math.pi, wall_within_m) < wall_within_m:
            n_enclosed += 1
    return n_enclosed / len(ts)


def annotate(wps, obst, clearance, doc, cmap=None, solid=None):
    """Per-waypoint numbers for scoring and for refnav's leg metadata: clearance at the point, the
    facing distance along the next heading (solid surfaces, nets excluded), gate alignment geometry,
    passage detection."""
    z = doc["cruise_z_m"]
    solid = obst if solid is None else solid
    for k, w in enumerate(wps):
        ij = obst.ij(w["xy"])
        w["clearance_m"] = round(float(clearance[ij[0], ij[1]]), 3)
        if k + 1 < len(wps):
            nxt = np.asarray(wps[k + 1]["xy"]) - np.asarray(w["xy"])
            if np.linalg.norm(nxt) > 1e-6:
                f = corridor_distance(solid, w["xy"], math.atan2(nxt[1], nxt[0]), CORRIDOR_HALF_M)
                w["facing_m"] = None if not math.isfinite(f) else round(f, 2)
        if 0 < k:
            inc = np.asarray(w["xy"]) - np.asarray(wps[k - 1]["xy"])
            if np.linalg.norm(inc) > 1e-6:
                f = corridor_distance(obst, w["xy"], math.atan2(inc[1], inc[0]), CORRIDOR_HALF_M)
                w["overrun_m"] = None if not math.isfinite(f) else round(f, 2)
        if w.get("part") == "centre":
            n = np.asarray(w["normal"], float)
            left, right = opening_edges(obst, w["xy"], n)
            gate = doc["chicane_options"][w["gate"]]
            dist = float(w["pre_m"])
            align = {"group": w["gate"], "dist_m": round(dist, 3), "width_m": round(left + right, 3),
                     "half_left_m": round(left, 3), "half_right_m": round(right, 3)}
            if gate.get("z_m"):
                align["dz_lo"] = round(float(gate["z_m"][0]) - z, 3)
                align["dz_hi"] = round(float(gate["z_m"][1]) - z, 3)
            far = max(left, right)
            bearing = math.degrees(math.atan2(far, max(dist - CAMERA_FORWARD_M, 0.05))) if math.isfinite(far) else 90.0
            align["edge_bearing_deg"] = round(bearing, 1)
            align["visible"] = bool(bearing < CAMERA_HALF_FOV_DEG)
            w["align"] = align
        if w.get("mode") == "tunnel" and k > 0:
            a = wps[k - 1]["xy"]
            frac = passage_fraction(cmap, a, w["xy"], z)
            L = math.dist(a, w["xy"])
            w["passage"] = {"enclosed_frac": round(frac, 2), "length_m": round(L, 3),
                            "map_clear": bool(segment_clearance(obst, clearance, a, w["xy"]) >= doc["vehicle"]["radius_m"]
                                              + doc["vehicle"]["margin_m"]),
                            "is_passage": bool(frac >= float(doc.get("leg_policy", {}).get("passage_min_frac", 0.4)))}
            if w["passage"]["is_passage"]:
                prev = wps[k - 1]
                u = _unit(np.asarray(w["xy"]) - np.asarray(a))
                # the mouth: the first enclosed point along the leg; its free width = the bore
                mouth = _mouth_offset(cmap, a, w["xy"], z)
                left, right = opening_edges(obst, np.asarray(a) + u * mouth, u)
                prev["align"] = {"group": "tunnel", "dist_m": round(mouth, 3),
                                 "width_m": round(left + right, 3), "half_left_m": round(left, 3),
                                 "half_right_m": round(right, 3)}
                far = max(left, right)
                bearing = (math.degrees(math.atan2(far, max(prev["align"]["dist_m"] - CAMERA_FORWARD_M, 0.05)))
                           if math.isfinite(far) else 90.0)
                prev["align"]["edge_bearing_deg"] = round(bearing, 1)
                prev["align"]["visible"] = bool(bearing < CAMERA_HALF_FOV_DEG)


def _mouth_offset(cmap, a, b, z, roof_within_m=1.5, step=0.05):
    """Distance along a->b to the first point with a roof above (the passage mouth)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    u = (b - a) / L
    k0 = max(0, cmap.z_index(z + 0.3))
    k1 = min(cmap.shape[2] - 1, cmap.z_index(z + roof_within_m))
    for t in np.arange(0.0, L + 1e-9, step):
        ij = cmap.xy_to_ij(a + u * t)
        if np.any(ij < 0) or np.any(ij >= np.array(cmap.shape[:2])):
            continue
        if cmap.occupied[ij[0], ij[1], k0:k1 + 1].any():
            return float(t)
    return L


# ---------------------------------------------------------------------------- scoring
def estimate_time(wps, doc):
    """Estimated flight time (s) for refnav's speed profile: climb, then per leg the trapezoidal
    travel time, the in-place turn (unless blended), the check and the hold; alignment legs add
    the alignment time."""
    rp = {k: v for k, v in doc.get("refnav_params", {}).items() if not k.startswith("_")}
    v = float(rp.get("speed_mps", 0.6))
    a = float(rp.get("accel_mps2", 0.0))
    yaw = math.radians(float(rp.get("yaw_rate_dps", 45.0)))
    hold = float(rp.get("hold_s", 1.0))
    blend = float(rp.get("blend_turn_deg", 0.0))
    t = doc["cruise_z_m"] / float(rp.get("climb_rate_mps", 0.5)) + 2.0
    heading = math.radians(doc["start"].get("yaw_deg", 90.0))
    wps = flight_waypoints(wps)
    for i in range(1, len(wps)):
        d = np.asarray(wps[i]["xy"]) - np.asarray(wps[i - 1]["xy"])
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            continue
        h = math.atan2(d[1], d[0])
        turn = abs(_wrap(h - heading))
        stops = (wps[i].get("mode") in ("gate", "tunnel", "align") or wps[i - 1].get("mode") in ("gate", "tunnel", "align")
                 or blend <= 0 or math.degrees(turn) >= blend)
        if a > 0:
            ramp = v * v / a                                   # accelerate + brake distance at full stop
            t += (L / v + v / a) if L >= ramp else 2 * math.sqrt(L / a)
        else:
            t += L / v
        if stops:
            t += turn / yaw + 0.4 + hold                       # turn, check (a few frames), hold
        if wps[i - 1].get("align"):
            t += 1.5
        heading = h
    return t


def score_plan(p, doc, option):
    """Points for a chicane option: its checkpoint points minus the estimated time, minus penalties
    for clearance below clearance_min_m and for leg ends facing a surface closer than face_min_m
    (or an alignment view the camera cannot see)."""
    sc = doc.get("chicane_scoring", {})
    pol = doc.get("leg_policy", {})
    pts = float(option.get("points", 0.0))
    time_w = float(sc.get("time_points_per_s", 0.05))
    cmin = float(sc.get("clearance_min_m", 0.45))
    cpen = float(sc.get("clearance_penalty_per_cm", 0.5))
    face_min = float(pol.get("face_min_m", 0.8))
    fpen = float(sc.get("facing_penalty", 10.0))
    apen = float(sc.get("alignment_penalty", 10.0))
    t = p["time_estimate_s"]
    clr_pen = max(0.0, cmin - p["min_clearance_m"]) * 100.0 * cpen
    face_relaxed = float(pol.get("face_relaxed_m", 0.6))
    facing_bad = [w["name"] for w in p["waypoints"][:-1]
                  if w.get("facing_m") is not None and w["facing_m"] < face_relaxed]
    facing_soft = [w["name"] for w in p["waypoints"][:-1]
                   if w.get("facing_m") is not None and face_relaxed <= w["facing_m"] < face_min]
    align_bad = [w["name"] for w in p["waypoints"] if w.get("align") and not w["align"].get("visible", True)]
    cross_bad = [k for k, g in p.get("gate_crossings", {}).items() if not g.get("feasible", True)]
    false_blocks = p.get("predicted_false_blocks", [])
    score = (pts - time_w * t - clr_pen - fpen * (len(facing_bad) + len(cross_bad) + len(false_blocks))
             - 0.1 * fpen * len(facing_soft) - apen * len(align_bad))
    return {"score": round(score, 2), "points": pts, "time_s": round(t, 1), "time_penalty": round(time_w * t, 2),
            "min_clearance_m": p["min_clearance_m"], "clearance_penalty": round(clr_pen, 2),
            "legs": len(p["waypoints"]) - 1, "length_m": p["length_m"],
            "facing_violations": facing_bad, "facing_relaxed": facing_soft, "crossing_infeasible": cross_bad,
            "alignment_not_visible": align_bad, "predicted_false_blocks": false_blocks}


def plan_options(doc, obst, blocked, clearance, start_xy=None, cmap=None):
    """Plan every chicane option, score each, choose the best. -> {"chicane", "plan", "scores", "reason", "plans"}."""
    results, scores, errors = {}, {}, {}
    for key, opt in doc["chicane_options"].items():
        if "center" not in opt:
            continue
        try:
            p = plan(doc, obst, blocked, clearance, start_xy=start_xy, chicane=key, cmap=cmap)
        except PlanError as e:
            errors[key] = str(e)
            continue
        results[key] = p
        scores[key] = score_plan(p, doc, opt)
    if not results:
        raise PlanError("no chicane option could be planned: " + "; ".join(f"{k}: {v}" for k, v in errors.items()))
    best = max(scores, key=lambda k: scores[k]["score"])
    ranked = sorted(scores.items(), key=lambda kv: -kv[1]["score"])
    reason = "; ".join(f"{k}: score {s['score']:+.2f} (points {s['points']:g}, time {s['time_s']:.0f} s, "
                       f"min clearance {s['min_clearance_m']:.2f} m"
                       + (f", facing {s['facing_violations']}" if s["facing_violations"] else "")
                       + (f", crossing infeasible {s['crossing_infeasible']}" if s["crossing_infeasible"] else "")
                       + (f", alignment not visible {s['alignment_not_visible']}" if s["alignment_not_visible"] else "")
                       + (f", predicted false blocks {s['predicted_false_blocks']}" if s["predicted_false_blocks"] else "")
                       + ")" for k, s in ranked)
    for k, e in errors.items():
        reason += f"; {k}: NO PLAN ({e})"
    results[best]["chicane"] = best
    results[best]["scores"] = scores
    results[best]["choice_reason"] = reason
    return {"chicane": best, "plan": results[best], "scores": scores, "reason": reason, "plans": results, "errors": errors}


# ---------------------------------------------------------------------------- refnav route
def flight_waypoints(waypoints):
    """The waypoints refnav flies: a gate's centre is crossed in ONE straight leg from the pre point
    to the post point (pre -> centre -> post are collinear by construction), so the aircraft does
    not stop, turn and re-check inside the opening. The centre's alignment geometry moves to the
    post waypoint (the leg that crosses the plane)."""
    out = []
    k = 0
    while k < len(waypoints):
        w = waypoints[k]
        if w.get("part") == "centre" and k + 1 < len(waypoints) and waypoints[k + 1].get("part") == "post":
            post = dict(waypoints[k + 1])
            if w.get("align"):
                post["align"] = dict(w["align"])
            post["crosses"] = w["name"]
            out.append(post)
            k += 2
            continue
        out.append(w)
        k += 1
    return out


def to_route(waypoints, start_xy, start_yaw_rad):
    """Course-frame waypoints -> refnav route [(forward_m, left_m)] from the take-off pose."""
    c, s = math.cos(start_yaw_rad), math.sin(start_yaw_rad)
    route = []
    for w in flight_waypoints(waypoints)[1:]:
        dx, dy = w["xy"][0] - start_xy[0], w["xy"][1] - start_xy[1]
        route.append((round(dx * c + dy * s, 4), round(-dx * s + dy * c, 4)))
    return route


def to_legs(waypoints):
    """Per-leg metadata for refnav (Params.legs), one dict per route entry (the leg ENDING at that
    waypoint): mode, hold (stop at the end), gate_group, align (geometry measured on the map) and
    tunnel (passage) flags. Names only: no coordinates."""
    legs = []
    waypoints = flight_waypoints(waypoints)
    for k in range(1, len(waypoints)):
        w, prev = waypoints[k], waypoints[k - 1]
        leg = {"name": w["name"], "mode": w.get("mode", "cruise")}
        if w.get("crosses"):
            leg["crosses"] = w["crosses"]
        if w.get("mode") in ("gate", "tunnel", "align") or w.get("land"):
            leg["hold"] = True
        if w.get("gate"):
            leg["gate_group"] = w["gate"]
        if w.get("align") and w["align"].get("group") != "tunnel":   # arriving at this waypoint crosses an opening
            leg["align"] = dict(w["align"])
        if prev.get("align") and prev["align"].get("group") == "tunnel":   # the leg from the mouth through it
            leg["align"] = dict(prev["align"])
            leg["gate_group"] = "tunnel"
        if w.get("passage", {}).get("is_passage"):
            leg["tunnel"] = {"length_m": w["passage"]["length_m"], "map_clear": w["passage"]["map_clear"],
                             "enclosed_frac": w["passage"]["enclosed_frac"]}
        legs.append(leg)
    return legs


def refnav_block_check(waypoints, obst_band: Grid2D, corridor_half, stop_m, end_margin, rmin=0.4, step=0.05):
    """Replay refnav's obstacle rule against the MAP for every leg (pre-flight, offline).
    At each point along a leg, a mapped surface blocks if it is ahead within [rmin, stop_m), within
    +-corridor_half across the leg, and closer than the remaining leg + end_margin.
    -> [(leg index, along_m, clearance_m, across_m)] of predicted false 'route blocked' stops."""
    occ_xy = obst_band.xy(np.argwhere(obst_band.occ))
    hits = []
    for k in range(len(waypoints) - 1):
        a, b = np.asarray(waypoints[k]["xy"], float), np.asarray(waypoints[k + 1]["xy"], float)
        L = float(np.linalg.norm(b - a))
        if L < 1e-6:
            continue
        u = (b - a) / L
        nrm = np.array([-u[1], u[0]])
        for s in np.arange(0.0, L, step):
            rel = occ_xy - (a + u * s)
            along, across = rel @ u, rel @ nrm
            sel = (along >= rmin) & (along < stop_m) & (np.abs(across) < corridor_half)
            if sel.any():
                c = float(along[sel].min())
                if c < (L - s) + end_margin:
                    hits.append((k + 1, round(float(s), 2), round(c, 2), round(float(np.abs(across[sel]).min()), 2)))
                    break
    return hits


def effective_stop_m(rp):
    """refnav's stop distance at cruise for a refnav_params dict (scaled rule when stop_per_mps > 0)."""
    per = float(rp.get("stop_per_mps", 0.0))
    if per > 0:
        return max(float(rp.get("stop_min_m", 0.55)), float(rp.get("stop_base_m", 0.45)) + per * float(rp.get("speed_mps", 0.6)))
    return float(rp.get("stop_m", 0.9))
