"""Course goals derived from the published map's named parts (rule 9.5). PURE (numpy).

The course file says WHAT each checkpoint is ("the end of the guideline", "the tube's axis",
"the opening in wall C2a", "the gap past wall C2b's end", "the pad's centre") and names the OBJ
group it lives on. This module measures WHERE from the published OBJ. No goal coordinate is written
in code or in the course file: move a wall in the OBJ and the goals move with it.

Shapes are measured on the published OBJ's own triangles (sampled every 1 cm). The 5 cm voxel map is
used only for free space: centring a goal across a corridor and marching to the next obstacle.

Features (course file key "from": {"feature": ..., "group": ..., parameters}):
  launch_mark    start xy = centroid of the mark; heading = the guideline's first direction.
  guideline_end  Gate A: the far end of the floor guideline (walked along the paint from the launch
                 mark), then past_m further along its last direction so the whole drone is past the
                 line; centred across the corridor (max clearance, moved sideways only).
  tube_axis      entry/exit: the tube's principal axis; the ends pushed out by entry_m / exit_m
                 (default approach_m). The end nearer the previous goal is the entry. The entry sits
                 far enough back for the camera to see both rim edges (the mouth alignment).
  wall_opening   window: the largest gap along the wall in its surface at cruise height. Centre,
                 width along the wall, height span, normal across it.
  gap_after_wall_end   Gate B: from each end of the wall, march along the wall's axis to the next
                 obstacle; the gate is the middle of the narrowest gap the drone fits (min_width_m).
  pad_center     centroid and radius of the pad's floor marking.
The enclosure (course file "enclosure": {"from": {"group": "Net side"}}) is the bounding box of the
net's side panels. A gate's normal points from the previous goal's side to the pad's side (the
crossing direction).
"""
from __future__ import annotations

import copy
import heapq
import math

import numpy as np

from .objmap import CourseMap, load_obj, sample_triangles

FLOOR_Z = 0.05      # floor paint lies below this height


class SemanticsError(RuntimeError):
    pass


# ---------------------------------------------------------------------------- published geometry
class Geometry:
    """Surface points of the published OBJ's named groups, sampled every `spacing` metres."""

    def __init__(self, obj_path, spacing=0.01):
        self.groups = load_obj(obj_path)
        self.spacing = spacing
        self._cache = {}

    def triangles(self, prefix):
        tri = [V[F] for name, (V, F) in self.groups.items() if name.startswith(prefix)]
        if not tri:
            raise SemanticsError(f"the published OBJ has no group starting with '{prefix}'")
        return np.concatenate(tri)

    def points(self, prefix, zlo=-math.inf, zhi=math.inf):
        if prefix not in self._cache:
            self._cache[prefix] = sample_triangles(self.triangles(prefix), self.spacing)
        P = self._cache[prefix]
        P = P[(P[:, 2] >= zlo) & (P[:, 2] <= zhi)]
        if len(P) == 0:
            raise SemanticsError(f"group '{prefix}' has no surface between z {zlo} and {zhi}")
        return P


# ---------------------------------------------------------------------------- geometry helpers
def principal_axis(xy):
    """-> (centre, unit axis, t_min, t_max) of a point set along its longest direction."""
    c = xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy - c, full_matrices=False)
    a = vt[0] / np.linalg.norm(vt[0])
    t = (xy - c) @ a
    return c, a, float(t.min()), float(t.max())


def perp(a):
    return np.array([-a[1], a[0]])


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def raster(xy, res):
    """Points -> (bool grid, lower corner) covering them at res."""
    lo = xy.min(axis=0) - res
    ij = np.floor((xy - lo) / res).astype(int)
    m = np.zeros(ij.max(axis=0) + 2, bool)
    m[ij[:, 0], ij[:, 1]] = True
    return m, lo


def geodesic(m, seed_ij):
    """Dijkstra distance in cells (diagonal sqrt 2) from seed_ij inside the 8-connected mask m."""
    d = np.full(m.shape, np.inf)
    d[seed_ij] = 0.0
    pq = [(0.0, seed_ij)]
    nx, ny = m.shape
    while pq:
        dc, (i, j) = heapq.heappop(pq)
        if dc > d[i, j]:
            continue
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if not (di or dj):
                    continue
                a, b = i + di, j + dj
                if 0 <= a < nx and 0 <= b < ny and m[a, b]:
                    nd = dc + (math.sqrt(2.0) if di and dj else 1.0)
                    if nd < d[a, b]:
                        d[a, b] = nd
                        heapq.heappush(pq, (nd, (a, b)))
    return d


def centre_across(xy, direction, blocked, clearance, span_m=0.3, step=0.025):
    """Slide xy perpendicular to `direction` (never along it) to the most open free point."""
    n = perp(unit(direction))
    best, best_c = None, -math.inf
    for s in np.arange(-span_m, span_m + 1e-9, step):
        p = np.asarray(xy) + s * n
        ij = blocked.ij(p)
        if not blocked.inside(ij) or blocked.occ[ij[0], ij[1]]:
            continue
        c = float(clearance[ij[0], ij[1]]) - 0.05 * abs(s)      # most open, then nearest the line
        if c > best_c:
            best, best_c = p, c
    if best is None:
        raise SemanticsError(f"no free point within {span_m} m across {np.round(xy, 3)}")
    return best


def march(occ_grid, cmap, p0, direction, max_m=2.0):
    """Distance along direction from p0 to the first occupied cell of occ_grid (inf: none)."""
    step = cmap.res * 0.25
    d = unit(direction)
    for t in np.arange(step, max_m, step):
        ij = cmap.xy_to_ij(np.asarray(p0) + t * d)
        if np.any(ij < 0) or np.any(ij >= np.array(occ_grid.shape)):
            return math.inf
        if occ_grid[ij[0], ij[1]]:
            return float(t)
    return math.inf


# ---------------------------------------------------------------------------- features
def launch_mark(geo, spec):
    return geo.points(spec["group"], zhi=FLOOR_Z)[:, :2].mean(axis=0)


def guideline(geo, spec, start_xy, res=0.01):
    """-> (start direction, end point on the centreline, end direction) of the painted line,
    walked along the paint from the end nearest start_xy."""
    xy = geo.points(spec["group"], zhi=FLOOR_Z)[:, :2]
    m, lo = raster(xy, res)
    ij = np.argwhere(m)
    cxy = lo + (ij + 0.5) * res
    seed = tuple(ij[np.argmin(np.linalg.norm(cxy - np.asarray(start_xy), axis=1))])
    d = geodesic(m, seed)[ij[:, 0], ij[:, 1]] * res
    if not np.isfinite(d).all():
        cxy, d = cxy[np.isfinite(d)], d[np.isfinite(d)]       # ignore paint not joined to the start
    look = spec.get("direction_from_m", 0.5)
    head = d <= look
    tail = d >= d.max() - look
    _, a0, _, _ = principal_axis(cxy[head])
    if np.dot(a0, cxy[~head].mean(axis=0) - cxy[head].mean(axis=0)) < 0:
        a0 = -a0
    c1, a1, _, _ = principal_axis(cxy[tail])
    if np.dot(a1, cxy[tail].mean(axis=0) - cxy[~tail].mean(axis=0)) < 0:
        a1 = -a1
    end = c1 + a1 * float(((cxy[tail] - c1) @ a1).max())
    return a0, end, a1


def tube_axis(geo, spec, prev_xy):
    """The tube's axis is the direction its surface normals avoid (every normal of a tube is
    perpendicular to its axis): the least eigenvector of the area-weighted normal scatter. This works
    for any cross-section, including a tube about as wide as it is long."""
    tri = geo.triangles(spec["group"])
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])           # |n| = 2 x area
    area = np.linalg.norm(n, axis=1)
    u = n[area > 1e-12] / area[area > 1e-12, None]
    _, v = np.linalg.eigh((u * area[area > 1e-12, None]).T @ u)          # ascending eigenvalues
    ax3 = v[:, 0]
    if np.hypot(ax3[0], ax3[1]) < 0.7:
        raise SemanticsError(f"'{spec['group']}' axis is not horizontal ({np.round(ax3, 2)})")
    a = unit(ax3[:2])
    xy = geo.points(spec["group"])[:, :2]
    c = xy.mean(axis=0)
    t = (xy - c) @ a
    ends = [c + a * t.min(), c + a * t.max()]
    if np.linalg.norm(ends[1] - prev_xy) < np.linalg.norm(ends[0] - prev_xy):
        ends = ends[::-1]
    d = unit(ends[1] - ends[0])
    k = spec.get("approach_m", 0.55)
    k_in, k_out = spec.get("entry_m", k), spec.get("exit_m", k)
    return ends[0] - d * k_in, ends[1] + d * k_out, dict(axis_xy=[round(float(v_), 4) for v_ in c],
                                                        length_m=round(float(t.max() - t.min()), 3))


def wall_opening(geo, spec, cruise_z, band=0.02):
    """The largest gap along the wall's length in its surface at cruise height."""
    P = geo.points(spec["group"])
    c, a, t0, t1 = principal_axis(P[:, :2])
    at_z = P[np.abs(P[:, 2] - cruise_z) <= band]
    t = np.concatenate([[t0], np.sort((at_z[:, :2] - c) @ a), [t1]])
    k = int(np.argmax(np.diff(t)))
    lo_t, hi_t = t[k], t[k + 1]
    if hi_t - lo_t < 0.2:
        raise SemanticsError(f"wall '{spec['group']}' has no opening at z {cruise_z}")
    mid = (lo_t + hi_t) / 2
    across = perp(a)
    near = P[np.abs((P[:, :2] - c) @ a - mid) <= 0.02]           # the sill and lintel at the middle
    centre = c + a * mid + across * float(np.median((near[:, :2] - c) @ across))
    zs = near[:, 2]
    below, above = zs[zs < cruise_z], zs[zs > cruise_z]
    z = [float(below.max()) if len(below) else 0.0, float(above.min()) if len(above) else math.inf]
    return centre, across, dict(width_m=round(float(hi_t - lo_t), 3), z_m=[round(v, 3) for v in z])


def gap_after_wall_end(geo, cmap, spec, zlo=0.3, zhi=1.6):
    """Gate past one end of a wall: march along the wall's axis to the next obstacle in the band; of
    the gaps at least min_width_m wide (the drone must fit), take the narrowest."""
    P = geo.points(spec["group"], zlo, zhi)
    c, a, t0, t1 = principal_axis(P[:, :2])
    wall = cmap.band(zlo, zhi, only_prefixes=(spec["group"],))
    others = cmap.band(zlo, zhi) & ~wall
    best = None
    for end, d in ((c + a * t1, a), (c + a * t0, -a)):
        g = march(others, cmap, end, d, max_m=spec.get("max_gap_m", 2.0))
        if math.isfinite(g) and g >= spec.get("min_width_m", 0.6) and (best is None or g < best[0]):
            best = (g, end, d)
    if best is None:
        raise SemanticsError(f"no passable gap past either end of '{spec['group']}'")
    g, end, d = best
    return end + d * (g / 2.0), perp(a), dict(width_m=round(g, 3), wall_end_xy=[round(float(v), 4) for v in end])


def pad_center(geo, spec):
    """Pad marking centre/radius, with an optional measured course-frame goal offset."""
    xy = geo.points(spec["group"], zhi=FLOOR_Z)[:, :2]
    c = xy.mean(axis=0)
    radius = float(np.linalg.norm(xy - c, axis=1).max())
    offset = np.asarray(spec.get("offset_xy_m", [0.0, 0.0]), float)
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise SemanticsError("pad_center offset_xy_m must contain two finite course-frame metres")
    return c + offset, radius


# ---------------------------------------------------------------------------- the course
def _r(v):
    return [round(float(x), 4) for x in v]


def resolve_course(doc, cmap: CourseMap, geo: Geometry = None):
    """Course file with "from" specs -> a copy with every xy / centre / normal filled in from the
    published OBJ (geo) and its voxel map (cmap: free space for centring and gap marching).
    Adds doc["derived"] describing what was measured. Raises SemanticsError if a feature is missing."""
    from .course import plan_grids, resolve
    geo = geo or Geometry(resolve(doc, "obj"))
    out = copy.deepcopy(doc)
    _, blocked, clearance = plan_grids(cmap, doc)
    z = doc["cruise_z_m"]
    derived = {}

    st = out["start"]
    if "from" in st:
        f = st["from"]
        sxy = launch_mark(geo, f)
        st["xy"] = _r(sxy)
        if f.get("heading_from"):
            heading, _, _ = guideline(geo, {"group": f["heading_from"]}, sxy)
            st["yaw_deg"] = round(math.degrees(math.atan2(heading[1], heading[0])), 2)
        # a declared spawn offset from the mark belongs to the SIM/TRAINING profile, never to the
        # competition configuration (the venue's start is the mark itself; the localizer refines it)
        fixture = (out.get("sim_profile") or {}).get("fixture_offset_m")
        if fixture:
            st["spawn_xy"] = _r(np.asarray(st["xy"], float) + np.asarray(fixture, float))
        derived["start"] = dict(xy=st["xy"], yaw_deg=st.get("yaw_deg"))

    enc = out.get("enclosure")
    if enc and "from" in enc:                   # the net's inner planes, measured on the OBJ
        pts = geo.points(enc["from"].get("group", "Net side"))[:, :2]
        inset = float(enc["from"].get("inset_m", 0.0))
        enc["lo"] = _r(pts.min(axis=0) + inset)
        enc["hi"] = _r(pts.max(axis=0) - inset)
        derived["enclosure"] = dict(lo=enc["lo"], hi=enc["hi"])

    ret = out.get("return")
    if ret and "from" in ret:
        ret["target_xy"] = _r(launch_mark(geo, ret["from"]))
        derived["return"] = dict(target_xy=ret["target_xy"])

    pad_xy = None                       # gates cross towards the pad
    for g in out["sequence"]:
        if g.get("from", {}).get("feature") == "pad_center":
            c, r = pad_center(geo, g["from"])
            g["xy"], g["radius_m"] = _r(c), round(r, 3)
            derived[g["name"]] = dict(xy=g["xy"], radius_m=g["radius_m"])
            pad_xy = c

    prev = np.asarray(st["xy"], float)
    seq = []
    for g in out["sequence"]:
        f = g.get("from", {})
        feat = f.get("feature")
        if feat == "guideline_end":
            _, end, d = guideline(geo, f, np.asarray(st["xy"]))
            p = centre_across(end + d * f.get("past_m", 0.4), d, blocked, clearance, f.get("centre_span_m", 0.3))
            g["xy"] = _r(p)
            derived[g["name"]] = dict(xy=g["xy"], line_end_xy=_r(end), direction=_r(d))
            seq.append(g)
        elif feat == "tube_axis":
            entry, exit_, info = tube_axis(geo, f, prev)
            e1 = {"name": f"{g['name']}_entry", "xy": _r(entry), "mode": "align"}
            e2 = {"name": f"{g['name']}_exit", "xy": _r(exit_), "mode": "tunnel", "straight": True}
            derived[g["name"]] = dict(entry=e1["xy"], exit=e2["xy"], **info)
            seq += [e1, e2]
        elif "choose" in g:
            dropped = {}
            for name, opt in list(out["chicane_options"].items()):
                of = opt.get("from")
                if not of:
                    continue
                try:
                    if of["feature"] == "wall_opening":
                        c, n, info = wall_opening(geo, of, z)
                    elif of["feature"] == "gap_after_wall_end":
                        c, n, info = gap_after_wall_end(geo, cmap, of)
                    else:
                        raise SemanticsError(f"unknown chicane feature {of['feature']}")
                except SemanticsError as e:            # an option whose part is not in this course: drop it
                    dropped[name] = str(e)
                    continue
                if pad_xy is not None and np.dot(n, pad_xy - c) < 0:
                    n = -n
                if "cross_deg" in opt:
                    raise SemanticsError(f"chicane option '{name}' declares cross_deg: crossing directions are "
                                         "searched by the planner (coursenav.planner.gate_crossing), never typed in")
                opt.update(center=_r(c), normal=_r(n), **info)
                derived[f"chicane:{name}"] = dict(center=opt["center"], normal=opt["normal"], **info)
            for name, why in dropped.items():
                out["chicane_options"].pop(name)
                derived[f"chicane:{name}"] = dict(dropped=why)
            if not any("center" in o for o in out["chicane_options"].values()):
                raise SemanticsError("no chicane option can be found in this course: " + "; ".join(dropped.values()))
            seq.append(g)
            # the next goal is measured from the chicane area: use the mean of the options' centres
            centres = [np.asarray(o["center"], float) for o in out["chicane_options"].values() if "center" in o]
            prev = np.mean(centres, axis=0) if centres else prev
            continue
        elif feat and feat != "pad_center":
            raise SemanticsError(f"unknown feature {feat}")
        else:
            seq.append(g)
        if "xy" in seq[-1]:
            prev = np.asarray(seq[-1]["xy"], float)
    out["sequence"] = seq
    out["derived"] = derived
    return out
