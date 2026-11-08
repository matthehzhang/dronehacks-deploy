"""Course file + map loading and the derived 2D grids (planning, keep-out, scan matching)."""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from .objmap import FLOOR_MARKING_PREFIXES, CourseMap
from .airframe import vehicle_geometry

PROJECT = Path(__file__).resolve().parent.parent


_RESOLVED = {}


# The course the stack flies by default: the team's reconstructed origin map (2026-11-07). Override with
# COURSENAV_COURSE=<course file> (the regression tests pin course/course_v3.json in tests/conftest.py).
DEFAULT_COURSE = PROJECT / "course" / "course_new_origin.json"


def default_course_path():
    env = os.environ.get("COURSENAV_COURSE")
    if not env:
        return DEFAULT_COURSE
    p = Path(env)
    return p if p.is_absolute() else PROJECT / p


def load_course(path=None, derive=True):
    """Course file -> dict (default: default_course_path()). With derive (default), every goal position is
    measured from the published OBJ by coursenav.semantics (the file itself stores none: rule 9.5);
    cached per file version."""
    path = Path(path) if path is not None else default_course_path()
    doc = json.loads(path.read_text())
    doc['vehicle'] = vehicle_geometry(doc['vehicle'])
    doc["_dir"] = str(PROJECT)
    if not derive:
        return doc
    key = (str(path.resolve()), path.stat().st_mtime_ns, str(resolve(doc, "map")), str(resolve(doc, "obj")))
    if key not in _RESOLVED:
        from .semantics import resolve_course
        _RESOLVED[key] = resolve_course(doc, CourseMap.load(resolve(doc, "map")))
    return copy.deepcopy(_RESOLVED[key])


def resolve(doc, key):
    p = Path(doc[key])
    return p if p.is_absolute() else Path(doc.get("_dir", PROJECT)) / p


@dataclass
class Grid2D:
    """A 2D bool grid sharing the CourseMap's x/y indexing."""
    occ: np.ndarray
    origin: np.ndarray      # xy of the lower corner
    res: float

    def ij(self, xy):
        return np.floor((np.asarray(xy, float) - self.origin) / self.res).astype(int)

    def xy(self, ij):
        return self.origin + (np.asarray(ij, float) + 0.5) * self.res

    def inside(self, ij):
        ij = np.asarray(ij)
        return np.all((ij >= 0) & (ij < np.array(self.occ.shape)), axis=-1)

    def lookup(self, xy, outside=True):
        ij = self.ij(xy)
        ok = self.inside(ij)
        out = np.full(ij.shape[:-1], outside, bool)
        out[ok] = self.occ[ij[ok][:, 0], ij[ok][:, 1]]
        return out

    def distance(self):
        """Metres from every cell to the nearest occupied cell (EDT)."""
        return ndimage.distance_transform_edt(~self.occ) * self.res


SURFACE_SAMPLE_M = 0.01
_EXACT = {}


def surface_clearance(doc, cmap: CourseMap, zlo, zhi, spacing=SURFACE_SAMPLE_M):
    """Metres from every cell CENTRE to the nearest published surface in [zlo, zhi] (floor marking
    excluded, as in CourseMap.band), measured on the OBJ's triangles instead of the voxel raster.
    -> array like Grid2D.distance(), or None when the map is not provably the published layout.

    Why: the voxel EDT measures cell centre to occupied-cell centre, which errs by up to
    res*sqrt(2)/2 = 3.5 cm either way at 5 cm cells. At the window (0.914 m wide in the OBJ, so
    0.457 m centre clearance) the raster read 0.412 m and refused a 0.414 m requirement; elsewhere
    it can read up to 3.5 cm too much. Bound used here: triangles are sampled every `spacing` m
    (barycentric, objmap.sample_triangles), so any surface point lies within spacing*sqrt(2)/2 of a
    sample; that is subtracted, making the field a lower bound on the true clearance.
    Only an unmodified published map (meta source/obj_sha256 match, no simulated 'world') uses it."""
    import hashlib
    meta = getattr(cmap, "meta", {}) or {}
    try:
        obj = resolve(doc, "obj")
    except KeyError:
        return None
    if meta.get("source") != "published OBJ" or "world" in meta or not obj.is_file():
        return None
    sha = hashlib.sha256(obj.read_bytes()).hexdigest()
    if sha != meta.get("obj_sha256"):
        return None
    key = (sha, float(cmap.res), tuple(cmap.origin[:2]), cmap.shape[:2], round(zlo, 4), round(zhi, 4), spacing)
    if key not in _EXACT:
        from scipy.spatial import cKDTree
        from .objmap import load_obj, sample_triangles
        tri = [V[F] for name, (V, F) in load_obj(obj).items() if not name.startswith(FLOOR_MARKING_PREFIXES)]
        P = sample_triangles(np.concatenate(tri), spacing)
        P = P[(P[:, 2] >= zlo) & (P[:, 2] <= zhi)][:, :2]
        nx, ny = cmap.shape[:2]
        centres = cmap.origin[:2] + (np.stack(np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij"), -1)
                                     .reshape(-1, 2) + 0.5) * cmap.res
        d, _ = cKDTree(P).query(centres, k=1)
        _EXACT[key] = np.maximum(d - spacing * np.sqrt(2.0) / 2.0, 0.0).reshape(nx, ny)
    return _EXACT[key].copy()


def plan_grids(cmap: CourseMap, doc, cruise_z=None, surface=True):
    """-> (obstacle band, blocked = inflated obstacles | keep-outs, clearance field metres).
    The clearance field comes from the published surfaces when the map is the published layout
    (surface_clearance) and `surface` is set, else from the voxel raster."""
    z = doc["cruise_z_m"] if cruise_z is None else cruise_z
    band = doc.get("plan_band_m", 0.25)
    obst = Grid2D(cmap.band(z - band, z + band), cmap.origin[:2].copy(), cmap.res)
    dist = surface_clearance(doc, cmap, z - band, z + band) if surface else None
    if dist is None:
        dist = obst.distance()
    v = doc["vehicle"]
    blocked = dist < (v["radius_m"] + v["margin_m"])
    for ko in doc.get("keepout", []):
        kg = cmap.band(ko["zlo"], ko["zhi"], only_prefixes=ko["groups"])
        kd = ndimage.distance_transform_edt(~kg) * cmap.res
        blocked |= kd < ko["radius_m"]
    return obst, Grid2D(blocked, obst.origin, cmap.res), dist


def scan_field(grid: Grid2D, doc):
    """Likelihood-field distances for scan matching: metres to the nearest mapped surface inside
    the enclosure, and 0 beyond it (unknown space seen through the net costs nothing)."""
    dist = grid.distance()
    enc = doc.get("enclosure")
    if enc:
        nx, ny = grid.occ.shape
        xy = grid.xy(np.stack(np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij"), -1))
        outside = np.any((xy < np.asarray(enc["lo"])) | (xy > np.asarray(enc["hi"])), axis=-1)
        dist[outside] = 0.0
    return dist


def see_through_grid(cmap: CourseMap, zlo=0.3, zhi=1.6, min_fill=0.6):
    """Cells of scan_grid a horizontal beam can legitimately pass: the nets (2 mm strands, returned only
    sometimes), and columns whose surfaces fill less than min_fill of the band's height (the window
    opening, the tunnel roof, boulder tops). Seeing through these is not a map conflict."""
    k0, k1 = max(0, cmap.z_index(zlo)), min(cmap.shape[2] - 1, cmap.z_index(zhi))
    solid = cmap.occupied[:, :, k0:k1 + 1] & ~cmap.group_mask(FLOOR_MARKING_PREFIXES)[:, :, k0:k1 + 1]
    band = solid.any(axis=2)
    return cmap.band(zlo, zhi, only_prefixes=("Net",)) | (band & (solid.mean(axis=2) < min_fill))


def scan_grid(cmap: CourseMap, zlo=0.3, zhi=1.6):
    """Surfaces a horizontal depth/ToF scan can hit between zlo and zhi (floor paint excluded)."""
    return Grid2D(cmap.band(zlo, zhi), cmap.origin[:2].copy(), cmap.res)


def land_columns(doc, cmap: CourseMap, start_xy, start_yaw_deg):
    """Where a fault landing may descend: the 2-D cells whose whole vertical column, from the floor up
    to the top of the flight band, has no published surface within the vehicle's rotor reach + margin
    (floor marking excluded: landing on paint is fine). Built from the full 3-D map, not the flight
    band: the tube, its cradles and the boulders lie below the band, and run012's fault landing in
    place over the tube came down onto it. Surface-derived when the map is the published layout
    (surface_clearance), else the voxel raster less half a cell diagonal (a lower bound).
    -> JSON-able dict for the route file: grid rows ('1' = clear), origin, res, and the take-off pose
    that relates refnav's (forward, left) frame to the course frame (planner.to_route's convention)."""
    v = doc["vehicle"]
    need = float(v["radius_m"]) + float(v["margin_m"])
    top = float(doc["cruise_z_m"]) + float(doc.get("plan_band_m", 0.25))
    dist = surface_clearance(doc, cmap, 0.0, top)
    if dist is None:
        g = Grid2D(cmap.band(0.0, top), cmap.origin[:2].copy(), cmap.res)
        dist = g.distance() - cmap.res * np.sqrt(2.0) / 2.0
    clear = dist >= need
    return {"origin": [float(c) for c in cmap.origin[:2]], "res": float(cmap.res),
            "rows": ["".join("1" if c else "0" for c in row) for row in clear],
            "start": [float(start_xy[0]), float(start_xy[1]), float(start_yaw_deg)],
            "need_m": need, "top_m": top}
