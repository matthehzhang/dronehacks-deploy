"""Course file + map loading and the derived 2D grids (planning, keep-out, scan matching)."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from .objmap import FLOOR_MARKING_PREFIXES, CourseMap

PROJECT = Path(__file__).resolve().parent.parent / "data"   # kestrel_nav/data: course/ and maps/


_RESOLVED = {}


def load_course(path=PROJECT / "course" / "course_v3.json", derive=True):
    """Course file -> dict. With derive (default), every goal position is measured from the published
    OBJ by coursenav.semantics (the file itself stores none: rule 9.5); cached per file version."""
    path = Path(path)
    doc = json.loads(path.read_text())
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


def plan_grids(cmap: CourseMap, doc, cruise_z=None):
    """-> (obstacle band, blocked = inflated obstacles | keep-outs, clearance field metres)."""
    z = doc["cruise_z_m"] if cruise_z is None else cruise_z
    band = doc.get("plan_band_m", 0.25)
    obst = Grid2D(cmap.band(z - band, z + band), cmap.origin[:2].copy(), cmap.res)
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
