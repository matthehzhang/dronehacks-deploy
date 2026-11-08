"""Waypoint configuration and generic metric grid primitives; no origin geometry."""
from pathlib import Path
import json
import os
from dataclasses import dataclass
import numpy as np
from scipy import ndimage
from .airframe import vehicle_geometry
PROJECT = Path(__file__).resolve().parent.parent

def load_course(path=None, derive=True):
    path=Path(path or os.environ.get('KESTREL_COURSE_FILE') or PROJECT/'course/course_depth_waypoints.json')
    doc=json.loads(path.read_text())
    doc['vehicle']=vehicle_geometry(doc['vehicle'])
    doc['_dir']=str(PROJECT)
    return doc

def resolve(doc,key):
    p=Path(doc[key])
    return p if p.is_absolute() else Path(doc.get('_dir',PROJECT))/p

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
