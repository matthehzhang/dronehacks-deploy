"""Conservative observed-free certification for recovery paths; no ROS or simulator inputs.

OccupancyGrid -1 and ambiguous probabilities are NOT free. Distances account for
cell area, not just its centre. The published grid is in the fixed nominal frame;
the same DriftEKF used for the vehicle converts course samples back into that frame.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy import ndimage


@dataclass
class ObservedFree:
    origin: np.ndarray
    resolution: float
    clearance: np.ndarray
    stamp: float

    @classmethod
    def from_grid(cls, origin, resolution, width, height, data, stamp, free_probability=35):
        if not math.isfinite(resolution) or resolution <= 0 or width <= 0 or height <= 0:
            raise ValueError('invalid local grid dimensions')
        origin = np.asarray(origin, float)
        if origin.shape != (2,) or not np.isfinite(origin).all() or not math.isfinite(stamp):
            raise ValueError('invalid local grid origin or time')
        probabilities = np.asarray(data, dtype=np.int16).reshape(height, width).T
        free = (probabilities >= 0) & (probabilities <= free_probability)
        # Pad with unknown: the edge of the local window is never open space.
        centres = ndimage.distance_transform_edt(np.pad(free, 1, constant_values=False))[1:-1, 1:-1]
        # EDT measures centre-to-centre distance. Subtract the obstacle cell's
        # circumradius AND the query cell's circumradius to bound every point in it.
        clearance = np.maximum(0., centres * resolution - math.sqrt(2.) * resolution)
        clearance[~free] = 0.
        return cls(origin, float(resolution), clearance, float(stamp))

    def fresh(self, now, max_age):
        return math.isfinite(now) and 0 <= now - self.stamp <= max_age

    def distances(self, nominal_points):
        points = np.asarray(nominal_points, float).reshape(-1, 2)
        out = np.zeros(len(points))
        finite = np.isfinite(points).all(axis=1)
        ij = np.zeros(points.shape, dtype=int)
        ij[finite] = np.floor((points[finite] - self.origin) / self.resolution).astype(int)
        ok = finite & np.all((ij >= 0) & (ij < np.array(self.clearance.shape)), axis=1)
        out[ok] = self.clearance[ij[ok, 0], ij[ok, 1]]
        return out

    def allowed_course_cells(self, grid, to_nominal, need):
        indices = np.indices(grid.occ.shape).reshape(2, -1).T
        points = grid.xy(indices)
        # Certify the entire course cell, preventing diagonal/smoothing shortcuts
        # from reaching unknown space between certified cell centres.
        radius = need + grid.res * math.sqrt(2.) / 2
        return (self.distances(to_nominal(points)) >= radius).reshape(grid.occ.shape)

    def certifies_path(self, points_course, to_nominal, need):
        points = np.asarray(points_course, float)
        if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
            return False
        step = min(self.resolution / 2., .025)
        samples = [points[:1]]
        for a, b in zip(points[:-1], points[1:]):
            n = max(1, int(math.ceil(float(np.linalg.norm(b-a)) / step)))
            samples.append(a + (b-a) * np.linspace(0., 1., n+1)[:, None])
        # Every segment point is <= step/2 from a sample: include that in the
        # required radius, so checking samples certifies the continuous sweep.
        return bool(np.all(self.distances(to_nominal(np.concatenate(samples))) >= need + step/2.))
