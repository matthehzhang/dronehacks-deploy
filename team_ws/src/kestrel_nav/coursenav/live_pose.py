"""Declared launch frame and sensor-only map geometry; no saved course obstacles."""
import math
import numpy as np
from .course import Grid2D
from .pf import compose


def arena_grid(half_m=3.048, res=.05, fence=False):
    origin = np.full(2, -half_m-.10)
    n = int(math.ceil((2*half_m+.20)/res))
    occ = np.zeros((n, n), bool)
    if fence:
        x = origin[0]+(np.arange(n)+.5)*res
        occ = (np.abs(x[:, None]) >= half_m) | (np.abs(x[None, :]) >= half_m)
    return Grid2D(occ, origin, res)


def vio_pose(fixture, pose, covariance):
    """Express VIO in the declared course frame, retaining its measured covariance."""
    cov = np.asarray(covariance, float).reshape(6, 6)[np.ix_([0, 1, 5], [0, 1, 5])]
    if not np.isfinite(cov).all() or np.any(np.diag(cov) <= 0):
        raise ValueError('VIO covariance missing or invalid')
    c, s = math.cos(fixture[2]), math.sin(fixture[2])
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return compose(np.asarray(fixture, float), np.asarray(pose, float)), R @ cov @ R.T
