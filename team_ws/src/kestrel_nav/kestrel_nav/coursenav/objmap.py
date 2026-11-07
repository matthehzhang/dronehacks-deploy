"""Published course OBJ -> 3D voxel occupancy -> 2D height-band grids. PURE (numpy).

Frame: the OBJ's own source frame (metres, Z up, origin at the floor centre),
which this project calls the *course* frame (x east, y north, z up: ENU).

Voxels mark triangle SURFACES (dense barycentric sampling, spacing <= res/2).
Thin walls (10 cm) therefore occupy both faces; closed interiors are not filled,
which does not matter for 2D planning (inflation) or scan matching (surfaces).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

# Flat floor paint: never an obstacle, never a scan target above the floor.
FLOOR_MARKING_PREFIXES = ("Guideline", "Launch mark", "Landing pad", "Approach ring", "Floor")


def load_obj(path):
    """-> {group name: (V (n,3) float, F (m,3) int triangles, 0-based into V)}.

    Polygons are fan-triangulated. Vertex indices may be negative (relative)."""
    verts = []
    faces = {}
    cur = "default"
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("g "):
                cur = line[2:].strip()
                faces.setdefault(cur, [])
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    i = int(tok.split("/")[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                for k in range(1, len(idx) - 1):
                    faces.setdefault(cur, []).append((idx[0], idx[k], idx[k + 1]))
    V = np.asarray(verts, float)
    return {g: (V, np.asarray(F, int).reshape(-1, 3)) for g, F in faces.items() if len(F)}


def sample_triangles(tri, spacing):
    """Points covering triangles (k,3,3) with spacing <= `spacing`. -> (n,3)."""
    tri = np.asarray(tri, float)
    out = []
    edges = np.stack([np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
                      np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
                      np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)], axis=1).max(axis=1)
    n_div = np.maximum(1, np.ceil(edges / spacing)).astype(int)
    for n in np.unique(n_div):
        sel = tri[n_div == n]
        i, j = np.mgrid[0:n + 1, 0:n + 1]
        keep = (i + j) <= n
        a = (i[keep] / n)[:, None]
        b = (j[keep] / n)[:, None]
        # p = v0 + a (v1 - v0) + b (v2 - v0), for every triangle in `sel`
        p = sel[:, None, 0] + a[None] * (sel[:, None, 1] - sel[:, None, 0]) + b[None] * (sel[:, None, 2] - sel[:, None, 0])
        out.append(p.reshape(-1, 3))
    return np.concatenate(out) if out else np.zeros((0, 3))


@dataclass
class CourseMap:
    """3D occupancy (ix, iy, iz) on a regular grid, plus per-voxel group labels."""
    occupied: np.ndarray              # bool (nx, ny, nz)
    origin: np.ndarray                # course-frame xyz of the grid's lower corner
    res: float
    labels: np.ndarray                # int16 (nx, ny, nz): group id + 1 (0 = free); last writer wins
    groups: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- indexing
    @property
    def shape(self):
        return self.occupied.shape

    def xy_to_ij(self, xy):
        xy = np.asarray(xy, float)
        return np.floor((xy - self.origin[:2]) / self.res).astype(int)

    def ij_to_xy(self, ij):
        return self.origin[:2] + (np.asarray(ij, float) + 0.5) * self.res

    def z_index(self, z):
        return int(math.floor((z - self.origin[2]) / self.res))

    def group_mask(self, prefixes):
        ids = [k + 1 for k, g in enumerate(self.groups) if g.startswith(tuple(prefixes))]
        return np.isin(self.labels, ids)

    # ---------------------------------------------------------------- 2D grids
    def band(self, zlo, zhi, exclude_prefixes=FLOOR_MARKING_PREFIXES, only_prefixes=None):
        """2D bool (nx, ny): any occupied voxel whose centre is in [zlo, zhi]."""
        k0 = max(0, self.z_index(zlo))
        k1 = min(self.shape[2] - 1, self.z_index(zhi))
        occ = self.occupied[:, :, k0:k1 + 1]
        if exclude_prefixes:
            occ = occ & ~self.group_mask(exclude_prefixes)[:, :, k0:k1 + 1]
        if only_prefixes is not None:
            occ = occ & self.group_mask(only_prefixes)[:, :, k0:k1 + 1]
        return occ.any(axis=2)

    # ---------------------------------------------------------------- 3D rays
    def raycast(self, origins, dirs, max_range, step=None, ignore_prefixes=()):
        """Vectorised ray march. origins/dirs (n,3) course frame (dirs normalised inside).
        -> (n,) distance to the first occupied voxel (inf: none within max_range or left the grid)."""
        origins = np.atleast_2d(np.asarray(origins, float))
        dirs = np.atleast_2d(np.asarray(dirs, float))
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        step = step or self.res * 0.5
        ts = np.arange(step, max_range + 1e-9, step)
        occ = self.occupied
        if ignore_prefixes:
            occ = occ & ~self.group_mask(ignore_prefixes)
        out = np.full(len(origins), np.inf)
        chunk = max(1, 400000 // max(1, len(ts)))
        for s in range(0, len(origins), chunk):
            o, d = origins[s:s + chunk], dirs[s:s + chunk]
            p = o[:, None, :] + d[:, None, :] * ts[None, :, None]
            ijk = np.floor((p - self.origin) / self.res).astype(int)
            inside = np.all((ijk >= 0) & (ijk < np.array(self.shape)), axis=2)
            ijk = np.where(inside[..., None], ijk, 0)
            hit = occ[ijk[..., 0], ijk[..., 1], ijk[..., 2]] & inside
            first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
            out[s:s + chunk] = np.where(first >= 0, ts[np.maximum(first, 0)], np.inf)
        return out

    def distance_field(self):
        """3D metres from each voxel centre to the nearest occupied voxel centre (cached)."""
        if getattr(self, "_edt", None) is None:
            from scipy import ndimage
            self._edt = (ndimage.distance_transform_edt(~self.occupied) * self.res).astype(np.float32)
        return self._edt

    def raycast_fast(self, origins, dirs, max_range, max_iter=600):
        """Same answer as raycast() (first occupied voxel, marched at res/2 near surfaces) but
        sphere-traced through the distance field in open space. -> (distance (n,), voxel ijk (n,3);
        inf / -1 where nothing is hit within max_range or the ray leaves the grid).

        Safe step: a point inside voxel v is at least edt(v) - res*sqrt(3) from any occupied voxel."""
        origins = np.atleast_2d(np.asarray(origins, float))
        dirs = np.atleast_2d(np.asarray(dirs, float))
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        edt = self.distance_field()
        shape = np.array(self.shape)
        n = len(origins)
        dist = np.full(n, np.inf)
        hit_ijk = np.full((n, 3), -1, int)
        t = np.full(n, self.res * 0.5)
        alive = np.arange(n)
        slack, fine = self.res * math.sqrt(3.0), self.res * 0.5
        for _ in range(max_iter):
            if len(alive) == 0:
                break
            p = origins[alive] + dirs[alive] * t[alive, None]
            ijk = np.floor((p - self.origin) / self.res).astype(int)
            inside = np.all((ijk >= 0) & (ijk < shape), axis=1)
            ijk_c = np.where(inside[:, None], ijk, 0)
            occ = self.occupied[ijk_c[:, 0], ijk_c[:, 1], ijk_c[:, 2]] & inside
            hit = alive[occ]
            dist[hit] = t[hit]
            hit_ijk[hit] = ijk[occ]
            step = np.maximum(edt[ijk_c[:, 0], ijk_c[:, 1], ijk_c[:, 2]] - slack, fine)
            t[alive] += step
            keep = inside & ~occ & (t[alive] <= max_range)
            alive = alive[keep]
        return dist, hit_ijk

    # ---------------------------------------------------------------- io
    def save(self, path):
        meta = dict(self.meta, groups=self.groups, res=self.res, origin=[float(c) for c in self.origin])
        np.savez_compressed(path, occupied=self.occupied, labels=self.labels, meta=np.array(json.dumps(meta)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            return cls(z["occupied"].astype(bool), np.asarray(meta["origin"], float), float(meta["res"]),
                       z["labels"], list(meta["groups"]), meta)


def build(groups, res=0.05, bounds=((-3.2, -3.2, -0.05), (3.2, 3.2, 3.15))):
    """groups: {name: (V, F)} -> CourseMap. Every group is kept and labelled."""
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    shape = tuple(np.ceil((hi - lo) / res).astype(int))
    occ = np.zeros(shape, bool)
    labels = np.zeros(shape, np.int16)
    names = sorted(groups)
    for gid, name in enumerate(names):
        V, F = groups[name]
        pts = sample_triangles(V[F], res * 0.5)
        ijk = np.floor((pts - lo) / res).astype(int)
        ok = np.all((ijk >= 0) & (ijk < np.array(shape)), axis=1)
        ijk = ijk[ok]
        occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
        labels[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = gid + 1
    return CourseMap(occ, lo, res, labels, names, {"source": "published OBJ", "algorithm": "surface-sample/1"})
