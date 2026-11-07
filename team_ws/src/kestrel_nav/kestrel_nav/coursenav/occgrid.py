"""Live 2D occupancy grid, built during ONE run from the depth scan, plus the landmark field the
localizer matches against. PURE (numpy + scipy).

Two maps, kept separate:
  * the PRIOR map: surfaces of the published course OBJ in the scan band (coursenav.course.scan_grid).
    Loaded before the run; allowed by rule 6.4. This is the localizer's ground reference.
  * the LIVE map (this module): log-odds occupancy filled from this run's depth scans, placed with
    the localizer's own pose estimate. It starts EMPTY (unknown) at every launch.

How a scan fills the live map (inverse sensor model, per beam):
  * cells the beam passed through, up to one cell before its return  -> log-odds += l_free (< 0);
  * the cell of the return                                            -> log-odds += l_occ  (> 0);
  * a bearing with no return adds NOTHING (no return is unknown, never free; see coursenav.scan).
  Each cell is updated at most once per scan, and log-odds are clamped to [l_min, l_max] so the map
  can still change (a person walking through, a net that moves).
Integration is GATED on the localizer's confidence: a scan placed with a wrong pose would write
walls in the wrong place and then pull the localizer towards them (self-reinforcing drift).

What the live map is used for:
  * landmarks(): confidently occupied live cells that the prior map does NOT explain (net strands,
    the room behind the net, poles, anything not in the OBJ). With use_in_pf they join the prior's
    surfaces in the likelihood field, so open areas with few mapped walls still constrain the pose.
  * conflicts(): prior surfaces that this run keeps seeing THROUGH (the built course differs from
    the OBJ, or the pose is wrong). Diagnostic only: prior surfaces are never removed in flight.
    Surfaces a horizontal beam is EXPECTED to pass (coursenav.course.see_through_grid: the nets, the
    window opening, the tunnel roof, boulder tops) are passed as see_through and never count.
  * save(): a record for review and for the judges. RULE 6.4: a later run must NEVER load it.
    No flight code calls load(); it exists for the review tools only.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from .course import Grid2D
from .pf import compose
from .scan import Scan


@dataclass
class OccParams:
    l_occ: float = 0.85              # log-odds added at a return (p = 0.70)
    l_free: float = -0.40            # log-odds added where a beam passed through (p = 0.40)
    l_min: float = -2.0
    l_max: float = 3.5
    occ_thresh: float = 2.0          # "confidently occupied": >= 3 net hits more than misses
    free_thresh: float = -1.2        # "confidently free": >= 3 net pass-throughs
    gate_sigma_xy_m: float = 0.06    # integrate only when the localizer is this sure ...
    gate_sigma_yaw_deg: float = 2.0  # ... in position and heading
    novel_min_m: float = 0.15        # an occupied live cell farther than this from any prior surface is new
    max_range_m: float = 4.0         # do not carve or mark beyond this (stereo error ~ r^2)
    use_in_pf: bool = False          # add novel landmarks to the localizer's likelihood field
    landmark_refresh_s: float = 1.0  # rebuild the fused field at most this often
    min_landmark_cells: int = 4      # ignore isolated specks (connected components smaller than this)


class LiveGrid:
    def __init__(self, prior: Grid2D, params: OccParams = None, see_through=None):
        self.p = params or OccParams()
        self.prior = prior
        self.see_through = np.zeros(prior.occ.shape, bool) if see_through is None else np.asarray(see_through, bool)
        self.prior_dist = prior.distance()
        self.L = np.zeros(prior.occ.shape, np.float32)          # log-odds; 0 = unknown
        self.seen = np.zeros(prior.occ.shape, np.uint16)          # scans that touched each cell
        self.scans_in = 0
        self.scans_gated = 0
        self._field = None
        self._field_t = -math.inf

    # ------------------------------------------------------------------ filling
    def gate(self, cov) -> bool:
        """True if the localizer is confident enough to place a scan in the course frame."""
        cov = np.asarray(cov, float)
        sxy = math.sqrt(max(cov[0, 0], cov[1, 1], 0.0))
        syaw = math.degrees(math.sqrt(max(cov[2, 2], 0.0)))
        return sxy <= self.p.gate_sigma_xy_m and syaw <= self.p.gate_sigma_yaw_deg

    def beam_cells(self, pose, scan: Scan):
        """-> (free linear indices, occupied linear indices), unique, occupied wins within one scan."""
        keep = scan.range <= self.p.max_range_m
        r, b = scan.range[keep], scan.bearing[keep]
        shape = np.array(self.L.shape)
        if len(r) == 0:
            return np.zeros(0, int), np.zeros(0, int)
        x0, y0, th = float(pose[0]), float(pose[1]), float(pose[2])
        ang = th + b
        ca, sa = np.cos(ang), np.sin(ang)
        res, org = self.prior.res, self.prior.origin
        # free: samples every res/2 from the body to one cell short of the return
        ts = np.arange(0.0, self.p.max_range_m, res * 0.5)
        X = x0 + ca[:, None] * ts[None]
        Y = y0 + sa[:, None] * ts[None]
        ok = ts[None] < (r[:, None] - res)
        ij = np.floor((np.stack([X[ok], Y[ok]], -1) - org) / res).astype(int)
        ij = ij[np.all((ij >= 0) & (ij < shape), axis=1)]
        free = np.unique(np.ravel_multi_index(ij.T, self.L.shape)) if len(ij) else np.zeros(0, int)
        # occupied: the return cell
        e = np.floor((np.stack([x0 + ca * r, y0 + sa * r], -1) - org) / res).astype(int)
        e = e[np.all((e >= 0) & (e < shape), axis=1)]
        occ = np.unique(np.ravel_multi_index(e.T, self.L.shape)) if len(e) else np.zeros(0, int)
        free = np.setdiff1d(free, occ, assume_unique=True)
        return free, occ

    def integrate(self, pose, scan: Scan, cov=None) -> bool:
        """Add one scan seen from course pose (x, y, yaw). Returns False if gated out or empty."""
        self.scans_in += 1
        if scan is None or len(scan) == 0:
            return False
        if cov is not None and not self.gate(cov):
            self.scans_gated += 1
            return False
        free, occ = self.beam_cells(pose, scan)
        flat, seen = self.L.reshape(-1), self.seen.reshape(-1)
        flat[free] += self.p.l_free
        flat[occ] += self.p.l_occ
        np.clip(flat, self.p.l_min, self.p.l_max, out=flat)
        touched = np.concatenate([free, occ])
        seen[touched] = np.minimum(seen[touched].astype(np.int32) + 1, 65535)
        return True

    # ------------------------------------------------------------------ products
    def probability(self):
        return 1.0 - 1.0 / (1.0 + np.exp(self.L))

    def occupied(self):
        return self.L >= self.p.occ_thresh

    def free(self):
        return self.L <= self.p.free_thresh

    def landmarks(self):
        """Confident live surfaces the prior map does not explain, minus isolated specks."""
        novel = self.occupied() & (self.prior_dist > self.p.novel_min_m)
        if not novel.any() or self.p.min_landmark_cells <= 1:
            return novel
        lab, n = ndimage.label(novel, structure=np.ones((3, 3), bool))
        sizes = ndimage.sum(novel, lab, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, bool)
        keep[1:] = sizes >= self.p.min_landmark_cells
        return keep[lab]

    def conflicts(self):
        """Prior surface cells this run has confidently seen through."""
        return self.prior.occ & ~self.see_through & self.free()

    def landmark_field(self, now=None, force=False):
        """Distance field over prior surfaces + live landmarks for the localizer (cached; rebuilt at
        most every landmark_refresh_s). Returns None while there are no landmarks."""
        now = time.monotonic() if now is None else now
        if not force and self._field is not None and now - self._field_t < self.p.landmark_refresh_s:
            return self._field
        lm = self.landmarks()
        self._field_t = now
        self._field = None if not lm.any() else \
            ndimage.distance_transform_edt(~(self.prior.occ | lm)) * self.prior.res
        return self._field

    def stats(self):
        return dict(scans_in=self.scans_in, scans_gated=self.scans_gated,
                    known_cells=int((self.seen > 0).sum()), occupied=int(self.occupied().sum()),
                    landmarks=int(self.landmarks().sum()), conflicts=int(self.conflicts().sum()))

    # ------------------------------------------------------------------ record (never reloaded)
    def save(self, path, **meta):
        """Run record for review. RULE 6.4: data captured on one run must not feed a later run."""
        m = dict(meta, origin=[float(v) for v in self.prior.origin], res=float(self.prior.res),
                 params=asdict(self.p), stats=self.stats(),
                 rule_6_4="RUN RECORD ONLY. Flight code must never load this file.")
        np.savez_compressed(path, logodds=self.L, seen=self.seen, prior=self.prior.occ,
                            landmarks=self.landmarks(), conflicts=self.conflicts(), meta=np.array(json.dumps(m)))

    @staticmethod
    def load_record(path):
        """For review tools only (tools/occ_view.py). Not imported by any flight or localizer code."""
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files if k != "meta"} | {"meta": json.loads(str(z["meta"]))}


def pose_scan_to_world(pose, scan: Scan):
    """(n,2) course-frame endpoints of a scan seen from body pose (x, y, yaw)."""
    e = scan.endpoints()
    p = np.column_stack([e, np.zeros(len(e))])
    return compose(np.asarray(pose, float)[None].repeat(len(e), 0), p)[:, :2]
