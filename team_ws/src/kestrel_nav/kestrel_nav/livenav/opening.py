"""Depth-measured opening (window / tunnel mouth) relative to the body. PURE (numpy).

Ported from workspace ``refnav/gate.py`` (``find_opening``, ``combine``, ``to_leg_frame``) at
``matthew352am``/``diagnostic`` with no change to the measurement rules. That module works in level
body (forward, RIGHT, DOWN) points; livenav's points are level body (forward, LEFT, UP), so
``find_opening_flu`` converts and calls it.

Why it exists here: the window is 0.871 m wide on the 5 cm map grid (0.914 m in the OBJ) and a
7-inch-prop skybeta reaches 0.294 m from the body origin, so a centred crossing has 0.14-0.16 m per
side. Sim runs 3/4 measured 0.16-0.20 m of PX4/VIO estimate error against truth, so a crossing placed
from the global estimate alone can touch. The opening's own jambs, seen in the depth image from the
pre point, are a direct measurement of where the opening is relative to the body.

Results (``lateral_m`` positive = opening centre is to the LEFT of the body axis; ``heading_rad``
positive = crossing direction lies to the RIGHT of the body axis, 0.0 when the jambs are not on one
plane; ``width_m``; ``range_m`` = forward distance of the jambs' near corners).
"""
from __future__ import annotations

import math

import numpy as np


def find_opening(points, meta, slab_m=0.4, bin_m=0.02, band_m=0.25, min_edge_pts=6, width_tol=0.35,
                 edge_search_m=0.45):
    """Find the opening described by meta in one frame's level (forward, right, down) points."""
    if points is None or len(points) == 0 or not meta:
        return None
    dist = float(meta.get("dist_m", 0.0))
    width = float(meta.get("width_m", 0.0))
    if dist <= 0 or width <= 0:
        return None
    P = np.asarray(points, float)
    fwd, right, down = P[:, 0], P[:, 1], P[:, 2]
    slab = (fwd >= dist - slab_m) & (fwd <= dist + slab_m)
    band = np.abs(down) <= band_m
    sel = slab & band
    if sel.sum() < 2 * min_edge_pts:
        return None
    r, f = right[sel], fwd[sel]
    half = width / 2.0
    lo = -half - edge_search_m
    nb = int(math.ceil((2 * (half + edge_search_m)) / bin_m))
    k = np.floor((r - lo) / bin_m).astype(int)
    inside = (k >= 0) & (k < nb)
    counts = np.bincount(k[inside], minlength=nb)
    occ = counts >= 3

    def edge(from_bin, step):
        b = from_bin
        while 0 <= b < nb:
            if occ[b]:
                return b
            b += step
        return None

    centre_bin = int((0.0 - lo) / bin_m)
    inner_l = max(0, int((-half * 0.5 - lo) / bin_m))
    inner_r = min(nb - 1, int((half * 0.5 - lo) / bin_m))
    bl = edge(inner_l, -1)
    br = edge(inner_r, +1)
    have_l = bl is not None and bl < centre_bin
    have_r = br is not None and br > centre_bin
    if not have_l and not have_r:
        return None
    r_left = lo + (bl + 1) * bin_m if have_l else None
    r_right = lo + br * bin_m if have_r else None
    one_edge = not (have_l and have_r)
    if one_edge:
        if have_l:
            r_right = r_left + width
        else:
            r_left = r_right - width
    seen_w = r_right - r_left
    if abs(seen_w - width) > width_tol * width:
        return None

    def edge_pts(inner, outward):
        m = sel.copy()
        m[sel] = (r <= inner) & (r >= inner - 0.12) if outward < 0 else (r >= inner) & (r <= inner + 0.12)
        return m
    left_pts = edge_pts(r_left, -1) if have_l else np.zeros(len(P), bool)
    right_pts = edge_pts(r_right, +1) if have_r else np.zeros(len(P), bool)
    n_l, n_r = int(left_pts.sum()), int(right_pts.sum())
    if (have_l and n_l < min_edge_pts) or (have_r and n_r < min_edge_pts):
        return None
    f_l = float(np.percentile(fwd[left_pts], 15)) if have_l else None
    f_r = float(np.percentile(fwd[right_pts], 15)) if have_r else None
    heading = 0.0
    if not one_edge and abs(f_l - f_r) < 0.3:
        t_f, t_r = f_l - f_r, r_left - r_right
        d_f, d_r = -t_r, t_f
        if d_f < 0:
            d_f, d_r = -d_f, -d_r
        heading = math.atan2(d_r, d_f)
    centre_r = 0.5 * (r_left + r_right)
    rng = [v for v in (f_l, f_r) if v is not None]
    return {"lateral_m": -centre_r, "heading_rad": heading, "width_m": seen_w, "dz_m": None,
            "n_left": n_l, "n_right": n_r, "range_m": float(np.mean(rng)), "one_edge": one_edge}


def find_opening_flu(pts_level, meta, **kw):
    """livenav level points (forward, left, up) -> find_opening's (forward, right, down)."""
    if pts_level is None or len(pts_level) == 0:
        return None
    P = np.asarray(pts_level, float)
    return find_opening(np.column_stack([P[:, 0], -P[:, 1], -P[:, 2]]), meta, **kw)


def combine(estimates):
    """Median of several single-frame estimates (robust to one bad frame)."""
    ests = [e for e in estimates if e]
    if not ests:
        return None
    out = {}
    for key in ("lateral_m", "heading_rad", "width_m", "range_m"):
        vals = [e[key] for e in ests if e.get(key) is not None]
        out[key] = float(np.median(vals)) if vals else (0.0 if key != "width_m" else float("nan"))
    out["lateral_spread_m"] = float(np.ptp([e["lateral_m"] for e in ests]))
    out["two_edge_frames"] = sum(1 for e in ests if not e.get("one_edge"))
    out["n_frames"] = len(ests)
    return out


def to_leg_frame(est, bearing_rad):
    """Re-express a body measurement relative to the leg whose direction lies bearing_rad to the
    RIGHT of the body axis. The opening centre sits range_m ahead and lateral_m left of the body."""
    if est is None:
        return None
    b = float(bearing_rad)
    f, r = float(est["range_m"]), -float(est["lateral_m"])
    across_right = -f * math.sin(b) + r * math.cos(b)
    out = dict(est)
    out["lateral_m"] = -across_right
    out["heading_rad"] = float(est.get("heading_rad", 0.0)) - b
    out["bearing_rad"] = b
    return out
