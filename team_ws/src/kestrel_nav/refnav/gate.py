"""refnav.gate - depth-based gate alignment. PURE (numpy only).

Before crossing an opening (a window, the gap past a wall's end, a tunnel mouth) the mission hovers
facing it and measures the opening itself in the depth image, instead of trusting the global
position estimate to the last few centimetres. The expected geometry comes from the planner (which
measured it on the published map, coursenav.semantics): how far ahead the opening's plane should be,
how wide it is, and (for a wall opening) where its sill and header are relative to the body height.

Frame: level body points (forward, right, down) in metres, relative to the body origin
(refnav.depth.DepthFrame.points). Results:
  lateral_m    how far the opening's centre lies to the LEFT of the body's axis (move left by this);
  heading_rad  how far the perpendicular crossing direction lies to the RIGHT of the body's axis
               (NED yaw: positive = clockwise), from the two edges' forward distances; 0.0 when the
               two edges are not on one plane (a gap between two different walls);
  dz_m         how far UP the opening's vertical centre lies (None when the sill/header were not
               seen; a window can use one edge plus its mapped height);
  width_m      the measured opening width;
  n_left, n_right   the number of points supporting each edge.
"""
from __future__ import annotations

import math

import numpy as np


def find_opening(points, meta, slab_m=0.4, bin_m=0.02, band_m=0.25, min_edge_pts=6, width_tol=0.35,
                 edge_search_m=0.45, height_max_m=0.30):
    """Find the opening described by meta in one frame's level points. -> dict or None.

    meta: {"dist_m": expected distance ahead of the opening's plane (required),
           "width_m": expected width (required),
           "dz_lo": expected sill height relative to the body (negative = below, optional),
           "dz_hi": expected header height relative to the body (positive = above, optional)}
    The edges are searched in a forward slab around dist_m, in a vertical band around the body
    height, as the first occupied 2 cm bins outward from the expected edge positions."""
    if points is None or len(points) == 0 or not meta:
        return None
    dist = float(meta.get("dist_m", 0.0))
    width = float(meta.get("width_m", 0.0))
    if dist <= 0 or width <= 0:
        return None
    P = np.asarray(points, float)
    fwd, right, down = P[:, 0], P[:, 1], P[:, 2]
    slab = (fwd >= dist - slab_m) & (fwd <= dist + slab_m)
    # A window displaced vertically can put its sill/header inside the normal jamb band.
    # Use the body-height strip so those surfaces are not mistaken for side jambs.
    jamb_band = min(band_m, 0.10) if meta.get("group") == "window" else band_m
    band = np.abs(down) <= jamb_band
    sel = slab & band
    if sel.sum() < 2 * min_edge_pts:
        return None
    r, f = right[sel], fwd[sel]
    half = width / 2.0
    # occupied lateral bins (>= 3 points) in the slab at body height
    lo = -half - edge_search_m
    nb = int(math.ceil((2 * (half + edge_search_m)) / bin_m))
    k = np.floor((r - lo) / bin_m).astype(int)
    inside = (k >= 0) & (k < nb)
    counts = np.bincount(k[inside], minlength=nb)
    occ = counts >= 3

    def edge(from_bin, step):
        """First occupied bin walking from from_bin in direction step (-1 left, +1 right)."""
        b = from_bin
        while 0 <= b < nb:
            if occ[b]:
                return b
            b += step
        return None

    centre_bin = int((0.0 - lo) / bin_m)
    inner_l = max(0, int((-half * 0.5 - lo) / bin_m))          # start the search inside the opening
    inner_r = min(nb - 1, int((half * 0.5 - lo) / bin_m))
    bl = edge(inner_l, -1)
    br = edge(inner_r, +1)
    have_l = bl is not None and bl < centre_bin
    have_r = br is not None and br > centre_bin
    if not have_l and not have_r:
        return None
    r_left = lo + (bl + 1) * bin_m if have_l else None            # inner face of the left edge
    r_right = lo + br * bin_m if have_r else None
    one_edge = not (have_l and have_r)
    if one_edge:
        # one jamb in view (a yaw error swung the other out of the field of view): place the opening
        # with the expected width from the edge that is seen
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
    # each edge's NEAR corner (a low percentile of its forward distances): the jamb face itself may
    # be seen over any part of its depth, its near corner is a fixed feature of the opening's plane
    f_l = float(np.percentile(fwd[left_pts], 15)) if have_l else None
    f_r = float(np.percentile(fwd[right_pts], 15)) if have_r else None
    heading = 0.0
    if not one_edge and abs(f_l - f_r) < 0.3:                    # one plane: both jambs at about the same range
        # wall tangent from the right jamb to the left jamb; the crossing is its forward perpendicular
        t_f, t_r = f_l - f_r, r_left - r_right
        d_f, d_r = -t_r, t_f
        if d_f < 0:
            d_f, d_r = -d_f, -d_r
        heading = math.atan2(d_r, d_f)
    centre_r = 0.5 * (r_left + r_right)
    rng = [v for v in (f_l, f_r) if v is not None]
    out = {"lateral_m": -centre_r, "heading_rad": heading, "width_m": seen_w, "dz_m": None,
           "n_left": n_l, "n_right": n_r, "range_m": float(np.mean(rng)), "one_edge": one_edge}
    # vertical edges: the sill (below) and header (above) between the jambs, in the same slab
    if meta.get("group") == "window" and meta.get("dz_lo") is not None and meta.get("dz_hi") is not None:
        out.update(_window_height(P, meta, r_left, r_right, out["range_m"], heading,
                                  height_max_m, min_edge_pts))
    elif meta.get("dz_lo") is not None and meta.get("dz_hi") is not None:
        mid = slab & (right > r_left + 0.08) & (right < r_right - 0.08)
        below = mid & (down > 0.08)
        above = mid & (down < -0.08)
        if below.sum() >= min_edge_pts and above.sum() >= min_edge_pts:
            sill_down = float(np.percentile(down[below], 10))     # the sill's top edge (least far down)
            head_down = float(np.percentile(down[above], 90))     # the header's bottom edge
            exp_sill, exp_head = -float(meta["dz_lo"]), -float(meta["dz_hi"])   # as "down"
            if abs(sill_down - exp_sill) < 0.25 and abs(head_down - exp_head) < 0.25:
                out["dz_m"] = -0.5 * (sill_down + head_down)       # up is negative down
                out["height_m"] = sill_down - head_down
    return out


def _window_height(points, meta, left, right, plane_range, heading, max_shift, min_hits):
    """Use a visible window edge and mapped aperture height when its other edge is out of view.

    Match the measured jamb plane, not the whole range slab (which can include the floor or
    background). Require a wall patch with both horizontal and vertical extent. If both edges
    are measured, their separation must agree with the map; never pick one conflicting edge.
    """
    height = float(meta["dz_hi"]) - float(meta["dz_lo"])
    if not math.isfinite(height) or height <= 0 or not math.isfinite(max_shift) or max_shift <= 0:
        return {}
    fwd, across, down = points.T
    normal_dist = ((fwd - plane_range) * math.cos(heading)
                   + (across - (left + right) / 2) * math.sin(heading))
    mid = (np.abs(normal_dist) <= 0.08) & (across > left + 0.08) & (across < right - 0.08)
    edges = []
    for name, mask, percentile, expected, half_height in (
        ("sill", mid & (down > 0.08), 10, -float(meta["dz_lo"]), height / 2),
        ("header", mid & (down < -0.08), 90, -float(meta["dz_hi"]), -height / 2),
    ):
        if (mask.sum() < min_hits or np.ptp(down[mask]) < 0.04
                or np.ptp(across[mask]) < 0.15):
            continue
        edge = float(np.percentile(down[mask], percentile))
        # Allow a sampling tolerance, but leave correction magnitude bounded by the controller.
        if abs(edge - expected) > max_shift + 0.08:
            return {}
        edges.append((name, half_height - edge, edge))
    if not edges:
        return {}
    if len(edges) == 2 and abs(edges[0][2] - edges[1][2] - height) > 0.12:
        return {}
    return {"dz_m": float(np.mean([e[1] for e in edges])),
            "height_source": "+".join(e[0] for e in edges),
            "height_m": edges[0][2] - edges[1][2] if len(edges) == 2 else height}


def combine(estimates):
    """Median of several single-frame estimates (robust to one bad frame)."""
    ests = [e for e in estimates if e]
    if not ests:
        return None
    out = {}
    for key in ("lateral_m", "heading_rad", "width_m", "range_m"):
        vals = [e[key] for e in ests if e.get(key) is not None]
        out[key] = float(np.median(vals)) if vals else (0.0 if key != "width_m" else float("nan"))
    dz = [e["dz_m"] for e in ests if e.get("dz_m") is not None]
    out["dz_m"] = (float(np.median(dz))
                   if len(dz) >= max(1, (len(ests) + 1) // 2) and np.ptp(dz) <= 0.08 else None)
    out["n_frames"] = len(ests)
    return out


def to_leg_frame(est, bearing_rad):
    """Re-express a body-frame measurement relative to the LEG the aircraft is about to fly, whose
    direction lies bearing_rad to the right of the body's axis (NED: positive = clockwise). The
    opening's centre sits range_m ahead and lateral_m to the left of the body."""
    if est is None:
        return None
    b = float(bearing_rad)
    f, r = float(est["range_m"]), -float(est["lateral_m"])            # body (forward, right)
    across_right = -f * math.sin(b) + r * math.cos(b)                  # in the leg frame
    out = dict(est)
    out["lateral_m"] = -across_right
    out["heading_rad"] = float(est.get("heading_rad", 0.0)) - b
    out["bearing_rad"] = b
    return out
