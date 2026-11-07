"""Observation-based speed ceilings. No course names or controller gain updates.

These ceilings supplement corridor/estimator checks; they cannot certify unseen
space or compensate for localization bias. Delay/deceleration are assumptions
until identified on the actual aircraft.
"""
import math

import numpy as np


def braking_cap(distance, reserve, decel, delay):
    """Largest v satisfying v*delay + v**2/(2*decel) <= distance-reserve."""
    if not all(math.isfinite(x) for x in (reserve, decel, delay)) or min(reserve, delay) < 0 or decel <= 0:
        return 0.0
    if math.isnan(distance) or distance == -math.inf:
        return 0.0
    if distance == math.inf:
        return math.inf
    room = max(0.0, distance - reserve)
    return math.sqrt((decel * delay) ** 2 + 2 * decel * room) - decel * delay


def side_cap(points, bearing, cruise, near, extra, minimum, reach, band, min_hits):
    """Close returns on either side of the commanded bearing lower the ceiling.

    Points are level heading-frame (forward, right, down). Only a short forward
    region at body height contributes. Missing side returns never authorize
    motion: the caller must independently establish observed forward clearance.
    """
    if points is None or len(points) == 0:
        return cruise
    pts = np.asarray(points, float)
    c, s = math.cos(bearing), math.sin(bearing)
    along = pts[:, 0] * c + pts[:, 1] * s
    across = -pts[:, 0] * s + pts[:, 1] * c
    selected = np.isfinite(pts).all(axis=1) & (along > 0) & (along < reach) & (np.abs(pts[:, 2]) < band)
    distance = math.inf
    for sign in (-1, 1):
        hits = np.sort(sign * across[selected & (sign * across > 0)])
        if len(hits) >= min_hits:
            distance = min(distance, float(hits[min_hits - 1]))
    if not math.isfinite(distance):
        return cruise
    fraction = min(1.0, max(0.0, (distance - near) / extra)) if extra > 0 else 0.0
    low = min(cruise, max(0.0, minimum))
    return low + (cruise - low) * fraction
