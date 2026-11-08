"""From the planner's plan to the navigator's mission. PURE. Shared by fly_course.py (the ROS run) and
offline_sim (nav="path"), so both fly the same thing.

    refnav route  = the STOPS only: where the aircraft must hold (the tunnel mouth and exit, the
                    chicane's pre and post points, the pad), in (forward, left) from the take-off pose;
    refnav legs   = their metadata (alignment, passage, hold) from planner.to_legs;
    nav legs      = per refnav leg: the course-frame stop, the pass-through goals before it (named
                    goals such as the maze exit; the planner's via points are dropped, the navigator
                    re-plans), and whether it is a navigator path leg (crossings stay straight legs).

Frames: course_to_ned (the simulator's PX4 frame and the drone's odometry frame differ only in the
declared heading).
"""
from __future__ import annotations

import math

import numpy as np

from . import planner


def is_stop(w):
    return w.get("mode") in ("gate", "tunnel", "align") or bool(w.get("land"))


def is_path_leg(meta):
    """Crossings (an opening to align on, an enclosed passage) are straight legs; everything else,
    including the approach to such a point, is flown along a navigator path."""
    return not (meta.get("align") or meta.get("tunnel"))


def stops_and_legs(plan, start_xy, start_yaw_rad):
    """-> (route [(forward, left)], legs [meta], nav_legs [{name, stop, goals, path}])."""
    fw = planner.flight_waypoints(plan["waypoints"])
    legs_full = planner.to_legs(plan["waypoints"])
    idx = [i for i in range(1, len(fw)) if is_stop(fw[i])]
    c, s = math.cos(start_yaw_rad), math.sin(start_yaw_rad)
    route, legs, nav_legs = [], [], []
    prev = 0
    for i in idx:
        dx, dy = fw[i]["xy"][0] - start_xy[0], fw[i]["xy"][1] - start_xy[1]
        route.append((round(dx * c + dy * s, 4), round(-dx * s + dy * c, 4)))
        meta = legs_full[i - 1]
        legs.append(meta)
        via = [list(map(float, fw[j]["xy"])) for j in range(prev + 1, i) if not str(fw[j]["name"]).startswith("via")]
        nav_legs.append(dict(name=fw[i]["name"], stop=[float(v) for v in fw[i]["xy"]],
                             goals=via + [[float(v) for v in fw[i]["xy"]]], path=is_path_leg(meta)))
        prev = i
    return route, legs, nav_legs


def course_to_ned(pts_nominal, fixture, declared_heading_rad):
    """Nominal-course-frame points -> the flight frame's local NED.

    nominal = fixture o odometry (the localizer's frame relation: fixture = (x, y, yaw) of the VIO origin
    in the course frame); the flight frame's ENU = R(declared heading) o odometry, NED = (E_y, E_x):
      * simulator: PX4 local, the vision bridge rotates VIO by heading_enu_rad = 90 deg, the fixture yaw
        is 90 deg as well, so ENU = nominal - spawn;
      * skybeta drone: the flight interface permutes the odometry's ENU directly (declared heading 0).
    """
    q = np.atleast_2d(np.asarray(pts_nominal, float)) - np.asarray(fixture[:2], float)
    a = float(declared_heading_rad) - float(fixture[2])
    c, s = math.cos(a), math.sin(a)
    e = np.stack([c * q[:, 0] - s * q[:, 1], s * q[:, 0] + c * q[:, 1]], axis=1)
    return np.stack([e[:, 1], e[:, 0]], axis=1)


def smooth_params_for(nav_cfg, refnav_params, need_m):
    """The navigator's SmoothParams from the course file's navigation block (+ refnav's yaw rate)."""
    from .smooth import SmoothParams
    sp = SmoothParams(need_m=need_m)
    for k, v in (nav_cfg or {}).items():
        if not k.startswith("_") and hasattr(sp, k):
            setattr(sp, k, type(getattr(sp, k))(v))
    if "yaw_rate_dps" not in (nav_cfg or {}) and "yaw_rate_dps" in refnav_params:
        sp.yaw_rate_dps = float(refnav_params["yaw_rate_dps"])
    sp.need_m = need_m
    return sp


def headings_to_frame(pts, yaw, to_frame, d_m=0.1):
    """CRAB headings (course frame, rad) at `pts` -> the same headings in the frame `to_frame` maps points
    into (PX4 NED: (n, e)), through the same transform as the points (drift correction, VIO frame), by
    mapping a short step along each heading. None when there are no headings."""
    if yaw is None:
        return None
    pts = np.asarray(pts, float)
    yaw = np.asarray(yaw, float)
    a = np.asarray(to_frame(pts), float)
    b = np.asarray(to_frame(pts + d_m * np.stack([np.cos(yaw), np.sin(yaw)], axis=1)), float)
    d = b - a
    return np.arctan2(d[:, 1], d[:, 0])
