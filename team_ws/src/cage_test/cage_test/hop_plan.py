"""Geometry for ``hop_test``: the legs of the flight and the moving setpoint.

No ROS in here, so it is tested on its own (``test/test_hop_plan.py``).

All points are ENU in the ``odom`` frame, the same frame the node reads from
``/visual_slam/tracking/odometry`` and publishes on ``/skybeta/setpoint/pose``.
"""
import math
from collections import namedtuple

#: One straight segment, flown at a fixed speed, then held for ``hold_s``.
Leg = namedtuple('Leg', ['name', 'start', 'end', 'hold_s'])


def yaw_from_xyzw(x, y, z, w):
    """ENU yaw (rad) of a ROS xyzw quaternion. 0 = body +x along odom +x."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def along_heading(position, origin, yaw_enu):
    """(forward, left, up) of ``position - origin``, along heading ``yaw_enu``."""
    dx, dy, dz = (float(p) - float(o) for p, o in zip(position, origin))
    c, s = math.cos(yaw_enu), math.sin(yaw_enu)
    return (dx * c + dy * s, -dx * s + dy * c, dz)


def build_legs(start, yaw_enu, climb_m, forward_m, low_m, hold_s):
    """up -> forward -> back -> down, from ``start`` (the vehicle on the ground).

    "Forward" is along the vehicle's heading at ``start`` (``yaw_enu``), not
    along odom +x: the two only agree if nobody turned the vehicle after the
    stack came up. ``low_m`` is the height above ``start`` the descent ends
    at; it must sit above the flight interface's ``min_altitude_m`` floor or
    the last point is clamped and never reached.
    """
    x0, y0, z0 = (float(v) for v in start)
    fx, fy = math.cos(yaw_enu), math.sin(yaw_enu)
    top = (x0, y0, z0 + climb_m)
    out = (x0 + forward_m * fx, y0 + forward_m * fy, z0 + climb_m)
    low = (x0, y0, z0 + low_m)
    return [
        Leg('up', (x0, y0, z0), top, hold_s),
        Leg('forward', top, out, hold_s),
        Leg('back', out, top, hold_s),
        Leg('down', top, low, hold_s),
    ]


def leg_length(leg):
    return math.dist(leg.start, leg.end)


def carrot(leg, progress_m):
    """The point ``progress_m`` along ``leg``, stopping at its end.

    Publishing this instead of ``leg.end`` keeps PX4's position error small:
    an Offboard position setpoint goes straight to the P controller with no
    trajectory smoothing, so a 1 m step is flown as a 1 m step.
    """
    length = leg_length(leg)
    if length <= 0.0:
        return tuple(leg.end)
    f = min(max(progress_m / length, 0.0), 1.0)
    return tuple(s + f * (e - s) for s, e in zip(leg.start, leg.end))
