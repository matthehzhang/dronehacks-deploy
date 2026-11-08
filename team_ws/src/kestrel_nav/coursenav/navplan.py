"""VIO/course frame conversion and depth-horizon trajectory parameters."""
import math
import numpy as np
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
