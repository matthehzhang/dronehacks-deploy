"""Return geometry from the published map and this run's remembered launch. No ROS/truth."""
from dataclasses import dataclass
import math

import numpy as np

from .objmap import CourseMap, FLOOR_MARKING_PREFIXES


@dataclass(frozen=True)
class ReturnPlan:
    height_m: float
    course_height_m: float
    target_ne: tuple
    estimated_s: float


def route_spec(doc, start, landed_z):
    """Embedded by the normal course entry point; no return CLI flag."""
    from .course import resolve
    return dict(map_file=str(resolve(doc, 'map')), start_course=list(start),
                launch_mark=list(doc['start']['xy']), pad_xy=list(doc['sequence'][-1]['xy']),
                pad_radius_m=float(doc['sequence'][-1]['radius_m']), landed_z_m=float(landed_z),
                radius_m=float(doc['vehicle']['radius_m']), margin_m=float(doc['vehicle']['margin_m']),
                cruise_z_m=float(doc['return'].get('cruise_z_m', 2.1)),
                landing_radius_m=float(doc['return'].get('radius_m', .61)))


class ReturnGeometry:
    MAX_DRIFT_M = 0.5               # PX4-frame vs localizer disagreement on the pad beyond which the return refuses

    def __init__(self, spec, cmap=None):
        self.spec = spec
        self.map = cmap if cmap is not None else CourseMap.load(spec['map_file'])
        # Paint/floor is permitted at touchdown; all physical structures and netting remain.
        self.solid = self.map.occupied & ~self.map.group_mask(FLOOR_MARKING_PREFIXES)

    def course_xy(self, ne, launch):
        sx, sy, yaw_deg = self.spec['start_course']
        dn, de = np.asarray(ne) - np.asarray(launch[:2])
        h, y = launch[3], math.radians(yaw_deg)
        f, left = dn * math.cos(h) + de * math.sin(h), dn * math.sin(h) - de * math.cos(h)
        return np.array([sx + f * math.cos(y) - left * math.sin(y),
                         sy + f * math.sin(y) + left * math.cos(y)])

    def ne_xy(self, course, launch):
        """Inverse of course_xy: a course-frame point -> PX4 local NE through the remembered launch frame."""
        sx, sy, yaw_deg = self.spec['start_course']
        dx, dy = np.asarray(course, float) - [sx, sy]
        h, y = launch[3], math.radians(yaw_deg)
        f, left = dx * math.cos(y) + dy * math.sin(y), -dx * math.sin(y) + dy * math.cos(y)
        dn = f * math.cos(h) + left * math.sin(h)
        de = f * math.sin(h) - left * math.cos(h)
        return np.array([launch[0] + dn, launch[1] + de])

    def clear_segment(self, a, b, radius, z_margin):
        """Conservative box sweep, including the full voxel cells, inside map bounds.

        Physical body extends 0.18 m below and 0.10 m above origin. Floor is excluded
        only as the known takeoff/landing surface. Samples overlap by less than one voxel.
        """
        cm = self.map
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(2, int(math.ceil(np.linalg.norm(b-a)/(cm.res/2)))+1)
        for p in np.linspace(a, b, n):
            lo = p - [radius, radius, .18 + z_margin]
            hi = p + [radius, radius, .10 + z_margin]
            il = np.floor((lo-cm.origin)/cm.res).astype(int)
            ih = np.floor((hi-cm.origin)/cm.res).astype(int)
            # Below-floor volume belongs to the permitted touchdown surface only.
            il[2] = max(il[2], cm.z_index(0.0))
            if np.any(il < 0) or np.any(ih >= cm.shape) or np.any(ih < il):
                return False
            if self.solid[il[0]:ih[0]+1, il[1]:ih[1]+1, il[2]:ih[2]+1].any():
                return False
        return True

    def plan(self, launch, current, localized_xy, sigma, eph, epv, climb_rate, speed):
        s = self.spec
        start = np.asarray(s['start_course'][:2], float)
        remembered = self.course_xy(current[:2], launch)
        # PX4's local frame drifts with VIO (Gazebo/FlyX: 0.2-0.4 m over a run), so the REMEMBERED launch is no
        # longer where the launch is. Measure that drift with the localizer (the map) on the pad and correct the
        # destination by it: home is the PX4-frame point the localizer places on the launch, and every swept
        # volume is checked where the aircraft really is. Gazebo 2026-11-07: the old rule (refuse beyond
        # 0.15 m) refused the return of an otherwise ready run. Beyond MAX_DRIFT_M the frames disagree by more
        # than drift explains: refuse.
        drift = np.asarray(localized_xy, float) - remembered
        disagreement = float(np.linalg.norm(drift))
        if disagreement > self.MAX_DRIFT_M:
            raise ValueError('remembered launch frame disagrees with localizer')
        pad = np.asarray(localized_xy, float)
        # heading drift is not corrected by a translation: allow a quarter of the drift, plus 5 cm, over the transit
        uncertainty = max(2*sigma, eph) + .05 + .25 * disagreement
        if np.linalg.norm(np.asarray(localized_xy)-s['pad_xy']) + uncertainty + s['radius_m'] > s['pad_radius_m']:
            raise ValueError('landing is not confidently inside the destination pad')
        if np.linalg.norm(start-s['launch_mark']) + uncertainty + .15 > s['landing_radius_m']:
            raise ValueError('remembered launch cannot meet return landing radius')
        if abs(current[2]-launch[2]) > .12:
            raise ValueError('landed height changed since launch')
        pad_z = s['landed_z_m'] + launch[2]-current[2]
        radius = s['radius_m'] + s['margin_m'] + uncertainty
        zm = s['margin_m'] + epv
        # The configured altitude is a candidate, never permission to intersect a roof.
        z = float(s['cruise_z_m'])
        a, b = [*pad, pad_z], [*pad, z]
        c, d = [*start, z], [*start, s['landed_z_m']]
        for label, p, q in [('pad climb', a, b), ('overflight', b, c), ('home descent', c, d)]:
            if not self.clear_segment(p, q, radius, zm):
                raise ValueError(f'{label} volume is blocked or outside mapped enclosure')
        height = z-pad_z
        if height <= 0 or min(climb_rate, speed) <= 0:
            raise ValueError('invalid return height or speed')
        # Allow 40 s for qualification, priming/arming, turning, braking, observed-
        # corridor checks and reserve, in addition to translation/descent time.
        # This is an initial budget model, not a flight-time guarantee.
        estimated = height/climb_rate + float(np.linalg.norm(pad-start))/speed + (z-s['landed_z_m'])/.25 + 40
        target = self.ne_xy(start - drift, launch)          # the PX4-frame point the localizer calls the launch
        return ReturnPlan(height, z, (float(target[0]), float(target[1])), estimated)
