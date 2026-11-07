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
        pad = self.course_xy(current[:2], launch)
        # Retain the remembered local-frame destination. The current map estimate must
        # agree with that frame before trusting the saved coordinate and map sweep.
        disagreement = float(np.linalg.norm(pad-np.asarray(localized_xy)))
        if disagreement > .15:
            raise ValueError('remembered launch frame disagrees with localizer')
        uncertainty = max(2*sigma, eph) + disagreement
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
        return ReturnPlan(height, z, tuple(launch[:2]), estimated)
