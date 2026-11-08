"""Horizontal swept footprint about the body origin, independent of course/yaw."""
import math


def vehicle_geometry(vehicle):
    """Resolve a declared body envelope and optional rotor layout.

    A rotor disk centred at (x,y) reaches hypot(x,y) + diameter/2 from
    the origin. The largest such radius bounds every yaw angle. This is a
    horizontal, level-flight envelope, not a surveyed 3-D tilted-airframe model.
    Keep any larger declared envelope (e.g. guards or payload) intact.
    """
    result = dict(vehicle)
    radius, margin = float(result['radius_m']), float(result['margin_m'])
    if not all(math.isfinite(x) and x >= 0 for x in (radius, margin)) or radius == 0:
        raise ValueError('vehicle radius must be positive and margin nonnegative, both finite')
    if 'rotor_centres_xy_m' in result or 'propeller_diameter_m' in result:
        diameter = float(result['propeller_diameter_m'])
        centres = result['rotor_centres_xy_m']
        if not math.isfinite(diameter) or diameter <= 0 or not centres:
            raise ValueError('positive finite propeller diameter and rotor centres required')
        reaches = []
        for centre in centres:
            if len(centre) != 2 or not all(math.isfinite(float(x)) for x in centre):
                raise ValueError('rotor centres must be finite body-frame XY pairs')
            reaches.append(math.hypot(*centre) + diameter / 2)
        radius = max(radius, max(reaches))
        # Half-width across the direction of travel (the follower faces along its path).
        result['lateral_m'] = max(abs(float(c[1])) for c in centres) + diameter / 2
    result.update(radius_m=radius, margin_m=margin)
    return result


def planning_clearance(vehicle, turn_floor_m=0.05):
    """Centre clearance the 2-D planner requires.

    The follower always faces along its path, so the airframe's extent across the
    path is its lateral half-width; the margin is kept around that. The full
    rotor-tip circle (any yaw) plus turn_floor_m stays a hard lower bound, so an
    in-place turn on the route never reaches a surface."""
    lateral = float(vehicle.get('lateral_m', vehicle['radius_m']))
    return max(lateral + float(vehicle['margin_m']), float(vehicle['radius_m']) + turn_floor_m)
