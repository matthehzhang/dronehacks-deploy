"""Load coordinate intent without querying or planning through origin geometry."""
import json
import math
from .course import resolve
from .planner import to_route, to_legs


def plan_intent(doc, start_xy, yaw_deg):
    intent = json.loads(resolve(doc,'waypoint_file').read_text())
    waypoints = [dict(name='start',xy=list(start_xy),mode='takeoff')]
    for item in intent['waypoints']:
        xy = item['xy']
        if len(xy) != 2 or not all(math.isfinite(float(v)) for v in xy):
            raise ValueError('waypoint coordinates must be finite XY')
        waypoints.append(dict(name=item['name'],xy=list(map(float,xy)),mode='cruise',
                              stop=True,land=bool(item.get('land',False))))
    if len(waypoints) < 2 or not waypoints[-1]['land']:
        raise ValueError('waypoint intent must end at a landing target')
    length = sum(math.dist(a['xy'],b['xy']) for a,b in zip(waypoints[:-1],waypoints[1:]))
    p = dict(waypoints=waypoints,length_m=length,min_clearance_m=0.,time_estimate_s=0.,
             leg_end_violations=[],predicted_false_blocks=0,gate_lines=[],
             geometry_source='depth_only',clearance_pending=True)
    return p,to_route(waypoints,start_xy,math.radians(yaw_deg)),to_legs(waypoints),doc
