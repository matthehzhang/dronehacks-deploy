"""Load coordinate intent without querying or planning through origin geometry."""
import json
import math
from .course import resolve
def to_route(wps,start,yaw):
    c,s=math.cos(yaw),math.sin(yaw)
    return [(round((w['xy'][0]-start[0])*c+(w['xy'][1]-start[1])*s,4),
             round(-(w['xy'][0]-start[0])*s+(w['xy'][1]-start[1])*c,4)) for w in wps[1:]]

def to_legs(wps):
    return [dict(name=w['name'],mode='cruise',hold=True) for w in wps[1:]]


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
