#!/usr/bin/env python3
"""Plan the course offline and print what course_mission would fly. No ROS, no simulator.

    ros2 run kestrel_nav plan_course                          # from the course-file start prior
    ros2 run kestrel_nav plan_course --start -2.51 1.11 90    # from a measured start pose (course frame)
    python3 -m kestrel_nav.plan_course --json plan.json       # also dump the plan

Prints the chosen chicane and why, every waypoint in the course frame, the same waypoints as
(forward, left) metres from the start pose (what course_mission rotates into the odom frame), and the
replay of the old controller's blocking rule on the map (predicted false stops).
"""
import argparse
import json
import math

from kestrel_nav.coursenav import planner
from kestrel_nav.coursenav.course import load_course, plan_grids, resolve
from kestrel_nav.coursenav.objmap import CourseMap


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--start', nargs=3, type=float, metavar=('X', 'Y', 'YAW_DEG'))
    ap.add_argument('--chicane', choices=['auto', 'window', 'gate_b'], default='auto')
    ap.add_argument('--course', default=None, help='course json (default: the packaged course_v3.json)')
    ap.add_argument('--json', default=None, help='write the plan here')
    a = ap.parse_args(argv)

    doc = load_course(a.course) if a.course else load_course()
    cmap = CourseMap.load(resolve(doc, 'map'))
    obst, blocked, clearance = plan_grids(cmap, doc)
    if a.start:
        sx, sy, syaw = a.start
    else:
        sx, sy = doc['start'].get('spawn_xy', doc['start']['xy'])
        syaw = doc['start']['yaw_deg']
    if a.chicane == 'auto':
        res = planner.plan_options(doc, obst, blocked, clearance, start_xy=(sx, sy), cmap=cmap)
        p = res['plan']
        print(f"chicane: {res['chicane']}  ({res['reason']})")
        for k, v in res['scores'].items():
            print(f'   score {k}: {v}')
    else:
        p = planner.plan(doc, obst, blocked, clearance, start_xy=(sx, sy), chicane=a.chicane, cmap=cmap)
    wps = planner.flight_waypoints(p['waypoints'])
    route = planner.to_route(p['waypoints'], (sx, sy), math.radians(syaw))
    legs = planner.to_legs(p['waypoints'])
    print(f'start (course frame): ({sx:.3f}, {sy:.3f}) yaw {syaw:.1f} deg; cruise {doc["cruise_z_m"]} m above the floor')
    print(f"{len(route)} legs, {p['length_m']:.2f} m, min clearance {p['min_clearance_m']:.2f} m "
          f"(need {p['required_clearance_m']:.2f}), ~{p['time_estimate_s']:.0f} s")
    print(f"{'waypoint':14s} {'course x':>9s} {'course y':>9s} {'forward':>8s} {'left':>7s}  leg")
    for w, (f, l), leg in zip(wps[1:], route, legs):
        flags = leg['mode'] + (' hold' if leg.get('hold') else '') + (' tunnel' if leg.get('tunnel') else '')
        print(f"{w['name']:14s} {w['xy'][0]:9.2f} {w['xy'][1]:9.2f} {f:8.2f} {l:7.2f}  {flags}")
    pfb = p.get('predicted_false_blocks', [])
    print(f'predicted false blocks (old refnav rule replayed on the map): {pfb or "none"}')
    if a.json:
        out = {k: v for k, v in p.items() if k != 'plans'}
        out.update(start={'xy': [sx, sy], 'yaw_deg': syaw}, route_forward_left=route, legs=legs)
        with open(a.json, 'w') as f:
            json.dump(out, f, indent=1, default=str)
        print(f'wrote {a.json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
