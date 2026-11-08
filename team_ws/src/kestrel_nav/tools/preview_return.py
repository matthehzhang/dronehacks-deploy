#!/usr/bin/env python3
"""Preview the current outbound route, genuine touch-and-go sequence and circular return.

Run: python tools/preview_return.py --out plans/touch_and_go_review
The diagram shows targets only, not aircraft contact or takeoff-ramp behavior.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import preview_path
from fly_course import skybeta_return_spec
from coursenav.course import load_course
from refnav.skybeta_return import return_arc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=preview_path.ROOT / 'plans/touch_and_go_review')
    args = parser.parse_args()
    data, output = preview_path.main(['--out', str(args.out)])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    spec = skybeta_return_spec(load_course())
    start = np.asarray(data['start']['xy'])
    pad = np.asarray(data['nav_legs'][-1]['stop'])
    # NE handedness is the reverse of course XY: swap the components for drawing.
    ne, speeds = return_arc(pad[::-1], start[::-1], spec['arc_offset_m'], spec['arc_side'])
    arc = ne[:, ::-1]
    fig = plt.figure(figsize=(14, 7))
    top, height = fig.add_subplot(121), fig.add_subplot(122, projection='3d')
    previous, z = start, .8
    for i, record in enumerate(data['preprocessed_paths']):
        xy = np.asarray(record['path']['pts'] if record['path'] else [previous, record['stop_xy']])
        top.plot(xy[:, 0], xy[:, 1], color='#2878c8', lw=2)
        height.plot(xy[:, 0], xy[:, 1], np.full(len(xy), z), color='#2878c8', lw=2)
        if i+1 < len(data['legs']) and data['legs'][i+1].get('align', {}).get('group') == 'window':
            new_z = data['legs'][i+1]['align']['target_z_m']
            height.plot([xy[-1, 0]]*2, [xy[-1, 1]]*2, [z, new_z], color='#2878c8', lw=2)
            z = new_z
        previous = np.asarray(record['stop_xy'])
    top.plot([pad[0], start[0]], [pad[1], start[1]], ':', color='#9ca6af', lw=1.5, label='Direct chord (not flown)')
    top.plot(arc[:, 0], arc[:, 1], color='#df7a23', lw=3, label='Circular return')
    midpoint = (pad+start)/2
    apex = arc[len(arc)//2]
    top.plot([midpoint[0], apex[0]], [midpoint[1], apex[1]], '--', color='#9b59b6')
    top.annotate('2 ft = 0.6096 m', (midpoint+apex)/2, xytext=(10, 8), textcoords='offset points', color='#75429a')
    top.scatter(*pad, s=60, color='#df7a23', zorder=5)
    top.scatter(*start, s=120, color='#16805d', marker='*', zorder=5)
    top.annotate('Pad: confirmed contact\nStay armed; climb vertically', pad, xytext=(12, -30), textcoords='offset points', fontsize=9)
    top.annotate('Original launch\nLand + fully disarm', start, xytext=(10, 10), textcoords='offset points', fontsize=9)
    top.set_aspect('equal')
    top.set(xlabel='Course x / east (m)', ylabel='Course y / north (m)', title='Blue: outbound | orange: curved return')
    top.grid(alpha=.2)
    top.legend(loc='upper right', fontsize=8)
    target = spec['height_above_floor_m']
    height.plot([pad[0]]*3, [pad[1]]*3, [z, .1, target], color='#df7a23', lw=3)
    height.plot(arc[:, 0], arc[:, 1], np.full(len(arc), target), color='#df7a23', lw=3)
    height.plot([start[0]]*2, [start[1]]*2, [target, .1], color='#df7a23', lw=3)
    height.text(pad[0], pad[1], target+.1, f'XY held until {target:.2f} m\nsettles for 1 second', fontsize=9)
    height.set(xlabel='Course x (m)', ylabel='Course y (m)', zlabel='Body height above floor (m)', zlim=(0, target+.5),
               title='Touch -> up -> curved transit -> down -> disarm')
    height.view_init(elev=25, azim=35)
    fig.suptitle('Touch-and-go: no pad disarm, no rearm, no straight-line return', fontweight='bold')
    fig.text(.5, .025, 'Target geometry only. Touch must be reported by PX4. Aircraft auto-disarm/takeoff-ramp '
             'interaction remains untested; no FC parameter changes.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .07, 1, .93])
    path = output/'touch_and_go_return.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    (output/'return_arc.json').write_text(json.dumps(dict(spec=spec, points_course_xy=arc.tolist(),
                                                        speed_profile_mps=speeds.tolist()), indent=2), encoding='utf-8')
    print(f'Touch-and-go review: {path.resolve()}')


if __name__ == '__main__':
    main()
