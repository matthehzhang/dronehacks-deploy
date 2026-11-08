#!/usr/bin/env python3
"""Draw the outbound preprocessed path and the separate vertical-climb return.

Run: python tools/preview_return.py --out plans/return_review
Shows target geometry, not a flight simulation or clearance certification.
"""
import argparse
from pathlib import Path

import numpy as np

import preview_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=preview_path.ROOT / 'plans/return_review')
    args = parser.parse_args()
    data, output = preview_path.main(['--out', str(args.out)])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    start = np.asarray(data['start']['xy'])
    pad = np.asarray(data['nav_legs'][-1]['stop'])
    fig = plt.figure(figsize=(13, 7))
    top = fig.add_subplot(121)
    height = fig.add_subplot(122, projection='3d')
    previous = start
    z = .8
    for i, record in enumerate(data['preprocessed_paths']):
        xy = np.asarray(record['path']['pts'] if record['path'] else [previous, record['stop_xy']])
        top.plot(xy[:, 0], xy[:, 1], color='#2878c8', lw=2)
        height.plot(xy[:, 0], xy[:, 1], np.full(len(xy), z), color='#2878c8', lw=2)
        if i + 1 < len(data['legs']) and data['legs'][i + 1].get('align', {}).get('group') == 'window':
            new_z = data['legs'][i + 1]['align']['target_z_m']
            height.plot([xy[-1, 0]]*2, [xy[-1, 1]]*2, [z, new_z], color='#2878c8', lw=2)
            z = new_z
        previous = np.asarray(record['stop_xy'])
    height.plot([pad[0]]*2, [pad[1]]*2, [z, .1], color='#2878c8', lw=2)
    top.plot([pad[0], start[0]], [pad[1], start[1]], '--', color='#df7a23', lw=2.5)
    for ax in (top, height):
        ax.scatter(*pad, *([.1] if ax is height else []), color='#df7a23', s=55)
        ax.scatter(*start, *([.1] if ax is height else []), color='#16805d', s=65, marker='*')
    top.annotate('Pad: land + disarm\nThen rearm and climb vertically', pad, xytext=(12, -25),
                 textcoords='offset points', fontsize=9)
    top.annotate('Original launch\nReturn landing + disarm', start, xytext=(12, 10),
                 textcoords='offset points', fontsize=9)
    top.set_aspect('equal')
    top.set(xlabel='Course x / east (m)', ylabel='Course y / north (m)',
            title='Blue: outbound | orange dashed: return')
    top.grid(alpha=.2)
    height.plot([pad[0]]*2, [pad[1]]*2, [.1, 2.1], color='#df7a23', lw=3)
    height.plot([pad[0], start[0]], [pad[1], start[1]], [2.1, 2.1], color='#df7a23', lw=3)
    height.plot([start[0]]*2, [start[1]]*2, [2.1, .1], color='#df7a23', lw=3)
    height.text(pad[0], pad[1], 2.2, 'Hold XY until 2.1 m\nsettles for 1 second', fontsize=9)
    height.set(xlabel='Course x (m)', ylabel='Course y (m)', zlabel='Height above floor (m)',
               zlim=(0, 2.5), title='Return: up first, across second, down last')
    height.view_init(elev=24, azim=35)
    fig.suptitle('Outbound completion first; return starts only after pad disarm', fontweight='bold')
    fig.text(.5, .025, 'Target geometry only. Return height 2.1 m is operator selected; actual arm/Offboard '
             'acceptance and flight dynamics are not simulated.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .07, 1, .93])
    image = output / 'return_preview.png'
    fig.savefig(image, dpi=160)
    plt.close(fig)
    print(f'Return preview: {image.resolve()}')


if __name__ == '__main__':
    main()
