#!/usr/bin/env python3
"""Preprocess the actual route and draw its window climb and retained height, without ROS.

Run: python tools/preview_window_height.py --out plans/window_height_review
The side view shows targets along path distance, not simulated flight dynamics.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import preview_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=preview_path.ROOT / 'plans/window_height_review')
    args = parser.parse_args()
    data, output = preview_path.main(['--out', str(args.out)])
    from coursenav.course import load_course, plan_grids, resolve, Grid2D
    from coursenav.objmap import CourseMap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    doc = load_course()
    cmap = CourseMap.load(resolve(doc, 'map'))
    obst, blocked, _ = plan_grids(cmap, doc)
    fig, (top, side) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={'width_ratios': [1.1, 1.4]})
    background = blocked.occ.astype(int)
    background[obst.occ] = 2
    extent = [obst.origin[0], obst.origin[0] + obst.occ.shape[0] * obst.res,
              obst.origin[1], obst.origin[1] + obst.occ.shape[1] * obst.res]
    top.imshow(background.T, origin='lower', extent=extent, interpolation='nearest',
               cmap=ListedColormap(['#fafbfc', '#e1e5eb', '#4c5666']), vmin=0, vmax=2)
    xy_start = np.asarray(data['start']['xy'])
    top.scatter(*xy_start, marker='*', s=140, color='#2878c8', zorder=5)
    top.annotate('Start', xy_start, xytext=(7, 8), textcoords='offset points')
    distance, height = 0., doc['cruise_z_m']
    distances, heights = [0, 0], [.10, height]
    rows = []
    previous = xy_start
    raised = False
    climb_xy = None
    for index, record in enumerate(data['preprocessed_paths']):
        points = np.asarray(record['path']['pts'] if record['path'] else [previous, record['stop_xy']])
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        distance += length
        name = record['name']
        color = '#df7a23' if raised else '#2878c8'
        top.plot(points[:, 0], points[:, 1], color=color, lw=2.3)
        top.scatter(*record['stop_xy'], s=45, color=color, zorder=5)
        label = name.replace('_', ' ')
        top.annotate(label, record['stop_xy'], xytext=(7, 5), textcoords='offset points', fontsize=8,
                     bbox=dict(facecolor='white', edgecolor='none', alpha=.8))
        distances.append(distance)
        heights.append(height)
        side.axvline(distance, color='#b0b8c1', lw=.7, alpha=.5)
        side.text(distance, .04, label, rotation=65, ha='left', va='bottom', fontsize=8)
        row = dict(name=name, distance_m=distance, arrival_z_m=height)
        if index + 1 < len(data['legs']):
            alignment = data['legs'][index + 1].get('align', {})
            if alignment.get('group') == 'window' and 'target_z_m' in alignment:
                raised = True
                height = alignment['target_z_m']
                climb_xy = record['stop_xy']
                distances.append(distance)
                heights.append(height)
                trim = data['params']['align_window_max_dz_m']
                side.fill_between([distance, distance + alignment['dist_m']], [1., 1.], [1.8, 1.8],
                                  color='#d5eedf', alpha=.65, label='User opening: 1.00-1.80 m')
                side.annotate('STOP; XY target fixed\nClimb to 1.40 m\nSettle 1 s, then align',
                              (distance, 1.4), xytext=(-85, 50), textcoords='offset points', fontsize=10,
                              arrowprops=dict(arrowstyle='->', color='#df7a23'),
                              bbox=dict(facecolor='white', edgecolor='#df7a23', boxstyle='round,pad=.5'))
                row.update(departure_z_m=height, xy_hold=True, height_tolerance_m=.05,
                           settle_s=1., camera_trim_limit_m=trim)
                side.fill_between([distance, distance + .8], height - trim, height + trim,
                                  color='#df7a23', alpha=.18, label='Camera trim: +/-0.10 m')
        rows.append(row)
        previous = np.asarray(record['stop_xy'])
    distances.append(distance)
    heights.append(0.)
    side.plot(distances, heights, color='#df7a23', lw=2.5, label='Nominal commanded body height')
    side.annotate('Retain crossing height\nuntil final landing descent', (distance - .5, height),
                  xytext=(-5, -45), textcoords='offset points', ha='right', fontsize=10)
    top.scatter(*climb_xy, s=180, facecolors='none', edgecolors='#df7a23', lw=2.5, zorder=6)
    top.set(title='Precomputed horizontal path (unchanged)', xlabel='Course x / east (m)', ylabel='Course y / north (m)')
    top.set_aspect('equal')
    side.set(title='Height targets along the same route', xlabel='Distance along preprocessed path (m)',
             ylabel='Body origin above floor (m)', ylim=(0, 2.15))
    side.legend(loc='upper left', fontsize=8)
    for ax in (top, side):
        ax.grid(alpha=.15)
    # Column check uses the old map only; the measured opening is explicitly separate.
    column = Grid2D(cmap.band(.55, 1.75), cmap.origin[:2], cmap.res)
    clearance = float(column.distance()[tuple(column.ij(climb_xy))])
    report = dict(height_profile=rows, climb_xy=climb_xy, climb_column_map_clearance_m=clearance,
                  mapped_opening_z_m=doc['chicane_options']['window']['z_m'],
                  user_opening_z_m=doc['chicane_options']['window']['opening_z_m'],
                  note='Target diagram, not a flight simulation. Old map has a lower window; raised crossing clearance is not verified.')
    (output / 'height_profile.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    fig.suptitle('Window height review | stop, climb, settle, align, cross', fontsize=15, fontweight='bold')
    fig.text(.5, .025, 'Targets only; no flight simulation. Background uses the old 0.80 m planning band. '
             'Measured window bounds override alignment, not map geometry.', ha='center', fontsize=9, color='#5b6370')
    fig.tight_layout(rect=[0, .06, 1, .94])
    path = output / 'window_height_review.png'
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f'Window height review: {path.resolve()}')
    print(f'Old-map climb-column horizontal clearance: {clearance:.2f} m')


if __name__ == '__main__':
    main()
