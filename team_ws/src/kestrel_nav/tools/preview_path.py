#!/usr/bin/env python3
"""Precompute the flight paths and save a PNG/JSON preview, without ROS or a GPU.

Run: python tools/preview_path.py --out plans/path_preview
Requires numpy, scipy and matplotlib. Uses the current course start and profile.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fly_course
from coursenav.course import load_course, plan_grids, resolve
from coursenav.navplan import smooth_params_for, stops_and_legs
from coursenav.navigator import Navigator
from coursenav.objmap import CourseMap


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT / 'plans' / 'path_preview')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    doc = load_course()
    cmap = CourseMap.load(resolve(doc, 'map'))
    grids = plan_grids(cmap, doc)
    start = np.asarray(doc['start']['xy'], float)
    yaw = math.radians(doc['start']['yaw_deg'])
    plan, _, _, _ = fly_course.plan_now(start, math.degrees(yaw), prepared=(doc, cmap, grids))
    route, legs, nav_legs = stops_and_legs(plan, start, yaw)
    params = {k: v for k, v in doc['refnav_params'].items() if not k.startswith('_')}
    params.update({k: v for k, v in doc['navigation']['refnav'].items() if not k.startswith('_')})
    params['track_mode'] = 'path'
    params['correction_max_m'] = 0.0
    need = doc['vehicle']['radius_m'] + doc['vehicle']['margin_m']
    navigator = Navigator(*grids, need, smooth_params_for(doc['navigation'], params, need),
                          gate_lines=plan.get('gate_lines'))
    records, paths = [], []
    prev = start.copy()
    # Same per-leg preplanning as coursenav.nav_node.NavNode._preplan.
    for j, leg in enumerate(nav_legs):
        record = dict(leg=j, name=leg['name'], start_xy=prev.tolist(), stop_xy=leg['stop'],
                      goals_xy=leg['goals'], path_leg=leg['path'], path=None)
        if leg['path']:
            path = navigator.plan(prev, [np.asarray(g, float) for g in leg['goals']], t=0.0)
            record.update(path=asdict(path), length_m=path.length, estimated_duration_s=path.duration())
            paths.append((j, path))
        records.append(record)
        prev = np.asarray(leg['stop'], float)
    result = dict(frame='course ENU', start=dict(xy=start.tolist(), yaw_deg=math.degrees(yaw)),
                  route=route, legs=legs, nav_legs=nav_legs, params=params, preprocessed_paths=records,
                  note='Geometric preflight preview; not a PX4/sensor simulation or flown trajectory.')
    (args.out / 'preprocessed_paths.json').write_text(
        json.dumps(result, indent=2, default=serializable), encoding='utf-8')
    draw(doc, grids, start, yaw, nav_legs, paths, args.out / 'path_preview.png')
    print(f'Precomputed {len(paths)} path legs; {len(nav_legs)} mission legs.')
    for j, path in paths:
        print(f"  {nav_legs[j]['name']}: {path.length:.2f} m; {len(path.stops)} planned corner stops")
    print(f'Image: {(args.out / "path_preview.png").resolve()}')
    print(f'Data:  {(args.out / "preprocessed_paths.json").resolve()}')


def draw(doc, grids, start, yaw, nav_legs, paths, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    obst, blocked, _ = grids
    background = np.zeros(obst.occ.shape, dtype=np.uint8)
    background[blocked.occ] = 1
    background[obst.occ] = 2
    extent = [obst.origin[0], obst.origin[0] + obst.occ.shape[0] * obst.res,
              obst.origin[1], obst.origin[1] + obst.occ.shape[1] * obst.res]
    colors = ['#2878c8', '#9b59b6', '#159b80', '#dc8330', '#d44d6b']
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={'width_ratios': [1.3, 1]})
    for ax in axes:
        ax.imshow(background.T, origin='lower', extent=extent,
                  cmap=ListedColormap(['#fafbfc', '#e1e5eb', '#4c5666']),
                  vmin=0, vmax=2, interpolation='nearest')
        prev = start
        path_by_leg = dict(paths)
        for j, leg in enumerate(nav_legs):
            col = colors[j % len(colors)]
            if j in path_by_leg:
                p = path_by_leg[j]
                xy = p.pts
                ax.plot(xy[:, 0], xy[:, 1], color=col, lw=2.2)
                if p.stops:
                    corners = xy[p.stops]
                    ax.scatter(corners[:, 0], corners[:, 1], marker='s', s=46,
                               color='#eea923', edgecolors='white', zorder=5)
                    if j == 0:
                        for k, i in enumerate(p.stops):
                            if ax is axes[1]:
                                ax.annotate(f'Stop {k+1}: {p.s[i]:.2f} m along path', xy[i],
                                            xytext=(12, -18), textcoords='offset points', fontsize=9,
                                            bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
            else:
                xy = np.asarray([prev, leg['stop']])
                ax.plot(xy[:, 0], xy[:, 1], '--', color=col, lw=2.2)
            ax.scatter(*leg['stop'], s=45, color=col, edgecolors='white', zorder=5)
            if ax is axes[0]:
                ax.annotate(leg['name'].replace('_', ' '), leg['stop'], xytext=(7, 5),
                            textcoords='offset points', fontsize=8,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=.8))
            prev = leg['stop']
        ax.scatter(*start, marker='*', s=190, color='#16805d', edgecolors='white', zorder=6)
        ax.arrow(*start, .22 * math.cos(yaw), .22 * math.sin(yaw), width=.009,
                 head_width=.07, color='#16805d', length_includes_head=True, zorder=6)
        ax.set_aspect('equal')
        ax.set_xlabel('Course x / east (m)')
        ax.set_ylabel('Course y / north (m)')
        ax.grid(alpha=.18)
    axes[0].set_xlim(extent[:2])
    axes[0].set_ylim(extent[2:])
    axes[0].set_title('Full precomputed route', fontsize=13)
    axes[1].set_xlim(start[0] - .42, start[0] + 1.12)
    axes[1].set_ylim(start[1] - .2, start[1] + 1.5)
    axes[1].set_title('Launch and first few feet', fontsize=13)
    axes[1].annotate('New start (+8 inches)', start, xytext=(12, -19), textcoords='offset points', fontsize=9)
    handles = [Line2D([], [], color=colors[0], lw=2, label='Navigator path'),
               Line2D([], [], color='#9b59b6', ls='--', label='Straight crossing'),
               Line2D([], [], color='#eea923', marker='s', ls='', label='Planned stop and turn'),
               Line2D([], [], color='#4c5666', lw=7, label='Obstacle at cruise-height band'),
               Line2D([], [], color='#e1e5eb', lw=7, label='Inflation / keep-out')]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9, frameon=False)
    fig.suptitle(f"groupbad218 | start ({start[0]:.3f}, {start[1]:.4f}) | body height {doc['cruise_z_m']:.2f} m",
                 fontsize=14, fontweight='bold')
    fig.text(.5, .105, 'Planned geometry only; map is approximate. This is not a simulated or recorded flight.',
             ha='center', fontsize=9, color='#5b6370')
    fig.tight_layout(rect=[0, .13, 1, .93])
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
