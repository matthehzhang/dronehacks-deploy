#!/usr/bin/env python3
"""Offline evidence: run the production local ray integrator without ROS or flight."""
from pathlib import Path
import argparse
import json
import sys

parser=argparse.ArgumentParser()
parser.add_argument('--package',type=Path,default=Path(__file__).resolve().parents[4]/'team_ws/src/kestrel_nav')
parser.add_argument('--out',type=Path,default=Path(__file__).with_name('probe-results.json'))
args=parser.parse_args()
sys.path.insert(0,str(args.package.resolve()))
import numpy as np
from coursenav.localgrid import LocalGrid, LocalParams, free_points_from_grid
from coursenav.scan import Scan, local_scan_from_points, scan_from_points

cases=[('near_1m',1.,.07),('exact_horizon_3m',3.,.23),('far_5m',5.,.55),
       ('far_8m',8.,1.33),('far_uncertain',5.,10.),('zero',0.,.05),
       ('nan',float('nan'),.05),('infinity',float('inf'),.05)]
results=[]
for name,distance,sigma in cases:
    grid=LocalGrid(LocalParams(max_range_m=3.,decay_s=0.,min_cells=1))
    scan=Scan(np.array([0.]),np.array([distance]),np.array([sigma]),free_origin_xy=np.zeros(2))
    accepted=[grid.integrate(0.,(0.,0.,0.),scan) for _ in range(3)]
    free=free_points_from_grid(*grid.grid_fields(now=0.))
    occupied=grid.obstacle_points()
    def at(x):
        ij=tuple(grid.ij([x,0.])[0])
        return {'log_odds':float(grid.L[ij]),'seen':int(grid.seen[ij])}
    results.append({'case':name,'distance_m':distance if np.isfinite(distance) else str(distance),
                    'sigma_m':sigma,'accepted':accepted,'free_cells':len(free),
                    'occupied_cells':len(occupied),'max_free_x_m':None if not len(free) else float(free[:,0].max()),
                    'samples':{str(x):at(x) for x in [.5,1.,2.8,3.,3.1]},
                    'free_cells_after_expiry':len(free_points_from_grid(*grid.grid_fields(now=8.001)))})
near=np.column_stack([np.linspace(.6,1.5,10),np.zeros(10),np.zeros(10)])
far=np.column_stack([np.full(1000,8.),np.zeros(1000),np.zeros(1000)])
legacy=scan_from_points(near,.8,rmax=3.,n_bins=1)
combined=local_scan_from_points(np.vstack([near,far]),.8,.3,1.6,obstacle_range_m=3.,ray_range_m=12.,camera_origin_xy=np.zeros(2),n_bins=1)
quantile={'legacy_near_ranges':legacy.range.tolist(),'near_plus_1000_far_ranges':combined.range.tolist()}
args.out.write_text(json.dumps({'near_quantile_invariance':quantile,'package':str(args.package.resolve()),'local_horizon_m':3.,'integrations':3,'cases':results},indent=2,allow_nan=False)+'\n')
print(args.out)
