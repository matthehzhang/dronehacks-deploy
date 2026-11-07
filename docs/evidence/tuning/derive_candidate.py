"""Offline published-map search; never commands ROS or certifies physical geometry."""
import sys,json,math
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[3] / 'team_ws/src/kestrel_nav'
sys.path.insert(0,str(ROOT))
from kestrel_nav.coursenav.course import load_course,resolve
from kestrel_nav.coursenav.objmap import CourseMap
from kestrel_nav.coursenav import planner
from kestrel_nav.livenav.navgrid import NavGrid
from kestrel_nav.livenav.tuning import validate_config
c=load_course(); m=CourseMap.load(resolve(c,'map')); g=NavGrid(m,c)
centres=g.obst.xy(np.argwhere(g.combined.occ)); tree=cKDTree(centres)
r=g.obst.res; hd=r/math.sqrt(2)
def cert(q):return tree.query(np.asarray(q))[0]-hd
T=c['derived']['tunnel']; center=np.array(T['axis_xy']); entry=np.array(T['entry']); exit=np.array(T['exit'])
u=(entry-exit)/np.linalg.norm(entry-exit); mouth_n=center+u*T['length_m']/2; mouth_s=center-u*T['length_m']/2
starts=[mouth_n+u*d for d in np.arange(c['vehicle']['radius_m']+c['vehicle']['margin_m'],2,r/2)]
ends=[mouth_s-u*d for d in np.arange(c['vehicle']['radius_m']+c['vehicle']['margin_m'],2,r/2)]
starts=[x for x in starts if cert(x)>=.65]; ends=[x for x in ends if cert(x)>=.65]
print('mouths',mouth_n.tolist(),mouth_s.tolist(),'certified endpoints',len(starts),len(ends))
results=[]
for a in starts:
 try:
  p=planner.plan(c,g.obst,g.prior_blocked,g.prior_clearance,start_xy=c['start']['spawn_xy'],goals=[dict(name='diagnostic_lane_start',xy=a.tolist())],cmap=m)
 except planner.PlanError:continue
 for b in ends:
  length=float(np.linalg.norm(b-a))
  if length<2 or length>8 or not planner.segment_free(g.prior_blocked,a,b):continue
  n=int(math.ceil(length/(r/2)))+1
  lower=float(np.min(cert(np.linspace(a,b,n))))-length/(n-1)/2
  if lower<.5:continue
  cfg={'map_only':True,'start_course':[*c['start']['spawn_xy'],c['start']['yaw_deg']],'speeds_mps':[.1],'time_limit_s':170.,'candidates':[{'name':'published_tube_axis','kind':'tube','start_xy':a.tolist(),'end_xy':b.tolist(),'height_m':c['cruise_z_m'],'half_width_m':lower,'start_turn_radius_m':float(cert(a)),'end_turn_radius_m':float(cert(b)),'start_open':True,'end_open':True}]}
  try:cfg=validate_config(cfg)
  except ValueError:continue
  total=p['length_m']/.15+5+cfg['estimated_lane_s']
  results.append(dict(config=cfg,total_s=total,transit_length_m=p['length_m'],lane_length_m=length,waypoints=p['waypoints'],transit_min_clearance_m=p['min_clearance_m']))
results.sort(key=lambda x:x['total_s'])
print('pairs',len(results))
for x in results[:3]:print(json.dumps({k:v for k,v in x.items() if k!='waypoints'},indent=2))
if results:
 best=results[0];Path(__file__).with_name('course_map_only.json').write_text(json.dumps(best['config'],indent=2))
 Path(__file__).with_name('course_geometry.json').write_text(json.dumps(dict(derivation='Published map only, not physical survey. Tube axis from semantic geometry; full-planned-clearance-outside endpoints searched at half map resolution. Continuous disk clearance lower bound is distance to occupied-cell centres minus voxel half diagonal; sampled lane lower bound subtracts half sampling interval by Lipschitz distance. Radius threshold .65 comes from current diagnostic contract; .1 speed/.15 transit existing diagnostic bounds.',mouth_n=mouth_n.tolist(),mouth_s=mouth_s.tolist(),candidate=best),indent=2))
