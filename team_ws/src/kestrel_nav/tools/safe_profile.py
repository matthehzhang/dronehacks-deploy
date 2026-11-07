"""Phase-time profile and gate-centring truth check for one offline SafeMission run."""
import collections, math, sys
import numpy as np
from kestrel_nav.livenav.offline_sim import OfflineFlight, SimConfig
from kestrel_nav.livenav.mission import MissionParams
from kestrel_nav.livenav.safe import SafeMission, REACH_M
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
m = SafeMission(MissionParams(landed_z_m=0.067, time_limit_s=175))
f = OfflineFlight(m, SimConfig(seed=seed, spawn=tuple(m.doc['start']['xy']), prop_radius_m=REACH_M, max_time_s=185))
orig = m._align_step
truth_at = []
def spy(now, obs, pose_c, cmd):
    out = orig(now, obs, pose_c, cmd)
    if m.align is None and m.aligned_for is not None and (not truth_at or truth_at[-1][0] != m.aligned_for):
        g = m.align_meta[m.goals[m.aligned_for].name]
        left = np.array([-g['dir'][1], g['dir'][0]])
        truth_at.append((m.aligned_for, round(float((np.array([f.x, f.y]) - g['centre']) @ left), 3),
                         round(float((np.asarray(pose_c[:2]) - g['centre']) @ left), 3), round(m.cross_offset, 3)))
    return out
m._align_step = spy
r = f.run()
seq = [(e[0], e[2]) for e in m.events if e[1] != 'ALIGN']
tot = collections.Counter()
for (t0, p), (t1, _) in zip(seq, seq[1:] + [(r.t_end, None)]):
    tot[p] += t1 - t0
print('result', r.result, 't', round(r.t_end, 1), 'goals', r.goals_done, '/', r.goals, 'min_clear', round(r.min_clearance_m, 3),
      'land_err', r.land_err_m)
print('phase seconds', {k: round(v, 1) for k, v in tot.items()})
print('segments', sum(1 for e in m.events if e[2] == 'TURN'))
print('aligned (goal k, TRUTH lateral of vehicle from opening centre, BELIEF lateral, cross_offset):', truth_at)
print('align events', [e[3] for e in m.events if e[1] == 'ALIGN'])
print('log', [(round(a, 1), b, c, d[:60]) for a, b, c, d in m.log])
turns = [e[3] for e in m.events if e[2] == 'TURN']
print('turn notes', turns)
