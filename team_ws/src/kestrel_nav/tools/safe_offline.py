"""Closed-loop offline check of the safe policy (SafeMission) vs the carrot follower (LiveMission).

    python3 tools/safe_offline.py --seeds 0 1 2 --slip 0 1.0 --start-err 0.10

Scored against the 7-inch prop reach (0.294 m). Offline kinematics, synthetic depth and VIO drift;
NOT a flight qualification."""
import argparse, json, math, sys, time
from kestrel_nav.livenav.offline_sim import OfflineFlight, SimConfig
from kestrel_nav.livenav.mission import LiveMission, MissionParams
from kestrel_nav.livenav.safe import SafeMission, SafeParams, REACH_M


def run(policy, seed, slip, err, shift=0.0, group=''):
    mp = MissionParams(landed_z_m=0.067, time_limit_s=175.0)
    m = SafeMission(mp) if policy == 'safe' else LiveMission(mp)
    ex, ey = (err * math.cos(seed), err * math.sin(seed)) if err else (0.0, 0.0)
    cfg = SimConfig(seed=seed, spawn=tuple(m.doc['start']['xy']), prop_radius_m=REACH_M, yaw_slip_s=slip, max_time_s=185.0,
                    start_err_xy=(ex, ey), start_err_yaw_deg=2.0 * err / 0.1 * (1 if seed % 2 else -1) if err else 0.0,
                    shift_group=group, shift_m=(shift, 0.0) if group else (0.0, 0.0))
    t0 = time.time()
    r = OfflineFlight(m, cfg).run()
    ev = [e for e in getattr(m, 'events', []) if e[1] == 'ALIGN']
    return dict(policy=policy, seed=seed, slip=slip, err=err, result=r.result, t=round(r.t_end, 1),
                goals=f'{r.goals_done}/{r.goals}', min_clear=round(r.min_clearance_m, 3),
                land_err=None if math.isnan(r.land_err_m) else round(r.land_err_m, 3),
                align=[e[3] for e in ev][:4], wall=round(time.time() - t0, 1),
                states=[(round(a, 1), b, c[:50]) for a, b, c in r.states][-4:])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', nargs='+', default=['safe'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0])
    ap.add_argument('--slip', nargs='+', type=float, default=[0.0])
    ap.add_argument('--start-err', nargs='+', type=float, default=[0.0])
    a = ap.parse_args()
    for pol in a.policy:
        for s in a.seeds:
            for sl in a.slip:
                for e in a.start_err:
                    print(json.dumps(run(pol, s, sl, e)), flush=True)
