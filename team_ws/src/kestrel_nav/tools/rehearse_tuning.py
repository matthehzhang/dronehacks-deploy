#!/usr/bin/env python3
"""Offline-only rehearsal; imports no ROS and cannot command a vehicle.

--fixture exercises separate yaw and translation fits using synthetic records.
--config exercises the actual pure TuningController in ONE declared lane using a
delayed first-order velocity plant. It begins at the lane entrance, so it DOES
NOT validate maze transit. The geometry is a rectangle plus circular open end
zones; it is NOT the real course or PX4 simulator. Sensors here are ideal, with
continuous pose observations, and do not qualify visual-only dropout behavior.

The supplied response and yaw lag are assumptions for software rehearsal, not
measured Sky Beta values. All results remain synthetic and never write gains.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_tuning import analyze, response_basis  # noqa: E402


def record(kind, now, phase, trial, *, candidate='synthetic_lane', axis='translation',
           rotation_trial=None, **fields):
    return dict(schema=1, kind=kind, t_ros=float(now), t_mono=float(now),
                source_stamp=float(now), frame_id='odom', phase=phase,
                candidate=candidate, trial=trial, trial_id=(f'{candidate}:yaw:{rotation_trial}' if axis == 'yaw' else f'{candidate}:{trial}'),
                axis=axis, rotation_trial=rotation_trial,
                split='train' if (rotation_trial if axis == 'yaw' else trial) % 2 == 0 else 'validation', **fields)


def synthetic_recording(*, tau_s=.3, delay_s=.15, gain=1., bias_mps=0.,
                        validation_tau_s=None, dt=.05, position_noise_m=0., seed=0):
    """Known-response fixture. Train/validation repetitions are labelled upfront.

    Two speeds test linear-response identification; these are test stimuli only,
    never proposed flight settings. The independent held-out variant can change
    tau to verify that the fitter rejects a response which fails to generalize.
    """
    rng = np.random.default_rng(seed)
    rows = [dict(schema=1, kind='header', synthetic=True, config={
        'purpose': 'log/fitter fixture, not controller/flight validation',
        'speeds_mps': [.1, .2], 'yaw_rates_rps': [], 'candidates': [{'name': 'synthetic_lane'}],
        'tau_s': tau_s, 'delay_s': delay_s, 'gain': gain, 'bias_mps': bias_mps})]
    offset, position = 10., 0.
    for trial, speed in enumerate([.1, .1, .2, .2]):
        t = np.arange(0., 7. + dt / 2, dt)
        u = np.where((t >= 2.) & (t < 4.), speed, 0.)
        tau = validation_tau_s if trial % 2 and validation_tau_s is not None else tau_s
        # A prehistory zero command allows exact delayed integration from t=0.
        ct = np.r_[-max(2., delay_s + dt), t]
        cu = np.r_[0., u]
        vb, pb = response_basis(t, ct, cu, tau, delay_s, 0.)
        coefficients = np.array([1., gain, bias_mps])
        p = pb @ coefficients
        rows.append(record('command', offset + ct[0], 'SETTLE', trial,
                           velocity_enu=[0., 0., 0.], yaw_rate=0., target_speed=speed,
                           clamped=[], offboard_active=True))
        for i, now in enumerate(t):
            phase = 'SETTLE' if now < 2. else ('PULSE' if now < 4. else 'BRAKE')
            rows.append(record('command', offset + now, phase, trial,
                               velocity_enu=[float(u[i]), 0., 0.], yaw_rate=0., target_speed=speed,
                               clamped=[], offboard_active=True, tracking_ok=True, imu_ok=True))
            rows.append(record('odom', offset + now, phase, trial,
                               position=[position + float(p[i]) + rng.normal(0, position_noise_m), 0., .8],
                               quaternion_xyzw=[0., 0., 0., 1.], covariance=[0.] * 36))
        position += float(p[-1])
        offset += 10.
    rows.append(dict(schema=1, kind='summary', dropped_count=0, synthetic=True,
                     terminal_phase='DONE', completed=[{'candidate': 'synthetic_lane', 'trial': i} for i in range(4)]))
    return rows


def synthetic_yaw_recording(*, yaw_tau_s=.3, yaw_delay_s=.1, yaw_gain=1.,
                            yaw_bias_rad_s=0., validation_yaw_tau_s=None, dt=.05,
                            **translation_options):
    """Combined independent yaw and translation fixture, with heading wrap.

    All four stationary yaw maneuvers precede translation. Opposite directions
    are declared held-out before model fitting; simulator truth is not logged.
    """
    translation = synthetic_recording(dt=dt, **translation_options)
    header = translation[0]
    header['config']['yaw_rates_rps'] = [.15, .25]
    rows = [header]
    offset, heading = 10., math.pi-.1
    completed = []
    for trial, rate in enumerate([.15, -.15, .25, -.25]):
        t = np.arange(0., 7.+dt/2, dt)
        u = np.where((t >= 2.) & (t < 4.), rate, 0.)
        tau = validation_yaw_tau_s if trial % 2 and validation_yaw_tau_s is not None else yaw_tau_s
        ct, cu = np.r_[-max(2., yaw_delay_s+dt), t], np.r_[0., u]
        vb, pb = response_basis(t, ct, cu, tau, yaw_delay_s, 0.)
        coef = np.array([1., yaw_gain, yaw_bias_rad_s])
        angle, measured_rate = pb @ coef, vb @ coef
        rows.append(record('command', offset+ct[0], 'YAW_SETTLE', 0,
                           axis='yaw', rotation_trial=trial, velocity_enu=[0., 0., 0.],
                           yaw_rate=0., clamped=[], offboard_active=True))
        for i, now in enumerate(t):
            phase = 'YAW_SETTLE' if now < 2. else ('YAW_PULSE' if now < 4. else 'YAW_BRAKE')
            fields = dict(axis='yaw', rotation_trial=trial)
            rows.append(record('command', offset+now, phase, 0, **fields,
                               velocity_enu=[0., 0., 0.], yaw_rate=float(u[i]),
                               clamped=[], offboard_active=True, tracking_ok=True, imu_ok=True))
            yaw = heading+float(angle[i])
            rows.append(record('odom', offset+now, phase, 0, **fields,
                               position=[0., 0., .8],
                               quaternion_xyzw=[0., 0., math.sin(yaw/2), math.cos(yaw/2)]))
        completed.append(dict(candidate='synthetic_lane', trial=trial, rate=rate,
                              peak_rate=float(np.max(np.abs(measured_rate))),
                              stopping_angle=float(angle[-1]-angle[np.searchsorted(t, 4.)])))
        heading += float(angle[-1])
        offset += 10.
    for row in translation[1:-1]:
        for key in ('source_stamp', 't_ros', 't_mono'):
            row[key] += 50.
        rows.append(row)
    rows.append(dict(translation[-1], completed_yaw=completed))
    return rows


def _forward_distance(position, yaw, lane):
    """Ray exit distance from union of lane rectangle and open-end disks.

    This ideal sensor model alone reads synthetic truth. No truth is passed to
    the fitter. Geometric interval intersection avoids sampling away thin walls.
    """
    a, b = np.asarray(lane['start_xy']), np.asarray(lane['end_xy'])
    axis = (b - a) / np.linalg.norm(b - a)
    normal = np.array([-axis[1], axis[0]])
    direction = np.array([math.cos(yaw), math.sin(yaw)])
    local = np.array([(position - a) @ axis, (position - a) @ normal])
    ray = np.array([direction @ axis, direction @ normal])
    bounds = [(0., np.linalg.norm(b - a)), (-lane['half_width_m'], lane['half_width_m'])]
    lo, hi = -math.inf, math.inf
    for value, step, (lower, upper) in zip(local, ray, bounds):
        if abs(step) < 1e-12:
            if not lower <= value <= upper:
                hi = -math.inf
        else:
            edges = sorted(((lower - value) / step, (upper - value) / step))
            lo, hi = max(lo, edges[0]), min(hi, edges[1])
    intervals = [(max(0., lo), hi)] if hi >= max(0., lo) else []
    for center, radius in [(a, lane['start_turn_radius_m']), (b, lane['end_turn_radius_m'])]:
        delta = position - center
        dot = float(delta @ direction)
        disc = dot * dot - float(delta @ delta) + radius ** 2
        if disc >= 0:
            low, high = -dot - math.sqrt(disc), -dot + math.sqrt(disc)
            if high >= 0:
                intervals.append((max(0., low), high))
    end = 0.
    for low, high in sorted(intervals):
        if low > end + 1e-9:
            break
        end = max(end, high)
    return end


def rehearse(config, *, tau_s=.45, delay_s=.15, yaw_tau_s=.3, yaw_delay_s=.1, dt=.05):
    """Drive the production pure core; preserve aborts and timeouts as failures."""
    from kestrel_nav.livenav.tuning import TuningController, TuningObservation
    if (tau_s <= 0 or delay_s < 0 or yaw_tau_s <= 0 or yaw_delay_s < 0 or dt <= 0
            or not all(map(math.isfinite, (tau_s, delay_s, yaw_tau_s, yaw_delay_s, dt)))):
        raise ValueError('finite positive tau/dt and nonnegative delay required')
    if len(config['candidates']) != 1:
        raise ValueError('rehearsal validates one lane only; inter-candidate navigation is not modeled')
    core = TuningController(config)
    lane = core.lane
    a, b = np.asarray(lane['start_xy']), np.asarray(lane['end_xy'])
    position = np.r_[a, lane['height_m']].astype(float)
    velocity = np.zeros(3)
    yaw = math.atan2(*(b - a)[::-1])
    yaw_rate = 0.
    history_t, history_u, history_yaw = [-1000.], [np.zeros(3)], [0.]
    rows = [dict(schema=1, kind='header', synthetic=True, config=config,
                 assumed_model={'tau_s': tau_s, 'delay_s': delay_s, 'yaw_tau_s': yaw_tau_s, 'yaw_delay_s': yaw_delay_s},
                 exclusions=['maze transit', 'ROS', 'PX4', 'VIO dropout', 'sensor noise'])]
    now, end = 10., 10. + config.get('time_limit_s', 160.) + 2 * dt
    while now <= end:
        front = _forward_distance(position[:2], yaw, lane)
        obs = TuningObservation(now=now, position=tuple(position), yaw=yaw,
                                velocity=tuple(velocity), front_clearance=front, coverage=1.,
                                transit_complete=True, yaw_rate=yaw_rate)
        command = core.step(obs)
        state = core.status()
        phase, trial, candidate = state['phase'], state['trial'], state['candidate']
        identification = dict(axis=state.get('axis', 'translation'), rotation_trial=state.get('rotation_trial'))
        u = np.array([command.vx, command.vy, command.vz])
        history_t.append(now)
        history_u.append(u)
        history_yaw.append(command.yaw_rate)
        rows.append(record('command', now, phase, trial, candidate=candidate, **identification,
                           velocity_enu=u.tolist(), yaw_rate=command.yaw_rate,
                           target_speed=state['target_speed'], clamped=[], offboard_active=True,
                           tracking_ok=True, imu_ok=True, note=command.note))
        rows.append(record('odom', now, phase, trial, candidate=candidate, **identification,
                           position=position.tolist(),
                           quaternion_xyzw=[0., 0., math.sin(yaw/2), math.cos(yaw/2)], covariance=[0.] * 36))
        if phase in ('DONE', 'FAULT'):
            break
        # Split integration at delayed command edges, retaining zero-order hold.
        edges = [now] + [t + delay_s for t in history_t if now < t + delay_s < now + dt] + [now + dt]
        for start, stop in zip(edges[:-1], edges[1:]):
            target = history_u[int(np.searchsorted(history_t, start - delay_s + 1e-10, side='right') - 1)]
            step = stop - start
            decay = math.exp(-step/tau_s)
            position += target*step + (velocity-target)*tau_s*(1-decay)
            velocity = target + (velocity-target)*decay
        # Independent exact delayed yaw plant; its measured rate reaches only
        # the controller sensor input. The fitter derives rate from pose heading.
        edges = [now] + [t+yaw_delay_s for t in history_t if now < t+yaw_delay_s < now+dt] + [now+dt]
        for start, stop in zip(edges[:-1], edges[1:]):
            target = history_yaw[int(np.searchsorted(history_t, start-yaw_delay_s+1e-10, side='right')-1)]
            step = stop-start
            decay = math.exp(-step/yaw_tau_s)
            yaw += target*step + (yaw_rate-target)*yaw_tau_s*(1-decay)
            yaw_rate = target+(yaw_rate-target)*decay
        now = round(now + dt, 9)
    rows.append(dict(schema=1, kind='summary', dropped_count=0, synthetic=True,
                     terminal_phase=core.state, reason=core.reason, elapsed_s=now-10.,
                     completed=core.completed, completed_yaw=core.completed_yaw))
    return rows


def check_course(config):
    """Construct the actual map-aware program, without its ROS adapter.

    This checks declared geometry against the prior and exercises its budget
    refusal. Distance divided by a speed ceiling is not an upper time bound:
    turns, qualification, depth holds and tracking can make transit take longer.
    """
    from kestrel_nav.livenav.tuning import TuningProgram
    from kestrel_nav.livenav.mission import MissionParams
    from kestrel_nav.livenav.follower import FollowParams
    program = TuningProgram(config, MissionParams(), FollowParams())
    return {'passed': True, 'transit_length_m': program.plan['length_m'],
            'transit_min_clearance_m': program.plan['min_clearance_m'],
            'estimated_total_s': program.estimated_total_s,
            'meaning': 'Prior-map and constructor checks only; no maze transit was simulated.',
            'time_limitation': 'Distance/cruise arithmetic is not a guaranteed upper time bound.'}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--fixture', action='store_true')
    mode.add_argument('--config', type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--report', required=True, type=Path)
    ap.add_argument('--tau-s', type=float, default=.45)
    ap.add_argument('--delay-s', type=float, default=.15)
    ap.add_argument('--yaw-tau-s', type=float, default=.3)
    ap.add_argument('--yaw-delay-s', type=float, default=.1)
    ap.add_argument('--check-course', action='store_true',
                    help='also construct the real map-aware TuningProgram; requires --config')
    args = ap.parse_args(argv)
    if args.check_course and args.config is None:
        ap.error('--check-course requires --config, not the synthetic fixture')
    config = json.loads(args.config.read_text()) if args.config else None
    course_check = check_course(config) if args.check_course else None
    rows = (synthetic_yaw_recording(tau_s=args.tau_s, delay_s=args.delay_s,
                                    yaw_tau_s=args.yaw_tau_s, yaw_delay_s=args.yaw_delay_s) if args.fixture
            else rehearse(config, tau_s=args.tau_s, delay_s=args.delay_s,
                          yaw_tau_s=args.yaw_tau_s, yaw_delay_s=args.yaw_delay_s))
    args.out.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows))
    report = analyze(rows)
    report['synthetic_only'] = True
    report['controller_summary'] = rows[-1]
    if course_check is not None:
        report['course_check'] = course_check
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f"Synthetic rehearsal: {args.out}; response analysis: {args.report}")
    return 0 if report['accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
