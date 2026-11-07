#!/usr/bin/env python3
"""Offline identification of ENU translation and world-heading yaw response.

This does NOT tune PX4 or write flight parameters. It fits a delayed first-order
response to observed velocity, then replays whole, separately labelled validation
trials without feeding their observations back into the model. Position comes
from odometry, not simulator truth. Therefore an accepted fit describes the
command-to-estimate response; it cannot separate estimator delay from dynamics.

JSONL schema 1: command records contain velocity_enu and source_stamp (publish time);
odom records contain position, source_stamp and frame_id; both carry candidate,
trial, phase and split ('train' or 'validation'). A summary with dropped=0 is
required. Commands and positions must share the adapter's fixed ENU 'odom' frame. Only PULSE,
BRAKE and post-pulse SETTLE observations are fitted. Transit/turn/return are never
training examples. Yaw uses YAW_PULSE/YAW_BRAKE/YAW_SETTLE with axis='yaw'
and rotation_trial; timestamped quaternion_xyzw or yaw_enu supplies unwrapped
heading. World yaw rate is derived from heading, never body IMU Z. Configured
yaw trials precede translation; signed opposite-direction trials are held out.
A fault or clamp invalidates a trial, not just its bad samples.

Run from the source checkout: python3 tools/fit_tuning.py recording.jsonl --out fit.json
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


FIT_PHASES = {'PULSE', 'BRAKE', 'SETTLE'}
YAW_PHASES = {'YAW_PULSE', 'YAW_BRAKE', 'YAW_SETTLE'}
EPS = np.finfo(float).eps


@dataclass
class Trial:
    candidate: str
    trial: str
    split: str
    time: np.ndarray
    velocity: np.ndarray
    displacement: np.ndarray
    command_time: np.ndarray
    command: np.ndarray
    pulse_duration: float


def finite(value, name):
    a = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError(f'{name} must be finite')
    return a


def response_basis(t, command_time, command, tau, delay, initial_velocity):
    """Exact integration of v'=(gain*u(t-delay)+bias-v)/tau, ZOH input.

    Columns are the initial-condition contribution, unit gain, and unit bias.
    Both velocity and displacement are propagated across EVERY command boundary;
    observation timestamps need not be equally spaced. No observed state after
    the first sample enters this replay.
    """
    t, ct, u = map(lambda x: np.asarray(x, float), (t, command_time, command))
    if tau <= 0 or delay < 0 or ct[0] + delay > t[0] + 1e-9:
        raise ValueError('invalid response parameters or missing command prehistory')
    v = np.array([initial_velocity, 0., 0.])
    p = np.zeros(3)
    vv, pp = [v.copy()], [p.copy()]
    boundaries = ct + delay
    k = int(np.searchsorted(boundaries, t[0], side='right') - 1)
    now = t[0]
    for end in t[1:]:
        while now < end:
            edge = boundaries[k + 1] if k + 1 < len(ct) else math.inf
            stop = min(end, edge)
            dt = stop - now
            if dt > 0:
                target = np.array([0., u[k], 1.])
                decay = math.exp(-dt / tau)
                p += target * dt + (v - target) * tau * (-math.expm1(-dt / tau))
                v = target + (v - target) * decay
            now = stop
            if edge <= end:
                k += 1
        vv.append(v.copy())
        pp.append(p.copy())
    return np.asarray(vv), np.asarray(pp)


def predict(trial, tau_s, delay_s, gain=1., bias_mps=0., bias_rad_s=None):
    vb, pb = response_basis(trial.time, trial.command_time, trial.command,
                            tau_s, delay_s, trial.velocity[0])
    coef = np.array([1., gain, bias_mps if bias_rad_s is None else bias_rad_s])
    return vb @ coef, pb @ coef


def error_metrics(trial, velocity, displacement, *, yaw=False):
    ev = velocity - trial.velocity
    ep = displacement - trial.displacement
    return {('rate_rmse_rad_s' if yaw else 'velocity_rmse_mps'): float(np.sqrt(np.mean(ev ** 2))),
            ('angle_rmse_rad' if yaw else 'position_rmse_m'): float(np.sqrt(np.mean(ep ** 2))),
            ('angle_max_error_rad' if yaw else 'position_max_error_m'): float(np.max(np.abs(ep))),
            ('angle_end_error_rad' if yaw else 'position_end_error_m'): float(ep[-1])}


def _instantaneous(trial):
    """No-lag baseline, integrated with the actual command event times."""
    ct, u, t = trial.command_time, trial.command, trial.time
    velocity = u[np.searchsorted(ct, t, side='right') - 1]
    displacement = [0.]
    for a, b in zip(t[:-1], t[1:]):
        edges = np.r_[a, ct[(ct > a) & (ct < b)], b]
        values = u[np.searchsorted(ct, edges[:-1], side='right') - 1]
        displacement.append(displacement[-1] + float(np.dot(np.diff(edges), values)))
    return velocity, np.asarray(displacement)


def fit_candidate(trials, *, nominal_tau_s=.45, max_delay_s=None, max_tau_s=None, yaw=False):
    """Fit on train trials ONLY. Yaw uses the same scalar response equations
    with rad/s and radians and a separately declared .3 s nominal assumption.
    The nominal .45 s is the existing offline
    velocity model's explicit guess (livenav/offline_sim.py), not a measurement.
    Search limits are identification limits, never flight settings: absent CLI
    overrides, delay is limited by half the shortest pulse and recorded history;
    tau by half the shortest recorded response window. A boundary fit is rejected.
    """
    train = [x for x in trials if x.split == 'train']
    val = [x for x in trials if x.split == 'validation']
    reasons = []
    if not train or not val:
        return {'accepted': False, 'reasons': ['need separate train and validation trials']}
    if nominal_tau_s <= 0 or not math.isfinite(nominal_tau_s):
        raise ValueError('nominal_tau_s must be finite and positive')
    dt = float(np.median(np.concatenate([np.diff(x.time) for x in train])))
    # Subsample response cannot be resolved by pose-derived velocity.
    tau_lo = dt / 2
    tau_hi = (min(x.time[-1] - x.time[0] for x in train) / 2
              if max_tau_s is None else float(max_tau_s))
    delay_hi = (min(x.pulse_duration for x in train) / 2
                if max_delay_s is None else float(max_delay_s))
    history = min(x.time[0] - x.command_time[0] for x in trials)
    delay_hi = min(delay_hi, history)
    if not (math.isfinite(tau_hi) and math.isfinite(delay_hi)
            and tau_hi > tau_lo and delay_hi > 0):
        return {'accepted': False, 'reasons': ['insufficient duration or command prehistory']}

    def solve(params, details=False):
        tau, delay = params
        bases = [response_basis(x.time, x.command_time, x.command,
                                tau, delay, x.velocity[0])[0] for x in train]
        design = np.concatenate([b[:, 1:] for b in bases])
        target = np.concatenate([x.velocity - b[:, 0] for x, b in zip(train, bases)])
        coef, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        mse = float(np.mean((design @ coef - target) ** 2))
        if rank < 2 or coef[0] <= 0:
            mse = math.inf
        return (mse, coef, rank) if details else mse

    # Coarse coverage avoids a single local minimum across delayed step edges.
    grid = [(solve((tau, delay)), tau, delay)
            for tau in np.geomspace(tau_lo, tau_hi, 9)
            for delay in np.linspace(0, delay_hi, 9)]
    best = min(grid)
    if not math.isfinite(best[0]):
        return {'accepted': False, 'reasons': ['unidentifiable or nonpositive command response']}
    result = minimize(solve, best[1:], method='Nelder-Mead',
                      bounds=((tau_lo, tau_hi), (0., delay_hi)),
                      options={'xatol': dt / 1000, 'fatol': 1e-12, 'maxiter': 300})
    tau, delay = map(float, result.x)
    mse, coef, rank = solve((tau, delay), details=True)
    model = {'tau_s': tau, 'delay_s': delay, 'gain': float(coef[0]), ('bias_rad_s' if yaw else 'bias_mps'): float(coef[1])}
    if not result.success or rank < 2 or not math.isfinite(mse):
        reasons.append('fit did not converge to an identifiable response')
    # A numerical boundary tolerance tied to timestamp resolution; not a safety margin.
    if tau - tau_lo <= dt / 100 or tau_hi - tau <= dt / 100 or delay_hi - delay <= dt / 100:
        reasons.append('response hits identification search boundary; collect more data or review bounds')
    evaluations = []
    for x in trials:
        fitted = error_metrics(x, *predict(x, **model), yaw=yaw)
        nominal = error_metrics(x, *predict(x, nominal_tau_s, 0.), yaw=yaw)
        instant = error_metrics(x, *_instantaneous(x), yaw=yaw)
        evaluations.append({'trial': x.trial, 'split': x.split, 'fitted': fitted,
                            'nominal': nominal, 'instantaneous': instant})
        if x.split == 'validation':
            # Every held-out trial must improve both velocity and displacement,
            # relative to BOTH predeclared baselines. No aggregate hides regression.
            for label, baseline in [('nominal', nominal), ('instantaneous', instant)]:
                for metric in (('rate_rmse_rad_s', 'angle_rmse_rad') if yaw else
                               ('velocity_rmse_mps', 'position_rmse_m')):
                    if fitted[metric] >= baseline[metric]:
                        reasons.append(f'validation {x.trial}: {metric} does not improve {label}')
    return {'accepted': not reasons, 'reasons': reasons, 'model': model,
            'nominal_model': {'tau_s': nominal_tau_s, 'delay_s': 0., 'gain': 1., ('bias_rad_s' if yaw else 'bias_mps'): 0.},
            'search': {'tau_bounds_s': [tau_lo, tau_hi], 'delay_bounds_s': [0., delay_hi]},
            ('training_rate_rmse_rad_s' if yaw else 'training_velocity_rmse_mps'): math.sqrt(mse), 'trials': evaluations}


def heading_enu(row):
    """World ENU heading from pose orientation, never body IMU angular Z."""
    heading = None
    if 'quaternion_xyzw' in row:
        q = finite(row['quaternion_xyzw'], 'quaternion')
        if q.shape != (4,) or not .9 <= np.linalg.norm(q) <= 1.1:
            raise ValueError('invalid quaternion shape or normalization')
        x, y, z, w = q / np.linalg.norm(q)
        forward = np.array([1-2*(y*y+z*z), 2*(x*y+w*z)])
        if np.linalg.norm(forward) < .1:
            raise ValueError('heading is ill-conditioned near vertical attitude')
        heading = math.atan2(forward[1], forward[0])
    if 'yaw_enu' in row:
        explicit = finite(row['yaw_enu'], 'yaw_enu')
        if explicit.ndim != 0:
            raise ValueError('yaw_enu must be a scalar')
        explicit = float(explicit)
        if heading is not None and abs(math.remainder(explicit-heading, 2*math.pi)) > 1e-5:
            raise ValueError('yaw_enu and quaternion heading disagree')
        heading = explicit
    if heading is None:
        raise ValueError('yaw identification requires quaternion_xyzw or yaw_enu')
    return heading


def prepare_trials(records, *, yaw=False):
    """Strict schema adapter. Invalid trials are reported, never silently repaired."""
    if not isinstance(records, list) or not records or not all(isinstance(r, dict) for r in records):
        return [], ['recording must contain JSON objects'], []
    phases = YAW_PHASES if yaw else FIT_PHASES
    pulse_phase, brake_phase = ('YAW_PULSE', 'YAW_BRAKE') if yaw else ('PULSE', 'BRAKE')
    identity = 'rotation_trial' if yaw else 'trial'
    summaries = [r for r in records if r.get('kind') == 'summary']
    global_errors = []
    if not summaries or summaries[-1].get('dropped_count', summaries[-1].get('dropped')) != 0:
        global_errors.append('missing complete recording summary or dropped records')
    if summaries and summaries[-1].get('logging_error'):
        global_errors.append('recording reports a logging failure')
    headers = [r for r in records if r.get('kind') == 'header' and isinstance(r.get('config'), dict)]
    if not headers:
        global_errors.append('missing recording configuration header')
    summary = summaries[-1] if summaries else {}
    final_status = summary.get('result', {})
    if not isinstance(final_status, dict):
        final_status = {}
    if summary.get('terminal_phase', final_status.get('phase')) != 'DONE':
        global_errors.append('diagnostic did not finish DONE; partial runs are not accepted')
    if any(r.get('phase') == 'FAULT' or r.get('terminal_phase') == 'FAULT' for r in records):
        global_errors.append('diagnostic ended or entered a fault; do not accept partial success')
    if any(r.get('schema') != 1 for r in records):
        global_errors.append('unsupported or missing schema version')
    commands = [r for r in records if r.get('kind') == 'command']
    odom = [r for r in records if r.get('kind') == 'odom']
    if not commands or not odom:
        return [], global_errors + ['missing command or odometry stream'], []
    try:
        ct = finite([r['source_stamp'] for r in commands], 'command publish times')
        cu = finite([r['velocity_enu'] for r in commands], 'commands')
        ot = finite([r['source_stamp'] for r in odom], 'odometry times')
        op = finite([r['position'] for r in odom], 'positions')
        if yaw:
            yaw_commands = finite([r['yaw_rate'] for r in commands], 'yaw commands')
            if yaw_commands.shape != ct.shape:
                raise ValueError('yaw commands must be scalar')
        if cu.shape != (len(ct), 3) or op.shape != (len(ot), 3):
            raise ValueError('commands and positions must have three coordinates')
        if np.any(np.diff(ct) <= 0) or np.any(np.diff(ot) <= 0):
            raise ValueError('source times must advance without duplicates/resets')
        frames = {r.get('frame_id') for r in commands + odom}
        if frames != {'odom'}:
            raise ValueError("schema 1 requires command and odometry in the fixed ENU 'odom' frame")
    except (KeyError, TypeError, ValueError) as exc:
        return [], global_errors + [str(exc)], []
    keys = sorted({(str(r.get('candidate')), str(r.get(identity))) for r in commands
                   if r.get('phase') == pulse_phase})
    rates = []
    try:
        config = headers[-1]['config']
        rates = config.get('yaw_rates_rps', [.15, .25]) if yaw else config['speeds_mps']
        if not isinstance(rates, list) or any(not math.isfinite(float(x)) or float(x) <= 0 for x in rates):
            raise ValueError('configured rates must be finite positive values')
        lanes = config['candidates'][:1] if yaw else config['candidates']
        expected = {(str(lane['name']), str(i)) for lane in lanes
                    for i in range(2 * len(rates))}
        completed_key = 'completed_yaw' if yaw else 'completed'
        completed = summary.get(completed_key, final_status.get(completed_key, []))
        finished = [(str(r['candidate']), str(r['trial'])) for r in completed]
        if (not expected and not yaw) or set(keys) != expected or set(finished) != expected or len(finished) != len(expected):
            global_errors.append('configured trial schedule was not fully recorded and completed')
    except (KeyError, TypeError, IndexError, ValueError):
        global_errors.append('missing or invalid configured/completed trial schedule')
    if yaw:
        yaw_pulses = [r['source_stamp'] for r in commands if r.get('phase') == 'YAW_PULSE']
        linear_pulses = [r['source_stamp'] for r in commands if r.get('phase') == 'PULSE']
        if yaw_pulses and linear_pulses and max(yaw_pulses) >= min(linear_pulses):
            global_errors.append('yaw identification must finish before translation begins')
    trials, rejected = [], []
    for candidate, trial_id in keys:
        rows = [r for r in records if str(r.get('candidate')) == candidate
                and str(r.get(identity)) == trial_id]
        try:
            if candidate == 'None' or trial_id == 'None':
                raise ValueError('missing candidate/trial identity')
            splits = {r.get('split') for r in rows if r.get('kind') == 'command'
                      and r.get('phase') in phases}
            if len(splits) != 1 or next(iter(splits)) not in ('train', 'validation'):
                raise ValueError('explicit consistent train/validation split required')
            pulse = [r for r in rows if r.get('kind') == 'command' and r.get('phase') == pulse_phase]
            brake = [r for r in rows if r.get('kind') == 'command' and r.get('phase') == brake_phase]
            if not pulse or not brake:
                raise ValueError('need a pulse and recorded braking response')
            start, brake_start = pulse[0]['source_stamp'], brake[0]['source_stamp']
            if brake_start <= start:
                raise ValueError('braking must follow excitation')
            if yaw:
                expected_rate = float(rates[int(trial_id)//2]) * (1 if int(trial_id)%2 == 0 else -1)
                expected_split = 'train' if int(trial_id)%2 == 0 else 'validation'
                if next(iter(splits)) != expected_split:
                    raise ValueError('yaw split differs from configured signed schedule')
                if any(r.get('axis') != 'yaw' for r in rows if r.get('phase') in phases):
                    raise ValueError('yaw response rows must explicitly declare axis yaw')
                excitation = np.array([r['yaw_rate'] for r in pulse])
                nonzero = excitation[np.abs(excitation) > EPS]
                if not len(nonzero) or not np.allclose(nonzero, expected_rate, atol=1e-6, rtol=0):
                    raise ValueError('yaw excitation differs from configured signed rate')
                if any(np.linalg.norm(r['velocity_enu'][:2]) > 1e-6
                       or abs(r['velocity_enu'][2]) > .1+1e-6 for r in rows
                       if r.get('kind') == 'command' and r.get('phase') in phases):
                    raise ValueError('yaw identification requires zero horizontal command and bounded altitude hold')
            else:
                axis = np.mean([r['velocity_enu'][:2] for r in pulse], axis=0)
                norm = np.linalg.norm(axis)
                if norm <= EPS:
                    raise ValueError('unexcited command axis')
                axis /= norm
            # The return trip has its own BRAKE/SETTLE under the same trial ID.
            # End the response at the first non-fit command after the pulse;
            # otherwise return braking would contaminate outbound identification.
            end_rows = [r for r in rows if r.get('kind') == 'command'
                        and r['source_stamp'] > start and r.get('phase') not in phases]
            end = min((r['source_stamp'] for r in end_rows), default=math.inf)
            if any(r['source_stamp'] >= end for r in pulse):
                raise ValueError('trial identity reused for a later pulse after return/transit')
            quality_rows = [r for r in rows if start <= r.get('t_ros', -math.inf) <= end]
            # Preflight legitimately has no flight authority; inspect the whole
            # excitation/braking interval, not the earlier waiting/transit phase.
            if any(r.get('phase') == 'FAULT' or r.get('clamped') or r.get('tracking_ok') is False
                   or r.get('imu_ok') is False or r.get('offboard_active') is False
                   or r.get('health') is False or r.get('fault') or r.get('accepted') is False
                   for r in quality_rows):
                raise ValueError('trial contains clamp, fault, unhealthy sensor or authority loss')
            usable = [r for r in rows if r.get('kind') == 'odom' and r.get('phase') in phases
                      and start <= r['source_stamp'] < end]
            t = np.asarray([r['source_stamp'] for r in usable], float)
            p = (np.unwrap([heading_enu(r) for r in usable]) if yaw else
                 np.asarray([r['position'][:2] for r in usable], float) @ axis)
            # Four model parameters + the two derivative endpoints need more
            # than six raw points to leave residual degrees of freedom.
            if len(t) < 8 or t[-1] <= brake_start:
                raise ValueError('insufficient pulse/braking samples')
            # The centered secant uses raw pose timestamps, never arrival times.
            # Its effective timestamp is the interval midpoint. No IMU propagation
            # is fabricated; gaps remain visible and are rejected below.
            gaps = np.diff(t)
            if np.max(gaps) > 2 * np.median(gaps):
                raise ValueError('odometry gap exceeds two observed sample intervals')
            velocity = np.diff(p) / gaps
            tm = (t[1:] + t[:-1]) / 2
            pm = (p[1:] + p[:-1]) / 2
            u = yaw_commands if yaw else cu[:, :2] @ axis
            in_trial = (ct >= start) & (ct <= t[-1])
            command_gaps = np.diff(ct[in_trial])
            if len(command_gaps) < 2 or np.max(command_gaps) > 2 * np.median(command_gaps):
                # No unlimited ZOH through missing publishes: the physical
                # interface can expire a velocity command and request a hold.
                raise ValueError('command gap exceeds two observed publish intervals')
            if np.ptp(u[in_trial]) <= EPS:
                raise ValueError('constant command cannot identify response')
            trials.append(Trial(candidate, trial_id, next(iter(splits)), tm, velocity,
                                pm - pm[0], ct, u, brake_start - start))
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            rejected.append({'candidate': candidate, 'trial': trial_id, 'reason': str(exc)})
    return trials, global_errors, rejected


def analyze(records, *, nominal_yaw_tau_s=.3, **fit_options):
    candidates, yaw_candidates, errors, rejected = {}, {}, [], []
    for yaw, output in [(False, candidates), (True, yaw_candidates)]:
        trials, axis_errors, axis_rejected = prepare_trials(records, yaw=yaw)
        label = 'yaw' if yaw else 'translation'
        errors.extend(f'{label}: {e}' for e in axis_errors)
        rejected.extend(dict(r, axis=label) for r in axis_rejected)
        for candidate in sorted({x.candidate for x in trials}):
            options = dict(fit_options)
            if yaw:
                options['nominal_tau_s'] = nominal_yaw_tau_s
            result = fit_candidate([x for x in trials if x.candidate == candidate], yaw=yaw, **options)
            if any(r['candidate'] == candidate for r in axis_rejected):
                result['accepted'] = False
                result['reasons'].append('candidate has rejected trials; no selective deletion')
            if axis_errors:
                result['accepted'] = False
                result['reasons'].extend(axis_errors)
            output[candidate] = result
    return {'schema': 1, 'accepted': bool(candidates) and not errors and not rejected
            and all(x['accepted'] for x in [*candidates.values(), *yaw_candidates.values()]), 'errors': errors,
            'rejected_trials': rejected, 'candidates': candidates, 'yaw_candidates': yaw_candidates,
            'meaning': 'Accepted means held-out response prediction improved; NOT controller or flight approval.',
            'limitations': ['Effective command-to-odometry response includes estimator delay/error.',
                            'Horizontal translation and ENU heading response are identified separately.',
                            'Pose and unwrapped-heading secants use source-time interval centers; noise remains.',
                            'Yaw uses pose heading, never raw body IMU Z. Turnarounds are excluded.',
                            'Yaw response includes bounded altitude-hold operation, not locked 3D position.',
                            'No gain/configuration writes. No simulator truth used by fitting.',
                            'No absolute tracking-error acceptance budget or physical uncertainty bound established.',
                            'First-order model omits saturation, cross-axis coupling and changing dynamics.']}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('recording', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--nominal-tau-s', type=float, default=.45)
    ap.add_argument('--nominal-yaw-tau-s', type=float, default=.3)
    ap.add_argument('--max-delay-s', type=float)
    ap.add_argument('--max-tau-s', type=float)
    args = ap.parse_args(argv)
    try:
        records = [json.loads(line) for line in args.recording.read_text().splitlines() if line.strip()]
        report = analyze(records, nominal_tau_s=args.nominal_tau_s, nominal_yaw_tau_s=args.nominal_yaw_tau_s,
                         max_delay_s=args.max_delay_s, max_tau_s=args.max_tau_s)
    except (ValueError, KeyError, TypeError) as exc:
        report = {'schema': 1, 'accepted': False, 'errors': [str(exc)]}
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f"{'ACCEPTED RESPONSE MODEL' if report['accepted'] else 'REJECTED RESPONSE MODEL'}: {args.out}")
    return 0 if report['accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
