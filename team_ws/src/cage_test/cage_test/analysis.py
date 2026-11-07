"""Per-segment numbers from a run: tracking, step response, yaw, hover noise, vibration.

No ROS in here. ``flight_test`` calls :func:`summarize` when a run ends and
writes ``summary.json`` / ``summary.txt``; ``tools/summarize_run.py`` re-runs it
from a run directory's CSVs on a laptop.

A sample is a dict with (missing or None where unknown):

    t, seg, phase ('ramp' | 'hold'), cmd_x/y/z, cmd_yaw (rad, ENU or None),
    vio_x/y/z, vio_yaw, px4_x/y/z, px4_yaw, odom_age, battery_v

IMU rows are ``(t, ax, ay, az, gx, gy, gz)``.
"""
import math
import statistics

from cage_test.tune_plan import distance_to_path, wrap_pi


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _xyz(row, prefix):
    p = tuple(_f(row.get(f'{prefix}_{a}')) for a in 'xyz')
    return None if any(v is None for v in p) else p


def _r(v, nd=3):
    return None if v is None else round(v, nd)


def _std(values):
    return statistics.pstdev(values) if len(values) > 1 else None


def _unwrap(angles):
    out, prev = [], None
    for a in angles:
        if prev is None:
            out.append(a)
        else:
            out.append(out[-1] + wrap_pi(a - prev))
        prev = a
    return out


def _slope(ts, ys):
    if len(ts) < 3:
        return None
    mt, my = statistics.fmean(ts), statistics.fmean(ys)
    den = sum((t - mt) ** 2 for t in ts)
    if den <= 0.0:
        return None
    return sum((t - mt) * (y - my) for t, y in zip(ts, ys)) / den


def _speeds(rows, prefix, direction):
    """Speed along ``direction`` (unit) from finite differences over >= 0.2 s."""
    pts = [(r['t'], _xyz(r, prefix)) for r in rows]
    pts = [(t, p) for t, p in pts if p is not None]
    out, j = [], 0
    for i in range(len(pts)):
        while j < i and pts[i][0] - pts[j][0] > 0.2:
            j += 1
        if j > 0:
            t0, p0 = pts[j - 1]
            t1, p1 = pts[i]
            if t1 > t0:
                out.append(sum((b - a) * u for a, b, u in zip(p0, p1, direction)) / (t1 - t0))
    return out


def _imu_stats(imu, t0, t1):
    rows = [r for r in imu if t0 <= r[0] <= t1]
    if len(rows) < 10:
        return None
    acc_std = math.sqrt(sum(_std([r[i] for r in rows]) ** 2 for i in (1, 2, 3)))
    gyro_std = math.sqrt(sum(_std([r[i] for r in rows]) ** 2 for i in (4, 5, 6)))
    rate = (len(rows) - 1) / (rows[-1][0] - rows[0][0]) if rows[-1][0] > rows[0][0] else None
    return {'acc_std_mps2': _r(acc_std), 'gyro_std_dps': _r(math.degrees(gyro_std), 2),
            'rate_hz': _r(rate, 1)}


def _step_response(rows, seg):
    """Rise time (10-90 %), overshoot, settling (2 % of the step or 3 cm)."""
    size = math.dist(seg.start, seg.end)
    if size <= 0.0:
        return {}
    unit = tuple((e - s) / size for s, e in zip(seg.start, seg.end))
    pts = []
    for r in rows:
        p = _xyz(r, 'vio')
        if p is not None:
            pts.append((r['t'], sum((pi - si) * u for pi, si, u in zip(p, seg.start, unit))))
    if len(pts) < 5:
        return {}
    t0 = pts[0][0]
    t10 = next((t for t, x in pts if x >= 0.1 * size), None)
    t90 = next((t for t, x in pts if x >= 0.9 * size), None)
    peak = max(x for _, x in pts)
    band = max(0.02 * size, 0.03)
    settle = None
    for i in range(len(pts)):
        if all(abs(x - size) <= band for _, x in pts[i:]):
            settle = pts[i][0] - t0
            break
    tail = [x for t, x in pts if t >= pts[-1][0] - 1.0]
    return {
        'step_m': _r(size),
        'rise_time_s': _r(t90 - t10, 2) if t10 is not None and t90 is not None else None,
        'overshoot_pct': _r(max(0.0, peak - size) / size * 100.0, 1),
        'settling_time_s': _r(settle, 2),
        'steady_state_error_m': _r(abs(statistics.fmean(tail) - size)) if tail else None,
    }


def summarize_segment(seg, rows, imu_oak=(), imu_px4=()):
    out = {'name': seg.name, 'pattern': seg.pattern, 'kind': seg.kind}
    if not rows:
        out['samples'] = 0
        return out
    t0, t1 = rows[0]['t'], rows[-1]['t']
    out['samples'] = len(rows)
    out['duration_s'] = _r(t1 - t0, 2)
    ramp = [r for r in rows if r.get('phase') == 'ramp']
    hold = [r for r in rows if r.get('phase') == 'hold']
    out['ramp_s'] = _r((ramp[-1]['t'] - ramp[0]['t']) if ramp else 0.0, 2)

    # Distance from the path, VIO (what is flown on) and PX4 (what EKF2 thinks).
    for src in ('vio', 'px4'):
        d = [distance_to_path(p, seg) for p in (_xyz(r, src) for r in rows) if p is not None]
        if d:
            out[f'{src}_max_off_path_m'] = _r(max(d))
            out[f'{src}_rms_off_path_m'] = _r(math.sqrt(statistics.fmean(v * v for v in d)))
    # PX4 minus VIO: the vision link / EKF2 disagreement.
    offs = []
    for r in rows:
        a, b = _xyz(r, 'vio'), _xyz(r, 'px4')
        if a is not None and b is not None:
            offs.append(math.dist(a, b))
    if offs:
        out['px4_vio_offset_mean_m'] = _r(statistics.fmean(offs))
        out['px4_vio_offset_max_m'] = _r(max(offs))
    ages = [_f(r.get('odom_age')) for r in rows]
    ages = [a for a in ages if a is not None]
    if ages:
        out['max_odom_age_s'] = _r(max(ages))
    volts = [_f(r.get('battery_v')) for r in rows]
    volts = [v for v in volts if v]
    if volts:
        out['battery_v_start'] = _r(volts[0], 2)
        out['battery_v_end'] = _r(volts[-1], 2)
        out['battery_v_min'] = _r(min(volts), 2)
    if hold:
        tail = [r for r in hold if r['t'] >= hold[-1]['t'] - 1.0]
        errs = [math.dist(p, seg.end) for p in (_xyz(r, 'vio') for r in tail) if p is not None]
        if errs:
            out['final_error_m'] = _r(statistics.fmean(errs))

    if seg.kind == 'move' and seg.speed:
        length = math.dist(seg.start, seg.end)
        out['commanded_speed_mps'] = _r(seg.speed, 2)
        if length > 0:
            unit = tuple((e - s) / length for s, e in zip(seg.start, seg.end))
            for src in ('vio', 'px4'):
                sp = _speeds(rows, src, unit)
                if sp:
                    out[f'{src}_peak_speed_mps'] = _r(max(sp))
            pts = [_xyz(r, 'vio') for r in rows]
            along = [sum((pi - si) * u for pi, si, u in zip(p, seg.start, unit))
                     for p in pts if p is not None]
            if along:
                out['overshoot_m'] = _r(max(0.0, max(along) - length))
            # Lag: how far behind the moving setpoint, while it moves.
            lags = []
            for r in ramp:
                c, p = _xyz(r, 'cmd'), _xyz(r, 'vio')
                if c is not None and p is not None:
                    lags.append(math.dist(c, p))
            if lags:
                out['max_lag_m'] = _r(max(lags))
    elif seg.kind == 'step':
        out.update(_step_response(rows, seg))
    elif seg.kind == 'yaw':
        out['commanded_rate_dps'] = _r(math.degrees(seg.yaw_rate), 1)
        out['commanded_turn_deg'] = _r(math.degrees(seg.yaw_delta), 1)
        yr = [(r['t'], _f(r.get('px4_yaw')), _f(r.get('cmd_yaw'))) for r in ramp]
        yr = [v for v in yr if v[1] is not None]
        if len(yr) > 5:
            n = len(yr)
            mid = yr[n // 5: n - n // 5] or yr
            unwrapped = _unwrap([y for _, y, _ in mid])
            rate = _slope([t for t, _, _ in mid], unwrapped)
            out['achieved_rate_dps'] = _r(abs(math.degrees(rate)), 1) if rate is not None else None
            lags = [abs(math.degrees(wrap_pi(c - y))) for _, y, c in yr if c is not None]
            if lags:
                out['mean_yaw_lag_deg'] = _r(statistics.fmean(lags), 1)
                out['max_yaw_lag_deg'] = _r(max(lags), 1)
        if hold:
            tail = [r for r in hold if _f(r.get('px4_yaw')) is not None
                    and _f(r.get('cmd_yaw')) is not None]
            if tail:
                out['final_yaw_error_deg'] = _r(abs(math.degrees(wrap_pi(
                    _f(tail[-1]['cmd_yaw']) - _f(tail[-1]['px4_yaw'])))), 1)
    elif seg.kind == 'hold':
        for src in ('vio', 'px4'):
            pts = [_xyz(r, src) for r in rows]
            pts = [p for p in pts if p is not None]
            if len(pts) > 2:
                out[f'{src}_std_xy_m'] = _r(math.hypot(_std([p[0] for p in pts]),
                                                       _std([p[1] for p in pts])))
                out[f'{src}_std_z_m'] = _r(_std([p[2] for p in pts]))
        yaws = [_f(r.get('px4_yaw')) for r in rows]
        yaws = [y for y in yaws if y is not None]
        if len(yaws) > 2:
            out['yaw_std_deg'] = _r(math.degrees(_std(_unwrap(yaws))), 2)

    for name, imu in (('oak_imu', imu_oak), ('px4_imu', imu_px4)):
        stats = _imu_stats(imu, t0, t1)
        if stats:
            out[name] = stats
    return out


def summarize(segments, samples, imu_oak=(), imu_px4=()):
    """One dict per segment that has samples, in flight order."""
    by_seg = {}
    for row in samples:
        seg = row.get('seg')
        if seg is None or seg == '' or int(seg) < 0:
            continue
        by_seg.setdefault(int(seg), []).append(row)
    return [summarize_segment(segments[i], by_seg[i], imu_oak, imu_px4)
            for i in sorted(by_seg) if i < len(segments)]


#: Columns of the text table, per kind: (key, header).
_TABLE = {
    'move': [('commanded_speed_mps', 'cmd v'), ('vio_peak_speed_mps', 'peak v'),
             ('max_lag_m', 'max lag'), ('vio_max_off_path_m', 'off path'),
             ('overshoot_m', 'overshoot'), ('final_error_m', 'final err')],
    'step': [('step_m', 'step'), ('rise_time_s', 'rise s'), ('overshoot_pct', 'OS %'),
             ('settling_time_s', 'settle s'), ('steady_state_error_m', 'ss err'),
             ('vio_max_off_path_m', 'off path')],
    'yaw': [('commanded_rate_dps', 'cmd dps'), ('achieved_rate_dps', 'got dps'),
            ('mean_yaw_lag_deg', 'lag deg'), ('final_yaw_error_deg', 'end err'),
            ('vio_max_off_path_m', 'drift m'), ('max_odom_age_s', 'odom age')],
    'hold': [('vio_std_xy_m', 'std xy'), ('vio_std_z_m', 'std z'),
             ('px4_vio_offset_mean_m', 'px4-vio'), ('yaw_std_deg', 'yaw std'),
             ('vio_max_off_path_m', 'max drift'), ('battery_v_end', 'batt V')],
}


def format_table(summary):
    """Readable per-segment table, grouped by kind as segments come."""
    lines = []
    last_kind = None
    for s in summary:
        cols = _TABLE.get(s['kind'], [])
        if s['kind'] != last_kind:
            lines.append('')
            lines.append(f'{"segment":<24}' + ''.join(f'{h:>10}' for _, h in cols)
                         + f'{"oak acc":>10}')
            last_kind = s['kind']
        vals = []
        for key, _ in cols:
            v = s.get(key)
            vals.append(f'{"-" if v is None else v:>10}')
        oak = (s.get('oak_imu') or {}).get('acc_std_mps2')
        lines.append(f'{s["name"]:<24}' + ''.join(vals) + f'{"-" if oak is None else oak:>10}')
    return '\n'.join(lines).lstrip('\n')
