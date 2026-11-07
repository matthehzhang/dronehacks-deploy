"""Offline depth-silence advisory for waypoint or navigator-path execution.

Sample complete moving legs with acceleration/braking, then holds and turns at
execution stops. Path mode uses Navigator's smoothed points, headings and speed
profile; planner via points are not stops. Silence carries across all samples.
The unchanged refnav static-floor and hold/turn-pause rules are applied here.

This is an ideal timing/geometry model, not a PX4 replay: it cannot predict
tracking error, capture/settling dynamics, alignment retries, live replans or
visual texture loss. pose_margin_m tests an additional forward-offset view.
Flight E calibration (403 truth-pose frames) agreed on below-30% depth for 99%
of frames with the 0.4 m camera near limit and see-through net. That calibrates
visibility at those poses, not execution timing or real-aircraft performance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .render import Camera, DepthModel, DepthRenderer, body_pose_matrix
from .smooth import Path, SmoothParams, speed_profile


@dataclass
class SensingParams:
    hold_s: float = 2.5
    yaw_rate_dps: float = 30.0
    speed_mps: float = 0.3
    accel_mps2: float = 0.0
    brake_mps2: float = 0.0
    tunnel_speed_mps: float = 0.3
    min_valid_frac: float = 0.3
    depth_lost_land_s: float = 3.0
    depth_lost_pause_static: bool = False
    min_valid_frac_static: float = -1.0
    warn_margin_s: float = 1.0
    pose_margin_m: float = 0.10
    approach_m: float = 1.0  # compatibility only: now render full legs, never assume an omitted leg sees
    check_s: float = 0.3
    dt: float = 0.2
    stride: int = 8
    border_px: int = 24

    @classmethod
    def from_refnav(cls, rn: dict, **kw):
        keys = ('hold_s', 'yaw_rate_dps', 'speed_mps', 'accel_mps2', 'brake_mps2',
                'tunnel_speed_mps', 'min_valid_frac', 'depth_lost_land_s', 'min_valid_frac_static')
        vals = {k: float(rn[k]) for k in keys if k in rn}
        if 'depth_lost_pause_static' in rn:
            vals['depth_lost_pause_static'] = bool(rn['depth_lost_pause_static'])
        return cls(**{**vals, **kw})

    def motion_params(self, speed=None):
        # Same fallback as refnav.core._go_pose / _go_path; no new braking knob.
        brake = self.brake_mps2 if self.brake_mps2 > 0 else (self.accel_mps2 if self.accel_mps2 > 0 else 0.5)
        accel = self.accel_mps2 if self.accel_mps2 > 0 else brake
        return SmoothParams(cruise_mps=self.speed_mps if speed is None else speed,
                            accel_mps2=accel, brake_mps2=brake, slow_clear_m=0)


@dataclass
class Sample:
    phase: str
    xy: np.ndarray
    yaw: float
    duration: float


def make_renderer(cmap, cam: Camera = None):
    """Deterministic forward camera; net misses see the room behind it."""
    return DepthRenderer(cmap, cam or Camera(), DepthModel(noise=False, net_return_p=0.0), stride=8)


def valid_frac(renderer, xy, z, yaw, p: SensingParams = SensingParams()):
    R, t = body_pose_matrix(xy[0], xy[1], z, yaw)
    img = renderer.render(R, t)
    b, s = p.border_px, p.stride
    zz = img[b:img.shape[0] - b:s, b:img.shape[1] - b:s]
    return float(np.mean(np.isfinite(zz) & (zz > 0.0)))


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _heading(p, q):
    return math.atan2(q[1] - p[1], q[0] - p[0])


def _intervals(duration, dt):
    if dt <= 0:
        raise ValueError('dt must be positive')
    for t in np.arange(0.0, duration, dt):
        yield float(t), min(dt, duration - t)


def _stationary(xy, incoming, outgoing, p, hold=True):
    samples = []
    if hold:
        samples.extend(Sample('hold', xy, incoming, dt) for _, dt in _intervals(p.hold_s, p.dt))
    if outgoing is not None:
        err = _wrap(outgoing - incoming)
        rate = math.radians(p.yaw_rate_dps)
        if rate <= 0:
            raise ValueError('yaw_rate_dps must be positive')
        for t, dt in _intervals(abs(err) / rate, p.dt):
            yaw = _wrap(incoming + math.copysign(min(abs(err), (t + dt) * rate), err))
            samples.append(Sample('turn', xy, yaw, dt))
        samples.extend(Sample('check', xy, outgoing, dt) for _, dt in _intervals(p.check_s, p.dt))
    return samples


def _straight_path(a, b, p, speed=None):
    length = math.dist(a, b)
    if length <= 0:
        raise ValueError('zero-length execution leg')
    sp = p.motion_params(speed)
    # At least two intervals so a short rest-to-rest leg has an interior peak.
    n = max(3, int(math.ceil(length / sp.ds_m)) + 1)
    pts = np.linspace(a, b, n)
    path = Path(pts, np.linspace(0, length, n), np.zeros(n),
                np.full(n, _heading(a, b)), np.zeros(n), np.full(n, math.inf))
    return speed_profile(path, sp)


def _moving(path, p, first=0, last=None):
    """Integrate piecewise constant acceleration from v^2(s); sample in time.

    dt_segment = 2 ds / (v0+v1). Linear interpolation in distance versus time
    would incorrectly make the final braking interval constant-speed.
    """
    last = len(path.s) - 1 if last is None else last
    distances = path.s[first:last + 1]
    speeds = path.v[first:last + 1]
    ds = np.diff(distances)
    sums = speeds[:-1] + speeds[1:]
    if np.any((ds > 0) & (sums <= 0)):
        raise ValueError('positive-length interval has no executable speed')
    durations = 2 * ds / sums
    times = np.r_[0., np.cumsum(durations)]
    headings = np.unwrap(path.heading[first:last + 1].copy())
    # Gradient headings at sharp stops blend both legs; actual stop headings
    # are the incoming/outgoing segments, with an explicit turn between them.
    if first in path.stops:
        headings[0] = headings[1] + _wrap(_heading(path.pts[first], path.pts[first + 1]) - headings[1])
    if last in path.stops:
        headings[-1] = headings[-2] + _wrap(_heading(path.pts[last - 1], path.pts[last]) - headings[-2])
    for t, dt in _intervals(float(times[-1]), p.dt):
        mid = t + dt / 2
        i = min(int(np.searchsorted(times, mid, side='right')) - 1, len(ds) - 1)
        elapsed = mid - times[i]
        accel = (speeds[i + 1] - speeds[i]) / durations[i]
        s = distances[i] + speeds[i] * elapsed + 0.5 * accel * elapsed ** 2
        xy = np.array([np.interp(s, distances, path.pts[first:last + 1, axis]) for axis in (0, 1)])
        yield Sample('approach', xy, _wrap(float(np.interp(s, distances, headings))), dt)


def _path_samples(path, p):
    samples = []
    ends = [0] + [i for i in path.stops if 0 < i < len(path.s) - 1] + [len(path.s) - 1]
    for first, last in zip(ends[:-1], ends[1:]):
        samples.extend(_moving(path, p, first, last))
        if last != ends[-1]:
            incoming = _heading(path.pts[last - 1], path.pts[last])
            outgoing = _heading(path.pts[last], path.pts[last + 1])
            # refnav interior path stops turn/check immediately, without hold_s.
            samples.extend(_stationary(path.pts[last], incoming, outgoing, p, hold=False))
    return samples


def _waypoint_execution(waypoints, start_yaw, p):
    from .planner import flight_waypoints
    waypoints = flight_waypoints(waypoints)
    xy = [np.asarray(w['xy'], float) for w in waypoints]
    for k, waypoint in enumerate(waypoints):
        incoming = start_yaw if k == 0 else _heading(xy[k - 1], xy[k])
        samples = [] if k == 0 else list(_moving(_straight_path(xy[k - 1], xy[k], p), p))
        outgoing = _heading(xy[k], xy[k + 1]) if k + 1 < len(xy) else None
        samples.extend(_stationary(xy[k], incoming, outgoing, p, hold=k > 0))
        yield waypoint, samples


def timeline(waypoints, k, start_yaw, p):
    """Compatibility view of a waypoint timeline; braking is included in approach."""
    _, samples = list(_waypoint_execution(waypoints, start_yaw, p))[k]
    return [(sample.phase, sample.xy, sample.yaw) for sample in samples]


def path_timeline(plan, start_yaw, navigator, p):
    """Plan the same stops/pass-through goals as nav_node, on the provided map.

    No live corrections or inferred actual poses are injected into this advisory.
    """
    from .navplan import stops_and_legs
    start = np.asarray(plan['waypoints'][0]['xy'], float)
    _, metadata, legs = stops_and_legs(plan, start, start_yaw)
    paths = []
    previous = start
    for leg, meta in zip(legs, metadata):
        if leg['path']:
            path = navigator.plan(previous, leg['goals'])
            # Refnav applies its own speed and braking caps in addition to the
            # navigator profile. Preserve both: never increase a profiled speed.
            profile = Path(path.pts, path.s, path.kappa, path.heading,
                           np.minimum(path.v, p.speed_mps), path.clearance, path.stops)
            # Re-propagate acceleration/braking from any newly reduced cap.
            ds = np.diff(profile.s)
            sp = p.motion_params()
            for i in range(1, len(profile.v)):
                profile.v[i] = min(profile.v[i], math.sqrt(profile.v[i - 1] ** 2 + 2 * sp.accel_mps2 * ds[i - 1]))
            for i in range(len(profile.v) - 2, -1, -1):
                profile.v[i] = min(profile.v[i], math.sqrt(profile.v[i + 1] ** 2 + 2 * sp.brake_mps2 * ds[i]))
            path = profile
        else:
            speed = min(p.speed_mps, p.tunnel_speed_mps) if meta.get('tunnel') else p.speed_mps
            path = _straight_path(previous, leg['stop'], p, speed)
        paths.append(path)
        previous = np.asarray(leg['stop'], float)
    first_heading = float(paths[0].heading[0]) if paths else None
    yield plan['waypoints'][0], _stationary(start, start_yaw, first_heading, p, hold=False)
    for k, (leg, path) in enumerate(zip(legs, paths)):
        samples = _path_samples(path, p)
        outgoing = float(paths[k + 1].heading[0]) if k + 1 < len(paths) else None
        samples.extend(_stationary(path.pts[-1], float(path.heading[-1]), outgoing, p))
        yield dict(name=leg['name'], xy=leg['stop']), samples


STATIC_PHASES = ('hold', 'turn', 'check')
PAUSED_PHASES = ('hold', 'turn')


def _check_execution(execution, renderer, body_z, p):
    results = []
    run, run_phase = 0.0, None
    for k, (waypoint, samples) in enumerate(execution):
        best, best_phase = run, run_phase
        worst, worst_phase = 1.0, None
        for sample in samples:
            phase, xy, yaw = sample.phase, sample.xy, sample.yaw
            valid = valid_frac(renderer, xy, body_z, yaw, p)
            if p.pose_margin_m > 0:
                forward = np.asarray(xy) + p.pose_margin_m * np.array([math.cos(yaw), math.sin(yaw)])
                valid = min(valid, valid_frac(renderer, forward, body_z, yaw, p))
            if valid < worst:
                worst, worst_phase = valid, phase
            if p.depth_lost_pause_static and phase in PAUSED_PHASES:
                run, run_phase = 0.0, None
                continue
            floor = p.min_valid_frac_static if p.min_valid_frac_static >= 0 and phase in STATIC_PHASES else p.min_valid_frac
            if valid < floor:
                run_phase = run_phase or phase
                run += sample.duration
                if run > best:
                    best, best_phase = run, run_phase
            else:
                run, run_phase = 0.0, None
        verdict = ('FAIL' if best >= p.depth_lost_land_s else
                   'WARN' if best >= p.depth_lost_land_s - p.warn_margin_s else 'OK')
        results.append(dict(index=k, name=waypoint.get('name', f'wp{k}'), xy=list(waypoint['xy']),
                            blind_s=round(best, 2), worst_valid=round(worst, 3),
                            phase=best_phase or worst_phase, verdict=verdict))
    return results


def sensing_check(waypoints, start_yaw, renderer, body_z, p: SensingParams = SensingParams()):
    return _check_execution(_waypoint_execution(waypoints, start_yaw, p), renderer, body_z, p)


def path_sensing_check(plan, start_yaw, renderer, body_z, navigator, p: SensingParams = SensingParams()):
    return _check_execution(path_timeline(plan, start_yaw, navigator, p), renderer, body_z, p)
