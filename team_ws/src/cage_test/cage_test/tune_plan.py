"""The flight as a list of segments: hop, hover, step, speed, yaw and vertical patterns.

No ROS in here, so it is tested on its own (``test/test_tune_plan.py``).

All points are ENU in the ``odom`` frame, the frame ``flight_test`` reads from
``/visual_slam/tracking/odometry`` and publishes on ``/skybeta/setpoint/pose``.
Every flight is:

    up      ground -> H = start + climb_height_m, at speed_mps
    <the patterns in ``sequence``, each starting and ending at H>
    down    H -> start + low_height_m, at speed_mps
    (touchdown, flown by the node as a twist)

Segment kinds:

    move   the setpoint slides start -> end at ``speed`` (a carrot), then holds
    step   the setpoint jumps to ``end`` at once (a step input), then holds
    yaw    position held at ``end``; yaw slides by ``yaw_delta`` at ``yaw_rate``
    hold   position held at ``end`` for ``hold_s``
"""
import math
from collections import namedtuple

Segment = namedtuple('Segment', [
    'name', 'pattern', 'kind', 'start', 'end', 'speed', 'hold_s',
    'yaw_delta', 'yaw_rate'])

KINDS = ('move', 'step', 'yaw', 'hold')
PATTERNS = ('hop', 'hover', 'steps', 'speed', 'yaw', 'vertical', 'square')

#: Defaults, also the launch-file defaults. Units in the names.
DEFAULTS = {
    'climb_height_m': 1.0,
    'low_height_m': 0.5,
    'speed_mps': 0.3,
    'hold_seconds': 3.0,
    'forward_m': 1.0,
    'hover_seconds': 20.0,
    'step_m': 0.5,
    'step_z_m': 0.3,
    'step_hold_seconds': 5.0,
    'speed_list_mps': '0.25,0.5,0.75,1.0',
    'speed_distance_m': 1.0,
    'speed_axes': 'forward',
    'yaw_rates_dps': '30,60,90,120',
    'yaw_turn_deg': 360.0,
    'vertical_speeds_mps': '0.2,0.4,0.6',
    'vertical_span_m': 0.4,
    'square_m': 1.0,
}

#: Above these the plan is refused at startup. Cage limits, not PX4 limits.
MAX_SPEED_MPS = 2.0
MAX_VERTICAL_SPEED_MPS = 1.0
MAX_YAW_RATE_DPS = 180.0


def parse_list(text):
    """'0.25, 0.5,1' -> [0.25, 0.5, 1.0]. Empty -> []."""
    if isinstance(text, (list, tuple)):
        return [float(v) for v in text]
    return [float(v) for v in str(text).replace(';', ',').split(',') if v.strip()]


def parse_names(text):
    if isinstance(text, (list, tuple)):
        return [str(v).strip() for v in text if str(v).strip()]
    return [v.strip() for v in str(text).split(',') if v.strip()]


def yaw_from_xyzw(x, y, z, w):
    """ENU yaw (rad) of a ROS xyzw quaternion. 0 = body +x along odom +x."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def xyzw_from_yaw(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def wrap_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def seg_length(seg):
    return math.dist(seg.start, seg.end)


def carrot(seg, progress_m):
    """The point ``progress_m`` along ``seg``, stopping at its end."""
    length = seg_length(seg)
    if length <= 0.0:
        return tuple(seg.end)
    f = min(max(progress_m / length, 0.0), 1.0)
    return tuple(s + f * (e - s) for s, e in zip(seg.start, seg.end))


def distance_to_path(point, seg):
    """Distance from ``point`` to the straight piece start -> end.

    Lagging behind the moving setpoint is not "off the path"; drifting
    sideways or overshooting past the end is.
    """
    a, b, p = seg.start, seg.end, point
    ab = [e - s for s, e in zip(a, b)]
    denom = sum(v * v for v in ab)
    if denom <= 0.0:
        return math.dist(p, b)
    t = sum((pi - ai) * v for pi, ai, v in zip(p, a, ab)) / denom
    t = min(max(t, 0.0), 1.0)
    return math.dist(p, tuple(ai + t * v for ai, v in zip(a, ab)))


def segment_duration(seg):
    """Nominal time to fly the ramp of ``seg`` (without its hold)."""
    if seg.kind == 'move' and seg.speed:
        return seg_length(seg) / seg.speed
    if seg.kind == 'yaw' and seg.yaw_rate:
        return abs(seg.yaw_delta) / seg.yaw_rate
    return 0.0


def plan_duration(segments):
    return sum(segment_duration(s) + s.hold_s for s in segments)


def _cfg(config):
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in (config or {}).items() if v is not None})
    return cfg


def validate(config, sequence):
    """Raise ValueError on a plan that should not be flown."""
    cfg = _cfg(config)
    names = parse_names(sequence)
    if not names:
        raise ValueError('sequence is empty')
    for name in names:
        if name not in PATTERNS:
            raise ValueError(f'unknown pattern "{name}" (known: {", ".join(PATTERNS)})')
    climb, low = float(cfg['climb_height_m']), float(cfg['low_height_m'])
    if not 0.0 < low < climb:
        raise ValueError('need 0 < low_height_m < climb_height_m')
    if float(cfg['speed_mps']) <= 0.0:
        raise ValueError('speed_mps must be > 0')
    for v in parse_list(cfg['speed_list_mps']):
        if not 0.0 < v <= MAX_SPEED_MPS:
            raise ValueError(f'speed_list_mps: {v} not in (0, {MAX_SPEED_MPS}]')
    for v in parse_list(cfg['vertical_speeds_mps']):
        if not 0.0 < v <= MAX_VERTICAL_SPEED_MPS:
            raise ValueError(
                f'vertical_speeds_mps: {v} not in (0, {MAX_VERTICAL_SPEED_MPS}]')
    for r in parse_list(cfg['yaw_rates_dps']):
        if not 0.0 < r <= MAX_YAW_RATE_DPS:
            raise ValueError(f'yaw_rates_dps: {r} not in (0, {MAX_YAW_RATE_DPS}]')
    for axis in parse_names(cfg['speed_axes']):
        if axis not in ('forward', 'left'):
            raise ValueError(f'speed_axes: "{axis}" is not forward or left')
    if 'steps' in names and climb - float(cfg['step_z_m']) <= low:
        raise ValueError('steps: climb_height_m - step_z_m must stay above low_height_m')
    if 'vertical' in names and climb - float(cfg['vertical_span_m']) <= low:
        raise ValueError(
            'vertical: climb_height_m - vertical_span_m must stay above low_height_m')


def build_plan(start, yaw_enu, config=None, sequence='hop'):
    """Every segment of the flight, from ``start`` (the vehicle on the ground).

    "Forward" is the nose at ``start`` (``yaw_enu``), "left" is 90 deg left of
    it, both level. Every pattern starts and ends at H, above ``start``.
    """
    cfg = _cfg(config)
    validate(cfg, sequence)
    x0, y0, z0 = (float(v) for v in start)
    climb = float(cfg['climb_height_m'])
    hold = float(cfg['hold_seconds'])
    speed = float(cfg['speed_mps'])
    fwd = (math.cos(yaw_enu), math.sin(yaw_enu), 0.0)
    left = (-math.sin(yaw_enu), math.cos(yaw_enu), 0.0)
    up = (0.0, 0.0, 1.0)
    home = (x0, y0, z0 + climb)

    def at(*terms):
        """home + sum(distance * unit)."""
        p = list(home)
        for dist, unit in terms:
            for i in range(3):
                p[i] += dist * unit[i]
        return tuple(p)

    segs = [Segment('up', 'takeoff', 'move', (x0, y0, z0), home, speed, hold, 0.0, 0.0)]

    def move(name, pattern, a, b, v, hold_s=hold):
        segs.append(Segment(name, pattern, 'move', a, b, v, hold_s, 0.0, 0.0))

    def step(name, pattern, a, b, hold_s):
        segs.append(Segment(name, pattern, 'step', a, b, None, hold_s, 0.0, 0.0))

    for pattern in parse_names(sequence):
        if pattern == 'hop':
            out = at((float(cfg['forward_m']), fwd))
            move('forward', 'hop', home, out, speed)
            move('back', 'hop', out, home, speed)
        elif pattern == 'hover':
            segs.append(Segment('hover', 'hover', 'hold', home, home, None,
                                float(cfg['hover_seconds']), 0.0, 0.0))
        elif pattern == 'steps':
            s, sz = float(cfg['step_m']), float(cfg['step_z_m'])
            h = float(cfg['step_hold_seconds'])
            for label, target in (('forward', at((s, fwd))),
                                  ('left', at((s, left))),
                                  ('up', at((sz, up))),
                                  ('down', at((-sz, up)))):
                step(f'step {label} {s if label in ("forward", "left") else sz:.2f}',
                     'steps', home, target, h)
                step(f'step {label} return', 'steps', target, home, h)
        elif pattern == 'speed':
            d = float(cfg['speed_distance_m'])
            for axis in parse_names(cfg['speed_axes']):
                unit = fwd if axis == 'forward' else left
                away = 'forward' if axis == 'forward' else 'left'
                back = 'back' if axis == 'forward' else 'right'
                for v in parse_list(cfg['speed_list_mps']):
                    out = at((d, unit))
                    move(f'{away} @{v:.2f}', 'speed', home, out, v)
                    move(f'{back} @{v:.2f}', 'speed', out, home, v)
        elif pattern == 'yaw':
            turn = math.radians(float(cfg['yaw_turn_deg']))
            for r in parse_list(cfg['yaw_rates_dps']):
                rate = math.radians(r)
                # Left (CCW from above) first, then back: ends on the start heading.
                segs.append(Segment(f'yaw left @{r:.0f}dps', 'yaw', 'yaw', home, home,
                                    None, hold, turn, rate))
                segs.append(Segment(f'yaw right @{r:.0f}dps', 'yaw', 'yaw', home, home,
                                    None, hold, -turn, rate))
        elif pattern == 'vertical':
            span = float(cfg['vertical_span_m'])
            top, bottom = at((span, up)), at((-span, up))
            move('vertical to top', 'vertical', home, top, speed)
            for v in parse_list(cfg['vertical_speeds_mps']):
                move(f'down @{v:.2f}', 'vertical', top, bottom, v)
                move(f'up @{v:.2f}', 'vertical', bottom, top, v)
            move('vertical to home', 'vertical', top, home, speed)
        elif pattern == 'square':
            side = float(cfg['square_m'])
            corners = [home, at((side, fwd)), at((side, fwd), (side, left)),
                       at((side, left)), home]
            for i, (a, b) in enumerate(zip(corners, corners[1:])):
                move(f'square {i + 1}/4', 'square', a, b, speed)

    low = (x0, y0, z0 + float(cfg['low_height_m']))
    segs.append(Segment('down', 'landing', 'move', home, low, speed, hold, 0.0, 0.0))
    return segs
