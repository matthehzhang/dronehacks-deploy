"""Input supervision for the hardware adapter. No ROS or flight authority here.

Zero velocity is a request to PX4, not a guarantee of hover with a bad estimator.
Receipt times use a monotonic clock; capture stamps use the ROS clock. Neither
replayed packets nor a stopped ROS clock can keep a dead stream qualified.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SafetyParams:
    odom_max_age_s: float = 0.5
    tracking_max_age_s: float = 0.5
    depth_max_age_s: float = 0.5
    state_max_age_s: float = 1.0
    land_max_age_s: float = 2.6  # initial finite policy; hardware cadence must be measured
    px4_reference_max_age_s: float = 0.5
    future_tolerance_s: float = 0.05
    qualify_s: float = 1.0
    min_depth_valid_fraction: float = 0.3
    min_corridor_coverage: float = 0.6
    min_corridor_rays: int = 12

    def __post_init__(self):
        for name in ('odom_max_age_s', 'tracking_max_age_s', 'depth_max_age_s',
                     'state_max_age_s', 'land_max_age_s', 'px4_reference_max_age_s',
                     'future_tolerance_s', 'qualify_s'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('min_depth_valid_fraction', 'min_corridor_coverage'):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in (0, 1]')
        if self.min_corridor_rays < 1:
            raise ValueError('min_corridor_rays must be positive')


class InputSafety:
    """One mission per process. Faults and authority loss require ground restart."""

    def __init__(self, params=None):
        self.p = params or SafetyParams()
        self.samples = {}
        self.errors = {}
        self.fault = None
        self.active = False
        self.saw_disarmed = False
        self.healthy_since = None
        self.reason = 'waiting for inputs'
        self.action = 'silent'
        self.waived = frozenset()  # streams the current mission phase does not need (e.g. depth on final descent)

    def trip(self, reason):
        if self.fault is None:
            self.fault = reason
        self.reason = self.fault

    def reject(self, stream, reason):
        self.errors[stream] = reason
        self.healthy_since = None
        if self.active:
            self.trip(reason)

    def accept(self, stream, stamp, ros_now, received):
        """Validate advancing capture time before the caller replaces stored data."""
        limit = getattr(self.p, f'{stream}_max_age_s')
        if not all(math.isfinite(v) for v in (stamp, ros_now, received)) or stamp <= 0:
            self.reject(stream, f'{stream}: invalid timestamp')
            return False
        age = ros_now - stamp
        if age < -self.p.future_tolerance_s or age > limit:
            self.reject(stream, f'{stream}: capture timestamp outside freshness window')
            return False
        previous = self.samples.get(stream)
        if previous is not None and stamp <= previous[0]:
            # A queued duplicate must not refresh the receive watchdog.
            return False
        self.samples[stream] = (stamp, received)
        self.errors.pop(stream, None)
        return True

    def stream_error(self, stream, ros_now, now):
        if stream in self.errors:
            return self.errors[stream]
        sample = self.samples.get(stream)
        if sample is None:
            return f'{stream}: no valid sample'
        stamp, received = sample
        limit = getattr(self.p, f'{stream}_max_age_s')
        if not (-self.p.future_tolerance_s <= ros_now - stamp <= limit):
            return f'{stream}: stale capture or clock discontinuity'
        if not 0 <= now - received <= limit:
            return f'{stream}: stale receipt or clock discontinuity'
        return None

    def evaluate(self, ros_now, now, *, offboard, armed, landed, link, position_valid, frozen=False,
                 land_error=None):
        """Return silent / ready (zero twist) / run / stop (latched zero twist).

        Loss of fresh authority means silence. Known Offboard plus a sensor fault
        means keep requesting zero twist until the pilot takes over. Never call
        the interface's hold service: it may recall an old position target.
        """
        state_error = self.stream_error('state', ros_now, now)
        if state_error or not link or frozen:
            self.healthy_since = None
            self.reason = state_error or ('flight-interface hold requested' if frozen else 'PX4 link lost')
            if self.active:
                self.trip(self.reason)
            else:
                self.saw_disarmed = False
            self.action = 'silent'
            return self.action
        if not armed:
            self.saw_disarmed = True
        if self.active and (not offboard or not armed):
            self.trip('Offboard authority lost; pilot owns vehicle')
        if self.fault:
            self.reason = self.fault
            self.action = 'stop' if offboard and armed else 'silent'
            return self.action
        error = land_error or next((e for s in ('odom', 'tracking', 'depth')
                      if s not in self.waived and (e := self.stream_error(s, ros_now, now))), None)
        if not position_valid:
            error = error or 'PX4 local position invalid'
        if error:
            self.healthy_since = None
            self.reason = error
            if self.active or offboard:
                self.active = True
                self.trip(error)
            self.action = 'stop' if offboard and armed else 'silent'
            return self.action
        if self.active:
            self.action, self.reason = 'run', 'inputs healthy'
            return self.action
        if self.healthy_since is None:
            self.healthy_since = now
        qualified = now - self.healthy_since >= self.p.qualify_s
        if offboard:
            if not (qualified and self.saw_disarmed and armed and landed is True):
                self.active = True
                self.trip('Offboard started without qualified ground readiness')
                self.action = 'stop' if armed else 'silent'
            else:
                self.active = True
                self.action, self.reason = 'run', 'qualified Offboard start'
        elif qualified and self.saw_disarmed and landed is True:
            self.action, self.reason = 'ready', 'qualified; waiting for pilot'
        else:
            self.action, self.reason = 'silent', 'waiting for qualified ground readiness'
        return self.action


def corridor_error(frame, vx_body, vy_body, *, half_width, band, at_m, params):
    """Check validity in the commanded motion direction, including reverse HOLD.

    Uses the existing DepthFrame ray-coverage calculation, not fabricated range
    returns. This is a missing-data guard, not proof of collision-free stopping.
    """
    if not all(math.isfinite(v) for v in (vx_body, vy_body)):
        return 'non-finite horizontal command'
    if math.hypot(vx_body, vy_body) <= 1e-6:
        return None
    if frame is None:
        return 'no depth frame for motion corridor'
    bearing = math.atan2(vy_body, vx_body)
    coverage, count = frame.corridor_coverage(bearing, half_width, band, at_m)
    if count < params.min_corridor_rays or not math.isfinite(coverage) or coverage < params.min_corridor_coverage:
        return f'motion corridor unobserved (coverage={coverage:.2f}, rays={count})'
    return None
