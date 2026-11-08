"""Usable-depth watchdog. Rejected frames never refresh usable timestamps."""
from dataclasses import dataclass
import math


def verify_small_reset(previous, current, message):
    """Validate exactly one bounded reset per changed PX4 uint8 counter.

    Ignore delta fields whose counters did not change: PX4 retains old values
    there, including the initial ~90 degree ground yaw alignment.
    """
    if previous is None or len(previous) != 3 or len(current) != 3:
        return None, 'missing reset-counter history'
    if any(type(x) is not int or not 0 <= x <= 255 for x in (*previous, *current)):
        return None, 'invalid reset counters'
    changed = [b != a for a, b in zip(previous, current)]
    if not any(changed):
        return None, 'no reset-counter change'
    if any(ch and (b-a) % 256 != 1 for a, b, ch in zip(previous, current, changed)):
        return None, 'multiple/unobserved resets cannot be reconstructed'
    try:
        xy = tuple(float(x) for x in message.delta_xy[:2]) if changed[0] else (0., 0.)
        z = float(message.delta_z) if changed[1] else 0.
        heading = float(message.delta_heading) if changed[2] else 0.
    except (AttributeError, TypeError, ValueError):
        return None, 'missing reset delta for changed counter'
    if len(xy) != 2 or not all(math.isfinite(x) for x in (*xy, z, heading)):
        return None, 'nonfinite/incomplete reset delta'
    if math.hypot(*xy) > .10 or abs(z) > .10 or abs(heading) > math.radians(5.):
        return None, 'reset exceeds conservative .10m XY/Z or 5deg heading bounds'
    return dict(previous=list(previous), current=list(current), changed=changed,
                delta_xy_ned=list(xy), delta_z_ned=z, delta_heading_ned=heading,
                delta_z_enu=-z), ''


class DepthWatchdog:
    def __init__(self, max_age_s=.5):
        self.max_age_s = max_age_s
        self.capture = None
        self.received = None
        self.error = ''

    def usable(self, capture, received):
        if (not all(math.isfinite(x) for x in (capture, received))
                or not -.05 <= received-capture <= self.max_age_s
                or self.capture is not None and capture <= self.capture):
            return False
        self.capture, self.received, self.error = capture, received, ''
        return True

    def reject(self, reason):
        self.error = reason

    def action(self, now):
        if (self.capture is None or self.received is None
                or not -.05 <= now-self.capture <= self.max_age_s
                or not 0 <= now-self.received <= self.max_age_s):
            return 'stale'
        return 'pause' if self.error else 'run'

    def status(self, now):
        return dict(action=self.action(now), error=self.error,
                    usable_capture_age_s=None if self.capture is None else now-self.capture,
                    usable_receipt_age_s=None if self.received is None else now-self.received)


class Px4ClockAnchor:
    """Observe advancing position-clock pairs before aligning PX4 boot time.

    A startup /clock of zero or a held first nonzero tick cannot establish the
    offset. Only stable advancing pairs over .25s can. After flight starts a
    clock discontinuity is a fault, never an opportunity to accept a replay.
    """
    def __init__(self):
        self.offset = None
        self.pairs = []
        self.first_ros = None
        self.first_boot = None
        self.reason = 'waiting for positive advancing ROS/PX4 clocks'
        self.resets = 0
        self.fault = ''
        self.anchor = None

    def observe(self, ros, boot, *, active=False):
        if self.first_ros is None:
            self.first_ros, self.first_boot = ros, boot
        if not all(math.isfinite(x) and x > 0 for x in (ros, boot)):
            return None
        if self.pairs and boot <= self.pairs[-1][1]:
            if boot >= self.pairs[-1][1]-.5:
                return None  # duplicate/out-of-order packet cannot refresh.
            self.reason = 'PX4 boot clock moved backwards: new ground epoch required'
            if active:
                self.fault = self.reason
                return None
            self.offset, self.anchor, self.pairs = None, None, []
            self.resets += 1
        if self.offset is not None and not -.05 <= ros-(boot+self.offset) <= .5:
            self.reason = 'PX4/ROS clock discontinuity exceeds existing freshness window'
            if active:
                self.fault = self.reason
                return None
            self.offset, self.anchor = None, None
            self.pairs = []
            self.resets += 1
        if self.pairs and ros <= self.pairs[-1][0]:
            # Do not ratchet a minimum offset downward while ROS time is held.
            return self.offset
        self.pairs.append((ros, boot))
        self.pairs = self.pairs[-256:]
        if self.offset is None:
            offsets = [r-b for r,b in self.pairs]
            if max(offsets)-min(offsets) > .05:
                self.pairs = [(ros, boot)]
                self.reason = 'qualifying stable relative ROS/PX4 clock rate'
                return None
            if len(self.pairs) < 5 or ros-self.pairs[0][0] < .25:
                self.reason = 'qualifying advancing ROS/PX4 clocks for .25s'
                return None
            self.offset = min(offsets)
            self.anchor = dict(ros=ros, px4_boot=boot, samples=len(self.pairs), span_s=ros-self.pairs[0][0])
            self.reason = 'stable advancing clocks aligned'
        else:
            self.offset = min(self.offset, ros-boot)
        return self.offset

    def status(self):
        return dict(offset_s=self.offset, first_ros=self.first_ros, first_px4_boot=self.first_boot,
                    reason=self.reason, resets=self.resets, anchor=self.anchor,
                    latest_pair=None if not self.pairs else list(self.pairs[-1]), fault=self.fault)
