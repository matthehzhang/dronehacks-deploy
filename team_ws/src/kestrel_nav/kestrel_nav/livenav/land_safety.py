"""Raw PX4 landing-state supervision, independent of the wrapper heartbeat.

PX4 may use a different clock epoch. Advancing local-position timestamps supply
a reference; the smallest observed ROS-minus-source offset is retained so later
transport delay cannot be rebased into freshness. Absolute common delay of both
streams at startup is unobservable here: bootstrap assumes a live PX4 reference.
Two advancing land samples, source-age and independent receipt deadlines provide
bounded supervision, not proof against coordinated replay of every PX4 stream.
"""
import math


class Px4LandSafety:
    def __init__(self, params):
        self.p = params
        self.reference = self.sample = None
        self.offset = None
        self.advances = 0
        self.errors = {}
        self.fault = None

    def _ordered(self, stream, stamp, ros_now, mono_now, previous):
        if not all(math.isfinite(x) for x in (stamp, ros_now, mono_now)) or stamp <= 0:
            self.errors[stream] = f'raw land {stream}: invalid timestamp'
            return False
        if previous is not None:
            if stamp < previous[0] or ros_now < previous[1] or mono_now < previous[2]:
                self.fault = f'raw land {stream}: clock rewind; restart on ground'
                return False
            if stamp == previous[0]:
                return False  # duplicate cannot change value or refresh receipt
        return True

    def observe_reference(self, stamp, ros_now, mono_now):
        old = self.reference
        if not self._ordered('reference', stamp, ros_now, mono_now, old):
            return False
        if old is not None and stamp-old[0] > ros_now-old[1]+self.p.px4_reference_max_age_s:
            self.fault = 'raw land reference: source clock jumped; restart on ground'
            return False
        offset = ros_now-stamp
        self.offset = offset if self.offset is None else min(self.offset, offset)
        if offset-self.offset > self.p.px4_reference_max_age_s:
            self.errors['reference'] = 'raw land reference: delayed capture'
            return False
        self.reference = (stamp, ros_now, mono_now)
        self.errors.pop('reference', None)
        return True

    def _reference_error(self, ros_now, mono_now):
        if self.fault:
            return self.fault
        if 'reference' in self.errors:
            return self.errors['reference']
        if self.reference is None:
            return 'raw land: no PX4 clock reference'
        stamp, ros, mono = self.reference
        limit = self.p.px4_reference_max_age_s
        if (not all(math.isfinite(x) for x in (ros_now, mono_now))
                or not 0 <= ros_now-ros <= limit or not 0 <= mono_now-mono <= limit
                or not -self.p.future_tolerance_s <= ros_now-self.offset-stamp <= limit):
            return 'raw land: stale PX4 clock reference or clock discontinuity'
        return None

    def observe_land(self, stamp, landed, ros_now, mono_now):
        if not self._ordered('sample', stamp, ros_now, mono_now, self.sample):
            return False
        if type(landed) is not bool:
            self.errors['sample'] = 'raw land: invalid landed flag'
            return False
        error = self._reference_error(ros_now, mono_now)
        if error:
            self.errors['sample'] = error
            return False
        age = ros_now-self.offset-stamp
        if not -self.p.future_tolerance_s <= age <= self.p.land_max_age_s:
            self.errors['sample'] = 'raw land: delayed or future source timestamp'
            return False
        self.sample = (stamp, ros_now, mono_now, landed)
        self.advances += 1
        self.errors.pop('sample', None)
        return True

    def error(self, ros_now, mono_now):
        error = self._reference_error(ros_now, mono_now)
        if error:
            return error
        if 'sample' in self.errors:
            return self.errors['sample']
        if self.sample is None:
            return 'raw land: no accepted sample'
        stamp, ros, mono, _ = self.sample
        age = ros_now-self.offset-stamp
        if (not 0 <= ros_now-ros <= self.p.land_max_age_s
                or not 0 <= mono_now-mono <= self.p.land_max_age_s
                or not -self.p.future_tolerance_s <= age <= self.p.land_max_age_s):
            return 'raw land: stale source/receipt or clock discontinuity'
        if self.advances < 2:
            return 'raw land: waiting for advancing heartbeat'
        return None

    @property
    def landed(self):
        return self.sample[3] if self.sample else None

    def status(self, ros_now, mono_now):
        return dict(landed=self.landed, error=self.error(ros_now, mono_now),
                    accepted_samples=self.advances, limit_s=self.p.land_max_age_s,
                    source_stamp_s=self.sample[0] if self.sample else None,
                    source_age_s=ros_now-self.offset-self.sample[0] if self.sample and self.offset is not None else None,
                    receipt_age_s=mono_now-self.sample[2] if self.sample else None,
                    reference_stamp_s=self.reference[0] if self.reference else None,
                    reference_receipt_age_s=mono_now-self.reference[2] if self.reference else None,
                    clock_offset_s=self.offset)
