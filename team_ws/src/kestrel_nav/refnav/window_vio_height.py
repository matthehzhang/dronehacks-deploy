"""Local altitude feedback handoff; route, timing and PX4 parameters stay unchanged.

Align ROS VIO's up axis to PX4's down axis during existing stationary periods.
Freeze that offset before the window crossing leg, then retain it to landing:
PX4's terrain-related height error can persist after leaving the window.
"""
from collections import deque
from dataclasses import replace
import math


class WindowVioHeight:
    MAX_AGE_S = 1.0
    CALIBRATION_S = 2.0

    def __init__(self, warn):
        self.warn = warn
        self.vio = None                 # receipt time, source stamp, down position
        self.samples = deque()
        self.calibrated_stamp = None
        self.offset = None
        self.active = False
        self.last_down = self.last_px4_down = None
        self.source = 'px4'
        self.missing_logged = False
        self.raw_down = None
        self.return_offset = None

    def begin_return(self, vehicle, launch_down):
        """Called by the return mission's existing confirmed-touchdown transition.

        Same floor/pad height as launch: align this PX4 ground sample to the
        remembered launch Z. This changes our feedback, not PX4's estimator.
        """
        self.return_offset = launch_down - self.raw_down
        self.active, self.source = False, 'px4_reanchored'
        self.warn(f'!!!!!!!!!! RETURN HEIGHT REANCHORED: PX4 NED offset {self.return_offset:+.3f} m; '
                  'pad contact uses original launch height; VIO height disabled for return !!!!!!!!!!')
        return replace(vehicle, pos=(*vehicle.pos[:2], launch_down))

    def observe(self, now, stamp, up):
        if not all(math.isfinite(x) for x in (now, stamp, up)):
            return
        # Repeated messages with a frozen source stamp must not refresh stale data.
        if self.vio is not None and stamp <= self.vio[1]:
            return
        self.vio = (now, stamp, -up)

    def vehicle(self, now, mission, vehicle, px4_received):
        if vehicle.pos is None:
            return vehicle
        self.raw_down = vehicle.pos[2]
        if self.return_offset is not None:
            self.last_down = self.raw_down + self.return_offset
            return replace(vehicle, pos=(*vehicle.pos[:2], self.last_down))
        legs = mission.p.legs
        windows = [i for i, leg in enumerate(legs)
                   if leg.get('align', {}).get('group') == 'window']
        if not windows and not self.active:
            return vehicle
        # Skybeta return modes replace their route/leg metadata after pad contact.
        # Retain the calibrated coordinate frame across that replacement too.
        crossing = self.active or mission.leg >= windows[0]
        fresh = self.vio is not None and 0 <= now - self.vio[0] <= self.MAX_AGE_S
        px4_fresh = px4_received is not None and 0 <= now - px4_received <= .25
        if not self.active and not crossing:
            still = (vehicle.landed is True or
                     (vehicle.vel is not None and math.hypot(*vehicle.vel[:2]) < .08
                      and abs(vehicle.vel[2]) < .08))
            if (fresh and px4_fresh and vehicle.pos_ok and still
                    and self.vio[1] != self.calibrated_stamp):
                self.samples.append((now, vehicle.pos[2] - self.vio[2]))
                while self.samples and now - self.samples[0][0] > self.CALIBRATION_S:
                    self.samples.popleft()
                self.offset = sum(value for _, value in self.samples) / len(self.samples)
                self.calibrated_stamp = self.vio[1]
            return vehicle
        if not self.active and crossing:
            if not fresh or self.offset is None:
                if not self.missing_logged:
                    self.warn('!!!!!!!!!! WINDOW VIO HEIGHT UNAVAILABLE: no fresh calibrated VIO; '
                              'keeping existing PX4 height control without an added wait !!!!!!!!!!')
                    self.missing_logged = True
                return vehicle
            self.active = True
            self.warn(f'!!!!!!!!!! WINDOW VIO HEIGHT ACTIVE: fixed NED offset {self.offset:+.3f} m '
                      f'from {len(self.samples)} steady samples; retaining VIO through pad approach; '
                      'height target unchanged !!!!!!!!!!')
        if fresh:
            down = self.vio[2] + self.offset
            source = 'vio'
        else:
            # Bridge a dropout with PX4 *changes*, preserving the calibrated frame.
            # Never jump back to PX4's absolute, potentially corrupted altitude.
            down = self.last_down + vehicle.pos[2] - self.last_px4_down
            source = 'px4_delta'
        if source != self.source and self.active:
            if source == 'px4_delta':
                self.warn('!!!!!!!!!! WINDOW VIO HEIGHT STALE: temporarily following PX4 height changes '
                          'from last VIO height; no added mission wait !!!!!!!!!!')
            elif self.source == 'px4_delta':
                self.warn('!!!!!!!!!! WINDOW VIO HEIGHT RESUMED: using original fixed calibration !!!!!!!!!!')
        self.source = source
        self.last_down, self.last_px4_down = down, vehicle.pos[2]
        return replace(vehicle, pos=(*vehicle.pos[:2], down))

    def status(self):
        return dict(source=self.source, active=self.active, offset_ned_m=self.offset,
                    calibration_samples=len(self.samples), down_m=self.last_down,
                    return_offset_ned_m=self.return_offset)
