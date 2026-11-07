"""Automatic, single-attempt return after a completed course landing.

This extends the native-PX4 mission, not the organizer hardware adapter. Failed
checks before the second takeoff keep the completed run on the pad. Once airborne,
the existing controller's obstacle/health/landing policies still apply.
"""
from dataclasses import replace
import math

import numpy as np

from . import core as C


class ReturnMission(C.Mission):
    def __init__(self, params, geometry):
        super().__init__(params)
        self.geometry = geometry
        self.stage = 'outbound'
        self.launch_pose = None
        self.launch_resets = None
        self.launch_localizer_resets = None
        self.run_started = None
        self.return_reason = 'pending outbound landing'
        self.return_plan = None
        self.return_started = False
        self.return_airborne = False
        self.pad_landed_at = None
        self.no_return = None
        self.samples = {}
        self.source_stamps = {}
        self.px4_clock_offset = None
        self.frame_fault = None
        self.outbound_legs_done = 0
        self._wall_now = 0.

    @property
    def accepts_outbound_paths(self):
        return self.stage == 'outbound'

    def observe(self, name, stamp, now, wall_now, **data):
        """PX4 clocks need not equal ROS time: require advancement and bounded receipt age.

        Localizer additionally carries its ROS capture stamp. Duplicate heartbeats
        never refresh the data watchdog; regressions invalidate the run's frame.
        """
        if not math.isfinite(stamp) or stamp <= 0:
            self.samples.pop(name, None)
            return False
        previous = self.source_stamps.get(name)
        if previous is not None and stamp <= previous:
            if stamp < previous:
                self.samples.pop(name, None)
                self.frame_fault = f'{name} timestamp regressed'
            return False
        self.source_stamps[name] = stamp
        if name != 'localizer':
            offset = now-stamp
            self.px4_clock_offset = offset if self.px4_clock_offset is None else min(self.px4_clock_offset, offset)
        self.samples[name] = dict(stamp=stamp, now=now, wall=wall_now, **data)
        if self.launch_resets is not None and name == 'position' and data.get('resets') != self.launch_resets:
            self.frame_fault = 'PX4 local frame reset since original launch'
        if self.launch_localizer_resets is not None and name == 'localizer' and data.get('resets') != self.launch_localizer_resets:
            self.frame_fault = 'localizer reset since original launch'
        return True

    def request_land(self, why='operator'):
        self.no_return = self.no_return or f'landing requested: {why}'
        super().request_land(why)

    def on_path(self, *args, **kwargs):
        return super().on_path(*args, **kwargs) if self.accepts_outbound_paths else False

    def on_correction(self, *args, **kwargs):
        if self.accepts_outbound_paths:
            return super().on_correction(*args, **kwargs)

    def _fresh(self, name, now, age):
        row = self.samples.get(name)
        if row is None or not 0 <= now-row['now'] <= age or not 0 <= self._wall_now-row['wall'] <= age:
            raise ValueError(f'{name} missing or stale')
        if name != 'localizer':
            newest = max(stamp for topic, stamp in self.source_stamps.items() if topic != 'localizer')
            if newest-row['stamp'] > age or now-row['stamp']-self.px4_clock_offset > age:
                raise ValueError(f'{name} PX4 capture delayed or incoherent')
        return row

    def _check(self, now, v, *, ground):
        if self.no_return or self.frame_fault:
            raise ValueError(self.no_return or self.frame_fault)
        if self.launch_pose is None or self.run_started is None or self.launch_resets is None:
            raise ValueError('original launch pose/reset history unavailable')
        for name, age in [('position', .5), ('status', 1.), ('land', 1.), ('attitude', .5), ('battery', 2.)]:
            self._fresh(name, now, age)
        if not v.pos_ok or v.pos is None or v.heading is None or not all(math.isfinite(x) for x in (*v.pos, v.heading)):
            raise ValueError('PX4 position or heading invalid')
        if v.vel is None or not all(math.isfinite(x) for x in v.vel):
            raise ValueError('PX4 velocity unavailable')
        if v.q is None or not all(math.isfinite(x) for x in v.q) or not .9 <= math.hypot(*v.q) <= 1.1:
            raise ValueError('attitude invalid')
        pos = self.samples['position']
        eph, epv = pos.get('eph'), pos.get('epv')
        if any(x is None or not math.isfinite(x) or x < 0 for x in (eph, epv)) or eph > .15 or epv > .10:
            raise ValueError('PX4 position uncertainty too large or unavailable')
        if pos.get('resets') != self.launch_resets:
            raise ValueError('PX4 frame changed')
        if not self.health.arm_ok(now)[0] or not self.depth_ready_for_arm(now):
            raise ValueError('estimator/depth not continuously healthy')
        bat = self.samples['battery']
        remaining = bat.get('remaining', math.nan)
        if (bat.get('connected') is not True or not math.isfinite(remaining) or not .30 <= remaining <= 1
                or bat.get('warning') != 0 or bat.get('faults') != 0 or bat.get('powering_off') is not False):
            raise ValueError('battery not ready for return')
        if ground:
            if 1-2*(v.q[1]**2+v.q[2]**2) < math.cos(math.radians(10)):
                raise ValueError('aircraft not level on landing pad')
            if v.landed is not True or np.linalg.norm(v.vel) > .08:
                raise ValueError('pad touchdown not confirmed and settled')
            allowed = (C.NAV_AUTO_LAND,) if self.stage == 'qualify' else (C.NAV_AUTO_LAND, C.NAV_OFFBOARD)
            if v.nav_state not in allowed:
                raise ValueError('pilot/failsafe changed mode on pad')
            if self.stage == 'qualify' and v.armed is not False:
                raise ValueError('pad disarm not confirmed')
            loc = self._fresh('localizer', now, 1.)
            if not -.05 <= now-loc['stamp'] <= 1.:
                raise ValueError('localizer capture stale or future')
            if self.launch_localizer_resets is None or loc.get('resets') != self.launch_localizer_resets:
                raise ValueError('localizer continuity unavailable')
            sigma = loc.get('sigma_xy', math.nan)
            xy = (loc.get('x', math.nan), loc.get('y', math.nan))
            if (loc.get('ready') is not True or loc.get('updates', 0) < 30
                    or not math.isfinite(sigma) or not 0 <= sigma <= .10
                    or not all(math.isfinite(x) for x in xy)):
                raise ValueError('localizer not converged')
            self.return_plan = self.geometry.plan(self.launch_pose, v.pos, xy, sigma, eph, epv,
                                                   self.p.climb_rate_mps, min(self.p.speed_mps, .3))
            if now-self.run_started + self.return_plan.estimated_s > 180:
                raise ValueError('insufficient time for the complete return plus reserve')
            seconds = bat.get('time_remaining_s', math.nan)
            if math.isfinite(seconds) and seconds < self.return_plan.estimated_s+10:
                raise ValueError('battery time estimate cannot cover return')
        elif now-self.run_started >= 170:
            raise ValueError('return time reserve reached')

    def _finish_on_pad(self, now, why, v):
        self.stage = 'finished'
        self.return_reason = f'skipped: {why}'
        out = C.Output('done')
        # If the arm acknowledgement arrived while a check failed, only a normal
        # checked disarm is permitted on the confirmed ground. No retry/relaunch.
        if self.return_started and v.armed is True and v.landed is True and v.nav_state == C.NAV_OFFBOARD:
            out.commands.append('disarm')
        self._end(now, out, 'complete')
        out.phase = self.phase
        out.events.append(f'return {self.return_reason}; retaining completed outbound landing')
        return out

    def _begin_return(self, now, v):
        plan = self.return_plan
        dn, de = np.asarray(plan.target_ne)-np.asarray(v.pos[:2])
        c, s = math.cos(v.heading), math.sin(v.heading)
        params = replace(self.p, route=[(dn*c+de*s, dn*s-de*c)],
                         legs=[dict(name='original_launch', mode='return', hold=True)],
                         height_m=plan.height_m, track_mode='pose', speed_mps=min(self.p.speed_mps, .3),
                         correction_max_m=0., correction_rate_mps=0., accept_armed_start=False,
                         tunnel_mode=False, blocked_hold_s=0., blend_turn_deg=0., hover_s=0.,
                         arrive_m=min(self.p.arrive_m, .10), arrive_speed_mps=.08,
                         wait_s=8., arm_wait_s=8.)
        retained = {k: getattr(self, k) for k in ('health', 'depth_ready_since', 'depth_t', 'frame',
                                                 'depth_pts', 'last_valid_frac')}
        C.Mission.__init__(self, params)
        for k, value in retained.items():
            setattr(self, k, value)
        self.stage = 'return'
        self.return_started = True
        self.return_reason = 'checks passed; returning to original launch'
        return C.Output('wait', events=[self.return_reason])

    def step(self, now, v, wall_now=None):
        self._wall_now = now if wall_now is None else wall_now
        if self.stage == 'finished':
            return super().step(now, v)
        if self.stage == 'qualify':
            try:
                self._check(now, v, ground=True)
            except (ValueError, TypeError, KeyError) as exc:
                return self._finish_on_pad(now, str(exc), v)
            if now-self.pad_landed_at >= 2.:
                return self._begin_return(now, v)
            return C.Output('return_check')
        if self.stage == 'return':
            if v.landed is False and self.offboard_seen:
                self.return_airborne = True
            try:
                self._check(now, v, ground=not self.return_airborne)
            except (ValueError, TypeError, KeyError) as exc:
                self.return_reason = f'aborted: {exc}'
                if not self.return_airborne and v.landed is True and v.armed is False:
                    return self._finish_on_pad(now, str(exc), v)
                # No stale-state commands or takeover overrides. Healthy live state
                # can use the original controller's landing response.
                authority_fresh = True
                try:
                    for name, age in [('position', .5), ('status', 1.), ('land', 1.)]:
                        self._fresh(name, now, age)
                except ValueError:
                    authority_fresh = False
                if not authority_fresh or self.frame_fault or not v.pos_ok:
                    out = C.Output('done')
                    self._end(now, out, f'yielded: {exc}')
                    out.phase = self.phase
                    self.stage = 'finished'
                    return out
                self.request_land(str(exc))
        out = super().step(now, v)
        if self.stage == 'return' and self.home is not None and out.phase != 'done':
            if self.phase in C.AIRBORNE and self.offboard_seen:
                # Land detection can stay true for the first centimetres of climb.
                # Ground settling checks end when PX4 accepts the second takeoff.
                self.return_airborne = True
            # Core may capture the second takeoff pose a tick after planning. The
            # destination and absolute cruise altitude remain the original run's.
            self.targets = [tuple(self.launch_pose[:2])]
            self.p.height_m = self.home[2] - (self.launch_pose[2] + self.geometry.spec['landed_z_m']
                                             - self.return_plan.course_height_m)
        if self.stage == 'outbound':
            if self.launch_pose is None and self.home is not None and self.phase == 'climb' and self.offboard_seen:
                self.launch_pose = tuple(self.home)
                resets = self.samples.get('position', {}).get('resets')
                self.launch_resets = (resets if resets is not None and len(resets) == 3
                                      and all(isinstance(x, int) and 0 <= x <= 255 for x in resets) else None)
                self.launch_localizer_resets = self.samples.get('localizer', {}).get('resets')
            if self.run_started is None and v.armed is True and v.landed is False:
                self.run_started = now
            if out.phase == 'done' and self.result == 'complete' and self.land_reason == 'route complete':
                self.outbound_legs_done = self.legs_done
                self.stage = 'qualify'
                self.pad_landed_at = now
                self.result = None
                self.phase = 'return_check'
                self.return_reason = 'outbound landing complete; checking return'
                # Qualify immediately: a failure is terminal, not a retry when inputs recover.
                return self.step(now, v, self._wall_now)
            elif out.phase == 'done':
                self.stage = 'finished'
                self.return_reason = f'skipped: outbound {self.result}'
        elif self.stage == 'return' and out.phase == 'done':
            self.stage = 'finished'
            if self.result == 'complete':
                if (v.landed is True and v.armed is False and v.pos is not None
                        and math.dist(v.pos[:2], self.launch_pose[:2]) <= .15):
                    self.return_reason = 'returned and disarmed at original launch'
                else:
                    self.result = 'return landing unconfirmed'
                    self.return_reason = 'return touchdown/disarm or saved-launch position not confirmed'
        return out

    def status(self, now):
        result = super().status(now)
        result.update(mission_stage=self.stage, return_reason=self.return_reason,
                      return_started=self.return_started, launch_pose=self.launch_pose,
                      outbound_legs_done=self.outbound_legs_done,
                      return_estimated_s=None if self.return_plan is None else self.return_plan.estimated_s)
        return result
