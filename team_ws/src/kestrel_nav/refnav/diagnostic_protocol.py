"""Bounded velocity-response experiment. Pure; no ROS, arming, or gain writes.

Coordinates are course ENU. TRANSIT is executed by the map navigator, never by
this controller. Lane endpoints must be measured open turning-zone centres.
"""
from dataclasses import dataclass, replace
import math

import numpy as np


STATIONARY_PHASES = frozenset(('SETTLE', 'TURN', 'YAW_SETTLE', 'YAW_PULSE', 'YAW_BRAKE'))


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


@dataclass
class TuningObservation:
    now: float
    position: tuple
    yaw: float
    velocity: tuple
    front_clearance: float
    coverage: float
    imu_ok: bool = True
    offboard: bool = True
    transit_complete: bool = False
    clamped: bool = False
    tracking_ok: bool = True
    yaw_rate: float | None = None  # observed ENU heading rate, rad/s; never raw body gyro z
    turn_clearance: float | None = None  # radial obstacle distance for open-disk checks


@dataclass
class TuningCommand:
    vx: float = 0.
    vy: float = 0.
    vz: float = 0.
    yaw_rate: float = 0.
    state: str = ''
    note: str = ''


def validate_config(config):
    """No guessed lane, tube, or turn geometry is supplied by defaults."""
    c = dict(config)
    try:
        minimum = float(c.get('minimum_clearance_m', .40))
    except (TypeError, ValueError) as exc:
        raise ValueError('minimum_clearance_m must be finite and at least 0.40 m') from exc
    if isinstance(c.get('minimum_clearance_m'), bool) or not math.isfinite(minimum) or minimum < .40:
        raise ValueError('minimum_clearance_m must be finite and at least 0.40 m')
    c['minimum_clearance_m'] = minimum
    policy = c.get('stationkeeping_policy', 'velocity_only')
    if not isinstance(policy, str) or policy not in ('velocity_only', 'native_position_hold_v1'):
        raise ValueError('stationkeeping_policy must be velocity_only or native_position_hold_v1')
    c['stationkeeping_policy'] = policy
    reference = c.get('turn_excursion_reference', 'nominal_zone')
    if not isinstance(reference, str) or reference not in ('nominal_zone', 'captured_hold'):
        raise ValueError('turn_excursion_reference must be nominal_zone or captured_hold')
    if reference == 'captured_hold' and policy != 'native_position_hold_v1':
        raise ValueError('captured_hold excursion requires native_position_hold_v1')
    c['turn_excursion_reference'] = reference
    start = np.asarray(c.get('start_course', []), float)
    if start.shape != (3,) or not np.isfinite(start).all():
        raise ValueError('measured start_course [x,y,yaw_deg] is required')
    lanes = c.get('candidates')
    if not isinstance(lanes, list) or not lanes:
        raise ValueError('at least one measured candidate lane is required')
    names = set()
    for lane in lanes:
        name = lane.get('name')
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('candidate names must be nonempty and unique')
        names.add(name)
        if lane.get('kind') not in ('open', 'tube'):
            raise ValueError('candidate kind must be open or tube')
        a, b = np.asarray(lane.get('start_xy', []), float), np.asarray(lane.get('end_xy', []), float)
        if a.shape != (2,) or b.shape != (2,) or not np.isfinite([a, b]).all():
            raise ValueError('measured lane start_xy/end_xy required')
        length = float(np.linalg.norm(b-a))
        if not 2. <= length <= 8.:
            raise ValueError('lane length must be 2–8 m, including braking reserve')
        for key, lower, upper in [('height_m', .5, 1.2), ('half_width_m', .5, 3.),
                                  ('start_turn_radius_m', .65, 3.), ('end_turn_radius_m', .65, 3.)]:
            value = float(lane.get(key, math.nan))
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f'{name}: measured {key} must be in [{lower}, {upper}]')
        if any(float(lane[k]) < minimum for k in
               ('half_width_m', 'start_turn_radius_m', 'end_turn_radius_m')):
            raise ValueError(f'{name}: measured lane/turn geometry cannot contain minimum_clearance_m')
        if 'validation_offset_m' in lane:
            offset = float(lane['validation_offset_m'])
            if not math.isfinite(offset) or not 0 < offset < length:
                raise ValueError('validation_offset_m must be a finite location inside the lane')
        if lane.get('kind') == 'tube' and (lane.get('start_open') is not True or lane.get('end_open') is not True):
            raise ValueError('tube endpoints must explicitly be measured open turning zones')
    if len({float(x['height_m']) for x in lanes}) != 1:
        raise ValueError('one measured flight height per run; no unverified vertical transfers')
    speeds = list(c.get('speeds_mps', [.1]))
    if (not speeds or not all(math.isfinite(x) and .05 <= x <= .4 for x in speeds)
            or speeds[0] > .1 or any(b <= a or b-a > .100001 for a, b in zip(speeds, speeds[1:]))):
        raise ValueError('speeds must start <=0.10, increase by <=0.10, and stay <=0.40 m/s')
    c['speeds_mps'] = speeds
    rates = list(c.get('yaw_rates_rps', [.15, .25]))
    if (len(rates) > 2 or not all(math.isfinite(x) and .05 <= x <= .25 for x in rates)
            or (rates and rates[0] > .15)
            or any(b <= a or b-a > .100001 for a, b in zip(rates, rates[1:]))):
        raise ValueError('yaw rates must start <=0.15, increase by <=0.10, stay <=0.25 rad/s; at most two rates')
    c['yaw_rates_rps'] = rates
    c['estimated_yaw_s'] = 13.*len(rates)  # once, in the first lane's open start zone
    c['time_limit_s'] = float(c.get('time_limit_s', 160.))
    if not math.isfinite(c['time_limit_s']) or not 10 <= c['time_limit_s'] <= 170:
        raise ValueError('time_limit_s must be 10–170 seconds')
    pairs = len(speeds)
    estimate = sum(2*float(np.linalg.norm(np.asarray(x['end_xy'])-x['start_xy']))/.1*pairs
                   + (2*pairs-1)*math.pi/.25 + pairs*14 for x in lanes)
    estimate += c['estimated_yaw_s']
    if policy == 'native_position_hold_v1':
        # One position-to-velocity release per translation trial pair. This is
        # the known one-second settling dwell, not an upper bound on settling.
        estimate += len(lanes)*pairs
    c['estimated_lane_s'] = estimate
    if estimate+10 > c['time_limit_s']:
        raise ValueError('paired trial schedule cannot fit time budget; select a shorter lane/smaller schedule')
    return c


class TuningController:
    """Open-start yaw trials, translation pulse/brake, forward-camera return.

    A fresh successful stop is required before the next trial. No recovery or
    automatic retry follows a fault. Remaining-distance checks use deliberately
    conservative initial delay/deceleration bounds, not unqualified fit output.
    """
    def __init__(self, config):
        self.config = validate_config(config)
        self.candidate = 0
        self.trial = 0
        self.state = 'TRANSIT'
        self.reason = ''
        self.started = self.phase_t = self.settled_since = None
        self.after_brake = 'CREEP'
        self.after_turn = 'PULSE'
        self.brake_start = None
        self.validation_offset = None
        self.peak_speed = 0.
        self.completed = []
        self.completed_yaw = []
        self.rotation_trial = 0
        self.yaw_started = self.yaw_reference = self.yaw_brake_start = None
        self.peak_yaw_rate = 0.
        self.events = []
        self.turn_guard = None  # immutable snapshot of the last evaluated observation
        self.stationary_anchor_xy = None

    @property
    def lane(self):
        return self.config['candidates'][self.candidate]

    @property
    def target_speed(self):
        return self.config['speeds_mps'][self.trial // 2]

    def _set(self, state, now, reason=''):
        self.events.append(dict(t_ros=now, phase=state, previous=self.state, reason=reason,
                                candidate=self.lane['name'], trial=self.trial,
                                axis='yaw' if state.startswith('YAW_') else 'translation',
                                rotation_trial=self.rotation_trial))
        self.state, self.phase_t, self.settled_since = state, now, None
        self.reason = reason

    def abort(self, now, reason):
        if self.state != 'FAULT':
            self._set('FAULT', now, reason)
        return TuningCommand(state='FAULT', note=self.reason)

    @staticmethod
    def stopping_travel(speed):
        # Existing conservative model assumptions, NOT measured/validated bounds.
        return max(speed, 0.) * .8 + max(speed, 0.) ** 2 / (2 * .15)

    @staticmethod
    def stopping_reserve(speed):
        return .30 + TuningController.stopping_travel(speed)

    def _settled(self, now, speed):
        if speed > .035:
            self.settled_since = None
        elif self.settled_since is None:
            self.settled_since = now
        return self.settled_since is not None and now-self.settled_since >= 1.

    @property
    def target_yaw_rate(self):
        rates = self.config['yaw_rates_rps']
        if not rates:
            return 0.
        magnitude = rates[min(self.rotation_trial//2, len(rates)-1)]
        return magnitude if self.rotation_trial % 2 == 0 else -magnitude

    def _turn_zone(self):
        end = self.state in ('SETTLE', 'TURN') and self.after_turn == 'RETURN'
        zone = 'end' if end else 'start'
        return zone, self.lane[f'{zone}_xy'], self.lane[f'{zone}_turn_radius_m']

    def bind_stationary_anchor(self, xy):
        """Bind the actual native hold once; callers cannot absorb later drift."""
        anchor = np.asarray(xy, dtype=float)
        if anchor.shape != (2,) or not np.isfinite(anchor).all():
            raise ValueError('stationary anchor must be finite XY')
        if self.config['turn_excursion_reference'] != 'captured_hold':
            raise ValueError('stationary anchor requires captured_hold reference')
        if self.state not in STATIONARY_PHASES:
            raise ValueError('stationary anchor may only bind in an angular phase group')
        if self.stationary_anchor_xy is not None:
            if not np.array_equal(anchor, self.stationary_anchor_xy):
                raise ValueError('stationary anchor changed during angular phase group')
            return
        _, centre, radius = self._turn_zone()
        offset = float(np.linalg.norm(anchor-centre))
        # Preserve the existing capture envelope. This is not permission to
        # move a nominal zone to an arbitrary pose or shrink its obstacle test.
        if offset > .15 or radius-offset < self.config['minimum_clearance_m']:
            raise ValueError('stationary anchor outside certified turn envelope')
        self.stationary_anchor_xy = tuple(float(v) for v in anchor)

    def clear_stationary_anchor(self):
        if self.state in STATIONARY_PHASES:
            raise ValueError('cannot clear stationary anchor during angular phase group')
        self.stationary_anchor_xy = None

    def _turn_guard_snapshot(self, obs):
        """Separate physical map clearance from excursion about the held pose.

        For a certified disk B(c,R), clearance at p is at least R-|p-c| by
        the triangle inequality. Keep this ORIGINAL certificate and observed
        depth requirement. Moving its centre to the anchor would change them.
        The .15 m experiment bound instead measures |p-a| for the immutable
        held point a. Both constraints apply; no distance threshold is raised.
        Legacy velocity-only protocols continue to measure offset from c.
        """
        yaw = self.state.startswith('YAW_')
        enforced = self.state == 'TURN' or yaw
        zone, centre, radius = self._turn_zone()
        xy = np.asarray(obs.position[:2])
        offset = float(np.linalg.norm(xy-centre))
        required = radius-offset
        reference = self.config['turn_excursion_reference']
        captured = reference == 'captured_hold'
        anchor = self.stationary_anchor_xy if captured else None
        excursion = (None if anchor is None else float(np.linalg.norm(xy-anchor))) if captured else offset
        observed = obs.front_clearance if obs.turn_clearance is None else obs.turn_clearance
        speed = float(np.linalg.norm(obs.velocity[:2]))
        checks = ({'anchor_missing': anchor is None and enforced,
                   'excursion': excursion is not None and excursion > .15}
                  if captured else {'offset': offset > .15})
        minimum = self.config['minimum_clearance_m']
        travel = self.stopping_travel(speed) if captured else None
        projected_excursion = None if excursion is None or travel is None else excursion+travel
        # Only a fixed native hold supplies restoring XY authority. Require the
        # whole assumed stopping travel to fit BOTH the original safe disk and
        # the immutable experiment excursion envelope, in any travel direction.
        # The legacy .30 m static stopping buffer is replaced here by the
        # configured body clearance (>=.40), never counted as travel distance.
        if captured:
            checks.update(stopping_margin=required < minimum+travel,
                          stopping_observed_clearance=observed+1e-6 < minimum+travel,
                          stopping_excursion=projected_excursion is not None and projected_excursion > .15)
        checks.update(margin=required < minimum,
                      measured_clearance=observed+1e-6 < required,
                      coverage=obs.coverage < .6)
        return dict(t_ros=obs.now, evaluated_phase=self.state, enforced=enforced,
                    zone=zone, centre_xy_m=list(centre), radius_m=radius,
                    offset_m=offset, offset_limit_m=.15, offset_limit_enforced=not captured,
                    excursion_m=excursion, excursion_limit_m=.15, excursion_reference=reference,
                    anchor_xy_m=None if anchor is None else list(anchor),
                    anchor_offset_m=None if anchor is None else float(np.linalg.norm(np.asarray(anchor)-centre)),
                    required_clearance_m=required, required_clearance_min_m=minimum,
                    observed_clearance_m=observed, comparison_epsilon_m=1e-6,
                    clearance_source='front_fallback' if obs.turn_clearance is None else 'radial',
                    clearance_slack_m=observed+1e-6-required,
                    coverage=obs.coverage, coverage_min=.6, speed_mps=speed, speed_max_mps=.06,
                    speed_limit_enforced=not captured,
                    stopping_guard_policy='fixed_anchor_containment_v1' if captured else 'legacy_speed_limit',
                    stopping_travel_m=travel, projected_excursion_m=projected_excursion,
                    stopping_required_clearance_m=None if travel is None else minimum+travel,
                    stopping_delay_s=.8, stopping_deceleration_mps2=.15,
                    stopping_model_source='existing stopping_reserve assumptions; not measured or validated bounds',
                    translation_during_turn=speed > .06,
                    wrong_yaw_candidate=yaw and self.candidate != 0,
                    failed_checks=[name for name, failed in checks.items() if failed])

    def _turn_refusal(self, reason):
        g = self.turn_guard
        failed = list(g['failed_checks'])
        if g['translation_during_turn'] and g['speed_limit_enforced']:
            failed.append('speed')
        if g['wrong_yaw_candidate']:
            failed.append('candidate')
        if 'anchor_missing' in failed:
            reason = 'stationary experiment anchor missing'
        elif failed == ['excursion']:
            reason = 'stationary experiment excursion exceeded'
        excursion_note = (f'excursion={g["excursion_m"]:.6f}m '
                          f'(max {g["excursion_limit_m"]:.2f}, {g["excursion_reference"]}); '
                          if g['excursion_m'] is not None else 'excursion=unavailable; ')
        return (f'{reason}: failed={",".join(failed)}; '
                f'nominal_offset={g["offset_m"]:.6f}m; {excursion_note}'
                f'radius={g["radius_m"]:.6f}m; required={g["required_clearance_m"]:.6f}m '
                f'(min {g["required_clearance_min_m"]:.2f}); '
                f'observed_{g["clearance_source"]}={g["observed_clearance_m"]:.6f}m '
                f'(epsilon {g["comparison_epsilon_m"]:g}); '
                f'coverage={g["coverage"]:.6f} (min {g["coverage_min"]:.2f}); '
                f'speed={g["speed_mps"]:.6f}m/s '
                f'({"max" if g["speed_limit_enforced"] else "legacy reference, not enforced"} '
                f'{g["speed_max_mps"]:.2f}); stopping_policy={g["stopping_guard_policy"]}')

    def _yaw_step(self, obs, cmd, xy, speed):
        """Excite yaw only in the FIRST lane's certified open start disk."""
        now = obs.now
        if obs.yaw_rate is None or not math.isfinite(obs.yaw_rate):
            return self.abort(now, 'observed world yaw rate required for rotational diagnostic')
        if (self.candidate != 0 or (speed > .06 and self.turn_guard['speed_limit_enforced'])
                or self.turn_guard['failed_checks']):
            return self.abort(now, self._turn_refusal('rotational diagnostic left observed open start zone'))
        if self.yaw_started is None:
            self.yaw_started, self.yaw_reference = now, obs.yaw
        if (now-self.yaw_started > self.config['estimated_yaw_s']
                or abs(wrap(obs.yaw-self.yaw_reference)) > .8 or abs(obs.yaw_rate) > .6):
            return self.abort(now, 'rotational time/angle/rate envelope exceeded')
        angular_speed = abs(obs.yaw_rate)
        self.peak_yaw_rate = max(self.peak_yaw_rate, angular_speed)
        if self.state == 'YAW_PULSE':
            if now-self.phase_t >= 2.:
                self.yaw_brake_start = obs.yaw
                self._set('YAW_BRAKE', now)
            else:
                cmd.yaw_rate = self.target_yaw_rate
        else:
            settled = self._settled(now, max(speed, angular_speed))
            if settled and self.state == 'YAW_SETTLE':
                self.peak_yaw_rate = 0.
                self._set('YAW_PULSE', now)
            elif settled:
                stopped_angle = abs(wrap(obs.yaw-self.yaw_brake_start))
                reserve = .05 + .8*self.peak_yaw_rate + self.peak_yaw_rate**2/(2*.5)
                if self.peak_yaw_rate < .6*abs(self.target_yaw_rate) or stopped_angle > reserve:
                    return self.abort(now, 'rotational excitation or braking quality failed')
                self.completed_yaw.append(dict(candidate=self.lane['name'], trial=self.rotation_trial,
                                               rate=self.target_yaw_rate, peak_rate=self.peak_yaw_rate,
                                               stopping_angle=stopped_angle))
                if self.rotation_trial+1 == 2*len(self.config['yaw_rates_rps']):
                    self.after_turn = 'PULSE'
                    self._set('SETTLE', now, 'rotation measurements complete; face lane before translation')
                else:
                    self.rotation_trial += 1
                    self._set('YAW_SETTLE', now)
            elif now-self.phase_t > 6:
                return self.abort(now, 'rotational motion did not settle')
        cmd.state, cmd.note = self.state, self.reason
        return cmd

    def step(self, obs):
        now = obs.now
        self.turn_guard = None
        if self.state in ('FAULT', 'DONE'):
            return TuningCommand(state=self.state, note=self.reason)
        if not all(math.isfinite(v) for v in (now, *obs.position, obs.yaw, *obs.velocity, obs.coverage)):
            return self.abort(now, 'invalid observation')
        if math.isnan(obs.front_clearance) or obs.front_clearance < 0:
            return self.abort(now, 'invalid front clearance')
        if obs.turn_clearance is not None and (math.isnan(obs.turn_clearance) or obs.turn_clearance < 0):
            return self.abort(now, 'invalid radial turn clearance')
        self.turn_guard = self._turn_guard_snapshot(obs)
        if self.started is None:
            self.started = self.phase_t = now
        if not obs.offboard or not obs.tracking_ok or not obs.imu_ok or obs.clamped:
            return self.abort(now, 'authority, tracking, IMU or interface clamp failed')
        if now-self.started > self.config['time_limit_s']:
            return self.abort(now, 'diagnostic time limit')
        if self.state == 'TRANSIT':
            if obs.transit_complete:
                self.after_turn = ('YAW_SETTLE' if self.candidate == 0 and self.config['yaw_rates_rps']
                                   and not self.completed_yaw else 'PULSE')
                self._set('SETTLE', now)
            return TuningCommand(state=self.state)
        lane = self.lane
        a, b = np.asarray(lane['start_xy'], float), np.asarray(lane['end_xy'], float)
        axis = b-a
        length = float(np.linalg.norm(axis))
        u = axis/length
        lateral = np.array([-u[1], u[0]])
        xy = np.asarray(obs.position[:2])
        along, cross = float((xy-a)@u), float((xy-a)@lateral)
        speed = float(np.linalg.norm(obs.velocity[:2]))
        if (along < -.15 or along > length+.15 or abs(cross) > lane['half_width_m']-self.config['minimum_clearance_m']
                or abs(obs.position[2]-lane['height_m']) > .15 or speed > .65):
            return self.abort(now, 'measured lane/altitude/speed envelope exceeded')
        forward = self.state != 'RETURN' and not (self.state == 'TURN' and self.after_turn == 'RETURN')
        heading = math.atan2(u[1], u[0]) + (0 if forward else math.pi)
        cmd = TuningCommand(state=self.state)
        cmd.vz = float(np.clip(lane['height_m']-obs.position[2], -.1, .1))
        if self.state.startswith('YAW_'):
            return self._yaw_step(obs, cmd, xy, speed)
        if self.state == 'SETTLE':
            if self._settled(now, speed):
                self._set('TURN', now)
            elif now-self.phase_t > 8:
                return self.abort(now, 'failed to settle before turn')
        elif self.state == 'TRANSFER_SETTLE':
            if self._settled(now, speed):
                self.trial += 1
                self.peak_speed = 0.
                self._set('PULSE', now, 'held-out pulse after observed forward transfer and settling')
            elif now-self.phase_t > 6:
                return self.abort(now, 'held-out transfer failed to settle')
        elif self.state == 'VELOCITY_SETTLE':
            # The native adapter has released XY position hold. Observe a fresh
            # stop under the same zero-velocity mode used for the ensuing fit.
            if now-self.phase_t > 6:
                return self.abort(now, 'position-hold release failed to settle')
            if self._settled(now, speed):
                self.peak_speed = 0.
                self._set('PULSE', now)
        elif self.state == 'TURN':
            if speed > .06 and self.turn_guard['speed_limit_enforced']:
                return self.abort(now, self._turn_refusal('translation during in-place turn'))
            if self.turn_guard['failed_checks']:
                return self.abort(now, self._turn_refusal('open turning-zone clearance not observed'))
            err = wrap(heading-obs.yaw)
            if abs(err) < math.radians(6):
                if self._settled(now, speed):
                    self.peak_speed = 0.
                    next_state = ('VELOCITY_SETTLE' if self.after_turn == 'PULSE'
                                  and self.config['stationkeeping_policy'] == 'native_position_hold_v1'
                                  else self.after_turn)
                    self._set(next_state, now)
            else:
                self.settled_since = None
                cmd.yaw_rate = float(np.clip(err, -.25, .25))
            if now-self.phase_t > 20:
                return self.abort(now, 'turn timeout')
        elif self.state == 'BRAKE':
            if self._settled(now, speed):
                if self.after_brake == 'CREEP':
                    travel = float(np.linalg.norm(xy-self.brake_start))
                    if self.peak_speed < .6*self.target_speed or travel > self.stopping_reserve(self.peak_speed):
                        return self.abort(now, 'trial excitation or braking quality failed')
                    self.completed.append(dict(candidate=lane['name'], trial=self.trial,
                                               speed=self.target_speed, peak_speed=self.peak_speed,
                                               stopping_distance=travel))
                    if self.trial % 2 == 0:
                        needed = 2*self.target_speed + self.stopping_reserve(self.target_speed) + self.config['minimum_clearance_m']
                        if length-along <= needed:
                            return self.abort(now, 'insufficient lane for held-out pulse')
                        self.validation_offset = min(length*.5, length-needed-.10)
                        requested = lane.get('validation_offset_m', self.validation_offset)
                        if not math.isfinite(requested) or not along+.10 < requested <= length-needed-.05:
                            return self.abort(now, 'held-out transfer position cannot fit lane reserve')
                        self.validation_offset = float(requested)
                        self._set('TRANSFER', now, 'moving forward to distinct held-out location')
                    else:
                        self._set('CREEP', now)
                else:
                    self.after_turn = self.after_brake
                    self._set('SETTLE', now)
                    if self.after_brake == 'PULSE':
                        self.trial += 1
                        if self.trial >= 2*len(self.config['speeds_mps']):
                            self.trial = 0
                            if self.candidate+1 >= len(self.config['candidates']):
                                self._set('DONE', now, 'diagnostic complete; pilot lands')
                            else:
                                self.candidate += 1
                                self._set('TRANSIT', now)
            elif now-self.phase_t > 6:
                return self.abort(now, 'braking did not settle within six seconds')
        else:  # PULSE / CREEP / RETURN: every translation faces the observed corridor
            returning = self.state == 'RETURN'
            direction = -u if returning else u
            remaining = (self.validation_offset-along if self.state == 'TRANSFER'
                         else along if returning else length-along)
            err = wrap(math.atan2(direction[1], direction[0])-obs.yaw)
            if abs(err) > math.radians(12) or obs.coverage < .6:
                return self.abort(now, 'translation corridor not forward and observed')
            stop_speed = max(speed, self.target_speed if self.state == 'PULSE' else .1)
            observed_reserve = max(self.stopping_reserve(stop_speed),
                                   self.config['minimum_clearance_m']+self.stopping_travel(stop_speed))
            if obs.front_clearance < observed_reserve:
                return self.abort(now, 'insufficient observed stopping corridor')
            self.peak_speed = max(self.peak_speed, speed)
            if self.state == 'PULSE' and (now-self.phase_t >= 2. or remaining <= self.stopping_reserve(max(speed, self.target_speed))+self.config['minimum_clearance_m']):
                self.brake_start = xy.copy()
                self.after_brake = 'CREEP'
                self._set('BRAKE', now)
            elif self.state == 'TRANSFER' and remaining <= .035:
                self._set('TRANSFER_SETTLE', now)
            elif self.state in ('CREEP', 'RETURN') and remaining <= .07:
                self.after_brake = 'PULSE' if returning else 'RETURN'
                self._set('BRAKE', now)
            else:
                target = self.target_speed if self.state == 'PULSE' else min(.1, max(0., remaining)*.5)
                if target <= 0:
                    return self.abort(now, 'lane endpoint overshot')
                velocity = direction*target - lateral*float(np.clip(cross*.25, -.025, .025))
                cmd.vx, cmd.vy = map(float, velocity)
                cmd.yaw_rate = float(np.clip(err*.5, -.15, .15))
        cmd.state, cmd.note = self.state, self.reason
        return cmd

    def status(self):
        yaw = self.state.startswith('YAW_')
        return dict(phase=self.state, candidate=self.lane['name'], trial=self.trial,
                    stationkeeping_policy=self.config['stationkeeping_policy'],
                    turn_excursion_reference=self.config['turn_excursion_reference'],
                    stationary_anchor_xy=None if self.stationary_anchor_xy is None else list(self.stationary_anchor_xy),
                    axis='yaw' if yaw else 'translation', rotation_trial=self.rotation_trial,
                    target_yaw_rate_rps=self.target_yaw_rate,
                    target_speed=self.target_speed, split='train' if (self.rotation_trial if yaw else self.trial) % 2 == 0 else 'validation',
                    trial_id=(f'{self.lane["name"]}:yaw:{self.rotation_trial}' if yaw else f'{self.lane["name"]}:{self.trial}'),
                    reason=self.reason, completed=self.completed, completed_yaw=self.completed_yaw)
