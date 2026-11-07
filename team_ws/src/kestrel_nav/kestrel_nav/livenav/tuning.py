"""Bounded velocity-response experiment. Pure; no ROS, arming, or gain writes.

Coordinates are course ENU. TRANSIT is executed by the map navigator, never by
this controller. Lane endpoints must be measured open turning-zone centres.
"""
from dataclasses import dataclass, replace
import math

import numpy as np


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
    def stopping_reserve(speed):
        return .30 + max(speed, 0.) * .8 + max(speed, 0.) ** 2 / (2 * .15)

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

    def _yaw_step(self, obs, cmd, xy, speed):
        """Excite yaw only in the FIRST lane's certified open start disk."""
        now = obs.now
        if obs.yaw_rate is None or not math.isfinite(obs.yaw_rate):
            return self.abort(now, 'observed world yaw rate required for rotational diagnostic')
        offset = float(np.linalg.norm(xy-np.asarray(self.lane['start_xy'])))
        clear = self.lane['start_turn_radius_m']-offset
        if (self.candidate != 0 or offset > .15 or speed > .06 or clear < .40
                or (obs.turn_clearance if obs.turn_clearance is not None else obs.front_clearance)+1e-6 < clear
                or obs.coverage < .6):
            return self.abort(now, 'rotational diagnostic left observed open start zone')
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
        if self.state in ('FAULT', 'DONE'):
            return TuningCommand(state=self.state, note=self.reason)
        if not all(math.isfinite(v) for v in (now, *obs.position, obs.yaw, *obs.velocity, obs.coverage)):
            return self.abort(now, 'invalid observation')
        if math.isnan(obs.front_clearance) or obs.front_clearance < 0:
            return self.abort(now, 'invalid front clearance')
        if obs.turn_clearance is not None and (math.isnan(obs.turn_clearance) or obs.turn_clearance < 0):
            return self.abort(now, 'invalid radial turn clearance')
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
        if (along < -.15 or along > length+.15 or abs(cross) > lane['half_width_m']-.40
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
        elif self.state == 'TURN':
            if speed > .06:
                return self.abort(now, 'translation during in-place turn')
            centre = b if self.after_turn == 'RETURN' else a
            radius = lane['end_turn_radius_m'] if self.after_turn == 'RETURN' else lane['start_turn_radius_m']
            offset = float(np.linalg.norm(xy-centre))
            required_clearance = radius-offset
            if (offset > .15 or required_clearance < .40
                    or (obs.turn_clearance if obs.turn_clearance is not None else obs.front_clearance)+1e-6 < required_clearance
                    or obs.coverage < .6):
                return self.abort(now, 'open turning-zone clearance not observed')
            err = wrap(heading-obs.yaw)
            if abs(err) < math.radians(6):
                if self._settled(now, speed):
                    self.peak_speed = 0.
                    self._set(self.after_turn, now)
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
                        needed = 2*self.target_speed + self.stopping_reserve(self.target_speed) + .4
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
            if obs.front_clearance < self.stopping_reserve(max(speed, self.target_speed if self.state == 'PULSE' else .1)):
                return self.abort(now, 'insufficient observed stopping corridor')
            self.peak_speed = max(self.peak_speed, speed)
            if self.state == 'PULSE' and (now-self.phase_t >= 2. or remaining <= self.stopping_reserve(max(speed, self.target_speed))+.4):
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
                    axis='yaw' if yaw else 'translation', rotation_trial=self.rotation_trial,
                    target_yaw_rate_rps=self.target_yaw_rate,
                    target_speed=self.target_speed, split='train' if (self.rotation_trial if yaw else self.trial) % 2 == 0 else 'validation',
                    trial_id=(f'{self.lane["name"]}:yaw:{self.rotation_trial}' if yaw else f'{self.lane["name"]}:{self.trial}'),
                    reason=self.reason, completed=self.completed, completed_yaw=self.completed_yaw)


from ..coursenav import planner
from .mission import LiveMission, Command
from .replanner import Replanner, ReplanParams, goals_from_plan


class TuningProgram(LiveMission):
    """Existing map navigator for transit, independent bounded experiment at the lane."""
    use_pad_detector = False

    def __init__(self, config, mp, fp, course_file=None):
        self.controller = TuningController(config)
        self.config = self.controller.config
        if self.config.get('synthetic') is True:
            raise ValueError('synthetic rehearsal geometry is forbidden in the physical adapter')
        self.transit_index = 0
        self.imu_ok = False
        self.clamped = False
        self.frame = None
        self._velocity_sample = None
        self.measured_velocity = np.zeros(3)
        self.measured_yaw_rate = None  # supplied from accepted odometry capture timestamps
        sx, sy, yaw = self.config['start_course']
        mp = replace(mp, start_xy=(sx, sy), start_yaw_deg=yaw,
                     cruise_z_m=self.controller.lane['height_m'], time_limit_s=self.config['time_limit_s'])
        fp = replace(fp, v_cruise=.15, v_gate=.15, v_floor=.03, hold_backoff_mps=0.)
        super().__init__(mp, replan_params=ReplanParams(arrive_m=.08, arrive_gate_m=.08),
                         follow_params=fp, course_file=course_file)
        if abs(self.p.cruise_z_m-float(self.doc['cruise_z_m'])) > 1e-6:
            raise ValueError('measured lane height must match course map planning-band height')
        for lane in self.config['candidates']:
            if not planner.segment_free(self.grid.prior_blocked, lane['start_xy'], lane['end_xy']):
                raise ValueError(f"{lane['name']}: measured lane conflicts with inflated published map")
            if planner.segment_clearance(self.grid.obst, self.grid.prior_clearance,
                                         lane['start_xy'], lane['end_xy'])+1e-6 < lane['half_width_m']:
                raise ValueError(f"{lane['name']}: declared full lane width conflicts with published map")
            for end in ('start', 'end'):
                if self.grid.clearance_at(lane[end+'_xy']) < lane[end+'_turn_radius_m']:
                    raise ValueError(f"{lane['name']}: {end} open turning zone conflicts with map")
        transit_estimate = self.plan['length_m']/.15 + 5
        for a, b in zip(self.config['candidates'], self.config['candidates'][1:]):
            route = planner.astar(self.grid.prior_blocked, self.grid.prior_clearance, a['start_xy'], b['start_xy'])
            transit_estimate += float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())/.15+5
        if transit_estimate+self.config['estimated_lane_s'] > self.config['time_limit_s']:
            raise ValueError('measured lane trials plus map transit estimate exceed time budget')
        self.estimated_total_s = transit_estimate+self.config['estimated_lane_s']

    def _plan_course(self):
        sx, sy = self.p.start_xy
        lane = self.controller.lane
        plan = planner.plan(self.doc, self.grid.obst, self.grid.prior_blocked, self.grid.prior_clearance,
                            start_xy=(sx, sy), goals=[dict(name='diagnostic_lane_start', xy=lane['start_xy'])],
                            cmap=self.cmap)
        plan['start_course'] = (sx, sy, math.radians(self.p.start_yaw_deg))
        return plan

    def abort(self, now, reason):
        if self.controller.state == 'DONE':
            # Pilot takeover/landing after completion is expected. The adapter
            # still suppresses commands through InputSafety; preserve the result.
            return Command(state='DONE', note=self.controller.reason)
        self.controller.abort(now, reason)
        return super().abort(now, reason)

    def step(self, obs):
        if self.fault or self.controller.state == 'FAULT':
            return self.abort(obs.now, self.fault or self.controller.reason)
        if not self.imu_ok or self.clamped:
            return self.abort(obs.now, 'IMU missing/stale or organizer clamped command')
        if self.controller.started is None:
            self.controller.started = obs.now
        if obs.now-self.controller.started > self.config['time_limit_s']:
            return self.abort(obs.now, 'diagnostic time limit')
        if self.T is not None:
            course = self.odom_to_course(obs.odom_xyz[:2], obs.odom_yaw)
            position = np.array([*course[:2], obs.odom_xyz[2]+self.p.landed_z_m])
            if self._velocity_sample is not None:
                t, previous = self._velocity_sample
                dt = obs.now-t
                if not 0 < dt <= .5 or np.linalg.norm(position-previous) > .25:
                    return self.abort(obs.now, 'odometry discontinuity during diagnostic')
                measured = (position-previous)/dt
                self.measured_velocity = .5*self.measured_velocity+.5*measured
            self._velocity_sample = (obs.now, position)
        else:
            course, position = None, None
        if self.controller.state == 'TRANSIT':
            if self.transit_index != self.controller.candidate:
                self.transit_index = self.controller.candidate
                self.p.start_xy, self.p.start_yaw_deg = tuple(course[:2]), math.degrees(course[2])
                self.plan = self._plan_course()
                self.goals = goals_from_plan(self.plan)
                self.replanner = Replanner(self.grid, self.goals, self.replan_params)
                self.state = 'FLY'
            lane = self.controller.lane
            arrived = (course is not None and math.dist(course[:2], lane['start_xy']) < .12
                       and abs(obs.odom_xyz[2]-self.z_odom_cruise) < .10 and self.state not in ('WAIT', 'CLIMB'))
            if not arrived:
                result = super().step(obs)
                if result.state in ('HOLD', 'FAULT', 'PAUSED', 'DESCEND'):
                    return self.abort(obs.now, 'conservative transit stopped: '+result.note)
                # Existing follower already caps geometry, depth, heading and corners.
                speed = math.hypot(result.vx, result.vy)
                if speed > .150001:
                    return self.abort(obs.now, 'transit speed cap exceeded')
                return result
        if self.frame is None:
            return self.abort(obs.now, 'depth unavailable for lane diagnostic')
        coverage, rays = self.frame.corridor_coverage(0., .40, .25, 1.)
        if rays < 12:
            coverage = 0.
        pts = np.asarray(obs.pts_level) if obs.pts_level is not None else np.empty((0, 3))
        hits = pts[(pts[:, 0] > 0) & (np.abs(pts[:, 1]) < .40) & (np.abs(pts[:, 2]) < .25), 0]
        clearance = float(np.min(hits)) if len(hits) else math.inf
        band = pts[np.isfinite(pts).all(axis=1) & (np.abs(pts[:, 2]) < .25)]
        radial = float(np.min(np.linalg.norm(band[:, :2], axis=1))) if len(band) else math.inf
        tuning = self.controller.step(TuningObservation(obs.now, tuple(position), course[2],
                                      tuple(self.measured_velocity), clearance, coverage,
                                      imu_ok=self.imu_ok, offboard=obs.offboard_active,
                                      transit_complete=True, clamped=self.clamped,
                                      yaw_rate=self.measured_yaw_rate, turn_clearance=radial))
        self.state = tuning.state
        if tuning.state == 'FAULT':
            return self.abort(obs.now, tuning.note)
        vx, vy = self.course_vec_to_odom(tuning.vx, tuning.vy)
        return Command(vx=vx, vy=vy, vz=tuning.vz, yaw_rate=tuning.yaw_rate,
                       state=tuning.state, note=tuning.note, speed=math.hypot(vx, vy))

    def status(self):
        return dict(super().status(), tuning=self.controller.status(), estimated_total_s=self.estimated_total_s)
