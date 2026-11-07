"""Native PX4 transit followed by bounded, explicitly separated response trials.

No ROS, simulator truth, gain changes, or second controller. The protocol uses
course ENU; the adapter's commands and telemetry use PX4 local NED. A validated
localizer affine transform is frozen exactly once at the lane handoff. Stationary
yaw uses the existing native position controller; translation is velocity-only.
"""
from dataclasses import replace
import math

import numpy as np

from coursenav.drift import DriftEKF
from coursenav.navplan import course_to_ned
from . import core as C
from .diagnostic_protocol import STATIONARY_PHASES, TuningController, TuningObservation


# A land-state heartbeat is slower than flight-control telemetry. Run 005's
# recording contains 26 samples / 33.42 wall seconds (mean period 1.285 s);
# runs 003/004 average 1.109/1.116 s. Policy: tolerate two mean periods,
# rounded UP to a tenth: ceil(2 * (33.42 / 26) * 10) / 10 = 2.6 s.
# This is NOT a measured maximum gap or publish-on-change assumption. Keep a
# finite deadline in every clock; position/status/attitude limits stay intact.
# See docs/RUN005_DIAGNOSTIC_REFUSAL.md; callbacks now record actual intervals.
LAND_HEARTBEAT_MAX_AGE_S = 2.6
POSITION_HOLD_PHASES = STATIONARY_PHASES


class DiagnosticMission(C.Mission):
    def __init__(self, params, config):
        self.controller = TuningController(config)
        self.config = self.controller.config
        if len(self.config['candidates']) != 1:
            raise ValueError('native diagnostic requires one candidate per run')
        for key in ('landed_z_m', 'declared_heading_rad'):
            if key not in config or not math.isfinite(float(config[key])):
                raise ValueError(f'diagnostic requires finite {key}')
        if not 0 <= float(config['landed_z_m']) <= .4:
            raise ValueError('invalid landed body height')
        estimate = float(config.get('estimated_total_s', math.inf))
        if not math.isfinite(estimate) or estimate > self.config['time_limit_s']:
            raise ValueError('complete diagnostic estimate exceeds time budget or is missing')
        params = replace(params, track_mode='path', speed_mps=min(params.speed_mps, .15),
                         tunnel_mode=False, blocked_hold_s=0., correction_max_m=0.,
                         correction_rate_mps=0., blend_turn_deg=0.)
        super().__init__(params)
        self.stage = 'transit'
        self.samples = {}
        self.source_stamps = {}
        self.px4_clock_offset = None
        self.frame_fault = None
        self.launch_resets = None
        self.launch_velocity_resets = None
        self.drift_resets = None
        self.total_started = None
        self._wall_now = 0.
        self._depth_wall = None
        self.frozen_transform = None
        self.protocol_reason = ''
        self.landing_confirmed = False
        self.yaw_rate_enu = None
        self.yaw_rate_received = None
        self.protocol_tick = None
        self.stationary_anchor = None

    @property
    def accepts_outbound_paths(self):
        return self.stage == 'transit'

    def observe(self, name, stamp, now, wall_now, **data):
        if name not in ('position', 'status', 'land', 'attitude'):
            return False
        if not all(math.isfinite(x) for x in (stamp, now, wall_now)) or stamp <= 0:
            self.samples.pop(name, None)
            return False
        previous = self.source_stamps.get(name)
        if previous is not None and stamp <= previous:
            if stamp < previous:
                self.frame_fault = f'{name} source timestamp regressed'
            return False
        self.source_stamps[name] = stamp
        offset = now-stamp
        self.px4_clock_offset = offset if self.px4_clock_offset is None else min(offset, self.px4_clock_offset)
        old = self.samples.get(name)
        self.samples[name] = dict(stamp=stamp, now=now, wall=wall_now, **data)
        if name == 'position':
            # The PX4 heading is a world-frame yaw; body gyro.z is not generally
            # its derivative. Raw-source time differences need no ROS clock
            # offset. NED and ENU yaw have opposite sign, including wrapped turns.
            self.yaw_rate_enu = self.yaw_rate_received = None
            heading = data.get('heading')
            old_heading = None if old is None else old.get('heading')
            if (heading is not None and old_heading is not None
                    and math.isfinite(heading) and math.isfinite(old_heading)
                    and 0 < stamp-old['stamp'] <= .3):
                delta = (heading-old_heading+math.pi) % (2*math.pi)-math.pi
                self.yaw_rate_enu = -delta/(stamp-old['stamp'])
                self.yaw_rate_received = (now, wall_now)
        if name == 'position' and self.launch_resets is not None and data.get('resets') != self.launch_resets:
            self.frame_fault = 'PX4 local frame reset during diagnostic'
        if (name == 'position' and self.launch_velocity_resets is not None
                and data.get('velocity_resets') != self.launch_velocity_resets):
            self.frame_fault = 'PX4 velocity reset during diagnostic'
        return True

    def on_imu(self, stamp, now, wall_now, angular_velocity=(), linear_acceleration=()):
        try:
            vectors = np.asarray([angular_velocity, linear_acceleration], float)
        except (ValueError, TypeError):
            self.samples.pop('imu', None)
            return False
        if vectors.shape != (2, 3) or not np.isfinite(vectors).all():
            self.samples.pop('imu', None)
            return False
        return self._observe_ros('imu', stamp, now, wall_now)

    def _observe_ros(self, name, stamp, now, wall_now, **data):
        age = .3 if name == 'imu' else 1.
        if not all(math.isfinite(x) for x in (stamp, now, wall_now)) or stamp <= 0:
            self.samples.pop(name, None)
            return False
        previous = self.source_stamps.get(name)
        if previous is not None and stamp <= previous:
            if stamp < previous:
                self.frame_fault = f'{name} source timestamp regressed'
            return False
        if not -.05 <= now-stamp <= age:
            self.samples.pop(name, None)
            return False
        self.source_stamps[name] = stamp
        self.samples[name] = dict(stamp=stamp, now=now, wall=wall_now, **data)
        return True

    def on_drift(self, data, now, wall_now=None):
        wall_now = self._wall_now if wall_now is None else wall_now
        try:
            stamp = float(data['stamp'])
            vectors = np.asarray([data['delta'], data['nominal'], data['fixture']], float)
            if vectors.shape != (3, 3) or not np.isfinite(vectors).all():
                raise ValueError('invalid drift transform')
            sigma = float(data['sigma_xy'])
            resets = data['resets']
            if (not math.isfinite(sigma) or not 0 <= sigma <= .10 or data.get('accepted', 0) < 1
                    or not isinstance(resets, int)):
                raise ValueError('drift uncertainty, time, or continuity invalid')
            if self.drift_resets is not None and resets != self.drift_resets:
                self.frame_fault = 'localizer reset during diagnostic'
            if self.drift_resets is None:
                self.drift_resets = resets
            return self._observe_ros('drift', stamp, now, wall_now, data=dict(data))
        except (ValueError, TypeError, KeyError):
            self.samples.pop('drift', None)
            return False

    def on_depth(self, *args, **kwargs):
        before = self.depth_t
        accepted = super().on_depth(*args, **kwargs)
        if accepted and (before is None or self.depth_t > before):
            self._depth_wall = self._wall_now
        return accepted

    def abort_diagnostic(self, reason, now=None):
        now = self.t_last if now is None else now
        now = 0. if now is None else now
        self.controller.abort(now, reason)
        self.request_land(f'diagnostic fault: {reason}')

    def on_path(self, *args, **kwargs):
        return super().on_path(*args, **kwargs) if self.accepts_outbound_paths else False

    def _valid_floor(self):
        if self.stage == 'protocol':
            return self.p.min_valid_frac, 'min_valid_frac'
        return super()._valid_floor()

    def _depth_silent_s(self, now):
        if self.stage == 'protocol':
            return math.inf if self.depth_t is None else now-self.depth_t
        return super()._depth_silent_s(now)

    def _fresh(self, name, now, age):
        row = self.samples.get(name)
        if row is None:
            raise ValueError(f'{name} missing or stale (no accepted sample; limit {age:.3f}s)')
        ros_age, wall_age = now-row['now'], self._wall_now-row['wall']
        if not 0 <= ros_age <= age or not 0 <= wall_age <= age:
            raise ValueError(f'{name} missing or stale (ROS age {ros_age:.3f}s, '
                             f'wall age {wall_age:.3f}s, limit {age:.3f}s)')
        if name in ('imu', 'drift'):
            if not -.05 <= now-row['stamp'] <= age:
                raise ValueError(f'{name} capture stale or future')
        else:
            newest = max(t for k, t in self.source_stamps.items() if k in ('position', 'status', 'land', 'attitude'))
            if newest-row['stamp'] > age or now-row['stamp']-self.px4_clock_offset > age:
                raise ValueError(f'{name} PX4 capture delayed or incoherent '
                                 f'(source lag {newest-row["stamp"]:.3f}s, '
                                 f'offset-adjusted age {now-row["stamp"]-self.px4_clock_offset:.3f}s, '
                                 f'limit {age:.3f}s)')
        return row

    def _authority(self, now, v):
        if self.frame_fault:
            raise ValueError(self.frame_fault)
        for name, age in [('position', .5), ('status', 1.),
                          ('land', LAND_HEARTBEAT_MAX_AGE_S), ('attitude', .5)]:
            self._fresh(name, now, age)
        if (v.pos is None or v.vel is None or v.heading is None or v.q is None or not v.pos_ok
                or not all(math.isfinite(x) for x in (*v.pos, *v.vel, *v.q, v.heading))
                or not .9 <= math.hypot(*v.q) <= 1.1):
            raise ValueError('PX4 position, velocity, heading or attitude invalid')
        resets = self.samples['position'].get('resets')
        if resets is None or len(resets) != 3 or not all(isinstance(x, int) and 0 <= x <= 255 for x in resets):
            raise ValueError('PX4 reset history unavailable')
        velocity_resets = self.samples['position'].get('velocity_resets')
        if (velocity_resets is None or len(velocity_resets) != 2
                or not all(isinstance(x, int) and 0 <= x <= 255 for x in velocity_resets)):
            raise ValueError('PX4 velocity reset history unavailable')
        uncertainty = (self.samples['position'].get('eph', math.nan),
                       self.samples['position'].get('epv', math.nan))
        # Existing native ReturnMission readiness bounds, not fitted/tuned here.
        if (not all(math.isfinite(x) and x >= 0 for x in uncertainty)
                or uncertainty[0] > .15 or uncertainty[1] > .10):
            raise ValueError('PX4 position uncertainty too large or unavailable')

    def _next_leg(self, now, out, v, blended=False):
        if self.stage == 'protocol':
            # Core increments both before this hook, including hold_s == 0.
            self.leg = max(0, self.leg-1)
            self.legs_done = max(0, self.legs_done-1)
            return  # Base hold supervision still runs; it must not advance/land.
        if self.leg >= len(self.targets):
            try:
                self._freeze(now, v)
            except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
                self.controller.abort(now, str(exc))
                self._land(now, out, f'diagnostic handoff refused: {exc}')
                self.stage = 'landing'
                return
            self.stage = 'protocol'
            self.leg = max(0, len(self.targets)-1)
            self._set('hold', now, out, 'diagnostic protocol: frozen course transform')
            return
        super()._next_leg(now, out, v, blended)

    def _freeze(self, now, v):
        data = self._fresh('drift', now, 1.)['data']
        ekf = DriftEKF.from_state(data['delta'], data['nominal'])
        basis = course_to_ned(ekf.to_nominal([[0., 0.], [1., 0.], [0., 1.]], data['nominal']),
                              data['fixture'], self.config['declared_heading_rad'])
        origin, matrix = basis[0], (basis[1:]-basis[0]).T
        if not np.allclose(matrix.T@matrix, np.eye(2), atol=1e-8) or not np.isfinite(origin).all():
            raise ValueError('invalid course-to-NED affine transform')
        self.frozen_transform = (origin.copy(), matrix.copy(), np.linalg.inv(matrix))
        pos, _, _ = self._course_vehicle(v)
        lane = self.controller.lane
        if math.dist(pos[:2], lane['start_xy']) >= .12 or abs(pos[2]-lane['height_m']) >= .10:
            self.frozen_transform = None
            raise ValueError('measured lane start or altitude not reached')

    def _course_vehicle(self, v):
        origin, matrix, inverse = self.frozen_transform
        xy = inverse@(np.asarray(v.pos[:2])-origin)
        vel = inverse@np.asarray(v.vel[:2])
        direction = inverse@np.array([math.cos(v.heading), math.sin(v.heading)])
        z = self.home[2]+float(self.config['landed_z_m'])-v.pos[2]
        return (*xy, z), math.atan2(direction[1], direction[0]), (*vel, -v.vel[2])

    def _begin_protocol_tick(self, now, v):
        transform = self.frozen_transform
        self.protocol_tick = dict(
            t_ros=now, evaluated_phase=self.controller.state,
            depth_stamp_s=self.depth_t,
            depth_ros_age_s=None if self.depth_t is None else now-self.depth_t,
            depth_watchdog_wall_age_s=None if self._depth_wall is None else self._wall_now-self._depth_wall,
            frame_available=self.frame is not None,
            depth_valid_fraction=None if self.frame is None else getattr(self.frame, 'valid_frac', None),
            position_source_stamp_s=self.samples.get('position', {}).get('stamp'),
            position_ned_m=None if v.pos is None else list(v.pos),
            velocity_ned_mps=None if v.vel is None else list(v.vel),
            heading_ned_rad=v.heading,
            frozen_course_to_ned=None if transform is None else dict(
                origin_ne_m=transform[0].tolist(), matrix=transform[1].tolist()),
            coverage_raw=None, corridor_rays=None, coverage_effective=None,
            front_clearance_m=None, turn_clearance_m=None,
            observation=None, turn_guard=None, refusal=None)

    def _protocol(self, now, v, out):
        if self.protocol_tick is None:  # The handoff can happen inside this tick.
            self._begin_protocol_tick(now, v)
        evidence = self.protocol_tick
        if (not self.depth_fresh(now) or self._depth_wall is None
                or not 0 <= self._wall_now-self._depth_wall <= self.p.depth_stale_s or self.frame is None):
            raise ValueError('fresh depth unavailable during protocol')
        minimum = self.config['minimum_clearance_m']
        coverage, rays = self.frame.corridor_coverage(0., minimum, .25, 1.)
        evidence.update(coverage_raw=float(coverage), corridor_rays=int(rays))
        coverage = float(coverage) if rays >= 12 else 0.
        evidence['coverage_effective'] = coverage
        if not math.isfinite(coverage) or coverage < .6:
            raise ValueError(f'protocol corridor not sufficiently observed: coverage={coverage:.6f} '
                             f'(min 0.60), rays={rays} (min 12)')
        pts = np.asarray(self.depth_pts)
        front_indices = np.flatnonzero((pts[:, 0] > 0) & (np.abs(pts[:, 1]) < minimum) & (np.abs(pts[:, 2]) < .25))
        hits = pts[front_indices, 0]
        clearance = float(np.min(hits)) if len(hits) else math.inf
        # Translation consumes axial stopping distance in its swept corridor.
        # In-place rotation consumes an obstacle-free DISK: its certificate is
        # radial, so an off-axis point must be compared using hypot(x,y), not x.
        band_indices = np.flatnonzero(np.abs(pts[:, 2]) < .25)
        band = pts[band_indices]
        turn_clearance = float(np.linalg.norm(band[:, :2], axis=1).min()) if len(band) else math.inf
        # Evidence only: retain the actual minimum and its nearby ordered
        # samples so an isolated stereo return can be distinguished from a
        # supported surface. The control decision still uses the same minimum.
        nearest = np.argsort(np.linalg.norm(band[:, :2], axis=1), kind='stable')[:8]
        front_index = None if not len(hits) else int(front_indices[np.argmin(hits)])
        evidence.update(front_clearance_m=clearance, turn_clearance_m=turn_clearance,
            point_frame='level heading: forward/right/down, body origin, metres',
            turn_band_point_count=len(band), front_corridor_point_count=len(hits),
            nearest_turn_points=[dict(index=int(band_indices[i]), point_m=band[i].tolist()) for i in nearest],
            nearest_front_point=None if front_index is None else dict(index=front_index, point_m=pts[front_index].tolist()))
        position, yaw, velocity = self._course_vehicle(v)
        rate_time = self.yaw_rate_received
        yaw_rate = (self.yaw_rate_enu if rate_time is not None and
                    0 <= now-rate_time[0] <= .3 and 0 <= self._wall_now-rate_time[1] <= .3 else None)
        observation = TuningObservation(now, position, yaw, velocity, clearance, coverage,
                                       transit_complete=True, offboard=v.nav_state == C.NAV_OFFBOARD,
                                       tracking_ok=self.health.bad_for(now) == 0.,
                                       yaw_rate=yaw_rate,
                                       turn_clearance=turn_clearance)
        evidence['observation'] = dict(vars(observation))
        command = self.controller.step(observation)
        evidence['turn_guard'] = self.controller.turn_guard
        if command.state == 'FAULT':
            raise ValueError(command.note)
        if command.state == 'DONE':
            self.controller.reason = 'diagnostic complete; native landing requested'
            self.stage = 'landing'
            self._land(now, out, 'route complete')
            return False
        native_hold = self.config.get('stationkeeping_policy') == 'native_position_hold_v1'
        if native_hold and command.state in POSITION_HOLD_PHASES:
            if self.stationary_anchor is None:
                # The endpoint has already passed arrival/geometry checks.
                # Hold that accepted measured position, not the outstanding
                # transit waypoint: run008 inherited 0.099 m of position error
                # and accelerated toward it during the stationary experiment.
                # Capture ONCE per angular group; recapturing every sample would
                # follow drift. Existing disk/altitude and motion guards remain.
                anchor = np.asarray(v.pos[:3], dtype=float)
                origin, _, inverse = self.frozen_transform
                anchor_xy = inverse@(anchor[:2]-origin)
                anchor_z = self.home[2]+float(self.config['landed_z_m'])-anchor[2]
                # Validate the TARGET against the same certified turning disk
                # and altitude envelope as the measured pose. A valid current
                # pose must remain inside the certified zone at each capture.
                guard = self.controller._turn_guard_snapshot(observation)
                offset = float(np.linalg.norm(anchor_xy-guard['centre_xy_m']))
                evidence['anchor_validation'] = dict(position_ned_m=anchor.tolist(),
                    position_course_m=[*anchor_xy, anchor_z], offset_m=offset,
                    zone=guard['zone'], remaining_radius_m=guard['radius_m']-offset)
                if (not np.isfinite(anchor).all() or offset > guard['offset_limit_m']
                        or guard['radius_m']-offset < guard['required_clearance_min_m']
                        or abs(anchor_z-self.controller.lane['height_m']) > .15):
                    raise ValueError('position hold target outside certified turn/altitude envelope')
                if self.config['turn_excursion_reference'] == 'captured_hold':
                    self.controller.bind_stationary_anchor(anchor_xy)
                self.stationary_anchor = tuple(anchor)
            out.position_hold = (*self.stationary_anchor, -command.yaw_rate)
            evidence['control_mode'] = 'position_hold'
            evidence['position_target_ned_m'] = list(self.stationary_anchor)
        else:
            self.stationary_anchor = None
            if self.config['turn_excursion_reference'] == 'captured_hold':
                self.controller.clear_stationary_anchor()
            ne = self.frozen_transform[1]@np.array([command.vx, command.vy])
            out.velocity_only = (float(ne[0]), float(ne[1]), -command.vz, -command.yaw_rate)
            evidence['control_mode'] = 'velocity_only'
            evidence['position_target_ned_m'] = None
        # Course ENU and NED have opposite handedness in the horizontal plane.
        out.velocity = out.accel = None
        out.events.extend(f'diagnostic {e["previous"]} -> {e["phase"]}: {e["reason"]}'
                          for e in self.controller.events)
        self.controller.events.clear()
        return True

    def step(self, now, v, wall_now=None):
        self._wall_now = now if wall_now is None else wall_now
        self.protocol_tick = None
        if self.stage == 'protocol':
            self._begin_protocol_tick(now, v)
        if self.phase == 'done':
            return super().step(now, v)
        try:
            self._authority(now, v)
        except (ValueError, TypeError) as exc:
            if self.protocol_tick is not None:
                self.protocol_tick['refusal'] = str(exc)
            if self.home is not None or v.armed is True:
                if self.controller.state != 'DONE':
                    self.controller.abort(now, str(exc))
                out = C.Output(self.phase)
                self._end(now, out, f'yielded: {exc}')
                self.stage = 'finished'
                out.phase = self.phase
                return out
            self.t_started = now if self.t_started is None else self.t_started
            self.why = str(exc)
            out = C.Output(self.phase)
            if now-self.t_started > self.p.wait_s:
                self.controller.abort(now, str(exc))
                self._end(now, out, f'refused: diagnostic inputs {exc}')
                out.phase = self.phase
            return out
        try:
            if self.phase not in ('land', 'done'):
                self._fresh('imu', now, .3)
            if (self.phase not in ('land', 'done') and self.total_started is not None
                    and now-self.total_started >= self.config['time_limit_s']):
                raise ValueError('complete diagnostic time budget reached')
        except ValueError as exc:
            if self.protocol_tick is not None:
                self.protocol_tick['refusal'] = str(exc)
            if self.home is None and v.armed is False:
                self.t_started = now if self.t_started is None else self.t_started
                self.why = str(exc)
                if now-self.t_started <= self.p.wait_s:
                    return C.Output(self.phase)
            self.controller.abort(now, str(exc))
            self.request_land(f'diagnostic fault: {exc}')
        # Protocol phases may hold position or command velocity; native landing
        # streams a position hold until AUTO_LAND acknowledges. Keep that
        # fallback at the CURRENT accepted pose, never the old lane-start
        # waypoint. Do this before base.step handles external/health faults too.
        if self.stage == 'protocol' and self.phase != 'retreat':
            self.sp = [*v.pos, v.heading]
            self.v_cmd = self.a_cmd = 0.
            self.v_vec = self.a_vec = None
            # Prevent hold's bookkeeping from accumulating fictitious legs.
            self.t_phase = now
        out = super().step(now, v)
        if (self.protocol_tick is not None and self.phase in ('land', 'done')
                and self.controller.state != 'DONE' and self.protocol_tick['refusal'] is None):
            # Native supervision may preempt the experiment before its sensor
            # checks run (operator request, authority takeover, health fault).
            self.protocol_tick['refusal'] = self.land_reason or self.result or 'native supervision stopped protocol'
        if self.total_started is None and self.home is not None:
            self.total_started = now
            self.controller.started = now
            self.launch_resets = tuple(self.samples['position']['resets'])
            self.launch_velocity_resets = tuple(self.samples['position']['velocity_resets'])
        if self.stage == 'protocol' and self.phase == 'hold':
            try:
                if not self._protocol(now, v, out):
                    return super().step(now, v)
            except (ValueError, TypeError, IndexError) as exc:
                self.protocol_tick['refusal'] = str(exc)
                self.controller.abort(now, str(exc))
                self.stage = 'landing'
                # The first protocol tick may reject the retained target before
                # the usual protocol-step landing fallback has run.
                self.sp = [*v.pos, v.heading]
                self.v_cmd = self.a_cmd = 0.
                self.v_vec = self.a_vec = None
                self.request_land(f'diagnostic fault: {exc}')
                return super().step(now, v)
        if self.phase == 'retreat' and self.stage == 'protocol':
            # a fault landing over a column that is not clear: the protocol is over, the base mission
            # retraces the flown path to a clear column and lands there (core.Mission._land)
            if self.controller.state not in ('DONE', 'FAULT'):
                self.controller.abort(now, self.retreat['why'] if self.retreat else 'retreat to a clear column')
            self.stage = 'landing'
        if self.phase in ('land', 'done'):
            if self.controller.state not in ('DONE', 'FAULT'):
                self.controller.abort(now, self.land_reason or self.result or 'native transit stopped')
            self.stage = 'finished' if self.phase == 'done' else 'landing'
            if self.phase == 'done' and self.result == 'complete':
                self.landing_confirmed = v.landed is True and v.armed is False
                if not self.landing_confirmed:
                    self.result = 'diagnostic landing unconfirmed'
        return out

    def diagnostic_status(self):
        return dict(self.controller.status(), native_stage=self.stage, native_result=self.result,
                    landing_confirmed=self.landing_confirmed,
                    land_max_age_s=LAND_HEARTBEAT_MAX_AGE_S)

    def status(self, now):
        return dict(super().status(now), diagnostic=self.diagnostic_status())
