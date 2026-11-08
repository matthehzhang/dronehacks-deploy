"""Pure bounded exploration and stop/replan controller. No ROS, files or clock reads."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .grid import DepthGrid
from .opening import OpeningCenteringController


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class MissionParams:
    cruise_height_m: float = .8
    v_cruise_mps: float = .4
    v_gap_mps: float = .20
    acceleration_mps2: float = .35
    braking_mps2: float = .5
    command_delay_s: float = .25
    jerk_mps3: float = 1.0
    yaw_rate_dps: float = 25.
    align_tolerance_deg: float = 8.
    gap_extra_m: float = .18
    arrival_m: float = .12
    scan_timeout_s: float = 24.
    time_limit_s: float = 180.
    max_frontier_visits: int = 24
    max_blocked_scans: int = 3
    target_fresh_s: float = 2.
    scan_plan_interval_s: float = .5
    frontier_failure_cooldown_s: float = 5.
    max_reset_recoveries: int = 3
    reset_recovery_timeout_s: float = 5.

    def __post_init__(self):
        if self.v_cruise_mps > .6 or self.v_gap_mps > .25:
            raise ValueError('search speed exceeds existing conservative Skybeta envelope')
        if not 0 < self.v_gap_mps <= self.v_cruise_mps or self.braking_mps2 <= 0:
            raise ValueError('invalid speeds/braking')
        if self.command_delay_s < .25 or self.time_limit_s <= 0:
            raise ValueError('invalid delay/time bound')
        if not 1 <= self.max_reset_recoveries <= 3 or not 0 < self.reset_recovery_timeout_s <= 5.:
            raise ValueError('reset recovery must remain bounded to <=3 resets and <=5 seconds')
        if self.scan_plan_interval_s < .5:
            raise ValueError('scan planning evaluations are capped at 2Hz')


@dataclass
class Observation:
    now: float
    position: tuple
    yaw: float
    velocity: tuple = (0., 0., 0.)
    ready: bool = False
    offboard: bool = False
    armed: bool = False
    landed: bool = True
    fault: str = ''
    target_xy: tuple | None = None
    target_stamp: float = -math.inf
    inputs_fresh: bool = True
    opening: object | None = None
    certify_sweep: object | None = None
    slab_ready: bool = True


@dataclass
class Command:
    phase: str
    reason: str
    velocity: tuple | None = None  # ENU x,y,z and ENU yaw rate; None yields authority.


class SearchMission:
    def __init__(self, grid=None, params=None):
        self.grid = grid or DepthGrid()
        self.p = params or MissionParams()
        self.phase, self.reason = 'wait', 'waiting for qualified sensors and pilot'
        self.route, self.route_revision, self.route_index = [], -1, 1
        self.route_goal, self.goal_kind = None, None
        self.visited, self.rejected_goals = [], []
        self.events = []
        self.counts = dict(replans=0, invalidations=0, scans=0, frontiers=0, obstacle_stops=0,
                           scan_plan_checks=0, reset_recoveries=0)
        self.started = None
        self.phase_t = None
        self.hold_xy, self.hold_z = None, None
        self.last_yaw, self.scan_angle = None, 0.
        self.last_t = None
        self.command_v = np.zeros(2)
        self.command_a = np.zeros(2)
        self.blocked_scans = 0
        self.stop_pending = False
        self.last_invalidated_revision = -1
        self.frontier_choices = []
        self.plan_rejections = []
        self.retry_goal = None
        self.failed_frontiers = []
        self.scan_plan_t, self.scan_plan_revision = -math.inf, -1
        self.recovery_started = None
        self.recovery_events = []
        self.centering = OpeningCenteringController()
        self.centering_status = {}
        self.centering_started, self.centering_blocked_since = None, None
        self.opening_cooldowns = []
        self.counts.update(opening_attempts=0, opening_centered=0)

    def transition(self, phase, reason, now):
        self.phase, self.reason, self.phase_t = phase, reason, now
        self.events.append(dict(t=now, phase=phase, reason=reason, revision=self.grid.revision))
        self.events = self.events[-100:]

    def invalidate_route(self, reason, now):
        """CLEAR the route before any failed replan can expose it to the follower."""
        old_goal = None if self.route_goal is None else np.array(self.route_goal)
        self.route = []
        self.route_index = 1
        self.command_v[:] = 0.
        self.command_a[:] = 0.
        self.counts['invalidations'] += 1
        self.counts['obstacle_stops'] += 1
        self.last_invalidated_revision = self.grid.revision
        if old_goal is not None:
            self.retry_goal = (self.goal_kind, old_goal)
        self.route_goal = None
        self.stop_pending = True
        self.transition('plan', reason, now)

    def begin_scan(self, o, reason):
        self.hold_xy = np.asarray(o.position[:2], float)
        self.last_yaw, self.scan_angle = o.yaw, 0.
        self.route = []
        self.command_v[:] = 0.
        self.command_a[:] = 0.
        self.counts['scans'] += 1
        # Preserve the global evaluation rate across view refusals; restarting
        # a scan must not turn the 2Hz planner into a 20Hz reject/retry loop.
        self.scan_plan_revision = -1
        self.transition('scan', reason, o.now)

    def reject_motion_view(self, o, reason):
        """Withdraw a refused route and choose a different observation view."""
        goal = None if self.route_goal is None else np.array(self.route_goal)
        frontier = self.goal_kind == 'frontier'
        self.invalidate_route(reason, o.now)
        if frontier and goal is not None:
            self.failed_frontiers.append(dict(xy=goal,until=o.now+self.p.frontier_failure_cooldown_s))
            self.retry_goal = None
            if not any(np.linalg.norm(goal-old) < .2 for old in self.rejected_goals):
                self.rejected_goals.append(goal)
        self.begin_scan(o, reason+'; observe another direction before a new route')

    def begin_reset_recovery(self, now, position, delta_z_enu, description):
        """Stop before rebuilding spatial data; never restart a yielded mission."""
        if self.grid.launch_volume is not None:
            self.grid.launch_volume.revoke('EKF frame reset: launch assertion withdrawn')
        if self.started is None or self.phase in ('yield', 'exhausted'):
            return False
        if (self.counts['reset_recoveries'] >= self.p.max_reset_recoveries
                or not math.isfinite(delta_z_enu) or self.hold_z is None):
            self.route = []
            self.transition('yield', 'bounded reset-recovery count or altitude validity exceeded', now)
            return False
        self.counts['reset_recoveries'] += 1
        self.recovery_started = now if self.recovery_started is None else self.recovery_started
        self.recovery_events.append(dict(t=now, reason=description, delta_z_enu=delta_z_enu))
        self.hold_z += delta_z_enu
        self.hold_xy = np.asarray(position[:2], float)
        self.route, self.route_goal, self.retry_goal = [], None, None
        self.visited, self.rejected_goals, self.failed_frontiers = [], [], []
        self.frontier_choices, self.plan_rejections = [], []
        self.route_revision, self.route_index = -1, 1
        self.centering.reset()
        self.centering_status, self.opening_cooldowns = {}, []
        self.centering_started, self.centering_blocked_since = None, None
        self.command_v[:], self.command_a[:] = 0., 0.
        self.grid.reset(position[:2], self.hold_z)
        self.stop_pending = False
        self.transition('recover', 'verified small reset: stop, rebuild fresh map and requalify', now)
        return True

    def end_centering(self, o, reason, *, success=False):
        """Hold the reached height, rebuild that slab; never climb back blindly."""
        if o.opening is not None:
            self.opening_cooldowns.append(dict(center=np.asarray(o.opening.center_world),
                                               until=o.now+(20. if success else 5.)))
        self.hold_z = float(o.position[2])
        self.hold_xy = np.asarray(o.position[:2], float)
        self.grid.reset(o.position[:2], self.hold_z)
        self.route, self.route_goal = [], None
        self.visited, self.failed_frontiers, self.rejected_goals = [], [], []
        self.command_v[:], self.command_a[:] = 0., 0.
        self.centering_started, self.centering_blocked_since = None, None
        self.centering.reset()
        self.counts['opening_centered'] += int(success)
        self.transition('rebuild_height', reason, o.now)

    def _opening_worth_centering(self, o):
        opening = o.opening
        if opening is None or not opening.body_fits or not 0 <= o.now-opening.stamp <= .5:
            return False
        if not self.grid.segment_safe(o.position[:2], o.position[:2]):
            return False
        self.opening_cooldowns = [x for x in self.opening_cooldowns if x['until'] > o.now]
        if any(np.linalg.norm(np.asarray(opening.center_world)-x['center']) < .3
               for x in self.opening_cooldowns):
            return False
        delta = np.asarray(opening.center_world)-np.asarray(o.position)
        ahead = float(delta @ np.asarray(opening.normal_world))
        side = abs(float(delta @ np.asarray(opening.right_world)))
        height = abs(float(delta @ np.asarray(opening.up_world)))
        # Only a measured nearby entrance with adequate stand-off is an
        # exploration candidate. Aligned openings already suit normal A*.
        return self.grid.inflation_m+.15 < ahead < 2.5 and (side > .08 or height > .08)

    def _hold(self, o, yaw_rate=0.):
        # Zero ground velocity is the stationary action. A lateral correction
        # toward a retained hold point has no active forward route certificate
        # and can fight a yaw scan or repeatedly invalidate an absent route.
        z = 0. if self.hold_z is None else float(np.clip(self.hold_z - o.position[2], -.25, .25))
        return Command(self.phase, self.reason, (0., 0., z, float(yaw_rate)))

    def _plan(self, o, *, during_scan=False):
        self.route = []  # fail closed even if every candidate A* fails.
        self.plan_rejections = []
        self.frontier_choices = []
        if len(self.rejected_goals) >= self.p.max_frontier_visits:
            self.transition('exhausted', 'bounded rejected observation points reached; hold for pilot', o.now)
            return
        target_valid = o.target_xy is not None and 0 <= o.now - o.target_stamp <= self.p.target_fresh_s
        options = []
        retry = self.retry_goal
        if retry is not None and retry[0] == 'target' and not target_valid:
            self.retry_goal, retry = None, None
        if retry is not None and (retry[0] != 'target' or target_valid):
            options.append(retry)
        if target_valid:
            options.append(('target', np.asarray(o.target_xy, float)))
        # A target in unknown/unreachable space must not suppress exploration.
        self.failed_frontiers = [f for f in self.failed_frontiers if f['until'] > o.now]
        excluded = self.visited + [f['xy'] for f in self.failed_frontiers]
        frontiers = self.grid.frontiers(o.position[:2], excluded, heading=o.yaw,
                                       speed_mps=self.p.v_cruise_mps)
        self.frontier_choices = getattr(self.grid, 'frontier_details', []) or [
            dict(score=float(score), xy=point.tolist()) for score, point in frontiers]
        options += [('frontier', point) for _, point in frontiers]
        if not self.grid.segment_safe(o.position[:2], o.position[:2]):
            self.plan_rejections.append(dict(kind='start', xy=list(o.position[:2]),
                                            reason=self.grid.route_rejection(o.position[:2], o.position[:2])))
        seen_goals = []
        for kind, goal in options:
            if any(kind == k and np.linalg.norm(goal-g) < .05 for k, g in seen_goals):
                continue
            seen_goals.append((kind, goal))
            self.counts['replans'] += 1
            path = self.grid.astar(o.position[:2], goal)
            if path is None or len(path) < 2:
                self.plan_rejections.append(dict(kind=kind, xy=goal.tolist(),
                                                reason=self.grid.route_rejection(o.position[:2], goal)))
                if retry is not None and kind == retry[0] and np.linalg.norm(goal-retry[1]) < .05:
                    self.retry_goal = None
                    if kind == 'frontier':
                        self.rejected_goals.append(np.asarray(goal))
                        self.failed_frontiers.append(dict(xy=np.asarray(goal),
                            until=o.now+self.p.frontier_failure_cooldown_s))
                continue
            self.route, self.route_index = path, 1
            self.route_goal, self.goal_kind = goal, kind
            self.route_revision = self.grid.revision
            self.blocked_scans = 0
            self.retry_goal = None
            self.transition('align', f'new {kind} route on map revision {self.route_revision}', o.now)
            return True
        if during_scan:
            self.reason = 'scanning: no reachable observed-safe observation point yet'
            return False
        self.blocked_scans += 1
        why = self.grid.route_rejection(o.position[:2], o.position[:2])
        if self.blocked_scans >= self.p.max_blocked_scans or len(self.visited) >= self.p.max_frontier_visits:
            self.transition('exhausted', f'no safe frontier after bounded scans ({why}); pilot must recover', o.now)
        else:
            self.begin_scan(o, f'no safe frontier ({why}): bounded rotation to observe another direction')

    def _remaining_route(self, xy):
        return [np.asarray(xy, float)] + list(self.route[self.route_index:])

    def tick(self, o):
        dt = .05 if self.last_t is None else float(np.clip(o.now - self.last_t, 0., .15))
        clock_bad = self.last_t is not None and o.now < self.last_t
        self.last_t = o.now
        active = self.started is not None
        if self.phase == 'yield':
            return Command(self.phase, self.reason)
        if clock_bad or o.fault or (active and (not o.offboard or not o.armed
                                               or (not o.ready and self.phase != 'recover'))):
            self.route = []
            if self.grid.launch_volume is not None:
                self.grid.launch_volume.revoke('clock/sensor/authority loss: launch assertion withdrawn')
                self.grid.refresh_launch_volume(o.now)
            self.transition('yield', o.fault or ('clock reset' if clock_bad else 'authority/sensor qualification lost'), o.now)
            return Command(self.phase, self.reason)
        if not active:
            if not o.ready:
                return Command(self.phase, 'waiting for fresh calibrated sensors')
            if not o.offboard:
                self.transition('arm', 'qualified; waiting for pilot', o.now)
                return Command(self.phase, self.reason, (0., 0., 0., 0.))
            if not o.armed or not o.landed or self.hold_xy is not None:
                self.transition('yield', 'start requires qualified pilot-controlled ground transition', o.now)
                return Command(self.phase, self.reason)
            self.started = o.now
            self.hold_xy = np.asarray(o.position[:2], float)
            # Relative climb: no original course altitude or start coordinates.
            self.hold_z = o.position[2] + self.p.cruise_height_m
            self.grid.reset(o.position[:2], self.hold_z)
            self.transition('takeoff', 'pilot-authorized vertical takeoff at launch XY', o.now)
        if o.now - self.started > self.p.time_limit_s:
            self.route = []
            self.transition('exhausted', 'bounded mission time reached; hold for pilot recovery', o.now)
            return self._hold(o)
        self.grid.expire(o.now)
        if self.phase == 'recover':
            if not o.inputs_fresh or o.now-self.recovery_started > self.p.reset_recovery_timeout_s:
                self.route = []
                self.transition('yield', 'critical inputs stale or bounded reset-recovery timeout', o.now)
                return Command(self.phase, self.reason)
            if o.ready:
                self.recovery_started = None
                self.begin_scan(o, 'fresh spatial data requalified after small reset')
            # No horizontal correction or yaw until fresh-map qualification.
            z = float(np.clip(self.hold_z-o.position[2], -.25, .25))
            return Command(self.phase, self.reason, (0., 0., z, 0.))
        if self.phase == 'takeoff':
            if not o.landed and abs(self.hold_z - o.position[2]) <= .08 and abs(o.velocity[2]) <= .10:
                self.begin_scan(o, 'stationary initial 360-degree observation')
            return self._hold(o)
        if self.phase == 'center_opening':
            command = self.centering.command(o.opening, o.position, o.yaw, o.velocity, o.now,
                authority=o.ready and o.armed and o.offboard,
                certify_sweep=o.certify_sweep, traverse=False)
            self.centering_status = command.status()
            if command.phase == 'centered':
                self.end_centering(o, 'opening centered at measured height; rebuild slab and run live A*', success=True)
                return self._hold(o)
            blocked = (command.phase in ('hold', 'stop') or
                       (command.requires_3d_clearance and not any(command.authorized_velocity[:3])))
            self.centering_blocked_since = ((o.now if self.centering_blocked_since is None else self.centering_blocked_since)
                                           if blocked else None)
            if (o.now-self.centering_started > 20. or self.centering_blocked_since is not None
                    and o.now-self.centering_blocked_since > 2.):
                self.end_centering(o, 'opening centering stopped: '+command.reason)
                return self._hold(o)
            self.reason = command.reason
            return Command(self.phase, self.reason, command.authorized_velocity)
        if self.phase == 'rebuild_height':
            if o.now-self.phase_t > 5.:
                self.transition('exhausted', 'bounded height-slab rebuild timeout; hold for pilot', o.now)
            elif o.slab_ready:
                self.begin_scan(o, 'calibrated depth rebuilt at reached height; choose live A* route')
            # Replay runs in bounded adapter slices; never move/yaw while
            # partially rebuilt geometry is being interpreted at a new height.
            z = float(np.clip(self.hold_z-o.position[2], -.25, .25))
            return Command(self.phase, self.reason, (0., 0., z, 0.))
        if self.phase in ('exhausted', 'hold_goal'):
            if self.phase == 'hold_goal' and (o.target_xy is None or o.now - o.target_stamp > self.p.target_fresh_s):
                self.begin_scan(o, 'target confidence lost; observe again')
            return self._hold(o)
        if self.phase == 'scan':
            self.scan_angle += max(0., min(.5, wrap(o.yaw - self.last_yaw)))
            self.last_yaw = o.yaw
            if np.linalg.norm(np.asarray(o.position[:2]) - self.hold_xy) > .20:
                self.transition('yield', 'stationary search drift exceeded .20m; pilot recovery', o.now)
                return Command(self.phase, self.reason)
            if self.stop_pending:
                if math.hypot(*o.velocity[:2]) > .05:
                    return self._hold(o)
                self.stop_pending = False
            if self._opening_worth_centering(o):
                self.route = []
                self.command_v[:], self.command_a[:] = 0., 0.
                self.centering.reset()
                self.centering_started, self.centering_blocked_since = o.now, None
                self.counts['opening_attempts'] += 1
                self.transition('center_opening', 'stop and center on measured depth aperture before planning through it', o.now)
                return self._hold(o)
            if (o.now-self.scan_plan_t >= self.p.scan_plan_interval_s
                    and self.grid.revision != self.scan_plan_revision):
                self.scan_plan_t, self.scan_plan_revision = o.now, self.grid.revision
                self.counts['scan_plan_checks'] += 1
                if self._plan(o, during_scan=True):
                    return self._hold(o)
                if self.phase != 'scan':
                    return self._hold(o)
            if self.scan_angle >= 2 * math.pi - .12 or o.now - self.phase_t >= self.p.scan_timeout_s:
                self.transition('plan', 'scan complete; select a reachable observation point', o.now)
                return self._hold(o)
            return self._hold(o, math.radians(self.p.yaw_rate_dps))
        if self.route:
            self.hold_xy = np.asarray(o.position[:2], float)
            remaining = self._remaining_route(o.position[:2])
            if self.grid.revision != self.route_revision:
                if not self.grid.route_safe(remaining):
                    self.invalidate_route('map changed across remaining swept route: STOP then replan', o.now)
                    return self._hold(o)
                self.route_revision = self.grid.revision
            # Recheck each tick even without a revision: current-position drift
            # can enter an unsafe part of a previously valid path corridor.
            if not self.grid.route_safe(remaining):
                self.invalidate_route('current swept route is unsafe: STOP then replan', o.now)
                return self._hold(o)
            if self.goal_kind == 'target' and (o.target_xy is None or o.now - o.target_stamp > self.p.target_fresh_s):
                self.invalidate_route('target confirmation expired', o.now)
                return self._hold(o)
        if self.phase == 'plan':
            if self.stop_pending:
                if math.hypot(*o.velocity[:2]) > .05:
                    return self._hold(o)
                self.stop_pending = False
            self._plan(o)
            return self._hold(o)
        if not self.route:
            self.transition('plan', 'route absent; choose new frontier', o.now)
            return self._hold(o)
        pos = np.asarray(o.position[:2], float)
        self.hold_xy = pos.copy()
        dest = np.asarray(self.route[self.route_index], float)
        distance = float(np.linalg.norm(dest - pos))
        if distance < self.p.arrival_m and math.hypot(*o.velocity[:2]) < .08:
            if self.route_index + 1 < len(self.route):
                self.route_index += 1
                self.transition('align', 'stop and align at checked path corner', o.now)
                self.command_v[:] = 0.
                self.command_a[:] = 0.
            else:
                self.hold_xy = pos.copy()
                self.visited.append(np.asarray(self.route_goal).copy())
                self.counts['frontiers'] += self.goal_kind == 'frontier'
                if self.goal_kind == 'target':
                    self.route = []
                    self.transition('hold_goal', 'confirmed contrast target reached; hold for pilot, no automatic landing', o.now)
                else:
                    self.begin_scan(o, 'frontier reached; stationary observation')
            return self._hold(o)
        direction = (dest - pos) / max(distance, 1e-6)
        desired_yaw = math.atan2(direction[1], direction[0])
        yaw_error = wrap(desired_yaw - o.yaw)
        yawrate = float(np.clip(2. * yaw_error, -math.radians(self.p.yaw_rate_dps), math.radians(self.p.yaw_rate_dps)))
        if abs(yaw_error) > math.radians(self.p.align_tolerance_deg):
            self.transition('align', 'slow/stop at entrance or turn; align camera with route', o.now)
            self.command_v[:] = 0.
            self.command_a[:] = 0.
            return self._hold(o, yawrate)
        self.phase = 'move'
        safe, clearance = self.grid.navigation()
        # Look ahead to decelerate BEFORE narrow passages, not after entry.
        probe = np.linspace(pos, dest, max(2, int(distance / .05) + 1))
        clearance_ahead = min(clearance[self.grid.cell(x)] for x in probe)
        narrow = clearance_ahead < self.grid.inflation_m + self.p.gap_extra_m
        ceiling = self.p.v_gap_mps if narrow else self.p.v_cruise_mps
        # Exact solution of delay*v + v^2/(2*a) <= remaining-distance-reserve.
        remaining = max(0., distance - .04)
        brake_cap = -self.p.braking_mps2 * self.p.command_delay_s + math.sqrt(
            (self.p.braking_mps2 * self.p.command_delay_s) ** 2 + 2 * self.p.braking_mps2 * remaining)
        speed = min(ceiling, brake_cap)
        desired_v = direction * speed
        if dt > 0:
            desired_a = (desired_v - self.command_v) / dt
            anorm = np.linalg.norm(desired_a)
            if anorm > self.p.acceleration_mps2:
                desired_a *= self.p.acceleration_mps2 / anorm
            da = desired_a - self.command_a
            dn = np.linalg.norm(da)
            if dn > self.p.jerk_mps3 * dt:
                da *= self.p.jerk_mps3 * dt / dn
            self.command_a += da
            self.command_v += self.command_a * dt
        norm = np.linalg.norm(self.command_v)
        if norm > speed:
            self.command_v *= speed / norm
        # A real measured speed may exceed our current request. The full stop
        # tube must remain observed safe before any new nonzero command.
        actual_speed = math.hypot(*o.velocity[:2])
        stop_distance = actual_speed * self.p.command_delay_s + actual_speed**2 / (2 * self.p.braking_mps2) + .12
        ahead = pos + direction * max(stop_distance, .12)
        if not self.grid.segment_safe(pos, ahead, safe):
            self.invalidate_route('stopping corridor leaves observed free space', o.now)
            return self._hold(o)
        self.reason = 'bounded gap traversal' if narrow else 'observed-safe route with braking profile'
        z = float(np.clip(self.hold_z - o.position[2], -.25, .25))
        return Command(self.phase, self.reason, (float(self.command_v[0]), float(self.command_v[1]), z, yawrate))

    def status(self):
        return dict(phase=self.phase, reason=self.reason, map_revision=self.grid.revision,
                    route_revision=self.route_revision, route_index=self.route_index,
                    route_goal=None if self.route_goal is None else list(map(float, self.route_goal)),
                    goal_kind=self.goal_kind, counts=self.counts.copy(), visited=len(self.visited),
                    path=[list(map(float, p)) for p in self.route], scan_angle_deg=math.degrees(self.scan_angle),
                    rejected_frontiers=[p.tolist() for p in self.rejected_goals],
                    frontier_choices=self.frontier_choices, plan_rejections=self.plan_rejections,
                    frontier_summary=self.grid.frontier_summary,
                    scan_plan_revision=self.scan_plan_revision,
                    failed_frontier_cooldowns=[dict(xy=f['xy'].tolist(), until=f['until']) for f in self.failed_frontiers],
                    retry_goal=None if self.retry_goal is None else dict(kind=self.retry_goal[0],xy=self.retry_goal[1].tolist()),
                    recovery_started=self.recovery_started, recovery_events=self.recovery_events[-10:],
                    opening_centering=self.centering_status,
                    events=self.events[-5:])
