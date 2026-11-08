"""Receding-horizon path follower and preserved flight lifecycle/sensor supervision."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import depth as D
from .vertical import VerticalCentring

NAV_OFFBOARD = 14
NAV_AUTO_LAND = 18
AIRBORNE = ("climb", "excite", "hover", "turn", "align", "check", "go", "hold", "blocked", "backoff", "recheck", "retreat", "search",
            "punch")
TRANSLATING = ("go", "backoff", "climb", "punch")           # phases in which the aircraft is commanded to move


@dataclass
class Params:
    route: list = field(default_factory=list)   # [(forward_m, left_m)] from the takeoff point and heading
    legs: list = field(default_factory=list)    # optional per-leg metadata dicts (same length as route):
                                                # {"name", "mode", "hold": bool}
    height_m: float = 1.2
    speed_mps: float = 0.6
    yaw_rate_dps: float = 45.0
    yaw_accel_dps2: float = 0.0                 # > 0: in-place turns ramp the yaw rate up and down at this bound
                                                # (trapezoidal profile) and publish the profile's rate as yawspeed
                                                # feed-forward; 0: the original abrupt slew with yawspeed 0
    face_tol_deg: float = 8.0
    arrive_m: float = 0.3
    hold_s: float = 1.0
    climb_tol_m: float = 0.15
    climb_rate_mps: float = 0.5
    hover_s: float = 0.0                        # hold at height_m this long after the climb (0: straight on)
    excite_yaw_deg: float = 0.0                 # > 0: after the climb, yaw this far in place and back, then hold
                                                # excite_settle_s, before any leg. PX4 cannot separate a horizontal
                                                # accelerometer bias from tilt while level and still; the FIRST large
                                                # yaw resolves it, and until then its velocity drifts (run010: 0.078,
                                                # run012: 0.073 m/s peak on the first ~70 deg turn, <= 0.026 on every
                                                # later one). 90 deg swaps the N/E axes, so both components become
                                                # observable; done over the take-off point, away from walls.
    excite_settle_s: float = 2.0                # run010: the error was gone ~1 s after that first yaw ended
    health_quiet_s: float = 3.0                 # healthy this long before arming
    health_stale_s: float = 1.0
    health_bad_land_s: float = 1.0
    depth_stale_s: float = 0.5
    local_horizon: bool = True                 # observed-depth horizons may end before the mission waypoint
    search_sweep_deg: float = 60.0             # search yaws at most this far either side of the bearing to the
                                                # active waypoint (never a full-circle scan)
    search_step_deg: float = 30.0              # dwell headings inside that sector
    search_blind_s: float = 2.0                # a heading with no usable depth this long is passed over: a yaw in
                                                # place needs no depth, and turning is how depth comes back
    centre_vertical: bool = True               # altitude target = midpoint of depth-measured openings ahead
                                                # (cruise height_m when nothing constrains the vertical band)
    lateral_m: float = 0.37                    # corridor half-width examined for openings (airframe + margin)
    vertical_clearance_m: float = 0.24         # kept between the airframe's centre and an opening's edge
    alt_min_m: float = 0.35
    alt_max_m: float = 1.5                     # stays below the 1.727 m walls
    alt_rate_mps: float = 0.25
    vertical_search: bool = True               # a full heading sweep found no route while a face blocks the way
                                                # close ahead: try other heights (a sill above cruise, a bar below
                                                # it) with a short sweep at each, before the headings repeat
    vertical_search_window_m: float = 1.35      # first height tried, ABOVE THE FLOOR: the window's centre (~55 in).
                                                # 0.95 m above home (1.06 m above the floor) left the 0.914 m sill
                                                # inside the navigator's +/-0.25 m band and the drone hovered there
    floor_below_home_m: float = 0.10           # the landed body origin's height above the floor (fly_course: landed_z)
    vertical_search_offsets_m: tuple = (-0.3,)  # then these, from cruise, in order, clipped to alt_min..max
    vertical_face_m: float = 1.2               # a face this close ahead at the current height starts the ladder
    vertical_hold_s: float = 20.0              # a height found by searching is held until past the face, at most this
    punch_through: bool = True                 # at the window height with no navigator route: fly straight through.
                                                # The navigator only routes through space SEEN free; through a window at
                                                # 1.35 m its band (+/-0.25 m) returns nothing within its 3 m range (floor
                                                # below the band, nets beyond), so the far side stays unknown for ever.
                                                # Accepted risk (Matthew, 2026-11-08): no return can also be a blank wall.
    punch_wait_s: float = 6.0                  # at the window height, facing the waypoint, wait this long for a route
    punch_beyond_m: float = 1.0                # then fly the face distance plus this, straight on
    punch_corridor_m: float = 0.30             # only if depth shows no return within this half-width at our height
    punch_speed_mps: float = 0.25
    replan_reorient_deg: float = 45.0          # while moving, a receding replan is tracked in motion unless its
                                                # initial heading differs by more than this (then stop and turn)
    depth_stride: int = 8                       # every n-th pixel of the depth image is used
    min_valid_frac: float = 0.3                 # below this a frame is unusable (flight data: >= 0.76)
    depth_future_tol_s: float = 0.05            # a capture stamp later than now + this is refused
    depth_border_px: int = 24                   # image edge ignored (stereo matching is weakest there)
    stream_lead_s: float = 1.0                  # setpoints stream this long before Offboard is requested
    wait_s: float = 300.0                       # nothing ready this long: give up without arming
    arm_wait_s: float = 30.0
    cmd_resend_s: float = 1.0
    land_timeout_s: float = 60.0
    # ---- competition-profile options, all off by default (see the module docstring) ----
    accel_mps2: float = 0.0                     # > 0: velocity-profiled carrot + velocity feed-forward
    track_mode: str = "carrot"                  # "carrot" (default): a target point slides along the leg on its own
                                                # clock and PX4 chases it. "pose": path following from the ACTUAL
                                                # PX4 position/velocity: the position setpoint is the point of the leg
                                                # beside the aircraft (PX4 corrects only the cross-track error), the
                                                # forward motion is a velocity command braked from the aircraft's own
                                                # remaining distance; inside capture_m the setpoint is the leg's end
    brake_mps2: float = 0.0                     # pose mode: braking deceleration (0: accel_mps2, else 0.5)
                                                # track_mode "path": cruise legs (no align / tunnel metadata) follow
                                                # a smoothed, speed-profiled path given by on_path() (coursenav
                                                # navigator: A* on the fused map, corner arcs, time-optimal profile);
                                                # gate and tunnel legs keep the straight pose tracking above
    path_lookahead_m: float = 0.35              # path mode: heading looks this far (+ 0.5 s of speed) ahead
    path_heading_lead_deg: float = 180.0        # path mode: the heading never leads the current tangent by more
                                                # (run010: 50 deg of heading preview into a corner; 180 = off)
    path_lat_accel_max_mps2: float = 0.0        # > 0: path mode lateral feed-forward v^2*kappa clamped to this
    path_stale_s: float = 1.0                   # path mode: a path older than this stops forward motion
    path_turn_hold: bool = False                # path mode: a path leg's turn with no path yet keeps the current
                                                # heading instead of the straight bearing to the leg's STOP; check and
                                                # go then wait for the path as they always do (kestral_nav934am: the
                                                # first maze path arrived after the turn, which faced a divider).
                                                # A blocking wait in the turn deadlocked with the navigator. False = old
    capture_m: float = 0.10                     # pose mode: closer than this to a stopping leg's end -> hold the end
    arrive_speed_mps: float = 0.0               # > 0: arrival ALSO needs the measured speed below this ...
    arrive_settle_s: float = 0.3                # ... for this long (no arriving while still moving: run 1 overshoot)
    min_valid_frac_static: float = -1.0         # >= 0: valid-pixel floor while NOT translating (else min_valid_frac)
    accept_armed_start: bool = False            # rule 7.4 start: accept an armed, landed vehicle at start
    turn_settle_s: float = 0.0                  # > 0: after an in-place turn, stay stopped this long before going


@dataclass
class Vehicle:
    """The latest PX4 reports, updated by the adapter as messages arrive."""
    pos: tuple | None = None          # NED metres (vehicle_local_position)
    heading: float | None = None      # rad
    pos_ok: bool = False              # xy_valid and z_valid and a heading (good for control, or PX4 yaw-aligned)
    q: tuple | None = None            # attitude, FRD -> NED, (w, x, y, z)
    armed: bool | None = None
    nav_state: int | None = None
    landed: bool | None = None
    vel: tuple | None = None          # NED m/s (vehicle_local_position vx, vy, vz when v_xy_valid), None = unknown
    rng: float | None = None          # vehicle_local_position dist_bottom (m; 0 when below the sensor's minimum range)


@dataclass
class Output:
    phase: str
    setpoint: tuple | None = None     # (n, e, d, yaw) for the trajectory setpoint
    velocity: tuple | None = None     # (vn, ve) feed-forward m/s, None = not commanded (position only)
    accel: tuple | None = None        # (an, ae) feed-forward m/s2, None = not commanded
    yawspeed: float | None = None     # yaw-rate feed-forward rad/s (NED), None = the adapter's default (0)
    stream: bool = False              # publish offboard_control_mode + the setpoint this tick
    commands: list = field(default_factory=list)   # "offboard" | "arm" | "land" | "disarm" (checked)
    events: list = field(default_factory=list)


class Health:
    """Estimator health from a boolean stream, failing closed: nothing heard, stale or unhealthy
    is never permission, and a gap restarts the quiet window."""

    def __init__(self, quiet_s, stale_s):
        self.quiet_s, self.stale_s = float(quiet_s), float(stale_s)
        self.last_t = self.last_ok = self.ok_since = self.bad_since = None

    def on_msg(self, t, ok):
        ok = bool(ok)
        if ok:
            if not self.last_ok or self.ok_since is None or (self.last_t is not None and t - self.last_t > self.stale_s):
                self.ok_since = t
            self.bad_since = None
        else:
            self.ok_since = None
            if self.bad_since is None:
                self.bad_since = t
        self.last_t, self.last_ok = t, ok

    def fresh(self, now):
        return self.last_t is not None and now - self.last_t <= self.stale_s

    def arm_ok(self, now):
        if self.last_t is None:
            return False, "no estimator health received"
        if not self.fresh(now):
            return False, f"estimator health stale ({now - self.last_t:.1f} s)"
        if not self.last_ok:
            return False, "estimator unhealthy"
        if now - self.ok_since < self.quiet_s:
            return False, f"estimator healthy for only {now - self.ok_since:.1f}/{self.quiet_s:.0f} s"
        return True, "ok"

    def bad_for(self, now):
        if self.last_t is None:
            return math.inf
        if not self.fresh(now):
            return now - (self.last_t + self.stale_s)
        if not self.last_ok:
            return now - self.bad_since
        return 0.0


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Mission:
    def __init__(self, params: Params):
        self.p = params
        self.health = Health(params.health_quiet_s, params.health_stale_s)
        self.vc = None                # VerticalCentring once airborne at cruise height
        self.vsearch = None           # altitude search of the current search episode: levels, next index, face
        self.phase = "wait"
        self.why = ""
        self.sp = None
        self.home = None              # (n, e, d, heading) at the arm request
        self.targets = []             # NED (n, e) of each leg's end
        self.leg = 0
        self.legs_done = 0
        self.t_phase = None
        self.t_started = None         # the first time this mission saw (its clock): where a replay starts
        self.t_last = None
        self.t_cmd = None
        self.offboard_seen = False
        self.land_reason = None
        self.land_auto_seen = False
        self.result = None            # final: "complete" | "blocked" | "landed: <why>" | "refused: <why>" | "yielded: <why>"
        self.depth_ready_since = None # start of continuously usable depth before control requests
        self.depth_t = None           # capture stamp of the latest USABLE depth frame (depth alive)
        self.frame = None             # that frame (refnav.depth.DepthFrame) and its obstacle points
        self.depth_pts = None
        self.leg_start = None
        self.leg_dir = None
        self.leg_len = 0.0
        self.leg_heading = 0.0
        self.leg_v_end = 0.0          # speed to carry through this leg's end (> 0: a blended corner)
        self.a_cmd = 0.0              # pose mode: feed-forward acceleration along the leg (m/s2)
        self.path = None              # path mode: dict(t, leg, pts (n,2) NED, s, v, heading, kappa, stops)
        self.path_s = 0.0             # path mode: progress along the current path (m)
        self.path_stop_done = -1      # path mode: index of the last interior stop already turned at
        self.v_vec = None             # path mode: commanded velocity (vn, ve) ...
        self.a_vec = None             # ... and feed-forward acceleration (an, ae)
        self.slow_since = None        # arrive_speed_mps: first tick the aircraft was slow enough
        self.last_pos = None          # for a speed estimate when PX4 reports no velocity
        self.s = 0.0
        self.v_cmd = 0.0              # the carrot's commanded speed this tick (velocity feed-forward)
        self.counts = {"depth_frames": 0, "depth_unusable": 0, "depth_future": 0, "depth_stale": 0,
                       "depth_out_of_order": 0, "path_waits": 0}
        self.tube_hold = False        # tube_gate: forward motion held until back inside half the tube
        self.last_v = None            # the latest Vehicle seen by step (for the safe-landing decision)
        self.yaw_rate_cmd = 0.0          # yaw_accel_dps2: the turn profile's current yaw rate (rad/s, NED)
        self.last_sp_yaw = None          # yaw_accel_dps2: the previous tick's yaw setpoint (yawspeed feed-forward)
        self.excite = None            # estimator excitation headings still to reach (excite_yaw_deg)
        self.excite_t = None
        self.land_request = None      # why an outside landing request was made (request_land), until acted on
        self.depth_reject = None      # (key, detail): why the latest refused depth frame was refused
        self.last_valid_frac = None   # fraction of pixels returning depth in the latest decoded frame

    # ------------------------------------------------------------------ leg metadata
    def _leg_meta(self, leg=None):
        return {'mode': 'cruise', 'hold': True}




    def on_path(self, t, leg, pts_ne, speeds, stops=(), pid=0, tubes=None, terminal=True):
        """track_mode "path": the current path for leg `leg` in PX4 NED (refreshed by the navigator as
        the map correction and the live map change). pts_ne: (n, 2), speeds: (n,) the time-optimal profile,
        stops: indices of interior stop-and-turn corners; pid: changes when the path is RE-PLANNED (new
        geometry from the aircraft's position: progress restarts), not when it is only re-expressed in the
        flight frame. The leg's target becomes the path's end."""
        import numpy as _np
        if (self.p.local_horizon and not terminal
                and getattr(self,'horizon_reached',None) == dict(leg=int(leg),pid=pid)):
            return False
        pts = _np.asarray(pts_ne, float)
        sp = _np.asarray(speeds, float)
        if (len(pts) < 2 or pts.ndim != 2 or pts.shape[1] != 2 or len(sp) != len(pts) or leg < 0
                or leg >= len(self.targets or []) or not _np.all(_np.isfinite(pts)) or not _np.all(_np.isfinite(sp))):
            return False
        seg = _np.linalg.norm(_np.diff(pts, axis=0), axis=1)
        s = _np.concatenate([[0.0], _np.cumsum(seg)])
        d = _np.gradient(pts, axis=0)
        heading = _np.arctan2(d[:, 1], d[:, 0])
        dh = _np.angle(_np.exp(1j * _np.diff(heading)))
        kappa = _np.concatenate([dh / _np.maximum(seg, 1e-6), [0.0]])
        # a stop-and-turn corner is turned in place, not flown as an arc: its heading kink is not curvature
        # (run024: kappa ~13 1/m there gave a 0.54 m/s^2 lateral feed-forward leaving the corner, which
        # pushed the aircraft 0.2 m toward M1)
        for i in stops:
            kappa[max(int(i) - 2, 0):int(i) + 3] = 0.0
        new_geometry = self.path is None or self.path.get("pid") != pid or self.path.get("leg") != leg
        if new_geometry:
            self.path_s, self.path_stop_done = 0.0, -1
        tb = None
        if tubes is not None:
            tb = _np.asarray(tubes, float)
            if tb.shape != (len(pts),) or not _np.all(_np.isfinite(tb)):
                tb = None
        self.path = dict(t=float(t), leg=int(leg), pid=pid, pts=pts, s=s, v=_np.maximum(sp, 0.0), heading=heading,
                         kappa=kappa, stops=sorted(int(i) for i in stops), tube=tb, terminal=bool(terminal))
        self.targets[leg] = (float(pts[-1, 0]), float(pts[-1, 1]))
        if leg == self.leg and self._path_leg():
            self.leg_len = float(s[-1])
            # A new route can point away from the currently observed corridor.
            if new_geometry and self.phase == "go":
                desired = self._path_at(self.p.path_lookahead_m)[1]
                if (self.last_v is not None and self.last_v.heading is not None
                        and abs(_wrap(desired - self.last_v.heading)) > math.radians(self.p.replan_reorient_deg)):
                    self.path_reorient = True
        return True

    def _path_leg(self, leg=None):
        return True

    def _path_fresh(self, now):
        P = self.path
        return P is not None and P["leg"] == self.leg and now - P["t"] <= self.p.path_stale_s

    def request_land(self, why="requested"):
        """An outside request to end the flight through the mission's own landing (no second controller)."""
        if self.land_request is None:
            self.land_request = str(why) or "requested"

    def depth_refused(self, key, detail):
        """A depth frame was refused (here, or before it reached the mission: no calibration, no
        attitude, an undecodable image). The reason is what the wait reports, with its numbers."""
        self.depth_reject = (str(key), str(detail))
        # Obsolete transport packets do not contradict the newer usable stream.
        # Its existing freshness deadline still expires if current data stops.
        if key not in ("out_of_order", "stale"):
            self.depth_ready_since = None

    # ------------------------------------------------------------------ inputs
    def on_health(self, t, ok):
        self.health.on_msg(t, ok)

    def _valid_floor(self):
        """The valid-pixel floor for the current phase: the static floor applies while not translating."""
        p = self.p
        if p.min_valid_frac_static >= 0 and self.phase not in TRANSLATING:
            return p.min_valid_frac_static, "min_valid_frac_static"
        return p.min_valid_frac, "min_valid_frac"

    def on_depth(self, stamp, depth_m, K, R_body_cam, t_body_cam, q_wxyz, now=None):
        """One depth frame with its own calibration; stamp = capture time on the mission clock, now =
        that clock at arrival (None: not judged for being from the future). -> True if it was usable."""
        if q_wxyz is None:
            self.depth_refused("attitude_missing", "no attitude for this depth frame")
            return False
        stamp = float(stamp)
        if now is not None and stamp > now + self.p.depth_future_tol_s:
            self.counts["depth_future"] += 1
            self.depth_refused("future", f"stamped {stamp - now:.2f} s in the future "
                                         f"(depth_future_tol_s {self.p.depth_future_tol_s:g} s)")
            return False
        if now is not None and now - stamp > self.p.depth_stale_s:
            self.counts["depth_stale"] += 1                  # too old to act on when it arrives
            self.depth_refused("stale", f"{now - stamp:.2f} s old on arrival (depth_stale_s {self.p.depth_stale_s:g} s)")
            return False
        if self.depth_t is not None and stamp <= self.depth_t:
            self.counts["depth_out_of_order"] += 1
            self.depth_refused("out_of_order", "stamped no later than the last frame used")
            return False
        f = D.DepthFrame(depth_m, K, R_body_cam, t_body_cam, D.level_rotation(q_wxyz), stride=self.p.depth_stride,
                         border_px=self.p.depth_border_px)
        self.last_valid_frac = f.valid_frac
        floor, name = self._valid_floor()
        if f.valid_frac < floor:
            self.counts["depth_unusable"] += 1
            self.depth_refused("too_little_depth", f"depth on {f.valid_frac:.0%} of the pixels, this mission needs "
                                                   f"{floor:.0%} ({name})")
            return False
        if (self.depth_ready_since is None or self.depth_t is None
                or stamp - self.depth_t > self.p.depth_stale_s):
            self.depth_ready_since = stamp
        self.frame, self.depth_t, self.depth_pts = f, stamp, f.points(0.3, 6.0)
        self.depth_reject = None                             # a refusal is only reported until a frame is used
        lv = self.last_v
        if self.vc is not None and lv is not None and lv.pos is not None and lv.heading is not None and self.home:
            self.vc.observe(stamp, self.depth_pts, self.home[2] - lv.pos[2], lv.pos[:2], lv.heading)
        self.counts["depth_frames"] += 1
        return True

    def depth_fresh(self, now):
        """A usable depth frame is recent (the camera is alive), whatever it looked at."""
        return self.depth_t is not None and now - self.depth_t <= self.p.depth_stale_s

    def depth_ready_for_arm(self, now):
        """Same quiet interval as estimator health, measured in usable capture history.

        A rare good snapshot among unusable frames cannot authorize motors. This
        only gates the first control request; airborne/authority policy is unchanged.
        """
        return (self.depth_fresh(now) and self.depth_ready_since is not None
                and math.isfinite(self.p.health_quiet_s) and self.p.health_quiet_s >= 0
                and self.depth_t - self.depth_ready_since >= self.p.health_quiet_s)


    # ------------------------------------------------------------------ step
    def _set(self, phase, now, out, why=""):
        if self.phase == "search" and phase != "search":
            self._end_altitude_search(now, out, phase)
        out.events.append(f"{self.phase} -> {phase}" + (f": {why}" if why else ""))
        self.phase, self.t_phase = phase, now
        if why:
            self.why = why

    def _end(self, now, out, result):
        self.result = result
        self._set("done", now, out, result)

    def _land(self, now, out, why, immediate=False):
        """Land where the aircraft is (PX4 Auto Land)."""
        self.land_reason = why
        self._set("land", now, out, why)


    # ------------------------------------------------------------------ safe fault landing





    def _resend(self, now):
        if self.t_cmd is None or now - self.t_cmd >= self.p.cmd_resend_s:
            self.t_cmd = now
            return True
        return False

    def _requalify(self, now, out, why):
        # Cancel the old priming interval and its held position. The original
        # preflight deadline remains t_started; this cannot renew its budget.
        self.sp, self.home, self.targets = None, None, []
        self._set("wait", now, out, why)


    def step(self, now, v: Vehicle) -> Output:
        p = self.p
        out = Output(self.phase)
        dt = 0.0 if self.t_last is None else min(max(now - self.t_last, 0.0), 0.25)
        self.t_last = now
        self.last_v = v
        if self.t_phase is None:
            self.t_phase = now
        if self.t_started is None:
            self.t_started = now
        if self.phase == 'done':
            return out
        armed_start_ok = p.accept_armed_start and v.landed is True
        if self.phase in ('wait', 'arm') and self.t_cmd is None:
            if v.armed is True and (not armed_start_ok):
                self._end(now, out, 'yielded: vehicle armed before this mission requested control')
            elif now - self.t_started > p.wait_s:
                self._end(now, out, f"refused: {self.why or 'not ready'} after {p.wait_s:.0f} s")
            elif self.phase == 'arm' and v.armed is None:
                self._requalify(now, out, 'waiting for an explicit disarmed PX4 report')
        if self.phase in AIRBORNE or (self.phase == 'land' and (not self.land_auto_seen)):
            allowed = (NAV_OFFBOARD, NAV_AUTO_LAND) if self.phase == 'land' else (NAV_OFFBOARD,)
            if self.offboard_seen and v.nav_state is not None and (v.nav_state not in allowed):
                self._end(now, out, f'yielded: PX4 left Offboard (nav_state {v.nav_state}): a pilot or a failsafe has it')
                out.phase = self.phase
                return out
        if self.land_request is not None and self.phase in ('wait', 'arm') + AIRBORNE:
            why = f'landing requested: {self.land_request}'
            if self.phase == 'wait':
                self._end(now, out, f'refused: {why} before arming')
            elif self.phase == 'arm':
                if v.armed and (not armed_start_ok):
                    out.commands.append('disarm')
                self._end(now, out, f'refused: {why} while arming')
            else:
                self._land(now, out, why)
            self.land_request = None
            out.phase = self.phase
        if self.phase in AIRBORNE:
            # Flight-critical estimator failsafes only. Motion is limited solely by the
            # planner's measured walls and declared room boundary.
            bad = self.health.bad_for(now)
            if bad > p.health_bad_land_s:
                self._land(now, out, f'estimator unhealthy or silent for {bad:.1f} s', immediate=True)
            elif not v.pos_ok or v.pos is None:
                self._land(now, out, 'PX4 position or heading invalid', immediate=True)
        if getattr(self, 'path_reorient', False) and self.phase == 'go':
            self.path_reorient = False
            if self._path_fresh(now) and v.pos is not None:
                self.sp[:2] = v.pos[:2]
                self.v_cmd = self.a_cmd = self.yaw_rate_cmd = 0.0
                self.v_vec = self.a_vec = (0.0, 0.0)
                self._set('turn', now, out, 'new A* route: face path before checking fresh depth')
        if p.local_horizon and self.phase in ('turn', 'go') and (not self._path_fresh(now)):
            self._search_path(now, out, v, 'no executable depth horizon: observe other headings')
        if p.local_horizon and self.phase == 'search' and self._path_fresh(now):
            self.path_reorient = False
            self.leg_heading = self._path_at(p.path_lookahead_m)[1]
            self._set('turn', now, out, 'executable depth horizon found: face route')
        ph = self.phase
        if ph == 'wait':
            ok, why = self.health.arm_ok(now)
            if now - self.t_started > p.wait_s:
                self._end(now, out, f"refused: {self.why or 'not ready'} after {p.wait_s:.0f} s")
            elif v.armed is not False and (not (v.armed is True and armed_start_ok)):
                self.why = 'waiting for an explicit disarmed PX4 report' if not p.accept_armed_start else 'waiting for a PX4 arming report (an armed vehicle must report landed)'
            elif not v.pos_ok or v.pos is None or v.heading is None:
                self.why = 'waiting for PX4 position and heading'
            elif not ok:
                self.why = why
            elif not self.depth_fresh(now):
                self.why = 'waiting for live depth: ' + (f'the latest frame was refused: {self.depth_reject[1]}' if self.depth_reject else 'no depth frame has arrived' if self.depth_t is None else f'the latest usable frame is {now - self.depth_t:.1f} s old')
            elif not self.depth_ready_for_arm(now):
                self.why = f'waiting for {p.health_quiet_s:g} s of continuously usable depth'
            else:
                n, e, d = v.pos
                self.home = (n, e, d, v.heading)
                self.sp = [n, e, d, v.heading]
                c, s = (math.cos(v.heading), math.sin(v.heading))
                self.targets = [(n + f * c + l * s, e + f * s - l * c) for f, l in p.route]
                # True mission waypoints. targets[leg] becomes a horizon endpoint while
                # following; search bearings always use these instead.
                self.waypoints_ne = list(self.targets)
                self._set('arm', now, out, 'vehicle already armed on the ground: event start' if v.armed else '')
        elif ph == 'arm':
            ok, why = self.health.arm_ok(now)
            if not ok:
                if self.t_cmd is None and v.armed is False:
                    self._requalify(now, out, f'{why}; requalifying before any control request')
                else:
                    self._end(now, out, f'refused: {why} before arming')
            elif self.t_cmd is None and (not self.depth_ready_for_arm(now)):
                if v.armed is False:
                    self._requalify(now, out, 'depth not continuously usable; requalifying before any control request')
                else:
                    self._end(now, out, 'yielded: depth lost before this mission requested control')
            elif now - self.t_phase > p.arm_wait_s:
                if v.armed and (not armed_start_ok):
                    out.commands.append('disarm')
                self._end(now, out, f'refused: not armed in Offboard after {p.arm_wait_s:.0f} s')
            elif v.armed and v.nav_state == NAV_OFFBOARD:
                self.offboard_seen = True
                self._set('climb', now, out)
            elif now - self.t_phase >= p.stream_lead_s and self._resend(now):
                out.commands += ['offboard'] + ([] if v.armed else ['arm'])
        elif ph == 'climb':
            target = self.home[2] - p.height_m
            self.sp[2] = max(target, self.sp[2] - p.climb_rate_mps * dt)
            if self.sp[2] <= target + 1e-09 and abs(v.pos[2] - target) < p.climb_tol_m:
                if p.centre_vertical and self.vc is None:
                    self.vc = VerticalCentring(p.height_m, p.lateral_m, p.vertical_clearance_m,
                                               p.alt_min_m, p.alt_max_m, p.alt_rate_mps)
                if p.excite_yaw_deg > 0 and self.excite is None:
                    self.excite = [_wrap(self.sp[3] + math.radians(p.excite_yaw_deg)), self.sp[3]]
                    self._set('excite', now, out, f'estimator excitation: yaw {p.excite_yaw_deg:.0f} deg in place and back')
                elif p.hover_s > 0:
                    self._set('hover', now, out, f'holding {p.hover_s:.0f} s at {p.height_m:.1f} m')
                else:
                    self._next_leg(now, out, v)
        elif ph == 'excite':
            if self.excite:
                err = _wrap(self.excite[0] - self.sp[3])
                step = math.radians(p.yaw_rate_dps) * dt
                self.sp[3] = _wrap(self.sp[3] + max(-step, min(step, err)))
                if abs(err) < 0.001 and abs(_wrap(self.excite[0] - v.heading)) < math.radians(p.face_tol_deg):
                    self.excite.pop(0)
                    self.excite_t = now
            elif now - self.excite_t >= p.excite_settle_s:
                if p.hover_s > 0:
                    self._set('hover', now, out, f'holding {p.hover_s:.0f} s at {p.height_m:.1f} m')
                else:
                    self._next_leg(now, out, v)
        elif ph == 'hover':
            if now - self.t_phase >= p.hover_s:
                self._next_leg(now, out, v)
        elif ph == 'turn':
            if self._path_leg() and self._path_fresh(now) and (self.path_stop_done < 0) and (self.path_s == 0.0):
                self.leg_heading = self._path_at(p.path_lookahead_m)[1]
            elif p.path_turn_hold and self._path_leg() and (self.path_stop_done < 0) and (self.path_s == 0.0):
                if self.leg_heading != self.sp[3]:
                    out.events.append(f'turn: no path yet for leg {self.leg + 1}: keeping the heading (not the straight bearing to the stop); go waits for the path')
                self.leg_heading = self.sp[3]
            err = _wrap(self.leg_heading - self.sp[3])
            if p.yaw_accel_dps2 > 0:
                out.yawspeed = self._slew_yaw(err, dt)
            else:
                step = math.radians(p.yaw_rate_dps) * dt
                self.sp[3] = _wrap(self.sp[3] + max(-step, min(step, err)))
            if abs(err) < 0.001 and abs(_wrap(self.leg_heading - v.heading)) < math.radians(p.face_tol_deg):
                self.yaw_rate_cmd = 0.0
                out.yawspeed = 0.0 if p.yaw_accel_dps2 > 0 else None
                self._after_turn(now, out)
        elif ph == 'search':
            err = _wrap(self.search_target - self.sp[3])
            step = math.radians(p.yaw_rate_dps) * dt
            self.sp[3] = _wrap(self.sp[3] + max(-step, min(step, err)))
            facing = abs(_wrap(self.search_target - v.heading)) < math.radians(p.face_tol_deg)
            if facing and self._at_search_level(v):
                if self.search_settled is None:
                    self.search_settled = now
                dwelt = now - self.search_settled
                window_wait = self._window_level() and self.search_steps == 0 and not self._path_fresh(now)
                if window_wait and dwelt >= p.punch_wait_s and self._punch_clear(now, v):
                    self._begin_punch(now, out, v)
                elif window_wait and dwelt < p.punch_wait_s:
                    pass                                       # at the window height: give the navigator time
                elif (self.depth_fresh(now) and dwelt >= max(p.turn_settle_s, 0.7)) or dwelt >= p.search_blind_s:
                    if self.search_steps == 0:                 # looking along the bearing to the waypoint
                        self.search_face = self._face_ahead(v) if self.depth_fresh(now) else None
                    self.search_steps += 1
                    if self._path_fresh(now):
                        self.path_s, self.path_stop_done = (0.0, -1)
                        self.path_reorient = False
                        self._set('turn', now, out, 'sweep found a route: face it, then check fresh depth')
                    elif self.search_steps >= len(self.search_offsets):
                        if not self._next_search_level(now, out, v):
                            self._begin_sweep(v, self.search_cycles + 1)   # re-centre on the current bearing
                    else:
                        self.search_target = _wrap(self.search_center + self.search_offsets[self.search_steps])
                        self.search_settled = None
            else:
                self.search_settled = None
        elif ph == 'go':
            self._go_path(now, dt, out, v)
        elif ph == 'punch':
            self._punch(now, dt, out, v)
        elif ph == 'hold':
            if now - self.t_phase >= p.hold_s:
                self.legs_done += 1
                self.leg += 1
                self._next_leg(now, out, v)
        if self.phase == 'land':
            if v.nav_state == NAV_AUTO_LAND:
                self.land_auto_seen = True
            if not self.land_auto_seen and self._resend(now):
                out.commands.append('land')
            if self.land_auto_seen and v.armed is False:
                why = self.land_reason or ''
                self._end(now, out, 'complete' if why == 'route complete' else f'landed: {why}')
            elif now - self.t_phase > p.land_timeout_s:
                self._end(now, out, f'landed: {self.land_reason}; landing not confirmed within {p.land_timeout_s:.0f} s')
        if (self.vc is not None and self.home is not None and v.pos is not None
                and self.phase in AIRBORNE and self.phase != 'climb'):
            # Altitude: centred in openings measured by depth, else cruise; rate limited.
            alt = self.vc.step(now, dt, self.home[2] - v.pos[2], v.pos[:2])
            self.sp[2] = self.home[2] - alt
        if self.phase != 'go':
            self.v_cmd, self.a_cmd = (0.0, 0.0)
            self.v_vec = self.a_vec = None
        if self.phase not in ('go', 'hold', 'retreat'):
            self.slow_since = None
        out.phase = self.phase
        out.setpoint = tuple(self.sp) if self.sp is not None else None
        if p.yaw_accel_dps2 > 0 and self.sp is not None:
            if out.yawspeed is None and self.last_sp_yaw is not None and (dt > 0):
                out.yawspeed = _wrap(self.sp[3] - self.last_sp_yaw) / dt
            self.last_sp_yaw = self.sp[3]
        pose = p.track_mode in ('pose',)
        path_go = self.phase == 'go' and self.v_vec is not None and self._path_leg()
        if path_go:
            out.velocity, out.accel = (self.v_vec, self.a_vec)
        elif (p.accel_mps2 > 0 or pose or p.track_mode == 'path') and self.phase in AIRBORNE and (self.leg_dir is not None):
            out.velocity = (self.leg_dir[0] * self.v_cmd, self.leg_dir[1] * self.v_cmd)
            if pose or p.track_mode == 'path':
                out.accel = (self.leg_dir[0] * self.a_cmd, self.leg_dir[1] * self.a_cmd)
        out.stream = self.phase in ('arm',) + AIRBORNE or (self.phase == 'land' and (not self.land_auto_seen))
        return out

    # ------------------------------------------------------------------ legs
    def _stops_at_end(self,leg):
        return True

    def _next_leg(self, now, out, v, blended=False):
        if self.leg >= len(self.targets):
            self._land(now, out, "route complete")
            return
        start = (self.sp[0], self.sp[1])
        tn, te = self.targets[self.leg]
        dn, de = tn - start[0], te - start[1]
        self.leg_len = math.hypot(dn, de)
        self.leg_start = start
        self.leg_dir = (dn / self.leg_len, de / self.leg_len) if self.leg_len > 1e-6 else (1.0, 0.0)
        self.leg_heading = math.atan2(de, dn) if self.leg_len > 1e-6 else self.sp[3]
        self.path_s, self.path_stop_done = 0.0, -1
        if self._path_leg() and self.path is not None and self.path["leg"] == self.leg:
            self.leg_heading = float(self.path["heading"][0])
        self.leg_v_end = 0.0 if (self._stops_at_end(self.leg) or self.p.track_mode == "path") else self.p.speed_mps
        if blended:
            self.counts["blended_turns"] += 1
            self.s = 0.0
            out.events.append(f"blend -> leg {self.leg + 1}/{len(self.targets)} "
                              f"(heading change {math.degrees(abs(_wrap(self.leg_heading - self.sp[3]))):.0f} deg)")
            return
        self._set("turn", now, out, f"leg {self.leg + 1}/{len(self.targets)}")

    def _slew_yaw(self, err, dt):
        """Trapezoidal yaw-rate profile toward a heading err rad away: the rate rises and falls at no more
        than yaw_accel_dps2, never exceeds yaw_rate_dps, and brakes so it arrives at rate 0
        (|rate| <= sqrt(2 * accel * |err|)). Moves the yaw setpoint; -> the rate (rad/s, NED) for yawspeed."""
        p = self.p
        acc = math.radians(p.yaw_accel_dps2)
        # discrete-time braking curve: the largest rate from which steps of decreasing rate (acc*dt less
        # each tick) still stop exactly at the target (continuous sqrt(2*acc*err) overshoots the bound)
        cap = min(math.radians(p.yaw_rate_dps),
                  acc * (math.sqrt(max(dt, 1e-3) ** 2 / 4.0 + 2.0 * abs(err) / acc) - max(dt, 1e-3) / 2.0))
        r = abs(self.yaw_rate_cmd) if self.yaw_rate_cmd * err > 0 else 0.0      # a reversal restarts from rest
        r = min(cap, r + acc * dt)
        step = min(r * dt, abs(err))
        self.sp[3] = _wrap(self.sp[3] + math.copysign(step, err))
        self.yaw_rate_cmd = math.copysign(r, err) if step < abs(err) else 0.0
        return self.yaw_rate_cmd

    def _after_turn(self, now, out):
        """The turn is done: follow the route (no separate corridor check)."""
        self.s = 0.0
        self._set("go", now, out)

    def _leg_speed(self):
        """This leg's cruise ceiling: the leg's own "speed_mps" (planner.leg_speed_mps, from its
        tube) when the route gives one, else speed_mps."""
        v = self._leg_meta().get("speed_mps")
        return float(v) if v else self.p.speed_mps








    def _path_project(self, pos):
        """Arc length and segment of the point of the current path nearest to pos, searched forward from
        the last progress (a path can come back near itself; progress never jumps backwards much)."""
        import numpy as _np
        P = self.path
        pts, s = P["pts"], P["s"]
        i0 = max(0, int(_np.searchsorted(s, self.path_s - 0.3)) - 1)
        i1 = min(len(pts) - 1, int(_np.searchsorted(s, self.path_s + 1.5)) + 1)
        a, b = pts[i0:i1], pts[i0 + 1:i1 + 1]
        if len(a) == 0:
            return float(s[-1]), len(pts) - 2
        ab = b - a
        L2 = _np.maximum((ab ** 2).sum(1), 1e-12)
        q = _np.asarray(pos[:2], float)
        t = _np.clip(((q - a) * ab).sum(1) / L2, 0.0, 1.0)
        d = _np.linalg.norm(a + t[:, None] * ab - q, axis=1)
        j = int(_np.argmin(d))
        return float(s[i0 + j] + t[j] * _np.sqrt(L2[j])), i0 + j

    def _path_at(self, s_q):
        """(point (n, e), heading, curvature, profile speed) of the current path at arc length s_q."""
        import numpy as _np
        P = self.path
        s = P["s"]
        s_q = min(max(s_q, 0.0), float(s[-1]))
        i = min(max(int(_np.searchsorted(s, s_q)) - 1, 0), len(s) - 2)
        f = (s_q - s[i]) / max(s[i + 1] - s[i], 1e-9)
        pt = P["pts"][i] + f * (P["pts"][i + 1] - P["pts"][i])
        dh = math.atan2(P["pts"][i + 1][1] - P["pts"][i][1], P["pts"][i + 1][0] - P["pts"][i][0])
        return (float(pt[0]), float(pt[1])), dh, float(P["kappa"][i]), float(P["v"][i] + f * (P["v"][i + 1] - P["v"][i]))


    def _go_path(self, now, dt, out, v):
        """Follow the navigator's path from the ACTUAL pose (track_mode "path", cruise legs):
        position setpoint = the path point beside the aircraft, velocity = profile speed along the
        tangent (braked from the actual distance to the next stop), acceleration feed-forward = the
        profile's along-path acceleration + v^2 * kappa across; heading = the tangent a look-ahead
        ahead (the camera looks where the aircraft goes). Same gates as every leg: no fresh path or no
        live evidence of the way ahead stops forward motion; a confirmed obstacle within the stop
        distance ALONG THE PATH stops, holds, backs off, rechecks."""
        import numpy as _np
        p = self.p
        if not self._path_fresh(now):
            self.motion_limit = dict(stamp=now,reason='path_missing_or_stale')
            self.counts["path_waits"] += 1
            self.v_cmd, self.a_cmd = 0.0, 0.0
            self.v_vec, self.a_vec = (0.0, 0.0), (0.0, 0.0)
            return
        P = self.path
        s_now, seg_i = self._path_project(v.pos)
        self.path_s = max(self.path_s - 0.05, s_now)            # progress: no jumping back along the path
        L = float(P["s"][-1])
        # every interior stop not yet executed stays ahead: progress may not run past it. run026 (and run022)
        # skipped 7 of 12 maze corner stops: near a U-turn the projection jumped onto the segment beyond the
        # corner, the old filter (s[i] > path_s - 0.05) dropped the stop, and the corner was flown yawing
        # while moving (run026's M3 minimum, 0.268 m)
        nxt = [i for i in P["stops"] if i > self.path_stop_done]
        if nxt:
            self.path_s = min(self.path_s, float(P["s"][nxt[0]]))
        s_stop = float(P["s"][nxt[0]]) if nxt else L            # brake to the next interior stop, else the end
        remaining = s_stop - self.path_s
        look = self.path_s + p.path_lookahead_m + 0.5 * self.v_cmd
        _, head_ahead, _, _ = self._path_at(min(look, s_stop))
        pt, tang, kap, _ = self._path_at(self.path_s)
        lead = math.radians(p.path_heading_lead_deg)
        if lead < math.pi:                                      # the heading may not run ahead of the motion
            head_ahead = tang + max(-lead, min(lead, _wrap(head_ahead - tang)))
        # the profile's limit just ahead (curves, clearance, stops coming up), not at the aircraft: the
        # profile starts at 0 from rest, and reading it here would hold the aircraft there for ever; the
        # start is ramped by accel_mps2 below
        _, _, _, v_prof = self._path_at(self.path_s + max(0.1, 0.3 * self.v_cmd))
        self.leg_heading = head_ahead
        self.leg_dir = (math.cos(tang), math.sin(tang))
        err = _wrap(head_ahead - self.sp[3])
        step = math.radians(p.yaw_rate_dps) * dt
        self.sp[3] = _wrap(self.sp[3] + max(-step, min(step, err)))
        # The planner's route already respects measured walls and the room boundary;
        # no second live-depth stop, speed cap or tube hold here.
        cap = p.speed_mps
        across = math.hypot(v.pos[0] - pt[0], v.pos[1] - pt[1])
        tube = None
        a_b = p.brake_mps2 if p.brake_mps2 > 0 else (p.accel_mps2 if p.accel_mps2 > 0 else 0.5)
        a_up = p.accel_mps2 if p.accel_mps2 > 0 else a_b
        v_brake = math.sqrt(2.0 * a_b * max(remaining - p.capture_m, 0.0))
        v_along = self._along_speed(v, (math.cos(tang), math.sin(tang)))
        v_ramp = self._ramp_from(v_along) + a_up * dt
        v_ref = max(0.0, min(cap, v_prof, v_brake, v_ramp))
        limits = dict(cruise=cap,profile=v_prof,braking=v_brake,ramp=v_ramp)
        binding = min(limits,key=limits.get)
        self.motion_limit = dict(stamp=now,reason=binding,limits_mps=limits,
            path_s_m=self.path_s,remaining_m=remaining,terminal=P.get('terminal',True),
            across_m=across,tube_m=tube)
        # along-path feed-forward: the profile's slope (planned), or the braking/ramp limit that binds
        _, _, _, v_here = self._path_at(self.path_s)
        _, _, _, v_next = self._path_at(self.path_s + 0.05)
        a_along = (v_next ** 2 - v_here ** 2) / 0.1
        if v_ref <= 0.0:
            a_along = 0.0
        elif v_ref == v_brake and v_brake < min(cap, v_prof):
            a_along = -a_b
        elif v_ref == v_ramp and v_ramp < min(cap, v_prof):
            a_along = a_up
        a_along = self._consistent_accel(max(-a_b, min(a_up, a_along)), v_ref, v_along, dt, a_b)
        a_lat = v_ref ** 2 * kap
        if p.path_lat_accel_max_mps2 > 0:                       # never more than the navigator planned arcs for
            a_lat = max(-p.path_lat_accel_max_mps2, min(p.path_lat_accel_max_mps2, a_lat))
        captured = remaining <= p.capture_m
        if captured:
            self.motion_limit['reason'] = 'endpoint_capture'
            end = P["pts"][nxt[0]] if nxt else P["pts"][-1]
            self.sp[0], self.sp[1] = float(end[0]), float(end[1])
            v_ref, a_along, a_lat = 0.0, 0.0, 0.0
        elif v_ref > 1e-3:
            self.sp[0], self.sp[1] = pt                         # beside the aircraft: no along-track pull
        self.v_cmd, self.a_cmd = v_ref, a_along
        c, sn = math.cos(tang), math.sin(tang)
        self.v_vec = (c * v_ref, sn * v_ref)
        self.a_vec = (c * a_along - sn * a_lat, sn * a_along + c * a_lat)
        end = P["pts"][nxt[0]] if nxt else P["pts"][-1]
        if math.hypot(v.pos[0] - end[0], v.pos[1] - end[1]) < p.arrive_m and self._settled(now, dt, v):
            self.v_cmd, self.a_cmd = 0.0, 0.0
            self.v_vec, self.a_vec = (0.0, 0.0), (0.0, 0.0)
            if nxt:                                             # an interior sharp corner: turn in place, check, go
                self.path_stop_done = nxt[0]
                self.path_s = float(P["s"][nxt[0]])
                _, h_next, _, _ = self._path_at(self.path_s + p.path_lookahead_m)
                self.leg_heading = h_next
                self._set("turn", now, out, f"path corner on leg {self.leg + 1}: stop and turn")
            elif not P.get('terminal', True):
                self.horizon_reached = dict(leg=self.leg,pid=P['pid'])
                self.path = None
                self._search_path(now, out, v, 'observation horizon reached; extend depth route')
            else:
                self.sp[0], self.sp[1] = float(end[0]), float(end[1])
                self._set("hold", now, out, f"arrived leg {self.leg + 1}")

    def _search_path(self, now, out, v, why):
        self.sp[:2] = v.pos[:2]
        self.v_cmd = self.a_cmd = self.yaw_rate_cmd = 0.
        self.v_vec = self.a_vec = (0., 0.)
        self.tube_hold = False
        self._begin_sweep(v)
        self.vsearch = None
        if self.vc is not None:
            self.vc.override = None
        self.motion_limit = dict(stamp=now,reason='search',detail=why)
        self._set('search', now, out, why)

    # ------------------------------------------------------------------ altitude search
    def _height(self, v):
        return None if self.home is None or v.pos is None else self.home[2] - v.pos[2]

    def _face_ahead(self, v):
        """A face within vertical_face_m ahead at the current height in the latest depth, as
        (distance, highest allowed height): returns above the aircraft NEARER than the face are an
        overhang (a roof, a bar) that caps any climb by vertical_clearance_m. None: no face."""
        import numpy as _np
        p, h = self.p, self._height(v)
        if self.depth_pts is None or len(self.depth_pts) == 0 or h is None:
            return None
        pts = _np.asarray(self.depth_pts, float)
        x, y, up = pts[:, 0], pts[:, 1], -pts[:, 2]
        corridor = (x > .3) & (_np.abs(y) <= p.lateral_m)
        face = corridor & (x < p.vertical_face_m) & (_np.abs(up) <= .15)
        if face.sum() < 6:
            return None
        d = float(_np.quantile(x[face], .1))
        over = corridor & (x < d - .1) & (up > .15)
        cap = h + float(up[over].min()) - p.vertical_clearance_m if over.any() else p.alt_max_m
        return d, cap

    def _at_search_level(self, v):
        """No dwell counts while the altitude search is still moving to its level."""
        if self.vc is None or self.vc.override is None:
            return True
        h = self._height(v)
        return h is not None and abs(h - self.vc.override) < self.p.climb_tol_m

    def _next_search_level(self, now, out, v):
        """The sweep at this height found no route. -> True if a short sweep at another height begins."""
        p = self.p
        if not p.vertical_search or self.vc is None:
            return False
        if self.vsearch is None:
            face = getattr(self, 'search_face', None)
            if face is None:
                return False                                   # open space or a far wall: headings only
            levels = []
            window = p.vertical_search_window_m - p.floor_below_home_m      # above home, like every height here
            for lv in [window] + [p.height_m + off for off in p.vertical_search_offsets_m]:
                lv = min(max(lv, p.alt_min_m), p.alt_max_m, face[1])
                if abs(lv - p.height_m) >= .05 and all(abs(lv - q) >= .05 for q in levels):
                    levels.append(lv)
            self.vsearch = dict(levels=levels, i=0, face_m=face[0], heading=self.search_center)
        S = self.vsearch
        if S['i'] >= len(S['levels']):
            if self.vc.override is not None:
                self.vc.override = None
                out.events.append('altitude search: no route at any height; back to cruise, headings only')
            return False
        level = S['levels'][S['i']]
        S['i'] += 1
        self.vc.override = level
        out.events.append(f"altitude search: {level:.2f} m ({S['i']}/{len(S['levels'])}), face "
                          f"{S['face_m']:.2f} m ahead")
        self._begin_sweep(v, self.search_cycles + 1, reduced=True)
        return True

    def _window_level(self):
        """The altitude search is at its first level: the window height (vertical_search_window_m)."""
        S = self.vsearch
        return (self.p.punch_through and self.vc is not None and S is not None and S['i'] == 1
                and self.vc.override is not None and abs(self.vc.override - S['levels'][0]) < 1e-9)

    def _punch_clear(self, now, v, extra=0.0):
        """Fresh depth shows no return in the straight corridor ahead at our own height, out to the face
        distance + punch_beyond_m (+ extra): what the punch-through flies."""
        import numpy as _np
        p, S = self.p, self.vsearch
        if not self.depth_fresh(now) or self.depth_pts is None or S is None:
            return False
        pts = _np.asarray(self.depth_pts, float)
        if len(pts) == 0:
            return True
        reach = S['face_m'] + p.punch_beyond_m + extra if self.phase != 'punch' else self.punch['left_m'] + extra
        x, y, up = pts[:, 0], pts[:, 1], -pts[:, 2]
        hits = (x > .3) & (x < reach) & (_np.abs(y) <= p.punch_corridor_m) & (_np.abs(up) <= .25)
        return int(hits.sum()) < 3

    def _begin_punch(self, now, out, v):
        p, S = self.p, self.vsearch
        dist = S['face_m'] + p.punch_beyond_m
        hdg = v.heading
        self.punch = dict(start=(v.pos[0], v.pos[1]), dir=(math.cos(hdg), math.sin(hdg)), dist=dist, s=0.0,
                          left_m=dist, heading=hdg)
        self.sp[3] = hdg
        out.events.append(f"punch-through: no route at the window height {self.vc.override:.2f} m after "
                          f"{p.punch_wait_s:.0f} s; corridor clear: flying {dist:.2f} m straight on")
        self._set('punch', now, out, 'window height, no route: straight through the opening')

    def _punch(self, now, dt, out, v):
        """Fly the punch-through: a carrot along the heading at punch_speed_mps, height held; stop and search
        again if a return appears in the corridor ahead, or once the distance is flown."""
        P, p = self.punch, self.p
        done = math.hypot(v.pos[0] - P['start'][0], v.pos[1] - P['start'][1])
        P['left_m'] = max(P['dist'] - done, 0.0)
        if not self._punch_clear(now, v, extra=.1):
            self._search_path(now, out, v, 'punch-through stopped: a return in the corridor ahead')
            return
        if P['left_m'] <= .05:
            self._search_path(now, out, v, f"punch-through done ({done:.2f} m): replan from beyond the opening")
            return
        P['s'] = min(P['s'] + p.punch_speed_mps * dt, done + .3, P['dist'])
        self.sp[0] = P['start'][0] + P['dir'][0] * P['s']
        self.sp[1] = P['start'][1] + P['dir'][1] * P['s']
        self.sp[3] = P['heading']

    def _end_altitude_search(self, now, out, phase):
        """Leaving search at a searched height with a route: hold that height until past the face."""
        vc, S, v = self.vc, self.vsearch, self.last_v
        if vc is None or vc.override is None:
            return
        level = vc.override
        vc.override = None
        if phase == 'land' or S is None or v is None or v.pos is None:
            return
        d = S['face_m'] + .1                                       # the face plus a wall's thickness
        at = (v.pos[0] + d * math.cos(S['heading']), v.pos[1] + d * math.sin(S['heading']))
        vc.hold_level(level, at, S['heading'], now, self.p.vertical_hold_s)
        out.events.append(f"altitude search: route found at {level:.2f} m; held until past the face")

    def _waypoint_ne(self):
        wp = getattr(self, 'waypoints_ne', None) or self.targets
        return wp[self.leg] if wp and 0 <= self.leg < len(wp) else None

    def _begin_sweep(self, v, cycle=0, reduced=False):
        """Bounded yaw sweep centred on the bearing to the ACTIVE mission waypoint.

        Dwell headings: the bearing itself, then outward on the side the aircraft
        already faces, then the other side; never beyond +/-search_sweep_deg, so the
        camera keeps looking where the aircraft must go (no full-circle scans).
        reduced (altitude search levels): the bearing and one step either side."""
        p, g = self.p, self._waypoint_ne()
        centre = v.heading
        if g is not None and v.pos is not None and math.hypot(g[0]-v.pos[0], g[1]-v.pos[1]) >= .3:
            centre = math.atan2(g[1]-v.pos[1], g[0]-v.pos[0])
        half, step = math.radians(max(p.search_sweep_deg, 0.)), math.radians(max(p.search_step_deg, 1.))
        side = [k*step for k in range(1, int(half/step+1e-9)+1)]
        if half > 0 and (not side or side[-1] < half-1e-6):
            side.append(half)
        if reduced:
            side = side[:1]
        first = 1. if _wrap(v.heading-centre) >= 0 else -1.
        self.search_center = centre
        self.search_offsets = [0.] + [first*a for a in side] + [-first*a for a in side]
        self.search_steps, self.search_settled, self.search_cycles = 0, None, cycle
        self.search_target = _wrap(centre)
        self.search_face = None

    def _speed(self, v, dt):
        """Measured horizontal speed: PX4's velocity when reported, else from successive positions."""
        if v.vel is not None and all(math.isfinite(c) for c in v.vel[:2]):
            spd = math.hypot(v.vel[0], v.vel[1])
        elif self.last_pos is not None and dt > 0:
            spd = math.hypot(v.pos[0] - self.last_pos[0], v.pos[1] - self.last_pos[1]) / dt
        else:
            spd = 0.0
        self.last_pos = (v.pos[0], v.pos[1])
        return spd

    def _settled(self, now, dt, v):
        """arrive_speed_mps > 0: slower than that for arrive_settle_s (counted from the first slow tick)."""
        p = self.p
        if p.arrive_speed_mps <= 0:
            return True
        if self._speed(v, dt) >= p.arrive_speed_mps:
            self.slow_since = None
            return False
        if self.slow_since is None:
            self.slow_since = now
        return now - self.slow_since >= p.arrive_settle_s

    # ------------------------------------------------------------------ consistent feed-forward
    # run022 (flow-corners) at M3: a hold set the command to 0 while PX4 still moved at 0.15 m/s; the
    # ramp then restarted from 0 and published +0.5 m/s^2 along the path on every tick while PX4 was
    # already faster than the command (0.19-0.27 vs 0.03-0.15 m/s). PX4 followed the acceleration to
    # 0.31 m/s and coasted 11 cm off the path (the run's minimum, tracking -0.111 m).
    @staticmethod
    def _along_speed(v, u):
        """PX4's speed along the unit direction u (NE), or None when its velocity is unknown."""
        if v.vel is None or not all(math.isfinite(c) for c in v.vel[:2]):
            return None
        return v.vel[0] * u[0] + v.vel[1] * u[1]

    def _ramp_from(self, v_along):
        """The speed a ramp continues from: the last command, or the aircraft's own along-track speed
        when that is higher (a resumed ramp must not demand acceleration the aircraft already has)."""
        return max(self.v_cmd, v_along) if v_along is not None else self.v_cmd

    def _consistent_accel(self, a, v_ref, v_along, dt, a_b):
        """Acceleration feed-forward consistent with the velocity command: never positive while the
        aircraft is already more than 0.03 m/s faster than the command. (Also capping it at the command's
        growth per tick was tried and dropped: offline on the reference course at 0.6 m/s it removed the
        profile's planned shaping and produced a contact, 0.279 m vs 0.325 m without the cap.)"""
        if v_ref <= 0.0:
            return 0.0
        if v_along is not None and v_along > v_ref + 0.03:
            a = min(a, 0.0)
        return a

    def status(self, now):
        return {"phase": self.phase, "why": self.why, "result": self.result, "started_t": self.t_started, "leg": self.leg,
                "legs": len(self.targets), "legs_done": self.legs_done, "home": self.home,
                "targets": [list(t) for t in self.targets], "setpoint": list(self.sp) if self.sp else None,
                
                "v_cmd": round(self.v_cmd, 3),
                "motion_limit": getattr(self,'motion_limit',None),
                "horizon_reached": getattr(self,'horizon_reached',None),
                "vertical": None if self.vc is None else dict(target_m=round(self.vc.target, 3),
                    command_m=None if self.vc.out is None else round(self.vc.out, 3), opening=self.vc.info,
                    search_level_m=self.vc.override,
                    held_kind=None if self.vc.held is None else self.vc.held[4]),
                "search": (None if self.phase != 'search' or getattr(self, 'search_offsets', None) is None else
                           dict(centre_deg=round(math.degrees(self.search_center), 1),
                                target_deg=round(math.degrees(self.search_target), 1),
                                step=self.search_steps, steps=len(self.search_offsets),
                                cycle=self.search_cycles, half_deg=self.p.search_sweep_deg)),
                "depth_age_s": None if self.depth_t is None else round(now - self.depth_t, 3),
                "depth_valid_frac": None if self.last_valid_frac is None else round(self.last_valid_frac, 3),
                "depth_reject": None if self.depth_reject is None else self.depth_reject[1],
                "health_bad_s": (None if math.isinf(self.health.bad_for(now)) else round(self.health.bad_for(now), 2)),
                
                
                **self.counts}
