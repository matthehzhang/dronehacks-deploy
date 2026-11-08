"""Receding-horizon path follower and preserved flight lifecycle/sensor supervision."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import depth as D
from .speed import braking_cap, side_cap

NAV_OFFBOARD = 14
NAV_AUTO_LAND = 18
AIRBORNE = ("climb", "excite", "hover", "turn", "align", "check", "go", "hold", "blocked", "backoff", "recheck", "retreat", "search")
TRANSLATING = ("go", "backoff", "climb")           # phases in which the aircraft is commanded to move
SILENCE_PAUSED = ("turn", "hold", "excite")                # no translation: depth silence is not counted here


@dataclass
class Params:
    route: list = field(default_factory=list)   # [(forward_m, left_m)] from the takeoff point and heading
    legs: list = field(default_factory=list)    # optional per-leg metadata dicts (same length as route):
                                                # {"name", "mode", "hold": bool, "align": {...}, "tunnel": {...}}
    height_m: float = 1.2
    speed_mps: float = 0.6
    lead_max_m: float = 1.0                     # the carrot never leads the aircraft by more
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
    depth_lost_land_s: float = 3.0
    stop_m: float = 0.9
    correction_max_m: float = 0.0               # > 0: accept map corrections of the PX4-frame targets up to this
                                                # total (m), applied only at leg starts; 0 (default): off
    correction_step_m: float = 0.15             # largest change applied at one leg start
    correction_deadband_m: float = 0.06         # smaller changes are ignored (no chasing localizer noise)
    correction_stale_s: float = 1.0
    correction_max_sigma_m: float = 0.10        # a correction this uncertain is ignored
    correction_rate_mps: float = 0.0            # > 0: also follow the correction DURING a leg, at most this fast
                                                # (m/s of target shift; 0 (default): leg starts only)
    leg_end_margin_m: float = -1.0              # >= 0: an obstacle closer than stop_m blocks only if it lies within
                                                # the REST OF THE LEG + this margin. A course wall planned beyond a
                                                # leg's end (maze corridors) is then expected, not a blockage.
                                                # < 0 (default): any obstacle closer than stop_m blocks (original rule)
    corridor_search: bool = True               # active yaw search instead of landing for unobserved corridor
    local_horizon: bool = True                 # observed-depth horizons may end before the mission waypoint
    search_sweep_deg: float = 60.0             # search yaws at most this far either side of the bearing to the
                                                # active waypoint (never a full-circle scan)
    search_step_deg: float = 30.0              # dwell headings inside that sector
    replan_reorient_deg: float = 45.0          # while moving, a receding replan is tracked in motion unless its
                                                # initial heading differs by more than this (then stop and turn)
    corridor_wait_land_s: float = -1.0          # >= 0: land when the current leg's corridor has gone this long
                                                # without an observing frame (in check or go). < 0 (default): hold
                                                # and wait indefinitely (original rule)
    corridor_half_m: float = 0.6
    band_m: float = 0.4
    min_obstacle_px: int = 1200                 # an obstacle covers at least this many image pixels (~35 x 35):
                                                # a 5 cm pole does out to ~4 m, sparse stereo junk does not
    depth_stride: int = 8                       # every n-th pixel of the depth image is used
    min_valid_frac: float = 0.3                 # below this a frame is unusable (flight data: >= 0.76)
    min_corridor_coverage: float = 0.6          # of the corridor's rays must return depth (flight data: >= 0.86)
    depth_future_tol_s: float = 0.05            # a capture stamp later than now + this is refused
    depth_border_px: int = 24                   # image edge ignored (stereo matching is weakest there)
    confirm_frames: int = 2                     # an obstacle must be in this many consecutive frames to stop
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
    path_tube_gate: bool = False                # path mode: outside the path's tube (navigator, per point) forward
                                                # motion holds and the setpoint is the path point beside the
                                                # aircraft; it advances again inside half the tube
    path_stale_s: float = 1.0                   # path mode: a path older than this stops forward motion
    path_turn_hold: bool = False                # path mode: a path leg's turn with no path yet keeps the current
                                                # heading instead of the straight bearing to the leg's STOP; check and
                                                # go then wait for the path as they always do (kestral_nav934am: the
                                                # first maze path arrived after the turn, which faced a divider).
                                                # A blocking wait in the turn deadlocked with the navigator. False = old
    capture_m: float = 0.10                     # pose mode: closer than this to a stopping leg's end -> hold the end
    arrive_speed_mps: float = 0.0               # > 0: arrival ALSO needs the measured speed below this ...
    arrive_settle_s: float = 0.3                # ... for this long (no arriving while still moving: run 1 overshoot)
    arrive_tight_m: float = 0.0                 # > 0 (pose mode, stopping legs): arrival needs the aircraft within this
                                                # of the leg end AND slower than arrive_tight_mps for arrive_settle_s;
    arrive_tight_mps: float = 0.02              # after arrive_timeout_s in the capture zone the looser arrive_m /
    arrive_timeout_s: float = 3.0               # arrive_speed_mps test applies (never waits for ever at a vertex)
    blend_turn_deg: float = 0.0                 # > 0: fly through a waypoint whose heading change is smaller
    blocked_hold_s: float = 0.0                 # > 0: hover this long on a confirmed obstacle before backing off
    blocked_backoff_m: float = 0.2              # ... then back off this far along the path just flown ...
    blocked_recheck_s: float = 2.0              # ... and recheck this long before landing
    blocked_clear_margin_m: float = 0.15        # resume only once the clearance exceeds the stop distance by this
    depth_lost_pause_static: bool = False       # depth silence is not counted while turning in place or holding
    stop_per_mps: float = 0.0                   # > 0: stop distance = max(stop_min_m, stop_base_m + this * v)
    stop_base_m: float = 0.45
    stop_min_m: float = 0.55                    # camera minimum range (0.4 m) + the camera's offset + margin
    min_valid_frac_static: float = -1.0         # >= 0: valid-pixel floor while NOT translating (else min_valid_frac)
    align_timeout_s: float = 4.0                # gate alignment: give up (hold-and-recheck, then land) after this
    align_frames: int = 3                       # ... agreeing measurements needed ...
    align_max_shift_m: float = 0.25             # ... largest lateral shift of the crossing ...
    align_max_dz_m: float = 0.15                # ... largest height shift ...
    align_window_max_dz_m: float = 0.30         # window only: bounded one/two-edge height correction
    align_max_turn_deg: float = 10.0            # ... largest heading change of the crossing
    tunnel_mode: bool = False                   # allow tunnel mode on legs the planner marked as a passage
    tunnel_speed_mps: float = 0.3
    tunnel_max_leg_m: float = 2.6
    tunnel_face_tol_deg: float = 5.0
    tunnel_align_max_age_s: float = 2.0         # alignment must have succeeded this recently at entry
    accept_armed_start: bool = False            # rule 7.4 start: accept an armed, landed vehicle at start
    geometry_speed: bool = False                # live depth can only LOWER existing speed ceilings
    response_delay_s: float = 0.75              # assumed total response delay, not a measured guarantee
    side_near_m: float = 0.40                   # centre-to-wall clearance at minimum speed
    side_extra_m: float = 0.30                  # extra clearance needed to allow cruise
    side_min_mps: float = 0.15
    side_reach_m: float = 1.0
    safe_land: bool = False                     # a FAULT landing descends only over a clear floor column (land_columns);
                                                # elsewhere the aircraft first retraces its own flown path to the
                                                # nearest clear column. Estimator / PX4-position failures still land
                                                # at once (no position to retrace with); "route complete" is the pad.
    retreat_speed_mps: float = 0.10
    turn_settle_s: float = 0.0                  # > 0: after an in-place turn, stay stopped this long before going
    first_turn_gate_s: float = 0.0              # > 0: after the FIRST in-place turn of first_turn_min_deg or more
                                                # since take-off, go only once back within first_turn_gate_m of the
                                                # setpoint and slower than first_turn_gate_mps (or after this timeout).
                                                # PX4's velocity estimate errs ~0.08-0.09 m/s after the first large yaw
                                                # only (runs 010-016; later yaws <= 0.033): leg 2, the leg after it, was
                                                # the minimum-clearance leg of every steady run
    first_turn_gate_m: float = 0.03
    sim_stuck_land_disarm_s: float = 0.0        # SIMULATOR-CONFIGURATION WORKAROUND, never on hardware (refnav.node
                                                # forces 0 unless use_sim_time). > 0: after AUTO_LAND, once the
                                                # rangefinder shows the ground and PX4's height estimate then runs away
                                                # for this long while PX4 stays armed, send a FORCED disarm. Cause (sim
                                                # only): range-only height (EKF2_HGT_REF 2) with a 0.04 m minimum range
                                                # vs a 0.049-0.078 m resting reading; run019's reading fell to 0 on the
                                                # pad, PX4's z ran to +117 m, its land detector never confirmed and it
                                                # stayed armed (graded FAIL). See docs/STEADY_RUN_930AM.md.
    first_turn_gate_mps: float = 0.02
    first_turn_min_deg: float = 20.0
    crumb_m: float = 0.10                       # breadcrumb spacing of the flown path (retrace resolution)
    land_columns: dict | None = None            # coursenav.course.land_columns() from the route file (never a ROS param)
    tube_gate: bool = False                     # pose mode: a leg's "tube_m" (planner.leg_tube_m) gates forward
                                                # motion: outside the tube across the leg the setpoint stops
                                                # advancing (PX4 pulls the aircraft back onto the line); it
                                                # advances again once inside half the tube


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
        self.corr_t = None            # stamp of the latest frame that OBSERVED the current corridor
        self.t_check = None           # when the current leg's check began (its corridor wait starts there)
        self._judged_t = None         # the latest frame judged against the corridor (each frame once)
        self.last_coverage = None
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
        self.prev_leg_dir = None      # direction of the leg just flown (the known-free path behind)
        self.s = 0.0
        self.v_cmd = 0.0              # the carrot's commanded speed this tick (velocity feed-forward)
        self.counts = {"depth_frames": 0, "obstacle_stops": 0, "depth_holds": 0, "unconfirmed_obstacles": 0,
                       "depth_unusable": 0, "depth_future": 0, "depth_stale": 0, "depth_out_of_order": 0,
                       "corridor_unobserved": 0, "corrections": 0, "recoveries": 0, "blended_turns": 0, "path_waits": 0,
                       "alignments": 0, "tunnel_ticks": 0, "tube_holds": 0, "tube_ticks": 0, "retreats": 0}
        self.tube_hold = False        # tube_gate: forward motion held until back inside half the tube
        self.last_v = None            # the latest Vehicle seen by step (for the safe-landing decision)
        self.flown = []               # PX4 NE breadcrumbs of the path actually flown (estimate), from take-off
        self.retreat = None
        self.yaw_rate_cmd = 0.0          # yaw_accel_dps2: the turn profile's current yaw rate (rad/s, NED)
        self.last_sp_yaw = None          # yaw_accel_dps2: the previous tick's yaw setpoint (yawspeed feed-forward)
        self.capture_t = None            # arrive_tight_m: when this leg's capture zone was entered
        self.tight_since = None
        self.turn_start_heading = None   # sp yaw when the current turn began (first_turn_gate)
        self.first_turn_pending = False  # this check follows the first large turn since take-off
        self.first_turn_done = False
        self.home_rng = None             # rangefinder reading on the ground at take-off
        self.rng_low = None              # the lowest non-zero rangefinder reading seen during the landing
        self.touchdown_t = None          # sim_stuck_land_disarm_s: when the rangefinder showed the ground
        self.excite = None            # estimator excitation headings still to reach (excite_yaw_deg)
        self.excite_t = None
        self.clr_hist = []            # [(depth stamp, clearance)] along the current leg, newest last
        self.last_clearance = None
        self.land_request = None      # why an outside landing request was made (request_land), until acted on
        self.depth_reject = None      # (key, detail): why the latest refused depth frame was refused
        self.last_valid_frac = None   # fraction of pixels returning depth in the latest decoded frame
        self.corr = None              # latest map correction (t, dn, de, sigma) for the PX4-frame targets
        self.corr_applied = (0.0, 0.0)
        self.silence_from = None      # when depth silence started counting (depth_lost_pause_static)
        self.recover = None           # hold-and-recheck state: {"kind", "stage", "t0", "why", ...}
        self.align_hist = []          # gate alignment measurements for the current leg
        self._align_judged_t = None
        self.align_ok_t = None        # when the current leg's gate alignment succeeded (and for which leg)
        self.align_ok_leg = None
        self.align_result = None
        self.last_align = None        # the latest successful alignment (reporting)
        self.tunnel_entry_t = None    # first tick this leg advanced without depth (tunnel mode entry)
        self.tunnel_active = False

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
            if new_geometry and self.phase in ("go", "check", "blocked", "backoff", "recheck"):
                desired = self._path_at(self.p.path_lookahead_m)[1]
                if self.phase in ("blocked", "backoff", "recheck") or (self.last_v is not None
                        and self.last_v.heading is not None
                        and abs(_wrap(desired - self.last_v.heading)) > math.radians(
                            self.p.replan_reorient_deg if self.phase == 'go' else self.p.face_tol_deg)):
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

    def corridor_fresh(self, now):
        """A recent frame OBSERVED the corridor of the current leg."""
        return self.corr_t is not None and now - self.corr_t <= self.p.depth_stale_s

    # ------------------------------------------------------------------ step
    def _set(self, phase, now, out, why=""):
        out.events.append(f"{self.phase} -> {phase}" + (f": {why}" if why else ""))
        if self.p.depth_lost_pause_static:
            paused_before, paused_after = self.phase in SILENCE_PAUSED, phase in SILENCE_PAUSED
            if paused_before and not paused_after:
                self.silence_from = now                      # silence counts again from here
        self.phase, self.t_phase = phase, now
        if why:
            self.why = why

    def _end(self, now, out, result):
        self.result = result
        self._set("done", now, out, result)

    def _land(self, now, out, why, immediate=False):
        """Land. With safe_land, a fault landing over a column that is not clear (or unknown) first
        retraces the flown path to the nearest clear column (phase retreat); immediate landings
        (no trustworthy position) and the planned landing ("route complete") descend where they are."""
        if self.phase == "retreat" and not immediate:
            return                                   # already on the way to a clear column
        if (not immediate and why != "route complete" and self.p.safe_land and self.p.land_columns
                and self.phase in AIRBORNE and self.last_v is not None and self.last_v.pos is not None):
            pos = self.last_v.pos[:2]
            if not self._column_clear(pos):
                pts = self._retreat_path(pos)
                if pts is not None:
                    self.retreat = {"pts": pts, "s": 0.0, "why": why}
                    self.counts["retreats"] += 1
                    self._set("retreat", now, out, f"{why}: column below not clear, retracing "
                                                   f"{self._poly_len(pts):.2f} m of the flown path to a clear one")
                    return
                out.events.append(f"{why}: no clear column on the flown path; landing here")
        self.land_reason = why
        self._set("land", now, out, why)

    def _stuck_landing(self, now, out, v):
        """SIMULATOR-CONFIGURATION WORKAROUND (sim_stuck_land_disarm_s; never enabled on hardware).
        Touchdown = after AUTO_LAND, the rangefinder reads within 3 cm of its take-off ground reading,
        or reads 0 (below its minimum range) right after a non-zero reading under 0.15 m. Stuck = armed
        for sim_stuck_land_disarm_s after that, the rangefinder still at the ground (no climb), and
        PX4's height estimate more than 0.25 m from the take-off height or moving faster than 0.5 m/s
        vertically. Then a forced disarm, resent every cmd_resend_s."""
        p = self.p
        if p.sim_stuck_land_disarm_s <= 0 or not self.land_auto_seen or v.armed is not True or self.home is None:
            return
        r = v.rng
        ground_ref = self.home_rng if self.home_rng is not None else 0.08
        if r is not None and r > 0:
            self.rng_low = r if self.rng_low is None else min(self.rng_low, r)
        at_ground = r is not None and ((0 < r <= ground_ref + 0.03)
                                       or (r <= 1e-3 and self.rng_low is not None and self.rng_low <= 0.15))
        if not at_ground:
            self.touchdown_t = None                      # any reading off the ground restarts the evidence
            return
        if self.touchdown_t is None:
            self.touchdown_t = now
            out.events.append(f"touchdown seen by the rangefinder ({r:.3f} m; ground at take-off "
                              f"{ground_ref:.3f} m)")
            return
        diverged = v.pos is not None and (abs(v.pos[2] - self.home[2]) > 0.25
                                          or (v.vel is not None and abs(v.vel[2]) > 0.5))
        if diverged and now - self.touchdown_t >= p.sim_stuck_land_disarm_s and self._resend(now):
            out.commands.append("force_disarm")
            out.events.append(f"SIM WORKAROUND: on the ground {now - self.touchdown_t:.1f} s by the rangefinder but "
                              f"PX4 height estimate {-(v.pos[2] - self.home[2]):+.2f} m and still armed: forced disarm")

    # ------------------------------------------------------------------ safe fault landing
    def _course_xy(self, ne):
        """PX4 NE -> course xy, through the take-off pose (planner.to_route's frame, inverted)."""
        g = self.p.land_columns
        hn, he, _, h = self.home
        dn, de = ne[0] - hn, ne[1] - he
        f = dn * math.cos(h) + de * math.sin(h)
        left = dn * math.sin(h) - de * math.cos(h)
        sx, sy, syaw = g["start"]
        c, s = math.cos(math.radians(syaw)), math.sin(math.radians(syaw))
        return sx + f * c - left * s, sy + f * s + left * c

    def _column_clear(self, ne):
        g = self.p.land_columns
        if self.home is None or not g:
            return False
        x, y = self._course_xy(ne)
        i = int(math.floor((x - g["origin"][0]) / g["res"]))
        j = int(math.floor((y - g["origin"][1]) / g["res"]))
        rows = g["rows"]
        return 0 <= i < len(rows) and 0 <= j < len(rows[i]) and rows[i][j] == "1"

    @staticmethod
    def _poly_len(pts):
        return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))

    def _retreat_path(self, pos, step=0.05):
        """The flown path backwards from pos (breadcrumbs of the estimate, newest first, to the take-off
        point), cut at the first sample whose column is clear. Breadcrumbs, not leg ends: a path-mode
        route curves around walls, and a straight line back to a leg start could cross one.
        -> [(n, e), ...] or None."""
        verts = [tuple(pos[:2])] + [tuple(v) for v in reversed(self.flown)]
        out = [verts[0]]
        for a, b in zip(verts, verts[1:]):
            L = math.dist(a, b)
            n = max(1, int(math.ceil(L / step)))
            for k in range(1, n + 1):
                q = (a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
                if self._column_clear(q):
                    out.append(q)
                    return out
            out.append(b)
        return None

    def _retreat(self, now, dt, out, v):
        """Carrot back along the retreat polyline at retreat_speed_mps (position setpoints, heading
        unchanged: the path is the one just flown), then land on the clear column at its end."""
        r = self.retreat
        pts = r["pts"]
        cum = [0.0]
        for a, b in zip(pts, pts[1:]):
            cum.append(cum[-1] + math.dist(a, b))
        L = cum[-1]
        # the aircraft's progress: nearest point on the polyline, never behind the carrot by more than a lead
        best, prog = math.inf, 0.0
        for k, (a, b) in enumerate(zip(pts, pts[1:])):
            ab = (b[0] - a[0], b[1] - a[1])
            l2 = max(ab[0] ** 2 + ab[1] ** 2, 1e-12)
            t = max(0.0, min(1.0, ((v.pos[0] - a[0]) * ab[0] + (v.pos[1] - a[1]) * ab[1]) / l2))
            d = math.hypot(a[0] + t * ab[0] - v.pos[0], a[1] + t * ab[1] - v.pos[1])
            if d < best:
                best, prog = d, cum[k] + t * math.sqrt(l2)
        r["s"] = min(L, r["s"] + self.p.retreat_speed_mps * dt, prog + self.p.lead_max_m)
        k = max(i for i in range(len(cum)) if cum[i] <= r["s"] + 1e-12)
        k = min(k, len(pts) - 2)
        f = (r["s"] - cum[k]) / max(cum[k + 1] - cum[k], 1e-12)
        self.sp[0] = pts[k][0] + f * (pts[k + 1][0] - pts[k][0])
        self.sp[1] = pts[k][1] + f * (pts[k + 1][1] - pts[k][1])
        end = pts[-1]
        if r["s"] >= L - 1e-9 and math.hypot(v.pos[0] - end[0], v.pos[1] - end[1]) < self.p.arrive_m \
                and self._settled(now, dt, v) and self._column_clear(v.pos[:2]):
            self.land_reason = r["why"]
            self._set("land", now, out, f"{r['why']}: over a clear column after retracing {L:.2f} m")

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

    def _depth_silent_s(self, now):
        """How long depth has been silent for landing purposes (the pause option excludes time
        spent in an in-place turn or a hold)."""
        if self.depth_t is None:
            return math.inf
        if self.p.depth_lost_pause_static:
            if self.phase in SILENCE_PAUSED:
                return 0.0
            since = self.depth_t if self.silence_from is None else max(self.depth_t, self.silence_from)
            return now - since
        return now - self.depth_t

    def step(self, now, v: Vehicle) -> Output:
        p = self.p
        out = Output(self.phase)
        dt = 0.0 if self.t_last is None else min(max(now - self.t_last, 0.0), 0.25)
        self.t_last = now
        self.last_v = v
        if self.phase in AIRBORNE and self.phase != 'retreat' and self.flown and (v.pos is not None) and (math.dist(self.flown[-1], v.pos[:2]) >= self.p.crumb_m):
            self.flown.append((v.pos[0], v.pos[1]))
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
            bad = self.health.bad_for(now)
            tunnel_dark = self.tunnel_active and self.phase in ('check', 'go') and self._tunnel_ok(now, v)[0]
            if bad > p.health_bad_land_s:
                self._land(now, out, f'estimator unhealthy or silent for {bad:.1f} s', immediate=True)
            elif not v.pos_ok or v.pos is None:
                self._land(now, out, 'PX4 position or heading invalid', immediate=True)
            elif self.depth_t is None or (self._depth_silent_s(now) > p.depth_lost_land_s and (not tunnel_dark)):
                self._land(now, out, f'depth silent for more than {p.depth_lost_land_s:.0f} s')
            elif p.corridor_wait_land_s >= 0 and self.phase in ('check', 'go') and (self.t_check is not None) and (not tunnel_dark):
                seen = self.t_check if self.corr_t is None else max(self.t_check, self.corr_t)
                if now - seen > p.corridor_wait_land_s:
                    if p.corridor_search:
                        self.sp[:2] = v.pos[:2]
                        self._begin_sweep(v)
                        self._set('search', now, out, 'corridor unobserved: sweep toward waypoint for live A*')
                    else:
                        self._land(now, out, f'corridor of leg {self.leg + 1} not observed for {now - seen:.1f} s (corridor_wait_land_s {p.corridor_wait_land_s:g} s)')
        if getattr(self, 'path_reorient', False) and self.phase in ('go', 'check', 'blocked', 'backoff', 'recheck'):
            self.path_reorient = False
            if self._path_fresh(now) and v.pos is not None:
                self.sp[:2] = v.pos[:2]
                self.v_cmd = self.a_cmd = self.yaw_rate_cmd = 0.0
                self.v_vec = self.a_vec = (0.0, 0.0)
                self.recover = None
                self.clr_hist, self.corr_t, self.t_check = ([], None, None)
                self._set('turn', now, out, 'new A* route: face path before checking fresh depth')
        if p.local_horizon and self.phase in ('turn', 'check', 'go', 'blocked', 'backoff', 'recheck') and self._path_leg() and (not self._path_fresh(now)):
            self._search_path(now, out, v, 'no executable depth horizon: observe other headings')
        if p.local_horizon and self.phase == 'search' and self._path_fresh(now):
            self.path_reorient = False
            self.clr_hist, self.corr_t = ([], None)
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
                self.home_rng = v.rng if v.rng is not None and v.rng > 0 else None
                self.flown = [(n, e)]
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
            if facing and self.depth_fresh(now):
                if self.search_settled is None:
                    self.search_settled = now
                if now - self.search_settled >= max(p.turn_settle_s, 0.7):
                    self.search_steps += 1
                    if self._path_fresh(now):
                        self.path_s, self.path_stop_done = (0.0, -1)
                        self.path_reorient = False
                        self.clr_hist, self.corr_t = ([], None)
                        self._set('turn', now, out, 'sweep found a route: face it, then check fresh depth')
                    elif self.search_steps >= len(self.search_offsets):
                        self._begin_sweep(v, self.search_cycles + 1)   # re-centre on the current bearing
                    else:
                        self.search_target = _wrap(self.search_center + self.search_offsets[self.search_steps])
                        self.search_settled = None
            else:
                self.search_settled = None
        elif ph == 'check':
            settled = now - self.t_phase >= p.turn_settle_s
            if self.first_turn_pending:
                off = math.hypot(v.pos[0] - self.sp[0], v.pos[1] - self.sp[1])
                spd = math.hypot(*v.vel[:2]) if v.vel is not None else math.inf
                if off < p.first_turn_gate_m and spd < p.first_turn_gate_mps or now - self.t_phase >= p.first_turn_gate_s:
                    self.first_turn_pending = False
                    out.events.append(f'first-turn gate released after {now - self.t_phase:.1f} s ({off:.3f} m off the setpoint, {spd:.3f} m/s)')
                else:
                    settled = False
            if settled and self.depth_t is not None and (self.depth_t >= self.t_phase) and self.depth_fresh(now):
                if self.clr_hist and self.clr_hist[0][0] < self.t_phase:
                    self.clr_hist, self.corr_t = ([], None)
                clr = self._clearance(v)
                if clr is not None and self.corridor_fresh(now):
                    self.last_clearance = clr
                    if self._blocks(clr, self.leg_len, 0.0):
                        self._obstacle(now, out, v, clr, 'at')
                    else:
                        self.s = 0.0
                        self._set('go', now, out)
            if self.phase == 'check' and self.tunnel_active and settled:
                ok, _ = self._tunnel_ok(now, v)
                if ok and (not self.corridor_fresh(now)):
                    self.s = 0.0
                    self._set('go', now, out, f'tunnel mode: leg {self.leg + 1} enters the passage on the map alone')
        elif ph == 'go':
            self._go(now, dt, out, v)
        elif ph == 'hold':
            if now - self.t_phase >= p.hold_s:
                self.legs_done += 1
                self.leg += 1
                self._next_leg(now, out, v)
        elif ph in ('blocked', 'backoff', 'recheck'):
            self._search_path(now,out,v,"refresh depth horizon")
        elif ph == 'retreat':
            self._retreat(now, dt, out, v)
        if self.phase == 'land':
            if v.nav_state == NAV_AUTO_LAND:
                self.land_auto_seen = True
            self._stuck_landing(now, out, v)
            if not self.land_auto_seen and self._resend(now):
                out.commands.append('land')
            if self.land_auto_seen and v.armed is False:
                why = self.land_reason or ''
                self._end(now, out, 'complete' if why == 'route complete' else f'landed: {why}')
            elif now - self.t_phase > p.land_timeout_s:
                self._end(now, out, f'landed: {self.land_reason}; landing not confirmed within {p.land_timeout_s:.0f} s')
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
        self.prev_leg_dir = self.leg_dir
        self.leg_len = math.hypot(dn, de)
        self.leg_start = start
        self.leg_dir = (dn / self.leg_len, de / self.leg_len) if self.leg_len > 1e-6 else (1.0, 0.0)
        self.leg_heading = math.atan2(de, dn) if self.leg_len > 1e-6 else self.sp[3]
        self.clr_hist, self.corr_t = [], None
        self.align_hist, self.align_result = [], None
        self.tunnel_entry_t = None
        self.tunnel_active = bool(self.p.tunnel_mode and self._leg_meta().get("tunnel"))
        self.path_s, self.path_stop_done = 0.0, -1
        self.capture_t, self.tight_since = None, None
        if self._path_leg() and self.path is not None and self.path["leg"] == self.leg:
            self.leg_heading = float(self.path["heading"][0])
        self.leg_v_end = 0.0 if (self._stops_at_end(self.leg) or self.p.track_mode == "path") else self.p.speed_mps
        if blended:
            self.counts["blended_turns"] += 1
            self.s = 0.0
            self.t_check = now
            out.events.append(f"blend -> leg {self.leg + 1}/{len(self.targets)} "
                              f"(heading change {math.degrees(abs(_wrap(self.leg_heading - self.sp[3]))):.0f} deg)")
            return
        self.turn_start_heading = self.sp[3]
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
        """The turn is done: align on a gate leg, else check the corridor."""
        self.t_check = now
        if (self.p.first_turn_gate_s > 0 and not self.first_turn_done and self.turn_start_heading is not None
                and abs(_wrap(self.sp[3] - self.turn_start_heading)) >= math.radians(self.p.first_turn_min_deg)):
            self.first_turn_done = True
            self.first_turn_pending = True
            out.events.append("first large turn since take-off: going only once back on the setpoint")
        self._set("check",now,out)

    def _leg_speed(self):
        """This leg's cruise ceiling: the leg's own "speed_mps" (planner.leg_speed_mps, from its
        tube) when the route gives one, else speed_mps."""
        v = self._leg_meta().get("speed_mps")
        return float(v) if v else self.p.speed_mps

    def _profile_speed(self, s_along):
        """The planned speed at s_along on this leg (a position, not a state: no oscillation with
        the stop distance). Trapezoid: cruise, braking to the leg's end speed at accel_mps2."""
        p = self.p
        if p.accel_mps2 <= 0:
            return self._leg_speed()
        remaining = max(self.leg_len - s_along, 0.0)
        return min(self._leg_speed(), math.sqrt(self.leg_v_end ** 2 + 2.0 * p.accel_mps2 * remaining))

    def _stop_m(self, s_along=0.0):
        p = self.p
        if p.stop_per_mps <= 0:
            return p.stop_m
        return max(p.stop_min_m, p.stop_base_m + p.stop_per_mps * self._profile_speed(s_along))

    def _clearance(self, v):
        """Clearance along the leg CONFIRMED by the last confirm_frames frames that OBSERVED the
        corridor: the largest of their clearances, so an obstacle counts only when every one of
        those frames shows it (a stereo false match lasts a frame; a wall does not go away). A frame
        whose rays through the corridor (at stop_m + lead_max_m ahead) mostly returned nothing has
        observed nothing there and is not evidence of free space. None = unknown (do not move)."""
        p = self.p
        bearing = _wrap(self.leg_heading - v.heading)        # positive = to the right (NED yaw)
        if self.frame is not None and self._judged_t != self.depth_t:
            self._judged_t = self.depth_t
            cov, _ = self.frame.corridor_coverage(bearing, p.corridor_half_m, p.band_m, self._stop_m(self.s) + p.lead_max_m)
            self.last_coverage = cov
            if cov >= p.min_corridor_coverage:
                hits = max(3, round(p.min_obstacle_px / p.depth_stride ** 2))
                ahead = (self._path_ahead_level(v) if self.phase in ("go", "check") and self._path_leg()
                         and self.path is not None and self.path["leg"] == self.leg else None)
                if ahead is not None:
                    c = D.path_clearance(self.depth_pts, ahead[0], ahead[1], p.corridor_half_m, p.band_m, hits)
                else:
                    c = D.clearance(self.depth_pts, bearing, p.corridor_half_m, p.band_m, hits)
                self.clr_hist = (self.clr_hist + [(self.depth_t, c)])[-max(1, p.confirm_frames):]
                self.corr_t = self.depth_t
            else:
                self.counts["corridor_unobserved"] += 1
        if len(self.clr_hist) < self.p.confirm_frames:
            return None
        confirmed = max(c for _, c in self.clr_hist)
        if min(c for _, c in self.clr_hist) < self._stop_m(self.s) <= confirmed:
            self.counts["unconfirmed_obstacles"] += 1
        return confirmed

    def _blocks(self, clr, remaining_m, s_along=None):
        """Is an obstacle clr metres ahead a blockage, with remaining_m of this leg still to fly?"""
        p = self.p
        stop = self._stop_m(self.s if s_along is None else s_along)
        if clr >= stop:
            return False
        return p.leg_end_margin_m < 0 or clr < max(remaining_m, 0.0) + p.leg_end_margin_m

    def _geometry_cap(self, clr, v):
        """Call only with fresh, observed corridor evidence; unknown stays stopped."""
        p = self.p
        if not p.geometry_speed:
            return self._leg_speed()
        decel = p.brake_mps2 if p.brake_mps2 > 0 else (p.accel_mps2 if p.accel_mps2 > 0 else 0.5)
        # Physical buffer, distinct from the camera's hard-stop threshold. Using
        # stop_min here would strand a valid endpoint 0.4 m before a wall. The
        # existing _blocks check still enforces its larger on-route threshold.
        front = braking_cap(clr, max(p.corridor_half_m, p.side_near_m), decel, p.response_delay_s)
        sides = side_cap(self.depth_pts, _wrap(self.leg_heading - v.heading), p.speed_mps,
                         p.side_near_m, p.side_extra_m, p.side_min_mps, p.side_reach_m,
                         p.band_m, max(3, round(p.min_obstacle_px / p.depth_stride ** 2)))
        return min(self._leg_speed(), front, sides)

    def _obstacle(self,now,out,v,clr,where):
        self.counts['obstacle_stops'] += 1
        self.path = None
        self._search_path(now,out,v,f'measured obstacle {clr:.2f} m: withdraw horizon and reobserve')

    def _go(self, now, dt, out, v):
        return self._go_path(now, dt, out, v)

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

    def _path_ahead_level(self, v):
        """The path ahead of the aircraft (stop distance + lead) in the levelled body frame (x forward,
        y right) with its arc lengths from the aircraft, for the along-path obstacle check."""
        import numpy as _np
        if self.path is None or v.pos is None or v.heading is None:
            return None
        P = self.path
        reach = self._stop_m(self.s) + self.p.lead_max_m
        s0 = self.path_s
        sel = (P["s"] >= s0) & (P["s"] <= s0 + reach)
        pts = _np.vstack([[v.pos[0], v.pos[1]], P["pts"][sel]])
        if len(pts) < 2:
            return None
        rel = pts - _np.array([v.pos[0], v.pos[1]])
        c, sn = math.cos(v.heading), math.sin(v.heading)
        fr = _np.stack([c * rel[:, 0] + sn * rel[:, 1], -sn * rel[:, 0] + c * rel[:, 1]], axis=1)
        s_rel = _np.concatenate([[0.0], _np.cumsum(_np.linalg.norm(_np.diff(pts, axis=0), axis=1))])
        return fr, s_rel

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
        clr = self._clearance(v) if self.depth_fresh(now) else None
        if clr is not None and not self.corridor_fresh(now):
            clr = None
        cap = p.speed_mps
        if clr is not None:
            cap = self._geometry_cap(clr, v)
            self.last_clearance = clr
            if self._blocks(clr, max(L - self.path_s, 0.0)):
                self.motion_limit = dict(stamp=now,reason='measured_obstacle',clearance_m=clr)
                self._obstacle(now, out, v, clr, "on")
                self.v_cmd, self.a_cmd = 0.0, 0.0
                self.v_vec, self.a_vec = (0.0, 0.0), (0.0, 0.0)
                return
        else:
            self.counts["depth_holds"] += 1
            cap = 0.0
        tube, across = None, None
        if p.path_tube_gate and P.get("tube") is not None:
            tube = float(P["tube"][min(max(seg_i, 0), len(P["tube"]) - 1)])
            across = math.hypot(v.pos[0] - pt[0], v.pos[1] - pt[1])
            if tube <= 0.005:
                pass        # no slack known at this point (run024: a 0.000 m tube right after the climb): no gate
            elif not self.tube_hold and across > tube:
                self.tube_hold = True
                self.counts["tube_holds"] += 1
                out.events.append(f"tube: {across:.3f} m off the path on leg {self.leg + 1} > {tube:.3f} m: "
                                  f"holding until back inside {tube / 2:.3f} m")
            elif self.tube_hold and across <= tube / 2:
                self.tube_hold = False
                out.events.append(f"tube: back inside {tube / 2:.3f} m on leg {self.leg + 1}: advancing")
            if self.tube_hold:
                self.counts["tube_ticks"] += 1
                cap = 0.0
        a_b = p.brake_mps2 if p.brake_mps2 > 0 else (p.accel_mps2 if p.accel_mps2 > 0 else 0.5)
        a_up = p.accel_mps2 if p.accel_mps2 > 0 else a_b
        v_brake = math.sqrt(2.0 * a_b * max(remaining - p.capture_m, 0.0))
        v_along = self._along_speed(v, (math.cos(tang), math.sin(tang)))
        v_ramp = self._ramp_from(v_along) + a_up * dt
        v_ref = max(0.0, min(cap, v_prof, v_brake, v_ramp))
        limits = dict(live_geometry=cap,profile=v_prof,braking=v_brake,ramp=v_ramp)
        binding = min(limits,key=limits.get)
        if self.tube_hold:
            binding = 'off_path_tube'
        elif clr is None:
            binding = 'depth_or_corridor_unobserved'
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
        elif p.geometry_speed and cap < p.speed_mps and v_ref == cap:
            # A reduced live ceiling must not retain an accelerating map-profile slope.
            a_along = 0.0
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
        elif v_ref > 1e-3 or self.tube_hold:
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
                self.clr_hist, self.corr_t = [], None
                self.turn_start_heading = self.sp[3]
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
        self.motion_limit = dict(stamp=now,reason='search',detail=why)
        self._set('search', now, out, why)

    def _waypoint_ne(self):
        wp = getattr(self, 'waypoints_ne', None) or self.targets
        return wp[self.leg] if wp and 0 <= self.leg < len(wp) else None

    def _begin_sweep(self, v, cycle=0):
        """Bounded yaw sweep centred on the bearing to the ACTIVE mission waypoint.

        Dwell headings: the bearing itself, then outward on the side the aircraft
        already faces, then the other side; never beyond +/-search_sweep_deg, so the
        camera keeps looking where the aircraft must go (no full-circle scans)."""
        p, g = self.p, self._waypoint_ne()
        centre = v.heading
        if g is not None and v.pos is not None and math.hypot(g[0]-v.pos[0], g[1]-v.pos[1]) >= .3:
            centre = math.atan2(g[1]-v.pos[1], g[0]-v.pos[0])
        half, step = math.radians(max(p.search_sweep_deg, 0.)), math.radians(max(p.search_step_deg, 1.))
        side = [k*step for k in range(1, int(half/step+1e-9)+1)]
        if half > 0 and (not side or side[-1] < half-1e-6):
            side.append(half)
        first = 1. if _wrap(v.heading-centre) >= 0 else -1.
        self.search_center = centre
        self.search_offsets = [0.] + [first*a for a in side] + [-first*a for a in side]
        self.search_steps, self.search_settled, self.search_cycles = 0, None, cycle
        self.search_target = _wrap(centre)

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





    # ------------------------------------------------------------------ tunnel mode
    def _tunnel_ok(self, now, v):
        return (False, 'depth horizon requires live observations')

    # ------------------------------------------------------------------ gate alignment



    # ------------------------------------------------------------------ hold-and-recheck




    def status(self, now):
        return {"phase": self.phase, "why": self.why, "result": self.result, "started_t": self.t_started, "leg": self.leg,
                "legs": len(self.targets), "legs_done": self.legs_done, "home": self.home,
                "targets": [list(t) for t in self.targets], "setpoint": list(self.sp) if self.sp else None,
                "clearance_m": (None if self.last_clearance is None or math.isinf(self.last_clearance)
                                else round(self.last_clearance, 2)),
                "stop_m": round(self._stop_m(self.s), 2),
                "v_cmd": round(self.v_cmd, 3),
                "motion_limit": getattr(self,'motion_limit',None),
                "horizon_reached": getattr(self,'horizon_reached',None),
                "search": (None if self.phase != 'search' or getattr(self, 'search_offsets', None) is None else
                           dict(centre_deg=round(math.degrees(self.search_center), 1),
                                target_deg=round(math.degrees(self.search_target), 1),
                                step=self.search_steps, steps=len(self.search_offsets),
                                cycle=self.search_cycles, half_deg=self.p.search_sweep_deg)),
                "depth_age_s": None if self.depth_t is None else round(now - self.depth_t, 3),
                "corridor_age_s": None if self.corr_t is None else round(now - self.corr_t, 3),
                "corridor_coverage": None if self.last_coverage is None else round(self.last_coverage, 2),
                "depth_valid_frac": None if self.last_valid_frac is None else round(self.last_valid_frac, 3),
                "depth_reject": None if self.depth_reject is None else self.depth_reject[1],
                "health_bad_s": (None if math.isinf(self.health.bad_for(now)) else round(self.health.bad_for(now), 2)),
                "recover": None if self.recover is None else {"kind": self.recover["kind"], "stage": self.phase},
                "tunnel": self.tunnel_active, "align": self.align_result,
                **self.counts}
