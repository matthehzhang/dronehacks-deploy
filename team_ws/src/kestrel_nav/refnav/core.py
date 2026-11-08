"""refnav.core - the reference mission's flight logic. PURE: no ROS, no clock reads, no files.

Everything arrives through calls that carry their own time, on ONE clock chosen by whoever runs
it: simulation time in the simulator and in replays of recorded data (ROS use_sim_time + /clock),
system time on hardware. The same code therefore runs against all three.

The mission: wait until PX4 has a valid position and heading, the estimator reports healthy and
the depth camera is live -> stream a hold setpoint, request Offboard and arm -> climb -> fly a
pre-planned route of legs (each leg: turn to face it, check the depth ahead, advance a carrot at a
bounded speed while the live depth shows the way clear) -> land.

Safety behaviour (each one is a test in tests/test_refnav_core.py):
* fail closed on estimator health: absent, stale or unhealthy refuses to arm; unhealthy or silent
  for health_bad_land_s in the air lands;
* no translation without OBSERVED free space: the carrot advances only while a depth frame no older
  than depth_stale_s returned depth on at least min_corridor_coverage of the rays crossing the
  corridor ahead (no return is unknown space, never free space); a blank, sparse (valid fraction
  < min_valid_frac), future-stamped, stale-on-arrival or out-of-order frame refreshes nothing; no
  usable depth for depth_lost_land_s lands. Insufficient corridor coverage alone does not impose
  a landing deadline;
* an obstacle closer than stop_m ahead stops the aircraft and ends the route (it lands where it
  is: route blocked; planning around obstacles is the student's job); with leg_end_margin_m >= 0 only
  an obstacle inside the remaining leg (+ that margin) counts, so a mapped wall beyond a planned leg
  end does not, while anything ON the leg still stops it;
* PX4 leaving Offboard while airborne (a pilot or a failsafe) is never fought: the mission stops
  streaming and commanding and ends ("yielded");
* PX4 owns the landing: once AUTO_LAND is observed the mission stops streaming setpoints (PX4's
  land detector reads the trajectory_setpoint topic; a position-only stream blanks its
  commanded-descent test and the aircraft never detects touchdown);
* a landing REQUEST from outside (the interactive runtime's `land`, request_land) goes through this
  same landing: airborne it lands where it is; before arming it ends without arming; while arming it
  ends with a normal checked disarm. It never adds a controller.

Competition-profile options (every one OFF by default, so the behaviour above is unchanged; the
course file's refnav_params turns them on; each has its own tests):
* accel_mps2 > 0: a velocity-profiled carrot (trapezoidal speed along each leg, velocity
  feed-forward in the setpoint) so PX4 arrives without overshoot;
* blend_turn_deg > 0: a heading change smaller than this is flown without stopping (no hold, no
  in-place turn); gate, alignment and passage legs always stop-turn-check-go;
* blocked_hold_s > 0: a confirmed obstacle inside the stop distance HOVERS first (a transient
  return clears), then backs off blocked_backoff_m along the path just flown, rechecks for
  blocked_recheck_s, and only then lands. Recovery judges the corridor with the same check as go (along
  the path on path legs) and only when every confirm frame reads clear; a path re-planned while blocked
  is followed (turn to it, check, go); blocked_max_per_leg > 0 lands on the (N+1)-th block of one leg;
* depth_lost_pause_static: depth silence is not counted during an in-place turn or a hold (no
  translation); while moving it still stops translation at once;
* stop_per_mps > 0: the stop distance follows the planned speed, max(stop_min_m, stop_base_m +
  stop_per_mps * v), never below the camera's blind range plus margin;
* min_valid_frac_static >= 0: a lower valid-pixel floor for frames judged while not translating;
* per-leg metadata (Params.legs, from the planner): "align" runs the depth-based gate alignment
  (refnav.gate) before a gate crossing; "tunnel" allows tunnel mode (advance without depth returns
  through a detected enclosed passage, only under every condition listed in _tunnel_ok);
* accept_armed_start: an already-armed, landed vehicle is accepted at start (rule 7.4: the team
  arms on the mark, the program starts on the referee's signal) instead of yielding.
What is never loosened: landing on an unhealthy estimator or an invalid PX4 position, yielding when
PX4 leaves Offboard, never translating into a confirmed obstacle inside the stop distance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import depth as D
from . import gate as G
from .speed import braking_cap, side_cap

NAV_OFFBOARD = 14
NAV_AUTO_LAND = 18
AIRBORNE = ("climb", "excite", "hover", "turn", "align", "check", "go", "hold", "blocked", "backoff", "recheck", "recovery_scan",
            "creep", "retreat")
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
    corridor_wait_land_s: float = -1.0          # ignored legacy parameter: retained for old route/config loading
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
    recovery_replan: bool = False              # explicit request/response; enabled by Sky Beta launch
    recovery_scan_deg: float = 0.0             # bounded yaw sweep, <=45deg each side; zero disables
    recovery_scan_rate_dps: float = 20.0
    recovery_scan_dwell_s: float = 0.5
    recovery_scan_route_m: float = 0.0         # > 0: centre the sweep on the blocked route's bearing over this
                                               # arc, starting at the navigator's rejoin lookahead (max(1 m,
                                               # stop distance)) past the block; the LAST look faces that route,
                                               # so the newest depth covers the rejoin when the last request goes.
                                               # 0 = the sweep is centred on the heading at the block (legacy)
    recovery_scan_route_max_deg: float = 90.0  # the sweep centre turns at most this far from the heading at the block
    recovery_timeout_s: float = 20.0           # whole recovery budget, never reset by planner replies
                                               # (with recovery_creep: only a hop that made progress extends it,
                                               # and never past creep_max_s after the block)
    recovery_creep: bool = False               # recovery_replan: accept short certified HOPS (navigator
                                               # recovery_hop paths) and fly them slowly, look again, repeat
    creep_speed_mps: float = 0.15              # hop speed: the carrot advances no faster than this
    creep_max_hops: int = 6                    # hops per recovery; then only full certified detours are accepted
    creep_max_stalls: int = 2                  # hops in a row that ended early (obstacle, unknown, withdrawn): land
    creep_max_s: float = 45.0                  # absolute recovery budget from the block, never extended
    creep_dwell_s: float = 1.0                 # hover after a hop before asking for the next (new depth evidence)
    creep_reply_s: float = 2.0                 # wait this long for that reply before sweeping
    creep_hop_timeout_s: float = 10.0          # a hop not finished in this long ends without progress
    creep_unobserved_s: float = 1.5            # no observed corridor for this long during a hop ends it
    creep_arrive_m: float = 0.10               # a hop is done within this of its end
    prior_required: bool = False                # the course map must describe the view (coursenav.priorstate):
                                                # no arming until a "placed" report, land when it reports "dropped"
    blocked_max_per_leg: int = 0                # > 0: the (N+1)-th confirmed block on one leg lands instead of another
                                                # hold-and-recheck (bag 20261107-201744: 18 blocks on leg 1, never landed)
    depth_lost_pause_static: bool = False       # depth silence is not counted while turning in place or holding
    stop_per_mps: float = 0.0                   # > 0: stop distance = max(stop_min_m, stop_base_m + this * v)
    stop_base_m: float = 0.45
    stop_min_m: float = 0.55                    # camera minimum range (0.4 m) + the camera's offset + margin
    min_valid_frac_static: float = -1.0         # >= 0: valid-pixel floor while NOT translating (else min_valid_frac)
    align_timeout_s: float = 4.0                # gate alignment: give up (hold-and-recheck, then land) after this
    align_frames: int = 3                       # ... agreeing measurements needed ...
    align_max_shift_m: float = 0.25             # ... largest lateral shift of the crossing ...
    align_max_dz_m: float = 0.15                # ... largest height shift ...
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
        if params.recovery_replan:
            bounds = {"recovery_timeout_s": (0.1, 30.0), "recovery_scan_deg": (0.0, 45.0),
                      "recovery_scan_rate_dps": (0.1, 20.0), "recovery_scan_dwell_s": (0.2, 2.0),
                      "recovery_scan_route_m": (0.0, 3.0), "recovery_scan_route_max_deg": (0.0, 135.0)}
            if params.recovery_creep:
                bounds.update({"creep_speed_mps": (0.05, 0.3), "creep_max_hops": (1, 12), "creep_arrive_m": (0.03, 0.3),
                               "creep_max_stalls": (1, 5), "creep_max_s": (5.0, 90.0),
                               "creep_dwell_s": (0.2, 5.0), "creep_reply_s": (0.2, 5.0),
                               "creep_hop_timeout_s": (1.0, 30.0), "creep_unobserved_s": (0.2, 5.0)})
            for name, (low, high) in bounds.items():
                value = getattr(params, name)
                if not math.isfinite(value) or not low <= value <= high:
                    raise ValueError(f"{name} must be finite and within [{low}, {high}]")
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
        self.plan_reference = None    # last valid path/progress, never used as a live command
        self.plan_failure = None      # normal no-route reply queued for supervised recovery in step()
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
                       "corridor_unobserved": 0, "corrections": 0, "recoveries": 0, "replan_resumes": 0, "blended_turns": 0, "path_waits": 0,
                       "creep_hops": 0, "creep_stalls": 0,
                       "alignments": 0, "tunnel_ticks": 0, "tube_holds": 0, "tube_ticks": 0, "retreats": 0,
                       "plan_stops": 0}
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
        self.replan_serial = 0
        self.replan_request = None
        self.plan_reply = None
        self.leg_blocks = 0           # confirmed blocks on the current leg (blocked_max_per_leg)
        self.prior = None             # latest prior-map report (t, state, fit) from the localizer
        self.clr_min = None           # the SMALLEST clearance of the confirm frames (recovery needs every one clear)
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
        leg = self.leg if leg is None else leg
        legs = self.p.legs or []
        return legs[leg] if 0 <= leg < len(legs) and isinstance(legs[leg], dict) else {}

    def on_correction(self, t, dn, de, sigma):
        """A map-based correction (m, NED) to ADD to this mission's PX4-frame targets, e.g. from a
        localizer matching depth against the course map. Only stored here; applied at leg starts."""
        self.corr = (float(t), float(dn), float(de), float(sigma))

    def _apply_correction(self, now, out, max_step=None, deadband=None, quiet=False):
        """Move the remaining targets towards the latest correction: at a leg start by up to
        correction_step_m (ignoring changes under the deadband), or continuously during a leg by a
        small per-tick amount (max_step), which keeps the aircraft on the corrected line between legs."""
        p = self.p
        if p.correction_max_m <= 0 or self.corr is None or self.leg >= len(self.targets):
            return
        t, dn, de, sig = self.corr
        if not all(math.isfinite(v) for v in (t, dn, de, sig)) or now - t > p.correction_stale_s \
                or sig > p.correction_max_sigma_m:
            return
        mag = math.hypot(dn, de)
        if mag > p.correction_max_m:
            dn, de = dn * p.correction_max_m / mag, de * p.correction_max_m / mag
        ddn, dde = dn - self.corr_applied[0], de - self.corr_applied[1]
        step = math.hypot(ddn, dde)
        if step < (p.correction_deadband_m if deadband is None else deadband):
            return
        cap = p.correction_step_m if max_step is None else max_step
        if step > cap:
            ddn, dde = ddn * cap / step, dde * cap / step
        self.targets[self.leg:] = [(n + ddn, e + dde) for n, e in self.targets[self.leg:]]
        self.corr_applied = (self.corr_applied[0] + ddn, self.corr_applied[1] + dde)
        self.counts["corrections"] += 1
        if quiet:
            self.counts["corrections_continuous"] = self.counts.get("corrections_continuous", 0) + 1
            return
        out.events.append(f"map correction: remaining targets moved ({ddn:+.2f} n, {dde:+.2f} e) m, "
                          f"total ({self.corr_applied[0]:+.2f}, {self.corr_applied[1]:+.2f}), sigma {sig:.2f}")

    def _follow_correction(self, now, dt, out):
        """During a leg: shift the targets a little towards the correction and keep the leg's
        geometry pointing at the (moved) target from the same start."""
        p = self.p
        if p.correction_rate_mps <= 0 or dt <= 0 or self.leg >= len(self.targets):
            return
        before = self.targets[self.leg]
        self._apply_correction(now, out, max_step=p.correction_rate_mps * dt, deadband=0.005, quiet=True)
        if self.targets[self.leg] != before:
            tn, te = self.targets[self.leg]
            n0, e0 = self.leg_start
            L = math.hypot(tn - n0, te - e0)
            if L > 1e-6:
                self.leg_len = L
                self.leg_dir = ((tn - n0) / L, (te - e0) / L)
                self.leg_heading = math.atan2(te - e0, tn - n0)

    def on_path(self, t, leg, pts_ne, speeds, stops=(), pid=0, tubes=None, request_id=None, recovery_certified=False,
                hop=False):
        """track_mode "path": the current path for leg `leg` in PX4 NED (refreshed by the navigator as
        the map correction and the live map change). pts_ne: (n, 2), speeds: (n,) the time-optimal profile,
        stops: indices of interior stop-and-turn corners; pid: changes when the path is RE-PLANNED (new
        geometry from the aircraft's position: progress restarts), not when it is only re-expressed in the
        flight frame. The leg's target becomes the path's end."""
        import numpy as _np
        if self.plan_failure is not None:
            return False  # the next supervised step must issue a certified recovery request first
        if self.recover is not None and self.p.recovery_replan:
            if (self.replan_request is None or request_id != self.replan_request["id"]
                    or leg != self.leg or recovery_certified is not True):
                return False
            if request_id == self.recover.get("consumed_request"):
                return False  # this request's hop has been flown (or abandoned): only a NEW request may move us
            if hop is True and not self.p.recovery_creep:
                return False
        elif request_id is not None or hop:
            return False  # delayed recovery replies cannot replace the resumed normal path
        try:
            pts = _np.asarray(pts_ne, float)
            sp = _np.asarray(speeds, float)
        except (ValueError, TypeError, OverflowError):
            return False
        if (pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2 or sp.shape != (len(pts),)
                or type(leg) is not int or leg < 0 or not math.isfinite(t)
                or leg >= len(self.targets or []) or not _np.all(_np.isfinite(pts)) or not _np.all(_np.isfinite(sp))):
            return False
        if any(type(i) is not int or not 0 <= i < len(pts) for i in stops):
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
        if self.path is None or self.path.get("pid") != pid or self.path.get("leg") != leg:
            self.path_s, self.path_stop_done = 0.0, -1
        tb = None
        if tubes is not None:
            tb = _np.asarray(tubes, float)
            if tb.shape != (len(pts),) or not _np.all(_np.isfinite(tb)):
                tb = None
        self.path = dict(t=float(t), leg=int(leg), pid=pid, pts=pts, s=s, v=_np.maximum(sp, 0.0), heading=heading,
                         kappa=kappa, stops=sorted(int(i) for i in stops), tube=tb)
        self.path["request_id"] = request_id
        self.path["recovery_certified"] = bool(recovery_certified)
        self.path["hop"] = hop is True
        if self.recover is not None and self.p.recovery_replan:
            self.plan_reply = {"state": "ready", "id": request_id, "t": float(t)}
        if hop is True:
            # a hop is a short certified step inside the recovery, NOT the leg: the leg's target, length and
            # the reference route a later request names all stay the original route's
            return True
        self.targets[leg] = (float(pts[-1, 0]), float(pts[-1, 1]))
        self.plan_reference = (self.path, self.path_s)
        if leg == self.leg and self._path_leg():
            self.leg_len = float(s[-1])
        return True

    def on_plan_result(self, now, leg, request_id, state, reason=""):
        """Explicit failure/pending invalidates a path; stale recovery replies cannot affect it."""
        if leg != self.leg or state not in ("pending", "failed"):
            return False
        if self.recover is not None and self.p.recovery_replan:
            if self.replan_request is None or request_id != self.replan_request["id"]:
                return False
        elif request_id is not None:
            return False
        if self.path is not None and self.path.get("leg") == leg:
            self.plan_reference = (self.path, self.path_s)
        if (state == "failed" and request_id is None and self.p.recovery_replan
                and self.recover is None and self.plan_failure is None
                and self.phase in ("turn", "check", "go") and self._path_leg()):
            reference = self.plan_reference
            if reference is None or reference[0]["leg"] != leg:
                reference = (None, 0.0)
            self.plan_failure = {"leg": leg, "reason": str(reason), "reference": reference}
        self.path = None
        if (self.sp is not None and self.last_v is not None and self.last_v.pos is not None
                and all(math.isfinite(c) for c in self.last_v.pos[:2])):
            self.sp[0], self.sp[1] = self.last_v.pos[:2]
        self.plan_reply = {"state": state, "id": request_id, "t": float(now), "reason": str(reason)}
        return True

    def _path_leg(self, leg=None):
        """Is leg `leg` flown along a navigator path (path mode, a cruise leg: no alignment, no passage)?"""
        if self.p.track_mode != "path":
            return False
        m = self._leg_meta(leg)
        # straight legs are the crossings: an opening to align on, or an enclosed passage. A leg whose mode is
        # "align"/"gate" without that metadata only ENDS at such a point (the approach): it is a path leg.
        return not (m.get("align") or m.get("tunnel"))

    def _path_fresh(self, now):
        P = self.path
        return P is not None and P["leg"] == self.leg and now - P["t"] <= self.p.path_stale_s

    def on_prior_state(self, t, state, fit=None):
        """The localizer's prior-map state (coursenav.priorstate): "placed" or "dropped"."""
        self.prior = (t, str(state), fit)

    def _prior_dropped_why(self):
        t, state, fit = self.prior
        return (f"prior map dropped: the course map does not describe what the camera sees "
                f"(scan fit {fit:.0%})" if fit is not None else "prior map dropped")

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
        self.plan_failure = None
        self.result = result
        self._set("done", now, out, result)

    def _land(self, now, out, why, immediate=False):
        """Land. With safe_land, a fault landing over a column that is not clear (or unknown) first
        retraces the flown path to the nearest clear column (phase retreat); immediate landings
        (no trustworthy position) and the planned landing ("route complete") descend where they are."""
        self.plan_failure = None
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
        if self.phase in AIRBORNE and self.phase != "retreat" and self.flown and v.pos is not None \
                and math.dist(self.flown[-1], v.pos[:2]) >= self.p.crumb_m:
            self.flown.append((v.pos[0], v.pos[1]))          # breadcrumbs of the path actually flown
        if self.t_phase is None:
            self.t_phase = now
        if self.t_started is None:
            self.t_started = now
        if self.phase == "done":
            return out

        armed_start_ok = p.accept_armed_start and v.landed is True
        if self.phase in ("wait", "arm") and self.t_cmd is None:
            if v.armed is True and not armed_start_ok:
                self._end(now, out, "yielded: vehicle armed before this mission requested control")
            elif now - self.t_started > p.wait_s:
                self._end(now, out, f"refused: {self.why or 'not ready'} after {p.wait_s:.0f} s")
            elif self.phase == "arm" and v.armed is None:
                self._requalify(now, out, "waiting for an explicit disarmed PX4 report")

        # ---- airborne supervision, before anything else may command -------------------
        if self.phase in AIRBORNE or (self.phase == "land" and not self.land_auto_seen):
            # only the mission's OWN landing may put PX4 in AUTO_LAND; anywhere else that is PX4 (a
            # failsafe) or a pilot taking over, and streaming on would blank PX4's land detector
            allowed = (NAV_OFFBOARD, NAV_AUTO_LAND) if self.phase == "land" else (NAV_OFFBOARD,)
            if self.offboard_seen and v.nav_state is not None and v.nav_state not in allowed:
                self._end(now, out, f"yielded: PX4 left Offboard (nav_state {v.nav_state}): a pilot or a failsafe has it")
                out.phase = self.phase
                return out
        if self.land_request is not None and self.phase in ("wait", "arm") + AIRBORNE:
            why = f"landing requested: {self.land_request}"
            if self.phase == "wait":
                self._end(now, out, f"refused: {why} before arming")
            elif self.phase == "arm":
                if v.armed and not armed_start_ok:
                    out.commands.append("disarm")          # a normal, checked disarm: on the ground, never forced
                self._end(now, out, f"refused: {why} while arming")
            else:
                self._land(now, out, why)
            self.land_request = None
            out.phase = self.phase
        if self.phase in AIRBORNE:
            bad = self.health.bad_for(now)
            tunnel_dark = self.tunnel_active and self.phase in ("check", "go") and self._tunnel_ok(now, v)[0]
            if bad > p.health_bad_land_s:
                self._land(now, out, f"estimator unhealthy or silent for {bad:.1f} s", immediate=True)
            elif not v.pos_ok or v.pos is None:
                self._land(now, out, "PX4 position or heading invalid", immediate=True)
            elif self.depth_t is None or (self._depth_silent_s(now) > p.depth_lost_land_s and not tunnel_dark):
                self._land(now, out, f"depth silent for more than {p.depth_lost_land_s:.0f} s")
            elif p.prior_required and self.prior is not None and self.prior[1] == "dropped":
                self._land(now, out, self._prior_dropped_why())
            elif self._begin_failed_plan_recovery(now, out, v):
                pass  # no-route recovery owns its existing bounded timer

        ph = self.phase
        if ph == "wait":
            ok, why = self.health.arm_ok(now)
            if now - self.t_started > p.wait_s:
                self._end(now, out, f"refused: {self.why or 'not ready'} after {p.wait_s:.0f} s")
            elif v.armed is not False and not (v.armed is True and armed_start_ok):
                self.why = ("waiting for an explicit disarmed PX4 report" if not p.accept_armed_start
                            else "waiting for a PX4 arming report (an armed vehicle must report landed)")
            elif not v.pos_ok or v.pos is None or v.heading is None:
                self.why = "waiting for PX4 position and heading"
            elif not ok:
                self.why = why
            elif not self.depth_fresh(now):
                self.why = "waiting for live depth: " + (
                    f"the latest frame was refused: {self.depth_reject[1]}" if self.depth_reject else
                    "no depth frame has arrived" if self.depth_t is None else
                    f"the latest usable frame is {now - self.depth_t:.1f} s old")
            elif not self.depth_ready_for_arm(now):
                self.why = f"waiting for {p.health_quiet_s:g} s of continuously usable depth"
            elif p.prior_required and self.prior is None:
                self.why = "waiting for the localizer's prior-map state (prior_required)"
            elif p.prior_required and self.prior[1] != "placed":
                self.why = "waiting: " + self._prior_dropped_why()
            else:
                n, e, d = v.pos
                self.home = (n, e, d, v.heading)
                self.home_rng = v.rng if v.rng is not None and v.rng > 0 else None
                self.flown = [(n, e)]
                self.sp = [n, e, d, v.heading]
                c, s = math.cos(v.heading), math.sin(v.heading)
                self.targets = [(n + f * c + l * s, e + f * s - l * c) for f, l in p.route]
                self._set("arm", now, out, "vehicle already armed on the ground: event start" if v.armed else "")
        elif ph == "arm":
            ok, why = self.health.arm_ok(now)
            if not ok:
                if self.t_cmd is None and v.armed is False:
                    self._requalify(now, out, f"{why}; requalifying before any control request")
                else:
                    self._end(now, out, f"refused: {why} before arming")
            elif self.t_cmd is None and not self.depth_ready_for_arm(now):
                if v.armed is False:
                    self._requalify(now, out, "depth not continuously usable; requalifying before any control request")
                else:
                    self._end(now, out, "yielded: depth lost before this mission requested control")
            elif now - self.t_phase > p.arm_wait_s:
                if v.armed and not armed_start_ok:
                    out.commands.append("disarm")          # a normal, checked disarm: on the ground, never forced
                self._end(now, out, f"refused: not armed in Offboard after {p.arm_wait_s:.0f} s")
            elif v.armed and v.nav_state == NAV_OFFBOARD:
                self.offboard_seen = True
                self._set("climb", now, out)
            elif now - self.t_phase >= p.stream_lead_s and self._resend(now):
                out.commands += ["offboard"] + ([] if v.armed else ["arm"])
        elif ph == "climb":
            target = self.home[2] - p.height_m
            self.sp[2] = max(target, self.sp[2] - p.climb_rate_mps * dt)
            if self.sp[2] <= target + 1e-9 and abs(v.pos[2] - target) < p.climb_tol_m:
                if p.excite_yaw_deg > 0 and self.excite is None:
                    self.excite = [_wrap(self.sp[3] + math.radians(p.excite_yaw_deg)), self.sp[3]]
                    self._set("excite", now, out, f"estimator excitation: yaw {p.excite_yaw_deg:.0f} deg in place and back")
                elif p.hover_s > 0:
                    self._set("hover", now, out, f"holding {p.hover_s:.0f} s at {p.height_m:.1f} m")
                else:
                    self._next_leg(now, out, v)
        elif ph == "excite":
            if self.excite:
                err = _wrap(self.excite[0] - self.sp[3])
                step = math.radians(p.yaw_rate_dps) * dt
                self.sp[3] = _wrap(self.sp[3] + max(-step, min(step, err)))
                if abs(err) < 1e-3 and abs(_wrap(self.excite[0] - v.heading)) < math.radians(p.face_tol_deg):
                    self.excite.pop(0)
                    self.excite_t = now
            elif now - self.excite_t >= p.excite_settle_s:
                if p.hover_s > 0:
                    self._set("hover", now, out, f"holding {p.hover_s:.0f} s at {p.height_m:.1f} m")
                else:
                    self._next_leg(now, out, v)
        elif ph == "hover":
            if now - self.t_phase >= p.hover_s:
                self._next_leg(now, out, v)
        elif ph == "turn":
            if (self._path_leg() and self._path_fresh(now) and self.path_stop_done < 0 and self.path_s == 0.0):
                self.leg_heading = self._path_at(p.path_lookahead_m)[1]   # face where the path first goes
            elif (p.path_turn_hold and self._path_leg() and self.path_stop_done < 0 and self.path_s == 0.0):
                if self.leg_heading != self.sp[3]:
                    out.events.append(f"turn: no path yet for leg {self.leg + 1}: keeping the heading "
                                      "(not the straight bearing to the stop); go waits for the path")
                self.leg_heading = self.sp[3]
            err = _wrap(self.leg_heading - self.sp[3])
            if p.yaw_accel_dps2 > 0:
                out.yawspeed = self._slew_yaw(err, dt)
            else:
                step = math.radians(p.yaw_rate_dps) * dt
                self.sp[3] = _wrap(self.sp[3] + max(-step, min(step, err)))
            if abs(err) < 1e-3 and abs(_wrap(self.leg_heading - v.heading)) < math.radians(p.face_tol_deg):
                self.yaw_rate_cmd = 0.0
                out.yawspeed = 0.0 if p.yaw_accel_dps2 > 0 else None
                self._after_turn(now, out)
        elif ph == "align":
            self._align(now, out, v)
        elif ph == "check":
            settled = now - self.t_phase >= p.turn_settle_s
            if self.first_turn_pending:
                off = math.hypot(v.pos[0] - self.sp[0], v.pos[1] - self.sp[1])
                spd = math.hypot(*v.vel[:2]) if v.vel is not None else math.inf
                if (off < p.first_turn_gate_m and spd < p.first_turn_gate_mps) or now - self.t_phase >= p.first_turn_gate_s:
                    self.first_turn_pending = False
                    out.events.append(f"first-turn gate released after {now - self.t_phase:.1f} s "
                                      f"({off:.3f} m off the setpoint, {spd:.3f} m/s)")
                else:
                    settled = False
            # a depth frame captured AFTER the turn ended looks along this leg
            if settled and self.depth_t is not None and self.depth_t >= self.t_phase and self.depth_fresh(now):
                if self.clr_hist and self.clr_hist[0][0] < self.t_phase:
                    self.clr_hist, self.corr_t = [], None    # frames from before the turn ended look elsewhere
                clr = self._clearance(v)                     # None: not enough observed frames yet, wait
                if clr is not None and self.corridor_fresh(now):
                    self.last_clearance = clr
                    if self._blocks(clr, self.leg_len, 0.0):
                        self._obstacle(now, out, v, clr, "at")
                    else:
                        self.s = 0.0
                        self._set("go", now, out)
            if self.phase == "check" and self.tunnel_active and settled:
                ok, _ = self._tunnel_ok(now, v)
                if ok and not self.corridor_fresh(now):
                    self.s = 0.0
                    self._set("go", now, out, f"tunnel mode: leg {self.leg + 1} enters the passage on the map alone")
        elif ph == "go":
            self._follow_correction(now, dt, out)
            self._go(now, dt, out, v)
        elif ph == "hold":
            if now - self.t_phase >= p.hold_s:
                self.legs_done += 1
                self.leg += 1
                self._next_leg(now, out, v)
        elif ph in ("blocked", "backoff", "recheck", "recovery_scan", "creep"):
            self._recover(now, dt, out, v)
        elif ph == "retreat":
            self._retreat(now, dt, out, v)

        if self.phase == "land":
            if v.nav_state == NAV_AUTO_LAND:
                self.land_auto_seen = True
            self._stuck_landing(now, out, v)
            if not self.land_auto_seen and self._resend(now):
                out.commands.append("land")
            if self.land_auto_seen and v.armed is False:
                why = self.land_reason or ""
                self._end(now, out, "complete" if why == "route complete" else f"landed: {why}")
            elif now - self.t_phase > p.land_timeout_s:
                self._end(now, out, f"landed: {self.land_reason}; landing not confirmed within {p.land_timeout_s:.0f} s")

        if self.phase != "go":
            self.v_cmd, self.a_cmd = 0.0, 0.0
            self.v_vec = self.a_vec = None
        if self.phase not in ("go", "hold", "retreat"):
            self.slow_since = None
        out.phase = self.phase
        out.setpoint = tuple(self.sp) if self.sp is not None else None
        # yaw_accel_dps2 > 0: whenever the yaw setpoint moves (in-place turns, and path-mode arcs that yaw
        # while moving), publish its rate as yawspeed feed-forward instead of the message default 0
        if p.yaw_accel_dps2 > 0 and self.sp is not None:
            if out.yawspeed is None and self.last_sp_yaw is not None and dt > 0:
                out.yawspeed = _wrap(self.sp[3] - self.last_sp_yaw) / dt
            self.last_sp_yaw = self.sp[3]
        pose = p.track_mode in ("pose",)
        path_go = self.phase == "go" and self.v_vec is not None and self._path_leg()
        if path_go:
            out.velocity, out.accel = self.v_vec, self.a_vec
        elif (p.accel_mps2 > 0 or pose or p.track_mode == "path") and self.phase in AIRBORNE and self.leg_dir is not None:
            out.velocity = (self.leg_dir[0] * self.v_cmd, self.leg_dir[1] * self.v_cmd)
            if pose or p.track_mode == "path":
                out.accel = (self.leg_dir[0] * self.a_cmd, self.leg_dir[1] * self.a_cmd)
        # PX4 owns the landing once AUTO_LAND is seen: nothing streams from then on
        out.stream = self.phase in ("arm",) + AIRBORNE or (self.phase == "land" and not self.land_auto_seen)
        return out

    # ------------------------------------------------------------------ legs
    def _stops_at_end(self, leg):
        """Does leg `leg` end with a full stop (hold, turn, check)? Gate/alignment/passage legs and
        their predecessors always do; otherwise a corner smaller than blend_turn_deg is blended."""
        p = self.p
        if p.blend_turn_deg <= 0 or leg + 1 >= len(self.targets):
            return True
        if self._leg_meta(leg).get("hold") or self._leg_meta(leg + 1).get("hold"):
            return True
        for k in (leg, leg + 1):
            m = self._leg_meta(k)
            if m.get("align") or m.get("tunnel") or m.get("mode") in ("gate", "tunnel", "align"):
                return True
        a = self.targets[leg]
        b = self.targets[leg + 1]
        start = self.leg_start if leg == self.leg and self.leg_start is not None else (
            self.targets[leg - 1] if leg > 0 else (self.home[0], self.home[1]))
        h1 = math.atan2(a[1] - start[1], a[0] - start[0])
        h2 = math.atan2(b[1] - a[1], b[0] - a[0])
        return abs(_wrap(h2 - h1)) >= math.radians(p.blend_turn_deg)

    def _next_leg(self, now, out, v, blended=False):
        self._apply_correction(now, out)
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
        self.leg_blocks = 0
        self.replan_request = self.plan_reply = None
        self.plan_failure = None
        self.plan_reference = ((self.path, self.path_s)
                               if self.path is not None and self.path.get("leg") == self.leg else None)
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
        if self._leg_meta().get("align") and self.align_ok_leg != self.leg:
            self._set("align", now, out, f"aligning on {self._leg_meta().get('name', 'the gate')}")
        else:
            self._set("check", now, out)

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
                # the same along-path check in go, check AND the recovery: judging recovery on the straight
                # leg heading while go judged the curved path flapped blocked <-> go 18 times (bag 20261107-201744)
                ahead = (self._path_ahead_level(v) if self.phase in ("go", "check", "blocked", "backoff", "recheck", "creep")
                         and self._path_leg()
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
        self.clr_min = min(c for _, c in self.clr_hist)
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

    def _obstacle(self, now, out, v, clr, where):
        """A confirmed obstacle inside the stop distance: stop where the aircraft is; land (the
        original rule) or hold-and-recheck (blocked_hold_s > 0)."""
        self.sp[0], self.sp[1] = v.pos[0], v.pos[1]
        self.counts["obstacle_stops"] += 1
        self.leg_blocks += 1
        why = f"route blocked: obstacle {clr:.2f} m ahead {where} leg {self.leg + 1}"
        if self.p.blocked_max_per_leg > 0 and self.leg_blocks > self.p.blocked_max_per_leg:
            self.recover = None
            self._land(now, out, f"{why}; blocked {self.leg_blocks} times on this leg "
                                 f"(blocked_max_per_leg {self.p.blocked_max_per_leg})")
        elif self.p.blocked_hold_s > 0:
            self._start_recover(now, out, "blocked", why, v)
        else:
            self._land(now, out, why)

    def _go(self, now, dt, out, v):
        p = self.p
        if p.track_mode == "path" and self._path_leg():
            return self._go_path(now, dt, out, v)
        if p.track_mode in ("pose", "path"):
            return self._go_pose(now, dt, out, v)
        n0, e0 = self.leg_start
        un, ue = self.leg_dir
        proj = (v.pos[0] - n0) * un + (v.pos[1] - e0) * ue
        if p.blend_turn_deg > 0 or self.leg_v_end > 0:
            err = _wrap(self.leg_heading - self.sp[3])      # a blended corner turns while moving
            step = math.radians(p.yaw_rate_dps) * dt
            self.sp[3] = _wrap(self.sp[3] + max(-step, min(step, err)))
        clr = self._clearance(v) if self.depth_fresh(now) else None
        if clr is not None and not self.corridor_fresh(now):
            clr = None                                       # the latest evidence of this corridor is too old
        s_old = self.s
        if clr is not None:
            self.last_clearance = clr
            if self._blocks(clr, self.leg_len - max(proj, 0.0)):
                self._obstacle(now, out, v, clr, "on")
                self.v_cmd = 0.0
                return
            self._advance(dt, proj, min(self._profile_speed(self.s), self._geometry_cap(clr, v)))
        else:
            tunnel_ok, _ = self._tunnel_ok(now, v) if self.tunnel_active else (False, "")
            if tunnel_ok:
                if self.tunnel_entry_t is None:
                    self.tunnel_entry_t = now
                    out.events.append(f"tunnel mode: entered leg {self.leg + 1} without depth returns at "
                                      f"{self.s:.2f} m of {self.leg_len:.2f} m")
                self.counts["tunnel_ticks"] += 1
                self._advance(dt, proj, min(p.tunnel_speed_mps, self._profile_speed(self.s)))
            else:
                self.counts["depth_holds"] += 1              # no live evidence: the carrot waits
        self.v_cmd = (self.s - s_old) / dt if dt > 0 else 0.0
        self.sp[0], self.sp[1] = n0 + un * self.s, e0 + ue * self.s
        tn, te = self.targets[self.leg]
        if self.s >= self.leg_len - 1e-6 and math.hypot(v.pos[0] - tn, v.pos[1] - te) < p.arrive_m \
                and (self.leg_v_end > 0 or self._settled(now, dt, v)):
            if self.tunnel_entry_t is not None:
                out.events.append(f"tunnel mode: left leg {self.leg + 1} (passage flown)")
            if self.leg_v_end > 0:
                self.legs_done += 1
                self.leg += 1
                self._next_leg(now, out, v, blended=True)
            else:
                self._set("hold", now, out, f"arrived leg {self.leg + 1}")

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
        if self.path_stop_done >= 0:
            # ... and never back behind a corner already turned at: arriving up to arrive_m short of the corner,
            # the projection falls on the segment BEFORE it, and the 0.05 m/tick allowance walked progress back
            # there, commanding along the old tangent (sideways after the turn) at a creep for ever
            # (+1 mm: _path_at at exactly a vertex returns the INCOMING segment's tangent)
            self.path_s = max(self.path_s, float(P["s"][self.path_stop_done]) + 1e-3)
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
                self._obstacle(now, out, v, clr, "on")
                self.v_cmd, self.a_cmd = 0.0, 0.0
                self.v_vec, self.a_vec = (0.0, 0.0), (0.0, 0.0)
                return
        else:
            self.counts["depth_holds"] += 1
            cap = 0.0
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
            else:
                self.sp[0], self.sp[1] = float(end[0]), float(end[1])
                self._set("hold", now, out, f"arrived leg {self.leg + 1}")

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

    def _arrived(self, now, dt, v, tn, te, captured):
        """Stopping-leg arrival. arrive_tight_m > 0: within arrive_tight_m of the end and slower than
        arrive_tight_mps for arrive_settle_s, so the next in-place turn starts from rest ON the vertex
        (run021: the looser test started turns 3.6 cm short at 3.5 cm/s median, 13 cm / 7.8 cm/s worst;
        the unfinished arrival then ran on through the turn and overshot by up to 11 cm, the visible
        reversal). After arrive_timeout_s in the capture zone, the looser arrive_m / arrive_speed_mps test."""
        p = self.p
        d = math.hypot(v.pos[0] - tn, v.pos[1] - te)
        if p.arrive_tight_m <= 0:
            return d < p.arrive_m and self._settled(now, dt, v)
        if captured and self.capture_t is None:
            self.capture_t = now
        if self.capture_t is not None and now - self.capture_t >= p.arrive_timeout_s:
            return d < p.arrive_m and self._settled(now, dt, v)
        spd = self._speed(v, dt)
        if d < p.arrive_tight_m and spd < p.arrive_tight_mps:
            if self.tight_since is None:
                self.tight_since = now
            return now - self.tight_since >= p.arrive_settle_s
        self.tight_since = None
        return False

    def _go_pose(self, now, dt, out, v):
        """Path following from the actual pose (track_mode "pose"). Same safety gates as the carrot:
        a confirmed obstacle inside the stop distance stops, no live evidence of the corridor stops
        forward motion (tunnel mode excepted), arrival needs the end reached AND (arrive_speed_mps)
        the aircraft slow."""
        p = self.p
        n0, e0 = self.leg_start
        un, ue = self.leg_dir
        proj = (v.pos[0] - n0) * un + (v.pos[1] - e0) * ue
        remaining = self.leg_len - proj
        if p.blend_turn_deg > 0 or self.leg_v_end > 0:
            err = _wrap(self.leg_heading - self.sp[3])
            step = math.radians(p.yaw_rate_dps) * dt
            self.sp[3] = _wrap(self.sp[3] + max(-step, min(step, err)))
        clr = self._clearance(v) if self.depth_fresh(now) else None
        if clr is not None and not self.corridor_fresh(now):
            clr = None
        cap = self._leg_speed()
        if self._tube_holds(now, out, v):
            cap = 0.0
        if clr is not None:
            cap = min(cap, self._geometry_cap(clr, v))
            self.last_clearance = clr
            if self._blocks(clr, max(remaining, 0.0)):
                self._obstacle(now, out, v, clr, "on")
                self.v_cmd, self.a_cmd = 0.0, 0.0
                return
        else:
            tunnel_ok, _ = self._tunnel_ok(now, v) if self.tunnel_active else (False, "")
            if tunnel_ok:
                if self.tunnel_entry_t is None:
                    self.tunnel_entry_t = now
                    out.events.append(f"tunnel mode: entered leg {self.leg + 1} without depth returns at "
                                      f"{max(proj, 0.0):.2f} m of {self.leg_len:.2f} m")
                self.counts["tunnel_ticks"] += 1
                cap = min(cap, p.tunnel_speed_mps)
            else:
                self.counts["depth_holds"] += 1          # no live evidence: stop where we are (latched setpoint)
                cap = 0.0
        a_b = p.brake_mps2 if p.brake_mps2 > 0 else (p.accel_mps2 if p.accel_mps2 > 0 else 0.5)
        a_up = p.accel_mps2 if p.accel_mps2 > 0 else a_b
        stopping = self.leg_v_end <= 0
        v_brake = math.sqrt(self.leg_v_end ** 2
                            + 2.0 * a_b * max(remaining - (p.capture_m if stopping else 0.0), 0.0))
        v_along = self._along_speed(v, (un, ue))
        v_ramp = self._ramp_from(v_along) + a_up * dt
        v_ref = max(0.0, min(cap, v_brake, v_ramp))
        # feed-forward acceleration of whichever limit binds (planned, not differentiated)
        if v_ref <= 0.0:
            a_ff = 0.0
        elif v_ref == v_brake and v_brake < cap:
            a_ff = -a_b
        elif v_ref == v_ramp and v_ramp < cap:
            a_ff = a_up
        else:
            a_ff = 0.0
        a_ff = self._consistent_accel(a_ff, v_ref, v_along, dt, a_b)
        captured = stopping and remaining <= p.capture_m
        if captured:
            self.s, v_ref, a_ff = self.leg_len, 0.0, 0.0             # hold the leg's end itself
        elif v_ref > 1e-3 or self.tube_hold:
            # beside the aircraft: no along-track pull. A tube hold only stops ADVANCING; latching the
            # setpoint there made PX4 pull the still-coasting aircraft back along the leg (run013 legs
            # 2-5: 7-13 cm reversals); beside it, PX4 corrects across the leg and brakes along it
            self.s = min(max(proj, 0.0), self.leg_len)
        # else: latched where forward motion last stopped (PX4 holds it)
        self.v_cmd, self.a_cmd = v_ref, a_ff
        self.sp[0], self.sp[1] = n0 + un * self.s, e0 + ue * self.s
        tn, te = self.targets[self.leg]
        if not stopping and remaining <= max(p.arrive_m, 0.0):
            if self.tunnel_entry_t is not None:
                out.events.append(f"tunnel mode: left leg {self.leg + 1} (passage flown)")
            self.sp[0], self.sp[1] = tn, te                          # the next leg starts at the waypoint
            self.legs_done += 1
            self.leg += 1
            self._next_leg(now, out, v, blended=True)
            return
        if stopping and self._arrived(now, dt, v, tn, te, captured):
            if self.tunnel_entry_t is not None:
                out.events.append(f"tunnel mode: left leg {self.leg + 1} (passage flown)")
            self.sp[0], self.sp[1] = tn, te
            self.v_cmd, self.a_cmd = 0.0, 0.0
            self._set("hold", now, out, f"arrived leg {self.leg + 1}")

    def _tube_holds(self, now, out, v):
        """tube_gate: is the aircraft too far across the leg to keep advancing? Outside tube_m (from
        the leg's metadata) -> hold forward motion; resume inside tube_m / 2 (hysteresis, so PX4 has
        pulled it most of the way back rather than sitting on the edge). Measured on PX4's estimate:
        this bounds tracking error only; estimation error has the other half of the leg's slack."""
        tube = self._leg_meta().get("tube_m") if self.p.tube_gate else None
        if not tube or v.pos is None or self.leg_start is None:
            self.tube_hold = False
            return False
        n0, e0 = self.leg_start
        un, ue = self.leg_dir
        across = abs(-(v.pos[0] - n0) * ue + (v.pos[1] - e0) * un)
        if not self.tube_hold and across > tube:
            self.tube_hold = True
            self.counts["tube_holds"] += 1
            out.events.append(f"tube: {across:.3f} m across leg {self.leg + 1} > {tube:.3f} m: holding until "
                              f"back inside {tube / 2:.3f} m")
        elif self.tube_hold and across <= tube / 2:
            self.tube_hold = False
            out.events.append(f"tube: back inside {tube / 2:.3f} m on leg {self.leg + 1}: advancing")
        if self.tube_hold:
            self.counts["tube_ticks"] += 1
        return self.tube_hold

    def _advance(self, dt, proj, speed):
        p = self.p
        if p.accel_mps2 > 0:
            speed = min(speed, self.v_cmd + p.accel_mps2 * dt)   # ramp up from rest
        self.s = min(self.s + speed * dt, self.leg_len, max(proj, 0.0) + p.lead_max_m)

    # ------------------------------------------------------------------ tunnel mode
    def _tunnel_ok(self, now, v):
        """Every condition for advancing through a detected enclosed passage without depth returns.
        -> (ok, why not)."""
        p = self.p
        m = self._leg_meta()
        t = m.get("tunnel")
        if not (p.tunnel_mode and t):
            return False, "not a passage leg"
        if not t.get("map_clear"):
            return False, "the map does not show the whole leg clear"
        if self.leg_len > p.tunnel_max_leg_m or float(t.get("length_m", self.leg_len)) > p.tunnel_max_leg_m:
            return False, f"leg longer than tunnel_max_leg_m ({self.leg_len:.2f} m)"
        if self.align_ok_leg != self.leg or self.align_ok_t is None:
            return False, "no successful gate alignment at the passage mouth"
        entry = self.tunnel_entry_t if self.tunnel_entry_t is not None else now
        if entry - self.align_ok_t > p.tunnel_align_max_age_s:
            return False, f"alignment {entry - self.align_ok_t:.1f} s old at entry"
        if v.heading is None or abs(_wrap(self.leg_heading - v.heading)) > math.radians(p.tunnel_face_tol_deg):
            return False, "not facing down the passage axis"
        if abs(_wrap(self.leg_heading - self.sp[3])) > 1e-6:
            return False, "yaw setpoint is turning"
        if self.recover is not None:
            return False, "recovering"
        return True, "ok"

    # ------------------------------------------------------------------ gate alignment
    def _measure_gate(self, v):
        """One alignment measurement from the latest frame, expressed in the LEG frame (the body may
        still be a few degrees off the leg heading when it measures)."""
        meta = self._leg_meta().get("align") or {}
        est = G.find_opening(self.frame.points(0.3, 6.0), meta)
        if est is None:
            return None
        bearing = _wrap(self.leg_heading - v.heading) if v.heading is not None else 0.0
        return G.to_leg_frame(est, bearing)

    def _align(self, now, out, v):
        p = self.p
        if self.frame is not None and self._align_judged_t != self.depth_t and self.depth_fresh(now):
            self._align_judged_t = self.depth_t
            est = self._measure_gate(v)
            if est is not None:
                self.align_hist.append(est)
        if len(self.align_hist) >= max(1, p.align_frames):
            res = G.combine(self.align_hist[-p.align_frames:])
            self._apply_alignment(now, out, res)
            self._set("check", now, out)
            return
        if now - self.t_phase > p.align_timeout_s:
            why = (f"gate not found: {len(self.align_hist)}/{p.align_frames} measurements in "
                   f"{p.align_timeout_s:g} s on leg {self.leg + 1}")
            if p.blocked_hold_s > 0:
                self._start_recover(now, out, "align", why, v)
            else:
                self._land(now, out, why)

    def _apply_alignment(self, now, out, res):
        """Shift the crossing (this leg's target and the following gate targets) by the measured
        offset, bounded; record the success for tunnel mode."""
        p = self.p
        lat = max(-p.align_max_shift_m, min(p.align_max_shift_m, res["lateral_m"]))
        dz = 0.0 if res.get("dz_m") is None else max(-p.align_max_dz_m, min(p.align_max_dz_m, res["dz_m"]))
        turn = max(-math.radians(p.align_max_turn_deg), min(math.radians(p.align_max_turn_deg), res.get("heading_rad", 0.0)))
        un, ue = self.leg_dir
        ln, le = ue, -un                                      # unit vector to the LEFT of the leg (NED, yaw clockwise)
        group = self._leg_meta().get("align", {}).get("group")
        k_last = self.leg
        for k in range(self.leg, len(self.targets)):
            if k != self.leg and (self._leg_meta(k).get("gate_group") != group or group is None):
                break
            k_last = k
        centre = self.targets[self.leg]
        for k in range(self.leg, k_last + 1):
            n, e = self.targets[k]
            n, e = n + ln * lat, e + le * lat
            if k > self.leg and abs(turn) > 0:                # rotate the exit about the (shifted) gate centre
                cn, ce = centre[0] + ln * lat, centre[1] + le * lat
                dn, de = n - cn, e - ce
                c, s = math.cos(turn), math.sin(turn)
                n, e = cn + c * dn - s * de, ce + s * dn + c * de
            self.targets[k] = (n, e)
        if dz:
            self.sp[2] -= dz                                  # NED: up is negative
        tn, te = self.targets[self.leg]
        n0, e0 = self.leg_start
        self.leg_len = math.hypot(tn - n0, te - e0)
        self.leg_dir = ((tn - n0) / self.leg_len, (te - e0) / self.leg_len) if self.leg_len > 1e-6 else self.leg_dir
        self.leg_heading = math.atan2(te - e0, tn - n0) if self.leg_len > 1e-6 else self.leg_heading
        self.align_ok_t, self.align_ok_leg, self.align_result = now, self.leg, res
        self.last_align = res
        self.counts["alignments"] += 1
        out.events.append(f"gate aligned: lateral {lat:+.2f} m, height {dz:+.2f} m, heading {math.degrees(turn):+.1f} deg "
                          f"(width seen {res.get('width_m', float('nan')):.2f} m, {len(self.align_hist)} frames)")

    # ------------------------------------------------------------------ hold-and-recheck
    def _begin_failed_plan_recovery(self, now, out, v):
        failure = self.plan_failure
        if failure is None:
            return False
        self.plan_failure = None
        if (failure["leg"] != self.leg or self.phase not in ("turn", "check", "go")
                or not self._path_leg() or self.recover is not None):
            return False
        why = f"navigator has no route for leg {self.leg + 1}: {failure['reason']}"
        self.sp[0], self.sp[1] = v.pos[:2]
        self.counts["plan_stops"] += 1
        self.leg_blocks += 1
        if self.p.blocked_max_per_leg > 0 and self.leg_blocks > self.p.blocked_max_per_leg:
            self._land(now, out, f"{why}; blocked {self.leg_blocks} times on this leg "
                                 f"(blocked_max_per_leg {self.p.blocked_max_per_leg})")
        elif self.p.blocked_hold_s <= 0:
            self._land(now, out, why)
        else:
            reference, progress = failure["reference"]
            self._start_recover(now, out, "blocked", why, v, reference, progress)
            self.recover["cause"] = "no_route"
            self.replan_request["require_detour"] = False
        return True

    def _start_recover(self, now, out, kind, why, v, reference=None, reference_s=None):
        old_path = self.path if reference is None else reference
        old_s = self.path_s if reference_s is None else reference_s
        self.recover = {"kind": kind, "t0": now, "why": why, "target": None, "t_back": None, "t_re": None,
                        "pid": old_path.get("pid") if (old_path is not None and self._path_leg()) else None,
                        "pos": (v.pos[0], v.pos[1]), "yaw": self.sp[3],
                        "old_path": old_path, "old_path_s": old_s, "scan_done": False,
                        "requests": 0, "hops": 0, "stalls": 0, "deadline": now + self.p.recovery_timeout_s,
                        "consumed_request": None, "hop": None, "await_reply": None}
        self.sp[0], self.sp[1] = v.pos[:2]   # brake at the measured pose, not the old forward carrot
        if kind == "blocked" and self._path_leg() and self.p.recovery_replan:
            self._request_recovery_plan(now)
        self._set("blocked", now, out, f"{why}; holding up to {self.p.blocked_hold_s:g} s")

    def _request_recovery_plan(self, now):
        if self.recover is None or self.recover["requests"] >= 5:
            return
        self.replan_serial += 1
        self.recover["requests"] += 1
        self.replan_request = {"id": self.replan_serial, "leg": self.leg, "observed_only": True,
                               "reference_pid": self.recover["pid"],
                               "reference_progress_s": float(self.recover["old_path_s"]),
                               "require_detour": self.recover.get("cause") != "no_route"}
        if self.p.recovery_creep:
            # the navigator may answer with a short certified hop only while hops remain
            self.replan_request["creep"] = self.recover.get("hops", 0) < self.p.creep_max_hops
            self.replan_request["creep_hops"] = int(self.recover.get("hops", 0))
        self.plan_reply = {"state": "pending", "id": self.replan_serial, "t": now}
        self.path = None   # old geometry retained only as a backoff reference, never a fresh command

    def _begin_backoff(self, now, out, v):
        r, p = self.recover, self.p
        back = self.leg_dir if r["kind"] == "blocked" or self.prev_leg_dir is None else self.prev_leg_dir
        d = p.blocked_backoff_m
        if r["kind"] == "blocked":
            if self._path_leg():
                d = min(d, max(r["old_path_s"], 0.0))
                P = r["old_path"]
                if P is not None:
                    import numpy as np
                    q = max(0.0, r["old_path_s"] - d)
                    target = tuple(float(np.interp(q, P["s"], P["pts"][:, i])) for i in range(2))
                    # Don't traverse a long chord after drift/replanning; existing backoff bound still applies.
                    dist = math.hypot(target[0]-v.pos[0], target[1]-v.pos[1])
                    if dist > d + 0.05:
                        target = (v.pos[0], v.pos[1])
                else:
                    target = (v.pos[0], v.pos[1])
            else:
                proj = sum((v.pos[i] - self.leg_start[i]) * back[i] for i in range(2))
                d = min(d, max(proj, 0.0))
                target = (v.pos[0]-back[0]*d, v.pos[1]-back[1]*d)
        else:
            target = (v.pos[0]-back[0]*d, v.pos[1]-back[1]*d)
        r["target"], r["t_back"] = target, now
        self.sp[0], self.sp[1] = target
        self._set("backoff", now, out, f"still blocked: backing off up to {d:.2f} m")

    def _route_scan_centre(self, r, v):
        """Yaw the recovery sweep is centred on.

        Legacy (recovery_scan_route_m == 0, no reference route, no usable pose): the heading at the block.
        Otherwise: the circular mean bearing from the aircraft to the blocked route over
        [s0 + lookahead, s0 + lookahead + recovery_scan_route_m], where lookahead = max(1 m, stop distance)
        is the navigator's own rejoin lookahead. Those are the only route points plan_recovery may rejoin,
        and the camera must see them, with their clearance disc, for any of them to be certified free.
        A route that turns off the nose otherwise stays at or beyond the 38 deg edge of the field of view
        (DIVERSE01: 36-73 deg right of the nose; a sweep centred on the nose certified nothing).
        The turn is bounded by recovery_scan_route_max_deg; it is a yaw in place, never a translation.
        """
        p, home = self.p, r["yaw"]
        P = r.get("old_path")
        if (p.recovery_scan_route_m <= 0 or P is None or v.pos is None
                or not all(math.isfinite(c) for c in v.pos[:2])):
            return home
        import numpy as np
        s = np.asarray(P["s"], float)
        pts = np.asarray(P["pts"], float)
        if len(s) < 2 or len(pts) != len(s):
            return home
        s0 = float(r.get("old_path_s") or 0.0)
        # After a recovery hop the aircraft is no longer at the block: start the window from where it now
        # projects onto the route (searched only ahead of the block, so a looping route cannot pull it back),
        # never before the block itself. The bearing below is always taken from the current position.
        ahead = (s >= s0) & (s <= s0 + 5.0)
        if ahead.any():
            dist = np.linalg.norm(pts[ahead] - np.asarray(v.pos[:2], float), axis=1)
            s0 = max(s0, float(s[ahead][int(np.argmin(dist))]))
        a = s0 + max(1.0, self._stop_m(0.0))
        q = np.clip(np.linspace(a, a + p.recovery_scan_route_m, 9), 0.0, float(s[-1]))
        d = np.stack([np.interp(q, s, pts[:, 0]) - v.pos[0], np.interp(q, s, pts[:, 1]) - v.pos[1]], axis=1)
        n = np.linalg.norm(d, axis=1)
        d = d[n > 0.2] / n[n > 0.2, None]     # a point on top of the aircraft has no bearing
        if not len(d):
            return home
        mean = d.mean(axis=0)
        if np.linalg.norm(mean) < 0.2:          # route points spread all round: no single direction to face
            return home
        off = _wrap(math.atan2(mean[1], mean[0]) - home)
        cap = math.radians(p.recovery_scan_route_max_deg)
        return _wrap(home + max(-cap, min(cap, off)))

    def _scan_recovery(self, now, dt, out, v):
        r, p = self.recover, self.p
        # Never rotate on missing depth, unsettled translation, or near obstacle returns.
        if (not self.depth_fresh(now) or v.vel is None or len(v.vel) < 3
                or not all(math.isfinite(c) for c in v.vel) or math.hypot(*v.vel[:2]) > 0.08
                or abs(v.vel[2]) > 0.08 or v.heading is None or not math.isfinite(v.heading)):
            r["settled"] = None
            return
        if (not all(math.isfinite(c) for c in v.pos)
                or math.dist(v.pos[:3], self.sp[:3]) > 0.10):
            r["settled"] = None
            return
        import numpy as np
        if self.depth_pts is not None and len(self.depth_pts):
            pts = self.depth_pts
            near = np.abs(pts[:, 2]) < p.band_m
            if np.any(np.linalg.norm(pts[near, :2], axis=1) < max(p.corridor_half_m, 0.45)):
                r["scan_done"] = True
                self._begin_backoff(now, out, v)
                return
        target = r["scan_yaws"][r["scan_i"]]
        err = _wrap(target-self.sp[3])
        step = math.radians(min(p.yaw_rate_dps, max(0.0, p.recovery_scan_rate_dps), 20.0))*dt
        self.sp[3] = _wrap(self.sp[3] + max(-step, min(step, err)))
        if abs(_wrap(target-v.heading)) > math.radians(3):
            r["settled"] = None
            return
        if r.get("settled") is None:
            r["settled"], r["settled_depth"] = now, self.depth_t
        if (now-r["settled"] < max(0.2, p.recovery_scan_dwell_s)
                or self.depth_t <= r["settled_depth"]):
            return
        self._request_recovery_plan(now)
        r["scan_i"] += 1
        r["settled"] = None
        if r["scan_i"] == len(r["scan_yaws"]):
            r["scan_done"] = True
            self._begin_backoff(now, out, v)

    # ------------------------------------------------------------------ creep hops (recovery_creep)
    def _creep_stop_m(self):
        """Stop distance at the hop speed (the leg's cruise profile does not apply to a creep)."""
        p = self.p
        if p.stop_per_mps <= 0:
            return p.stop_m
        return max(p.stop_min_m, p.stop_base_m + p.stop_per_mps * p.creep_speed_mps)

    def _begin_hop(self, now, out, v):
        """A certified hop arrived: face its first segment in place, then fly it at creep speed."""
        r, P = self.recover, self.path
        L = float(P["s"][-1])
        r["hop"] = {"t": now, "stage": "turn", "pid": P.get("pid"), "id": P.get("request_id"),
                    "start": (float(v.pos[0]), float(v.pos[1])), "carrot": 0.0, "unobserved": None}
        r["await_reply"] = None
        self.sp[0], self.sp[1] = v.pos[:2]
        self.path_s = 0.0
        self.clr_hist, self.corr_t = [], None
        self.leg_heading = self._path_at(min(0.10, L))[1]
        self._set("creep", now, out, f"creep hop {r['hops'] + 1}/{self.p.creep_max_hops}: {L:.2f} m of "
                                     f"certified observed-free space")

    def _creep(self, now, dt, out, v):
        r, p = self.recover, self.p
        h, P = r["hop"], self.path
        if h is None:
            return self._end_hop(now, out, v, "no hop in progress", False)
        if P is None or not P.get("hop") or P.get("pid") != h["pid"] or not self._path_fresh(now):
            return self._end_hop(now, out, v, "hop withdrawn (certificate lost or path stale)", False)
        if now - h["t"] > p.creep_hop_timeout_s:
            return self._end_hop(now, out, v, f"hop not finished in {p.creep_hop_timeout_s:g} s", False)
        L = float(P["s"][-1])
        step = math.radians(min(p.yaw_rate_dps, 30.0)) * dt
        if h["stage"] == "turn":
            # translation stays at the hop's start until the camera looks along the hop
            self.sp[0], self.sp[1] = h["start"]
            target = self._path_at(min(0.10, L))[1]
            self.leg_heading = target
            self.sp[3] = _wrap(self.sp[3] + max(-step, min(step, _wrap(target - self.sp[3]))))
            slow = v.vel is not None and len(v.vel) >= 2 and math.hypot(*v.vel[:2]) < 0.08
            if v.heading is not None and abs(_wrap(target - v.heading)) < math.radians(5) and slow:
                h["stage"], h["t_move"] = "move", now
                self.clr_hist, self.corr_t = [], None
            return
        s_now, _ = self._path_project(v.pos)
        self.path_s = max(self.path_s, s_now)
        remaining = max(L - self.path_s, 0.0)
        end = P["pts"][-1]
        if math.hypot(v.pos[0] - end[0], v.pos[1] - end[1]) < p.creep_arrive_m:
            return self._end_hop(now, out, v, f"hop done ({L:.2f} m)", True)
        clr = self._clearance(v) if self.depth_fresh(now) else None
        if clr is not None and not self.corridor_fresh(now):
            clr = None
        if clr is None:
            # unknown ahead: never advance; a hop that stays blind ends without progress
            h["unobserved"] = now if h["unobserved"] is None else h["unobserved"]
            self.counts["depth_holds"] += 1
            if now - h["unobserved"] > p.creep_unobserved_s:
                return self._end_hop(now, out, v, f"corridor not observed for {p.creep_unobserved_s:g} s on the hop",
                                     False)
            return
        h["unobserved"] = None
        self.last_clearance = clr
        if clr < self._creep_stop_m() and (p.leg_end_margin_m < 0 or clr < remaining + p.leg_end_margin_m):
            self.counts["obstacle_stops"] += 1
            self.leg_blocks += 1
            why = f"obstacle {clr:.2f} m ahead on the hop"
            if p.blocked_max_per_leg > 0 and self.leg_blocks > p.blocked_max_per_leg:
                self.sp[0], self.sp[1] = v.pos[:2]
                self.recover, self.replan_request = None, None
                self._land(now, out, f"{r['why']}; {why}; blocked {self.leg_blocks} times on this leg "
                                     f"(blocked_max_per_leg {p.blocked_max_per_leg})")
                return
            return self._end_hop(now, out, v, why, False)
        # carrot: at most creep_speed ahead of the aircraft (PX4's xy loop gain is ~1 /s), never past the end
        h["carrot"] = min(L, max(h["carrot"], self.path_s) + p.creep_speed_mps * dt,
                          self.path_s + p.creep_speed_mps)
        pt, tang, _, _ = self._path_at(h["carrot"])
        self.sp[0], self.sp[1] = pt
        self.leg_heading = tang
        self.leg_dir = (math.cos(tang), math.sin(tang))
        self.sp[3] = _wrap(self.sp[3] + max(-step, min(step, _wrap(tang - self.sp[3]))))

    def _end_hop(self, now, out, v, why, progressed):
        """Hover where the hop ended, then look again (dwell, ask, sweep). Bounded by hops, stalls and time."""
        r, p = self.recover, self.p
        h = r.get("hop") or {}
        r["hops"] += 1
        r["consumed_request"] = h.get("id")
        r["hop"] = None
        self.path = None
        self.sp[0], self.sp[1] = v.pos[:2]
        self.v_cmd, self.a_cmd = 0.0, 0.0
        if progressed:
            r["stalls"] = 0
            r["requests"] = 0
            r["deadline"] = min(r["t0"] + p.creep_max_s, now + p.recovery_timeout_s)
            self.counts["creep_hops"] += 1
        else:
            r["stalls"] += 1
            self.counts["creep_stalls"] += 1
            if r["stalls"] >= p.creep_max_stalls:
                self.recover, self.replan_request = None, None
                self._land(now, out, f"{r['why']}; creep made no progress ({why}; "
                                     f"{r['stalls']} hops in a row)")
                return
        out.events.append(f"creep hop {r['hops']}: {why}")
        r.update(scan_done=False, settled=None, yaw=self.sp[3], pos=(float(v.pos[0]), float(v.pos[1])),
                 await_reply={"t": now, "asked": None, "id": None})
        r.pop("t0_hold", None)
        self.clr_hist, self.corr_t = [], None
        self._set("blocked", now, out, f"creep hop {r['hops']}: {why}; looking again")

    def _recover_clear(self, now, v):
        """Has the reason for the recovery gone away? Blocked: the corridor reads clear. Align: the
        gate is found (its alignment is applied on resume)."""
        r = self.recover
        if r["kind"] == "blocked":
            if self.p.recovery_replan and self._path_leg():
                return False    # only an explicit certified response may release a requested recovery
            if not self.depth_fresh(now):
                return False
            clr = self._clearance(v)
            if clr is None or not self.corridor_fresh(now):
                return False
            self.last_clearance = clr
            # hysteresis: the obstacle must have clearly gone in EVERY confirm frame (one clear frame is
            # not evidence that a wall left), not just hover at the stop distance
            worst = clr if self.clr_min is None else self.clr_min
            # judged from where the block happened: backing off adds clearance without the obstacle moving
            # (bag 20261107-201744: blocked at 0.85-0.95 m, backed off 0.2 m, 'recovered', blocked again
            # 0.1-1.5 s later, 18 times). Only an obstacle that actually left clears.
            bp = r.get("pos")
            if bp is not None and math.isfinite(worst):
                worst -= math.hypot(v.pos[0] - bp[0], v.pos[1] - bp[1])
            if self._path_leg() and self.path is not None and self.path["leg"] == self.leg:
                L = float(self.path["s"][-1])
                return not self._blocks(worst - self.p.blocked_clear_margin_m, max(L - self.path_s, 0.0), self.path_s)
            n0, e0 = self.leg_start
            proj = (v.pos[0] - n0) * self.leg_dir[0] + (v.pos[1] - e0) * self.leg_dir[1]
            return not self._blocks(worst - self.p.blocked_clear_margin_m, self.leg_len - max(proj, 0.0), max(proj, 0.0))
        if r["kind"] == "align":
            if self.frame is not None and self._align_judged_t != self.depth_t and self.depth_fresh(now):
                self._align_judged_t = self.depth_t
                est = self._measure_gate(v)
                if est is not None:
                    self.align_hist.append(est)
            return len(self.align_hist) >= max(1, self.p.align_frames)
        return False

    def _resume(self, now, out, v):
        r, self.recover = self.recover, None
        self.replan_request = None
        self.counts["recoveries"] += 1
        if r["kind"] == "align":
            self._apply_alignment(now, out, G.combine(self.align_hist[-self.p.align_frames:]))
            self._set("check", now, out, "recovered: gate found")
            return
        if self._path_leg():
            # path legs: _go_path sets the setpoint from the path beside the aircraft; the straight-leg
            # geometry (leg_start + leg_dir * s) is not where the aircraft is
            self.sp[0], self.sp[1] = v.pos[0], v.pos[1]
        else:
            n0, e0 = self.leg_start
            proj = (v.pos[0] - n0) * self.leg_dir[0] + (v.pos[1] - e0) * self.leg_dir[1]
            self.s = min(max(proj, 0.0), self.leg_len)
            self.sp[0], self.sp[1] = n0 + self.leg_dir[0] * self.s, e0 + self.leg_dir[1] * self.s
        self.t_check = now
        self._set("go", now, out, "recovered: corridor clear again")

    def _recover(self, now, dt, out, v):
        p = self.p
        r = self.recover
        if r is None:
            self._set("check", now, out)
            return
        if self.p.recovery_replan and (now - r["t0"] >= self.p.recovery_timeout_s if not p.recovery_creep else
                                       now >= min(r.get("deadline", r["t0"] + p.recovery_timeout_s),
                                                  r["t0"] + p.creep_max_s)):
            self.recover = None
            self.replan_request = None
            self._land(now, out, f"{r['why']}; bounded recovery timed out"
                                 + (f" after {r.get('hops', 0)} creep hops" if p.recovery_creep else ""))
            return
        if self._recover_clear(now, v):
            self._resume(now, out, v)
            return
        if (r["kind"] == "blocked" and self.phase != "creep" and self._path_leg() and self._path_fresh(now)
                and ((r.get("pid") is not None and self.path.get("pid") != r["pid"])
                     or (p.recovery_replan and r.get("pid") is None))
                and (not p.recovery_replan or (self.replan_request is not None
                     and self.path.get("request_id") == self.replan_request["id"]
                     and self.path.get("recovery_certified")))):
            if p.recovery_creep and self.path.get("hop"):
                self._begin_hop(now, out, v)
                return
            # the navigator re-planned while we were blocked: turn to face the new path, check it, go. Waiting
            # for the OLD corridor to clear never turns the camera away from the obstacle
            self.recover = None
            self.replan_request = None
            self.counts["replan_resumes"] += 1
            self.sp[0], self.sp[1] = v.pos[0], v.pos[1]
            self.clr_hist, self.corr_t = [], None
            self.leg_heading = self._path_at(p.path_lookahead_m)[1]
            self.turn_start_heading = self.sp[3]
            self._set("turn", now, out, f"re-planned while blocked on leg {self.leg + 1}: turning to the new path")
            return
        if self.phase == "creep":
            self._creep(now, dt, out, v)
        elif self.phase == "recovery_scan":
            self._scan_recovery(now, dt, out, v)
        elif self.phase == "blocked" and r.get("await_reply") is not None:
            # after a hop: hover creep_dwell_s for fresh depth, ask once, wait for the answer, then sweep
            a = r["await_reply"]
            if a["asked"] is None:
                if now - a["t"] >= p.creep_dwell_s and self.depth_fresh(now):
                    self._request_recovery_plan(now)
                    a["asked"], a["id"] = now, (self.replan_request or {}).get("id")
            else:
                reply = self.plan_reply or {}
                failed = reply.get("id") == a["id"] and reply.get("state") == "failed"
                if failed or now - a["asked"] >= p.creep_reply_s:
                    r["await_reply"] = None
                    r["t0_hold"] = now - p.blocked_hold_s   # straight on to the sweep (or back-off once swept)
        elif self.phase == "blocked":
            if now - r.get("t0_hold", r["t0"]) > p.blocked_hold_s:
                if (p.recovery_replan and self._path_leg() and p.recovery_scan_deg > 0
                        and not r["scan_done"]):
                    angle = math.radians(min(45.0, p.recovery_scan_deg))
                    centre = self._route_scan_centre(r, v)
                    r.update(scan_i=0, settled=None, scan_centre=centre,
                             scan_yaws=[_wrap(centre-angle), _wrap(centre+angle), centre])
                    self._set("recovery_scan", now, out, "bounded stationary depth scan for an observed-free detour")
                else:
                    self._begin_backoff(now, out, v)
        elif self.phase == "backoff":
            tn, te = r["target"]
            if math.hypot(v.pos[0] - tn, v.pos[1] - te) < p.arrive_m or now - r["t_back"] > 4.0:
                r["t_re"] = now
                if p.recovery_replan and self._path_leg():
                    self._request_recovery_plan(now)
                self.clr_hist, self.corr_t = [], None
                self._set("recheck", now, out, f"rechecking for {p.blocked_recheck_s:g} s")
        elif self.phase == "recheck":
            if now - r["t_re"] > p.blocked_recheck_s:
                self.recover = None
                self.replan_request = None
                self._land(now, out, f"{r['why']}; still blocked after hold, back-off and recheck")

    def status(self, now):
        return {"phase": self.phase, "why": self.why, "result": self.result, "started_t": self.t_started, "leg": self.leg,
                "legs": len(self.targets), "legs_done": self.legs_done, "home": self.home,
                "targets": [list(t) for t in self.targets], "setpoint": list(self.sp) if self.sp else None,
                "clearance_m": (None if self.last_clearance is None or math.isinf(self.last_clearance)
                                else round(self.last_clearance, 2)),
                "stop_m": round(self._stop_m(self.s), 2),
                "v_cmd": round(self.v_cmd, 3),
                "depth_age_s": None if self.depth_t is None else round(now - self.depth_t, 3),
                "corridor_age_s": None if self.corr_t is None else round(now - self.corr_t, 3),
                "corridor_coverage": None if self.last_coverage is None else round(self.last_coverage, 2),
                "depth_valid_frac": None if self.last_valid_frac is None else round(self.last_valid_frac, 3),
                "depth_reject": None if self.depth_reject is None else self.depth_reject[1],
                "health_bad_s": (None if math.isinf(self.health.bad_for(now)) else round(self.health.bad_for(now), 2)),
                "recover": None if self.recover is None else {
                    "kind": self.recover["kind"], "stage": self.phase,
                    "hops": self.recover.get("hops", 0), "stalls": self.recover.get("stalls", 0),
                    **({} if self.recover.get("scan_centre") is None else
                       {"scan_centre_deg": round(math.degrees(self.recover["scan_centre"]), 1)})},
                "recovery_request": self.replan_request if self.recover is not None else None,
                "path_s": None if self.path is None else float(self.path_s),
                "path_pid": None if self.path is None else self.path.get("pid"),
                "plan_reply": self.plan_reply,
                "prior_state": None if self.prior is None else self.prior[1],
                "leg_blocks": self.leg_blocks,
                "tunnel": self.tunnel_active, "align": self.align_result,
                **self.counts}
