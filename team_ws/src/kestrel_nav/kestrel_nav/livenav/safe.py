"""Safe course policy: stop-turn-go along the live A* path. PURE (numpy). No ROS, no clocks.

Goal: finish the course without touching anything. Speed is spent only where clearance is large.

Rules, each from measured evidence (workspace sim runs 3-10 on 2026-11-07, docs/SAFE_POLICY.md):

1. NEVER YAW WHILE TRANSLATING (beyond a small heading hold). Run010's contact at maze divider M1
   happened while the vehicle yawed ~27 deg/s and translated at 0.15 m/s: true velocity swung toward
   the nose while the estimate did not, 12 cm of cross-track error, 0.268 m truth clearance against a
   0.294 m prop reach. Run004 lost 14 cm to tracking at M3 on a moving corner. Here every corner is a
   stop: brake, hold, turn in place at <= turn_rate, settle, then translate straight.
2. FACE WHERE YOU GO. Translation is only commanded along the body's forward axis (plus a bounded
   cross-track correction), so the forward depth camera always watches the swept corridor.
3. CLOSE THE POSITION LOOP ON VIO, NOT ON PX4. The hardware interface takes world velocity; this
   policy computes it from cuVSLAM odometry (via the course<-odom transform). In run010 VIO position
   followed truth while PX4's own velocity estimate lagged the slide.
4. HOLD WHERE YOU ARE. After every stop the station-keeping anchor is the CURRENT position, never
   the waypoint just targeted (run008: a stale hold target pulled the vehicle 10 cm sideways).
5. MEASURE GATES, DO NOT TRUST THE MAP FOR THEM. Before a crossing leg (window, tunnel) the vehicle
   faces the crossing, measures the opening's jambs in depth, side-steps onto its centre, re-measures,
   and keeps re-centring on the measured opening while it closes in. No measurement -> no crossing.
6. STOP DISTANCE IS A BUDGET. A return ahead in the body corridor must be farther than
   reach + front_buffer + v*delay + v^2/(2*a_brake); otherwise the segment is shortened or the
   vehicle holds. delay/a_brake are conservative assumptions, not measured hardware values.
7. LAND ON THE PAD. Centre over the pad, descend slowly (below PX4's land-detector vertical speed),
   keep commanding descent until PX4 disarms. The interface's 0.3 m floor clamps pose setpoints only.

Speeds are deliberately low (0.12-0.25 m/s). They are not tuned: nothing here was fitted to a flight.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .follower import FollowParams, scan_ahead, wrap
from .mission import Command, LiveMission, Observation
from .navgrid import NavParams
from .opening import combine, find_opening_flu, to_leg_frame
from .square_pad import FOOTPRINT_CENTRE_X_M, PadFuser
from .replanner import ReplanParams


# 7-inch propellers (confirmed by the team, 2026-11-07) on the fleet allocator geometry
# (deploy/px4/params/07_airframe_geometry.params, FRD metres). The horizontal yaw-independent reach
# of a rotor disk centred at (x, y) is hypot(x, y) + D/2; the airframe's is the largest of the four.
ROTOR_CENTRES_XY_M = ((0.08, 0.16), (-0.15, -0.14), (0.08, -0.16), (-0.15, 0.14))
PROP_DIAMETER_M = 7 * 0.0254


def prop_reach_m(rotors=ROTOR_CENTRES_XY_M, diameter=PROP_DIAMETER_M):
    """0.294 m for the skybeta fleet with 7-inch props. Level flight, no guards/tilt allowance."""
    return max(math.hypot(x, y) for x, y in rotors) + diameter / 2.0


REACH_M = prop_reach_m()


@dataclass
class SafeParams(FollowParams):
    # node-level motion-corridor guard (live_mission.tick reads these three via mission.follower.p)
    corridor_half_m: float = REACH_M + 0.05   # body corridor the depth image must observe
    stop_m: float = 0.45                      # depth is blind under ~0.4 m (stereo minimum range)
    reactive_ramp_m: float = 0.30
    # speeds: slow everywhere, slowest where it is tight
    v_cap: float = 0.30           # 930am: 2x the only speed flown in the steady profile (no data beyond)
    tube_response_s: float = 0.44  # 930am run015: per-leg p90 of steady cross-track p95 / speed
    v_gate: float = 0.20          # extra cap on wall-opening crossings (the window); tube speed applies too
    reach_m: float = REACH_M
    # stopping budget (assumptions, not measured; the workspace diagnostic assumed 0.8 s / 0.15 m/s^2)
    a_brake: float = 0.5          # commanded deceleration, m/s^2
    response_delay_s: float = 0.6
    front_buffer_m: float = 0.10  # always stop at least reach + this short of a return ahead
    # turning in place
    turn_rate: float = math.radians(35)       # well under MPC_YAWRAUTO_MAX 45 deg/s
    turn_kp: float = 3.0
    heading_hold_rate: float = math.radians(6)  # the only yaw allowed while translating (run010: 27 deg/s)
    flow_turn_max: float = math.radians(12)     # a vertex turning less than this is flown through, not stopped at
    yaw_tol_go: float = math.radians(3)
    settle_s: float = 0.5         # still (< settle_speed) this long before measuring an opening
    pre_turn_dwell_s: float = 0.2 # still this long before turning in place
    post_turn_dwell_s: float = 0.0  # 930am run015: a 2 s post-turn settle did not change set-off drift  # heading on target this long before translating
    settle_speed: float = 0.05
    settle_timeout_s: float = 6.0
    # station keeping and cross-track (bounded creep; see creep_mps)
    hold_kp: float = 0.6
    creep_mps: float = 0.06       # corrective velocity cap; below it the node needs no forward corridor
    ct_kp: float = 0.8
    # segments
    min_seg_m: float = 0.10
    arrive_tol_m: float = 0.05
    pad_enter_m: float = 0.30     # stopped this close to the pad goal -> DESCEND (which centres to pad_over_m)
    brake_timeout_s: float = 3.0  # still "moving" after this: hold position where the vehicle is
    no_progress_max: int = 3      # segments ending in a stop without progress, in a row -> HOLD and recheck
    obstacle_memory_s: float = 3.0
    # bounded localisation corrections (workspace matthew930am refnav: deadband 0.04 m, 0.05 m/s within a
    # leg, total <= 0.8 m, sigma <= 0.10). Gazebo @ 9608f04: unbounded PF pulls (30 % per scan) put the
    # belief 0.27 m / 7.4 deg off ground truth in the first corridor and the planner found no path.
    corr_sigma_max_m: float = 0.10
    corr_deadband_m: float = 0.04
    corr_rate_mps: float = 0.05
    corr_yaw_rate: float = math.radians(1.0)
    corr_total_m: float = 0.8
    corr_total_yaw: float = math.radians(10.0)
    live_grid: bool = False       # 930am plans on the static map; live depth is the stop check on the leg
    align_retry_s: float = 5.0    # a refused gate: hold this long, re-measure once, then cross on the map
    escape_max_m: float = 0.30    # no path from inside the planning margin: step to the nearest free cell this close
    v_escape: float = 0.08  # a return that drops inside stereo minimum range is remembered this long
    # gate alignment
    align_frames: int = 5
    align_tol_m: float = 0.03
    align_max_shift_m: float = 0.25   # a bigger measured offset than this is a localisation fault: hold
    align_max_rounds: int = 4
    align_timeout_s: float = 12.0
    align_spread_max_m: float = 0.06  # frames disagreeing more than this are not a measurement
    recentre_min_range_m: float = 0.55  # depth is unreliable inside ~0.45 m; stop re-centring there
    recentre_gain: float = 0.5
    gate_requires_measurement: bool = True
    # landing
    land_vz: float = 0.15         # PX4 LNDMC_Z_VEL_MAX 0.25: descend below it so the land detector fires
    pad_over_m: float = 0.03      # venue pad ~0.64 m square: the 0.50 x 0.41 m footprint has 7 / 11.5 cm per side
    pad_settle_s: float = 1.0     # centred and still this long before descending
    pad_yaw_tol: float = math.radians(3)
    landed_z_odom_m: float = 0.04


def stop_distance(v, p: SafeParams):
    v = max(0.0, float(v))
    return v * p.response_delay_s + v * v / (2.0 * p.a_brake)


class SafeMission(LiveMission):
    """LiveMission (planning, PF localizer, live grid, replanner, pad detector) with the follower
    replaced by the stop-turn-go segment driver above. States seen by the node/sim are unchanged:
    WAIT CLIMB FLY PAD HOLD DESCEND PAUSED FAULT; the driver's phase is in Command.note/`phase`."""

    def __init__(self, params=None, nav_params: NavParams = None, replan_params: ReplanParams = None,
                 safe_params: SafeParams = None, course_file=None):
        sp = safe_params or SafeParams()
        super().__init__(params, nav_params, replan_params, sp, course_file=course_file)
        self.sp = sp
        self.phase = "IDLE"
        self.t_phase = None
        self.anchor = None            # course xy held while stationary
        self.seg = None               # dict(a, b, dir, heading, v, gate)
        self.settle_since = None
        self.align = None             # dict for the gate alignment in progress
        self.aligned_for = None       # goal index whose crossing has been centred
        self.cross_offset = 0.0       # measured lateral offset of the crossing line (left +)
        self.climb_xy = None
        self.pf_hold_k = None         # goal index whose crossing froze the PF (the opening is the fix)
        self.align_meta = self._align_meta()
        self.events = []
        self.pad_fuser = PadFuser(self.goals[-1].xy.copy())
        st = self.doc["start"]                   # placement accuracy on the mark (course file: 0.05 m / 2 deg)
        self.p.pf_sigma_xy_m = float(st.get("sigma_xy_m", self.p.pf_sigma_xy_m))
        self.p.pf_sigma_yaw_deg = float(st.get("sigma_yaw_deg", self.p.pf_sigma_yaw_deg))
        self.pad_source = "CAD"
        self.land_yaw = None
        self.pad_ok_since = None

    # ------------------------------------------------------------------ gate geometry from the plan
    def load_doc(self, course_file=None):
        """The course document with the vehicle radius raised to the real 7-inch prop reach, as on
        workspace matthew930am: planning needs reach + 0.12 m (0.414 m), the live hard band reach +
        0.10 m. Feasible on the surface-measured clearance field (the window's centre is 0.457 m)."""
        doc = super().load_doc(course_file)
        # The course file's sim_profile spawn (FlyX training world: 0.2 m north of the mark) is not the
        # venue start: on the drone the aircraft sits on the launch mark itself. Plan from the mark.
        doc["start"].pop("spawn_xy", None)
        v = doc["vehicle"]
        v["radius_m"] = max(float(v["radius_m"]), REACH_M)
        return doc

    def _plan_course(self):
        """Plan from the start pose; if it lies inside the planning inflation (the launch mark is ~0.40 m
        from the barrier, under reach + 0.12 = 0.414 m), plan from the nearest free point within 0.3 m.
        The course<-odom anchor stays the real start (the drone sits there); the first leg then moves
        to increasing clearance, which the live chord rule allows."""
        doc = self.doc
        sx, sy = doc["start"]["xy"] if self.p.start_xy is None else self.p.start_xy
        start = np.array([float(sx), float(sy)])
        free = start
        if self.grid.is_blocked(start):
            free = self.grid.nearest_free(start, max_m=0.3)
            if free is None:
                raise ValueError(f"start {start.round(3).tolist()} has no free planning cell within 0.3 m")
        keep = self.p.start_xy
        self.p.start_xy = (float(free[0]), float(free[1]))
        try:
            plan = super()._plan_course()
        finally:
            self.p.start_xy = keep
        plan["start_course"] = (float(start[0]), float(start[1]), plan["start_course"][2])
        plan["start_plan_xy"] = [round(float(v), 3) for v in free]
        return plan

    def _align_meta(self):
        """goal name -> {centre, dir, meta} for every crossing leg that has a measurable opening."""
        wps = self.plan["waypoints"]
        out = {}
        by_name = {w["name"]: w for w in wps}
        for i, g in enumerate(self.goals):
            if not g.is_gate_leg or i == 0:
                continue
            pre = self.goals[i - 1]
            if g.crosses is not None:
                centre = next((w for w in wps if w.get("gate") == g.crosses and w.get("part") == "centre"), None)
                if centre is None or "align" not in centre:
                    continue
                d = np.asarray(centre["normal"], float)
                out[g.name] = dict(centre=np.asarray(centre["xy"], float), dir=d / np.linalg.norm(d),
                                   meta=dict(centre["align"]), pre=pre.xy.copy())
            elif g.passage:
                w = by_name.get(pre.name)
                if w is None or "align" not in w:
                    continue
                d = g.xy - pre.xy
                d = d / np.linalg.norm(d)
                mouth = pre.xy + float(w["align"]["dist_m"]) * d
                out[g.name] = dict(centre=mouth, dir=d, meta=dict(w["align"]), pre=pre.xy.copy())
        return out

    # ------------------------------------------------------------------ localisation from an opening
    def _shift_course(self, dv):
        """Move the vehicle's course-frame position by dv (course metres) by correcting course<-odom.
        Every stored course-frame point that describes the vehicle moves with it."""
        tx, ty, dyaw = self.T
        c, s = math.cos(dyaw), math.sin(dyaw)
        self.T = (tx - (c * dv[0] - s * dv[1]), ty - (s * dv[0] + c * dv[1]), dyaw)
        if self.anchor is not None:
            self.anchor = self.anchor + dv
        if self._last_odom is not None:
            self._last_odom = (self._last_odom[0], self._last_odom[1] + dv)

    def _localize(self, obs):
        """PF as in LiveMission, but its pull on course<-odom is BOUNDED: only when sigma <= 0.10 m,
        outside a 4 cm deadband, at most 0.05 m/s and 1 deg/s, at most 0.8 m / 10 deg in total, never
        during the climb, and not while a measured opening is the position reference."""
        sp = self.sp
        T0 = self.T
        super()._localize(obs)                   # runs the PF; proposes a (low-passed) new T
        Tp, self.T = self.T, T0
        if self.pf_hold_k is not None or self.state in ("CLIMB", "WAIT") or not (self.pf_sigma <= sp.corr_sigma_max_m):
            return
        dt = 0.0 if getattr(self, "_corr_t", None) is None else max(0.0, obs.now - self._corr_t)
        self._corr_t = obs.now
        if dt <= 0.0:
            return
        dx, dy, dyaw = Tp[0] - T0[0], Tp[1] - T0[1], wrap(Tp[2] - T0[2])
        d = math.hypot(dx, dy)
        if d < sp.corr_deadband_m and abs(dyaw) < math.radians(1.0):
            return
        step = min(d, sp.corr_rate_mps * dt)
        dyaw = float(np.clip(dyaw, -sp.corr_yaw_rate * dt, sp.corr_yaw_rate * dt))
        base = getattr(self, "_T_start", None) or T0
        self._T_start = base
        nx = T0[0] + (dx / d * step if d > 1e-9 else 0.0)
        ny = T0[1] + (dy / d * step if d > 1e-9 else 0.0)
        nyaw = wrap(T0[2] + dyaw)
        if math.hypot(nx - base[0], ny - base[1]) > sp.corr_total_m or abs(wrap(nyaw - base[2])) > sp.corr_total_yaw:
            return
        self.T = (nx, ny, nyaw)

    def _release_pf(self, now, obs):
        """Crossing done: re-seed the PF at the gate-corrected pose, so it does not pull back to the
        pre-gate estimate the opening measurement just disproved."""
        self.pf_hold_k = None
        if self.pf is not None:
            pose = self.odom_to_course(obs.odom_xyz[:2], obs.odom_yaw)
            self.pf.init(pose, 0.05, math.radians(2.0), np.array([obs.odom_xyz[0], obs.odom_xyz[1], obs.odom_yaw]))
            self._pf_last_t = now
        self.events.append((now, "PF", "PF", "re-seeded at the gate-corrected pose"))

    def _update_pad(self, obs, pose_c):
        """Fuse square-pad fixes (node: square_pad.fix_from_detection, level body frame) into the pad goal."""
        f = obs.pad_fix
        if not isinstance(f, dict) or obs.now - obs.pad_fix_t > self.p.pad_fix_max_age_s or obs.pad_fix_t <= getattr(self, "_pad_t", -math.inf):
            return
        self._pad_t = obs.pad_fix_t
        # the fix is relative to the body WHEN THE FRAME WAS TAKEN: use that odometry pose, not the current one
        x, y, yaw = self.odom_to_course(f["odom"][:2], f["odom"][2]) if "odom" in f else pose_c
        c, s = math.cos(yaw), math.sin(yaw)
        Rb = np.array([[c, -s], [s, c]])
        xy = np.array([x, y]) + Rb @ np.asarray(f["xy"], float)
        if np.linalg.norm(xy - self.pad_fuser.prior) > self.p.pad_shift_max_m:
            return
        self.pad_fuser.add(xy, Rb @ np.asarray(f["cov"], float) @ Rb.T, None if f.get("yaw") is None else f["yaw"] + yaw)
        self.pad_n = self.pad_fuser.n
        self.pad_est = self.pad_fuser.xy
        if self.pad_fuser.n >= self.p.pad_min_fixes:
            self.goals[-1].xy = self.pad_est.copy()
            self.pad_source = f"detected ({self.pad_fuser.n} fixes, sigma {self.pad_fuser.sigma:.3f} m)"

    # ------------------------------------------------------------------ small helpers
    def _phase(self, name, now, note=""):
        if name != self.phase:
            self.events.append((now, self.phase, name, note))
        self.phase, self.t_phase = name, now

    def _hold_vel(self, xy):
        """Bounded station keeping on the anchor (course frame)."""
        if self.anchor is None:
            return np.zeros(2)
        v = self.sp.hold_kp * (self.anchor - np.asarray(xy, float))
        n = float(np.linalg.norm(v))
        return v if n <= self.sp.creep_mps else v * (self.sp.creep_mps / n)

    def _cmd(self, cmd, v_course, yaw_rate=0.0, note=""):
        cmd.vx, cmd.vy = self.course_vec_to_odom(float(v_course[0]), float(v_course[1]))
        cmd.yaw_rate = float(yaw_rate)
        cmd.speed = float(np.linalg.norm(v_course))
        cmd.note = f"{self.phase}{': ' + note if note else ''}"
        return cmd

    def _seg_clearance(self, a, b):
        n = max(2, int(math.ceil(math.dist(a, b) / 0.05)) + 1)
        return min(self.grid.clearance_at(a + t * (b - a)) for t in np.linspace(0.0, 1.0, n))

    def _chord_blocked(self, a, b):
        """Live-grid validity of a straight move a->b at the HARD inflation (reach + 0.10 class).
        A vehicle already inside the band (a live return landed close) may move only if clearance never
        decreases until it is out, and the rest of the chord is free. Never deeper into the band."""
        rp = self.replanner
        if not rp.chord_blocked(a, b, hard=True):
            return False
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(2, int(math.ceil(math.dist(a, b) / 0.025)) + 1)
        pts = [a + t * (b - a) for t in np.linspace(0.0, 1.0, n)]
        clear = [self.grid.clearance_at(q) for q in pts]
        k = next((i for i, c in enumerate(clear) if c >= self.grid.hard_inflate_m), None)
        if k is None or k == 0:
            return True
        if any(clear[i + 1] < clear[i] - 0.01 for i in range(k)):
            return True
        return k < n - 1 and rp.chord_blocked(pts[k], b, hard=True)

    def _update_velocity_odom(self, now, odom_xy):
        """Velocity from VIO odometry differences rotated into the course frame. Unlike LiveMission's
        course-frame difference, a PF/gate correction of course<-odom is not mistaken for motion."""
        p = np.asarray(odom_xy, float)
        last = getattr(self, "_last_odom_xy", None)
        if last is not None:
            dt = now - last[0]
            if 0.02 <= dt <= 0.5:
                c, s = math.cos(-self.T[2]), math.sin(-self.T[2])
                d = (p - last[1]) / dt
                v = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
                self.vel_course = 0.6 * self.vel_course + 0.4 * v
        self._last_odom_xy = (now, p)

    def _escape_segment(self, xy):
        """Inside the planning margin with no path (a climb drifted toward a wall, or live returns
        thickened it): a short slow leg to the nearest margin-free cell, only if clearance never
        decreases along it. Then the next segment replans from free space."""
        sp = self.sp
        g = self.grid
        if not g.is_blocked(xy):
            return None
        target = g.nearest_free(xy, max_m=sp.escape_max_m)
        if target is None or math.dist(target, xy) < 0.02 or self._chord_blocked(xy, target):
            return None
        pts = [xy + t * (target - xy) for t in np.linspace(0.0, 1.0, 8)]
        clear = [g.clearance_at(q) for q in pts]
        if any(b < a - 0.01 for a, b in zip(clear, clear[1:])):
            return None
        d = (target - xy) / math.dist(target, xy)
        return dict(a=xy.copy(), b=np.asarray(target, float), dir=d, heading=math.atan2(d[1], d[0]),
                    v=sp.v_escape, gate=None, line_pt=xy.copy(), flow=False, tube=0.0, escape=True)

    def _no_progress(self, now, xy, cmd, why):
        """Stop. The same refusal a few times in a row (no progress between them) -> HOLD, which
        replans every hold_recheck_s and gives up after blocked_abort_s."""
        self.no_progress = getattr(self, "no_progress", 0) + 1
        self._stop(now, xy, "BRAKE", why)
        if self.no_progress >= self.sp.no_progress_max:
            self.no_progress = 0
            self._set("HOLD", now, f"no progress: {why}")
            self.hold_since = now
        return self._cmd(cmd, np.zeros(2), 0.0, why)

    def _tube_speed(self, a, b, cap):
        """930am leg rule: tube = (leg clearance - reach) / 2 (the other half is left for estimation
        error), speed = tube / 0.44 s (the speed whose p90 tracking error fills the tube), <= cap."""
        sp = self.sp
        tube = max(0.0, (self._seg_clearance(a, b) - sp.reach_m) / 2.0)
        return tube, max(0.05, min(cap, sp.v_cap, tube / sp.tube_response_s))

    def _stop(self, now, xy, phase="BRAKE", note=""):
        self.anchor = np.asarray(xy, float).copy()
        self.seg = None
        self.settle_since = None
        self._phase(phase, now, note)

    # ------------------------------------------------------------------ segment choice
    def _next_segment(self, now, pose_c):
        """Pick the next straight segment from the current position along a FRESH path."""
        sp, rp = self.sp, self.replanner
        xy = np.asarray(pose_c[:2], float)
        # goals already reached from here
        while not rp.done and rp.arrived(xy) and rp.goal is not self.goals[-1]:
            rp.advance()
        if rp.done:
            return None
        goal = rp.goal
        gate = self.align_meta.get(goal.name)
        if gate is not None and self.aligned_for != rp.k:
            return "ALIGN"
        if gate is not None:
            # crossing: straight along the measured crossing line to the goal
            d = gate["dir"]
            left = np.array([-d[1], d[0]])
            # the goal projected onto the measured crossing line, at its along-track position
            line_pt = gate["centre"] + self.cross_offset * left
            along = float((goal.xy - line_pt) @ d)
            b = line_pt + along * d
            if float((b - xy) @ d) <= sp.arrive_tol_m and goal is not self.goals[-1]:
                rp.advance()                     # past the crossing goal along the track: leg done
                return self._next_segment(now, pose_c)
            # the line is centred on the measured opening: its clearance is the map's centre line
            tube, v = self._tube_speed(line_pt, b, sp.v_gate if goal.crosses is not None else sp.v_cap)
            return dict(a=xy.copy(), b=b, dir=d, heading=math.atan2(d[1], d[0]), v=v, gate=goal.name,
                        line_pt=line_pt, tube=tube)
        path = rp.plan(xy, now)
        if path is None or rp.blocked:
            esc = self._escape_segment(xy)
            if esc is not None:
                self.events.append((now, "ESCAPE", "ESCAPE", f"no path ({rp.last_reason[:60]}); stepping "
                                    f"{math.dist(esc['b'], xy):.2f} m to free space"))
                return esc
            return "BLOCKED"
        # first vertex far enough away; the path starts at the vehicle
        b = None
        for q in path[1:]:
            if math.dist(q, xy) >= sp.min_seg_m:
                b = np.asarray(q, float)
                break
        if b is None:
            if math.dist(goal.xy, xy) < sp.min_seg_m:
                if goal is self.goals[-1]:
                    return None                  # at the pad: the caller descends
                rp.advance()
                return self._next_segment(now, pose_c)
            b = goal.xy.copy()
        d = (b - xy) / max(1e-9, math.dist(b, xy))
        tube, v = self._tube_speed(xy, b, sp.v_gate if goal.crosses is not None else sp.v_cap)
        # flow through b only when the path continues past it almost straight (no gate, no stop)
        flow = False
        idx = next((i for i, q in enumerate(path) if np.allclose(q, b)), None)
        if idx is not None and idx + 1 < len(path) and not goal.is_gate_leg:
            e = np.asarray(path[idx + 1], float) - b
            if np.linalg.norm(e) >= sp.min_seg_m:
                turn = abs(wrap(math.atan2(e[1], e[0]) - math.atan2(d[1], d[0])))
                flow = turn <= sp.flow_turn_max
        return dict(a=xy.copy(), b=b, dir=d, heading=math.atan2(d[1], d[0]), v=v, gate=None, line_pt=xy.copy(),
                    flow=flow, tube=tube)

    # ------------------------------------------------------------------ gate alignment
    def _align_step(self, now, obs, pose_c, cmd):
        """Face the crossing, measure the opening, side-step onto its centre. Repeat until centred."""
        sp, rp = self.sp, self.replanner
        gate = self.align_meta[rp.goal.name]
        xy = np.asarray(pose_c[:2], float)
        a = self.align
        if a is None:
            a = self.align = dict(t0=now, rounds=0, frames=[], last_pts_t=-math.inf, target=None)
            self.anchor = xy.copy()
            self.pf_hold_k = rp.k
        heading = math.atan2(gate["dir"][1], gate["dir"][0])
        err = wrap(heading - pose_c[2])
        if now - a["t0"] > sp.align_timeout_s:
            return self._align_fail(now, xy, cmd, "opening not measured in time")
        # 1. side-step in progress
        if a["target"] is not None:
            if self._chord_blocked(xy, a["target"]):
                return self._align_fail(now, xy, cmd, "side-step blocked on the live grid")
            self.anchor = a["target"]
            v = self._hold_vel(xy)
            if math.dist(xy, a["target"]) <= sp.arrive_tol_m and float(np.linalg.norm(self.vel_course)) < sp.settle_speed:
                a["target"], a["frames"], a["settle"] = None, [], None
                a["t0"] = now                    # each measuring round gets the full timeout
            return self._cmd(cmd, v, 0.0, "side-step onto the opening")
        # 2. face the crossing
        if abs(err) > sp.yaw_tol_go:
            yr = float(np.clip(sp.turn_kp * err, -sp.turn_rate, sp.turn_rate))
            a["frames"], a["settle"] = [], None
            a["t0"] = now                        # the timeout counts measuring, not turning
            return self._cmd(cmd, self._hold_vel(xy), yr, "facing the crossing")
        # 3. settle, then collect frames
        speed = float(np.linalg.norm(self.vel_course))
        if speed > sp.settle_speed:
            a["settle"] = None
            return self._cmd(cmd, self._hold_vel(xy), 0.0, "settling")
        a["settle"] = a.get("settle") or now
        if now - a["settle"] < sp.settle_s:
            return self._cmd(cmd, self._hold_vel(xy), 0.0, "settling")
        if obs.pts_level is not None and obs.pts_t > a["last_pts_t"]:
            a["last_pts_t"] = obs.pts_t
            meta = dict(gate["meta"])
            meta["dist_m"] = float((gate["centre"] - xy) @ gate["dir"])
            est = find_opening_flu(obs.pts_level, meta)
            a["frames"].append(to_leg_frame(est, wrap(pose_c[2] - heading)) if est else None)
        if len(a["frames"]) < sp.align_frames:
            return self._cmd(cmd, self._hold_vel(xy), 0.0, f"measuring opening {len(a['frames'])}/{sp.align_frames}")
        good = [f for f in a["frames"] if f]
        m = combine(good) if len(good) >= (sp.align_frames + 1) // 2 else None
        a["frames"] = []
        if m is None or m["lateral_spread_m"] > sp.align_spread_max_m:
            a["misses"] = a.get("misses", 0) + 1
            if a["misses"] >= 3:
                return self._align_fail(now, xy, cmd, f"opening not measured consistently ({m})")
            return self._cmd(cmd, self._hold_vel(xy), 0.0, "re-measuring opening")
        lat = float(m["lateral_m"])
        left = np.array([-gate["dir"][1], gate["dir"][0]])
        # The opening is a landmark: the vehicle is -lat across from its centre. Correct course<-odom
        # so the believed position agrees (lateral only), then steer onto the map's crossing line.
        corr = -lat - float((xy - gate["centre"]) @ left)
        self.events.append((now, "ALIGN", "measured", dict(lateral_m=round(lat, 3), width_m=round(m["width_m"], 3),
                                                            correction_m=round(corr, 3), round=a["rounds"])))
        if abs(corr) > sp.align_max_shift_m:
            return self._align_fail(now, xy, cmd, f"opening {corr:+.2f} m off the believed position: localisation fault")
        self._shift_course(corr * left)
        xy = xy + corr * left
        self.cross_offset = 0.0
        # one jamb in view (the other out of the field of view): good enough to side-step toward the
        # opening (it is placed from the seen jamb and the map width), never to declare it centred
        two_edges = m["two_edge_frames"] >= (sp.align_frames + 1) // 2
        if abs(lat) <= sp.align_tol_m and two_edges:
            self.aligned_for = rp.k
            self.align = None
            self._stop(now, xy, "SETTLE", "centred on the opening")
            return self._cmd(cmd, self._hold_vel(xy), 0.0, "centred")
        a["rounds"] += 1
        if a["rounds"] > sp.align_max_rounds:
            return self._align_fail(now, xy, cmd, "did not converge on the opening centre")
        a["target"] = xy + lat * left
        return self._cmd(cmd, self._hold_vel(xy), 0.0, f"opening {lat:+.3f} m left: side-stepping")

    def _align_fail(self, now, xy, cmd, why):
        if not self.sp.gate_requires_measurement:
            self.aligned_for, self.align, self.cross_offset = self.replanner.k, None, 0.0
            self.events.append((now, "ALIGN", "map-only crossing", why))
            self._stop(now, xy, "SETTLE", why)
            return self._cmd(cmd, self._hold_vel(xy), 0.0, why)
        self.align = None
        self.align_failures = getattr(self, "align_failures", 0) + 1
        if self.align_failures >= 2:             # measured twice without success: cross on the map (930am times out too)
            self.aligned_for, self.cross_offset = self.replanner.k, 0.0
            self.events.append((now, "ALIGN", "map-only crossing", f"opening not measured twice: {why}"))
            self._stop(now, xy, "SETTLE", "map-only crossing")
            return self._cmd(cmd, self._hold_vel(xy), 0.0, "map-only crossing")
        self._stop(now, xy, "REFUSED", why)
        self._set("HOLD", now, f"gate not measured, retrying in {self.sp.align_retry_s:.0f} s: {why}")
        self.hold_since = now
        self.gate_retry_at = now + self.sp.align_retry_s
        return self._cmd(cmd, self._hold_vel(xy), 0.0, why)

    # ------------------------------------------------------------------ the driver
    def _drive(self, now, obs, pose_c, cmd):
        sp, rp = self.sp, self.replanner
        xy = np.asarray(pose_c[:2], float)
        speed = float(np.linalg.norm(self.vel_course))
        if self.anchor is None:
            self.anchor = xy.copy()
        if self.phase in ("IDLE", "BRAKE"):
            self.seg = None
            if speed > sp.settle_speed and now - self.t_phase < sp.brake_timeout_s:
                return self._cmd(cmd, np.zeros(2), 0.0, "braking")
            self.anchor = xy.copy()          # rule 4: hold where the vehicle actually stopped
            self._phase("SETTLE", now)
        if self.phase == "SETTLE":
            if speed > sp.settle_speed:
                self.settle_since = None
            else:
                self.settle_since = self.settle_since or now
            if self.settle_since is None or now - self.settle_since < sp.pre_turn_dwell_s:
                if now - self.t_phase > sp.settle_timeout_s:
                    self.anchor = xy.copy()
                    self.t_phase = now
                return self._cmd(cmd, self._hold_vel(xy), 0.0, "settling")
            nxt = self._next_segment(now, pose_c)
            if nxt is None:
                return self._cmd(cmd, self._hold_vel(xy), 0.0, "route done")
            if nxt == "ALIGN":
                self._phase("ALIGN", now)
                return self._align_step(now, obs, pose_c, cmd)
            if nxt == "BLOCKED":
                self._set("HOLD", now, rp.last_reason or "no path")
                self.hold_since = now
                return self._cmd(cmd, self._hold_vel(xy), 0.0, rp.last_reason)
            self.seg = nxt
            self._phase("TURN", now, f"to {rp.goal.name} {math.degrees(wrap(nxt['heading'] - pose_c[2])):+.0f} deg")
        if self.phase == "ALIGN":
            return self._align_step(now, obs, pose_c, cmd)
        seg = self.seg
        if self.phase == "TURN":
            err = wrap(seg["heading"] - pose_c[2])
            if abs(err) > sp.yaw_tol_go:
                self.settle_since = None
                yr = float(np.clip(sp.turn_kp * err, -sp.turn_rate, sp.turn_rate))
                return self._cmd(cmd, self._hold_vel(xy), yr, f"turning {math.degrees(err):+.0f} deg")
            if speed > sp.settle_speed:
                self.settle_since = None
            else:
                self.settle_since = self.settle_since or now
            if self.settle_since is None or now - self.settle_since < sp.post_turn_dwell_s:
                return self._cmd(cmd, self._hold_vel(xy), sp.turn_kp * err, "settling after the turn")
            # the turn may have moved the vehicle a little: start the segment from here
            seg["a"] = xy.copy()
            if seg["gate"] is None:
                seg["line_pt"] = xy.copy()
                # the replanner just certified this chord (A* + hard fallback); only NEW live evidence
                # blocking it later stops the segment (re-refusing a certified chord livelocked offline)
                seg["chord_clear_at_go"] = not self._chord_blocked(xy, seg["b"])
            seg["ahead_mem"] = None
            self._phase("GO", now)
        if self.phase == "GO":
            d = seg["dir"]
            left = np.array([-d[1], d[0]])
            remaining = float((seg["b"] - xy) @ d)
            # live checks: the chord ahead on the live grid, the depth scan in the body corridor
            if seg["gate"] is None and seg.get("chord_clear_at_go") and self._chord_blocked(xy, seg["b"]):
                return self._no_progress(now, xy, cmd, "segment blocked on the live grid")
            if now - self.t_phase > 1.0:
                self.no_progress = 0
            ahead = scan_ahead(self.scan_nav, wrap(seg["heading"] - pose_c[2]), sp.corridor_half_m)
            # a return inside stereo minimum range (~0.4 m) vanishes from the scan: remember where it was
            s_here = float(xy @ d)
            if math.isfinite(ahead):
                seg["ahead_mem"] = (now, s_here + ahead)
            elif seg.get("ahead_mem") and now - seg["ahead_mem"][0] <= sp.obstacle_memory_s:
                ahead = seg["ahead_mem"][1] - s_here
            # always ramp down into the segment end: a flow-through is only confirmed there (fresh replan),
            # and braking from full speed after a refused flow overshot 7 cm toward a wall offline
            v = min(seg["v"], math.sqrt(max(0.0, 2.0 * sp.a_brake * max(0.0, remaining))))
            # rule 6: what is seen ahead caps the speed so the vehicle can still stop short of it
            room = ahead - sp.reach_m - sp.front_buffer_m
            while v > 0.03 and stop_distance(v, sp) > room:
                v *= 0.8
            if (room < sp.min_seg_m or v <= 0.03) and remaining > max(room, sp.arrive_tol_m):
                return self._no_progress(now, xy, cmd, f"return {ahead:.2f} m ahead")
            if remaining <= sp.arrive_tol_m and seg.get("flow"):
                nxt = self._next_segment(now, pose_c)
                if isinstance(nxt, dict) and nxt["gate"] is None \
                        and abs(wrap(nxt["heading"] - pose_c[2])) <= sp.flow_turn_max + math.radians(3):
                    self.seg = seg = nxt
                    self.events.append((now, "GO", "GO", f"flow to {rp.goal.name}"))
                    return self._drive(now, obs, pose_c, cmd)
            if remaining <= sp.arrive_tol_m:
                self._stop(now, xy, "BRAKE", "segment end")
                return self._cmd(cmd, np.zeros(2), 0.0, "arrived")
            # cross-track: back onto the segment line (or the measured crossing line), bounded
            cross = float((xy - seg["line_pt"]) @ left)
            if seg["gate"] is not None:
                self._recentre(obs, pose_c, seg)
                cross = float((xy - seg["line_pt"]) @ left)
            # 930am tube gate: off the leg by more than the tube -> no forward motion until back inside half
            tube = seg.get("tube") or 0.0
            if tube > 0 and abs(cross) > tube:
                seg["tube_hold"] = True
            elif seg.get("tube_hold") and abs(cross) <= tube / 2.0:
                seg["tube_hold"] = False
            if seg.get("tube_hold"):
                v = 0.0
            v_ct = float(np.clip(-sp.ct_kp * cross, -sp.creep_mps, sp.creep_mps))
            herr = wrap(seg["heading"] - pose_c[2])
            yr = float(np.clip(sp.turn_kp * herr, -sp.heading_hold_rate, sp.heading_hold_rate))
            if abs(herr) > math.radians(15):
                self._stop(now, xy, "BRAKE", "heading lost while translating")
                return self._cmd(cmd, np.zeros(2), 0.0, "heading lost")
            return self._cmd(cmd, v * d + v_ct * left, yr, f"{remaining:.2f} m to go at {v:.2f} m/s")
        return self._cmd(cmd, self._hold_vel(xy), 0.0, "hold")

    def _recentre(self, obs, pose_c, seg):
        """While still far enough to see both jambs, keep the crossing line on the measured opening."""
        sp = self.sp
        gate = self.align_meta.get(seg["gate"])
        if gate is None or obs.pts_level is None or obs.pts_t <= getattr(self, "_recentre_t", -math.inf):
            return
        self._recentre_t = obs.pts_t
        xy = np.asarray(pose_c[:2], float)
        dist = float((gate["centre"] - xy) @ gate["dir"])
        if dist < sp.recentre_min_range_m:
            return
        meta = dict(gate["meta"])
        meta["dist_m"] = dist
        est = find_opening_flu(obs.pts_level, meta)
        if not est or est.get("one_edge"):
            return
        est = to_leg_frame(est, wrap(pose_c[2] - seg["heading"]))
        left = np.array([-gate["dir"][1], gate["dir"][0]])
        measured = float((xy - gate["centre"]) @ left) + float(est["lateral_m"])
        if abs(measured - self.cross_offset) > 0.10:
            return                    # one frame disagreeing by > 10 cm is an outlier, not a correction
        self.cross_offset += sp.recentre_gain * (measured - self.cross_offset)
        line_pt = gate["centre"] + self.cross_offset * left
        seg["line_pt"] = line_pt
        seg["b"] = line_pt + float((seg["b"] - line_pt) @ gate["dir"]) * gate["dir"]

    # ------------------------------------------------------------------ the tick
    def step(self, obs: Observation) -> Command:
        p, sp = self.p, self.sp
        now = obs.now
        if self.fault is not None:
            return Command(state='FAULT', note=self.fault)
        if not all(math.isfinite(v) for v in (*obs.odom_xyz, obs.odom_yaw, now)):
            return self.abort(now, 'non-finite mission observation')
        cmd = Command(state=self.state)
        z = obs.odom_xyz[2]
        if self.state not in ("WAIT", "PAUSED") and not obs.offboard_active:
            self._set("PAUSED", now, "Offboard lost")
            cmd.state = self.state
            return cmd
        if self.state == "WAIT":
            cmd = super().step(obs)
            if self.state == "CLIMB":
                self.climb_xy = None
            return cmd
        if self.state == "PAUSED":
            return cmd
        # sensors -> scans, localizer, live grid (exactly as LiveMission)
        scan_fresh = (obs.pts_level is not None and math.isfinite(obs.pts_t)
                      and -0.05 <= now - obs.pts_t <= p.depth_max_age_s)
        if not scan_fresh:
            self.scan_nav = self.scan_pf = None
        new_scan = scan_fresh and obs.pts_t > self._pts_t
        if new_scan:
            from ..coursenav.scan import scan_from_points
            self._pts_t = obs.pts_t
            band = self.doc.get("plan_band_m", 0.25)
            self.scan_nav = scan_from_points(obs.pts_level, obs.height_m, zlo=p.cruise_z_m - band,
                                             zhi=p.cruise_z_m + band, rmax=p.scan_rmax_m)
            self.scan_pf = scan_from_points(obs.pts_level, obs.height_m, zlo=p.scan_band_lo,
                                            zhi=p.scan_band_hi, rmax=p.scan_rmax_m)
        if self.state in ('FLY', 'PAD', 'HOLD') and (self.scan_nav is None or not len(self.scan_nav)):
            return self.abort(now, 'missing, stale or empty navigation depth scan')
        if new_scan and self.pf is not None and self.pf.ready and self.state in ("CLIMB", "FLY", "PAD", "HOLD"):
            self._localize(obs)
        pose_c = self.odom_to_course(obs.odom_xyz[:2], obs.odom_yaw)
        cmd.pose_course = pose_c
        self._update_velocity_odom(now, obs.odom_xyz[:2])
        xy = np.asarray(pose_c[:2], float)
        if sp.live_grid and new_scan and self.scan_nav is not None and self.state in ("FLY", "PAD", "HOLD", "CLIMB"):
            self.grid.integrate(pose_c, self.scan_nav, now)
        if now - self.t_offboard > p.time_limit_s and self.state in ("FLY", "PAD"):
            self._stop(now, xy, "BRAKE", "time limit")
            self._set("HOLD", now, "time limit")
            self.hold_since = -math.inf
        if self.state == "CLIMB":
            if self.climb_xy is None:
                self.climb_xy = xy.copy()
            self.anchor = self.climb_xy
            cmd.vz = self._vz(z, self.z_odom_cruise, p.climb_rate_mps)
            self._cmd(cmd, self._hold_vel(xy), 0.0, "climbing straight up")
            if abs(z - self.z_odom_cruise) <= p.z_tol_m:
                self._set("FLY", now)
                self._phase("BRAKE", now)
            cmd.state = self.state
            return cmd
        rp = self.replanner
        if self.pf_hold_k is not None and rp.k > self.pf_hold_k:
            self._release_pf(now, obs)
            pose_c = self.odom_to_course(obs.odom_xyz[:2], obs.odom_yaw)
            xy = np.asarray(pose_c[:2], float)
        if self.state in ("FLY", "PAD", "HOLD"):
            cmd.vz = self._vz(z, self.z_odom_cruise, p.climb_rate_mps)
        if self.state in ("FLY", "PAD"):
            last = self.goals[-1]
            if rp.goal is last and self.state == "FLY":
                self._set("PAD", now)
            if rp.k >= len(self.goals) - 2:      # window crossing onward: the pad can be in view
                self._update_pad(obs, pose_c)
            at_pad = rp.done or (rp.goal is last and math.dist(xy, last.xy) <= sp.pad_enter_m
                                 and self.phase in ("SETTLE", "BRAKE", "IDLE"))
            if at_pad:
                self._set("DESCEND", now, "over the pad")
                self.anchor = last.xy.copy()
            else:
                self._drive(now, obs, pose_c, cmd)
                cmd.goal = rp.goal.name if rp.goal is not None else ""
                if self.seg is not None:
                    cmd.path_course = np.vstack([xy, self.seg["b"]])
        if self.state == "HOLD":
            self.anchor = self.anchor if self.anchor is not None else xy.copy()
            self._cmd(cmd, self._hold_vel(xy), 0.0, f"hold: {self.log[-1][3] if self.log else ''}")
            if getattr(self, "gate_retry_at", None) is not None and now >= self.gate_retry_at:
                self.gate_retry_at = None
                self._set("FLY", now, "re-measuring the opening")
                self._phase("SETTLE", now)
            recheck = (self.state == "HOLD" and self.hold_since is not None and now - self.t_state >= p.hold_recheck_s
                       and now - self.hold_since < p.blocked_abort_s and now - self.t_offboard <= p.time_limit_s
                       and getattr(self, "gate_retry_at", None) is None)
            if recheck:
                path = rp.plan(xy, now)
                if (path is not None and not rp.blocked) or self._escape_segment(xy) is not None:
                    self._set("FLY", now, "path clear again" if path is not None else "escaping to free space")
                    self._phase("SETTLE", now)
                else:
                    self.t_state = now
            elif (self.state == "HOLD" and self.hold_since is not None and now - self.hold_since >= p.blocked_abort_s
                  and getattr(self, "gate_retry_at", None) is None):
                # blocked for good: land straight down where it is rather than burn the clock in the air
                self.land_here = xy.copy()
                self.pad_source = "ABORT: landing in place (no path)"
                self.events.append((now, "ABORT", "DESCEND", self.pad_source))
                self._set("DESCEND", now, self.pad_source)
        if self.state == "DESCEND":
            if self.land_yaw is None and getattr(self, "land_here", None) is not None:
                self.land_yaw = pose_c[2]
            elif self.land_yaw is None:
                if self.pad_fuser.n < self.p.pad_min_fixes:
                    self.pad_source = "CAD FALLBACK (pad not detected)"
                    self.events.append((now, "PAD", "FALLBACK", "landing on the course-file pad coordinates"))
                # face a pad side (the nearest of the four) so the 0.41 x 0.50 m footprint is square to it
                py = self.pad_fuser.yaw
                self.land_yaw = pose_c[2] if py is None else py + round(wrap(pose_c[2] - py) / (math.pi / 2)) * (math.pi / 2)
                self.events.append((now, "PAD", "TARGET", dict(source=self.pad_source,
                                    xy=[round(float(v), 3) for v in self.goals[-1].xy],
                                    yaw_deg=round(math.degrees(self.land_yaw), 1))))
            # footprint centre (0.035 m behind the body origin) over the pad centre
            cl, sl = math.cos(self.land_yaw), math.sin(self.land_yaw)
            goal_xy = (np.asarray(self.land_here, float) if getattr(self, "land_here", None) is not None
                       else np.asarray(self.goals[-1].xy, float) - FOOTPRINT_CENTRE_X_M * np.array([cl, sl]))
            self.anchor = goal_xy
            off = math.dist(xy, goal_xy)
            yerr = wrap(self.land_yaw - pose_c[2])
            speed = float(np.linalg.norm(self.vel_course))
            if getattr(self, "land_here", None) is not None:
                ready, self.pad_ok_since = True, (self.pad_ok_since or now - sp.pad_settle_s)   # abort: straight down
            else:
                # odometry noise (~1 cm at 10 Hz) reads as ~0.05 m/s: settle_speed, and after 8 s centring
                # within 2x the tolerance is accepted
                t_desc = now - self.t_state
                ready = ((off <= sp.pad_over_m and abs(yerr) <= sp.pad_yaw_tol and speed < sp.settle_speed)
                         or (t_desc > 8.0 and off <= 2 * sp.pad_over_m and abs(yerr) <= 2 * sp.pad_yaw_tol))
            self.pad_ok_since = (self.pad_ok_since or now) if ready else (self.pad_ok_since if z < self.z_odom_cruise - 0.1 else None)
            descending = self.pad_ok_since is not None and now - self.pad_ok_since >= sp.pad_settle_s
            yr = 0.0 if descending else float(np.clip(sp.turn_kp * yerr, -sp.turn_rate, sp.turn_rate))
            note = ("descending onto the pad" if descending else
                    f"centring over the pad ({off:.3f} m, yaw {math.degrees(yerr):+.1f} deg)") + f" [{self.pad_source}]"
            self._cmd(cmd, self._hold_vel(xy), yr, note)
            cmd.vz = -sp.land_vz if descending else 0.0
            cmd.land = descending and z <= sp.landed_z_odom_m
            cmd.goal = "pad"
        cmd.state = self.state
        return cmd

    def status(self):
        st = super().status()
        st.update(pad_source=self.pad_source, pad_fixes=self.pad_fuser.n,
                  pad_xy=[round(float(v), 3) for v in self.goals[-1].xy],
                  phase=self.phase, cross_offset_m=round(self.cross_offset, 3),
                  seg=None if self.seg is None else dict(b=[round(float(v), 3) for v in self.seg["b"]],
                                                         v=self.seg["v"], gate=self.seg["gate"]),
                  gate_refused=getattr(self, "gate_refused", None))
        return st
