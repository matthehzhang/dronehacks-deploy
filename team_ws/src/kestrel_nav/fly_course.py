#!/usr/bin/env python3
"""The competition start: ONE command plans the course onboard and flies it (rules 5.2, 5.4, 7.4).

    python3 fly_course.py                      # localizer estimate -> goals from the OBJ -> plan -> fly
    python3 fly_course.py --dry-run            # plan and print only (no ROS node is started, nothing arms)
    python3 fly_course.py --event-start        # rule 7.4: the team armed on the mark; start on the signal

What happens, in order:
  1. The start pose comes from the coursenav localizer, which runs from boot as an onboard service
     (/coursenav/status: x, y, yaw, sigma). It has already converged on the ground. With --start the
     pose is given on the command line (tests); with --allow-prior the course file's declared start
     is used if the localizer is silent (training only).
  2. The landed body height comes from the downward rangefinder (/fmu/out/distance_sensor + its
     mount offset), so the body flies at the course's cruise height above the FLOOR.
  3. Goals are derived from the published OBJ (coursenav.semantics), every chicane option is planned
     and scored, the best is chosen (coursenav.planner.plan_options). The scores, the reason, the
     time estimate and refnav's blocking rule replayed on the map are printed. A predicted false block
     aborts (unless --force).
  4. A run record (plans/route + the scored plan) is written under logs/run_<stamp>/ (gitignored; a
     record of THIS run, never an input to a later one: rule 6.4).
  5. refnav (refnav.node) starts with that route, the course's refnav_params, the localizer's
     correction topic and, with --event-start, accept_armed_start. It takes off promptly (PX4
     auto-disarms an idle armed vehicle; that timeout is not changed).

route.json and plans/route_*.json are not needed. No coordinate is stored anywhere: the plan is
computed here from the map and the start estimate every time.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

RANGEFINDER_BELOW_ORIGIN_M = 0.027      # Sky Beta VL53L1X mount z (airframe file); loc_node uses the same


def _writable_dir(path):
    """Create path (and parents) and prove a file can be written there. -> path or None."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok")
        probe.unlink()
        return path
    except OSError:
        return None


def run_record_dir(stamp, project=PROJECT, env=None):
    """Where this run's record (route.json that refnav flies, plan.json) is written, first writable of:
    $RUN_RECORD_DIR; <project>/logs (terminal runs); the managed run's own output folder when the
    project is a read-only managed snapshot (<root>/companion-code/<name>/project ->
    <root>/companion-output/<name>/, so the record stays with that run's evidence); ~/.local/state.
    Live 2026-11-07: a managed "Run saved file" failed with Errno 30 writing <snapshot>/logs."""
    import os
    env = os.environ if env is None else env
    name = f"run_{stamp}"
    candidates = []
    if env.get("RUN_RECORD_DIR"):
        candidates.append(Path(env["RUN_RECORD_DIR"]) / name)
    candidates.append(Path(project) / "logs" / name)
    snap = Path(project).resolve()
    if snap.name == "project" and snap.parent.parent.name == "companion-code":
        candidates.append(snap.parent.parent.parent / "companion-output" / snap.parent.name / name)
    candidates.append(Path(env.get("HOME", "/tmp")) / ".local" / "state" / "dronehacks" / "runs" / name)
    for c in candidates:
        if _writable_dir(c) is not None:
            return c
    raise SystemExit(f"no writable place for the run record (tried {[str(c) for c in candidates]})")


def start_own_localizer(run_dir):
    """Start coursenav.loc_node as a CHILD of this run when no localizer is publishing (e.g. right
    after a Reset, whose default service profile has none), logging into this run's record folder.
    -> Popen or None (one is already running: never a second /coursenav publisher)."""
    import os
    import subprocess
    try:
        import rclpy
        from rclpy.node import Node
        own = not rclpy.ok()
        if own:
            rclpy.init()
        n = Node("fly_course_probe")
        t = time.monotonic()
        while time.monotonic() - t < 3.0:                 # DDS discovery
            rclpy.spin_once(n, timeout_sec=0.1)
        running = n.count_publishers("/coursenav/status") > 0
        n.destroy_node()
        if own:
            rclpy.shutdown()
    except ImportError:
        return None
    if running:
        print("localizer: using the one already running", flush=True)
        return None
    args = [sys.executable, "-m", "coursenav.loc_node", "--ros-args",
            "--params-file", str(PROJECT / "onboard" / "config" / "common.yaml")]
    deployment = Path("/etc/flyx/onboard.yaml")           # the deployment's clock selection (sim: use_sim_time)
    if deployment.is_file():
        args += ["--params-file", str(deployment)]
    # one scan snapshot per second into the run record, so the particle filter can be replayed offline
    # against truth (estimator analysis 2026-11-07: the localizer error repeats by location and its
    # reported sd is overconfident; replay needs the scans). A few MB per flight.
    args += ["-p", f"log_dir:={run_dir}", "-p", "snapshot_every_s:=1.0"]
    env = dict(os.environ, PYTHONPATH=f"{PROJECT}:{os.environ.get('PYTHONPATH', '')}")
    log = open(run_dir / "localizer.log", "w")
    print(f"localizer: none running, starting one for this run (log {run_dir / 'localizer.log'})", flush=True)
    return subprocess.Popen(args, cwd=str(PROJECT), env=env, stdout=log, stderr=subprocess.STDOUT)


def start_navigator(run_dir):
    """coursenav.nav_node as a CHILD of this run (the managed-run launcher ends it with the program),
    on this run's route record. -> Popen."""
    import os
    import subprocess
    args = [sys.executable, "-m", "coursenav.nav_node", "--ros-args", "-p", f"route_file:={run_dir / 'route.json'}"]
    if os.environ.get("KESTREL_NICE_NAV"):                 # skybeta: never compete with refnav's setpoint loop
        args = ["nice", "-n", str(int(os.environ["KESTREL_NICE_NAV"]))] + args
    deployment = Path("/etc/flyx/onboard.yaml")           # sim clock selection, like the localizer
    if deployment.is_file():
        args += ["--params-file", str(deployment)]
    if os.environ.get("KESTREL_USE_SIM_TIME") in ("true", "false"):   # skybeta launch (kestral_nav934am)
        args += ["-p", f"use_sim_time:={os.environ['KESTREL_USE_SIM_TIME']}"]
    if os.environ.get("KESTREL_DECLARED_HEADING_DEG"):            # skybeta: 0 (the interface permutes odometry ENU)
        args += ["-p", f"declared_heading_deg:={float(os.environ['KESTREL_DECLARED_HEADING_DEG'])}"]
    env = dict(os.environ, PYTHONPATH=f"{PROJECT}:{os.environ.get('PYTHONPATH', '')}")
    log = open(run_dir / "navigator.log", "w")
    print(f"navigator: starting (log {run_dir / 'navigator.log'})", flush=True)
    return subprocess.Popen(args, cwd=str(PROJECT), env=env, stdout=log, stderr=subprocess.STDOUT)


def wait_for_start(timeout_s, min_updates=30, max_sigma=0.15, want_height=True):
    """Spin a small ROS node until the localizer reports a converged estimate (and the rangefinder a
    landed height). -> (x, y, yaw_deg, sigma_xy, height_m or None, status dict)."""
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

    class Listener(Node):
        def __init__(self):
            super().__init__("fly_course_start")
            self.status = None
            self.height = None
            self.create_subscription(String, "/coursenav/status", self._st, 10)
            try:
                from px4_msgs.msg import DistanceSensor
                qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 history=HistoryPolicy.KEEP_LAST, depth=1)
                self.create_subscription(DistanceSensor, "/fmu/out/distance_sensor", self._rf, qos)
            except ImportError:
                pass

        def _st(self, m):
            try:
                self.status = json.loads(m.data)
            except ValueError:
                pass

        def _rf(self, m):
            if 0 < m.current_distance <= m.max_distance:
                self.height = float(m.current_distance) + RANGEFINDER_BELOW_ORIGIN_M

    rclpy.init(args=["fly_course_start", "--ros-args", "-p", "use_sim_time:=true"]
               if (PROJECT / "flight.yaml").exists() and "use_sim_time: true" in (PROJECT / "flight.yaml").read_text()
               else None)
    node = Listener()
    t0 = time.monotonic()
    out = None
    try:
        while time.monotonic() - t0 < timeout_s:
            rclpy.spin_once(node, timeout_sec=0.2)
            st = node.status
            if st and st.get("ready") and st.get("updates", 0) >= min_updates and st.get("sigma_xy") is not None \
                    and st["sigma_xy"] <= max_sigma and (node.height is not None or not want_height):
                out = (st["x"], st["y"], st["yaw_deg"], st["sigma_xy"], node.height, st)
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return out


def _free_start(blocked, start_xy, max_m=0.3):
    """start_xy, or the nearest unblocked cell centre within max_m of it (None-safe: start_xy if none)."""
    import numpy as np
    xy = np.asarray(start_xy, float)
    ij = blocked.ij(xy)
    if not blocked.inside(ij) or not blocked.occ[ij[0], ij[1]]:
        return tuple(start_xy)
    n = int(math.ceil(max_m / blocked.res))
    best, bd = None, math.inf
    for di in range(-n, n + 1):
        for dj in range(-n, n + 1):
            c = np.array([ij[0] + di, ij[1] + dj])
            if blocked.inside(c) and not blocked.occ[c[0], c[1]]:
                d = math.dist(blocked.xy(c), xy)
                if d < bd:
                    best, bd = blocked.xy(c), d
    if best is None or bd > max_m:
        return tuple(start_xy)
    print(f"start ({xy[0]:.3f}, {xy[1]:.3f}) is inside the planning margin: planning from the nearest free point "
          f"({best[0]:.3f}, {best[1]:.3f}), {bd:.3f} m away", flush=True)
    return (float(best[0]), float(best[1]))


def plan_now(start_xy, start_yaw_deg, chicane="auto", prepared=None):
    """Goals from the OBJ, every option planned and scored, the best chosen. -> (plan, route, legs, doc)."""
    from coursenav import planner
    from coursenav.course import load_course, plan_grids, resolve
    from coursenav.objmap import CourseMap
    if prepared is None:
        doc = load_course()
        cmap = CourseMap.load(resolve(doc, "map"))
        grids = plan_grids(cmap, doc)
    else:
        doc, cmap, grids = prepared
    obst, blocked, clearance = grids
    reach = float(doc["vehicle"]["radius_m"])
    # kestral_nav934am (skybeta), the one planning change to group119: the venue launch mark is ~0.40 m from
    # the barrier, inside the 0.414 m planning margin, so planning from it fails ("start is blocked"; the
    # FlyX spawn was 0.2 m north). Plan from the nearest free cell within 0.3 m; the route stays relative
    # to the REAL start, so the first leg simply leaves the margin.
    plan_xy = _free_start(blocked, start_xy)
    if chicane == "auto":
        res = planner.plan_options(doc, obst, blocked, clearance, start_xy=plan_xy, cmap=cmap)
        p = res["plan"]
    else:
        p = planner.plan(doc, obst, blocked, clearance, start_xy=plan_xy, chicane=chicane, cmap=cmap)
    yaw = math.radians(start_yaw_deg)
    route = planner.to_route(p["waypoints"], start_xy, yaw)
    legs = planner.to_legs(p["waypoints"], obst, clearance, reach)
    return p, route, legs, doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="plan and print only; start nothing")
    ap.add_argument("--event-start", action="store_true", help="rule 7.4: accept an already armed, landed vehicle")
    ap.add_argument("--chicane", choices=["auto", "window", "gate_b"], default="auto")
    ap.add_argument("--start", nargs=3, type=float, metavar=("X", "Y", "YAW_DEG"), help="skip the localizer (tests)")
    ap.add_argument("--allow-prior", action="store_true",
                    help="training only: use the course file's declared start if the localizer is silent")
    ap.add_argument("--localizer-timeout", type=float, default=20.0)
    ap.add_argument("--landed-z", type=float, default=None, help="landed body height above the floor (else the rangefinder)")
    ap.add_argument("--hover", type=float, default=0.0)
    ap.add_argument("--force", action="store_true", help="fly even with predicted false blocks")
    ap.add_argument("--no-own-localizer", action="store_true",
                    help="never start a localizer here (default: start one when none is publishing)")
    ap.add_argument("--skip-preflight", action="store_true", help="tests only: do not gate arming on preflight")
    ap.add_argument("--start-from-mark", action="store_true",
                    help="skybeta: the start pose is the course file's launch mark and guideline heading (rule 7.4: "
                         "the drone is placed on the mark); the localizer still corrects in flight")
    ap.add_argument("--skybeta-return", action="store_true", help="confirmed touch-and-go and curved hardware return")
    ap.add_argument("--no-return", action="store_true",
                    help="disable the native-PX4 return; explicit --skybeta-return selects the hardware return instead")
    ap.add_argument("--nav", choices=["path", "legs", "steady"], default=None,
                    help="path: stops only, cruise legs along the navigator's smoothed paths on the fused map "
                         "(coursenav.nav_node + the localizer's drift EKF); legs: the planner's waypoint route with "
                         "bounded corrections; steady: legs flown straight, stopping at every vertex, turning only "
                         "while stopped, inside each leg's tube (navigation.steady). Default: navigation.mode")
    ap.add_argument("ros_args", nargs="*", help="extra refnav -p name:=value settings")
    a = ap.parse_args(argv)

    t0 = time.monotonic()
    # prepare the map and goals first: the only part that takes seconds, independent of the start pose
    from coursenav.course import load_course, plan_grids, resolve
    from coursenav.objmap import CourseMap
    doc = load_course()
    cmap = CourseMap.load(resolve(doc, "map"))
    prepared = (doc, cmap, plan_grids(cmap, doc))
    print(f"course and map ready in {time.monotonic() - t0:.1f} s", flush=True)

    stamp = time.strftime("%Y%m%dT%H%M%S")
    run_dir = run_record_dir(stamp)
    print(f"run record: {run_dir}", flush=True)
    own_loc = None
    if not a.start and not a.no_own_localizer:
        own_loc = start_own_localizer(run_dir)
    try:
        return _fly(a, t0, doc, prepared, run_dir, stamp)
    finally:
        if own_loc is not None and own_loc.poll() is None:
            own_loc.terminate()
            try:
                own_loc.wait(timeout=5)
            except Exception:
                own_loc.kill()


def require_sim_plan_ready(ready, process):
    """A capped simulation must not silently fall back to planning in flight."""
    if os.environ.get('KESTREL_USE_SIM_TIME') != 'true' or ready.exists():
        return
    import subprocess
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    raise SystemExit('Navigator did not finish preplanning within the 300 wall-second watchdog; simulation flight not started')


def skybeta_return_spec(doc):
    # Validate optional return settings only after the outbound landing. A bad
    # return height must not prevent planning/starting the primary mission.
    config = doc.get('return')
    config = config if isinstance(config, dict) else {}
    start, pad = doc.get('start', {}).get('xy'), doc.get('sequence', [{}])[-1].get('xy')
    side = 1
    if start is not None and pad is not None:
        # NE-positive arc normal corresponds to the course-frame right normal.
        # Choose the bulge toward the room's centre, away from the outside net.
        dx, dy = start[0]-pad[0], start[1]-pad[1]
        mid = ((start[0]+pad[0])/2, (start[1]+pad[1])/2)
        side = 1 if -mid[0]*dy + mid[1]*dx >= 0 else -1
    return dict(height_above_floor_m=config.get('cruise_z_m', 2.1), speed_mps=.3,
                arc_offset_m=.6096, arc_side=side)


def _fly(a, t0, doc, prepared, run_dir, stamp):
    height = a.landed_z
    source = "command line"
    if a.start:
        sx, sy, syaw = a.start
    elif getattr(a, "start_from_mark", False):
        sx, sy = doc["start"]["xy"]
        syaw = doc["start"]["yaw_deg"]
        source = "launch mark (course file)"
    else:
        got = None
        try:
            got = wait_for_start(a.localizer_timeout, want_height=height is None)
        except ImportError as e:
            print(f"no ROS here ({e})", flush=True)
        if got is not None:
            sx, sy, syaw, sigma, h, st = got
            source = f"localizer (sigma {sigma:.3f} m, {st.get('updates')} updates)"
            if height is None:
                height = h
        elif a.allow_prior:
            sx, sy = doc["start"].get("spawn_xy", doc["start"]["xy"])
            syaw = doc["start"]["yaw_deg"]
            source = "course file prior (training only)"
        else:
            raise SystemExit("no converged localizer estimate: start the coursenav service, or --allow-prior (training)")
    if height is None:
        height = 0.18
        print("landed height not measured: assuming 0.18 m (refnav test fixture)", flush=True)

    t1 = time.monotonic()
    p, route, legs, doc = plan_now((sx, sy), syaw, a.chicane, prepared)
    dt_plan = time.monotonic() - t1
    cruise = float(doc["cruise_z_m"])
    height_m = cruise - height
    print(f"start {source}: ({sx:.3f}, {sy:.3f}) yaw {syaw:.1f} deg; landed body height {height:.3f} m -> "
          f"height_m {height_m:.3f} (body at {cruise} m above the floor)")
    print(f"planned in {dt_plan:.2f} s: chicane {p.get('chicane')}: {p.get('choice_reason', '(forced)')}")
    print(f"  {len(route)} legs, {p['length_m']:.2f} m, min clearance {p['min_clearance_m']:.2f} m, estimated "
          f"{p['time_estimate_s']:.0f} s; leg-end violations {p['leg_end_violations']}; predicted false blocks "
          f"{p['predicted_false_blocks']}")
    if p["predicted_false_blocks"] and not a.force:
        raise SystemExit("refnav's rule replayed on the map predicts a false stop: not flying (--force to override)")

    params = {k: v for k, v in doc.get("refnav_params", {}).items() if not k.startswith("_")}
    params["landed_z_m"] = float(height)
    # simulator-configuration workarounds; refnav ignores them unless the node runs on sim time
    params.update({k: v for k, v in doc.get("sim_profile", {}).get("refnav", {}).items() if not k.startswith("_")})
    navcfg = {k: v for k, v in doc.get("navigation", {}).items() if not k.startswith("_")}
    nav_mode = a.nav or navcfg.get("mode", "legs")
    nav_legs = None
    if nav_mode == "steady":
        steady = {k: v for k, v in navcfg.get("steady", {}).items() if not k.startswith("_")}
        resp, cap = steady.pop("tube_response_s", None), steady.pop("leg_speed_cap_mps", None)
        params.update(steady)
        if resp and cap:
            from coursenav.planner import leg_speed_mps
            for L in legs:
                if L.get("tube_m"):
                    L["speed_mps"] = min(leg_speed_mps(L["tube_m"], resp, cap), L.get("end_room_mps", cap))
        tubes = ", ".join(f"{L.get('tube_m', 0.0):.2f}" for L in legs)
        speeds = ", ".join(f"{L.get('speed_mps', params['speed_mps']):.2f}" for L in legs)
        print(f"navigation: steady mode, {len(route)} straight legs, stop and turn at every vertex; "
              f"cross-track tubes (m): {tubes}; leg speeds (m/s): {speeds}")
    if nav_mode == "path":
        from coursenav.navplan import stops_and_legs
        route, legs, nav_legs = stops_and_legs(p, (sx, sy), math.radians(syaw))
        params.update({k: v for k, v in navcfg.get("refnav", {}).items() if not k.startswith("_")})
        params["track_mode"] = "path"
        params["correction_max_m"] = 0.0                # the drift EKF + whole-path conversion replace it
        print(f"navigation: path mode, {len(route)} stops ({sum(L['path'] for L in nav_legs)} navigator legs)")
    route_doc = {"route": route, "legs": legs, "params": params,
                 "_note": f"RUN RECORD of fly_course.py at {stamp}: planned from start {source} "
                          f"({sx:.4f}, {sy:.4f}, {syaw:.2f} deg). Rule 6.4: never an input to a later run."}
    from coursenav.course import land_columns
    route_doc["land_columns"] = land_columns(doc, prepared[1], (sx, sy), syaw)
    if getattr(a, 'skybeta_return', False):
        route_doc['skybeta_return'] = skybeta_return_spec(doc)
        print(f"touch-and-go return enabled: confirmed pad contact (stay armed) -> vertical climb to "
              f"{route_doc['skybeta_return']['height_above_floor_m']} m -> 0.6096 m curved offset -> "
              "original launch -> landing/disarm", flush=True)
    elif doc.get("return", {}).get("enabled", True) and not getattr(a, "no_return", False):
        from coursenav.return_plan import route_spec
        route_doc["automatic_return"] = route_spec(doc, (sx, sy, syaw), height)
        print(f"automatic return after pad landing/disarm: candidate height {doc['return']['cruise_z_m']:.2f} m; "
              "only after readiness, time and clearance checks; otherwise remain landed", flush=True)
    if nav_legs is not None:
        route_doc["nav_legs"] = nav_legs
        route_doc["start_xy"] = [float(sx), float(sy)]   # skybeta: the navigator pre-plans each leg from its start
        route_doc["gate_lines"] = p.get("gate_lines", [])
        route_doc["navigation"] = {k: v for k, v in navcfg.items() if k not in ("mode", "refnav")}
    (run_dir / "route.json").write_text(json.dumps(route_doc, indent=1))
    (run_dir / "plan.json").write_text(json.dumps({k: v for k, v in p.items() if k != "plans"}, indent=1, default=str))
    if a.dry_run:
        for w in p["waypoints"]:
            print(f"   {w['name']:14s} ({w['xy'][0]:6.2f},{w['xy'][1]:6.2f})  clear {w['clearance_m']:.2f}")
    if not a.skip_preflight:
        from coursenav.preflight import report, run_checks
        rows = run_checks(6.0, allow_armed=a.event_start, need_localizer=not a.start)
        (run_dir / "preflight.txt").write_text("\n".join(f"{'ok' if r.ok else ('FAIL' if r.hard else 'warn')}\t{r.name}\t"
                                                       f"{r.detail}" for r in rows) + "\n")
        passed = report(rows)
        if a.dry_run:
            print("dry run: nothing started or armed", flush=True)
            return 0 if passed else 1
        if not passed:
            raise SystemExit("preflight blocked the run: nothing armed (every failed check is listed above)")
    elif a.dry_run:
        return 0

    from refnav import node
    from refnav.launch import arguments
    extra = ["-p", "correction_topic:=/coursenav/px4_correction"] if nav_mode != "path" else []
    extra += ["-p", f"trace_file:={run_dir / 'flight_trace.jsonl'}"]
    nav_proc = start_navigator(run_dir) if nav_mode == "path" else None
    if nav_proc is not None:                    # skybeta: refnav (and so arming) only after the paths exist
        ready, t_wait = run_dir / "navigator_ready", time.monotonic()
        while not ready.exists() and nav_proc.poll() is None and time.monotonic() - t_wait < 300.0:
            if int(time.monotonic() - t_wait) % 10 == 0:
                print(f"waiting for the navigator to pre-plan the legs ({time.monotonic() - t_wait:.0f} s)", flush=True)
            time.sleep(1.0)
        require_sim_plan_ready(ready, nav_proc)
        print(f"navigator: {'ready' if ready.exists() else 'NOT ready (planning in flight)'} after "
              f"{time.monotonic() - t_wait:.0f} s", flush=True)
    if a.event_start:
        extra += ["-p", "accept_armed_start:=true"]
    extra += [x for x in a.ros_args if x != "--"]
    args = arguments(PROJECT, height_m=height_m, hover_s=a.hover, ros_args=extra)
    args = [x if not x.startswith("route_file:=") else f"route_file:={run_dir / 'route.json'}" for x in args]
    print(f"Flight source: {Path(node.__file__).parent}; starting refnav {time.monotonic() - t0:.1f} s after start",
          flush=True)
    try:
        node.main(args)
    finally:
        if nav_proc is not None and nav_proc.poll() is None:
            nav_proc.terminate()
            try:
                nav_proc.wait(timeout=5)
            except Exception:
                nav_proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
