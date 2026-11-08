#!/usr/bin/env python3
"""Start the depth-only receding-horizon mission from global waypoint intent."""
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
    if os.environ.get('KESTREL_PILOT_VERIFIED_LAUNCH_RADIUS_M'):
        args += ['-p', f"pilot_verified_launch_radius_m:={float(os.environ['KESTREL_PILOT_VERIFIED_LAUNCH_RADIUS_M'])}"]
    env = dict(os.environ, PYTHONPATH=f"{PROJECT}:{os.environ.get('PYTHONPATH', '')}")
    log = open(run_dir / "navigator.log", "w")
    print(f"navigator: starting (log {run_dir / 'navigator.log'})", flush=True)
    return subprocess.Popen(args, cwd=str(PROJECT), env=env, stdout=log, stderr=subprocess.STDOUT)


def plan_now(start_xy,start_yaw_deg,chicane=None,prepared=None):
    from coursenav.course import load_course
    from coursenav.waypoint_intent import plan_intent
    return plan_intent(prepared[0] if prepared else load_course(),start_xy,start_yaw_deg)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="plan and print only; start nothing")
    ap.add_argument("--event-start", action="store_true", help="rule 7.4: accept an already armed, landed vehicle")
    ap.add_argument("--start", nargs=3, type=float, metavar=("X", "Y", "YAW_DEG"), help="skip the localizer (tests)")
    ap.add_argument("--landed-z", type=float, default=None, help="landed body height above the floor (else the rangefinder)")
    ap.add_argument("--hover", type=float, default=0.0)
    ap.add_argument("--no-own-localizer", action="store_true",
                    help="never start a localizer here (default: start one when none is publishing)")
    ap.add_argument("--skip-preflight", action="store_true", help="tests only: do not gate arming on preflight")
    ap.add_argument("--start-from-mark", action="store_true",
                    help="skybeta: the start pose is the course file's launch mark and guideline heading (rule 7.4: "
                         "the drone is placed on the mark); the localizer still corrects in flight")
    ap.add_argument("--no-return", action="store_true",
                    help="never fly the automatic return after the pad landing (skybeta: the pilot arms, not refnav)")
    ap.add_argument("ros_args", nargs="*", help="extra refnav -p name:=value settings")
    a = ap.parse_args(argv)

    t0 = time.monotonic()
    # prepare the map and goals first: the only part that takes seconds, independent of the start pose
    from coursenav.course import load_course
    doc = load_course()
    prepared = (doc,None,None)
    print(f"waypoint configuration ready in {time.monotonic() - t0:.1f} s", flush=True)

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


def _fly(a, t0, doc, prepared, run_dir, stamp):
    height = a.landed_z
    source = "command line"
    if a.start:
        sx, sy, syaw = a.start
    else:
        sx,sy=doc['start']['xy']
        syaw=doc['start']['yaw_deg']
        source='declared VIO launch fixture'
        if doc['start'].get('forward_offset_m'):
            source += (f" moved {doc['start']['forward_offset_m']:.4f} m forward of the mark "
                       f"{tuple(round(v, 3) for v in doc['start']['declared_xy'])} (start_forward_in)")
    if height is None:
        height=.18
        print('landed height not measured: assuming 0.18 m (test fixture)',flush=True)

    t1 = time.monotonic()
    p, route, legs, doc = plan_now((sx, sy), syaw, None, prepared)
    dt_plan = time.monotonic() - t1
    cruise = float(doc["cruise_z_m"])
    height_m = cruise - height
    print(f"start {source}: ({sx:.3f}, {sy:.3f}) yaw {syaw:.1f} deg; landed body height {height:.3f} m -> "
          f"height_m {height_m:.3f} (body at {cruise} m above the floor)")
    print(f"planned in {dt_plan:.2f} s: chicane {p.get('chicane')}: {p.get('choice_reason', '(forced)')}")
    if p.get('clearance_pending'):
        print(f"  {len(route)} waypoint targets; executable routes and clearance come from live depth only")
    else:
        print(f"  {len(route)} legs, {p['length_m']:.2f} m, min clearance {p['min_clearance_m']:.2f} m, estimated "
              f"{p['time_estimate_s']:.0f} s; leg-end violations {p['leg_end_violations']}; predicted false blocks "
              f"{p['predicted_false_blocks']}")

    params = {k: v for k, v in doc.get("refnav_params", {}).items() if not k.startswith("_")}
    navcfg = {k: v for k, v in doc.get("navigation", {}).items() if not k.startswith("_")}
    nav_mode = 'path'
    yaw=math.radians(syaw);c,s=math.cos(yaw),math.sin(yaw)
    nav_legs=[dict(name=w['name'],stop=w['xy'],goals=[w['xy']],path=True) for w in p['waypoints'][1:]]
    params.update(navcfg.get('refnav',{}))
    params.update(track_mode='path',local_horizon=True)
    route_doc = {"route": route, "legs": legs, "params": params,
                 "_note": f"RUN RECORD of fly_course.py at {stamp}: planned from start {source} "
                          f"({sx:.4f}, {sy:.4f}, {syaw:.2f} deg). Rule 6.4: never an input to a later run."}
    if nav_legs is not None:
        route_doc["nav_legs"] = nav_legs
        route_doc["start_xy"] = [float(sx), float(sy)]   # skybeta: the navigator pre-plans each leg from its start
        route_doc["gate_lines"] = p.get("gate_lines", [])
        route_doc["navigation"] = {k: v for k, v in navcfg.items() if k not in ("mode", "refnav")}
    (run_dir / "route.json").write_text(json.dumps(route_doc, indent=1))
    (run_dir / "plan.json").write_text(json.dumps({k: v for k, v in p.items() if k != "plans"}, indent=1, default=str))
    if a.dry_run:
        for w in p["waypoints"]:
            clearance_text = 'live depth pending' if p.get('clearance_pending') else f"clear {w['clearance_m']:.2f}"
            print(f"   {w['name']:14s} ({w['xy'][0]:6.2f},{w['xy'][1]:6.2f})  {clearance_text}")
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
                print(f"waiting for the navigator to initialise depth planning ({time.monotonic() - t_wait:.0f} s)", flush=True)
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
