# Package 154 new flight failure and paused fix

**Paused at Matthew's request on 2026-11-08. No navigation fix has been implemented or deployed.** This handoff records the deployment, the observed failure, the explanation, and where to resume. It is not approval for another flight.

## Plain-English explanation

The planner handed the controller a route that it labeled valid, but whose first approximately 23 cm had a speed limit of zero. The controller was in its `go` state while commanding zero forward progress. It could not reach the later part of the route where motion was permitted. That is the software defect: the planner accepts an initial escape route that its own speed rules make impossible to traverse, and the controller reports `go` rather than identifying the unusable route.

There is also a real safety constraint behind the zero. The planner masks every cell inside its obstacle margin to a clearance of zero. That loses the distinction between a physically clear point that is merely inside the extra margin, and a point whose modeled obstacle distance is smaller than the rotor footprint. In this run the unmasked prior-map distance at the estimated start was only about **0.21 m**, smaller than the configured **0.2941 m rotor reach**. A minimum-speed override would not be a safe fix.

The left turn and descent were separate events. The planner commanded a left turn to face the new path from the estimated takeoff position. Later, PX4 changed from Offboard to Position mode with no failsafe flagged and with RC throttle near minimum. The autonomy yielded; the descent followed this mode change. The policy did not enter its landing state.

## Repository and installed state

- Original repository: `https://github.com/matthehzhang/dronehacks-deploy.git`, remote `origin`.
- Requested branch: `package_154_new`; installed source commit: `92b0c1fbd5b3e9fad1b52b18cd8984465554451c`.
- Fix worktree: `/home/matthew/projects/hackathon/dronehacks/deploy-154new-fix`, local branch `fix/package154-start-clearance`, based on that exact commit.
- Drone identity: `skybeta-14`, SSH alias `dronehacks`; USB address `192.168.55.1`. The alias can fall back to `skybeta-14.local` over Wi-Fi. Both connectivity and mDNS have been intermittent.
- Active source: `/home/dronehacks/team_ws/src/kestrel_nav` (not a separate saved-policy directory).
- Source/overlay backup: `/home/dronehacks/backups/pre-package154-new-20261108T102245Z/`, including previous `kestrel_nav`, build, install, tools, environment, build/test output and startup log.
- Runtime: `skybeta-stack`, image `skybeta:0.3.3`, ROS Humble. Source is mounted at `/root/team_ws/src/kestrel_nav` inside the container.
- Boot selection: `COMPETITOR_PKG=kestrel_nav`, `COMPETITOR_LAUNCH=refnav_skybeta.launch.py`.
- Deployment verified all 152 package source files against the branch, completed a clean native colcon build, and passed 29 focused native tests. These tests did not expose the later flight failure and are not hardware-flight qualification.
- No PX4 parameters, organizer monitoring, camera calibration, or flight-controller firmware were changed for this deployment.

The existing local `deploy` checkout is on `kestral_nav934am` and has unrelated uncommitted changes to `team_ws/tools/fly.sh`. Do not reset or overwrite it. `deploy-group154` and `deploy-blocked-recovery` are separate worktrees used by other work.

## Run and evidence

Analyzed run on the Jetson:

```text
~/team_ws/src/kestrel_nav/logs/run_20261108T103950/
  flight_trace.jsonl       approximately 9.4 MB
  navigator.log
  route.json
  plan.json
~/team_ws/src/kestrel_nav/logs/coursenav_20261108T103951.csv
~/team_ws/flight_logs/20261108-023942_group154_gqa92E/bag/
  bag_0.mcap              approximately 23 MB
  metadata.yaml
```

The directory timestamps use differing time conventions. Correlate using the recorded epoch/ROS timestamps, not the apparent hour in the folder name.

Compact extracted evidence committed with this handoff:

- [First turn and path](evidence/package154_20261108/first_turn.json): commanded/observed yaw, initial preview, actual first path, speeds and lateral slack.
- [Stall and descent](evidence/package154_20261108/stop_and_landing.json): phase counts, first/last `go` status, final trace tick, PX4 mode/arming/failsafe changes, RC throttle and vertical-position samples.
- [Installed launcher snapshot](evidence/package154_20261108/launcher_154new.sh): the helper installed on the Jetson, preserved as evidence rather than a navigation change.

The full trace/bag were read on the Jetson, not copied into this handoff. The bag includes vehicle status, manual-control setpoints, failsafe flags, trajectory/twist commands, course status/drift, and odometry. It does **not** contain `/coursenav/local_occupancy` or depth images, so the exact 33-cell fused obstacle grid cannot be reconstructed from this bag alone. The prior map already reproduces the zero-speed problem.

## Observed sequence

| Epoch seconds | Event |
| --- | --- |
| 1794134450.764 | PX4 enters Offboard; policy goes `arm -> climb` |
| 1794134455.117 | Policy goes `climb -> turn`, first of six legs |
| 1794134455.197 | First valid A* path arrives; 0.65 m long with one stop corner |
| 1794134455.218 | `check -> turn: new A* route: face path before checking fresh depth` |
| 1794134456–4457 | Yaw follows approximately 90 degrees to 68.6 degrees NED, a roughly 22-degree left turn |
| 1794134460.668 | First-turn settling gate releases after 4.5 seconds |
| 1794134460.871–4475.968 | `go`, but reported forward `v_cmd` remains exactly 0.0 |
| 1794134476.010 | PX4 changes Offboard -> Position; user-intended mode also Position; `failsafe=false` |
| 1794134476.018 | Policy yields because PX4 left Offboard; no landing phase |
| 1794134476.530 | Recorded RC throttle is -1.0, near/bottom of its range |
| 1794134477.108 | Descent is evident: local NED z changes from approximately -0.70 m to -0.18 m |
| 1794134484.089 | Land detector reports landed |
| 1794134486.110 | PX4 reports disarmed |

Interpretation: the bag supports descent after return to RC-controlled Position mode with low throttle, not an autonomy landing or flagged PX4 failsafe. Do not infer the pilot's intent from this alone.

## Why the first turn was left

The declared launch pose is course ENU `(-2.515, 0.914)`, yaw 90 degrees. The first `entrance_approach` goal is `(-2.5, 1.35)`: approximately 0.436 m forward and 0.015 m right relative to that pose. The larger rightward route comes after this first stop.

At the first live plan, the estimated course position was `(-2.4067, 0.7239)`, about 11 cm right and 19 cm behind the mark. The first path points run toward `(-2.525, 1.025)` before reaching the goal. The follower deliberately faced that leftward correction. The published yaw-rate sign and measured heading agree; there is no evidence here of a reversed yaw command. Whether the estimated offset represented actual drift, estimator drift, or map-placement error remains unresolved.

## Exact deadlock mechanism

Relevant active modules are the top-level `coursenav/` and `refnav/` under `team_ws/src/kestrel_nav`, not the separate legacy `kestrel_nav/coursenav/` modules used by some tests.

1. `coursenav/navigator.py: Navigator._effective()` sets clearance to zero in every blocked cell, including cells blocked only by inflated safety margins and semantic keep-outs.
2. `Navigator._free_start()` and `Navigator.plan()` allow a connector from the aircraft's estimated position inside that margin to a free cell. They pass the masked clearance field into smoothing and speed profiling.
3. `coursenav/smooth.py: speed_profile()` computes lateral slack as `max((clearance - reach_m) / 2, 0)` and caps speed by `slack / tube_response_s`.
4. The connector's first six samples, spanning 0.2311 m, consequently have zero slack and zero speed. The first positive speed is at about 0.2773 m.
5. `refnav/core.py: Mission._go_path()` samples the speed profile ahead of the current projected position, then takes that speed into the minimum of its safety/ramp limits. This run never progressed far enough to sample a positive value. The hold position remained latched and forward speed stayed zero.

Trace data show no stale-path waits, depth holds, tube-gate holds, or obstacle-stop events during this `go` interval. Healthy/fresh depth alone cannot overcome a zero path-speed limit. A zero speed at a legitimate start/end/stop corner is normal; a finite initial segment of zero permitted speed that cannot be traversed is the defect.

## Offline reproduction already completed

Run from `team_ws/src/kestrel_nav` in this repository, with NumPy/SciPy available:

```python
from coursenav.course import load_course, resolve, plan_grids
from coursenav.objmap import CourseMap
from coursenav.navigator import Navigator
from coursenav.navplan import smooth_params_for

doc = load_course()
cmap = CourseMap.load(resolve(doc, 'map'))
obst, blocked, clear = plan_grids(cmap, doc)
need = doc['vehicle']['radius_m'] + doc['vehicle']['margin_m']
sp = smooth_params_for(doc['navigation'], doc['refnav_params'], need)
nav = Navigator(obst, blocked, clear, need, sp)
path = nav.plan([-2.4067, 0.7239], [[-2.5, 1.35]])
print(path.pts, path.v, path.clearance, path.duration())
```

Observed before any fix: essentially the same path geometry as the flight, six initial zero-speed samples, and duration approximately 234.26 seconds for a 0.65 m path. The unmasked prior-map clearances at the first six recorded points are approximately `[0.210, 0.260, 0.310, 0.360, 0.360, 0.410]` m. The effective field replaces all six with zero. Full required clearance is `0.4140828453` m; configured physical reach is `0.2941` m.

**The first two points are inside the modeled rotor reach, not merely inside the extra planning margin.** A raw-clearance repair alone cannot justify driving this exact start through the modeled obstacle footprint.

## Unimplemented repair direction

No production file or new regression test was changed before the pause. The following is a design direction, not an approved or validated patch:

- Preserve real obstacle distance separately from the inflated/semantic blocked mask; keep ordinary A*, shortcut pruning, gate constraints and smoothing at full required clearance.
- Explicitly distinguish any permitted initial margin-recovery connector from normal travel. Certify the entire connector, including intermediate cells, physical rotor clearance, keep-outs, gate boundaries and a return to full margin.
- Reject unsafe starts or impossible zero-speed travel with an explicit planner reason. Never label an unusable path valid and enter a silent `go` deadlock.
- For a genuinely clear margin-recovery connector, derive speed from verified available clearance; do not impose an arbitrary positive speed floor.
- Preserve fresh depth, estimator health, RC takeover and landing checks. Do not shrink the aircraft, delete prior walls, reinterpret missing depth as free, or alter PX4 parameters to get a pass.
- Decide how the controller should hold/replan or otherwise handle a rejected start. Current missing-path behavior may itself hold indefinitely; returning invalid alone is not proof of a complete recovery fix.
- Add regression cases for the recorded failure, a physically clear margin start, true footprint overlap, an intermediate obstacle/keep-out/gate crossing, legitimate stop points, and an obstacle appearing after a plan was published.
- Validate pure planner/controller behavior first, then native ROS package tests and a build. Any simulation or real-flight result must be recorded separately; none has been run for a fix.

## Launcher behavior and operator commands

From the laptop, the installed normal entry point is:

```bash
ssh -t dronehacks '~/team_ws/tools/fly.sh 154new'
```

The drone's `fly.sh` aliases `154new`, `group154_new`, and `package_154_new` to:

```bash
PACKAGE154_NO_WATCH=1 "$HERE/154new.sh" restart || exit $?
watch_mission
```

The helper verifies package provenance and boot selection, requires fresh direct PX4 disarmed-and-landed messages, stops the old recorder and competitor process, rebuilds/restarts **only the competitor pane**, starts a fresh recorder, and returns to the usual log view. It does not restart the camera, SLAM, bridge, MicroXRCE agent or organizer monitoring. It never arms. `arming_state=1` is disarmed; `wait -> arm` is waiting for the pilot, not an arming command.

Read-only status:

```bash
ssh dronehacks '~/team_ws/tools/154new.sh status'
```

The helper also supports `start`, `restart`, `watch`, `check`, and `--help`. Direct `start` attaches to an already-running mission rather than restarting it. `fly.sh 154new` explicitly requests restart. Recorder helpers are `group154_record.sh` and `group154_record.py`; recordings use the historical `_group154_` directory label. Ctrl-C in the `fly.sh` log view stops watching only, not the mission or recorder.

The source backup is `~/backups/pre-package154-new-20261108T102245Z`. The fly alias edit was backed up to `~/team_ws/tools/fly.sh.bak-154new-20261108T103139Z`. The last attempt to re-read the drone while preparing this handoff failed on mDNS; no current flight/readiness state is asserted by this document.
