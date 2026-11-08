# kestrel_nav: team wide-kestrel course code, packaged for skybeta


## Active group154 map (2026-11-08)

Window height: cruise remains **0.80 m above the floor** through the tunnel and
the approach to `window_pre`. At that existing stop, hold the XY/yaw target and
raise the body origin to **1.40 m** at a setpoint rate of 0.25 m/s. Start window
alignment only after measured height stays within **5 cm for 1 second**. Camera
height trim is limited to **+/-10 cm** around 1.40 m, followed by the same height
hold before crossing. Missing sill/header observations keep the nominal height.
Keep the crossing height on the route to the pad until the normal landing descent.
If the camera cannot confirm the window within the existing 4-second alignment
period, discard partial detections and continue the planned crossing at the settled
height, without lateral/height corrections, alignment recovery or an alignment-failure
landing. A prominent `WINDOW ALIGNMENT FALLBACK` message explains this. Tunnel
alignment, detected-obstacle responses and camera-stream-loss handling are unchanged.
This adds no automatic-landing trigger, velocity-validity requirement or continuous
height veto during crossing. Holding XY still allows feedback to counter drift.

Skybeta window altitude feedback: `/visual_slam/tracking/odometry` is aligned
to PX4's vertical coordinate using a rolling 2-second average of available
samples while landed or moving below 0.08 m/s horizontally and vertically.
Calibration runs during the existing route before the window crossing leg;
there is no minimum-sample wait. At `window_pre`, as the `window_post` leg
starts, freeze that offset and use VIO height for both mission height checks
and the twist altitude controller. Keep it through the pad approach because
PX4's terrain-height error may persist beyond the window. Targets stay in
their existing coordinate frame. Activation/fallback messages use `!!!!!!!!!!`.
With no fresh calibrated VIO, continue the existing PX4 behavior and warn.
After activation, a VIO gap over 1 second temporarily uses PX4 height changes
relative to the last VIO height; fresh VIO resumes with the frozen offset.
Neither case adds a mission hold or landing trigger. Status/trace records include
the height source and offset; raw PX4 local-position trace records are retained.
Native PX4 output, automatic return, and diagnostics do not use this handoff.
PX4 estimator parameters and its inner velocity controller are unchanged.
This change has CPU regression coverage, but has not been flight validated.

At the end, the skybeta launch disables the mission's landing timeout: the existing
0.30 m/s landing descent continues until PX4 reports disarmed. During that descent,
a fresh PX4 `landed=true` report triggers an explicit normal
`VEHICLE_CMD_COMPONENT_ARM_DISARM` command (`param1=0`, `param2=0`), retried once
per second while still landed and armed. The adapter publishes this command on
the version-resolved PX4 vehicle-command topic; motion still uses the flight
interface. The outbound flight sends no arm/mode commands; the automatic return
uses normal arm/Offboard commands only after the first landing and disarm.
It prints `DISARM COMMAND SENT`, then waits for actual PX4 disarm confirmation.
It then prints `LANDING CONFIRMED: PX4 reports DISARMED` and finishes. At 60 seconds without
confirmation it prints `DISARM NOT CONFIRMED` and keeps landing. PX4 owns the
normal disarm acceptance; a command being sent does not mark the mission complete.
No force-disarm command is added. This has not been flight validated.

The skybeta launch now enables `return_home:=true`. After successful pad landing
and disarm, a separate return layer holds the pad XY/yaw target, rearms, requests
Offboard and climbs vertically to **2.10 m above the original floor reference**.
The climb's commanded target must finish and the measured height must remain
within **5 cm for one second** before turning or translating toward the original
launch position. It then follows a circular arc home at 0.30 m/s, lands, and explicitly
disarms again. The arc has a 0.6096 m (2 ft) midpoint offset from the direct
pad-to-launch chord, bending toward the room centre. The first landing latch is cleared before second takeoff; late
outbound paths/corrections cannot change the return target. Recording spans both
flights, including a delayed rearm on the pad.

The existing outbound controller is unchanged. Return setup starts only after
successful outbound completion, never on abort/fault landings. Invalid return
configuration leaves the completed run on the pad. This skybeta return does not
use the native-PX4 return's battery/localizer-confidence/map-clearance/time-budget
qualification. Height 2.10 m is the operator-selected return height, not a checked
clearance guarantee. Existing controller input/obstacle handling still applies.
To fly only the committed outbound sequence, use
`ros2 launch kestrel_nav refnav_skybeta.launch.py return_home:=false`.

The user-supplied approximate window bounds **1.00-1.80 m** override alignment
geometry only. The old OBJ/voxel window is **0.305-1.219 m** and remains unchanged;
the horizontal planner still uses the existing cruise band. The old map shows
0.50 m horizontal clearance around the climb column over z=0.55-1.75 m, but cannot
verify the real raised crossing. Preview the actual preprocessed route and height
targets with `python tools/preview_window_height.py`; outputs go to
`plans/window_height_review/`. These are target diagrams, not flight simulations.

The final landing goal is now **(-2.0422, -1.5425) m**: 0.75 m farther from
the start along the vertical map axis (-course y), with x unchanged. The
pad's `from.offset_xy_m=[0.0, -0.75]` records this user-measured adjustment
relative to the imported marking; OBJ/voxel geometry remains unchanged.

After `tunnel_exit`, the route now requires a full stop at `tunnel_clearance`,
course ENU **(2.05, -1.85) m**, before turning toward the window. This is an
approximate placement from the user's pink waypoint to pass the rotated wall
tip. It is a separate stopping leg, so smoothing/replanning cannot shortcut
the goal. Existing `align` approach mode marks the stop; it has no gate
alignment measurement. The map is unchanged, so clearance to the real rotated
wall still needs checking. This detour has not been flight validated.

The configured launch position is now course ENU **(-2.515, 1.1172) m**, heading
**90 degrees**: 8 inches (0.2032 m) forward of the imported launch mark.
Place the drone at that new position before starting the stack; this changes
the planner and localizer's start assumption, not a commanded takeoff movement.

Path tracking uses `navigation.refnav.path_tube_recovery_mps=0.10`: outside
the narrow tracking tube, forward speed is capped at 0.10 m/s while position
feedback steers back toward the path. The planned speed ceiling returns inside
half the tube. Planned stops/turns and other motion limits still apply. Set
this parameter to 0 to restore the previous forward hold. This permits forward
travel outside the planner's tracking tube and has not been flight validated.

The refnav course profile now follows waypoints without requiring depth returns
along the path: `min_corridor_coverage`, `min_valid_frac`, and
`min_valid_frac_static` are 0, and `corridor_wait_land_s` is -1. Fresh blank or
sparse frames therefore permit motion; missing returns can hide real obstacles.
Measured obstacles still trigger hold/back-off/recheck at a fixed **0.35 m**
ahead of the body origin (`stop_m=0.35`, `stop_per_mps=0`), with three confirming
frames. This distance is not rotor clearance or a guaranteed stopping distance,
and is inside the documented OAK minimum range of about 0.4 m. Camera silence
still stops motion and triggers landing after 5 s. Estimator/PX4 faults, gate
alignment failures, persistent measured obstacles, and route completion can
still land. Source and packaged course profiles carry the same settings.

Refnav emits warning-level `!!!!!!!!!! AUTO LAND: <reason> !!!!!!!!!!` banners
when landing begins, including normal route completion. A fault that first
retraces to clear floor emits `AUTO LAND PENDING` with its reason. PX4 or pilot
takeover emits `CONTROL YIELDED`; refnav cannot infer PX4's internal failsafe
reason. These events also appear in the flight trace.

The default `course/course_v3.json` now selects `course/new_course_origin.obj`
and `maps/course_origin_5cm.npz`, imported from `new_course_origin_map.zip`.
Planning, localizer initialization/scan matching, occupancy, and safe-landing
columns use that same prior. The entrance route now plans toward the tunnel;
the obsolete maze-exit goal is absent. The source-tree and packaged deployment
assets match. Historical drawing fixtures are explicitly named `course_v3_legacy.json`.

This is the supplied approximate `topview_entrance` reconstruction, not a surveyed
venue map. Its unchanged historical scoring lines do not qualify this route as
an official course pass. Vehicle reach is 0.29408 m and the planning margin
remains 0.12 m. See `course/NEW_COURSE_README.txt` for the download's prior test
limitations; that record is not the result of the current deployment test.

The current compute test completed the route and disarmed in 86.79 s with zero
touches, but minimum measured outline clearance was 7.48 cm (below 12 cm), and
the unchanged scoring lines awarded zero points. The native workspace planner
rejects the exact supplied launch mark; the tested skybeta adapter uses its
existing bounded free-start projection. This is traversal evidence, not flight
qualification. See `deploy-group154/docs/evidence/new_course_20261108/summary.json`
in the parent checkout for the complete result.

`host_depth` assigns `array.array('B', depth_mm.tobytes())` to the ROS image field,
avoiding per-byte Python validation where enabled. Reproduce the payload and
serialization check with `python3 tools/benchmark_depth_payload.py` after sourcing ROS.


## Current branch: `kestral_nav934am` (safe policy)

The entry point for this branch is **`safe.launch.py`** (`COMPETITOR_LAUNCH=safe.launch.py`): a
safety-first course policy built on livenav's planning, localizer and live grid. Stop-turn-go (never
yaw while translating), depth-measured centring on the tunnel mouth and the window (no measurement, no
crossing), stopping-distance budget against the depth view, the 7-inch prop reach (0.294 m), and an
autonomous descent onto the pad. See **[docs/SAFE_POLICY.md](docs/SAFE_POLICY.md)** for the rules, the
evidence behind each, offline results (16/16 landed from the launch mark, 0 contacts, 150-168 s); matched to workspace matthew930am's steady-profile rules and the open risks. It has
not flown. `live.launch.py` and `tune.launch.py` are unchanged.

## Branch `diagnostic` (inherited)

This experimental branch adds `tune.launch.py`: slow map-guided transit, bounded
yaw measurements in the open area before the tube, straight-line response
measurements, forward-facing slow return, raw sensor and
command logging, and an offline fitter with held-out validation. It requires
measured lane geometry and never changes PX4 gains. See
[the diagnostic instructions](../../../docs/TUNING.md) for limits and evidence.
It has not flown; the synthetic rehearsal does not validate maze transit or the
cloud simulator's PX4 interface.

## Inherited mission from `matthew559am`

The entry point for this branch is **`live.launch.py`**. It keeps the hardware velocity follower
and organizer flight interface, adds the timestamp/tracking/depth watchdogs described below, and
ports the gate-planning fixes from workspace `matthew352am` at `dc2dd1c`.

Gate openings are excluded from ordinary routes, fallback searches, path shortcuts and follower
commands. Only the designated forward crossing leg opens its gate. A blocked crossing cannot
backtrack through the opening and cross it again. Gate B is refused at the configured 0.40 m
clearance; the default plan selects the window. The packaged alternative `navfield` planner gets
the same upstream gate fixes, but it is not the controller used by `live_mission`.

This is a **partial simulator port**. The simulator's trajectory follower, acceleration feedforward,
arming/mode commands, VIO wrapper and final landing sequence are not transferred. No controller
gains, particle-filter correction rates, global margins or stereo implementation are retuned here.
See [PORTING.md](PORTING.md) for provenance, verification and remaining limitations.

The older pose-waypoint mission documented next is retained for reference; it does **not** have
the new `live_mission` watchdogs. Use the live-mission section below for this branch's launch setup.

## Original pose-waypoint port

A ROS 2 (ament_python) package that drops into `~/team_ws/src` on a skybeta drone. It carries the
parts of the team workspace (github.com/AMMistry18/workspace) that are platform-independent and
replaces the parts that fought the skybeta stack.

| Ported as-is (import path changed only) | From | Why it carries over |
|---|---|---|
| `kestrel_nav/coursenav/{objmap,course,planner,semantics,navfield}.py` | `coursenav/` | Course map from the published OBJ, A* planner with chicane scoring, goal derivation. Pure numpy/scipy. |
| `kestrel_nav/coursenav/{pf,scan,occgrid,render}.py` | `coursenav/` | Particle-filter localizer core, depth-to-scan, live occupancy grid, map-rendered depth for tests. Pure. |
| `kestrel_nav/coursenav/depth.py` | `refnav/depth.py` | Quaternion / depth-decode helpers the localizer needs. Pure numpy, so vendored. |
| `kestrel_nav/coursenav/loc_node.py` | `coursenav/loc_node.py` | The ROS localizer node. Publishes only `/coursenav/*`. Needs a depth topic (see below). |
| `kestrel_nav/coursenav/viz.py`, `tools/plan_route.py`, `tools/build_map.py` | `coursenav/`, `tools/` | Offline planning PNG and map rebuild. OpenCV only for the PNG. |
| `kestrel_nav/data/course/*`, `kestrel_nav/data/maps/course_v3.npz` | `course/`, `maps/` | Course file, published OBJ, prebuilt 5 cm voxel map. Installed with the package. |
| `tests/test_{coursenav,occgrid,semantics}.py` | `tests/` | 52 tests, all passing here on numpy 2.5 / scipy 1.18. |

| New here | What it does |
|---|---|
| `kestrel_nav/course_mission.py` | The competitor node. Plans the course at start-up, then flies it through `/skybeta/setpoint/pose` the way `competitor_example/square_mission.py` does. Never touches `/fmu/in/*`. |
| `kestrel_nav/plan_course.py` | `ros2 run kestrel_nav plan_course [--start X Y YAW_DEG]`: offline dry run, prints what the mission will fly. |
| `launch/course.launch.py` | `flight_interface` exactly as the skybeta example starts it (`auto_offboard_on_arm:=true`) + `course_mission` (+ optional localizer). |
| `config/localizer.yaml` | Localizer topics for the skybeta stack. |

| Deliberately NOT ported | Why |
|---|---|
| `refnav/` (core, node, gate, gps, launch) | It is a PX4 offboard controller: publishes `/fmu/in/*`, arms, sets modes. skybeta forbids all three and `skybeta_flight_interface` already does the Offboard handshake and NED conversion. |
| `onboard/` (own cuVSLAM wrapper, own vision bridge, `/estimator/healthy`) | skybeta runs its own cuVSLAM and `px4_odom_bridge`; a second VIO publisher corrupts the EKF. x86-only `cuvslam` binary anyway. |
| `coursenav/preflight.py`, `tests/test_preflight.py` | Encodes the flyx contract (depth topic, `/estimator/healthy`, zero `/fmu/in` publishers). On skybeta the flight interface IS a `/fmu/in` publisher, so every check would fail. Write a skybeta one against `/skybeta/state` if wanted. |
| `coursenav/offline_sim.py`, `coursenav/tilt.py`, the closed-loop and gate-alignment tests, `tools/generality.py` | They drive `refnav.core`. |
| `fly_course.py`, `main.py`, `flight.yaml`, `simeval/`, `simsensors/`, `learn/`, the `.step` | Simulator-lab entry points and data. |

## Put it on skybeta-14

```bash
# from this repo, on the laptop (drone 14)
rsync -a --delete team_ws/src/kestrel_nav/ dronehacks@skybeta-14.local:team_ws/src/kestrel_nav/

# on the drone: point the competitor window at it, then restart the stack ON THE GROUND
ssh dronehacks@skybeta-14.local
sudoedit /etc/skybeta/drone.env       # COMPETITOR_PKG=kestrel_nav  COMPETITOR_LAUNCH=course.launch.py
docker rm -f skybeta-stack             # skybeta-stack.service restarts it in ~10 s; run_competitor.sh runs team_build.sh first
docker exec -it skybeta-stack tmux attach -t skybeta    # watch the competitor window; Ctrl-b d to leave
```

Dry run inside the container before any flight:

```bash
docker exec -it skybeta-stack bash -lc 'source skybeta_env.sh; ros2 run kestrel_nav plan_course'
```

Dependencies already in the `skybeta:0.3.2` image: ROS 2 Humble, numpy<2, python3-scipy 1.8 (apt), OpenCV.
Nothing extra to install. The code was tested here on numpy 2.x; numpy 1.x is the older, more permissive API.

## How the mission maps the plan onto the drone

The planner works in the COURSE frame (the OBJ's: origin floor centre, metres). cuVSLAM gives poses in
its own ENU `odom` frame, origin at the power-on pose, x along the nose at that moment. At the instant
the flight interface reports Offboard, the node takes the vehicle's odom pose `p0` and assumes it is
standing at the course start pose (the launch mark from the course file, or `start_x/start_y/start_yaw_deg`).
`planner.to_route` already expresses every waypoint as (forward, left) metres from the start pose, so
each one becomes `p0 + R(yaw0) * (forward, left)`. Heights: course `cruise_z_m` (0.8 m above the floor)
minus `landed_z_m` (body origin above the floor when landed) is the odom z to command.

With `use_localizer:=true` and the localizer running, every `/coursenav/pose` paired with the odometry
at that moment re-fits the course->odom transform, so VIO drift is corrected along the way.

## Open items before this flies

1. **Measure `landed_z_m` on skybeta-14** (body origin above the floor, landed). The default 0.15 m is a
   guess. Wrong by 10 cm = flying 10 cm off the planned band; the interface still floors at 0.3 m.
2. **Bench test on the ground, props off**: launch, arm, switch to Position, confirm `/skybeta/state`
   shows `offboard_active`, the log shows "climbing", and no `clamped` entries appear.
3. **Place the drone on the launch mark heading along the guideline** (the course-file prior) or pass
   the measured start pose. Without the localizer there is no correction for placement error.
4. **Localizer needs depth.** skybeta's `oakd_vslam.yaml` has no StereoDepth node. Add a `stereo:` block
   so depthai publishes `/oak/stereo/image_raw` + `camera_info`, check the extra load does not drop the
   cuVSLAM pair rate below what `fleet_guide.md` requires, and verify `/fmu/out/distance_sensor` exists.
   Until then fly without it (`use_localizer:=false`, the default).
5. **Gate and tunnel legs** are flown as plain waypoints with a `hold_seconds` pause. The old refnav gate
   alignment (depth-based lateral centering) is not ported; clearance is the planner's 0.40 m.
6. **Landing is the pilot's.** Over the pad the node descends to `min_altitude_m` and holds.

## Tests

```bash
cd team_ws/src/kestrel_nav && python3 -m pytest tests -q     # 52 passed, ~20 s
python3 -m kestrel_nav.plan_course                           # prints the 19-leg window plan
```

---

# livenav (branch `basic-obstacle-avoidance`): A* replanning on a live depth grid + velocity follower

`kestrel_nav/livenav/` is a second mission that reuses coursenav's map, goal derivation, A*,
particle-filter localizer, live occupancy grid and depth renderer, and replaces leg-by-leg
waypoint flying with:

| Module | What it does |
|---|---|
| `navgrid.py` | prior obstacle band (published OBJ) + live log-odds grid from this run's depth scans; only returns the prior does NOT explain become obstacles (`novel_min_m`), so VIO drift cannot thicken known walls; the net is a fence (returns beyond the enclosure are pulled onto it); two inflations: planning (radius + 0.12 m) and hard validity (radius + 0.10 m) |
| `replanner.py` | the planner's goal sequence (maze exit, tunnel, chicane pre/post, pad), A* from the vehicle to the next goal on the live grid, replanned every second, on deviation, or when the live grid blocks the path; plain vias continue into the next goal so corners are flown through; gate crossings are straight legs (a 0.9 m window cannot be represented on the inflated grid) |
| `follower.py` | carrot (pure-pursuit) velocity follower with lag compensation; speed = min(cruise, corner, map clearance, path-corridor scan guard, heading error); never commands a stop-and-turn beside a wall; yaw follows the motion (the camera looks where it flies) |
| `pad.py` | landing-pad ring detector: ground-plane rectification + known-radius matched filter; sub-centimetre on synthetic frames from 2.0 to 4.5 m, blind inside 1.7 m (level camera) |
| `mission.py` | WAIT → CLIMB → FLY → PAD → DESCEND state machine; owns the course<-odom transform, corrected by `coursenav.pf` from the full-band scan |
| `offline_sim.py` | closed loop with a kinematic vehicle (0.45 s velocity lag), drifting VIO, ray-cast depth and synthesized pad frames. `python3 -m kestrel_nav.livenav.offline_sim --png out.png` |
| `../live_mission.py`, `launch/live.launch.py` | the skybeta node: subscribes odometry, `/skybeta/state`, the depth image + camera_info, the left mono image, PX4 attitude and rangefinder; publishes `/skybeta/setpoint/twist` |

## Offline results (2026-11-07, 13 cases, `offline_sim`)

11/13 land in 54-93 s with 3-11 cm landing error (PF localizer on). Start-pose errors of 0.15 m / 5 deg,
unmapped pillars in the open areas, 0.5 and 0.6 m/s cruise and the Gate B route all finish. The two
non-finishes are a pillar that physically blocks a 1.0 m maze corridor (correct HOLD) and a VIO drift
model twice the measured one that hits a maze wall before the PF converges. **Not yet exercised: the
ring detector in the loop** — on the window route the pad is first visible from 1.7 m, inside the
detector's blind zone, so landing accuracy there is the localizer's. **Nothing here has flown.**

## Run on the drone

```bash
# /etc/skybeta/drone.env
COMPETITOR_PKG=kestrel_nav
COMPETITOR_LAUNCH=live.launch.py
# then, on the ground, still:
docker rm -f skybeta-stack
docker exec -it skybeta-stack tmux attach -t skybeta
```

`live.launch.py` starts host-side depth by default (`/kestrel_nav/depth` and
`/kestrel_nav/depth/camera_info`), without changing the organizer camera pipeline.
Check before the first flight:

```bash
ros2 topic hz /kestrel_nav/depth                           # host depth present
ros2 topic echo /kestrel_nav/live_status --once              # plan, pf_sigma, scan_age, counts
```

Open items: measure `landed_z_m`; a frame from the real camera for the ring detector; the pilot lands
(the interface floors altitude at `min_altitude_m`, DESCEND holds there).

## Hardware mission input supervision (2026-11-07)

`live_mission` now requires fresh, advancing odometry, successful
`/visual_slam/status`, usable depth and `/skybeta/state` before it sends readiness
setpoints. It must observe a disarmed vehicle and a continuous healthy interval
before accepting the pilot's ground Offboard handover. The ISAAC ROS status
message package is an explicit dependency; a missing status stream blocks readiness.

Preflight sends **zero twist**, never a pose. This avoids leaving the launch
position in the organizer interface's cached pose fallback. Start with a fresh
flight-interface instance on the ground: this change cannot erase a pose target
left in an already-running interface by another program. There must be one team
setpoint publisher. The organizer interface itself is unchanged.

Both capture age (ROS time) and receipt age (monotonic time) are checked. Duplicate
timestamps do not refresh the watchdog. Non-finite poses/covariance, invalid
quaternions, frame changes and unsuccessful tracking are rejected. Stale/missing
depth clears both retained scans; invalid pixels remain unknown, never invented
obstacle measurements. A sampled-ray coverage check tests the commanded horizontal
direction, including sideways and reverse commands. It is a missing-data guard,
not a proof of full swept-volume clearance or sufficient braking distance.

Once Offboard starts, a sensor fault latches `FAULT`, freezes mission/localizer/grid
progress and requests zero velocity and yaw rate while fresh state confirms
Offboard authority. The guard bypasses the ordinary HOLD state, which can back off
and yaw. Lost/stale authority or an operator-requested interface hold stops team
setpoint publication. Good sensor samples do not resume a faulted mission. The
pilot takes over and lands; restart/re-anchor the mission and interface on the
ground before another run. Zero twist cannot guarantee hover when PX4's estimate
is bad. This is an in-process guard: a killed/frozen process still depends on the
organizer interface/PX4 fallback and pilot, not on an independent watchdog.

`/kestrel_nav/live_status` includes `safety_action`, `safety_reason`,
`safety_fault` and the mission's `fault`. Launch parameters expose the initial
limits: odometry/tracking/depth 0.5 s, interface state 1.0 s, future timestamp
tolerance 0.05 s, readiness interval 1.0 s, valid-depth fraction 0.3, corridor
coverage 0.6 with at least 12 sampled rays. These are conservative initial settings,
**not measured hardware qualification**. Coverage thresholds reuse the simulator
policy's fractions; they may abort in dark or poorly textured areas.

Validation before flight: on a props-off setup, verify capture stamps share the
ROS clock, measure actual data ages/coverage with the full workload, check each
sensor dropout latches a zero-twist response, verify pilot takeover and the
interface hold remain authoritative, and verify node shutdown does not recall a
launch pose. Never restart VIO or inject sensor failures in flight for this check.
Local tests use ROS transport doubles and an offline kinematic simulator; neither
qualifies the deployed ROS graph, aircraft dynamics, or an autonomous landing.
