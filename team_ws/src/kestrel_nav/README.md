# kestrel_nav: team wide-kestrel course code, packaged for skybeta

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

### Adaptive maze trial (SITL)

On the Group154 adaptive branch, rebuild the team overlay and start the SAFE
mission with the adaptive option. Use the normal simulator arm/Offboard flow
once the stack reports healthy and this is the only mission publisher.

```bash
ros2 launch kestrel_nav safe.launch.py adaptive_maze:=true allow_mode_request:=true
```

Use the same course file and checkpoint goals for the original and modified
maze worlds. Startup plans only from the semantic maze exit onward, so it
does not require a route through the old maze. During the first stage A*
searches from the current pose to the exit on the evolving map. It gathers
recent flight-band depth evidence from the VIO-based pose even when matching
against the published map fails. Only course groups marked
`adaptive_obstacle_groups` are eligible for clearing. The enclosure,
keep-outs and gate policy remain in force. The downstream mission restores
the baseline grid after the maze exit. A blocked non-gate goal gets a
stationary look to both sides and a bounded replan budget.

Record mission status, live depth and camera info, VIO, PX4 status and
setpoints. Compare the `live_map` counters (including cleared prior cells),
goal progress, and whether the terminal state was a pad landing or an abort.
The depth map is planar, and this option is experimental. The field bag did
not capture raw depth, so it cannot validate the modified layout by replay.
