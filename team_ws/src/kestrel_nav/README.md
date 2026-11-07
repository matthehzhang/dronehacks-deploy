# kestrel_nav: team wide-kestrel course code, packaged for skybeta

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
# from this repo, on the laptop (drone 14, password 2rich)
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
