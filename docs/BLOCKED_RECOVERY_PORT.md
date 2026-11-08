# Blocked-recovery ROS port

Workspace `refnav/blocked-recovery` at `ed2271bd49c34bf5fd77cbd5d5124e28be778905`
is vendored into `team_ws/src/kestrel_nav`. Deploy branch: `kestral_blocked_recovery`.
The branch-specific port skill and exact upstream adapter patch are in `docs/port-to-skybeta`.

## Behavior and map

The policy retains upstream blocked recovery: confirm every clearance frame, hold/back off/recheck,
follow a replanned path after turning and checking, and land at the per-leg block limit.
The planner, navigator, local occupancy and prior monitor are the upstream implementations.
Only Sky Beta transport, startup, landing and simulation-clock adapters are applied.
The legacy start-relocation and navigator vertex-removal port patches were removed because upstream
now implements bounded escape and the old patches changed its behavior.

`coursenav.course.DEFAULT_COURSE` selects `course/course_new_origin.json` for the mission,
navigator, localization scan field, prior monitor and live-map initialization. The new OBJ/materials
match the download exactly; the voxel occupied cells and labels also match. See `map_identity.json`.
Upstream map metadata is retained. Historical v3 assets remain for explicit legacy regression tests;
they are not selected by the recovery launch. No thresholds were relaxed to accommodate the map.

## Run

```bash
colcon build --symlink-install --packages-select kestrel_nav
source install/setup.bash
ros2 launch kestrel_nav refnav_skybeta.launch.py
```

Use `use_sim_time:=true` only in simulation. ROS normal/copied installs are also supported.
The pilot supplies physical arming and mode authority; this package does not arm the aircraft.

## USB-C installation

Transferred via pinned `dronehacks` SSH identity at `192.168.55.1` to skybeta-14:
`~/team_ws/policies/blocked-recovery/src/kestrel_nav`.
Built with the existing skybeta-stack container into that isolated workspace. Its installed overlay is
`/root/team_ws/policies/blocked-recovery/install/setup.bash` inside the container.
The active policy, organizer stack, PX4 parameters and autostart selection were not switched.
Native ARM64/Humble colcon build and launch argument resolution passed.

## Validation

Evidence: `docs/evidence/blocked_recovery_20261108/`.
- Deployment regression suite: 318 passed, 2 expected failures.
- Recovery, new-map and simulation-clock focused suite: 21 passed.
- Copied-install wheel test: passed, including installed new-map dry-run planning.
- Native Jetson ROS build and `ros2 launch ... --show-args`: passed without starting a mission.
- Upstream broad suite: 713 passed, 16 fixture failures. All 16 bypassed ROS constructor initialization and lacked new adapter fields; the corrected direct-mode fixture rerun passed all 22 affected-module tests.
- Double-Gazebo result: recorded separately when complete.

Initial failures are retained: the historical adapter test assumed v3 route length (now explicitly
pins v3); old navigator vertex removal violated new-map creep speed (removed); upstream ROS mocks
bypassed initialization and lacked the new adapter fields (harness-only patch, direct mode unchanged).

The reconstructed map is not a surveyed venue map. Its historical grader lines do not certify the
new entrance route; report traversal and scorecard separately. Simulation does not qualify hardware.

## First real ROS startup failure

Both sim localizers refused startup because upstream declared `min_fit_frac` twice.
The prior monitor now reuses the PF parameter value, preserving threshold and overrides.
No simulated pilot armed during the failed attempt. Native build/import/launch-description
checks did not expose this constructor-only error; the live ROS startup did.
