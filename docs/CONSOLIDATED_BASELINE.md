# Consolidated navigation baseline

Canonical development branch: `kestral_group154_timed` in
`matthehzhang/dronehacks-deploy`. Future changes from this session belong here.

This consolidation merges the deployed `groupbad218` commit
`f9528572e1efadf6839ef0de0fda0f445d787997` into the existing local branch at
`5b2dd5a3756c65c8211e4bb1b4592989080a4915`. Both histories are preserved.

## What is included

- The group154-ideas timed trajectory follower and lead settings, derived from
  workspace `95a1c5ab4d6e54a423fcd5121f63897acfbe3b47`.
- The deployed new-course origin OBJ/MTL, voxel map and native course settings;
  the first semantic destination is the tunnel. Root and installed map assets
  must resolve to the same geometry. Entrance geometry is an approximate
  reconstruction, not a surveyed physical course.
- The deployed window-height alignment implementation and tests, including its
  bounded window correction setting.
- Hardware startup/recording tools under `team_ws/tools/group154*`.
- Simulation-time depth scheduling, serialization and preplanning readiness
  fixes, with hardware scheduling behavior retained.
- Generalized map-free search under `kestrel_nav/searchnav`, selected separately
  by `search_skybeta.launch.py`. It is experimental, not the default refnav flight
  policy, and has not completed a course in Gazebo.
- Recorded hardware diagnoses and existing validation evidence. Evidence refers
  to its recorded source, not automatically to this merged revision.

The separate `refnav/blocked-recovery` port tested in `deploy_refnav_blocked` is
not silently overlaid on the timed follower. Its distinct algorithm and original
upstream history remain separate; combining mutually different controllers needs
an explicit implementation decision.

## Merge decisions and current limitations

Health expiry follows ROS time in simulation and monotonic time on hardware,
preserving the deployed hardware behavior and the unchanged 0.5-second deadline.
The existing generalized policy and refnav stationary-search/live-replan work
are retained. No occupancy fit gate, yaw gate, safety margin, authority guard or
PX4 parameter was relaxed during consolidation.

The real recording `20261107-235044_group154_OvyGUo` ends immediately after climb
is commanded. Its map hash matches the new export; all 285 localization scans
were skipped and all 285 live-map scans gated. The later turn/landing was not
recorded. This consolidation does not fix or explain that missing failure interval.

The prior map-assisted simulator flights reached the pad but breached the 12cm
physical clearance reserve near launch. The generalized policy remains under
development. Build/unit-test success is not hardware or merged-flight qualification.

## Launch selection

`ros2 launch kestrel_nav refnav_skybeta.launch.py` starts the map-assisted policy.
`ros2 launch kestrel_nav search_skybeta.launch.py` starts experimental map-free
search, with takeoff approval and pilot launch volume disabled by default.
Run only one mission/interface at a time. The pilot owns arming and takeover.

Validation results for this merge are recorded in `consolidation_validation.txt`.
