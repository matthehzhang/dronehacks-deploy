# Depth-only receding horizon

Implemented on `package_154_new`. This is a new navigation mode; it has not
been flight-qualified. The separate Gazebo test of unchanged `groupbad218`
does not validate this mode.

## Inputs and authority

`course/waypoint_intent.json` contains only named XY targets and a final
landing marker. These targets were exported once from the new-course origin.
There are no walls, buffers, gate widths or origin-derived clearance claims
in that file. The declared launch fixture aligns their coordinates to VIO.

`course/course_depth_waypoints.json` selects `geometry_source: depth_only`.
The Skybeta launch now defaults to this configuration. Legacy navigation configurations and alternate launchers have been deleted.
Depth mode does not load origin geometry in fly_course, NavNode or LocNode,
does not run prior scan matching, and ignores PF drift corrections. Aircraft
pose and the rolling grid stay in the same VIO launch frame. Existing PX4
attitude, height, estimator-health and control-authority checks remain.

## Planning cycle

1. Depth ray updates yield observed free, uncertain/unknown and occupied cells.
   A valid finite return beyond the local 3 m horizon clears the ray inside
   that horizon; it never creates a wall at the range cap. Invalid/no return
   remains unknown. Old evidence decays and the grid rolls with the aircraft.
2. The complete aircraft footprint plus the existing 12 cm margin must fit
   in observed free space. Grid raster uncertainty is also accounted for.
   Unknown cells are distinct from obstacles, but are not certified flight
   space. Origin obstacles never participate.
3. A* reaches the current waypoint when it is reachable in that region.
   Otherwise a reachable observation pose is selected near unobserved space
   INSIDE the room. It must reduce an optimistic cost-to-go to the waypoint
   (measured obstacles and the room boundary block it, unknown is assumed
   passable for ranking only) by at least 0.15 m, falling back to 0.05 m.
   Detours around measured walls therefore count as progress; poses that move
   away from the waypoint are not chosen. Ties break on travel and visits.
4. Existing line-of-sight pruning, corner smoothing and acceleration/braking
   profiles produce the executable horizon. Every smoothed segment is checked
   densely against the current grid. No start snapping or unchecked escape
   connector is permitted; a finite zero-speed prefix is rejected explicitly.
5. The follower faces the actual route before translation. It retains a safe
   route while new observations arrive. Unsafe routes are withdrawn immediately;
   near the horizon end, changed observations can extend the route (planning
   throttled to at most 2 Hz, extensions at most 1 Hz).
6. A nonterminal horizon never completes a mission waypoint. Arrival is
   acknowledged by `(leg,pid)`; repeated packets cannot resurrect that consumed
   horizon. The planner records the visit and chooses the next horizon.
7. Without an executable route the aircraft holds XY and sweeps a bounded
   sector centred on the bearing to the ACTIVE mission waypoint (dwell at 0,
   then +/-30 and +/-60 deg, starting on the side it already faces), re-centred
   each cycle. It never performs a full-circle scan. The horizon endpoint never
   replaces the waypoint for this bearing (`goal_ne` on `/coursenav/path`). A fresh executable route returns it to face/check/follow. Actual
   sensor loss, estimator faults and pilot takeover retain their existing
   behavior. No arm/disarm logic or flight-interface settings were changed.

This is receding-horizon graph planning with a constrained trajectory follower,
not a nonlinear MPC optimizer and not a guarantee of globally shortest paths
through unknown space.

## Room boundary (hard, not depth)

Nets may be invisible to the camera, so `course/room_boundary.json` declares
the enclosure: the inside faces of the four perimeter nets (+/-3.048 m). It is
the OUTER boundary only; no internal obstacle, wall, buffer or passage from the
origin map is used. The aircraft footprint plus margin must stay inside it for
every start, segment and endpoint, regardless of what depth reports (depth may
see free hall space through a net). It overrides the launch disc. Explicit
failures: `start_violates_room_boundary_clearance`,
`waypoint_violates_room_boundary_clearance`. Frontiers outside the room are
ignored. Unknown depth is never turned into an obstacle. The boundary lives in
the VIO-aligned course frame, so VIO drift shifts it with the aircraft estimate.

## Launch blind spot

Default `pilot_verified_launch_radius_m:=0.0` makes no clearance exception.
A forward camera may be unable to observe its entire launch footprint. The
user-authorized opt-in is a **fixed disc at the declared launch point and
cruise altitude**, at most 2 m radius, never a disc that follows the aircraft.
Measured occupied cells always override it. Example, only after verifying
that actual volume:

```
ros2 launch kestrel_nav refnav_skybeta.launch.py pilot_verified_launch_radius_m:=0.8
```

The explicit waypoint list selects the window route.

## Diagnostics

- `/coursenav/local_occupancy`: depth probability grid in `coursenav_nominal`.
- `/coursenav/traversability`: current footprint-feasible grid; unknown=-1,
  feasible=0, room-boundary keep-out=99, measured-obstacle inflation=100.
  Footprint cells blocked only by incomplete coverage stay -1.
- `/coursenav/path`: also `goal_ne`, `room_boundary` (polygon, aircraft
  clearance) and `planner` (goal_state, cost_to_go_m, progress_m, candidates).
- `/coursenav/path`: `map_provenance`, `terminal`, `goal_course`,
  `horizon_course`, path geometry, speeds, tubes, `pid`, and explicit failures
  such as `start_footprint_not_observed_clear`.
- `/refnav/status`: `horizon_reached` and `motion_limit`, including profile,
  live-geometry, braking and ramp ceilings, or a depth/tube/path failure.

## Validation and limits

Validation on 2026-11-08: ROS Jazzy colcon build and launch argument loading
passed. The broad package regression run passed 365 tests with one expected
failure, excluding the long offline `test_square_pad.py` and `test_livenav_sim.py`
suites. After final horizon-handoff, coordinate-frame and CLI edge-case changes,
74 focused ROS-enabled tests passed. No new-policy Gazebo flight was run.

Offline regression tests exercise reveal-and-extend, a new-wall detour,
unobserved and obstacle-buffer starts, opt-in launch clearance, far returns,
frontier acknowledgements, search/reorientation and ROS startup without any
origin-map load. An additional regression uses the exact hardware estimated
start that previously deadlocked, with synthetic depth-certified free space.
It advances; this does not establish that the real space was actually clear.

This remains a 2-D cruise-height map, not a 3-D swept-volume certificate or
adaptive window-height centering. VIO drift/launch alignment, stereo errors,
thin nets, map resolution and genuinely impassable gaps can still prevent
progress. There is no forced minimum speed or claim that all deadlocks are
impossible. Raw depth and local occupancy must be retained for future flight
diagnosis. The new mode still requires Gazebo and real-flight validation.


## Legacy removal

All alternate package launchers, nested historical planner copies, origin OBJ/
voxel maps, PF/landmark matching, complete-course planning, return/diagnostic
missions, and pose/carrot movement implementations have been deleted. The ROS
shell uses the horizon path follower exclusively. There is no new startup
rejection check. Existing sensor/authority/arm/disarm supervision remains.
Historical evidence outside the runtime package remains as an audit record.


## Gazebo cleanup audit (2026-11-08)

The original-mark case without pilot launch clearance did not advance: unknown
footprint coverage repeatedly prevented a route. The recorded grid separates
that from obstacle clearance: late scan samples have about 0.035 m to uncertain
space versus 1.917 m to a measured obstacle. The display formerly showed an
unknown-footprint erosion band in purple; it now keeps that region unknown and
reserves purple for measured-obstacle inflation. This test failed its bounded
navigation objective and then disarmed through test cleanup.

With the explicitly enabled 0.8 m pilot-verified launch disc, the next original-
mark test reached the entrance target. This is a partial entrance test, not
full-course or real-flight qualification. The launch disc remains opt-in,
fixed at launch, and never overrides measured occupied cells.


The final east-5-cm case also reached the entrance with the opt-in disc, with
zero recorded touches and confirmed disarm. Both are partial first-target tests
(`flight_completed=false` by design), not full-course runs. An intermediate
Gazebo attempt never spawned the aircraft model and was stopped unarmed; its
retry is the east-offset result. Final reduced-package suite: 32 passed.


## FASTER-style committed horizons and reduced clearance (2026-11-08)

Gazebo run stack_20261108T123141Z reached waypoint 1, briefly had a leg-2
route, then stalled 4+ minutes in search: observed footprint clearance fell to
0.33-0.37 m (required 0.414) while measured-obstacle clearance stayed 0.50 m.
Mixed/decayed evidence near the aircraft revoked its own footprint coverage,
and brief depth gaps withdrew the route.

Method adopted from FASTER (Tordesillas, Lopez, Everett, How; IROS 2019,
T-RO 2022, arXiv:2001.04420): plan a guide through known-free AND unknown space
(measured obstacles and the room block it), commit only to its prefix inside
certified space with the stop (backup) there, and replan continuously so the
prefix extends ahead of the aircraft. Changes:

- Certified = observed at least once with probability < 70, not a measured
  obstacle, plus the airframe's own body disc. Never-observed stays unknown.
- A depth return now overrides accumulated free belief in that cell, so new
  obstacles still need only ~3 returns.
- Committed prefix of the optimistic guide is preferred; frontier observation
  poses (must advance) are the fallback when the prefix is shorter than 0.4 m.
- Navigator replans every 0.5 s while following a non-terminal route; a failed
  or shorter extension keeps the current route. Depth gaps up to 1.5 s keep
  the committed route instead of withdrawing it.
- Follower tracks receding replans in motion unless their initial heading
  differs by > 45 deg.
- margin_m 0.12 -> 0.05 and raster allowance res*sqrt(2)/2 -> res/2: effective
  centre clearance 0.449 -> 0.369 m. Offline widest-path analysis of the
  topview_entrance 5 cm map (analysis only, not loaded by the policy) gave
  bottlenecks of 0.400 m at the launch point and 0.412 m at the window: both
  were impossible at 0.449 m. Rotor-tip radius 0.294 m is the physical floor.
