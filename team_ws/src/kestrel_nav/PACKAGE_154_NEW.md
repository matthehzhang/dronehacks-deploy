# Copyable groupbad218 package update

Base: `groupbad218` at `f9528572e1efadf6839ef0de0fda0f445d787997`.
Copy this complete `kestrel_nav` directory into `team_ws/src/kestrel_nav` and
rebuild the overlay. All changes on `package_154_new` are confined to this
package. Arming, disarming, mode switching, recorder lifecycle, PX4 settings,
OBJ/voxel geometry and flight authority handling are unchanged. Refnav motion
recovery now turns to a changed A* route and searches other headings when
the corridor is unobserved; it never translates blindly during search. The course
sequence adds the requested `entrance_approach` stop at ENU(-2.50,1.35), in the
first passage west of the M2 tip, before the tunnel. Both source and installed
course JSON use this same goal; full-margin planning must pass before testing.
These are approximate prior-map coordinates, not surveyed venue truth. The refnav
ROS adapter withdraws invalid planner paths and holds the current XY position
so position feedback cannot continue pursuing an old carrot. The hold position
is captured only when a valid active path is withdrawn; repeated invalid
messages never reset the climb/hover anchor.

## How mapping rejection is handled

1. Poor scan-to-origin matching continues to reject unsafe localization
   corrections. The geometric origin map is a prior, not measured truth.
2. Fresh, calibrated depth independently fills a rolling VIO-frame obstacle
   grid, published as `/coursenav/local_occupancy`. It does not require the
   PF's 6cm/2deg confidence gate. Missing returns remain unknown, the floor
   is excluded, stale inputs do not refresh the map, and detected VIO resets
   or height-band changes withdraw incompatible observations.
3. The old PF-gated landmark grid is diagnostic only: persistent ghost cells
   in that layer closed an open passage in SITL. It is not fused into planning.
   The navigator transforms the fresh depth obstacles with the aircraft's own drift
   transform, adds them to the origin-map geometry (without legacy landmarks) and replans toward the
   original tunnel/window/pad goals. Vehicle inflation stays unchanged.
4. Changed cell contents invalidate blocked remaining routes, including equal-count obstacle
   moves. An unsafe old path is withdrawn before the next bounded A* retry;
   a failed plan never republishes it. A missing path also triggers a retry.
   Passed intermediate waypoints stay consumed for the entire leg. Obstacles
   behind the aircraft do not restart a safe remaining path.
6. A new route pointing elsewhere enters turn/check before forward travel.
   With valid sensors but an unobserved corridor, a stationary yaw scan replaces
   the old corridor-timeout landing. Fresh depth, estimator and pilot authority
   protections remain active. No claim of finding a route through every layout.
5. `/coursenav/path` reports validity, local-map freshness, obstacle counts and
   the actual A* rejection reason. Refnav holds when the route is invalid.

This does not guarantee a course completion. Prior walls are retained: if the
origin map contains a wall across the only real passage, independent depth
obstacle addition alone cannot remove it. Removing prior walls would require
validated free-space evidence for the complete vehicle envelope, not a weak
scan fit or an invalid depth hole. A real sensor/authority fault still stops
motion. This package prevents a prior-fit rejection from also disabling all
local obstacle mapping; it does not bypass motion safety checks.

The new local grid implementation is adapted from the locally tested
`refnav/blocked-recovery` port's pure `coursenav.localgrid`; its controller and
other behavior are not imported. The timed follower and map-free search policy
from `kestral_group154_timed` are not included here.

Simulator-only launch override: `course_file:=/home/anshul/package154_sim_course.json`
sets start ENU(-2.515,1.10), yaw90; physical spawn matches. Default course start
remains ENU(-2.515,.914). `/coursenav/traversability` displays the fused inflated
planning grid. Simulator diagnostics retain A* revisions in `astar_paths.jsonl`.

## Verification

Focused tests exercise mapping without any PF object, stale inputs/frames,
ground filtering, real ROS localizer construction, live obstacles reaching A*,
equal-count changes, and failed/missing-path withdrawal and retry. Source and
installed origin-map assets remain those of groupbad218. Gazebo outcomes are
recorded separately; build/test success is not physical-flight qualification.

### Diagnostic runs during development (2026-11-08)

- `stack_20261108T091859Z`: did not complete. 101 first-leg path revisions,
  737 path-wait ticks, 355 depth-hold ticks, corridor timeout landing.
- `stack_20261108T093257Z`: did not complete. An invalid-path callback reset
  XY to the measured position on every message, including climb, allowing
  drift toward the west net. Two near-contact events; harness commanded
  cleanup Auto Land at the first event. Final disarm was not confirmed by
  bounded cleanup. Simulation processes subsequently exited. Fixed by
  edge-triggered active-route withdrawal; regression test included.
- New test after this fix is recorded separately. These failures are retained;
  passing offline checks alone does not make this package flight-qualified.

`stack_20261108T093834Z` verified the hover-anchor fix and first entrance stop:
1/6 stops reached, zero contacts during the search period. It exposed a second
problem: 28 persistent legacy landmark cells closed the north passage while
fresh local depth had zero occupied cells in that region. The planner now
excludes the old layer entirely. The run was ended for this mapping correction;
it was not a completed course run.
