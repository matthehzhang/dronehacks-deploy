# Map-free search for the blue target

Team decision, 2026-11-07: remove the origin map completely from navigation.
The target is blue and the ground is red. The stereo cameras are monochrome;
the separately available `/oak/rgb/image_raw` camera provides color identity.
This supersedes the draft that retained original-map tunnel waypoints.

## Inputs and separation

Use calibrated forward depth, RGB and mono images with their own CameraInfo/TF,
VIO health, and PX4's local estimate (VIO fused with its IMU). Work in the local
odometry frame established at launch. No course-frame fixture, course OBJ, saved
occupancy, original waypoints, course-wall particle filter or map-derived landing
columns may enter the new mission. Start its map empty on every run. The simulator
still needs physical course geometry; that geometry is only environment truth and
independent test evidence.

## Mission

1. Qualify fresh position, health, calibrated camera/depth and pilot authority.
   Take off and perform a slow stationary yaw scan. Hold XYZ during search. Pause
   yaw on short sensor gaps; preserve existing health, depth and takeover behavior.
2. Integrate timestamp-matched depth into a local occupancy map. Keep unknown,
   observed free and occupied cells distinct. Plan only through observed free
   space with the complete vehicle envelope and margin. No clearing unknown
   space merely because a ray has no valid return.
3. During the scan (at most twice per second on changed maps), if no target is
   confirmed, choose a reachable observation point near an
   unexplored boundary. Score expected new visibility against travel time,
   required yaw and already-visited views. A* finds a route to that observation
   point. If no observation point is safely reachable, hold, complete a bounded
   scan, then end the attempt rather than repeatedly driving into the same wall.
4. Detect blue in the RGB image using HSV (OpenCV H=90..135, S>=80, V>=40).
   Use camera calibration and depth to reject walls/rocks and verify a
   candidate lies on the ground plane. Require consistent detections in several
   frames and from another viewpoint before accepting a goal. A bounding box or
   bearing alone is not a position fix.
5. Once the target position is confirmed, use A* for the shortest currently known
   safe route to an approach point. Continue mapping and replan when observations
   invalidate the route. If target confidence is lost, hold or resume search.
   Approach/centre from visual measurements; do not land on an unconfirmed dark
   patch. Preserve any competition-required checkpoint order as detected features,
   not stored course coordinates.

Exploration cannot promise the globally shortest route through an unseen course.
It can minimize the route through the currently observed map and improve it as new
space becomes visible.

## Camera qualification

The RGB path accepts explicit RGB/BGR channel encodings and honors row padding.
It undistorts using RGB calibration, intersects candidate rays with the measured
floor, and reprojects into the separately calibrated, timestamp-matched depth
camera. A blue wall or raised blue object cannot qualify as the floor target.
Four frames, at least 0.6 s and 0.20 m viewpoint baseline remain necessary;
multiple qualified blue patches are ambiguous. RGB/depth support expires after
0.5 s. A stale/failed RGB stream withdraws its goal while exploration can continue.
The initial hue/saturation/value bounds need lighting qualification on real images.

The retained mono detector is a fallback requiring explicit appearance qualification:
Blue/red contrast in a color photo does not guarantee separability on this mono
sensor. Measure the actual raw-image distributions under exposure and lighting
changes. Shadows, black paint and printed floor features are negative examples.
The initial detector proposes **contrast candidates**, not color identities.
If the target has no distinct mono appearance beyond its color, the requested
sensor stream cannot reliably identify it; a distinctive pattern or another
usable sensor channel would be necessary.

Depth and raw mono may use different intrinsics, frame IDs and capture times.
Project between their calibrated frames; do not copy image pixel coordinates
between streams without registration. Reject stale or missing depth instead of
assigning a guessed target distance.

## Smooth and bounded motion

Keep A* initially. Prune line-of-sight segments and round corners only where the
whole swept vehicle envelope remains in observed free space. Retime the path with
acceleration/braking limits. Slow and align before a narrow gap, traverse at a
bounded speed, then accelerate once the full vehicle clears. A failed replan
invalidates the old path immediately. The radius remains 0.29408 m and the existing
0.12 m planning margin remains; uncertainty may require more reserve.

Use measured command/response delay when setting the stopping corridor:
`distance >= speed * delay + speed^2 / (2 * braking_deceleration) + reserve`.

## Tests before a moving qualification

- Start with the original-map files absent: the mission must initialize and search.
- Move the entrance walls and target independently between fresh runs.
- Check blank floor, dark rocks, shadows, target occlusion and exposure changes.
- Confirm yaw search creates map coverage while XYZ stays held.
- Confirm paths never cross unknown/inflated cells, including smoothed corners.
- Exercise failed replans, equal-count moving obstacles, stale streams, VIO reset,
  authority loss, bounded search timeout and exhausted exploration.
- Record raw images, depth/calibration, fused pose, map revisions, rejected/accepted
  target candidates, frontier choices, paths, controller state and independent grade.

## Implemented module and launch

`team_ws/src/kestrel_nav/kestrel_nav/searchnav/` is a separate implementation. Its
pure `grid.py`, `mission.py`, and `target.py` do not load course assets. `node.py`
adapts the ordinary ROS interfaces to that core. The existing refnav/coursenav
launches remain separate; do not run two competitor command publishers together.

```bash
ros2 launch kestrel_nav search_skybeta.launch.py use_sim_time:=true
```

This starts the normal flight interface, host stereo depth, the health-only VIO
monitor and the new mission. Startup never arms or changes mode. If those three
infrastructure nodes already exist, use `flight_interface:=false host_depth:=false
health:=false`. `allow_takeoff` defaults to false: after the launch column is
approved, set it true for qualification. The pilot still owns arming and the
Position/Offboard transition. No new mission process publishes to `/fmu/in/*`.

The node uses PX4's fused local position, velocity, attitude and estimator flags;
VIO odometry and `/estimator/healthy` supervise the estimator. Camera pose uses
capture-time local pose/attitude plus measured body-to-camera TF. Depth and mono
intrinsics/extrinsics are separate. A valid downward range while landed establishes
the local floor plane. If that range is unavailable, calibrated depth fits a broad,
nearby horizontal plane below the landed body using robust support, area, residual
and a three-degree IMU/world-up normal check. At least three stable frames over
0.5 seconds qualify it. Neither method assumes a nominal body height or reads
course geometry. Missing calibration, TF, measured ground support, advancing
timestamps or fresh authority prevents readiness; there is no course-frame fallback. Qualification
requires three continuous seconds of valid inputs and an observed disarmed start.
After flight begins, an individual rejected stereo frame pauses translation and
yaw without refreshing the last usable frame. Recovery within the existing 0.5
second usable-depth deadline can resume; actual stale depth, other sensor/authority
loss or an unverified/large airborne EKF frame reset latches `yield` and stops streaming. A ground
EKF frame reset clears the old floor/target/map, pose and attitude capture histories
and input qualification. It requires new measurements and the full three-second
qualification in the new frame; pre-reset queued images cannot restore readiness.
The flight interface and PX4 retain their timeout/recovery
policy. A verified small reset now enters bounded recovery: changed uint8 counters
must advance exactly once, XY/Z corrections must be <=0.10 m and heading <=5 degrees.
Only deltas whose counters changed are read. Withdraw all map/target/capture geometry,
hold zero XY/yaw, and require new spatial samples plus three continuous seconds of
normal sensor qualification. The measured horizontal floor scalar and altitude hold
receive the exact verified Z translation. Recovery allows at most three resets and
five seconds per recovery window, never restarts the mission deadline, and cannot
resume a previously yielded mission. Independent VIO health timestamps are not in
the reset PX4 coordinate frame; their original .5 s capture/receipt deadlines remain.

The map starts entirely unknown. It samples a flight-height slab at 0.10 m vertical
spacing across +/-0.4 m and collapses a horizontal cell to free only if every layer
is observed free. A calibrated farther depth return can clear the traversed ray;
an invalid return cannot clear anything. Surface endpoints win over free evidence.
Three distinct free observations are required to clear an occupied voxel. Twenty
seconds without observation returns a voxel to unknown. This is a sampled depth
model and cannot guarantee detection of wires, thin mesh, stereo artifacts or
features below its resolution.

Usable-depth freshness is evaluated at the delivered sensor cadence, independently
of the capped full free-volume projection. Between those projections, a lightweight
occupied-hit update still changes occupancy/revision immediately. The mono sensor's
freshness is likewise separate from its three-Hz detector. Compute throttling must
never manufacture a stale sensor or postpone a newly observed blocking obstacle.

The default fresh local bound is 16 m square at 0.05 m resolution, centred on the
current launch estimate. This is a compute limit, not a course boundary. Horizontal
inflation includes radius 0.294082845 m, margin 0.12 m and a half-cell diagonal
reserve. A measured 0.9144 m gap can fit that discretized envelope, but approximately
3 ft from a photo is not sufficient evidence of actual clearance. Paths use A*,
checked line-of-sight pruning, stops/alignment at retained corners, acceleration,
jerk and braking limits, and a 0.20 m/s ceiling when the path ahead is narrow.
There is no unverified spline rounding.

Every changed voxel invalidates the revision/cache, including moving obstacles
that leave the total occupied count unchanged. On each control tick the remaining
swept route is checked against the latest revision and current pose. An intersection
clears the route immediately and requests zero velocity before replanning. A failed
replan leaves an empty route. Replan to the SAME goal first; choose a different
frontier if that goal becomes unreachable. Failed frontiers have a five-second
cooldown rather than a permanent blacklist. Blocked frontier views are excluded from subsequent
choices; no route leads to a bounded stationary scan, then another reachable view
or an exhausted hold. The maximum search time is 180 seconds, with bounded scans
and rejected/visited observation points. Exhaustion holds for pilot recovery; the
node never sends an arm/mode/land command or lands on an unconfirmed patch.

Frontier scores combine occlusion-checked unknown rays, connected travel distance,
turning time and extra clearance. Visibility stops at known walls or the first
unknown cell; it cannot award information hidden behind a solid wall or clear a
voxel. Candidate scores, chosen viewpoints and route-rejection reasons are published.

## Measured opening height and centering

`opening.py` fits approximately vertical frames in calibrated depth and rectifies
them into metric coordinates. A candidate requires all four observed frame sides,
actual valid returns at least .60 m behind the opening, .30 full-frame valid depth,
.60 opening support and .65 support on each side. Invalid/no-return stereo is never
an opening. Three stable captures over .4 s qualify a unique measured 3-D center,
width and height. Cropped/open-sided passages are left to ordinary exploration.

The controller stops, aligns yaw, then corrects lateral position and height at
<=.15 m/s. It never automatically pushes through: after centering, live A* selects
the route through observed space. Every translation/height command needs a fresh
full swept-volume certificate from `clearance.py`; a 2-D route is insufficient.
The certificate covers radius+margin and vertical half-height+margin on a .05 m
XY/.10 m Z lattice with raster reserve. Up to20s of bounded calibrated depth history
provides finite, supported ray evidence; unknown blocks movement, hits win and
require three later distinct free frames to clear. No launch bubble is fabricated.
The existing current-camera XY stopping-corridor check remains in force as well.

Changing height withdraws the old route and rebuilds the new flight slab from
capture-time calibrated views while holding zero XY/yaw. Replay is queued in
bounded callback slices; older rays cannot clear newer obstacles, and duplicate
captures cannot count twice toward occupied-voxel clearing. Only after the rebuild
does scan/frontier/live A* resume. Loss of authority or critical freshness still
yields; blocked centering and rebuild are time-bounded.

## Current-run visualizer

Open **http://127.0.0.1:8091/live.html** on this computer. It follows `active_run.json`
to each newly started diagnostics run, with measured occupancy versus inflated
traversability, route, commands/PX4 setpoints, mono/RGB/annotated/HSV-mask views,
frontier rejections and centering/clearance diagnostics. Closed recordings are
explicitly labeled; a closed result is not a continuing flight.

## Monochrome target qualification

The detector proposes both brighter and darker floor patches; blue paint can be
brighter than the red floor on the actual mono response. It rejects insufficient
contrast, missing depth, truncated image regions and points inconsistent with the
measured floor plane. Mono rays project onto the floor and then into the separate
depth camera for support. A track requires at least four frames, 0.6 seconds and
0.20 m of camera viewpoint baseline. Pure repeated frames at one optical origin
cannot confirm a goal. Multiple valid patches remain ambiguous.

`target_contrast_qualified` defaults to false. In that mode even a stable dark
shadow or bright painted floor pattern remains a candidate, and exploration
continues. This is deliberate: monochrome pixels cannot prove that a patch is blue.
Set the parameter true only after actual sensor footage establishes a sufficiently
distinct appearance and negative examples under the operating lighting. The module
does not currently have that qualification. A promoted target is an approach goal;
arrival holds for pilot inspection and never triggers automatic landing. Target
confidence expires, and loss clears its route or resumes observation.

## Diagnostics

The node publishes the following ordinary ROS topics for recording and visualization:

| Topic | Type | Meaning |
| --- | --- | --- |
| `/searchnav/occupancy` | `nav_msgs/OccupancyGrid` | Raw collapsed unknown/free/occupied observations |
| `/searchnav/traversability` | `nav_msgs/OccupancyGrid` | Full-envelope inflated observed-safe cells |
| `/searchnav/path` | `nav_msgs/Path` | Current valid route; empty after invalidation |
| `/searchnav/status` | `std_msgs/String` JSON | Readiness, per-source capture/receipt ages and deadlines, qualification duration/failure history, state/reason, map provenance/revision, footprint unknown/occupied counts, frontier choices/rejections, target uncertainty |
| `/searchnav/candidates` | `std_msgs/String` JSON | Per-frame mono candidates and temporal/viewpoint tracks |
| `/searchnav/command` | `std_msgs/String` JSON | Requested ENU velocity/yaw rate or explicit yield, with reason and map revision |
| `/skybeta/setpoint/twist` | `geometry_msgs/TwistStamped` | Actual flight-interface command contract |

Visualize raw occupancy and traversability separately. A surrounding dark ring in
the latter can be unknown space inflated around the vehicle, rather than observed
obstacles. `status.map.start_footprint_unknown_cells`, `start_footprint_occupied_cells`
and `start_traversable` distinguish those cases. A fresh map bound is not a sensed
wall. The status explicitly reports `prior_loaded=false` and
`provenance=calibrated_depth_only`.

For a future actual-drone recording, `host/run.sh bag --search --duration 90`
adds calibrated depth, mono, TF, health, search decisions, both grid views and
emitted commands to the normal stereo/PX4 diagnostic topic set. Recording does
not arm or start a mission. The newly supplied 70.9 s hardware bag lacks raw
images/depth, so it can diagnose mapping gates and commands but cannot qualify
the new contrast detector or replay its depth mapper.

## Validation and current limit

The dedicated `tests/test_searchnav.py` suite passes 34 tests on Windows. Six
ROS adapter regressions in `test_searchnav_ros.py` exercise the actual ground-reset
and sensor/clock/freshness callbacks; all 40 pass in WSL with ROS and `px4_msgs`. They cover
maze/frontier reachability, unknown and narrow-gap vetoes, swept-path smoothing,
same-count obstacle motion invalidating and changing the route, failed replans,
bounded yaw, stale/authority/reset yield, separate camera projection, bright/dark
mono candidates, raised-object and blank/gradient negatives, target ambiguity and
no origin-map access, no PF-confidence gate on depth integration, measured-floor
qualification/rejections and transient-depth watchdog recovery/expiry. ROS import/build and an unarmed integration smoke are separate
checks recorded with the baseline diagnostics.

The first unarmed smoke correctly refused readiness because this PX4 stream had
no valid downward range. Offline replay of its recorded calibrated depth found
horizontal ground in all 20 inspected frames: about 3.1 m² support, 95% inliers,
0.24-degree tilt and 0.007 m residual in the first frame. The third distinct frame
qualified the measured-depth fallback. This fixes that specific missing-input
assumption; it does not create unknown free space or qualify a moving mission.

The next attempt armed on the ground but never entered Offboard: the mission had
incorrectly coupled usable sensor freshness to its four-Hz mapping/three-Hz detector
throttles, causing short eligibility withdrawals. The bag shows the command callback
continued at about 20 Hz (maximum 0.104 s ROS gap); this was not a stalled controller
timer. The watchdogs now validate every usable sensor frame while limiting expensive
compute separately, with the original deadlines unchanged. The test harness must
also require `ready=true`, not merely a retained `phase=arm`, before its simulated
pilot transition. The failed run remains evidence and is not a successful flight.

A later unarmed startup exposed a clock initialization race: PX4 DDS arrived before
the node received its first `/clock`, and a minimum-offset anchor at ROS zero made
all subsequent PX4 samples appear stale. The recorded sources then advanced together
(for example PX4 boot 9.495831 s and ROS 9.499998 s), while the node retained accepted
capture/receipt zero. The adapter now rejects uninitialized clocks and requires five
consistent advancing position-clock pairs spanning 0.25 s before anchoring. A held
initial nonzero tick followed by a jump also cannot establish the anchor. A ground
clock-domain discontinuity clears the old spatial data, packet monotonic counters
and qualification before re-anchoring; an airborne discontinuity yields. The
`px4_clock` status records the first observations, offset, anchor span and reset
reason. Capture/receipt freshness limits remain unchanged.

With clock alignment corrected, the following ground attempt identified one
incorrect new readiness gate: estimator flags publish at a median 1.008333 seconds
and a measured maximum 1.020833 seconds. A newly introduced one-second flags
deadline repeatedly reset qualification even while depth, VIO, position and health
were fresh. The flags deadline now matches the existing reference policy's
two-second EV-yaw freshness contract; other deadlines and EV-yaw conditions are
unchanged. The regression accepts a 1.02-second flags age and rejects over two
seconds. That attempt never entered Offboard, and its bag confirms final disarm
at PX4 boot 34.270827 seconds and `landed=true` through the final sample.

A synthetic 360-degree scan exposes an important forward-camera limitation: the
upper/lower slab immediately around the aircraft can remain unobserved even while
the surrounding room is well mapped. With unknown blocked and the full envelope
preserved, the correct result is `no safe frontier` followed by bounded hold. The
implementation does not seed a fake free bubble or reuse old geometry to force
movement. The actual camera placement/FOV needs recorded coverage qualification;
additional views or a properly justified launch-volume observation may be needed
before this configuration can explore from rest. Forward-only depth also cannot
prove the entire overhead takeoff column; `allow_takeoff` makes that pilot-approved
assumption explicit and clears no horizontal cells.

No successful map-free moving mission, actual blue-color identification or physical
aircraft qualification is claimed. This policy also does not yet recognize semantic
tunnel/window checkpoints or enforce competition checkpoint order from imagery.

## Recorded moving result and remaining work

The corrected simulator launch took off and performed 131.8 degrees of its initial
stationary scan, with live occupancy integrating independently of any PF. The last
complete pre-yield map had 9,587 known cells and 1,364 occupied cells. It then yielded
on a 1.5 cm XY EKF correction. Cleanup reached Auto Land and physical ground contact,
but the recorded run ended before confirmed disarm. It did not reach frontier/path
execution. See the [complete moving-test diagnosis](../../ext/matthehzhang-dronehacks-sim/scenarios/topview_entrance/stack_20261108T053249Z/DIAGNOSIS.md).

That run motivated the bounded small-reset recovery described above. The next
moving run (`stack_20261108T060500Z`) made 13 early planning checks and entered
recovery on a verified 1.6 cm correction. It then exposed an over-broad epoch filter
that discarded fresh, delayed independent VIO health packets; the filter and its
regression are corrected. That run confirmed final disarm. Unknown near-body
coverage still prevented a certified starting route. Neither recorded moving run
validated horizontal exploration, obstacle rerouting or a complete course.

The source used in each moving test matches its saved SHA-256 manifest. The
navigation/adapter suite and two diagnostics-runner tests cover RGB/HSV/depth
geometry and height-centering/3-D clearance. Exact counts and source hashes are
recorded with validation runs.
The unarmed RGB integration run `stack_20261108T062838Z` passed readiness/transport
and confirmed disarmed; raw RGB, annotated proposals, HSV mask and opening/clearance
diagnostics were observed. No end target was in view, and no flight qualification
of RGB approach or height-centering is claimed. Semantic checkpoint order is still
not implemented by the mapless policy.

The separately requested temporary group154 origin-map policy lives in the isolated
`deploy_group154_newcourse` worktree. Its precomputed course map is deliberately not
an input to this mapless implementation. The shareable export is under
`deliverables/new_course_origin_map` at the repository root.

## Explicit pilot launch volume

The user approved an opt-in, pilot-verified volume to address the forward camera's
near-body blind region. It defaults off. `allow_takeoff` still approves only the
vertical launch column and does not enable horizontal clearance by itself.

To use a separately inspected launch volume, set `pilot_verified_launch:=true`,
`launch_volume_bounds_json` to a JSON string of six offsets
`[xmin,ymin,zmin,xmax,ymax,zmax]`, and `launch_volume_duration_s` (default 30 s,
maximum 60 s). The offsets are metres in the fixed local ENU frame, relative to
the qualified ground pose when the pilot starts. Each axis spans at most 3 m;
the origin must lie within the box. Bounds are not inferred from an old map or
automatically claimed to be clear. No deployment-ready bounds are prescribed.

This is a separate temporary operator assertion. Raw `/searchnav/occupancy`
remains measured depth; planner traversability may additionally use UNKNOWN
voxels wholly inside the inspected box. Vehicle inflation and the entire flight
height slab still apply, and centering certifies the whole swept 3-D envelope.
Current calibrated depth must remain fresh. A depth hit inside the assertion,
expiry, clock reversal, authority loss or an EKF frame reset withdraws it and
invalidates any route that depended on it. It never follows the aircraft, grows,
clears an obstacle, renews itself, or survives a frame reset as fresh approval.
The live viewer draws an amber box and publishes provenance, expiry and per-layer
pilot-supported unknown counts separately from observed-free counts.

Regression coverage includes raw-map separation, whole-voxel containment, active
route stop on expiry, new and pre-existing obstacle rejection, cached 3-D queries,
fresh-depth requirements and the actual ROS launch/adapter lifecycle. These are
implementation checks, not evidence of a successful map-free course flight.
