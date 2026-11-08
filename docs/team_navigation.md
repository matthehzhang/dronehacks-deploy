# Drone 14 navigation: two candidates under review

Status as of **2026-11-08**. Team code for `skybeta-14`. Everything below runs inside the
organizers' stack (this repo's `fleet-ready` image) as a competitor package in `~/team_ws/src`.

We have two approaches to the same course, each on its own branch:

| | **Exploratory / generalizable** | **Hardcoded route** |
|---|---|---|
| Branch | [`package_mcp_hopecore`](https://github.com/matthehzhang/dronehacks-deploy/tree/package_mcp_hopecore) (= `package_154_new`, `6ee2b2d`) | [`groupbad218`](https://github.com/matthehzhang/dronehacks-deploy/tree/groupbad218) (`7d5e95f`) |
| Author(s) | Anshul Mistry | Robert Schnurr |
| Course knowledge | 7 named target points + the net boundary. **No walls** | The reconstructed course map (OBJ + 5 cm voxels) |
| How it moves | Plans through space the depth camera has **seen** free, toward the next target, replanning every 0.5 s | Plans the full route on the map **before takeoff**, then follows it stop to stop |
| When the course differs from the map | Adapts: it only ever trusted what it saw | Map errors become route errors. A depth check stops it 0.35 m short of what the camera confirms |
| Best result | Gazebo `stack_20261108T135606Z`: all 7 targets, 0 touches, landed on the pad | Gazebo / offline only |
| Real flights | **None yet** | **None yet** |

**Neither has completed the course on the real drone.** Please review both with that in mind. The
[flight history](#what-has-happened-on-the-real-drone) below records what the real drone has done so far.

Contents:
1. [The course and its coordinates](#the-course-and-its-coordinates)
2. [What both share](#what-both-share)
3. [Exploratory: `package_mcp_hopecore`](#exploratory-package_mcp_hopecore)
4. [Hardcoded: `groupbad218`](#hardcoded-groupbad218)
5. [Side by side](#side-by-side)
6. [What has happened on the real drone](#what-has-happened-on-the-real-drone)
7. [How to run, record and replay](#how-to-run-record-and-replay)
8. [What we want reviewed](#what-we-want-reviewed)
9. [Other branches](#other-branches)

---

## The course and its coordinates

**Course frame:** ENU, metres. x is east, y is north, z is up, and the origin is the centre of the
enclosure floor. The enclosure is 20 × 20 ft: the nets' inner faces are at ±3.048 m.

**Map:** `new_course_origin.obj` is the team's **reconstruction** of the new course layout (top view,
photos, partial LiDAR). It is **not surveyed**: the LiDAR registration was rejected, and the entrance
panels (M1, M2, M3) are approximate. The tunnel, chicane, window and pad were carried over from the
previous published course.

**Course order:**
1. Launch mark.
2. North out of the start, then east along the lane above M2.
3. Tunnel entry at (2.31, 2.37).
4. South through the tube to the exit at (2.31, −0.85).
5. Chicane.
6. Window in wall C2a (centre (−0.47, −2.27)).
7. Land on the pad.

**Measured at the venue.** These numbers differ from the map and **only `groupbad218` uses them**:

| | Map | Measured (groupbad218) | hopecore uses |
|---|---|---|---|
| Launch position | (−2.515, 0.914) | (−2.515, **1.117**), 8 in further forward | map value |
| Pad centre | (−2.042, −0.793) | (−2.042, **−1.543**), 0.75 m further south | map value |
| Window opening height | 0.305–1.219 m | **1.00–1.80 m** (approximate) | measured by depth in flight |

> Reviewers: `package_mcp_hopecore`'s `course/waypoint_intent.json` and `course_depth_waypoints.json`
> still hold the **map** launch and pad. If the measured values are right, hopecore will land 0.75 m
> from the real pad and start 0.2 m off its own assumption. This needs to be reconciled.

**Frames on the drone:**
- cuVSLAM's odometry origin is wherever the drone was **when the stack started**.
- Both packages place the course on that origin with the declared launch pose (yaw 90°: the drone
  faces course north).
- So the drone must be on the launch position, facing north, **before** the stack starts, and must not
  be touched until takeoff.
- PX4's heading then reads 90° NED (`EKF2_EV_CTRL` fuses vision yaw; no magnetometer).

---

## What both share

- **The stack** (this repo):
  - OAK-D-S2 → cuVSLAM → `px4_odom_bridge` → PX4 EKF2, on PX4 v1.16.2 with the patched TELEM2 fix
    (see CLAUDE.md, bug #15).
  - `skybeta_flight_interface` turns team twist setpoints into Offboard.
  - The pilot arms and switches to Position; the interface enters Offboard. Team code never arms
    and never changes mode.
- **Package:** `team_ws/src/kestrel_nav`, launched with `refnav_skybeta.launch.py`.
  - **`refnav`** follows the path and outputs twist commands.
  - **`coursenav`** holds the navigator and the localizer node.
  - **`host_depth`** computes stereo depth on the Jetson: OpenCV SGBM at half resolution (512×320),
    covering 0.39 m out to a 12 m cap. It's about ±0.1 m accurate at 3 m and about ±0.4 m at 6 m.
- **Vehicle:**
  - 7-inch props.
  - Rotor-tip radius 0.294 m, which is the yaw-independent reach.
  - Lateral half-width 0.249 m.
  - Cruise height 0.8 m above the floor.
- **Recorder:** `team_ws/tools/group154_record.sh` (+ `.py`) writes one rosbag per run to
  `~/team_ws/flight_logs/<stamp>_group154_XXXX/`. It stops by itself 5 s after the drone is disarmed
  and on the ground.

---

## Exploratory: `package_mcp_hopecore`

Design notes: `team_ws/src/kestrel_nav/DEPTH_HORIZON.md` on that branch.

### Idea
Use no map geometry at all. The only course knowledge is the 7 targets in
`course/waypoint_intent.json` and the net boundary in `course/room_boundary.json`:
1. entrance approach (−2.5, 1.35)
2. tunnel entry
3. tunnel exit
4. window pre
5. window
6. window post
7. pad

The drone reaches each target by planning only through space its depth camera has seen free.

### Each cycle (navigator ~2 Hz)
1. **Depth grid.** A 2-D grid, rolling with the drone, at flight height ± 0.25 m. Each cell is seen
   free, seen occupied or unknown.
   - About 3 returns mark an obstacle.
   - A valid return beyond 3 m clears the ray up to 3 m.
   - No return leaves the cell unknown.
2. **Optimistic guide.** Plan a route to the target that treats unknown cells as passable. Only
   measured obstacles and the net boundary block it.
3. **Commit only to the safe part** (the FASTER method, Tordesillas et al. 2019). The drone commits to
   the first stretch of that route that runs through *certified* space, and the committed stretch
   ends with a stop. Certified means seen free, not occupied, and with clearance ≥ 0.369 m:
   lateral half-width 0.249 + margin 0.12, never below rotor-tip radius + 0.05.
4. **Fallback.** If that stretch is shorter than 0.4 m, fly to an observation pose next to unseen
   space. The pose must cut the estimated remaining distance to the target.
5. **Replan every 0.5 s while moving**, so the committed stretch keeps extending.
   - A route is withdrawn only for a real hazard: a measured obstacle or the net boundary.
   - Depth gaps of up to 1.5 s keep the current route.
6. **No route at all:** hold position and sweep ±60° around the bearing to the target, then retry.
7. **Height:**
   - VIO height, slowly corrected by the rangefinder (time constant 10 s), so a window sill no longer
     yanks it.
   - The drone centres vertically on an opening only when depth confirms one: surfaces above and below,
     a gap it fits through, and at least 15 returns seen through the gap.

### What the follower (refnav) no longer does
Removed in `883a68e`, about 540 lines:
- the forward-window depth-coverage check;
- live depth obstacle stops and speed caps;
- tube holds and the first-turn gate;
- the corridor-wait landing and the depth-silence landing;
- the safe-landing retreat.

**Motion is now limited only by the planner's grid** (measured walls and the net boundary).

Kept: yielding to the pilot or PX4, landing on estimator or position failure, and the landing at
route end.

### Evidence
Gazebo, 2026-11-08, from the commit messages:
- **`123141Z`:** stalled 4 min in search. The drone's own footprint coverage flickered off. Led to
  committed routes.
- **`124724Z`:** touched the window top. The rangefinder height dropped over the sill, and the map
  wiped on the height change. Led to the VIO-based height.
- **`134644Z`:** refnav crashed on a leftover `geometry_speed` reference. Fixed in `ac65620`, with a
  closed-loop test.
- **`135908Z`:** a wall cut off by the image edge looked like a sill. The height ratcheted to 1.15 m
  and it stalled at window pre. Fixed so it only acts on confirmed openings.
- **`135606Z`:** **all 7 waypoints, 0 touches, pad landing, PX4 auto-disarm.**

Offline: 53 tests pass; 2 more need ROS. I also checked the window with the branch's own
`DepthPlanner` and perfect depth: the 0.914 m window plans through in 30/30 grid alignments. It still
passes with jamb edges smeared 2 cm by stereo, and fails at 4 cm (2/30).

### Known risks
1. **Not flown on hardware.** Gazebo's tunnel is lit and its depth is clean. On the real drone:
   - a dark tube may return no depth, leaving it unknown, so there's no route;
   - the start estimate has been off by 19 cm (see the history below);
   - this airframe yaws about 11° left at liftoff.
2. **Spare clearance while turning.** The 0.369 m clearance assumes the drone faces along the path. It
   follows replans while up to 45° off heading, and at 45° the spare drops from 0.120 to **0.075 m**.
   Real sideways tracking error has measured 0.04–0.16 m.
3. **No live obstacle stop.** It reacts only to what is in the grid: about 3 frames plus a 0.5 s
   replan, roughly 0.5 m of travel at cruise speed. Fine for fixed walls, not for a person stepping in.
4. **It can hover forever.** The search sweep has no cap, and the depth-silence landing is gone. If
   depth fails while VIO is healthy, it holds until the pilot takes over or the battery runs low.
5. **It uses the map's launch and pad positions**, not the venue measurements (see the table above).

### Key files (`team_ws/src/kestrel_nav/`)
| File | What |
|---|---|
| `coursenav/depth_planner.py` | certified grid, the FASTER guide and committed stretch, observation poses |
| `coursenav/localgrid.py`, `coursenav/loc_node.py` | rolling depth grid; VIO height corrected by the rangefinder |
| `coursenav/nav_node.py` | replanning cycle, publishes `/coursenav/path` and `/coursenav/traversability` |
| `coursenav/room.py`, `course/room_boundary.json` | net boundary as a hard limit |
| `coursenav/airframe.py` | `planning_clearance()` (0.369 m) |
| `refnav/core.py`, `refnav/vertical.py` | follower, search sweep, vertical centring on openings |
| `course/waypoint_intent.json`, `course/course_depth_waypoints.json` | the 7 targets, vehicle, launch |
| `tests/test_depth_horizon.py`, `tests/test_room_search.py` | offline tests, including a closed loop through `Mission.step()` |

---

## Hardcoded: `groupbad218`

Notes: `team_ws/src/kestrel_nav/README.md` on that branch (the newest sections are at the end).

### Idea
Use the course map. Before takeoff, `fly_course.py` plans the route on it with the coursenav planner
(A*, at least 0.414 m clearance: rotor reach + 0.12 margin). The route has fixed stops:
1. tunnel entry (lined up on the tube's axis), straight through the tube
2. a **new** `tunnel_clearance` stop at (2.05, −1.85) to get round the rotated wall tip
3. the chicane, choosing window or gate B
4. the pad, moved to the venue position

refnav then follows the route, with turns at the corners and gate alignment at the window.

### What is changed relative to the group154 base (`f952857`)
- **Venue measurements:** launch 8 in further forward; pad moved 0.75 m south.
- **Window height:**
  1. Cruise at 0.8 m to `window_pre`.
  2. Hold position and climb to **1.40 m** at 0.25 m/s, until height is within 5 cm for 1 s.
  3. The camera may trim height by ±10 cm.
  4. If alignment fails inside 4 s, it crosses at the held height anyway and logs
     `WINDOW ALIGNMENT FALLBACK`.
  5. It keeps the crossing height until the landing descent.
- **Depth no longer gates motion:** `min_corridor_coverage = 0`, `min_valid_frac = 0`,
  `corridor_wait_land_s = −1`. Blank or sparse depth frames still allow motion.
- **Measured obstacles** trigger hold, back-off and recheck at a fixed **0.35 m** ahead
  (`stop_m 0.35`, `stop_per_mps 0`), after 3 confirming frames. That's inside the OAK's ~0.4 m
  minimum range.
- **Camera silence** still stops motion and lands after 5 s.
- **Tracking recovery:** outside the tracking tube it creeps forward at 0.10 m/s
  (`path_tube_recovery_mps`) instead of holding.
- **Landing:**
  - The landing timeout is off; it descends at 0.30 m/s until PX4 reports landed.
  - It then sends a normal disarm command (retried each second) and waits for PX4 to confirm.
  - This disarm is the only direct vehicle command the package sends.
  - Log banners: `AUTO LAND`, `CONTROL YIELDED`, `DISARM COMMAND SENT`.
- **Tools:**
  - `replay/replay_bag.py`: puts a flight's estimates on the course map in one offline HTML file,
    using plain Python.
  - `tools/preview_path.py` and `tools/preview_window_height.py`: preview the route and the height
    targets.

### Known risks
1. **Map error becomes route error.** The entrance panels are approximate. The tightest map
   clearances are the start (0.40 m, with Barrier 1 0.39 m south of the map's mark) and the window
   (0.41–0.42 m).
2. **The start estimate decides everything.** On 2026-11-08 the estimate was 19 cm behind the mark.
   That made the planner turn 22° left to recover, then deadlock on a zero-speed start path (history
   below). Matthew's fix for this is paused. The 8 in forward launch may move the start out of the
   barrier's margin; check with `tools/preview_path.py`.
3. **Ignoring depth coverage means blank or dark stretches don't stop it.** That's intended, but a
   wall that returns no depth isn't detected. Only measured obstacles stop it, at 0.35 m.
4. **The window crossing at 1.40 m relies on the measured 1.00–1.80 m opening.** The map's window
   (0.31–1.22 m) is not updated, so the map can't check the raised crossing.
5. The stop at (2.05, −1.85) is placed by eye. None of these changes has flown.

### Key files (`team_ws/src/kestrel_nav/`)
| File | What |
|---|---|
| `course/course_v3.json` | launch, sequence (tunnel → `tunnel_clearance` → chicane → pad + offset), refnav profile |
| `fly_course.py`, `coursenav/planner.py`, `coursenav/semantics.py` | route planning on the map |
| `coursenav/nav_node.py` | navigator, pre-planned legs |
| `refnav/core.py` | follower, window height and alignment fallback, 0.35 m obstacle stop, tube recovery |
| `refnav/skybeta.py` | twist output, landing descent, explicit disarm after touchdown |
| `tools/preview_path.py`, `tools/preview_window_height.py` | look at the route before flying |
| `replay/` | bag → course-map replay viewer |
| `tests/test_new_course.py`, `tests/test_refnav_depth_profile.py`, `tests/test_window_height.py` | offline tests |

---

## Side by side

| Question | hopecore (exploratory) | groupbad218 (hardcoded) |
|---|---|---|
| Uses map walls? | No | Yes |
| Uses venue launch/pad measurements? | **No** (map values) | Yes |
| What can stop it | No certified route → search sweep (no cap) | Measured obstacle within 0.35 m → hold, back-off, recheck; camera silence → land after 5 s |
| Handles a moved or extra wall | Yes: plans around what it sees | Only stops in front of it |
| Window | Depth-certified crossing at 0.369 m clearance; height centred on the seen opening | Map route at 0.414 m clearance; climbs to 1.40 m, aligns or falls back |
| Tunnel | Needs depth inside the tube to certify it | Lined up on the tube's axis from the map; ignores missing depth |
| Clearance kept from walls | 0.369 m (facing along the path) | 0.414 m (any heading) |
| Live obstacle reaction | Grid only (~0.5 m of travel) | Live depth, 3 frames, stops at 0.35 m |
| Lands itself on | estimator/position failure, route end | the same, plus camera silence and a persistent block |
| Hardware flights | none | none (its base `f952857` flew once; see below) |
| Biggest unknown | Real depth in the tube and at the window; start offset | Map accuracy at the entrance; the raised window crossing |

---

## What has happened on the real drone

All on `skybeta-14`. The bags are in `cram304:docs/evidence/drone_bags_20261108/` (Git LFS), and
Matthew's analysis is in `package_154_new:docs/PACKAGE154_ZERO_SPEED_HANDOFF.md`.

| When (local) | Code | What happened | Cause |
|---|---|---|---|
| 11-07 20:17 | `kestral_group154`, old map | Flew ~0.7 m north, then 45 s of block → back off → "clear" → block, until the pilot took over | The maze had changed on the day. The localizer never matched a scan, and refnav's recovery always "recovered" after backing off 0.2 m |
| 11-07 23:46 | `f952857` | No flight: armed in Manual for 1 s, disarmed. The drone was carried around after the stack started | (it would have been 14° off heading) |
| 11-07 23:50 | `f952857` + new map | Forward, turned, stopped | Bag cut off on the drone at takeoff (no metadata), so the cause is unrecorded |
| 11-08 ~10:39 | `package_154_new` `92b0c1f` | 22° left turn, then 15 s in "go" at zero speed. Pilot switched to Position with low throttle, so it descended | Start estimate 11 cm right of and 19 cm behind the mark → path back to the first target; start inside the clearance margin → zero-speed start path |

Airframe and stack facts that matter here:
- **Yaw kick at liftoff:** about 11° left at liftoff (20:17 bag) while refnav commanded right. It
  recovers within about 1 s. Check motor direction, props and motor order (the motors were rewired
  on 11-07).
- **Level calibration:** after the rewire, the FC's level calibration was off by 10° and the drone
  drifted backward. Fixed with QGC accel + level calibration.
- **Turn signs are correct.** PX4's heading matches VIO at takeoff (90.1° NED), and the yaw signs agree
  through refnav, the twist and the flight interface.
- **Depth range cap:** stereo depth returns nothing past 12 m or on blank surfaces. The old refnav
  rule (45 % forward coverage) stalled in open lanes. hopecore removed that rule; groupbad218 set it
  to 0.
- **The recorder can lose a flight.** It buffers up to 16 MB and is cut off if killed. Copy a bag only
  after `finished.json` appears.

---

## How to run, record and replay

**On the drone** (USB-C: `ssh dronehacks@192.168.55.1`; never log into another team's drone):
```bash
# on a laptop: put one branch's package on the drone (replace BRANCH)
git fetch origin && git worktree add /tmp/nav origin/BRANCH
rsync -a --delete /tmp/nav/team_ws/src/kestrel_nav/ dronehacks@192.168.55.1:team_ws/src/kestrel_nav/
```
```bash
# on the drone: /etc/skybeta/drone.env must say
#   COMPETITOR_PKG=kestrel_nav
#   COMPETITOR_LAUNCH=refnav_skybeta.launch.py
docker rm -f skybeta-stack            # the stack restarts: do this with the drone ON THE MARK, facing north
```
Only one `kestrel_nav` can be active. Back the current one up first (`cp -a ~/team_ws/src/kestrel_nav ~/kestrel_nav.bak`).
- hopecore's launch defaults to `course/course_depth_waypoints.json`. The opt-in
  `pilot_verified_launch_radius_m` is 0 by default and is no longer needed since `059ff2e`.
- groupbad218's launch takes `chicane:=auto|window|gate_b` and an optional
  `start_x/start_y/start_yaw_deg` override.
- Matthew's helper on the drone, `~/team_ws/tools/fly.sh 154new`, rebuilds and restarts only the
  competitor pane, starts a fresh recording and shows the logs.

**Before every attempt:**
1. Drone on the launch position, square to course north. Then (re)start the stack. Don't touch it again.
2. Check `/fmu/out/vehicle_local_position` heading is ~90° and `xy_valid` is true.
3. Pilot arms → Position. Offboard engages by itself. Raising the throttle or moving a stick takes over.

**Record:** `~/team_ws/tools/group154_record.sh start` (it stops itself after landing). Add these
topics, which the last bags lacked: `/kestrel_nav/depth`, `/coursenav/local_occupancy`,
`/coursenav/traversability`, `/coursenav/path`.

**Replay a bag on the map** (`groupbad218`):
```bash
python3 replay/replay_bag.py ~/team_ws/flight_logs/<stamp>_group154_XXXX --open
```
**refnav's own trace** for every run is at
`~/team_ws/src/kestrel_nav/logs/run_*/flight_trace.jsonl`, with `navigator.log` beside it. It
survives even when the bag doesn't.

---

## What we want reviewed

**hopecore**
1. Is 0.369 m clearance (lateral half-width + 0.12) safe when the follower accepts replans up to 45°
   off heading? Should planning use the full 0.294 m rotor-tip radius until the drone is facing along
   the path?
2. Should there be a cap on search sweeps and a depth-loss landing? Both were removed in `883a68e`.
3. Tunnel: what happens if the tube interior returns no depth? Is there a case for a declared passage
   leg, like the old `tunnel_mode`?
4. The committed stretch ends in a stop that assumes static walls. Is the ~0.5 m reaction distance
   acceptable without a live stop?
5. The launch and pad still use map values. Which are right?

**groupbad218**
1. Is the 0.35 m obstacle stop (inside the OAK's minimum range) meaningful, given that depth coverage
   no longer gates motion?
2. Window: climb to 1.40 m with an alignment fallback. Is crossing without alignment acceptable, and
   is the 1.00–1.80 m measurement reliable?
3. Start: with the 8 in forward launch, does the route avoid the zero-speed start path found on 11-08?
4. `tunnel_clearance` (2.05, −1.85) was placed by eye. Please check it against the real rotated wall.
5. Is an explicit disarm after `landed=true` safe if the land detector fires early on a slope or a
   boulder?

**Both**
- What is the minimum hardware test before a full attempt? Suggested:
  1. props-off hand-carry with the stack running (check the estimates on `replay`);
  2. a tethered or low hop to the first target;
  3. one leg at a time.

---

## Other branches

| Branch | What |
|---|---|
| `package_154_new` | same commit as `package_mcp_hopecore`, plus Matthew's handoff on the 11-08 stall (`docs/PACKAGE154_ZERO_SPEED_HANDOFF.md`) |
| `cram304`, `kestral_blocked_recovery` | Matthew's port of AsiakN's workspace `refnav/blocked-recovery`: local grid, prior-map switch, recovery creep. Also the drone bag archive |
| `kestral_group154_timed` | group154 plus timed trajectory and search policies (Anshul) |
| `kestral_group154`, `kestral_nav934am` | earlier group154/group119 ports to skybeta |
| `course-waypoints` (local), `cage-hop-test` | cage tests (`hop_test`, `tune_test`) and a map-only waypoint runner (`course.launch.py`) |
| `kestral_group154-livemap` | live-map/replan experiment, superseded by the two candidates above |

Upstream team workspace: `AMMistry18/workspace` (`group154`, `group154-ideas`, `refnav/blocked-recovery`).
