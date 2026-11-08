---
name: port-to-skybeta
description: Port a FlyX-lab workspace branch (refnav + coursenav, e.g. group119, matthew930am) unchanged onto the skybeta drone stack in the deploy repo, then validate it in the local Gazebo sim. Use when Matthew asks to "port <branch> to the drone / hardware / deploy / gazebo", to bring the latest lab code to skybeta, or to refresh a deploy branch from a newer workspace branch.
---

# Port a workspace branch to skybeta (drone + Gazebo)

**The rule: port the lab code UNCHANGED.** Do not re-implement its behaviour in another code base (the
first attempt, kestrel livenav "safe policy", re-derived 930am's rules by hand and every Gazebo failure
was a place the copy differed). The only edits are the skybeta adaptations below, all kept in
`assets/skybeta.patch` + `assets/new/`, each forced by a hardware fact and found by a Gazebo run.

Repos: `~/projects/hackathon/dronehacks/workspace` (FlyX lab code, branches like `origin/group119`) ->
`~/projects/hackathon/dronehacks/deploy` (skybeta, package `team_ws/src/kestrel_nav`). First port:
deploy `kestral_nav934am` from workspace `group119 @ 749bca0` (Gazebo: full course, checkpoints 1-4 in
143 s, 0 touches, 2026-11-07).

## Procedure

1. Run the port (vendors, patches, wires setup.py, dry-runs the plan from the launch mark, adapter tests):
   ```bash
   bash ~/projects/hackathon/dronehacks/.claude/skills/port-to-skybeta/scripts/port.sh origin/<ws-branch> <deploy-branch> [base]
   ```
   Exit 2 = an adapted upstream file changed: re-apply the failed hunk by hand from the table below.
2. Review `git -C deploy diff --stat`: only vendored dirs, the four adapted files, the new files, setup.py.
3. Full suite on the LAPTOP, at most 4 heavy processes (`nice`); never on `compute` while a sim runs.
4. Commit (message: workspace commit vendored + any new adaptation and its evidence), push ONLY that branch.
5. Gazebo: ask the "Rebuild sim environment in Gazebo" session to run it, or Matthew pastes:
   `cd ~/projects/hackathon/dronehacks/sim && scripts/policy.sh <deploy-branch> kestrel_nav refnav_skybeta.launch.py use_sim_time:=true`
   (that session is reachable by SendMessage only when it appears in ListAgents; else give Matthew the text).
   Ask for: grader scorecard, refnav log transitions, last /refnav/status, truth at the end.
6. Each failure: decide sim-side vs port-side with evidence (logs, /coursenav/status, a dumped frame);
   fix port-side ones as a new adaptation (never loosen a group rule), then `scripts/refresh_patch.sh`
   and add a row below. Repeat until the run completes.

## Adaptations (why each exists)

| Where | Change | Evidence |
|---|---|---|
| refnav/node.py | `px4_output` param: `direct` (lab, default) or `skybeta`: no /fmu/in publishers, organizer topic resolver, every tick to `refnav/skybeta.py` | Drone rules: competitors never publish /fmu/in, arm, or change mode |
| refnav/skybeta.py | setpoint + vel FF -> twist `v = v_ff + kp(sp - pos)`, yaw rate likewise, NED->ENU, clamped; offboard/arm dropped (pilot + interface); land -> descent | Interface takes ENU world twist (expires 1 s, 1.5 m/s cap); pose setpoints are unlimited in speed |
| refnav/skybeta.py | landing descent 0.30 m/s | PX4 1.16 land detector: in_descend needs vz >= 1.1 x LNDMC_Z_VEL_MAX (0.25) = 0.275; at 0.15 it sat armed |
| refnav/skybeta.py | pad landing: centre rotor footprint (0.035 m behind origin) on the last target, face nearest pad side (start heading + k x 90), 0.03 m / 3 deg / 1 s, then descend; aborts land in place | Gazebo: 9.7 cm off + yaw 131 -> prop 4.2 cm outside a 0.64 m pad |
| refnav/node.py | skybeta: disarm during land phase confirms touchdown | No AUTO_LAND on skybeta; refnav waited for it |
| refnav/node.py | skybeta: heading OK also when estimator_status_flags.cs_ev_yaw | FRD vision (pose_frame 2, EKF2_EV_CTRL 9): PX4 1.16 leaves yaw_align false, heading_good_for_control never sets (deploy CLAUDE.md, drone 16) |
| refnav/node.py | skybeta: cs_ev_yaw accepted if true within 2 s; in-flight EKF reset > 10 deg or > 0.30 m -> refnav `request_land` (smaller resets continue) | REALISM=1: a 330 ms VIO freeze cleared cs_ev_yaw -> "position or heading invalid" landing. The resets after it were tiny (1.3 cm, < 0.5 deg; an early "206 deg" report compared headings across a turn); the large-reset guard exists because refnav has no reset handling at all |
| refnav/skybeta_health.py | /estimator/healthy from cuVSLAM odometry + status with the lab's VioHealth rule; publishes nothing to PX4 | Lab's ev_bridge published it; a 2nd vision publisher would corrupt EKF2 |
| coursenav/loc_node.py | `height_source local_position`: dist_bottom (0.02-3 m, valid flag ignored) else landed_z - z | skybeta has no /fmu/out/distance_sensor; every scan was skipped |
| fly_course.py | plan from nearest free cell within 0.3 m if the start is inside the margin | Venue mark is 0.40 m from the barrier < 0.414 m; lab only flew from its spawn |
| fly_course.py | `--start-from-mark` (course file mark + guideline heading), `--no-return` | Rule 7.4 places the drone on the mark; auto-return re-arms (impossible/unsafe on skybeta) |
| fly_course.py | env KESTREL_USE_SIM_TIME / KESTREL_DECLARED_HEADING_DEG to the navigator child | navigator declared_heading 0 on skybeta (organizer bridge permutes ENU), 90 was the lab bridge: 3.4 m path error |
| coursenav/navigator.py | no vertex at the few-cm free-start hop; drop interior vertices within 0.30 m of the start | refnav faces the path's first segment: from the mark it swung 65-115 deg with cm of position; now 83-99 |
| refnav/core.py | `path_turn_hold` (default False; launch true): a path leg's turn with no path yet keeps the current heading; go waits for the path (a blocking wait in the turn deadlocked, 7733cd0) | Gazebo e313281/89918dc: first maze path arrived 0.5-2 s after the leg-1 turn ended; the turn used the bearing to the leg's STOP (163 deg NED) into the M1 divider |
| coursenav/loc_node.py | `publish_zero_drift` (launch true): zero drift until the drift EKF is ready (30 PF updates) | The navigator plans nothing without /coursenav/drift; on the ground the fit gate refused every scan: no leg-1 path in > 40 s (REALISM=1). Offline every leg plans in ~0 s |
| coursenav/nav_node.py + fly_course.py | pre-plan every path leg from its start (route `start_xy`, then previous stop), write `<run>/navigator_ready`; use cached path within 0.5 m; fly_course waits for the marker (300 s) before refnav | Paths exist before arming; removes the turn/path race |
| coursenav/loc_node.py, skybeta_health, launch | `adaptive_hz` 4 once sigma < 0.05 m / 2 deg; health at <= 4 Hz; localizer + navigator `nice 5`; per-scan CPU cost logged | 1.4-core Jetson-sized budget: localizer ~30 %, health ~11 % of a core |
| course/course_v3.json | sim_profile.fixture_offset_m [0, 0] | Lab spawn was 0.2 m north of the mark; localizer prior must be the mark |
| launch/refnav_skybeta.launch.py | interface, host_depth (max_range 12 m), health, loc_node (depth topic, height source), fly_course `--event-start --skip-preflight --no-own-localizer --no-return --start-from-mark --landed-z 0.10`, refnav `px4_output:=skybeta`, arm_wait_s 600, use_sim_time arg | Corridor rays through the nets hit the hall 6-9 m out: 6 m cap -> "corridor not observed" landing; drone 14 dist_bottom 0.100 landed |

## Must never

- Publish /fmu/in, arm, or change modes from team code; edit `skybeta_flight_interface` or the organizer camera config.
- Loosen a lab safety rule to pass (fit gate, corridor observation, stop distance, estimator health). Find the cause.
- Change vendored code beyond the patch; new behaviour goes in `refnav/skybeta*.py` or the launch file.
- Run test suites on `compute` during a sim run, or > ~4 heavy jobs on the laptop (FlyX lab tab drops).

## Sim-fidelity facts the Gazebo session fixed (re-check if a run looks wrong)

- VIO odometry must be start-relative (origin + heading at VIO start, ENU x forward), like cuVSLAM.
- PX4 must have ONE vision source (the stack's bridge); a hidden world-frame EV skews heading.
- Receipt clocks: a sim that pauses needs `use_sim_time:=true`.

## Open hardware items (tell Matthew)

- Pad size (0.64 m estimated) and orientation unmeasured; the drone has no colour image, so the pad is
  the course-file position plus drift correction.
- Localizer fit gate can stay closed (24 % fit seen in the sim on the ground); the start is the mark, so
  it only costs in-flight drift correction.
- Startup order: wait for refnav "wait -> arm" (it streams) before the pilot arms + switches to Position.

## Blocked-recovery port snapshot (2026-11-08)

This copy is pinned to workspace `refnav/blocked-recovery` at
`ed2271bd49c34bf5fd77cbd5d5124e28be778905`. Use its script from this directory.
The older table above records historical adaptations, not additional edits to apply.
For this upstream revision:

- Remove the old `fly_course._free_start` and navigator vertex-removal patches.
  Upstream now owns bounded start escape; the vertex patch broke its 0.1 m/s
  escape-speed test. `coursenav/navigator.py` is byte-identical to upstream.
- Preserve the upstream reconstructed-map default and its behavior. OBJ/materials
  match Downloads/new_course_origin_map.zip byte-for-byte; occupied cells and labels
  match exactly. Keep upstream generated-map metadata and algorithm unchanged.
- Carry forward group154 f952857 clock adapters: health freshness, depth scheduling,
  and localizer rate caps follow ROS time only in simulation. Hardware timing and
  safety thresholds remain unchanged. The simulator preplan watchdog fails closed.
- Install the full runtime tree for copied ROS installs; source/symlink installs keep
  their original layout. Pass use_sim_time to host_depth and declare its dependencies.
- `assets/new/` contains packaging, hardware adapter, and focused regression files.
  The script copies this tree recursively. Keep this branch-specific skill separate
  from the historical global skill.
