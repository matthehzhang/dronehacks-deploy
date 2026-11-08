# Blocked-recovery ROS port

Workspace `refnav/blocked-recovery` at `ed2271bd49c34bf5fd77cbd5d5124e28be778905`
is vendored into `team_ws/src/kestrel_nav`. Initial port branch: `kestral_blocked_recovery`. Approved recovery extensions: `cram304`.
The branch-specific port skill and exact upstream adapter patch are in `docs/port-to-skybeta`.

## Behavior and map

The initial port retained upstream blocked recovery and applied the Sky Beta transport, startup,
landing and simulation-clock adapters. The later user-requested recovery extensions below modify
the controller, navigator and prior monitor. Clearance confirmation, the per-leg block limit,
estimator supervision and pilot authority remain enforced.
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

## Constructor fix verified on the actual drone

`tools/check_localizer_startup.py` passed twice on skybeta-14 ARM64 / ROS Humble:
default fit 0.35 and custom fit 0.42. Both PF and prior monitor retained the requested
value. The complete constructor loaded the new map and live/local grids; timers ran
for 1.2 seconds; shutdown completed cleanly. Domain 77 isolated these checks from
flight traffic. This verifies startup and timer paths, not future sensor callbacks
or hardware flight reliability. Four strict constructor regression cases also pass.
Both Gazebo localizers subsequently initialized and published ready=true with prior
placed and fit=1.0; the original startup failure is preserved in simulation evidence.

## Range investigation

See `docs/evidence/blocked_recovery_20261108/RANGE_FINDINGS.md` and the reproducible
range probe. Out-of-range/invalid forward stereo becomes unknown, not a near wall.
Unknown coverage can stop and eventually land the policy. The retained real-bag
segment instead shows finite near obstacle hits and eventual Offboard exit; depth
images were absent, so the causes of those near returns cannot be established.

## Single-command start using the existing device launcher

An identical package is staged at `~/policies/group154_blocked_recovery/kestrel_nav`.
The existing device `fly.sh` supports the `group154_*` selector and was syntax checked.
From the laptop (`dronehacks` already selects USB-C automatically when connected):

```bash
ssh -t dronehacks '~/team_ws/tools/fly.sh group154_blocked_recovery'
```

This is a user-invoked start: it checks disarmed telemetry, asks for sudo credentials
if needed, saves the previous package, selects this package, restarts the stack and
watches mission output. The agent did not invoke it or arm hardware. Pilot arming
and mode switching are still required. The active package changes only when run.
The preloaded localizer and OBJ hashes were verified against the local committed copy.

## Randomized worlds with the deployed prior unchanged

User selected mismatch testing: L13 and L02, seed 8177313980703840530, identical
policy snapshots, no COURSENAV_COURSE overrides or injected map assets. The previous
same-map runs were discarded. Compact evidence is in `evidence/blocked_recovery_20261108/sim/`.

- L13: blocked at 0.94 m, held/backed off/rechecked, then deliberately landed and
  disarmed. Zero checkpoints, zero touches; grader records off-pad ground contact
  at 30.675 s. Prior dropped on poor fit but prior_required is false in the supplied
  policy, so it did not itself gate the flight.
- L02: PX4 external-vision position/yaw flags cleared, local velocity became invalid,
  then PX4 rejected velocity Offboard. Heartbeat was only 45.8 ms old, with no nearby
  gaps above 200 ms. This is estimator validity rejection, not evidence of a policy
  heartbeat timeout. Refnav yielded; the agent commanded simulation-only LAND and
  confirmed disarm. No checkpoints; grader remained waiting.

Both localizers stayed alive. Both worlds remain available; pilots and evidence
recorders were stopped after both aircraft were confirmed landed and disarmed.
These runs did not demonstrate successful completion under prior-map mismatch.


## Requested recovery extensions (cram304)

These implement the three approved policy changes without new sensors, simulator truth, PX4
parameter changes, arming authority or automatic Offboard re-entry:

1. A confirmed path blockage sends an explicit numbered replan request and stops publishing the
   old command path. Pending/failed plans also invalidate the controller path. Delayed or uncertified
   replies cannot change the current route or future targets. Normal failed planning retries at the
   existing one-second interval instead of remaining stuck without a route.
2. After the existing four-second hold, recovery can scan at fixed XYZ through -25, +25 and the
   original yaw, at at most 20 degrees/second. It requires settled translation, usable fresh depth,
   a new capture after each dwell, and no close return in the swept footprint. The entire recovery
   has a 20-second budget and at most five explicit requests; depth loss, estimator failure, PX4
   takeover and the existing block limit remain authoritative. A failed recovery still backs off,
   rechecks and lands. A successful plan turns and checks depth before translating.
3. Prior status begins unknown. It needs two seconds of consecutive informative fit >=35% to become
   placed, or three seconds below 30% to become dropped. Sparse/empty/invalid/out-of-order scans and
   gaps over 0.5 seconds interrupt qualification. Unknown prior status cannot explain away locally
   observed obstacles. Existing configured thresholds, hysteresis after certification and
   `prior_required` setting are retained.

The detour's changed prefix must have an observed-free swept footprint including vehicle clearance,
then rejoin the exact unchanged reference suffix. Unknown cells remain blocked; prior walls, nets,
keep-outs and gate walls remain constraints. The original mission goals are retained. A different
path ID without a real geometric departure is rejected. The local grid publication expires free
observations after eight seconds; republishing a grid cannot renew that evidence. The recipient
also requires a grid publication no older than two seconds.

The five friend files found in Downloads were reviewed, not copied into the deployment. Their
all-prior-removal fallback and clearing behavior are not in this change. That fallback could route
through unobserved cells; the supplied localizer would also restore the duplicate-parameter startup
crash and remove required Sky Beta adapters. The original prior map assets are unchanged.

Development and Gazebo results must be distinguished: the additional diverse-world batch is
being restarted using the exact final `cram304` commit, with the original policy prior unchanged.
The hardware launcher staging now contains `cram304` policy commit `c52cb7c`. It was not
activated or flown by the agent. Do not interpret earlier Gazebo evidence as tests of these extensions.


Local validation of the recovery extensions: full deployment suite 476 passed / 2 xfailed;
final controller, free-age, planner protocol, prior and full-localizer-constructor rerun 76 passed;
copied-install package test 1 passed, including the new observed-free module and bundled prior.
The dynamic controller harness completes a supplied detour after turn/check/go. That is not a
claim of full Gazebo course completion. Exact commands and source hashes are in
`evidence/blocked_recovery_20261108/recovery_extensions/validation.json`.


The `cram304` package was transferred via `ssh dronehacks` after the local checks. The native ARM64
build, full ROS localizer constructors at fit thresholds 0.35 and 0.42, actual OccupancyGrid
publication/subscription free-age check, observed-free import and launch argument checks passed.
The sensor-observed free cell published probability 12; the same evidence aged nine seconds
published unknown (-1). All 153 committed package files match in both the isolated workspace and
`~/policies/group154_blocked_recovery/kestrel_nav`. The map hash and existing `fly.sh` are unchanged.
Previous copies are backed up under `~/policies/_replaced/blocked-recovery-before-c52cb7c/`.
No hardware stack restart, arming or flight was performed. Native evidence is in
`evidence/blocked_recovery_20261108/recovery_extensions/device/`.


## Final diverse-world tests and remaining defect

Both final-policy runs used exact `c52cb7c` sources and the unchanged prior, in verified different
DIVERSE01/DIVERSE02 geometry. Both failed to complete: zero checkpoints, deliberate off-target
abort landings, then natural disarming. Grader durations were 24.450 and 17.425 simulation seconds.
There was no manual LAND intervention, airborne Offboard-loss transition or localizer crash.
Depth stayed usable and fresh. Neither run emitted a recovery request.

The missing integration is the normal-planner-failure case: `on_plan_result` invalidates the old
path, then `_go_path` returns before updating along-path clearance. The obstacle-triggered scan
is never entered and the 10-second corridor timer expires. Entering bounded recovery on loss of a
previously usable route is still needed; even that will not guarantee a valid detour when retained
prior constraints conflict with the live map. Exact blocking cells have not been isolated. These
runs do not validate successful mismatch flight. Evidence: `sim_diverse_final/` beside the earlier
records, including actual geometry/source verification, logs, traces and matching scorecards.

## Main-workspace installation correction

The initial isolated native build missed a deployment defect: the main workspace recursively
found both `src/kestrel_nav` and `policies/blocked-recovery/src/kestrel_nav`, producing the logged
duplicate-package build failure. An ignore marker first proved the discovery fix. Per Matthew's
subsequent request, the final arrangement instead installs the new exact package directly at
`~/team_ws/src/kestrel_nav` and deletes the redundant nested isolated workspace. The saved launcher
copy remains outside the build workspace at `~/policies/group154_blocked_recovery/kestrel_nav`.
The main workspace now discovers one package, and its actual package build and launch argument
check pass. No hardware stack restart or arming occurred. PX4 telemetry was unavailable; no
policy or flight-interface process was running during this inert source installation.


## Prior correction integration candidate

Matthew authorized integrating the friend's prior-wall clearing and local-only fallback, and
requested device transfer before testing. This candidate adds protected masks and persistent
camera-confirmed prior walls, clearing only while the prior is dropped using fresh confident
free-space cells (occupancy probability <=23%, a weighted log-odds threshold, not an exact count
of hits versus passes). Placement/unknown restores all prior cells and invalidates fallback plans.
Local-only fallback retains protected/confirmed surfaces, keep-outs, gate walls and a buffered
enclosure boundary; its entire changed swept footprint must be observed free. Normal full-leg
fallback requires full observation; bounded recovery may rejoin an unchanged suffix that still
passes the corrected prior map. Path certification persists after recovery resumes and expires
with its sensor evidence. Ordinary planner failure now enters the same bounded recovery.

The source map assets, goals, ROS adapters, existing depth/estimator rules and pilot authority
remain unchanged. Candidate validation is pending at the device-first handoff; this entry is not
a claim of passing tests or successful simulation flight.


The prior integration is now runtime commit `9b76afd` on `cram304`. It was installed in the main
hardware workspace and saved launcher copy before validation, per request. All 158 files match.
The main native build and actual ROS localizer/navigator constructors passed; all three path legs
and their non-commandable recovery references were prepared. Laptop suite: 545 passed / 2 xfailed;
focused integration: 49 passed. Patch replay reproduces the runtime and the prior assets remain
unchanged. The two diverse comparison flights are in progress; these checks do not yet establish
successful mismatch flight. Evidence: `evidence/blocked_recovery_20261108/newprior_integration/`.


The 9b76afd comparison flights have finished. Both entered bounded recovery and issued all five
requests, unlike c52cb7c, but neither found a certified detour. DIVERSE01 cleared 14 prior cells and
confirmed 42; DIVERSE02 cleared none. Both landed and naturally disarmed, with zero checkpoints
and no manual intervention. The limiting errors were no observed-free rejoin beyond the recovery
lookahead, and on DIVERSE01 an uncertified start footprint. This version is not validated for
successful mismatch flight; exact evidence is under `newprior_integration/sim/`.

The real device launcher subsequently reported missing /skybeta/state despite a running policy.
The underlying CLI error was the ROS daemon XMLRPC `RuntimeError: !rclpy.ok()`. Direct reads with
`ros2 topic echo --no-daemon --spin-time 2 --once --full-length /skybeta/state std_msgs/msg/String`
succeeded. The existing device fly.sh was patched only at that read, keeping its disarmed check.
A backup was saved outside team_ws and bash syntax validation passed. No stack restart occurred.
Live telemetry then showed arm phase, healthy estimator, fresh depth, PX4 linked, disarmed/landed.


## Finite distant depth rays (2026-11-08)

The local scan previously discarded every return beyond its 3 m obstacle horizon,
so even valid 5–8 m measurements could not certify nearby free space. Local scan
extraction now preserves the existing near-obstacle scan and adds measured distant
rays only in bins without nearer points. Their free segment starts at the calibrated
camera origin, ends before both the body-centred 3 m horizon and the measured range
minus three standard deviations (plus a cell margin), and creates no occupied
endpoint. Missing/invalid depth remains unknown. Prior assets, PF scan extraction,
clearance, evidence expiry and flight authority are unchanged. Device transfer must
precede validation; staging does not replace the already-running Python processes.
Validation for this change is recorded under `evidence/blocked_recovery_20261108/range_ray_fix/`.


## Recovery creep and recovery evidence (2026-11-08, `recovery_creep`, default ON since 2026-11-08; `recovery_creep:=false` restores the previous behaviour)

**Why.** Real-flight report: after the leg-1 turn the drone moved a little, paused, swept and never
entered visible free space. Through the real `LocalGrid` + `ObservedFree` (scratch evaluation: open
space, then approach → 70° turn → 4 s hold → ±25° sweep), recovery could certify only a strip ~1 m wide,
0.9–1.7 m ahead, and **never the drone's own position**. So every request failed ("recovery start footprint
is not certified free" / "no observed-free rejoin beyond recovery lookahead") however open the room was.
Causes, by measured effect: unknown counts as obstacle for the 0.414 m footprint (kept: it is the rule that
keeps paths out of unseen space); 48 free rays (gaps between rays beyond ~1.8 m); 3 m carving; 8 s free
memory (the cells beside the drone were seen ~16 s before the first post-sweep request).

**What the flag turns on** (one launch argument, `recovery_creep`, threaded to all three processes):

- localizer `recovery_evidence`: 160 dense **free-only** rays (`scan.with_dense_free_rays`: never mark an
  endpoint; a bin with any near point but too few for a return gives no ray and vetoes far carving;
  obstacle marking stays on the 48-bin scan), distant returns carve to 4 m (obstacles still only to 3 m),
  free evidence lasts 20 s (was 8), and the airframe's own 0.29 m disc is published free for 20 s
  (`LocalParams.body_radius_m` / `swept_age_s`; a confident obstacle, or any later return on the cell,
  overrides it). Explicit `local_*` parameters still override each value.
- navigator `recovery_creep`: when no certified detour exists and refnav's request says `creep`, answer with
  a **hop** (`Navigator.plan_creep_hop`): ≤ 0.5 m straight-line, every swept cell observed free, end outside
  the fused obstacle buffer, ≥ 0.10 m closer to the leg goal by fused-map path distance (`cost_to_go`,
  8-connected Dijkstra). Prior walls are never cleared by it. A hop in flight is re-certified every tick on
  what is left of it (map changes elsewhere do not withdraw it); a hop is never stored as a reference route.
- refnav `recovery_creep`: a hop (`recovery_hop` path) never replaces the leg's target/length/reference.
  Phase `creep`: turn in place to the hop, then a carrot at ≤ `creep_speed_mps` 0.15 m/s with the live-depth
  corridor check on (stop distance at creep speed); unobserved corridor → no advance; obstacle, withdrawal,
  staleness or timeout ends the hop without progress. After a hop: hover `creep_dwell_s`, one new request
  (`creep_hops` = n), then the usual sweep → back-off → recheck → land. Bounds: `creep_max_hops` 6 (then only
  full detours are requested), `creep_max_stalls` 2 in a row → land, `recovery_timeout_s` restarts only on a
  hop that made progress, never past `creep_max_s` 45 s from the block. A consumed hop's request id can never
  move the aircraft again. Every new parameter is range-checked.

**Validation (laptop).** `tests/test_recovery_creep.py`: 25 cases (sensing, hop planning, navigator protocol,
refnav controller, and a closed-loop DynamicPX4 flight around an unmapped wall: 2 hops then a certified
detour, course complete, ≥ 0.294 m from the wall throughout, ≤ creep speed within hops). Synthetic
panel scenario: deployed settings → "recovery start footprint is not certified free"; flag on → three chained
~0.5 m hops around the panel (goal distance 4.62 → 3.18 m).

**Not yet validated.** Gazebo DIVERSE01/02 with `recovery_creep:=true`; the drone. Known risks to review
before flight: hops are planned in VIO coordinates (drift), the sides of the airframe are outside the live
depth check during a hop, 20 s free memory assumes nothing walks into seen space unobserved. First real test
caged or tethered, pilot on the kill switch. `docs/port-to-skybeta/assets/skybeta.patch` needs
`scripts/refresh_patch.sh` on the machine that has the workspace checkout.
