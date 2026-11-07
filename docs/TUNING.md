# Bounded physical response diagnostic (branch `two`)

This code measures the response from organizer ENU velocity commands to observed
odometry. It does not tune PX4 rate/attitude gains, write parameters, arm, change
modes, or command autonomous landing. Nothing in this implementation has flown.

The pilot still arms and switches to Position after ground qualification. There
must be one team setpoint publisher and a fresh organizer flight-interface
instance. Do not run `live_mission` beside `tune_mission`. Completion and faults
request zero twist while authority is fresh; the pilot takes over and lands.
Stale authority means silence. A stopped process depends on organizer/PX4
fallback, not an independent safety processor.

## Measure the configuration first

`tuning_config` is required. An absent file, missing dimensions, conflicting map
geometry, or an impossible time budget prevents startup. Coordinates are the
published **course ENU** frame; `start_course` is the measured launch fixture
`[x, y, yaw_deg]`. The actual log's odometry and commanded velocity are in **odom
ENU**, with the transformation anchored at handover. No simulator truth is used.

Schema (replace every `null` with a measured value; this template is deliberately
not executable):

```json
{
  "start_course": [null, null, null],
  "speeds_mps": [0.10],
  "time_limit_s": 160,
  "candidates": [{
    "name": "measured_open_lane",
    "kind": "open",
    "start_xy": [null, null],
    "end_xy": [null, null],
    "height_m": null,
    "half_width_m": null,
    "start_turn_radius_m": null,
    "end_turn_radius_m": null
  }]
}
```

An open stretch or a tube can be selected. For a tube, use `"kind": "tube"`,
`"start_open": true`, and `"end_open": true`: endpoints must be measured centres
of **open zones beyond the tube mouths**, not points inside the tube. The routine
never turns around in a narrow tube or returns by flying blind backwards. Radius
means the measured obstacle-free disk around each endpoint. The current forward
depth view is checked while turning; it cannot prove unseen rear clearance, so
the measured disk is an essential assumption, not an inferred sensor result.

Current bounds are lane length 2–8 m, half-width at least 0.50 m, and open turning
radius at least 0.65 m. All candidates in a run share one height, which must match
the course map's planning-band height (normally 0.8 m). The planner checks lane
and turning-zone geometry against that map. These are initial conservative
limits, not measured aircraft qualification. The map and declared start fixture
can still be wrong; slower motion does not repair position bias.

## Sequence and budget

The existing map/depth navigator climbs and approaches the first lane at no more
than **0.15 m/s**, with existing geometry, heading, and front-depth caps. The
route ends at the lane start; it does not continue through the course. Transit
blockage ends the diagnostic. The particle-filter correction runs during transit;
its course-to-odom transform is frozen during measurement to avoid injecting
localizer corrections into measured response. Raw odom remains the fit input.

At the lane: settle, face forward in the open start zone, command a two-second
pulse, brake with zero horizontal velocity, and wait until measured speed is
below 0.035 m/s for one second. It then transfers forward at up to 0.10 m/s to a
separate held-out location, stops and settles, and runs the validation pulse at
the same speed. The default location is the lane midpoint, moved earlier if
needed to retain pulse and stopping reserve; optional measured
`validation_offset_m` selects distance from lane start. The `TRANSFER` and
`TRANSFER_SETTLE` phases are excluded from response fitting. A tube configuration
must be reviewed to ensure this location actually lies inside its measured
interior; the generic routine does not infer tube mouths. It then creeps to the open end, stops,
turns at at most 0.25 rad/s, reacquires forward depth, and returns at at most
0.10 m/s. Translation always faces the observed direction. A larger configured
speed begins only after the preceding pair and slow return succeed.

Default `[0.10]` means one training/validation pair. Speeds can increase by at
most 0.10 m/s per pair, with a 0.40 m/s hard maximum. The initial stopping reserve
is `0.30 + 0.8*v + v*v/(2*0.15)` metres: assumed delay 0.8 s, assumed braking
deceleration 0.15 m/s², and 0.30 m residual margin. The diagnostic does not replace
these limits with an automatically fitted value. A failed pulse, stop, sensor,
lane boundary, altitude, heading, clamp, or log write latches a stop.

Startup estimates the complete pulse/return schedule plus map transit at
0.15 m/s. It refuses an estimate beyond the declared budget, at most 170 seconds.
This is a necessary budget screen, not an upper time bound: transit distance
divided by its maximum speed is optimistic because corners, heading, and depth
can slow the navigator. The runtime limit includes transit and is independent of progress. Select fewer
pairs or a different measured lane when it does not fit; do not increase speed
or weaken clearance to meet the budget. Open and tube candidates can require
separate flights. An interrupted or incomplete pair is not a valid fitted trial.

## Build and props-off checks

Build the team overlay using the normal organizer workflow. The dedicated entry
point is `tune_mission`, and `tune.launch.py` starts the organizer interface, host
stereo depth, and this node only. A configuration can be reviewed and validated
offline by importing `validate_config` and constructing `TuningProgram`; that
does not start ROS. A launch command is intentionally not executed by development
or test scripts.

Before any separately approved physical run, check measured geometry, camera/IMU
clock agreement, `/oak/imu/data` frame and rate, recording disk space, pilot
handover/takeover, duplicate/stale-sensor faults, and organizer clamps with props
off. Then use a bounded low/tethered trial. In-process tests and synthetic
rehearsals do not qualify the aircraft, net depth, vibration, VIO accuracy, or
unseen obstacles.

## Record format and analysis

Each run creates a new `tuning_<time_ns>.jsonl` in `log_directory`. Schema 1 uses
`kind`, `t_ros`, `t_mono`, `source_stamp`, `frame_id`, `phase`, `candidate`, `trial`,
`trial_id`, `split`, and `target_speed`. `trial_id` is `candidate:trial`; even trials
are training, odd trials validation. The header contains the measured config;
the final summary contains completion and dropped/logging-error counts.

Odometry callbacks record position, XYZW orientation, covariance, and acceptance
at callback rate. IMU callbacks record raw acceleration and angular velocity in
the declared IMU frame. Command records use the exact published message capture
stamp, ENU velocity and yaw rate; they also include authority, clamps, watchdog
action, and fault. Tracking, interface state, depth coverage, and transition
events are recorded separately. Invalid nonfinite values are JSON `null`.

Odometry and commands must both be `odom`; other odom frames refuse qualification.
Duplicate IMU or odometry stamps do not refresh the guard. IMU age is limited to
0.3 s on both ROS capture and monotonic receipt clocks. The inherited odometry,
tracking, depth, and authority watchdogs remain in force. Logging failures stop
the experiment; a missing final summary after a crash must be treated as
incomplete data.

The separate `tools/fit_tuning.py` consumes these records to estimate response
delay, lag, and braking behavior with data-quality gates and held-out validation.
Use the companion synthetic rehearsal to exercise its refusal cases offline.
Fit results are review material; this routine never writes PX4 gains or silently
applies a fitted limit to another flight.
Acceptance requires a final `DONE` result and the entire configured trial
schedule in both the recorded pulses and completed-trial summary. A clean
process shutdown during braking is still incomplete and cannot produce an
accepted diagnostic. Expected pilot takeover after `DONE` preserves completion;
the authority watchdog still stops publishing.

## Published-course rehearsal evidence

The map-derived example in `docs/evidence/tuning/course_map_only.json` spans the
open approach and tube, with endpoints 0.40 m beyond the mouths. It is marked
`map_only`, which the physical ROS adapter rejects. It is **not** a physical
measurement file. `derive_candidate.py` searches the published geometry;
`course_geometry.json` records the candidate and clearance certificates.

The selected straight is 2.324 m long at 0.8 m body height. The map-derived
half-width is at least 0.588 m, and endpoint turning disks exceed 0.678 m radius.
The mapped maze approach is 8.729 m with 0.40 m minimum planned clearance.
Distance/cruise arithmetic gives approximately 136.24 s including the diagnostic;
corner slowdown, turns, uncertainty and holds can make the actual transit longer.
The 170 s runtime limit remains active; this arithmetic is not a completion
guarantee or a collision-free flight certificate.

The separate **lane-only** rehearsal completes in 71.65 s. Its first 0.10 m/s
pulse is outside the tube; its held-out pulse is inside, after a forward transfer
and settled stop. With injected lag 0.45 s and delay 0.15 s, the fitter recovers
approximately 0.45001 s and 0.15022 s and improves held-out prediction. This uses
ideal synthetic observations and a first-order plant, so it validates software
sequencing and identification, not real tuning values. Reports and pulse
positions are in `lane_rehearsal_report.json` and `lane_summary.json`.

Reproduce the pure map checks and lane rehearsal from the deployment root:

```sh
python3 team_ws/src/kestrel_nav/tools/rehearse_tuning.py --config docs/evidence/tuning/course_map_only.json --check-course --out /tmp/tuning-course.jsonl --report /tmp/tuning-course-report.json
```

**The maze transit has not been simulated end to end.** The cloud workspace uses
native PX4 topics; this node uses the organizer's skybeta interface. No verified
cloud adapter is included. The diagnostic is ready for offline review and
props-off integration work, not for a claimed ready-to-launch cloud or physical
flight. No flight command was issued during development.

Final package verification: `python3 -m pytest -q tests` from
`team_ws/src/kestrel_nav` — **203 passed** in 49.82 s; compile and diff checks
passed. Independent review found and verified the fix for incomplete-run fit
acceptance. Additional focused checks cover stale/duplicate IMU and odometry,
physical rejection of example geometry, lane/transit handoff, turn zones,
distinct measurement positions, missing command heartbeats, and pilot takeover.
Source hashes and the exact rehearsal command are in `provenance.json`.

## Offline commands and physical launch entry point

From this deployment checkout, these commands exercise synthetic software only:

```sh
python3 team_ws/src/kestrel_nav/tools/rehearse_tuning.py --config docs/tuning_synthetic.json --out /tmp/tuning-rehearsal.jsonl --report /tmp/tuning-rehearsal-report.json
python3 team_ws/src/kestrel_nav/tools/rehearse_tuning.py --fixture --out /tmp/tuning-fit-fixture.jsonl --report /tmp/tuning-fit-report.json
python3 team_ws/src/kestrel_nav/tools/fit_tuning.py /tmp/tuning-fit-fixture.jsonl --out /tmp/tuning-fit.json
```

The generic example is marked `synthetic`; the physical adapter rejects it.
Any separately supplied course-derived configuration marked `map_only` is also
refused by the ROS adapter until real clearance has been measured and reviewed.
It has ideal sensors and a delayed first-order plant, not native PX4, ROS
transport, real maze transit, or real obstacle geometry. A passed rehearsal
does not establish native simulator or hardware readiness. A fitter refusal
after a completed controller rehearsal must remain visible; completion is not
proof of sufficient excitation or reliable identification.

After separate mission approval and the props-off checks, the physical entry
point is the following, with a reviewed **measured** configuration path:

```sh
ros2 launch kestrel_nav tune.launch.py tuning_config:=/absolute/path/measured-lane.json log_directory:=/absolute/path/new-run-logs
```

This command starts readiness publication and may accept the pilot's subsequent
handover; it is not a read-only configuration check. It was not run during
development. To analyze an actual recorded run later, use `fit_tuning.py` on its
JSONL file and review the validity gates before interpreting fitted values.
