# Actual-drone run: mapping starvation and repeated avoidance

Source: `Downloads/20261107-201744_bag/20261107-201744_bag/bag/bag_0.mcap`,
13,242,368 bytes. The recording covers 70.90 seconds. The source was read only.
This report describes recorded behavior; the bag does not identify an exact Git
commit, and is not assumed to be the historical run022.

## Why the occupancy map did not produce a new route

The final localizer status reports **894 scans, 2 accepted updates and 892
rejected/gated scans**. The old map localizer's confidence gate was also gating
the live occupancy mapper. At the beginning of the recording its XY sigma was
0.065 m and yaw sigma 2.08 degrees, just beyond the 0.06 m / 2 degree gates.

Only two scans reached occupancy integration. The map requires repeated positive
evidence to declare an occupied cell: its log-odds threshold is 2, with a hit
increment of 0.85. Two accepted scans cannot build that evidence. Across 72
published grids, the content changed only once; the final grid contained 390
known cells, **zero occupied cells and zero novel landmarks**. The planner had
almost no new obstacle information to replace its original route.

This differs from the linked group154 simulation, where live occupancy updated
but the planner kept obsolete prior walls/mandatory waypoints and retained a
stale path after failed replans. Both failures require changes beyond choosing
a different shortest-path algorithm. Map-free integration must depend on fresh,
calibrated depth and the fused local pose, without an original-map PF confidence
gate; route validity must be checked against changes in cell content.

## Why it did not turn to find another direction

The recorded controller used path-corner turns and a hold/backoff/recheck loop.
It did not command an exploratory sweep:

- 16.17 s: navigator logged a preplanned 9.01 m route.
- 17.39 s: forward movement began.
- 21.04 s: a path-corner turn, followed by movement.
- 23.95 s: obstacle reported 0.93 m ahead; then hold, 0.20 m backoff and recheck.
- Through 64 s: repeated blocking at approximately 0.84–0.95 m, with short
  resumes along the same leg rather than a new observation goal.
- 67.71 s: PX4 left Offboard for Position; the controller yielded immediately.

PX4 heading spans only **14.92 degrees** in the bag (VIO: 14.27 degrees).
There are brief yaw commands up to 20.43 degrees/s, but no full search rotation.
This supports a missing search/replanning behavior, rather than proving that
the drone ignored a commanded search turn.

## What the bag can and cannot establish

All 354 recorded estimator-health messages were true. PX4 position, VIO,
commands and mode changes were recorded. The last recorded state is armed,
airborne, in Position mode; this bag does **not** confirm a later landing or
disarm. Small estimator reset counters recur in the logs and were accepted by
the old controller. They should be considered when reviewing map consistency.

The bag has camera calibration, but **no raw camera images or depth frames**,
and no recorded controller status/path topic. It cannot establish which exact
physical surface caused each depth stop, whether the apparent gap was visible,
or whether a complete collision-free route existed in the measured data.
The custom VisualSlamStatus type was unavailable to this offline environment;
those messages were not decoded. The other selected types decoded successfully.

## Evidence

- [Commands and measured heading](actual_commands_and_yaw.png)
- [Recorded occupancy](actual_occupancy.png)
- [Parsed counts, measurements and events](summary.json)
- [Complete recorded ROS logs](rosout.txt)
- [Recorded localizer, interface and health states](states.json)

The new `searchnav` strategy has no prior map or PF input. Its tests cover new
depth obstacles reaching the grid, same-count cell changes, immediate route
withdrawal/zero commands, alternative frontier selection and bounded stationary
scanning. Its runtime qualification is recorded separately; these tests do not
constitute a successful physical flight.
