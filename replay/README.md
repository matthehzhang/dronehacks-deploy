# replay: put a flight's rosbag back on the course map

Shows where the drone *thought* it was during a flight, drawn on the course map, so you can check
those estimates against the walls and against each other.

```powershell
python replay\replay_bag.py <bag> --open
```

`<bag>` can be a group154 run directory (`flight_logs/<stamp>_group154_XXXX`), a rosbag2 directory, or
a single `.mcap` file. The script writes one self-contained `replay.html` next to the bag (map and
flight are embedded inside it). It opens offline in any browser and can be copied to someone else.

It needs only plain Python 3.8+ with no packages, so it runs on a Windows laptop, a Mac, the Jetson, or
inside the skybeta container. Reading a 75 s group154 bag takes under a second.

## Getting the bags off the drone

`team_ws/tools/group154_record.sh` writes each run to `~/team_ws/flight_logs/<stamp>_group154_XXXX/`
on the drone (`bag/`, plus `course_v3.json` and `drone.env`); `flight_logs/group154_latest.txt` names
the newest run.

```powershell
scp -r dronehacks@skybeta-14.local:team_ws/flight_logs/20261107-235044_group154_OvyGUo .
python replay\replay_bag.py 20261107-235044_group154_OvyGUo --open
```

Copy the bag only after the recorder has stopped (`group154_record.sh stop`, or wait for
`finished.json`). A bag copied while it is still recording, or one whose recorder was killed, has no
`metadata.yaml` and ends in a cut-off chunk. The replay still reads everything before the cut and
says so, but the last few seconds are missing.

## What you see

Four position estimates, all converted to the course frame (metres, x east, y north, origin at the
enclosure floor centre):

| Trace | Source | Meaning |
|---|---|---|
| **Localizer (map match)**, orange | `/coursenav/pose`, else `/coursenav/status` | particle filter matching depth scans against the course map. The only estimate that uses the map. |
| **Navigator belief**, blue | PX4 local position + launch fixture + `/coursenav/drift` | the frame refnav/fly_course fly in: where the flight code thought it was |
| **PX4 EKF2 (uncorrected)**, green | `/fmu/out/vehicle_local_position*` + launch fixture | PX4's estimate with no map correction |
| **cuVSLAM VIO**, purple | `/visual_slam/tracking/odometry` + launch fixture | visual odometry dead-reckoned from the launch pose |

Map layers you can toggle:
- **Course map:** top-down view of the OBJ.
- **Cross-section:** a red slice of the map at the drone's current height, which shows what it could actually hit there (window wall, tube interior, boulders below the band drop out).
- **Live map:** the `/coursenav/occupancy` grid the drone built during the flight.
- **Planned targets:** refnav's targets and setpoint.
- **Course goals and launch fixture.**
- **Reach circle:** 0.294 m, the 7-inch prop reach.

The side panel shows the following:
- **Clearance.** Distance from the reference estimate to the nearest mapped wall at its height, plus the minimum over the whole flight (click it to jump there).
- **Gap chart.** How far each estimate is from the reference estimate over time.
- **Height chart.**
- **Event list.** Arm and land, PX4 mode changes, refnav phases, cuVSLAM tracking loss and jumps, drift-EKF resets, and ROS warnings/errors (repeats folded). Click an event to seek to it.

Controls: <kbd>Space</kbd> play/pause, <kbd>←</kbd>/<kbd>→</kbd> step 0.5 s (with Shift, 5 s), drag to pan,
wheel to zoom, double-click to reset the view. Add `#t=42.5` to the file URL to open at a time.

## How to read it: was the drone where we think?

**The bag contains no ground truth.** Every trace is the drone's own estimate. The replay cannot
tell you where the drone really was. It shows where each estimate put it, and what to look for:

- **The estimates disagree** (gap chart). The localizer drifting away from VIO/PX4 means the map
  match and the odometry no longer agree. One of them is wrong, and the flight was steered by the
  navigator belief (blue).
- **A path goes through a wall that the drone did not hit,** so that estimate is wrong there.
- **Live-map walls do not line up with the course map.** The live grid is built from depth at the
  *believed* pose. If the walls it saw are offset or rotated from the map's walls, the belief was
  offset or rotated by the same amount. This is the strongest check the bag allows.
- **The landing spot.** Compare where the drone actually landed (note it, or tape-measure it from
  the launch mark) with where each trace ends.

For a real ground-truth comparison, film the run from above or mark the actual landing spot. The replay
then shows which estimate was right.

## Options

| Option | Default | Use |
|---|---|---|
| `--start X Y YAW_DEG` | from the bag | launch pose in the course frame. Needed for bags without `/coursenav/drift` (e.g. `record_bag.sh` recordings); also "what if it was placed 10 cm off" |
| `--landed-z M` | 0.10 | body-origin height when landed (refnav_skybeta's `landed_z`) |
| `--map DIR` | `maps/new_course_origin` | another map package (an `.obj` + optional `manifest.json` / `course_v3.json`) |
| `-o FILE` | next to the bag | output HTML path |
| `--csv` | off | also write every trace as CSV (`estimate,t_s,x_m,y_m,z_m,yaw_deg`) |
| `--open` | off | open the result in the browser |

Where the launch pose comes from, in order: `--start`; the `fixture` field of `/coursenav/drift`
(what the localizer itself used); the `start` block of the run directory's `course_v3.json`; the
map's `course_v3.json`. The terminal output and the page both say which one was used.

## Assumptions and limits

- **Frame chain.** cuVSLAM odom → course is `fixture ⊕ odom` (SE(2)). PX4 local NED → odom is
  `(x, y) = (y_ned, x_ned)` and `yaw = π/2 − heading`, which is the skybeta `px4_odom_bridge`
  convention (FRD with the VIO heading, `declared_heading_deg = 0`). Checked against a real run: refnav's first
  PX4-frame target converts to (2.314, 2.367), against the map's tunnel_entry at (2.311, 2.369).
  Bags from other bridges (e.g. the FlyX lab one, declared heading 90°) put the PX4 trace in the
  wrong place.
- **Heights** are `landed_z` plus the rise since the bag's first sample, so they assume the bag starts on
  the ground (the group154 recorder always does).
- **The map is approximate.** `maps/new_course_origin` is the team's `topview_entrance`
  reconstruction, not a survey (see its `README.txt` and `manifest.json` limitations). A clearance
  number is the distance from an *estimate* to an *approximate* wall. Treat it as a flag, not a
  measurement.
- Chunk compression: none, or zstd (Python 3.14 has it built in; otherwise `pip install zstandard`),
  or lz4 (`pip install lz4`). group154 and `record_bag.sh` bags are uncompressed.

## Layout

```
replay/
  replay_bag.py            CLI
  skyreplay/mcap_reader.py MCAP reader (tolerates truncated bags)
  skyreplay/cdr.py         CDR decoder driven by the schema text embedded in the bag
  skyreplay/extract.py     bag -> course-frame traces, events, live-map frames
  skyreplay/course_map.py  OBJ/MTL + manifest -> viewer map
  skyreplay/viewer.html    the replay page (template)
  maps/new_course_origin/  the map from new_course_origin_map.zip (OBJ, MTL, course/manifest JSON, README)
  tests/test_replay.py     python -m unittest discover -s replay/tests
```
