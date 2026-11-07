# cage_test: first flight and tuning flights in the cage

Tools for the first autonomous flights of a skybeta drone in an open cage or net:

| | What it does | Armed? |
|---|---|---|
| `hop_test` (`hop_test.launch.py`) | take off, up 1.0 m, forward 1.0 m along the nose, back, down to 0.5 m, touch down, PX4 disarms | flies |
| `tune_test` (`tune_test.launch.py`) | same takeoff and landing, with tuning patterns in between: hover, step responses, a speed sweep, a yaw-rate sweep, a vertical sweep, a square ([Stage 6](#stage-6-tuning-flights-tune_test)) | flies |
| `frame_check` | you carry the vehicle; it prints forward/left/up/yaw as VIO and PX4 see them, plus the position offset between them | **disarmed only** |

It is a `team_ws` package, the way every team ships code here: it goes in `~/team_ws/src` on the
drone and needs **no image rebuild** and no change to the organiser checkout (`~/skybeta`). It uses
the stock skybeta interface only: `flight_interface` with `auto_offboard_on_arm:=true`,
`/skybeta/setpoint/pose`, `/skybeta/setpoint/twist`, `/skybeta/state`, cuVSLAM odometry. Nothing
touches `/fmu/in/*`.

**Nothing here has flown.** Neither has any Offboard takeoff on any skybeta airframe (CLAUDE.md,
competitor guide). This flight is the test.

## The flight

```
t ≈ 0     pilot: arm (switch on Altitude, throttle low), then Position
t ≈ 0.5   flight_interface enters Offboard; hop_test logs "Offboard active ... Leg 1/4: up"
          up 1.0 m at 0.3 m/s           hold 3 s
          forward 1.0 m along the nose  hold 3 s
          back to above the start       hold 3 s
          down to 0.5 m                 hold 3 s
          touch down at 0.4 m/s, PX4 lands and disarms 2 s later (COM_DISARM_LAND)
t ≈ 30 s  "disarmed ... Ready for the next hop"
```

Design choices, each checked against the PX4 v1.16 source:

- **The setpoint slides along each leg at `speed_mps`; it does not jump to the waypoint.** In
  Offboard, PX4 sends a position setpoint straight to its P controller with no smoothing
  (`PositionControl::_positionControl`), capped only at `MPC_XY_VEL_MAX` = 12 m/s on these airframes.
  It stops sliding while odometry is stale (> 0.5 s).
- **Forward is the nose at takeoff**, taken from the odometry orientation, not odom +x.
- **Touchdown uses a twist.** Poses are floored at `min_altitude_m` (0.3 m), and in Offboard PX4's
  land detector only reports ground contact while the setpoint commands a descent of at least
  1.1 × `LNDMC_Z_VEL_MAX` = 0.275 m/s (`MulticopterLandDetector::_get_ground_contact_state`). A pose
  never does that, because its velocity is NaN. So the last 0.5 m is a straight-down twist,
  refreshed every tick until PX4 disarms. Horizontally that part is a zero-velocity hold, not a
  position hold, which is why it starts low. `touchdown:=false` stops at 0.5 m and the pilot lands.
- **It starts only on a switch into Offboard it saw happen.** A node (re)started while already in
  Offboard never starts a hop from the air. After a disarm it is ready again.
- **When Offboard is lost (the pilot takes over), it stops for good** until the vehicle is disarmed.
- If a leg is not reached within `length / speed + 15 s`, it holds there and logs why (usually a
  clamp, or VIO).

Launch arguments (write numbers with a decimal point: `1.0`, not `1`):

| Argument | Default | |
|---|---|---|
| `climb_height_m` | 1.0 | |
| `forward_m` | 1.0 | |
| `low_height_m` | 0.5 | must be above `min_altitude_m` and below `climb_height_m` |
| `speed_mps` | 0.3 | |
| `hold_seconds` | 3.0 | |
| `arrival_radius_m` | 0.2 | |
| `touchdown` | true | false: hold at `low_height_m` and the pilot lands |
| `descend_speed_mps` | 0.4 | refused below 0.3 |
| `max_altitude_m` | **2.0** | the interface's ceiling, lowered from 3.0 for the cage |
| `anchor_setpoints` | false | see stage 3 |

---

## Stage 0: laptop, no ROS (1 minute)

```bash
cd team_ws/src/cage_test
python3 tools/offline_rehearsal.py      # state machine against a toy vehicle: 42 checks
python3 -m pytest -q test/              # geometry, plan, analysis: 30 tests
```

The rehearsal swaps rclpy for stand-ins and flies the real node against a vehicle that
follows the setpoint with PX4's position gain and yaw gain. It covers the nominal hop, a turned heading, a
restart while in Offboard, a pilot takeover, `touchdown:=false`, a landing PX4 never detects, an
odometry dropout, an unreachable leg, bad parameters, the full tuning sequence, a PX4/VIO yaw
offset and a sideways push that trips the abort. It also checks every run directory it writes.
It checks logic, not dynamics.

## Stage 1: SITL (optional; the sim profile has never been run)

`sim/` (Isaac Sim 5.1 + Pegasus + PX4 v1.16.2 SITL, x86 with an RTX GPU) is a **spike**: an ordered
checklist that nobody has executed (`sim/README.md`, top-level README "Simulation"). If you have a
machine for it, work through steps 1 to 6 there. Then, in place of step 7, put this package in
the sim container's `/root/team_ws/src`, run `team_build.sh`, and:

```bash
ros2 launch cage_test hop_test.launch.py
```
```
pxh> commander mode altctl
pxh> commander arm
pxh> commander mode posctl       # the "switch to Position after arming": Offboard follows
```

Score against Pegasus ground truth (`/drone1/state/pose`). The same steps catch a rotated
"forward" before the real drone does. If the sim isn't available, skip this stage: stages 3 and 4
are the real gate.

## Stage 2: put it on the drone

Check the drone first (on the drone, as `dronehacks`):

```bash
cd ~/skybeta && host/doctor.sh                 # all green
cat ~/skybeta/VERSION                          # 0.3.2 or later: arm, THEN Position
```

On the FC (QGC → Parameters, or the MAVLink shell):

| Must be | Why |
|---|---|
| firmware git hash `7a39c59` (patched v1.16.2) | stock v1.16.2 loses TELEM2 after every stack restart (bug #15) |
| `COM_RC_OVERRIDE = 3` (`09_autonomy_takeover.params`) | so the sticks take over from Offboard. **Drone 16's saved dump (2026-11-03) has 1**, which predates 09 |
| `EKF2_EV_CTRL = 9`, `EKF2_EV_DELAY` measured | VIO position hold (03b) |
| `COM_DISARM_LAND = 2`, `LNDMC_Z_VEL_MAX = 0.25` | what the touchdown relies on |

Copy the package in. From your laptop, on this branch:

```bash
rsync -a team_ws/src/cage_test dronehacks@<drone-ip>:~/team_ws/src/
```

or on the drone: `git clone -b cage-hop-test <this repo> /tmp/cage && cp -r /tmp/cage/team_ws/src/cage_test ~/team_ws/src/`.

Build and test it inside the running stack:

```bash
cd ~/skybeta && host/run.sh shell
team_build.sh --packages-select cage_test
python3 -m pytest -q /root/team_ws/src/cage_test/test/
ros2 pkg executables cage_test                 # frame_check, hop_test, tune_test
```

The stack's `competitor` window builds **all** of `~/team_ws` before it launches anything. If
another package in there (a team's `kestrel_nav`, say) fails to build, the window stops at a shell
and nothing flies. Build the whole workspace once with `team_build.sh` to be sure.

## Stage 3: frame check (disarmed, props off, in the cage)

Put the drone on its launch mark on the cage floor, nose along the direction "forward" should go,
and restart the stack there. The VIO origin and heading come from where the stack starts, and so
do the interface's altitude floor and ceiling:

```bash
docker rm -f skybeta-stack                     # back in ~10 s; keep the drone still for a minute
```

Then, in a shell on the stack:

```bash
cd ~/skybeta && host/run.sh shell
ros2 run cage_test frame_check
```

Carry it level and read both lines (`VIO` and `PX4`):

| Move | Expect on both lines |
|---|---|
| 1 m along the nose (measure it) | `fwd +1.00 left 0.00` within ~0.1 m |
| back, then 1 m to the right | `fwd 0.00 left -1.00` |
| lift 0.5 m | `up +0.50` |
| 90° clockwise seen from above | `yaw -90`; `yaw offset` stays constant through the turn |
| put it back on the mark | `pos offset` horizontal under ~0.1 m |

This is the "controlled repeat (nose fixed, measured 1 m forward / right, then 90° CW)" that
CLAUDE.md lists as pending for drone 16. **Do not fly forward until it passes.**

- VIO forward shows up as left/right: the camera mount yaw (`CAMERA_YAW` in `drone.env`) is wrong.
- VIO is right but PX4 is not, or `yaw offset` drifts: the problem is between the bridge and EKF2
  (CLAUDE.md "Toilet bowl").
- `pos offset` horizontal over ~0.1 m: PX4 would jump that far when Offboard starts. Launch with
  `anchor_setpoints:=true` (stage 5, by hand) after reading the competitor guide §1.

Ctrl-C when done. The stack does not need a restart: `frame_check` publishes nothing.

## Stage 4: props-off rehearsal (armed, props OFF)

Point the stack at the hop test in `/etc/skybeta/drone.env` (these two keys are yours to set):

```
COMPETITOR_PKG=cage_test
COMPETITOR_LAUNCH=hop_test.launch.py
```

Restart the stack **on the launch mark** (`docker rm -f skybeta-stack`) and keep it still for a minute.
Watch the `competitor` window:

```bash
docker exec -it skybeta-stack tmux attach -t skybeta      # Ctrl-b 3; Ctrl-b d to leave
```

Every 5 s while waiting, `hop_test` prints `ready: ...` or `NOT ready: <what> false`. Do not arm
on `NOT ready`.

1. Pilot arms (switch on Altitude, throttle low), then switches to Position within a few seconds.
   **Expect:** `flight_interface`: `auto offboard: switched to Position after arming: requesting
   Offboard`, then `Offboard active`. `hop_test`: `Offboard active. Start ENU ... Leg 1/4: up`.
   Motors spin up. With no props it cannot climb, so PX4 disarms ~10 s after arming
   (`COM_DISARM_PRFLT`) and `hop_test` logs `disarmed (was LEG)`. That is the expected result.
   If it says `staying out of Offboard: <reason>` instead, fix that reason (competitor guide §4).
2. Again, but the pilot **raises the throttle** a few seconds into Offboard. **Expect:** PX4 goes to
   Position, and `hop_test` logs `Offboard lost during LEG`. If the sticks do nothing,
   `COM_RC_OVERRIDE` is not 3: stop and fix it before flying.
3. Pilot practises the kill switch (channel 5) once.

## Stage 5: the cage flight

Props on, drone on the mark, stack restarted there and left still for a minute, net closed, pilot
with the transmitter and a hand on the throttle. Record a bag from a second terminal:

```bash
cd ~/skybeta && host/run.sh bag -- /skybeta/state /skybeta/setpoint/pose /skybeta/setpoint/twist \
    /fmu/in/trajectory_setpoint /fmu/in/offboard_control_mode
# mcap into ~/skybeta/bags; Ctrl-C after the flight
```

The default set (cameras, IMU, odometry, `/fmu/out/*`) leaves out the `/skybeta/*` topics; the
arguments after `--` add them.

**First flight, conservative:** half the distance, and the pilot lands it. Stop the `competitor`
window's launch (attach, `Ctrl-b 3`, Ctrl-C; two flight interfaces fight) and run it by hand from
`host/run.sh shell`:

```bash
ros2 launch cage_test hop_test.launch.py forward_m:=0.5 touchdown:=false
```

**Then the full hop** with the defaults. The stack's `competitor` window runs it after a restart,
or launch it by hand without arguments.

Pilot: arm on Altitude, switch to Position. **Take over** (throttle up toward centre) if it:

- goes sideways or backwards on the forward leg,
- drifts more than ~0.3 m from where it should be, or circles (toilet bowl),
- climbs past ~1.3 m, or does not stop descending at 0.5 m when `touchdown:=false`,
- has not landed ~15 s after "touching down" (it also logs `Pilot: if it is on the ground,
  disarm`; disarm with the arm switch if it is already down).

Things that look odd but are expected:

- It lifts to 0.3 m almost at once and then climbs at 0.3 m/s. Any pose is floored at 0.3 m, and
  PX4 runs its own takeoff ramp (`MPC_TKO_RAMP_T` = 3 s).
- Heading is never commanded (yaw is NaN), so PX4 holds the heading it took off with.
- Land detection can take a few seconds after touchdown: the range finder adds hysteresis.

After the flight, the bag has the setpoints, `/skybeta/state`, `/visual_slam/tracking/odometry`
and `/fmu/out/vehicle_local_position`.
Setpoint against odometry is the tracking error. `odom_enu` against `local_enu` in the state shows
whether the vision link held. Set `COMPETITOR_PKG`/`COMPETITOR_LAUNCH` back to the team's code
when you are done.

## Stage 6: tuning flights (`tune_test`)

Only after the hop (stage 5) has flown cleanly. Same takeoff, same landing, same pilot rules,
same take-over criteria. Between them the node flies the patterns in `sequence`, each starting and
ending at the hover point H (`climb_height_m` above the launch mark):

| Pattern | What it flies (defaults) | What it is for |
|---|---|---|
| `hover` | hold at H for 20 s | baseline: position and yaw noise (VIO and PX4), PX4-VIO offset, hover vibration on the camera IMU and the FC IMU, battery sag. Run it first on every new airframe |
| `steps` | **step** (no ramp) 0.5 m forward, back; 0.5 m left, back; 0.3 m up, back; 0.3 m down, back; 5 s hold after each | position-loop step response per axis: rise time, overshoot, settling, steady-state error. Roll and pitch respond differently on these frames (rotor arms are not symmetric) |
| `speed` | forward 1.0 m and back at 0.25, 0.5, 0.75, 1.0 m/s, slowest first (`speed_axes:=forward,left` adds the lateral axis) | tracking lag, overshoot at the end of the leg and achieved speed as speed rises. The sequence stops at the first speed that goes off the path |
| `yaw` | spin 360° left then 360° right in place at 30, 60, 90, 120 °/s, slowest first | yaw tracking (rate achieved, lag), **position drift while spinning** (a camera extrinsic or frame error shows up here first; it is the toilet-bowl test), and VIO tracking under rotation (odometry age) |
| `vertical` | to H+0.4 m, then down and up between H−0.4 m and H+0.4 m at 0.2, 0.4, 0.6 m/s | thrust and vertical-velocity response; range finder vs VIO vs EKF2 height (the "climbs and keeps climbing" symptom) |
| `square` | 1 m forward, 1 m left, back, back to H at `speed_mps` | frame alignment over a closed path: where it ends vs where it started |
| `hop` | the stage-5 hop (forward and back) | — |

`sequence` is a comma list, flown in order: `tune_test.launch.py` defaults to
`hover,steps,speed,yaw` (about 3.5 min in the air with takeoff and landing). Do them one per flight
first:

```bash
ros2 launch cage_test tune_test.launch.py sequence:=hover
ros2 launch cage_test tune_test.launch.py sequence:=steps
ros2 launch cage_test tune_test.launch.py sequence:=speed speed_list_mps:=0.25,0.5
ros2 launch cage_test tune_test.launch.py sequence:=yaw yaw_rates_dps:=30,60
ros2 launch cage_test tune_test.launch.py sequence:=vertical
ros2 launch cage_test tune_test.launch.py                      # all four, once each one is clean
```

Or set `COMPETITOR_LAUNCH=tune_test.launch.py` in `drone.env`, as in stage 4. Lists are comma
strings, slowest first. The node refuses speeds over 2 m/s, vertical speeds over 1 m/s and yaw rates
over 180 °/s. Integers are fine here (`speed_mps:=1`).

Before the flight, check the cage has room around H: forward +1 m (`speed`, `hop`, `square`),
left +0.5 m (`steps`) or +1 m (`square`, `speed_axes:=forward,left`), height H−0.4 m to H+0.4 m
(`vertical`). The node logs the segment count and time in the air at startup.

**What to expect, and what is not a fault:**

- **The vehicle trails the moving setpoint by about speed / `MPC_XY_P`** (0.95 on drone 16): ~1 m at
  1 m/s. On a 1 m leg it never reaches 1 m/s: the rehearsal's toy vehicle peaks at 0.63 m/s. Raise
  `speed_distance_m` if the cage allows, or read the result as tracking lag.
- Yaw is held from the first yaw segment onwards (it is not commanded before that, as in the hop).
  The spin starts from PX4's own heading, so a VIO/PX4 yaw offset causes no jump.
- **The sequence stops by itself** (holds; pilot lands) if the vehicle is more than
  `abort_distance_m` (0.6 m) off the current segment's path. Lag behind the moving setpoint does not
  count; drifting sideways or overshooting past the end does. The log line says which segment.
- The pilot takes over exactly as in stage 5. Spinning at 90-120 °/s blurs the cameras the most: if
  odometry goes stale during the spin, the setpoint freezes and the log says so. Take over if it
  drifts.

### What every run saves

A **run** is one arm → disarm (+3 s). Each one gets its own directory, `~/team_ws/runs/` on the host
(`/root/team_ws/runs` in the stack; `log_dir:=` changes it), named `<date-time>_<sequence>`:

| File | Contents |
|---|---|
| `meta.json` | every parameter, the planned segments (start/end/speed/yaw), start position and heading, drone (hostname, `ROS_DOMAIN_ID`), result (`completed`, `pilot took over during …`, `failed: …`), the bag's topic list |
| `samples.csv` | each tick (20 Hz in `tune_test`): state, segment, phase (ramp/settle/hold), **the command sent** (pose x/y/z/yaw or twist vz), VIO pose, yaw and velocity, PX4 position and yaw, odometry age, armed/offboard/landed, clamp reasons, battery voltage. Starts 10 s **before** the arm |
| `imu_oak.csv` | OAK-D IMU (BNO086, ~200 Hz) accel and gyro, receipt time and header stamp |
| `imu_px4.csv` | PX4 `sensor_combined` accel and gyro, as it arrives over DDS |
| `events.jsonl` | every log line, segment start and state change, timestamped |
| `summary.json`, `summary.txt` | per-segment numbers: lag, peak speed, off-path, overshoot, rise/settling time, yaw rate achieved, yaw lag, drift while spinning, hover std, PX4-VIO offset, IMU vibration (accel std), battery. Also printed in the window when the run closes |
| `bag/` | rosbag2 **mcap** of every `/fmu/in/*` and `/fmu/out/*` topic (the setpoints PX4 got, its attitude, local position, odometry, estimator flags, the pilot's sticks in `manual_control_setpoint`, battery, failsafe flags, `sensor_combined`), `/skybeta/*`, `/cage_test/event`, cuVSLAM odometry and status, the OAK IMU, camera_info, TF and `/rosout` (every node's log). Cameras only with `record_images:=true` (~39 MB/s) |

A bag is skipped (CSVs still written) with under 1 GB free. Runs are not in git (`team_ws/.gitignore`);
copy them off with `rsync -a dronehacks@<drone-ip>:~/team_ws/runs .` and re-summarize on a laptop
with `python3 tools/summarize_run.py <run dir>` (no ROS needed).

**The PX4 log is the other half, and the more important one for PID tuning.** The bag only has
what DDS exports: no rate-controller setpoints, no actuator outputs, IMU at DDS rate. PX4 writes a
`.ulg` on the SD card for every arm → disarm (`SDLOG_MODE=0`, already set). For tuning, set
before the flights (v1.16 `SDLOG_PROFILE` bits: 0 default, 1 EKF2 replay, 3 system ID, 4 high rate,
7 computer vision):

```
SDLOG_PROFILE = 147     # 1 + 2 + 16 + 128: default + EKF2 replay + high rate + vision
                        # (+8 = 155 adds system-ID topics; watch for logger dropouts)
```

Download the `.ulg` after the session (QGC → Analyze Tools → Log Download, FC on USB) and keep it
with the run directory: the arm times match (`started_wall` in `meta.json`). Flight Review /
PlotJuggler read it. The EKF2 replay bit lets you re-run the estimator offline with a different
`EKF2_EV_DELAY` or noise, which is how to settle the delay question without flying again.

## Known gaps between rehearsal and the real flight

- PX4 dynamics, the takeoff ramp and land detection timing are only modelled crudely offline. The
  0.4 m/s touchdown is above the 0.275 m/s threshold with margin, but it has not been seen to land.
- The altitude floor and ceiling are measured from the odometry origin at stack start. Start the
  stack on the cage floor, not on a bench.
- VIO needs texture and light. A bare cage floor or a lens facing the net can lose tracking. PX4
  then has no horizontal position and the vehicle drifts in Altitude behaviour, so the pilot takes
  over.
- The tuning patterns, the bag recorder and the run logs have only run offline (the recorder not
  at all: there is no `ros2` on the laptop). Check on the bench (stage 4, props off) that an arm
  creates a run directory with a non-empty `bag/` before the first tuning flight.
- After a cold boot, wait for `timedatectl` to report synchronized (the service does this). A
  clock step makes EKF2 reject every vision sample.
