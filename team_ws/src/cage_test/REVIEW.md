# Review brief: `cage_test` before the first cage flight

For whoever reviews this on a machine with ROS 2 Humble. Read `README.md` in this directory
first (what it does, and the flight procedure), then the repo's `CLAUDE.md` and
`docs/competitor_guide.md` §1–§4 for the interface it sits on.

**What has been checked:** the pure geometry (`test/test_hop_plan.py`, 14 tests) and the node's state
machine against a toy vehicle with rclpy stubbed out (`tools/offline_rehearsal.py`, 17 checks). Both
pass, on a Mac with **no ROS installed**. **Nothing has been built with colcon, run under real
rclpy, or launched.** That is what this review is for.

## Setup (any Humble machine; x86 is fine)

Only three packages are needed: `px4_msgs`, `skybeta_flight_interface` and `cage_test`. The rest of
the repo needs Isaac ROS and an OAK-D, so do not build it.

```bash
mkdir -p ~/cage_ws/src && cd ~/cage_ws
git clone -b cage-hop-test https://github.com/matthehzhang/dronehacks-deploy.git repo
git clone -b release/1.16 https://github.com/PX4/px4_msgs.git src/px4_msgs   # the image pins 392e831c
ln -s ../repo/src/skybeta_flight_interface src/
ln -s ../repo/team_ws/src/cage_test src/
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Run

```bash
cd ~/cage_ws/src/cage_test
python3 -m pytest -q test/
python3 tools/offline_rehearsal.py
cd ~/cage_ws/src/skybeta_flight_interface && python3 -m pytest -q test/     # untouched by this branch; must still pass
ros2 pkg executables cage_test                                            # frame_check, hop_test
ros2 launch cage_test hop_test.launch.py                                  # must start both nodes cleanly
ros2 launch cage_test hop_test.launch.py forward_m:=0.5 touchdown:=false  # args parse as float/bool
```

## The live check that matters most

Write a throwaway script in the style of
`src/skybeta_flight_interface/test/integration_offboard.py`. It should stand in for PX4 (publish
`vehicle_status`, `vehicle_local_position`, `vehicle_land_detected` on the topic names
`skybeta_flight_interface.topics.resolve_endpoints()` gives, BEST_EFFORT + TRANSIENT_LOCAL) and for
cuVSLAM (`/visual_slam/tracking/odometry`). Then run `hop_test.launch.py` against it and walk the
pilot sequence:

1. disarmed, `nav_state` POSCTL or ALTCTL, landed, valid local position
2. armed with `nav_state` ALTCTL
3. `nav_state` POSCTL → expect a `VehicleCommand` DO_SET_MODE(1, 6) on `/fmu/in/vehicle_command`
4. `nav_state` OFFBOARD → then move the fake vehicle toward each `/fmu/in/trajectory_setpoint`

Confirm on `/fmu/in/trajectory_setpoint` (NED):

- up: `position[2]` ramps from -0.3 toward -1.0 at about 0.03 m per 10 Hz tick, and x/y stay at the start
- forward: with odometry yaw 0 (nose along odom +x = ENU east), NED **y** grows by 1.0 and x stays put
  (ENU→NED swaps x/y). Repeat with yaw 90° and the change should move to NED **x**
- touchdown: `position` NaN, `velocity = [0, 0, +0.4]` (NED down), `OffboardControlMode.velocity`
  true, repeated every tick
- after the fake vehicle reports `landed`, then disarmed: `hop_test` logs "disarmed ... Ready"
- a pilot takeover (`nav_state` back to POSCTL mid-leg) logs "Offboard lost" and nothing advances

And `ros2 run cage_test frame_check` against the same fakes plus a fake
`/fmu/out/vehicle_attitude`. It must start (that topic name comes from `fmu_topic('out',
'vehicle_attitude')`) and print sensible fwd/left/yaw as the fake odometry moves.

## Things I was not sure of; please look

1. **Launch argument types.** `LaunchConfiguration` strings become typed parameters through YAML
   parsing (`'1.0'` → float, `'true'` → bool), the same way `competitor_example/launch/competitor.launch.py`
   does it. An integer (`forward_m:=1`) should fail the float declaration. Confirm both.
2. **`VehicleAttitude`** in `px4_msgs` release/1.16: does it carry a `MESSAGE_VERSION`, and is
   `msg.q` a 4-element wxyz array? `frame_check` assumes so.
3. **`skybeta_flight_interface.frames`** imports in `frame_check` (`wrap_pi`, `yaw_from_wxyz`,
   `yaw_ned_to_enu`): check they exist with those meanings at the image's version.
4. **Touchdown hand-off.** During touchdown the interface is in twist mode. If `hop_test` dies, the
   twist expires after 1 s and the interface holds the *last pose* (the 0.5 m low point), so the
   vehicle climbs back to 0.5 m. Intended (it hovers rather than drifts); flag it if you disagree.
5. **PX4 claims**, from the v1.16.0 source (`MulticopterLandDetector.cpp`
   `_get_ground_contact_state`; `MulticopterPositionControl.cpp` offboard `want_takeoff`;
   `PositionControl::_positionControl`). If you can, check them against v1.16.2:
   - an Offboard pose never lands, because ground contact needs `trajectory_setpoint.velocity[2] >=
     1.1 * LNDMC_Z_VEL_MAX`
   - takeoff starts when the setpoint z is above the vehicle
   - position setpoints go to the P controller unsmoothed
6. **PX4 `nav_state` after an auto-disarm.** If it stays OFFBOARD while disarmed, `hop_test` waits
   to see it leave Offboard before it will start again. The pilot moving the switch to Altitude
   before re-arming does that.
7. **Anything that publishes to `/fmu/in/*` or would fight `flight_interface`:** there should be
   none. `hop_test` publishes only `/skybeta/setpoint/pose` and `/skybeta/setpoint/twist`.

## Ground rules from the repo

Do not modify `src/skybeta_flight_interface` or anything outside `team_ws/src/cage_test`. Do not put
`px4_msgs` in the drone's team workspace (it is only in this desk setup). Report findings rather
than pushing to `fleet-ready`.
