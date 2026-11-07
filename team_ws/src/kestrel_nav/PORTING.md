# Hardware branch port, 2026-11-07

Branch: `matthew-05-59am`, created from deploy `basic-obstacle-avoidance` at
`6178b4de0c88d3115c1072c6f5fd3a7819d13b75`. The earlier uncommitted sensor-safety
work is included. Separate IMU calibration files under `team_ws/config` are untouched.

Simulator source: workspace `matthew352am`, commit `dc2dd1c`.
`coursenav/planner.py` and `coursenav/navfield.py` are copied byte-for-byte:

| File | SHA-256 |
|---|---|
| planner.py | `7a65e88c1aa89c49a18ae92d1327c61a1a7e33a4af15c08c36e9a1d59f67f5c9` |
| navfield.py | `f15ce6b1f9586f071cfc823909f3d94c1417d682a3001b2c3f208ff65a05e017` |

## What runs

`ros2 launch kestrel_nav live.launch.py` starts the existing hardware velocity
mission, host stereo-depth node and organizer flight interface. The mission publishes
only `/skybeta/setpoint/twist`; the pilot and organizer interface retain arming and
Offboard control. Landing remains the pilot's responsibility at the interface floor.
This command is documentation, not a record of a hardware launch in this session.

The selected gate's opening is closed to ordinary A* searches and the hard-margin
fallback. Path shortening, nearest-free connectors, continued via legs and retained
paths preserve that constraint. The designated crossing opens only its own gate,
in the planned direction. A blocked crossing cannot retreat from the plane or post
side through the gate and re-enter it. Follower look-ahead, acceleration-limited
command direction and HOLD backoff also obey the gate rule.

The earlier safety changes require fresh odometry, successful cuVSLAM tracking,
usable depth and interface state. They reject stale/replayed inputs, latch active
sensor faults, reject motion into an unobserved depth corridor, and yield to pilot
takeover or an explicit interface hold. A zero-velocity command is not a guarantee
of physical hover when position estimation is invalid. The watchdog lives in the
mission process and cannot supervise that process's death.

## Scope retained for the time and hardware constraints

The hardware route follower and its tuning remain in use. No simulator PX4 input
publishers, arming commands, VIO binaries, acceleration feedforward or autonomous
landing logic are copied. The organizer checkout is unchanged. CPU stereo remains
in use; no VPI/GPU dependency or particle-filter tuning is added.

The simulator's routecheck advisory was not copied: it replays the simulator's
trajectory controller and would not validate this velocity follower. The alternative
navfield planner is updated and tested but is not activated in the live mission.

## Verification and review

- Baseline safety/mission subset: 72 passed before the port.
- Upstream gate regressions were transferred with package import paths adapted.
- Fresh independent Codex review checked the port and its hardware interface.
  Its findings about metadata, shortcuts, crossing backtracking and HOLD backoff
  were addressed; its final targeted run passed all 12 live-gate tests.
- The first full-course attempt timed out at the first maze corner: an extra final
  physical-inflation check rejected a 3 cm escape connector starting at estimated
  clearance 0.361 m, below the existing 0.38 m planning threshold. That unintended
  obstacle-recovery change was removed. Final added motion vetoes concern gate
  policy; existing physical-clearance and recovery behavior is retained.
- The corrected offline full-course regression passes: completion under 120 s,
  minimum clearance at least 0.28 m, landing error below 0.30 m. These are the
  existing assertions, not relaxed criteria.
- Final complete package suite: `python3 -m pytest -q tests` — **146 passed in
  49.63 s**. `compileall` and `git diff --check` pass. The offline planning CLI
  selects the window and explains why Gate B cannot meet the gate rule at 0.40 m.

The offline course uses modeled depth/VIO and a guessed velocity-response lag.
It exercises the mission core, not a ROS-connected aircraft or the complete depth
coverage watchdog. Adapter tests use ROS message doubles. ROS 2 Humble execution,
Jetson timing, camera visibility and physical flight are not qualified by these tests.

Mapped gate constraints govern planned and commanded motion; they are not an
independent measurement that the real airframe crossed exactly once. The simulator's
M1-contact problem remains unresolved. No hardware launch, upload, restart, arming,
flight, push or PX4 parameter change was performed for this port.
