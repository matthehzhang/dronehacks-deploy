# Safe course policy (`safe.launch.py`, branch `kestral_nav934am`)

Goal: **finish the course without touching anything.** Speed is spent only where the evidence says it
is safe. Nothing here was tuned to a flight; nothing here has flown. Offline evidence only.

## How to run it

```bash
# laptop -> drone 14
rsync -a --delete team_ws/src/kestrel_nav/ dronehacks@skybeta-14.local:team_ws/src/kestrel_nav/
# on the drone, /etc/skybeta/drone.env:
COMPETITOR_PKG=kestrel_nav
COMPETITOR_LAUNCH=safe.launch.py
# restart the stack ON THE GROUND, then watch:
docker rm -f skybeta-stack
docker exec -it skybeta-stack tmux attach -t skybeta
ros2 topic echo /kestrel_nav/live_status --once      # state, phase, gate_refused, pf_sigma
```

Sequence: the pilot arms, switches to Position; the interface enters Offboard; the node climbs straight
up, flies the course, measures and centres on the tunnel mouth and the window, flies to the pad,
descends at 0.15 m/s until PX4 detects landing and auto-disarms (`COM_DISARM_LAND` 2 s). The pilot
never has to touch the sticks on a good run. **Any stick input ends the run (rule).**

## What it does differently from `live.launch.py` (the carrot follower), and why

| Rule | Evidence it comes from |
|---|---|
| **Never yaw while translating.** Every corner is a stop: brake, hold, turn in place at <= 30 deg/s (PX4 `MPC_YAWRAUTO_MAX` is 45), settle, then go straight. While moving, yaw is only a heading hold (<= 6 deg/s). A vertex turning < 12 deg is flown through with that heading hold. | Run010 (cloud sim, 2026-11-07): contact at maze divider M1 while yawing ~27 deg/s and translating 0.15 m/s. Truth velocity swung toward the nose, PX4's estimate did not; 12.3 cm cross-track, 0.268 m truth clearance. Run004: 14 cm lost to tracking on a moving corner at M3. |
| **Face where you go.** Translation only along the body axis (plus <= 0.06 m/s cross-track), so the forward depth camera watches the swept corridor. | The OAK-D sees forward only; sideways and reverse motion is unobserved. |
| **Close the position loop on VIO.** Twist setpoints computed from cuVSLAM odometry (through the PF-corrected course frame), not pose setpoints. | Run010: VIO position followed truth while PX4's velocity estimate lagged the slide. Interface: a pose setpoint has no speed limit (flown at `MPC_XY_VEL_MAX` 12 m/s) and never expires; a twist is capped at 1.5 m/s and expires to hold after 1 s. |
| **Hold where you stopped.** After every stop the station-keeping anchor is the current position, never the waypoint. | Run008: a stale hold target pulled the vehicle 10 cm sideways. |
| **Measure gates; no measurement, no crossing.** At the tunnel entry and the window pre-point the vehicle faces the crossing, measures the jambs in depth (5 frames, median, both jambs), corrects its lateral position estimate to the measured opening, side-steps onto the centre line, re-measures, and keeps re-centring on the opening while it is still > 0.55 m away. PF corrections pause through the crossing and the PF is re-seeded at the gate-corrected pose afterwards. If the opening is not measured consistently within 12 s, or disagrees with the believed position by > 0.25 m, the vehicle **holds** (`gate_refused`), it does not cross blind. | Window: 0.914 m square in the OBJ (0.871 m on the 5 cm grid); prop-tip span 0.588 m, so 0.14-0.16 m per side when perfectly centred. Sim runs 3/4: PX4/VIO estimate 0.16-0.20 m off truth. Offline here: belief was 5-8 cm off at both gates; after measured centring, **truth** was within 2 cm of the opening centre in every case. |
| **Stop distance is a budget.** Speed is capped so that `reach + 0.10 + v*0.6 s + v^2/(2*0.25)` fits before the nearest depth return in the body corridor; otherwise the segment is shortened or the vehicle holds. Segment chords are re-checked on the live grid at the hard inflation; a vehicle already inside it may only move to increasing clearance. | `0.6 s` / `0.25 m/s^2` are conservative assumptions, **not measured** on hardware (the workspace diagnostic assumed 0.8 s / 0.15 m/s^2). |
| **Size is the 7-inch airframe.** Prop reach 0.294 m = max over rotors of hypot(x,y) + 7 in/2, rotor XY from `px4/params/07_airframe_geometry.params`. Used for the corridor guard and offline contact scoring. The planner graph still uses the course file's 0.28 m + 0.12 m (0.40 m): that is reach 0.294 + 0.106 m margin. | 7-inch props confirmed by the team, 2026-11-07. Raising the planner radius to 0.294 + 0.12 makes the window route infeasible on the 5 cm grid (0.412 < 0.414 m); the measured crossing replaces that margin at the window instead. |
| **Land on the pad.** Centre within 0.10 m, descend at 0.15 m/s (< `LNDMC_Z_VEL_MAX` 0.25) with xy station keeping; depth is waived only in this phase (stereo is blind < 0.4 m); odometry/tracking/state/PX4 land state stay required. | CP5 is "land and disarm with the whole drone inside the pad's outer ring". The interface's 0.3 m floor clamps pose setpoints only, not twist. |

Speeds: 0.35 m/s on straight segments whose map clearance is >= 0.55 m, 0.25 m/s elsewhere,
0.12 m/s through the window. All launch arguments (`v_open`, `v_tight`, `v_gate_window`,
`turn_rate_deg`, `land_vz`, `gate_requires_measurement`).

## Offline evidence (`tools/safe_offline.py`, results in `safe_policy_offline.jsonl`)

Kinematic offline sim (`livenav/offline_sim.py`): first-order velocity lag 0.45 s, drifting VIO
(scale 2 %, 0.3 deg/m yaw drift, random walk), rendered depth, PF localizer. Contact is scored
against the **0.294 m** reach. New here: `yaw_slip_s`, which rotates the achieved velocity toward the
turn by `yaw_rate * yaw_slip_s` (1.0 s used): a crude stand-in for the run010 mechanism.

| Policy | Cases | Landed | Contact | Min clearance | Time | Pad error |
|---|---|---|---|---|---|---|
| **safe** (seeds 0-3 x slip {0, 1.0} x start error {0, 0.10 m + 2 deg}) | 16 | **16** | **0** | 0.335-0.364 m | 156-170 s | 0.09-0.13 m |
| live carrot follower, slip 1.0 (seeds 0, 1) | 2 | 0 | **2** (maze M1 corner, t ~ 8-10 s) | 0.292 m | - | - |
| live carrot follower, no slip (seeds 0, 1) | 2 | 1 | 1 (M1 corner) | 0.292 / 0.300 m | 67 s | 0.06 m |

The carrot follower's offline contact point (-2.24, 2.27)-(-2.24, 2.30) is the same M1 corner as
run010's real contact (-2.18, 2.30).

## Known limits and risks (read before flying)

1. **Time.** 156-170 s offline against a 180 s limit that starts when the drone leaves the mark and
   stops on disarm. Hardware will be slower if VIO, depth or holds misbehave. `time_limit_s` 175: past
   it the vehicle holds where it is (the pilot then lands; the run is scored on checkpoints reached).
   If time is short, the levers are `v_tight`/`v_open` (straight segments only) and `turn_rate_deg`,
   not the safety rules.
2. **Turning in place beside a wall** has lost VIO before (managed run 2026-11-07, see `follower.py`
   docstring). 30 deg/s is below PX4's auto limit, but visual-only cuVSLAM near a close, plain wall is
   the main unmeasured risk of stop-turn-go. A tracking loss latches FAULT (zero twist); the pilot lands.
3. **The tunnel is dark.** VIO and depth in it are unmeasured on hardware.
4. **Window vertical clearance** is not measured (cruise 0.8 m body height; sill 0.305 m, top 1.219 m).
5. **Landing** relies on PX4's land detector firing under a 0.15 m/s offboard velocity descent and on
   `COM_DISARM_LAND` 2 s. Not verified on hardware.
6. **Stopping model** (0.6 s, 0.25 m/s^2) and creep exemption (0.06 m/s corrective motion does not need
   an observed corridor) are assumptions.
7. Offline sim decouples yaw from XY except through the crude slip term; it does not model VIO dropouts
   or PX4's EKF. Passing it is necessary, not sufficient.

## Before the first flight

Props off: launch, check the "SAFE policy ready" line (chicane window, 19 goals), arm, switch to
Position, confirm `offboard_active` and a CLIMB command (vz up, vx/vy ~0) in `live_status`. Then a
tethered or low hover test of CLIMB + first segment + one in-place turn, watching `/visual_slam/status`
through the turn. Then the course.
