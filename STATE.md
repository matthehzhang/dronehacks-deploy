# STATE — read this first, rewrite ## now on every handoff

## now
goal: PAUSED by Matthew on 2026-11-08; preserve context and push to original `origin/package_154_new`. Do not continue the navigation fix or start a flight without a new request.
next: On resume, read `docs/PACKAGE154_ZERO_SPEED_HANDOFF.md` and its evidence; reproduce the start-clearance deadlock before editing `team_ws/src/kestrel_nav/coursenav/navigator.py` or `smooth.py`.
verified: Original repo is `matthehzhang/dronehacks-deploy`; navigation source on the drone is `package_154_new` at `92b0c1fbd5b3e9fad1b52b18cd8984465554451c`.
verified: Installed directly in `skybeta-14:~/team_ws/src/kestrel_nav`; 152 source files matched, native build passed, 29 focused native tests passed before the observed flight.
verified: Run `run_20261108T103950` commanded a roughly 22-degree left yaw to face its newly planned first path; measured yaw followed the command.
verified: During 15.1 seconds of `go`, forward `v_cmd` stayed 0.0; first 0.231 m of the path had zero speed and zero reported lateral slack.
verified: Prior-map-only reproduction from estimated start `(-2.4067, 0.7239)` to `(-2.5, 1.35)` recreated the stalled path and approximately 234-second duration estimate.
verified: `_effective()` zeroes clearance in inflated blocked cells; the start's unmasked modeled clearance was about 0.210 m versus rotor reach 0.2941 m and full planning clearance 0.4140828453 m.
verified: Descent followed Offboard -> Position with `failsafe=false`, user-intended mode Position and RC throttle near minimum; policy yielded and never entered its landing state.
verified: Jetson launcher `~/team_ws/tools/fly.sh 154new` dispatches to `154new.sh restart`, records a fresh bag, then watches logs; laptop command is `ssh -t dronehacks '~/team_ws/tools/fly.sh 154new'`.
open: NO navigation fix or new regression test has been written or deployed. A software start-clearance deadlock is established; safe recovery behavior remains to be designed.
open: Whether the takeoff offset was actual motion, VIO drift, or map alignment error is unresolved; the original bag omits local occupancy/depth images, so exact fused-grid replay is unavailable.
decisions: Do not add a minimum-speed floor, erase obstacles, shrink the footprint, weaken PX4 checks, or treat the estimated 21 cm clearance as safe for a 29 cm rotor radius.
decisions: Proposed only: separate geometric clearance from blocked-cell masks, certify any initial recovery connector, and reject untraversable paths with an explicit reason instead of publishing valid zero-speed travel.
decisions: Fix worktree `/home/matthew/projects/hackathon/dronehacks/deploy-154new-fix`, branch `fix/package154-start-clearance`; other worktrees have unrelated edits and must be preserved.

## log
- 2026-11-08: Installed package_154_new, added the 154new launcher and fly.sh alias, diagnosed the first-turn/stall/descent from on-drone traces and rosbag, reproduced the clearance-mask deadlock offline, then paused at Matthew's request before implementing a fix.
