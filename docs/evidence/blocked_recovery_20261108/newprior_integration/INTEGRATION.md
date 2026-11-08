# Prior correction integration

Runtime commit: `9b76afd` on `cram304`. Both active hardware source and saved launcher copy were
updated before final validation, as requested. No hardware stack restart, arming or flight command.

The five downloaded files were integrated selectively. Sky Beta startup, depth/height/topic/clock
adapters and the fixed duplicate min_fit_frac declaration were preserved. The original prior OBJ,
voxel map and course JSON remain byte-identical to c52cb7c. Mission goals are unchanged.

## Behavior

- Clearing operates only on explicit dropped prior state. A recent local grid must confidently see
  through the whole candidate prior cell. Its probability threshold is 23%, a weighted/clamped
  log-odds confidence rule, not an exact arithmetic count of hits versus passes.
- Nets, see-through columns (window/tunnel roof), keep-outs, virtual gate walls and enclosure
  boundaries remain protected. The enclosure limit includes vehicle clearance even across an
  empty prior boundary cell.
- A prior cell within 0.15 m of a local obstacle is restored and remembered as confirmed while dropped,
  including after the local grid forgets that obstacle. Cleared cells stay cleared during that dropped
  interval; placement or unknown restores all prior cells and invalidates fallback paths/caches.
- When fused planning fails, local-only fallback retains protected/confirmed prior surfaces and live
  obstacles. Unknown space is never certified free. A normal full-leg fallback needs its entire swept
  footprint observed; bounded recovery can certify only a changed prefix if it rejoins an unchanged
  suffix that still passes the corrected prior map. Original goals and clearance limits remain.
- Per-cell free evidence expires after 8 seconds; publication must be no older than 2 seconds. A fallback
  certificate is checked after recovery resumes, using accepted PID/progress to discard only the
  already-flown prefix. Restored walls and changed map constraints invalidate ready recovery replies.
- A normal failed plan now holds at measured position and requests bounded recovery before the
  generic corridor timer. Existing 20-second total budget, maximum 5 requests, per-leg block count,
  depth/estimator supervision and PX4 authority remain. Failure still leads to landing.
- Initial failed planning can also recover using non-commandable preplan references. Without a flown
  path there is no guessed backoff motion. No-route recovery may reuse newly certified geometry;
  physical-obstacle recovery still requires a geometric departure from the blocked reference.

## Validation status

Copy hashes, native main build and complete ROS constructor/preplanning evidence are under
`device/`. Final laptop regression passed 545 tests with 2 expected failures (198.13 seconds); the focused
prior-integration suite passed 49 tests (13.33 seconds). The branch-specific patch replay reproduces
all 10 checked runtime files. Two identical-geometry comparison flights are now starting with exact
9b76afd and are recorded separately when complete. A passing build is not evidence of successful mismatch flight.
