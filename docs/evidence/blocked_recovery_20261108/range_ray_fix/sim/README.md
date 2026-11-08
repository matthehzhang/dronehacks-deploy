# Range-ray fix: same-world flight comparison

Both flights of `50978085fedb6b3f56e1b23cb904167c15383eec` failed completion. Each exhausted bounded recovery without an observed-free rejoin, landed away from the target, and disarmed naturally. There was no manual flight intervention or policy/configuration change during either run.

| World | Previous 9b76afd duration | Range-fix duration | Checkpoints / points | Obstacle contacts | Minimum obstacle clearance |
|---|---:|---:|---:|---:|---:|
| DIVERSE01 | 31.175 sim s | 18.900 sim s | 0 / 0 | 0 | 0.6377 m |
| DIVERSE02 | 18.900 sim s | 18.800 sim s | 0 / 0 | 0 | 0.5348 m |

Both scorecards report `crash_ground_contact`, corresponding to the policy's deliberate off-target abort landing. Abort durations are not completion-time improvements.

Native traces show recovery requests 1–5 in each run, zero certified-ready replies, and only `fused` plan mode. Both rejected replans with `no observed-free rejoin beyond recovery lookahead` and ended `still blocked after hold, back-off and recheck`. Maximum prior cells cleared were zero in both runs; maximum confirmed cells were eight in DIVERSE01 and zero in DIVERSE02. The range interpretation defect is covered by the passing focused tests and offline probe, but these flights do not establish successful recovery or course completion. The traces do not establish the precise missing free cells or prove that distant valid returns occurred during the relevant scan.

## Inputs

Exact archive: `50978085fedb6b3f56e1b23cb904167c15383eec`, 159 files hash-verified inside each isolated instance. Same DIVERSE01/DIVERSE02 worlds, original unchanged policy prior, selection seed `1049038789930006439`, hardware sensor profile, normal impairments, and existing compute budget. Actual episode world, collision-transform and live viewer scene/course hashes match the preceding comparisons exactly. No world-specific policy goal/map override was supplied.

Episodes:

- DIVERSE01: `93035d0e53c94299add889a44f026596`.
- DIVERSE02: `8ae7f01774ed46a0a9d96d9faefe0e44`.

Focused tests passed 36 cases after the active drone package copy succeeded. The parent reported full-suite validation of 576 passed and 2 expected failures. The parent subsequently confirmed both drone package locations matched all 159 files; native device checks are recorded separately by the device-staging agent.

## Startup

| Stage | DIVERSE01 | DIVERSE02 |
|---|---:|---:|
| Course/map loading | 66.9 wall s | 72.6 wall s |
| Navigator initialization/preplanning | 61 wall s | 63 wall s |
| Policy start → RefNav launch | 204.8 wall s | 212.1 wall s |
| Batch creation → first native RefNav status | 245.435 wall s | 254.089 wall s |

Readiness-gated pilots were queued for the new episode IDs. No readiness or safety checks were bypassed.

## Evidence completeness

The auxiliary observer initially failed because its copied script referenced a nonexistent filename. Intermittent SSH connectivity delayed correction until after flight. The retained `run-*.jsonl` files therefore contain **postflight observations only**, not flight telemetry. No inference about flight-wide behavior is made from those files.

Independent native policy recording was intact: DIVERSE01 has 3,126 trace records, DIVERSE02 2,928, both with zero reported dropped records. These traces include policy ticks/status and full navigator path messages, so recovery requests, planner failures, prior clearing/confirmation counts and plan modes are derived from the native traces. Native navigator logs, matching-episode grader scorecards, localizer CSVs and final live viewer snapshots are also retained. Continuous external ground-truth/localizer topic capture during flight is missing; this limitation is not concealed by the native trace recovery.

`results.json` identifies each evidence source and contains scorecards, phase transitions, path metrics, startup measurements and final disarmed snapshots. Large native traces are gzip-compressed. Only scorecards matching the two current episode IDs are retained. Pilot services and auxiliary capture were stopped after natural disarming; viewers remain available.
