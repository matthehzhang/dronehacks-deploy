# Prior-correction integration comparison

Both runs of `9b76afd` failed to finish. The new recovery trigger operated, but neither run obtained a certified detour or accepted a local-only fallback. Both executed bounded recovery, landed away from the target, and disarmed naturally. No manual flight intervention or policy/configuration changes occurred during the runs.

| World | c52cb7c duration | 9b76afd duration | New checkpoints / points | Grader result |
|---|---:|---:|---:|---|
| DIVERSE01 | 24.450 sim s | 31.175 sim s | 0 / 0 | `crash_ground_contact` |
| DIVERSE02 | 17.425 sim s | 18.900 sim s | 0 / 0 | `crash_ground_contact` |

These durations end at off-target abort landings; longer duration is not evidence of progress or success. Both had zero obstacle contacts; minimum obstacle clearance was 0.5425/0.5466 m. The grader's ground-contact label corresponds to the policy's deliberate abort landing.

The previous version held after a normal planner failure until the corridor timer expired, with no recovery request. This version requested bounded recovery and scanned/rechecked after losing a route. Each run emitted recovery request IDs 1–5, with zero certified-ready replies. The navigator rejected attempts with `no observed-free rejoin beyond recovery lookahead`. DIVERSE01 also reported `recovery start footprint is not certified free` during normal fallback planning. Those messages identify the failed certificates; no unrecorded geometric cell cause is inferred.

Prior clearing was exercised in DIVERSE01 (maximum 14 prior cells cleared and 42 confirmed during airborne phases); DIVERSE02 cleared none. Only `fused` paths were published, with no accepted `local_only` path or certified recovery resume. DIVERSE01's prior monitor briefly restored `placed` during the scan before dropping again. The unchanged target/prior geometry and observed-space restrictions remain constraints; these results do not establish completion on arbitrary courses.

## Inputs and verification

- Exact archived policy: `9b76afd`; all 158 committed files matched inside each isolated instance.
- Device was updated and verified before tests. Focused tests: 49 passed. Parent full suite: 545 passed, 2 expected failures.
- Same DIVERSE01/DIVERSE02 geometry as the previous `c52cb7c` comparison; same selection seed `1049038789930006439`, hardware sensor profile, normal impairments, and original policy prior. No course override was injected.
- Actual episode world files, collision-transform hashes, live viewer scene/course hashes, and policy hashes were verified. Their hashes exactly match the corresponding earlier worlds and differ between worlds; see `loaded-world-check.json`.
- Episodes: DIVERSE01 `04709ebf53ed45a58f3e08ed52c8ae9a`; DIVERSE02 `be7fa2f68b3849ca891044d3d12d24f3`.

## Startup and evidence

Map loading took 68.8/60.5 wall seconds; navigator initialization/preplanning took 56/47 seconds. RefNav launched 194.8/167.2 seconds after policy start. Gated pilots were queued promptly for the new episode IDs and waited for actual RefNav readiness. The existing RTF-scaled companion CPU budget was preserved; snapshots are in `companion-budget-*.txt`.

`results.json` contains final scorecards, phase/prior transitions, recovery request IDs, clearing/confirmation maxima, plan modes, failure messages, and exact startup measurements. Only scorecards matching the current episode IDs are retained. Telemetry includes `/coursenav/path` in addition to policy/localizer/prior/flight-interface/grader/truth topics. Earlier-batch shutdown messages were filtered by the new batch timestamp. Large telemetry and flight traces are gzip-compressed. Pilot services and capture were stopped after natural disarming; viewers remain available.
