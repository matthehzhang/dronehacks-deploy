# Diverse followup environments

The existing L01–L20 catalog uses one `official_serpentine_chicane` topology, with wall jitter and global rotations/reflections. Inspection of actual Gazebo collision transforms and viewer scene assets confirmed that L10/L09 loaded distinct geometry, but their underlying corridor arrangement remained the same family. The followup L10/L09 batch was superseded at the user's request; it is not a completed evaluation.

`world-layouts.png` shows the replacement pair. These are seeded generic boundary chains with all ten original panels, three maze dividers, five boulders, the tube, window, gate and landing pad in the original enclosure. Tube placement, chicane connectivity, divider anchors/angles, gate and pad differ. The deployed policy prior is unchanged.

Generation is isolated in `sim/tools/procgen/gen_diverse_layouts.py`; it does not change the default generator or policy. Candidate seeds increment from 918246100. The timed search records its exact attempt count; passing that count via `--attempts` reproduces the candidate set. Four candidates passed full inventory, existing dense open-area bounds, existing route/window clearance, sealed tube barrier, sealed pad pocket, launch clearance, and minimum 10 m route. The pair maximizes wall-centre RMS separation after global rotations/reflections are removed; no flight outcome influenced selection. Their blocked-area Jaccard distance remains 0.338 after the best global rotation/reflection alignment.

| World | Generator seed | Route | Minimum route clearance | Window clearance |
|---|---:|---:|---:|---:|
| DIVERSE01 | 918246141 | 11.514 m | 0.474 m | 0.352 m |
| DIVERSE02 | 918247045 | 11.163 m | 0.434 m | 0.401 m |

The original reference route is 18.414 m. These runs test behavior under different geometry; their route lengths are only 62.5% and 60.6% of the reference, so scores/times are not a matched-difficulty comparison. Full inventory and existing density/clearance checks were preserved. A bounded search did not find a pair meeting the original 90% route-length condition. The parent explicitly authorized prioritizing diverse valid bugfinding worlds over matching route length.

Assets were built using the existing builder, with world collision/visual transforms and grader geometry generated from the same catalog entries. No oracle map voxels were generated or supplied to the policy. The exported rounded geometry was independently reconstructed and revalidated; results are in `selection.json`. Geometry checks are not evidence of successful 3D flight.

Both runs tested commit `c52cb7c` from `cram304`, archived before synchronization. All 153 archived policy files were hash-verified in each isolated instance. Sensor profile: hardware. Impairments: normal. Original policy prior SHA-256: `ef86d5ea34edb7d655551ad2faf5bedff6c916425013aeb4e03e927e0b46809b`.

## Results

Both runs failed to finish, landed away from the target, and disarmed naturally. No manual LAND command or policy/parameter changes were made during the runs. Both pilot services and telemetry capture were stopped after disarming; viewers remain available.

| World | Grader duration | Checkpoints / points | Obstacle contacts | Minimum obstacle clearance | Final result |
|---|---:|---:|---:|---:|---|
| DIVERSE01 | 24.450 sim s | 0 / 0 | 0 | 0.5631 m | Abort landing; `crash_ground_contact` |
| DIVERSE02 | 17.425 sim s | 0 / 0 | 0 | 0.5422 m | Abort landing; `crash_ground_contact` |

The grader's ground-contact result corresponds to the policy's deliberate off-target abort landing. Both RefNav logs say `landed: corridor of leg 1 not observed for 10.0 s (corridor_wait_land_s 10 s)`. The scorecards, policy traces, navigation logs and captured truth telemetry are retained.

Both prior monitors transitioned unknown→dropped before flight. Navigator initially published prior-derived paths, then repeatedly reported no path to the unchanged prior's first target. Trace samples show `plan_reply.state=failed` with `awaiting bounded planner retry`, followed by path holds and the corridor timer landing. There were **zero recovery requests and zero recoveries**. During failed-plan holds, depth-valid fraction remained 0.913–0.914 (DIVERSE01) and 0.887–0.894 (DIVERSE02), and maximum recorded depth age was 0.296/0.346 s. No Offboard-loss transition appears in the airborne policy/flight-interface logs.

The source path matches the observed missing recovery: `refnav/core.py:on_plan_result` clears the path on pending/failed replies; `_go_path` returns immediately when no fresh path exists, before its clearance/obstacle branch. The new obstacle recovery scan therefore was not requested in these traces. This identifies an unhandled normal-planner-failure path, not proof that scanning alone would make either mission solvable. Retained prior constraints, unchanged target locations, local occupancy, and limited observed-free space can still prevent a valid path. The tests do not certify completion on arbitrary geometry.

## Startup evidence

| Stage | DIVERSE01 | DIVERSE02 |
|---|---:|---:|
| Map ready, from policy start | 71.4 wall s | 66.6 wall s |
| Navigator constructor/preplanning wait | 59 wall s | 55 wall s |
| Policy start → RefNav launch | 203.3 wall s | 187.8 wall s |
| Batch creation → first captured RefNav status | 242.808 wall s | 227.517 wall s |

The pilot gates were active and correctly waiting for current RefNav arm phase and a fresh hold setpoint; there was no missing arm command. Navigator CPU use was about 8.2/9.2 CPU seconds across roughly 60/70 wall seconds. At the sampled preflight point, the existing companion CPU quotas were 0.5829/0.6299 CPUs, with 10327/10361 and 10322/10362 quota periods throttled. This is the existing RTF-scaled synthetic compute budget shared by depth, localization and planning; its allocation was not changed. Full snapshot values are in `companion-budget-*.txt`. Startup progressed and completed without a constructor exception.

`loaded-world-check.json` records the actual running episode files, collision-transform hashes, live viewer scene/course hashes, and per-instance policy verification. `results.json` provides compact outcomes, startup timestamps, prior/phase transitions and freshness checks. The new scenes were verified through each running viewer endpoint, not inferred from map IDs alone.

Large telemetry JSONL and flight traces are gzip-compressed (`*.jsonl.gz`); compact results and matching episode scorecards remain plain JSON. Historical scorecards were removed only from the copied evidence, leaving original simulator records intact.
