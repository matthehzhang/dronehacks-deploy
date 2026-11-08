# Depth range investigation

The range limit can cause a stop followed by landing through loss of observed
space. It does not turn missing returns into a nearby wall in this pipeline.
No behavior was changed for this investigation.

Reproduce from the deploy repository root:

```bash
python3 docs/evidence/blocked_recovery_20261108/range_probe.py --output docs/evidence/blocked_recovery_20261108/range_probe.json
```

The script records source hashes,
uses the actual deployed depth decoder/core, and extracts `HostDepth.process` from
its AST. Only image transport and stereo matching are replaced with synthetic
inputs. This isolates return handling; it does not validate real stereo accuracy.

## Results

| Input | Core result |
|---|---|
| Zero, NaN, +inf | Unknown: no valid pixels, no obstacle points, frame refused |
| Finite 8 m | Observed, no obstacle points inside the 6 m obstacle horizon |
| 8 m through host max 6 m | Zeroed by host filtering; unknown |
| 8 m through host max 12 m | Preserved; observed |
| 13 m through host max 12 m | Zeroed; unknown |
| False finite 0.6 m plane, three frames | Confirmed obstacle stop |
| False finite 0.9 or 1 m plane | Real obstacle points, but beyond this probe's 0.75 m stop threshold |
| External uint16 sentinel 65535 | Decodes to 65.535 m and counts as observed; not a near wall |

The probe uses the new course's base profile, a 5 m straight leg with 4 m remaining,
and nominal 0.3 m/s. Stop distance is `max(0.55, 0.45 + speed)`, hence 0.75 m here.
Actual leg speeds and path geometry affect the threshold and along-path clearance.
The external 65535 case is a decoder contract caveat, not a demonstrated live fault:
the current host filter rejects depths beyond 12 m before encoding, so it cannot
produce that saturation value from its accepted depth range.

## Supported mechanisms, ranked

1. **Range/texture losses cause unknown space and can trigger a safety landing.**
   `host_depth.py:234-239` rejects disparity <=0.5 and depth outside the strict
   min/max interval, encoding rejected pixels as zero. `refnav/depth.py:37-49`
   turns zero/nonfinite values into NaN. `core.py:555-563` refuses frames below
   the valid fraction threshold (0.30 translating, 0.15 static). Far finite
   returns count toward visibility although obstacle extraction uses only
   0.3 < depth < 6 m (`depth.py:72-91`). Corridor coverage must reach 0.45
   (`core.py:1108-1124`). Without fresh observations, motion stops; the profile
   has a 10 s corridor-observation deadline and 5 s usable-depth deadline
   (`core.py:797-813`). Static-phase pauses and qualified tunnel mode retain
   their existing exceptions.
2. **Finite incorrect stereo matches can be obstacles.** A false near plane is
   not distinguishable from a real plane by range filtering alone. The core
   requires enough points and three confirming observed frames. The 0.6 m
   probe demonstrates a stop without modifying those rules. This is a plausible
   source of real/sim divergence, not proof of what happened in the real flight.
3. **The sim's optional 6 m OAK emulator does not constrain the current policy.**
   `sim/scripts/run_stack.sh:97-101` enables it only with `OAK_DEPTH=1`, and it
   publishes `/oak/stereo/image_raw`. The policy launch always runs host stereo
   on `/kestrel_nav/depth`, with max range 12 m. Its rendered mono cameras see
   to 50 m (`sim/models/oak_d_s2/model.sdf:58,84`). Rendered textures/noise and
   calibration may give much better or different host-SGBM matches than reality.
   An emulator existing in the sim does not prove the policy consumes it.
4. **The downward VL53L1X is a different sensor.** It supplies height rather than
   forward obstacle depth. Sky Beta localizer height uses plausible PX4
   `dist_bottom` (0.02-3 m), else landed height minus NED z
   (`coursenav/loc_node.py:228-238`). It cannot impose a forward lidar maximum.

The organizer repository camera configuration intentionally enables no on-device
StereoDepth block (`src/oakd_vslam_bringup/config/oakd_vslam.yaml:9-15`). The current
forward-depth source is the team's host SGBM on OAK mono images. The parent agent
also confirmed this on the live Jetson.

## What remains unproven

Existing real-bag notes report repeated finite near blocks and poor scan fit, but
that bag did not record depth (`workspace docs/LOCAL_MAP.md:175`). A range cap alone
cannot manufacture the finite near points. Distinguishing false matches, camera
calibration/TF error, a real obstacle, and path-check differences requires captured
depth + CameraInfo + TF alongside refnav rejection/coverage/clearance status. No
threshold increase or conversion of unknown space to free space is justified by
this probe.

## Direct check of the retained real bag

`extract_real_rosout.py` reads the locally recovered MCAP segment of
`20261107-201744_bag`; extracted messages are in `real_bag_range_logs.json`.
This retained segment contains 16 `go -> blocked` transitions with finite
obstacles 0.84–0.95 m ahead. Host-depth log samples show 63–80% valid pixels.
Its final transition is `yielded: PX4 left Offboard (nav_state 2)`, not a
range/coverage timeout landing. This does not establish whether the near hits
were real obstacles or stereo artifacts: no depth images were recorded.
The recovered segment may be incomplete and does not represent every flight.
