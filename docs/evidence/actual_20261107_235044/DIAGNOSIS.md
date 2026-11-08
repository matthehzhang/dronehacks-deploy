# New-origin group154 hardware recording: incomplete flight coverage

The available recording is `20261107-235044_group154_OvyGUo/bag/bag_0.mcap`, SHA256 822b30dc048fddc5fab898b249e4bd84e04eb273c6851b634dfdb01abd702a8f. It spans 74.988 s including stack startup. It ends only 0.167 s after `arm -> climb` (74.821 s). Last interface status is armed=true, landed=true, OFFBOARD; the turn, forward refusal and landing described by the pilot are not recorded. No failure cause or disarm can be inferred for that missing interval.

## Supported observations

- The recorded map NPZ hash f0c73aeb0b459707c4f0f5a0e23b4e23375991eed36fce4ee5cacc61e6569d1e matches the simulator-tested new-course map exactly. An accidentally unchanged original-map NPZ is not supported by this evidence. Physical venue agreement/launch alignment are not established by a matching file.
- Recorded refnav/core.py hash dc4c5c7d... and gate.py ab0436d8... DIFFER from the simulator-tested source (42eb15cd... and b35d611a...). LF/CRLF comparisons do not explain the difference. Actual source bytes were not included, so the behavioral difference cannot yet be identified. Recorded course configuration also adds align_window_max_dz_m=.30 and an updated alignment note; the rest of course configuration matches.
- Navigator successfully preplanned first leg 5.66 m/two stop corners, plus legs 3 and 5. This rules out startup A* failing to find any route; it does not establish a live opening or physical clearance.
- Final localizer status: 285 scans, 285 skipped, zero updates. Final map: 285 scans in, 285 gated, zero known/occupied cells. Initial fit 5/29 (~.172); final fit 1/27 (~.037); configured min fit fraction is .35 in the tested source. Sigma yaw remains about2.07deg versus live-map 2deg gate. Thus mapping/localization were not updating before takeoff. No raw occupancy topic was recorded; these facts come from localizer status counters.
- Host stereo depth was running about7–8Hz, 63–68% full-frame valid, 37–78ms reported processing. These aggregate values do NOT prove enough depth along the specific commanded corridor after turning.
- TF initially unavailable for one refnav frame, then calibration succeeded (base_link <- oak_left_rect_frame, t~.106,.038,.031; optical axis~.5deg off forward). No evidence this resolved startup message caused the later unrecorded landing.
- Healthy VIO/PX4 Offboard observed at bag end. No later authority loss or reset can be evaluated.

## Hypotheses requiring the missing flight interval

A specific corridor remaining unobserved for10s, a confirmed close obstacle/backoff failure, estimator/authority loss, wrong route bearing from map-start alignment, or different deployed control code could produce turn/hold/land. None is proven by this truncated recording. The successful sim used approximate entrance geometry and a simulated sensor/error profile, so it cannot validate the real launch transform, depth texture/occlusion or actual deployed source.

## Needed evidence

The complete flight bag or refnav flight_trace.jsonl and navigator/localizer logs spanning turn through landing, plus actual deployed core.py/gate.py/launch/config source. Record /refnav/status,/refnav/control,/coursenav/path,/coursenav/occupancy,/coursenav/status,/coursenav/drift,/kestrel_nav/depth and CameraInfo, TF, authority/estimator topics and commands. Raw stereo/mono images enable independent depth/calibration inspection. Do not weaken fit/depth/authority checks based on the incomplete bag.
