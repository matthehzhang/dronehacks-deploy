# Receding-horizon waypoint navigation

This package contains one movement architecture: global waypoint intent,
VIO/depth rolling occupancy, observed-space A*, smoothed local horizons and
its path follower. See DEPTH_HORIZON.md for behavior, limitations and validation.

Launch: `ros2 launch kestrel_nav refnav_skybeta.launch.py`.
Pilot-verified launch clearance is explicitly opt-in and defaults to zero.
Arm/disarm, estimator checks and the organizer flight interface are retained.
Origin geometry, prior matching, alternate missions and old launchers were deleted.
