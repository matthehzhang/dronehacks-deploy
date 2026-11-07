"""skybeta flight interface + the kestrel_nav LIVE mission (livenav: A* replanning on a depth grid).

Same shape as competitor_example/competitor.launch.py: ``flight_interface`` is kept exactly as the
example starts it (auto_offboard_on_arm:=true) and the competitor node is ``live_mission``.

    ros2 launch kestrel_nav live.launch.py
    ros2 launch kestrel_nav live.launch.py v_cruise:=0.5 depth_topic:=/oak/stereo/image_raw
    ros2 launch kestrel_nav live.launch.py start_x:=-2.51 start_y:=0.91 start_yaw_deg:=90.0
    ros2 launch kestrel_nav live.launch.py mission:=false        # flight interface only

Point the stack at it from /etc/skybeta/drone.env:
    COMPETITOR_PKG=kestrel_nav
    COMPETITOR_LAUNCH=live.launch.py
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGUMENTS = [
    # --- flight interface envelope (the interface's own defaults) ---
    ('max_altitude_m', '3.0', 'ceiling above the odometry origin, metres'),
    ('min_altitude_m', '0.3', 'floor above the odometry origin, metres'),
    ('max_speed_mps', '1.5', 'cap on the norm of a twist setpoint, m/s'),
    ('setpoint_rate_hz', '10.0', 'OffboardControlMode heartbeat rate'),
    ('allow_mode_request', 'false', 'SIM ONLY'),
    ('auto_offboard_on_arm', 'true', 'enter Offboard once when the pilot arms, then switches to Position'),
    ('odometry_topic', '/visual_slam/tracking/odometry', 'ENU pose source'),
    # --- live mission ---
    ('mission', 'true', 'start live_mission'),
    ('chicane', 'auto', 'auto | window | gate_b'),
    ('start_x', 'nan', 'start pose in the COURSE frame; nan = course-file prior (launch mark)'),
    ('start_y', 'nan', ''),
    ('start_yaw_deg', 'nan', ''),
    ('cruise_z_m', '0.8', 'body height above the floor in flight'),
    ('landed_z_m', '0.10', 'body origin above the floor when landed: PX4 dist_bottom read 0.100 m on skybeta-14, landed, 2026-11-07'),
    ('descend_to_z_m', '0.3', 'descend to this height above the odom origin over the pad (interface floor)'),
    ('v_cruise', '0.4', 'cruise speed, m/s'),
    ('v_gate', '0.25', 'speed through the tunnel and the window, m/s'),
    ('stop_m', '0.55', 'reactive stop distance along the path, m'),
    ('time_limit_s', '170.0', 'stop driving after this long in Offboard'),
    ('odom_max_age_s', '0.5', 'maximum odometry capture/receipt age; bench qualification required'),
    ('tracking_max_age_s', '0.5', 'maximum successful cuVSLAM status age'),
    ('depth_max_age_s', '0.5', 'maximum depth capture/receipt age'),
    ('state_max_age_s', '1.0', 'maximum flight-interface state age; then yield authority'),
    ('future_tolerance_s', '0.05', 'maximum future sensor timestamp tolerance'),
    ('qualify_s', '1.0', 'continuous healthy input interval before preflight zero twists'),
    ('min_depth_valid_fraction', '0.3', 'sampled depth validity floor; not hardware qualified'),
    ('min_corridor_coverage', '0.6', 'valid fraction of sampled motion-corridor rays'),
    ('min_corridor_rays', '12', 'minimum sampled rays viewing commanded motion corridor'),
    ('host_depth', 'true', 'run kestrel_nav host_depth (CPU stereo from the mono pair; on-device depth added 190 ms to VIO)'),
    ('depth_topic', '/kestrel_nav/depth', 'depth image (16UC1 mm or 32FC1 m)'),
    ('depth_info_topic', '/kestrel_nav/depth/camera_info', ''),
    ('depth_max_hz', '8.0', 'host_depth rate cap'),
    ('mono_topic', '/oak/left/image_mono', 'image for the pad ring detector'),
    ('mono_info_topic', '/oak/left/camera_info', ''),
    ('camera_from_tf', 'true', 'camera extrinsics from TF (skybeta publishes base_link <- oak_*); else camera_* params'),
    ('camera_x', '0.116', 'fallback camera position in body FLU, metres'),
    ('camera_y', '0.037', ''),
    ('camera_z', '0.029', ''),
    ('camera_pitch', '0.0', 'fallback mount pitch, rad (nose-down positive)'),
]


def generate_launch_description():
    declared = [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in ARGUMENTS]

    flight_interface = Node(
        package='skybeta_flight_interface', executable='flight_interface', name='flight_interface',
        output='screen', emulate_tty=True,
        parameters=[{k: LaunchConfiguration(k) for k in (
            'max_altitude_m', 'min_altitude_m', 'max_speed_mps', 'setpoint_rate_hz', 'allow_mode_request',
            'auto_offboard_on_arm', 'odometry_topic')}],
    )

    mission = Node(
        package='kestrel_nav', executable='live_mission', name='live_mission',
        output='screen', emulate_tty=True,
        condition=IfCondition(LaunchConfiguration('mission')),
        parameters=[{k: LaunchConfiguration(k) for k in (
            'chicane', 'start_x', 'start_y', 'start_yaw_deg', 'cruise_z_m', 'landed_z_m', 'descend_to_z_m',
            'v_cruise', 'v_gate', 'stop_m', 'time_limit_s', 'depth_topic', 'depth_info_topic', 'mono_topic',
            'mono_info_topic', 'camera_from_tf', 'camera_x', 'camera_y', 'camera_z', 'camera_pitch', 'min_altitude_m',
            'odom_max_age_s', 'tracking_max_age_s', 'depth_max_age_s', 'state_max_age_s', 'future_tolerance_s',
            'qualify_s', 'min_depth_valid_fraction', 'min_corridor_coverage', 'min_corridor_rays')}],
    )

    host_depth = Node(
        package='kestrel_nav', executable='host_depth', name='host_depth',
        output='screen', emulate_tty=True,
        condition=IfCondition(LaunchConfiguration('host_depth')),
        prefix='nice -n 5',
        parameters=[{'depth_topic': LaunchConfiguration('depth_topic'), 'max_hz': LaunchConfiguration('depth_max_hz')}],
    )

    return LaunchDescription(declared + [flight_interface, host_depth, mission])
