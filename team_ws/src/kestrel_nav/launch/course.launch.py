"""skybeta flight interface + the kestrel_nav course mission (+ optional localizer).

Same shape as competitor_example/competitor.launch.py: ``flight_interface`` is kept exactly as the
example starts it (auto_offboard_on_arm:=true: Offboard after the pilot arms, then switches to
Position), and the competitor node is swapped for ``course_mission``.

    ros2 launch kestrel_nav course.launch.py                       # plan from the course-file start prior
    ros2 launch kestrel_nav course.launch.py start_x:=-2.51 start_y:=1.11 start_yaw_deg:=90.0
    ros2 launch kestrel_nav course.launch.py localizer:=true       # also run coursenav.loc_node (needs depth)
    ros2 launch kestrel_nav course.launch.py mission:=false        # flight interface only

Point the stack at it from /etc/skybeta/drone.env:
    COMPETITOR_PKG=kestrel_nav
    COMPETITOR_LAUNCH=course.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
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
    ('setpoint_rate_hz', '10.0', 'OffboardControlMode heartbeat rate -- PX4 needs > 2 Hz'),
    ('allow_mode_request', 'false', 'SIM ONLY: enable /skybeta/request_offboard'),
    ('auto_offboard_on_arm', 'true', 'enter Offboard once when the pilot arms, then switches to Position'),
    ('odometry_topic', '/visual_slam/tracking/odometry', 'ENU pose source for the altitude band'),
    # --- course mission ---
    ('mission', 'true', 'start course_mission'),
    ('chicane', 'auto', 'auto | window | gate_b'),
    ('start_x', 'nan', 'start pose in the COURSE frame; nan = course-file prior (launch mark)'),
    ('start_y', 'nan', ''),
    ('start_yaw_deg', 'nan', ''),
    ('landed_z_m', '0.15', 'body origin above the floor when landed (MEASURE on skybeta-14)'),
    ('arrival_radius_m', '0.25', 'how close counts as reaching a waypoint'),
    ('hold_seconds', '1.0', 'pause at gate/tunnel/align waypoints'),
    ('use_localizer', 'false', 'refine the course->odom transform from /coursenav/pose while flying'),
    ('leg_timeout_s', '45.0', 'give up a leg (hold) after this long'),
    # --- localizer ---
    ('localizer', 'false', 'also run kestrel_nav localizer (coursenav.loc_node); needs a depth topic'),
]


def generate_launch_description():
    declared = [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in ARGUMENTS]
    share = get_package_share_directory('kestrel_nav')

    flight_interface = Node(
        package='skybeta_flight_interface', executable='flight_interface', name='flight_interface',
        output='screen', emulate_tty=True,
        parameters=[{
            'max_altitude_m': LaunchConfiguration('max_altitude_m'),
            'min_altitude_m': LaunchConfiguration('min_altitude_m'),
            'max_speed_mps': LaunchConfiguration('max_speed_mps'),
            'setpoint_rate_hz': LaunchConfiguration('setpoint_rate_hz'),
            'allow_mode_request': LaunchConfiguration('allow_mode_request'),
            'auto_offboard_on_arm': LaunchConfiguration('auto_offboard_on_arm'),
            'odometry_topic': LaunchConfiguration('odometry_topic'),
        }],
    )

    mission = Node(
        package='kestrel_nav', executable='course_mission', name='course_mission',
        output='screen', emulate_tty=True,
        condition=IfCondition(LaunchConfiguration('mission')),
        parameters=[{
            'chicane': LaunchConfiguration('chicane'),
            'start_x': LaunchConfiguration('start_x'),
            'start_y': LaunchConfiguration('start_y'),
            'start_yaw_deg': LaunchConfiguration('start_yaw_deg'),
            'landed_z_m': LaunchConfiguration('landed_z_m'),
            'arrival_radius_m': LaunchConfiguration('arrival_radius_m'),
            'hold_seconds': LaunchConfiguration('hold_seconds'),
            'use_localizer': LaunchConfiguration('use_localizer'),
            'leg_timeout_s': LaunchConfiguration('leg_timeout_s'),
            'min_altitude_m': LaunchConfiguration('min_altitude_m'),
            'max_altitude_m': LaunchConfiguration('max_altitude_m'),
        }],
    )

    localizer = Node(
        package='kestrel_nav', executable='localizer', name='coursenav_localizer',
        output='screen', emulate_tty=True,
        condition=IfCondition(LaunchConfiguration('localizer')),
        parameters=[os.path.join(share, 'config', 'localizer.yaml')],
    )

    return LaunchDescription(declared + [flight_interface, mission, localizer])
