"""One organizer interface, host depth, and measured-lane diagnostic. No auto-arm."""
from pathlib import Path
import runpy

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    arguments = runpy.run_path(str(Path(__file__).with_name('live.launch.py')))['ARGUMENTS']
    arguments = [a for a in arguments if a[0] not in ('mission', 'host_depth')]
    arguments += [('tuning_config', '', 'Required JSON with measured course-frame lane and open end zones'),
                  ('log_directory', '~/team_ws/logs/tuning', 'Fresh per-run JSONL recording directory'),
                  ('imu_topic', '/oak/imu/data', 'Raw calibrated IMU; capture timestamps must use ROS clock')]
    declared = [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in arguments]
    interface_keys = ('max_altitude_m', 'min_altitude_m', 'max_speed_mps', 'setpoint_rate_hz',
                      'auto_offboard_on_arm', 'odometry_topic')
    interface = Node(package='skybeta_flight_interface', executable='flight_interface', output='screen',
                     parameters=[{**{k: LaunchConfiguration(k) for k in interface_keys},
                                  'allow_mode_request': False}])
    mission_keys = [n for n, _, _ in arguments if n not in (*interface_keys, 'allow_mode_request', 'depth_max_hz')]
    mission = Node(package='kestrel_nav', executable='tune_mission', output='screen',
                   parameters=[{k: LaunchConfiguration(k) for k in mission_keys}])
    depth = Node(package='kestrel_nav', executable='host_depth', output='screen', prefix='nice -n 5',
                 parameters=[{'depth_topic': LaunchConfiguration('depth_topic'),
                              'max_hz': LaunchConfiguration('depth_max_hz')}])
    return LaunchDescription(declared+[interface, depth, mission])
