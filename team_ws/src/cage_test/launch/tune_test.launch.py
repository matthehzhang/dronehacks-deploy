"""Tuning flights: hop_test.launch.py with the tuning sequence and 20 Hz setpoints.

    ros2 launch cage_test tune_test.launch.py                       # hover,steps,speed,yaw
    ros2 launch cage_test tune_test.launch.py sequence:=hover       # one pattern per flight
    ros2 launch cage_test tune_test.launch.py sequence:=yaw yaw_rates_dps:=30,60
    ros2 launch cage_test tune_test.launch.py sequence:=speed speed_list_mps:=0.3,0.6,0.9

Every argument of hop_test.launch.py works here too. 20 Hz (instead of 10)
halves the stair-step of the moving setpoint at the sweep's top speed.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

TUNE_DEFAULTS = [
    ('sequence', 'hover,steps,speed,yaw', 'patterns, in order'),
    ('publish_rate_hz', '20.0', 'setpoint publish and log rate, Hz'),
    ('setpoint_rate_hz', '20.0', 'flight_interface heartbeat / setpoint rate, Hz'),
]


def generate_launch_description():
    hop = os.path.join(get_package_share_directory('cage_test'), 'launch',
                       'hop_test.launch.py')
    # Declared first, so the included file's own defaults do not apply to these.
    return LaunchDescription(
        [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in TUNE_DEFAULTS]
        + [IncludeLaunchDescription(PythonLaunchDescriptionSource(hop))])
