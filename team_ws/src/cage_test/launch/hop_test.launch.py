"""flight_interface + the cage hop test (up, forward, back, down).

Run it from the stack by setting, in /etc/skybeta/drone.env,

    COMPETITOR_PKG=cage_test
    COMPETITOR_LAUNCH=hop_test.launch.py

and restarting the stack (``docker rm -f skybeta-stack``), or by hand after
stopping the ``competitor`` window (two flight interfaces fight):

    ros2 launch cage_test hop_test.launch.py
    ros2 launch cage_test hop_test.launch.py forward_m:=0.5 touchdown:=false

Write numbers with a decimal point (``forward_m:=1.0``, not ``1``): an
integer fails the node's float parameter declaration at startup.

``flight_interface`` is started exactly as in competitor_example's
``competitor.launch.py``, with
the ceiling lowered to 2.0 m for the cage and ``anchor_setpoints`` exposed.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# name -> (default, help)
INTERFACE_ARGUMENTS = [
    ('max_altitude_m', '2.0', 'ceiling above the odometry origin, metres (cage)'),
    ('min_altitude_m', '0.3', 'floor above the odometry origin, metres'),
    ('max_speed_mps', '1.5', 'cap on the norm of a twist setpoint, m/s'),
    ('setpoint_rate_hz', '10.0',
     'OffboardControlMode heartbeat rate -- PX4 needs > 2 Hz'),
    ('auto_offboard_on_arm', 'true',
     'enter Offboard once when the pilot arms, then switches to Position'),
    ('odometry_topic', '/visual_slam/tracking/odometry',
     'source of the ENU pose the altitude band is measured against'),
    ('anchor_setpoints', 'false',
     'true only if frame_check shows a PX4-VIO position offset over ~0.1 m'),
]
HOP_ARGUMENTS = [
    ('climb_height_m', '1.0', 'height of the hop above the start, metres'),
    ('forward_m', '1.0', 'distance flown along the nose and back, metres'),
    ('low_height_m', '0.5',
     'height the descent ends at before touchdown; keep above min_altitude_m'),
    ('speed_mps', '0.3', 'speed the setpoint moves along each leg, m/s'),
    ('hold_seconds', '3.0', 'hold at the end of each leg, s'),
    ('arrival_radius_m', '0.2', 'how close counts as reaching a leg end, m'),
    ('touchdown', 'true',
     'true: descend from low_height_m until PX4 lands and disarms; '
     'false: hold at low_height_m for the pilot to land'),
    ('descend_speed_mps', '0.4', 'touchdown descent rate, m/s (>= 0.3)'),
]


def generate_launch_description():
    declared = [
        DeclareLaunchArgument(name, default_value=default, description=help_)
        for name, default, help_ in INTERFACE_ARGUMENTS + HOP_ARGUMENTS
    ]

    flight_interface = Node(
        package='skybeta_flight_interface',
        executable='flight_interface',
        name='flight_interface',
        output='screen',
        emulate_tty=True,
        parameters=[{name: LaunchConfiguration(name)
                     for name, _, _ in INTERFACE_ARGUMENTS}],
    )

    hop_test = Node(
        package='cage_test',
        executable='hop_test',
        name='hop_test',
        output='screen',
        emulate_tty=True,
        parameters=[{name: LaunchConfiguration(name)
                     for name, _, _ in HOP_ARGUMENTS}],
    )

    return LaunchDescription(declared + [flight_interface, hop_test])
