"""flight_interface + the cage flight test node (hop_test / tune_test).

Run it from the stack by setting, in /etc/skybeta/drone.env,

    COMPETITOR_PKG=cage_test
    COMPETITOR_LAUNCH=hop_test.launch.py      # or tune_test.launch.py

and restarting the stack (``docker rm -f skybeta-stack``), or by hand after
stopping the ``competitor`` window (two flight interfaces fight):

    ros2 launch cage_test hop_test.launch.py
    ros2 launch cage_test hop_test.launch.py forward_m:=0.5 touchdown:=false
    ros2 launch cage_test hop_test.launch.py sequence:=hop,hover

Lists are comma strings: ``speed_list_mps:=0.3,0.6``. Every run (arm ->
disarm) is logged under ``log_dir`` (``~/team_ws/runs`` on the host) with a
rosbag2 mcap: see the package README.

``flight_interface`` is started exactly as in competitor_example's
``competitor.launch.py``, with the ceiling lowered to 2.0 m for the cage and
``anchor_setpoints`` exposed.
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
TEST_ARGUMENTS = [
    ('sequence', 'hop',
     'comma list of patterns flown between takeoff and landing: '
     'hop, hover, steps, speed, yaw, vertical, square'),
    ('climb_height_m', '1.0', 'height of every pattern above the start, metres'),
    ('forward_m', '1.0', 'hop: distance flown along the nose and back, metres'),
    ('low_height_m', '0.5',
     'height the descent ends at before touchdown; keep above min_altitude_m'),
    ('speed_mps', '0.3', 'speed of the hop, takeoff, landing and square legs, m/s'),
    ('hold_seconds', '3.0', 'hold at the end of each leg, s'),
    ('arrival_radius_m', '0.2', 'how close counts as reaching a leg end, m'),
    ('abort_distance_m', '0.6',
     'stop the sequence (hold, pilot lands) when this far off the path, m'),
    ('touchdown', 'true',
     'true: descend from low_height_m until PX4 lands and disarms; '
     'false: hold at low_height_m for the pilot to land'),
    ('descend_speed_mps', '0.4', 'touchdown descent rate, m/s (>= 0.3)'),
    ('publish_rate_hz', '10.0', 'setpoint publish and log rate, Hz'),
    ('hover_seconds', '20.0', 'hover: hold time, s'),
    ('step_m', '0.5', 'steps: horizontal step size (forward, left), m'),
    ('step_z_m', '0.3', 'steps: vertical step size (up, down), m'),
    ('step_hold_seconds', '5.0', 'steps: hold after each step, s'),
    ('speed_list_mps', '0.25,0.5,0.75,1.0', 'speed: the sweep, slowest first, m/s'),
    ('speed_distance_m', '1.0', 'speed: length of each out and back leg, m'),
    ('speed_axes', 'forward', 'speed: forward, left, or forward,left'),
    ('yaw_rates_dps', '30,60,90,120', 'yaw: spin rates, slowest first, deg/s'),
    ('yaw_turn_deg', '360.0', 'yaw: turn per spin, each way, deg'),
    ('vertical_speeds_mps', '0.2,0.4,0.6', 'vertical: up/down sweep speeds, m/s'),
    ('vertical_span_m', '0.4', 'vertical: +/- this around climb_height_m, m'),
    ('square_m', '1.0', 'square: side length (forward, then left), m'),
    ('log_runs', 'true', 'write a run directory per arm -> disarm'),
    ('log_dir', 'auto', 'auto = /root/team_ws/runs (host ~/team_ws/runs)'),
    ('record_bag', 'true', 'rosbag2 mcap of every /fmu topic etc. per run'),
    ('record_images', 'false', 'also record both mono images (~39 MB/s)'),
]


def generate_launch_description():
    declared = [
        DeclareLaunchArgument(name, default_value=default, description=help_)
        for name, default, help_ in INTERFACE_ARGUMENTS + TEST_ARGUMENTS
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

    test_node = Node(
        package='cage_test',
        executable='hop_test',
        name='hop_test',
        output='screen',
        emulate_tty=True,
        parameters=[{name: LaunchConfiguration(name)
                     for name, _, _ in TEST_ARGUMENTS}],
    )

    return LaunchDescription(declared + [flight_interface, test_node])
