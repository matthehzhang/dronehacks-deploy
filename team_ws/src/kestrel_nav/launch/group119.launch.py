"""group119 (workspace origin/group119 @ 749bca0) flying on skybeta: refnav + coursenav + fly_course.py,
vendored unchanged, with refnav's px4_output:=skybeta (twist through the organizer flight interface).

    ros2 launch kestrel_nav group119.launch.py
    ros2 launch kestrel_nav group119.launch.py chicane:=gate_b

/etc/skybeta/drone.env:  COMPETITOR_PKG=kestrel_nav  COMPETITOR_LAUNCH=group119.launch.py

Start-up order on the drone: drone on the launch mark, stack up. Wait for refnav's "arm" phase in the
log (it streams a hold setpoint; the interface needs a team setpoint within 1 s of the switch), THEN the
pilot arms and switches to Position. refnav accepts the armed vehicle (accept_armed_start), climbs and flies.
"""
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ROOT = Path(__file__).resolve().parent.parent          # package source root (symlink install): fly_course.py lives here

ARGUMENTS = [
    ('max_altitude_m', '3.0', 'flight interface ceiling above the odometry origin'),
    ('min_altitude_m', '0.3', 'flight interface floor (pose setpoints only)'),
    ('max_speed_mps', '1.5', 'flight interface cap on a twist norm'),
    ('auto_offboard_on_arm', 'true', 'enter Offboard once when the pilot arms, then switches to Position'),
    ('chicane', 'auto', 'auto | window | gate_b'),
    ('depth_topic', '/kestrel_nav/depth', 'depth image for refnav and the localizer (host_depth)'),
    ('depth_info_topic', '/kestrel_nav/depth/camera_info', ''),
    ('depth_max_hz', '8.0', 'host_depth rate cap'),
    ('arm_wait_s', '600.0', 'refnav: how long the arm phase waits for the pilot to arm and switch to Position'),
    ('wait_s', '900.0', 'refnav: how long it waits for readiness before giving up'),
    ('landed_z', '0.10', 'landed body-origin height above the floor: PX4 dist_bottom 0.100 m on skybeta-14 (no distance_sensor topic there)'),
    ('localizer_timeout', '120.0', 'fly_course: how long to wait for the localizer to converge on the mark'),
    ('use_sim_time', 'false', 'true only in a simulator that publishes /clock'),
]


def generate_launch_description():
    declared = [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in ARGUMENTS]
    L = LaunchConfiguration
    env = dict(os.environ, PYTHONPATH=f"{ROOT}:{os.environ.get('PYTHONPATH', '')}")
    flight_interface = Node(
        package='skybeta_flight_interface', executable='flight_interface', name='flight_interface',
        output='screen', emulate_tty=True,
        parameters=[{'max_altitude_m': L('max_altitude_m'), 'min_altitude_m': L('min_altitude_m'),
                     'max_speed_mps': L('max_speed_mps'), 'setpoint_rate_hz': 10.0, 'allow_mode_request': False,
                     'auto_offboard_on_arm': L('auto_offboard_on_arm'),
                     'odometry_topic': '/visual_slam/tracking/odometry'}])
    host_depth = Node(package='kestrel_nav', executable='host_depth', name='host_depth', output='screen',
                      prefix='nice -n 5', parameters=[{'depth_topic': L('depth_topic'), 'max_hz': L('depth_max_hz')}])
    health = ExecuteProcess(cmd=['python3', '-m', 'refnav.skybeta_health', '--ros-args', '-p',
                                 ['use_sim_time:=', L('use_sim_time')]], cwd=str(ROOT), additional_env=env,
                            output='screen', name='skybeta_health')
    localizer = ExecuteProcess(
        cmd=['python3', '-m', 'coursenav.loc_node', '--ros-args', '--params-file', str(ROOT / 'onboard/config/common.yaml'),
             '-p', ['depth_topic:=', L('depth_topic')], '-p', ['depth_info_topic:=', L('depth_info_topic')],
             '-p', ['use_sim_time:=', L('use_sim_time')]],
        cwd=str(ROOT), additional_env=env, output='screen', name='localizer')
    fly_env = dict(env)
    fly = ExecuteProcess(
        cmd=['python3', str(ROOT / 'fly_course.py'), '--event-start', '--skip-preflight', '--no-own-localizer', '--no-return',
             '--chicane', L('chicane'), '--landed-z', L('landed_z'), '--localizer-timeout', L('localizer_timeout'), '--',
             '-p', 'px4_output:=skybeta', '-p', ['depth_topic:=', L('depth_topic')],
             '-p', ['depth_info_topic:=', L('depth_info_topic')],
             '-p', ['arm_wait_s:=', L('arm_wait_s')], '-p', ['wait_s:=', L('wait_s')],
             '-p', ['use_sim_time:=', L('use_sim_time')]],
        cwd=str(ROOT), additional_env={**fly_env, 'KESTREL_USE_SIM_TIME': L('use_sim_time')},
        output='screen', name='fly_course')
    return LaunchDescription(declared + [flight_interface, host_depth, health, localizer, fly])
