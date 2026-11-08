"""Skybeta VIO/depth receding-horizon waypoint mission."""
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ROOT = Path(__file__).resolve().parent.parent          # package source root (symlink install): fly_course.py lives here

ARGUMENTS = [
    ('pilot_verified_launch_radius_m', '0.0', 'OPT-IN: radius of pilot-verified free launch disc at cruise altitude; never overrides depth obstacles'),
    ('course_file', str(ROOT / 'course' / 'course_depth_waypoints.json'), 'depth-only waypoint and vehicle configuration'),
    ('max_altitude_m', '3.0', 'flight interface ceiling above the odometry origin'),
    ('min_altitude_m', '0.3', 'flight interface floor (pose setpoints only)'),
    ('max_speed_mps', '1.5', 'flight interface cap on a twist norm'),
    ('auto_offboard_on_arm', 'true', 'enter Offboard once when the pilot arms, then switches to Position'),
        ('depth_topic', '/kestrel_nav/depth', 'depth image for refnav and the localizer (host_depth)'),
    ('depth_info_topic', '/kestrel_nav/depth/camera_info', ''),
    ('depth_max_hz', '8.0', 'host_depth rate cap'),
    ('depth_max_range_m', '12.0', "host_depth range cap. refnav counts a corridor ray as OBSERVED if it returns any depth "
                                  "(its obstacle points stop at 6 m, the localizer's at 4 m): the hall beyond the "
                                  "see-through nets is 6-9 m away and at the old 6 m cap every such ray read unobserved"),
    ('arm_wait_s', '600.0', 'refnav: how long the arm phase waits for the pilot to arm and switch to Position'),
    ('wait_s', '900.0', 'refnav: how long it waits for readiness before giving up'),
    ('landed_z', '0.10', 'landed body-origin height above the floor: PX4 dist_bottom 0.100 m on skybeta-14 (no distance_sensor topic there)'),
    ('use_sim_time', 'false', 'true only in a simulator that publishes /clock'),
    ('declared_heading_deg', '0.0', "navigator: flight frame heading vs the VIO origin. 0 on skybeta (the organizer bridge "
                                     "permutes cuVSLAM's start-relative ENU odometry directly); 90 was the FlyX lab bridge"),
]


def generate_launch_description():
    declared = [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in ARGUMENTS]
    L = LaunchConfiguration
    env = dict(os.environ, PYTHONPATH=f"{ROOT}:{os.environ.get('PYTHONPATH', '')}", KESTREL_COURSE_FILE=L('course_file'))
    flight_interface = Node(
        package='skybeta_flight_interface', executable='flight_interface', name='flight_interface',
        output='screen', emulate_tty=True,
        parameters=[{'max_altitude_m': L('max_altitude_m'), 'min_altitude_m': L('min_altitude_m'),
                     'max_speed_mps': L('max_speed_mps'), 'setpoint_rate_hz': 10.0, 'allow_mode_request': False,
                     'auto_offboard_on_arm': L('auto_offboard_on_arm'),
                     'odometry_topic': '/visual_slam/tracking/odometry'}])
    host_depth = Node(package='kestrel_nav', executable='host_depth', name='host_depth', output='screen',
                      prefix='nice -n 5', parameters=[{'depth_topic': L('depth_topic'), 'max_hz': L('depth_max_hz'),
                                   'max_range_m': L('depth_max_range_m')}])
    health = ExecuteProcess(cmd=['python3', '-m', 'refnav.skybeta_health', '--ros-args', '-p',
                                 ['use_sim_time:=', L('use_sim_time')]], cwd=str(ROOT), additional_env=env,
                            output='screen', name='skybeta_health')
    localizer = ExecuteProcess(
        cmd=['nice', '-n', '5', 'python3', '-m', 'coursenav.loc_node', '--ros-args', '--params-file', str(ROOT / 'onboard/config/common.yaml'),
             '-p', ['depth_topic:=', L('depth_topic')], '-p', ['depth_info_topic:=', L('depth_info_topic')],
             '-p', ['use_sim_time:=', L('use_sim_time')], '-p', 'height_source:=local_position', '-p', 'publish_zero_drift:=true', '-p', 'adaptive_hz:=4.0',
             '-p', ['landed_z_m:=', L('landed_z')], '-p', ['course_file:=', L('course_file')]],
        cwd=str(ROOT), additional_env=env, output='screen', name='localizer')
    fly_env = dict(env)
    fly = ExecuteProcess(
        cmd=['python3', str(ROOT / 'fly_course.py'), '--event-start', '--skip-preflight', '--no-own-localizer', '--no-return', '--start-from-mark',
             '--landed-z', L('landed_z'), '--',
             '-p', 'px4_output:=skybeta', '-p', ['depth_topic:=', L('depth_topic')],
             '-p', ['depth_info_topic:=', L('depth_info_topic')],
             '-p', ['arm_wait_s:=', L('arm_wait_s')], '-p', ['wait_s:=', L('wait_s')],
             '-p', ['use_sim_time:=', L('use_sim_time')], '-p', 'path_turn_hold:=true'],
        cwd=str(ROOT), additional_env={**fly_env, 'KESTREL_USE_SIM_TIME': L('use_sim_time'),
                                       'KESTREL_DECLARED_HEADING_DEG': L('declared_heading_deg'),
                                       'KESTREL_PILOT_VERIFIED_LAUNCH_RADIUS_M': L('pilot_verified_launch_radius_m'),
                                       'KESTREL_NICE_NAV': '5'},
        output='screen', name='fly_course')
    return LaunchDescription(declared + [flight_interface, host_depth, health, localizer, fly])
