"""group119 (workspace origin/group119 @ 749bca0) flying on skybeta: refnav + coursenav + fly_course.py,
vendored unchanged, with refnav's px4_output:=skybeta (twist through the organizer flight interface).

    ros2 launch kestrel_nav refnav_skybeta.launch.py
    ros2 launch kestrel_nav refnav_skybeta.launch.py chicane:=gate_b
    ros2 launch kestrel_nav refnav_skybeta.launch.py start_x:=-2.0 start_y:=0.5 start_yaw_deg:=90.0

Start pose: by default the drone must sit on the course file's launch mark (course_v3.json "start").
To take off from somewhere else, place it there still, then pass start_x/start_y/start_yaw_deg in the
course frame (metres, x east, y north, origin at the enclosure floor centre; yaw 90 = nose north, CCW +).
All three or none. They replace the mark for both the localizer prior and the planner; the route is
flown relative to the VIO origin, i.e. the pose at stack start, so the placement must be as exact as on
the mark (course file: 0.05 m / 2 deg). Preview the plan first:
    python3 fly_course.py --dry-run --start X Y YAW_DEG

/etc/skybeta/drone.env:  COMPETITOR_PKG=kestrel_nav  COMPETITOR_LAUNCH=refnav_skybeta.launch.py

Start-up order on the drone: drone on the launch mark, stack up. Wait for refnav's "arm" phase in the
log (it streams a hold setpoint; the interface needs a team setpoint within 1 s of the switch), THEN the
pilot arms and switches to Position. refnav accepts the armed vehicle (accept_armed_start), climbs and flies.
"""
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
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
    ('depth_max_range_m', '12.0', "host_depth range cap. refnav counts a corridor ray as OBSERVED if it returns any depth "
                                  "(its obstacle points stop at 6 m, the localizer's at 4 m): the hall beyond the "
                                  "see-through nets is 6-9 m away and at the old 6 m cap every such ray read unobserved"),
    ('arm_wait_s', '600.0', 'refnav: how long the arm phase waits for the pilot to arm and switch to Position'),
    ('wait_s', '900.0', 'refnav: how long it waits for readiness before giving up'),
    ('landed_z', '0.10', 'landed body-origin height above the floor: PX4 dist_bottom 0.100 m on skybeta-14 (no distance_sensor topic there)'),
    ('localizer_timeout', '120.0', 'fly_course: how long to wait for the localizer to converge on the mark'),
    ('use_sim_time', 'false', 'true only in a simulator that publishes /clock'),
    ('start_x', '', 'course-frame start x (m); empty = the course file launch mark. Needs start_y and start_yaw_deg'),
    ('start_y', '', 'course-frame start y (m); empty = the course file launch mark'),
    ('start_yaw_deg', '', 'course-frame start heading (deg, 90 = north); empty = the course file launch mark'),
    ('declared_heading_deg', '0.0', "navigator: flight frame heading vs the VIO origin. 0 on skybeta (the organizer bridge "
                                     "permutes cuVSLAM's start-relative ENU odometry directly); 90 was the FlyX lab bridge"),
]


def start_pose(context):
    """(x, y, yaw_deg) from the start_* arguments, or None to use the course file's launch mark."""
    raw = [LaunchConfiguration(n).perform(context).strip() for n in ('start_x', 'start_y', 'start_yaw_deg')]
    if not any(raw):
        return None
    if not all(raw):
        raise RuntimeError('start_x, start_y and start_yaw_deg must be given together (or none of them)')
    return tuple(float(v) for v in raw)


def generate_launch_description():
    declared = [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in ARGUMENTS]
    return LaunchDescription(declared + [OpaqueFunction(function=_nodes)])


def _nodes(context):
    L = LaunchConfiguration
    start = start_pose(context)
    loc_start, fly_start = [], []
    if start is not None:
        x, y, yaw = start
        loc_start = ['-p', f'init_x:={x}', '-p', f'init_y:={y}', '-p', f'init_yaw_deg:={yaw}']
        fly_start = ['--start', str(x), str(y), str(yaw)]
        print(f'[refnav_skybeta] start pose from launch arguments: ({x}, {y}) yaw {yaw} deg (course frame)')
    env = dict(os.environ, PYTHONPATH=f"{ROOT}:{os.environ.get('PYTHONPATH', '')}")
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
             '-p', ['landed_z_m:=', L('landed_z')]] + loc_start,
        cwd=str(ROOT), additional_env=env, output='screen', name='localizer')
    fly_env = dict(env)
    fly = ExecuteProcess(
        cmd=['python3', str(ROOT / 'fly_course.py'), '--event-start', '--skip-preflight', '--no-own-localizer', '--no-return', '--start-from-mark',
             '--chicane', L('chicane'), '--landed-z', L('landed_z'), '--localizer-timeout', L('localizer_timeout')] + fly_start + ['--',
             '-p', 'px4_output:=skybeta', '-p', 'land_timeout_s:=0.0', '-p', ['depth_topic:=', L('depth_topic')],
             '-p', ['depth_info_topic:=', L('depth_info_topic')],
             '-p', ['arm_wait_s:=', L('arm_wait_s')], '-p', ['wait_s:=', L('wait_s')],
             '-p', ['use_sim_time:=', L('use_sim_time')], '-p', 'path_turn_hold:=true'],
        cwd=str(ROOT), additional_env={**fly_env, 'KESTREL_USE_SIM_TIME': L('use_sim_time'),
                                       'KESTREL_DECLARED_HEADING_DEG': L('declared_heading_deg'),
                                       'KESTREL_NICE_NAV': '5'},
        output='screen', name='fly_course')
    return [flight_interface, host_depth, health, localizer, fly]
