"""Map-free search bringup. No prior/PF/course process is launched.

Default startup is unarmed and blocks takeoff until allow_takeoff is explicitly
set after approving the launch column. A pilot still owns arm/mode transitions.
Only one flight interface and competitor mission may run at a time.
"""
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ROOT = Path(__file__).resolve().parent.parent
ARGUMENTS = [
    ('use_sim_time', 'false', 'true only with a simulator /clock'),
    ('flight_interface', 'true', 'start organizer flight interface; false if already running'),
    ('host_depth', 'true', 'CPU stereo depth using calibrated mono pair'),
    ('health', 'true', 'VIO health-only node; publishes no external vision'),
    ('allow_takeoff', 'false', 'approve a clear vertical launch column; pilot still arms'),
    ('pilot_verified_launch', 'false', 'opt-in inspected bounded launch volume; never changes measured occupancy'),
    ('launch_volume_bounds_json', '', 'six fixed ENU offsets [xmin,ymin,zmin,xmax,ymax,zmax] from launch pose; metres'),
    ('launch_volume_duration_s', '30.0', 'expires within 60s; never extended or recentered'),
    ('target_contrast_qualified', 'false', 'only after actual mono target/floor/negative-image qualification'),
    ('depth_topic', '/kestrel_nav/depth', 'rectified calibrated depth'),
    ('depth_info_topic', '/kestrel_nav/depth/camera_info', ''),
    ('mono_topic', '/oak/left/image_mono', 'raw MONO8 image, never RGB observer'),
    ('mono_info_topic', '/oak/left/camera_info', ''),
    ('rgb_topic', '/oak/rgb/image_raw', 'color target detection; sim may use /oak/rgb/image'),
    ('rgb_info_topic', '/oak/rgb/camera_info', 'RGB calibration, separate from stereo depth'),
    ('blue_hue_low', '90', 'OpenCV hue 0..179'),
    ('blue_hue_high', '135', ''),
    ('blue_saturation_min', '80', ''),
    ('blue_value_min', '40', ''),
    ('base_frame', 'base_link', 'measured camera TF parent'),
    ('cruise_height_m', '0.8', 'climb relative to the current landed fused local position'),
    ('v_cruise_mps', '0.4', 'm/s, capped at .6'),
    ('v_gap_mps', '0.2', 'm/s, capped at .25'),
    ('time_limit_s', '180.0', 'bounded search then hold for pilot'),
    ('mapping_hz', '4.0', 'maximum depth integration rate'),
    ('detector_hz', '3.0', 'maximum raw mono detector rate'),
    ('resolution_m', '0.05', 'fresh occupancy raster; radius and margin are fixed'),
    ('map_extent_m', '16.0', 'bounded square in local frame, not course dimensions'),
]


def generate_launch_description():
    L = LaunchConfiguration
    env = dict(os.environ, PYTHONPATH=f'{ROOT}:{os.environ.get("PYTHONPATH", "")}')
    declared = [DeclareLaunchArgument(name, default_value=value, description=description)
                for name, value, description in ARGUMENTS]
    interface = Node(package='skybeta_flight_interface', executable='flight_interface',
        name='flight_interface', output='screen', condition=IfCondition(L('flight_interface')),
        parameters=[dict(use_sim_time=ParameterValue(L('use_sim_time'), value_type=bool),
                         allow_mode_request=False, auto_offboard_on_arm=True,
                         max_altitude_m=3., min_altitude_m=.3, max_speed_mps=1.5,
                         setpoint_rate_hz=10., odometry_topic='/visual_slam/tracking/odometry')])
    depth = Node(package='kestrel_nav', executable='host_depth', name='host_depth', output='screen',
        condition=IfCondition(L('host_depth')), prefix='nice -n 5', parameters=[{
            'use_sim_time': ParameterValue(L('use_sim_time'), value_type=bool),
            'depth_topic': L('depth_topic'), 'max_hz': 8., 'max_range_m': 12.}])
    health = ExecuteProcess(cmd=['python3', '-m', 'refnav.skybeta_health', '--ros-args', '-p',
                                ['use_sim_time:=', L('use_sim_time')]], cwd=str(ROOT),
                            additional_env=env, output='screen', name='search_vio_health',
                            condition=IfCondition(L('health')))
    args = ['python3', '-m', 'kestrel_nav.searchnav.node', '--ros-args']
    for name, _, _ in ARGUMENTS:
        if name not in ('flight_interface', 'host_depth', 'health'):
            # ROS parses CLI values as YAML. Keep JSON (including the disabled
            # empty default) a string rather than a null/array override.
            value = ([name + ':="', L(name), '"'] if name == 'launch_volume_bounds_json'
                     else [name + ':=', L(name)])
            args.extend(['-p', value])
    mission = ExecuteProcess(cmd=args, cwd=str(ROOT), additional_env=env,
                             output='screen', name='search_mission')
    return LaunchDescription(declared + [interface, depth, health, mission])
