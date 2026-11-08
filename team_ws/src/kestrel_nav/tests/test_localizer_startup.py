"""Construct the complete production localizer with strict ROS declaration doubles.

Geometry, particle filter, live/local maps, and prior monitor are real. Only ROS
transport/TF objects are replaced. Duplicate declarations raise as rclpy does.
"""
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('fit', [None, .42])
@pytest.mark.parametrize('maps_enabled', [True, False])
def test_full_localizer_constructor_declares_fit_once_and_shares_override(monkeypatch, tmp_path, fit, maps_enabled):
    overrides = dict(course_file=str(ROOT/'course/course_new_origin.json'), log_dir=str(tmp_path),
                     height_source='local_position', live_map=maps_enabled, local_map=maps_enabled)
    if fit is not None:
        overrides['min_fit_frac'] = fit

    class Node:
        def __init__(self, name):
            self.declarations = {}
            self.publishers, self.subscriptions, self.timers = [], [], []

        def declare_parameter(self, name, default):
            if name in self.declarations:
                raise RuntimeError(f'ParameterAlreadyDeclaredException: {name}')
            parameter = NS(value=overrides.get(name, default))
            self.declarations[name] = parameter
            return parameter

        def create_publisher(self, message, topic, qos):
            self.publishers.append(topic)
            return NS(publish=lambda value: None)

        def create_subscription(self, message, topic, callback, qos):
            self.subscriptions.append(topic)
            return NS()

        def create_timer(self, period, callback):
            self.timers.append((period, callback))
            return NS()

        def get_logger(self):
            return NS(info=lambda *_: None, error=lambda *_: None)

        def destroy_node(self):
            return True

    modules = {
        'rclpy': {}, 'rclpy.node': {'Node': Node},
        'rclpy.executors': {'ExternalShutdownException': type('ExternalShutdownException', (Exception,), {})},
        'rclpy.qos': {'QoSProfile': lambda **kw: NS(**kw), 'qos_profile_sensor_data': object(),
                      'ReliabilityPolicy': NS(BEST_EFFORT=1, RELIABLE=2),
                      'DurabilityPolicy': NS(TRANSIENT_LOCAL=1), 'HistoryPolicy': NS(KEEP_LAST=1)},
        'geometry_msgs.msg': {name: NS for name in ('PoseWithCovarianceStamped', 'Vector3Stamped')},
        'nav_msgs.msg': {name: NS for name in ('OccupancyGrid', 'Odometry')},
        'sensor_msgs.msg': {name: NS for name in ('CameraInfo', 'Image')},
        'std_msgs.msg': {'String': NS},
        'px4_msgs.msg': {name: NS for name in ('DistanceSensor', 'VehicleAttitude', 'VehicleLocalPosition')},
        'tf2_ros': {'Buffer': NS, 'TransformListener': lambda *args: NS()},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        vars(module).update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, 'coursenav.loc_node', raising=False)
    module = importlib.import_module('coursenav.loc_node')
    node = module.LocNode()
    try:
        expected = .35 if fit is None else fit
        assert node.pf.p.min_fit_frac == expected
        assert node.prior_mon.p.restore_fit == expected
        assert node.declarations['min_fit_frac'].value == expected
        assert (node.local is not None) == maps_enabled
        assert (node.live is not None) == maps_enabled
        assert node.height_source == 'local_position'
        assert not any(topic.startswith('/fmu/in/') for topic in node.publishers)
        # Exercise the startup status timer; this invokes prior publication too.
        node._status()
        assert not node.log_f.closed
    finally:
        node.destroy_node()
        sys.modules.pop('coursenav.loc_node', None)
    assert node.log_f.closed
