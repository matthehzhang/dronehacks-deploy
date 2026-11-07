"""safe_mission node hooks with ROS stand-ins (no DDS): corridor stop-and-wait."""
import importlib
import sys
import types
from types import SimpleNamespace as NS

import pytest

from kestrel_nav.livenav.mission import Command
from kestrel_nav.livenav.safe import SafeParams
from kestrel_nav.livenav.safety import SafetyParams


@pytest.fixture
def node(monkeypatch):
    modules = {
        'rclpy': {}, 'rclpy.node': {'Node': object},
        'rclpy.qos': {'QoSProfile': lambda **kw: kw, 'qos_profile_sensor_data': object(),
                      'ReliabilityPolicy': NS(BEST_EFFORT=1), 'DurabilityPolicy': NS(TRANSIENT_LOCAL=1),
                      'HistoryPolicy': NS(KEEP_LAST=1)},
        'geometry_msgs.msg': {'PoseStamped': object, 'TwistStamped': object},
        'nav_msgs.msg': {'Odometry': object, 'Path': object},
        'sensor_msgs.msg': {'CameraInfo': object, 'Image': object},
        'std_msgs.msg': {'String': object},
        'isaac_ros_visual_slam_interfaces.msg': {'VisualSlamStatus': object},
    }
    for name, attrs in modules.items():
        mod = types.ModuleType(name)
        vars(mod).update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
    for mod in ('kestrel_nav.live_mission', 'kestrel_nav.safe_mission'):
        monkeypatch.delitem(sys.modules, mod, raising=False)
    module = importlib.import_module('kestrel_nav.safe_mission')
    n = object.__new__(module.SafeMissionNode)
    n._now = 100.0
    n.now_s = lambda: n._now
    n.warnings = []
    n.get_logger = lambda: NS(warning=n.warnings.append, info=lambda s: None)
    n.sp = SafeParams()
    n._corridor_wait = None
    n.yaw = 0.0
    n.depth_frame = None                       # nothing observed
    n.mission = NS(follower=NS(p=n.sp), doc={'plan_band_m': 0.25})
    n.safety = NS(p=SafetyParams())
    return n


def test_unobserved_corridor_stops_and_waits_then_latches(node):
    cmd = Command(vx=0.25, vy=0.0)
    assert node.motion_corridor_error(cmd) is None          # not latched: stopped instead
    assert cmd.vx == 0.0 and cmd.vy == 0.0 and node.warnings
    node._now += 4.9
    assert node.motion_corridor_error(Command(vx=0.25)) is None
    node._now += 0.2
    assert 'for 5 s' in node.motion_corridor_error(Command(vx=0.25))


def test_creep_needs_no_corridor(node):
    cmd = Command(vx=0.05, vy=0.0)
    assert node.motion_corridor_error(cmd) is None and cmd.vx == 0.05


def test_receipt_clock_sim_uses_the_ros_clock(node):
    node._receipt_sim = True
    assert node.monotonic_s() == node.now_s()


def test_brief_depth_loss_holds_then_latches(node, monkeypatch):
    from kestrel_nav.livenav.safety import InputSafety
    import kestrel_nav.safe_mission as sm
    node.safety = InputSafety()
    node.safety.active = True
    node.mission = NS(state='FLY', follower=NS(p=node.sp), doc={'plan_band_m': 0.25})
    node._depth_bad_since = None
    node._receipt_sim = True
    monkeypatch.setattr(sm.LiveMissionNode, 'tick', lambda self: None)
    node.tick()                                          # no depth sample at all
    assert node.safety.fault is None and 'depth' in node.safety.waived
    node._now += 4.0
    node.tick()
    assert node.safety.fault is None
    node._now += 1.5
    node.tick()
    assert node.safety.fault and 'for 5 s' in node.safety.fault
