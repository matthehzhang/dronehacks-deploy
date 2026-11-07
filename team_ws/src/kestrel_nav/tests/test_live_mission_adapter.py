"""Exercise real adapter callbacks/tick with ROS transport doubles, no DDS peers."""
import importlib
import json
import math
import sys
import types
from types import SimpleNamespace as NS

import numpy as np
import pytest

from kestrel_nav.livenav.follower import FollowParams
from kestrel_nav.livenav.mission import Command
from kestrel_nav.livenav.safety import InputSafety


def header(now):
    return NS(stamp=NS(sec=int(now), nanosec=round((now - int(now)) * 1e9)), frame_id='odom')


class Twist:
    def __init__(self):
        self.header = header(0)
        self.twist = NS(linear=NS(x=0, y=0, z=0), angular=NS(x=0, y=0, z=0))


class MissionDouble:
    def __init__(self):
        self.state = 'WAIT'
        self.follower = NS(p=FollowParams())
        self.p = NS(landed_z_m=.1, cruise_z_m=.8)
        self.doc = {'plan_band_m': .25}
        self.scan_nav = self.scan_pf = object()
        self.calls = 0
        self.fault = None
        self.velocity = (.2, 0, 0)

    def step(self, obs):
        self.calls += 1
        self.state = 'FLY'
        return Command(vx=self.velocity[0], vy=self.velocity[1], vz=self.velocity[2], state='FLY')

    def abort(self, now, reason):
        self.state, self.fault = 'FAULT', reason


@pytest.fixture
def node(monkeypatch):
    # Only ROS bindings are replaced; validation, depth geometry and tick are production code.
    modules = {
        'rclpy': {}, 'rclpy.node': {'Node': object},
        'rclpy.qos': {'QoSProfile': lambda **kw: kw, 'qos_profile_sensor_data': object(),
                      'ReliabilityPolicy': NS(BEST_EFFORT=1), 'DurabilityPolicy': NS(TRANSIENT_LOCAL=1),
                      'HistoryPolicy': NS(KEEP_LAST=1)},
        'geometry_msgs.msg': {'PoseStamped': object, 'TwistStamped': Twist},
        'nav_msgs.msg': {'Odometry': object, 'Path': object},
        'sensor_msgs.msg': {'CameraInfo': object, 'Image': object},
        'std_msgs.msg': {'String': object},
        'isaac_ros_visual_slam_interfaces.msg': {'VisualSlamStatus': object},
    }
    for name, attrs in modules.items():
        mod = types.ModuleType(name)
        vars(mod).update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, 'kestrel_nav.live_mission', raising=False)
    module = importlib.import_module('kestrel_nav.live_mission')
    n = object.__new__(module.LiveMissionNode)
    n._now = 100.0
    n.now_s = lambda: n._now
    n.monotonic_s = lambda: n._now + 1000
    n.get_clock = lambda: NS(now=lambda: NS(to_msg=lambda: header(n._now).stamp))
    n.messages = []
    n.twist_pub = NS(publish=n.messages.append)
    n.get_logger = lambda: NS(info=lambda s: None)
    n.safety = InputSafety()
    n.mission = MissionDouble()
    n.position = n.yaw = n.odom_frame = None
    n.offboard_active = n.armed = n.px4_link = n.position_valid = False
    n.interface_frozen = False
    n.landed = None
    n.clamped = []
    n.log_once = set()
    n.K_depth = (40., 40., 32., 24.)
    n.depth_info_topic = '/depth/info'
    n.att = n.height = None
    n.camera_pose = lambda _: (np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]), np.zeros(3))
    n.pts = n.depth_frame = None
    n.pts_h = .8
    n.pts_t = -math.inf
    n.pad_fix, n.pad_fix_t = None, -math.inf
    n.counts = {'depth': 0, 'depth_bad': 0}
    yield n
    sys.modules.pop('kestrel_nav.live_mission', None)


def odom(now, x=0):
    return NS(header=header(now), child_frame_id='base_link',
              pose=NS(pose=NS(position=NS(x=x, y=0., z=.7), orientation=NS(x=0., y=0., z=0., w=1.)),
                      covariance=[0.] * 36))


def depth(now, valid=True):
    data = np.full((48, 64), 2000 if valid else 0, dtype=np.uint16)
    return NS(header=header(now), encoding='16UC1', height=48, width=64, step=128, data=data.tobytes())


def feed(n, now, offboard=False, skip=()):
    n._now = now
    if 'odom' not in skip:
        n.odom_cb(odom(now))
    if 'tracking' not in skip:
        n.tracking_cb(NS(header=header(now), vo_state=1))
    if 'state' not in skip:
        n.state_cb(NS(data=json.dumps(dict(stamp=now, offboard_active=offboard, armed=offboard,
                                         landed=not offboard, px4_link=True, frozen=False, local_enu=[0., 0., .7]))))
    if 'depth' not in skip:
        n.depth_cb(depth(now))


def ready(n):
    for t in np.arange(100, 101.21, .2):
        feed(n, float(t))
        n.tick()
    assert n.safety.action == 'ready'


def start(n):
    ready(n)
    feed(n, 101.3, offboard=True)
    n.landed = True  # pilot's Offboard ground handover
    n.tick()
    assert n.safety.action == 'run'
    assert n.messages[-1].twist.linear.x == .2


def last_twist(n):
    t = n.messages[-1].twist
    return t.linear.x, t.linear.y, t.linear.z, t.angular.z


def test_ready_stream_contains_only_zero_velocity_no_pose_target(node):
    ready(node)
    assert node.messages
    assert all((m.twist.linear.x, m.twist.linear.y, m.twist.linear.z, m.twist.angular.z) == (0, 0, 0, 0)
               for m in node.messages)
    assert not hasattr(node, 'pose_pub')
    assert node.mission.calls == 0


@pytest.mark.parametrize('missing', ['odom', 'tracking', 'depth'])
def test_stale_sensor_stops_without_running_planner_then_latches(node, missing):
    start(node)
    calls = node.mission.calls
    feed(node, 102, offboard=True, skip=(missing,))
    node.tick()
    assert last_twist(node) == (0, 0, 0, 0)
    assert node.mission.calls == calls
    assert node.mission.state == 'FAULT'
    assert node.mission.scan_nav is None and node.mission.scan_pf is None
    feed(node, 102.1, offboard=True)
    node.tick()
    assert last_twist(node) == (0, 0, 0, 0)
    assert node.mission.calls == calls


def test_delayed_depth_uses_capture_time_not_callback_time(node):
    start(node)
    feed(node, 102, offboard=True, skip=('depth',))
    node.depth_cb(depth(101.2))
    node.tick()
    assert last_twist(node) == (0, 0, 0, 0)
    assert 'timestamp' in node.safety.fault


@pytest.mark.parametrize('bad', ['nan', 'quaternion', 'covariance', 'frame'])
def test_invalid_odometry_rejected_without_overwriting_last_pose(node, bad):
    start(node)
    m = odom(101.4, x=5)
    if bad == 'nan':
        m.pose.pose.position.x = math.nan
    elif bad == 'quaternion':
        m.pose.pose.orientation.w = 0
    elif bad == 'covariance':
        m.pose.covariance[0] = math.nan
    else:
        m.header.frame_id = 'new_odom'
    node._now = 101.4
    node.odom_cb(m)
    node.tick()
    assert node.position[0] == 0
    assert last_twist(node) == (0, 0, 0, 0)
    assert node.safety.fault


def test_tracking_loss_between_ticks_cannot_be_hidden_by_reacquisition(node):
    start(node)
    node._now = 101.4
    node.tracking_cb(NS(header=header(101.4), vo_state=2))
    feed(node, 101.5, offboard=True)
    node.tick()
    assert 'tracking failed' in node.safety.fault
    assert last_twist(node) == (0, 0, 0, 0)


def test_blank_depth_clears_old_scan_and_stops(node):
    start(node)
    node._now = 101.4
    node.depth_cb(depth(101.4, valid=False))
    assert node.pts is None and node.depth_frame is None
    assert node.mission.scan_nav is None
    node.tick()
    assert last_twist(node) == (0, 0, 0, 0)


@pytest.mark.parametrize('velocity', [(0, .2, 0), (-.2, 0, 0)])
def test_unobserved_sideways_and_reverse_commands_are_not_published(node, velocity):
    ready(node)
    feed(node, 101.3, offboard=True)
    node.landed = True
    node.mission.velocity = velocity
    node.tick()
    assert last_twist(node) == (0, 0, 0, 0)
    assert 'corridor unobserved' in node.safety.fault


@pytest.mark.parametrize('loss', ['takeover', 'stale_state'])
def test_authority_loss_stops_publication_and_does_not_resume(node, loss):
    start(node)
    count = len(node.messages)
    if loss == 'takeover':
        feed(node, 101.4)
    else:
        feed(node, 102.4, offboard=True, skip=('state',))
    node.tick()
    assert len(node.messages) == count
    assert node.safety.fault
    feed(node, 102.5, offboard=True)
    node.tick()
    assert last_twist(node) == (0, 0, 0, 0)


def test_exception_in_mission_is_a_latched_stop(node):
    start(node)
    def fail(obs):
        raise RuntimeError('injected planner failure')
    node.mission.step = fail
    feed(node, 101.4, offboard=True)
    node.tick()
    assert last_twist(node) == (0, 0, 0, 0)
    assert 'injected planner failure' in node.safety.fault
