"""Actual tuning adapter callbacks over transport doubles; no ROS graph."""
import importlib
import io
import json
import math
import sys
from types import SimpleNamespace as NS

import pytest

from test_live_mission_adapter import node, header, odom, ready  # shared transport-only fixture


@pytest.fixture
def tuning_node(node, monkeypatch):
    monkeypatch.setattr(sys.modules['sensor_msgs.msg'], 'Imu', object, raising=False)
    monkeypatch.delitem(sys.modules, 'kestrel_nav.tune_mission', raising=False)
    module = importlib.import_module('kestrel_nav.tune_mission')
    node.__class__ = module.TuningNode
    node.trace_file = io.StringIO()
    node.trace_error = None
    node.dropped_count = 0
    node.imu_sample = node.imu_frame = None
    node.yaw_sample = node.measured_yaw_rate = None
    node.mission.controller = NS(status=lambda: dict(phase='PULSE', candidate='test', trial=0,
                                      trial_id='test:0', split='train', target_speed=.1), events=[])
    yield node
    sys.modules.pop('kestrel_nav.tune_mission', None)


def imu(now, frame='oak_imu_frame'):
    h = header(now); h.frame_id = frame
    return NS(header=h, linear_acceleration=NS(x=0., y=0., z=9.8),
              angular_velocity=NS(x=0., y=0., z=0.))


def rows(n):
    return [json.loads(line) for line in n.trace_file.getvalue().splitlines()]


def test_no_imu_means_no_readiness_publication(tuning_node):
    n = tuning_node
    n.tick()
    assert not n.messages
    assert n.safety.healthy_since is None


def test_imu_duplicate_does_not_refresh_and_frame_change_faults(tuning_node):
    n = tuning_node
    n.imu_cb(imu(100.))
    sample = n.imu_sample
    n._now = 100.1
    n.imu_cb(imu(100.))
    assert n.imu_sample == sample
    n.safety.active = True
    n.imu_cb(imu(100.1, 'changed'))
    assert n.imu_sample is None and n.safety.fault


def test_raw_callbacks_keep_capture_receipt_and_duplicate_acceptance(tuning_node):
    n = tuning_node
    n.odom_cb(odom(100.))
    n._now = 100.1
    n.odom_cb(odom(100.))
    n.imu_cb(imu(100.1))
    data = rows(n)
    assert [x['accepted'] for x in data if x['kind'] == 'odom'] == [True, False]
    assert data[0]['source_stamp'] == 100. and data[0]['frame_id'] == 'odom'
    assert data[-1]['frame_id'] == 'oak_imu_frame'
    assert data[-1]['t_mono'] == 1100.1


def test_other_odometry_frame_cannot_feed_enu_commands(tuning_node):
    n = tuning_node
    msg = odom(100.); msg.header.frame_id = 'different_world'
    n.odom_cb(msg)
    assert n.position is None and 'odom' in n.safety.errors
    assert rows(n)[-1]['accepted'] is False


def test_command_record_uses_published_header_and_enu(tuning_node):
    n = tuning_node
    n.send_twist(.1, .02, 0., .1)
    published = n.messages[-1]
    record = rows(n)[-1]
    assert record['source_stamp'] == published.header.stamp.sec+published.header.stamp.nanosec*1e-9
    assert record['velocity_enu'] == [.1, .02, 0.] and record['frame_id'] == 'odom'
    assert record['split'] == 'train' and record['trial_id'] == 'test:0'


def test_yaw_trial_identity_and_split_do_not_reuse_translation_trial(tuning_node):
    n = tuning_node
    n.mission.controller.status = lambda: dict(phase='YAW_PULSE', candidate='test', trial=0,
        trial_id='test:yaw:3', split='validation', target_speed=.1, axis='yaw', rotation_trial=3,
        target_yaw_rate_rps=-.25)
    n.send_twist(0., 0., 0., -.25)
    record = rows(n)[-1]
    assert record['split'] == 'validation' and record['trial_id'] == 'test:yaw:3'
    assert record['axis'] == 'yaw' and record['rotation_trial'] == 3


def test_yaw_rate_uses_accepted_capture_time_and_unwraps_heading(tuning_node):
    n = tuning_node
    def message(t, yaw):
        msg = odom(t)
        msg.pose.pose.orientation = NS(x=0., y=0., z=math.sin(yaw/2), w=math.cos(yaw/2))
        return msg
    n.odom_cb(message(100., math.pi-.01))
    assert n.measured_yaw_rate is None
    n._now = 100.1
    n.odom_cb(message(100.1, -math.pi+.01))
    assert n.measured_yaw_rate == pytest.approx(.2)
    sample = n.yaw_sample
    n._now = 100.2
    n.odom_cb(message(100.1, -.5))
    assert n.yaw_sample == sample and n.measured_yaw_rate == pytest.approx(.2)
    # A large capture gap cannot supply the rotation gate with a fresh estimate.
    n._now = 100.5
    n.odom_cb(message(100.5, -math.pi+.02))
    assert n.measured_yaw_rate is None


def test_logging_failure_latches_fault(tuning_node):
    n = tuning_node
    n.trace_file = NS(write=lambda _: (_ for _ in ()).throw(OSError('disk full')))
    n.record('event', reason='test')
    assert n.safety.fault and n.dropped_count == 1


def test_missing_measured_configuration_refuses_constructor(tuning_node):
    n = tuning_node
    n.declare_parameter = lambda *args: NS(value='')
    with pytest.raises(ValueError, match='measured lane geometry'):
        n.make_mission(None, None, None)


def test_map_only_configuration_refused_by_physical_ros_adapter(tuning_node, tmp_path):
    from test_tuning import config
    c = config(); c['map_only'] = True
    path = tmp_path/'map_only.json'; path.write_text(json.dumps(c))
    tuning_node.declare_parameter = lambda *args: NS(value=str(path))
    with pytest.raises(ValueError, match='map_only geometry'):
        tuning_node.make_mission(None, None, None)
