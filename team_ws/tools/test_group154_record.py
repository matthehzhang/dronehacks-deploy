"""The recorder must not end merely because the aircraft starts on the ground."""
import re
from group154_record import EndOfRun, TOPICS


def test_waiting_on_ground_keeps_recording():
    end = EndOfRun()
    for t in range(20):
        end.status({'phase': 'arm'}, t)
        end.state({'armed': False, 'landed': True}, t)
        assert end.reason(t) is None


def test_landing_closes_after_postroll():
    end = EndOfRun()
    end.state({'armed': True, 'landed': False}, 0)
    end.status({'phase': 'done'}, 1)
    assert end.reason(1) is None
    for t in range(2, 7):
        end.state({'armed': False, 'landed': True}, t)
        assert end.reason(t) is None
    end.state({'armed': False, 'landed': True}, 7)
    assert 'post-roll' in end.reason(7)


def test_aborted_before_takeoff_closes_when_ground_confirmed():
    end = EndOfRun()
    end.status({'phase': 'done'}, 0)
    for t in range(6):
        end.state({'armed': False, 'landed': True}, t)
    assert end.reason(5)


def test_stale_ground_report_cannot_end_flight():
    end = EndOfRun()
    end.state({'armed': True, 'landed': False}, 0)
    end.state({'armed': False, 'landed': True}, 1)
    assert end.reason(7) is None
    end.state({'armed': False, 'landed': True}, 8)
    assert end.reason(8) is None
    assert end.reason(69) == 'mission/state telemetry absent for 60 seconds'


def test_topic_filter_includes_policy_and_px4_but_no_images():
    for topic in ('/refnav/status', '/fmu/out/vehicle_status_v1', '/coursenav/status', '/skybeta/setpoint/twist'):
        assert re.fullmatch(TOPICS, topic)
    for topic in ('/oak/left/image_raw', '/kestrel_nav/depth', '/oak/points'):
        assert not re.fullmatch(TOPICS, topic)
