"""Freshness follows ROS time in slow simulation; stale/tracking guards remain."""
from types import SimpleNamespace
import time
import pytest
pytest.importorskip('rclpy')
from refnav.skybeta_health import SkybetaHealth


def state(t):
    return SimpleNamespace(tracking_ok=True,last_good=12.,stale_s=.5,
        get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=int(t*1e9))))


def test_slow_simulation_does_not_expire_a_fresh_ros_sample(monkeypatch):
    s=state(12.4)
    monkeypatch.setattr(time,'monotonic',lambda:100.)
    assert SkybetaHealth._healthy(s)
    monkeypatch.setattr(time,'monotonic',lambda:101.)
    assert SkybetaHealth._healthy(s)


@pytest.mark.parametrize('t',[11.9,12.5,12.6])
def test_old_or_backwards_ros_time_is_unhealthy(t):
    assert not SkybetaHealth._healthy(state(t))


def test_tracking_failure_remains_unhealthy():
    s=state(12.1);s.tracking_ok=False
    assert not SkybetaHealth._healthy(s)
