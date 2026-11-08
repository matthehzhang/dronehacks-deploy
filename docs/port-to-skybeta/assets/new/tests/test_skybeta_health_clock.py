"""Health expiry uses the sampling clock in SITL, monotonic time on hardware."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def make_health(monkeypatch):
    # Import the real node's methods without requiring a ROS installation.
    monkeypatch.syspath_prepend(str(ROOT))
    for name in ('rclpy','rclpy.node','nav_msgs','nav_msgs.msg','std_msgs','std_msgs.msg'):
        monkeypatch.setitem(sys.modules,name,ModuleType(name))
    sys.modules['rclpy.node'].Node=object
    sys.modules['nav_msgs.msg'].Odometry=object
    sys.modules['std_msgs.msg'].Bool=object
    spec=importlib.util.spec_from_file_location('clock_test_health',ROOT/'refnav/skybeta_health.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def make(simulated):
        clocks=NS(sim=10.,wall=1000.)
        module.time=NS(monotonic=lambda:clocks.wall)
        node=module.SkybetaHealth.__new__(module.SkybetaHealth)
        node.get_parameter=lambda name:NS(value=simulated)
        node.get_clock=lambda:NS(now=lambda:NS(nanoseconds=round(clocks.sim*1e9)))
        node.stale_s=.5;node.tracking_ok=True;node.last_good=None
        node.health=module.VioHealth(max_speed_mps=5.,jump_m=1.,quiet_s=5.)
        return node,clocks
    return make


def odom(t,position=(0.,0.,0.)):
    return NS(header=NS(stamp=NS(sec=int(t),nanosec=round((t-int(t))*1e9))),
              pose=NS(pose=NS(position=NS(x=position[0],y=position[1],z=position[2]))))


def test_slow_simulation_does_not_expire_between_valid_samples(make_health):
    node,clock=make_health(True)
    node._odom(odom(10.))
    # 0.125 real-time factor: two wall seconds per 0.25 simulation seconds.
    clock.wall+=2;clock.sim+=.25
    assert node._healthy()
    node._odom(odom(10.25))
    clock.wall+=3;clock.sim+=.375
    assert node._healthy()
    # Actual sensor silence still expires at the unchanged 0.5 s threshold.
    clock.sim=10.75
    assert not node._healthy()


def test_hardware_retains_monotonic_expiry(make_health):
    node,clock=make_health(False)
    node._odom(odom(10.))
    clock.sim+=10000;clock.wall+=.499
    assert node._healthy()
    clock.wall=1000.5
    assert not node._healthy()


def test_freeze_and_tracking_loss_still_fail_closed(make_health):
    node,clock=make_health(True)
    node._odom(odom(10.))
    clock.sim=10.6
    assert not node._healthy()
    node._odom(odom(10.6))
    assert node._healthy()
    node._status(NS(vo_state=0))
    assert not node._healthy()


def test_bad_kinematics_does_not_refresh_good_sample(make_health):
    node,clock=make_health(True)
    node._odom(odom(10.))
    clock.sim=10.5
    node._odom(odom(10.5,(2.,0.,0.)))
    assert node.last_good==10.
    assert not node._healthy()
    assert 'jump' in node.reason
