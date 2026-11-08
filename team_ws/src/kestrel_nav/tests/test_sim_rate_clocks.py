"""Rate caps count simulation seconds in SITL; profiling remains wall/CPU time."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import os

import pytest

ROOT=Path(__file__).resolve().parents[1]


def methods(path,owner,names,namespace):
    # Compile the production scheduling methods without importing ROS/OpenCV.
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==owner)
    body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    test_class=ast.ClassDef(name=owner,bases=[],keywords=[],body=body,decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[test_class],type_ignores=[]))
    exec(compile(module,str(path),'exec'),namespace)
    return namespace[owner]


def clocks(simulated,rtf=.25):
    state=NS(sim=20.,wall=1000.,sleeps=[])
    def sleep(seconds):
        state.sleeps.append(seconds)
        assert len(state.sleeps)<1000, 'rate wait did not converge'
        state.wall+=seconds;state.sim+=seconds*rtf
    fake_time=NS(time=lambda:state.wall,monotonic=lambda:state.wall,sleep=sleep)
    def wire(node):
        node.get_parameter=lambda name:NS(value=simulated)
        node.get_clock=lambda:NS(now=lambda:NS(nanoseconds=round(state.sim*1e9)))
        return node
    return state,fake_time,wire


def test_depth_rate_wait_scales_with_simulation_clock():
    state,t,wire=clocks(True)
    cls=methods(ROOT/'kestrel_nav/host_depth.py','HostDepth',{'_rate_now','_wait_for_rate'},
                {'time':t,'rclpy':NS(ok=lambda:True)})
    node=wire(cls());node.max_hz=8.
    node._wait_for_rate(20.)
    assert node._rate_now()>=20.125
    assert .5<=state.wall-1000.<.51
    assert max(state.sleeps)<=.05
    assert min(state.sleeps)>=.005


def test_depth_hardware_retains_wall_time_rate_limit():
    state,t,wire=clocks(False)
    cls=methods(ROOT/'kestrel_nav/host_depth.py','HostDepth',{'_rate_now','_wait_for_rate'},
                {'time':t,'rclpy':NS(ok=lambda:True)})
    node=wire(cls());node.max_hz=8.
    node._wait_for_rate(1000.)
    assert state.sleeps==[.125]
    assert node._rate_now()==1000.125


def test_depth_clock_rewind_does_not_wait_for_old_episode():
    state,t,wire=clocks(True)
    cls=methods(ROOT/'kestrel_nav/host_depth.py','HostDepth',{'_rate_now','_wait_for_rate'},
                {'time':t,'rclpy':NS(ok=lambda:True)})
    node=wire(cls());node.max_hz=8.
    node._wait_for_rate(200.)
    assert not state.sleeps


@pytest.mark.parametrize('simulated',[True,False])
def test_localizer_rate_cap_uses_selected_clock(simulated):
    state,t,wire=clocks(simulated)
    cls=methods(ROOT/'coursenav/loc_node.py','LocNode',{'_scan_rate_now','_scan_rate_limited'},{'time':t})
    node=wire(cls());node.adaptive_hz=4.;node._t_proc=node._scan_rate_now()
    state.wall+=.25
    assert node._scan_rate_limited() is simulated
    state.sim+=.25
    assert not node._scan_rate_limited()


def test_preplan_watchdog_fails_closed_only_in_sim(monkeypatch,tmp_path):
    tree=ast.parse((ROOT/'fly_course.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='require_sim_plan_ready')
    namespace={'os':os}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'fly_course.py','exec'),namespace)
    check=namespace['require_sim_plan_ready'];events=[]
    process=NS(poll=lambda:None,terminate=lambda:events.append('terminate'),wait=lambda timeout:events.append('wait'))
    ready=tmp_path/'navigator_ready'
    monkeypatch.setenv('KESTREL_USE_SIM_TIME','false')
    check(ready,process)
    assert events==[]
    monkeypatch.setenv('KESTREL_USE_SIM_TIME','true')
    with pytest.raises(SystemExit,match='simulation flight not started'):
        check(ready,process)
    assert events==['terminate','wait']
    events.clear();ready.touch();check(ready,process)
    assert events==[]
