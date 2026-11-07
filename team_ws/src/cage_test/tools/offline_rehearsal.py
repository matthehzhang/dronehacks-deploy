#!/usr/bin/env python3
"""Rehearse hop_test's state machine against a toy vehicle. No ROS, no drone.

    python3 tools/offline_rehearsal.py          # from team_ws/src/cage_test

rclpy and the message packages are replaced by stand-ins, so this runs on a
laptop. The "vehicle" follows the pose setpoint with PX4's position gain
(MPC_XY_P = MPC_Z_P ~ 1), follows a twist exactly, never leaves the ground
below z = 0, and is "landed" + auto-disarmed 2 s after touching down under a
descending twist (COM_DISARM_LAND). The flight interface's 0.3 m floor is
and ceiling are applied to poses. It checks the node's logic and the shape of the flight,
not PX4 dynamics or anything on the wire. Exits 1 on any failed check.
"""
import json
import math
import os
import sys
import types
from types import SimpleNamespace as NS

DT = 0.1
FLOOR_M = 0.3
NOW = [0.0]
FAILURES = []


# ---------------------------------------------------------------- stand-ins
class _Clock:
    def now(self):
        return NS(nanoseconds=int(NOW[0] * 1e9), to_msg=lambda: None)


class _Pub:
    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


class _Node:
    def __init__(self, name, parameter_overrides=None, **kwargs):
        self._params = {}
        self._overrides = parameter_overrides or {}
        self.logs = []

    def declare_parameter(self, name, default):
        self._params[name] = self._overrides.get(name, default)

    def get_parameter(self, name):
        return NS(value=self._params[name])

    def create_publisher(self, *args):
        return _Pub()

    def create_subscription(self, *args):
        pass

    def create_timer(self, *args):
        pass

    def get_clock(self):
        return _Clock()

    def get_logger(self):
        logs = self.logs
        return NS(info=lambda m: logs.append(('I', NOW[0], m)),
                  warn=lambda m: logs.append(('W', NOW[0], m)),
                  error=lambda m: logs.append(('E', NOW[0], m)))


def _xyz():
    return NS(x=0.0, y=0.0, z=0.0)


class _PoseStamped:
    def __init__(self):
        self.header = NS(stamp=None, frame_id='')
        self.pose = NS(position=_xyz(), orientation=NS(x=0.0, y=0.0, z=0.0, w=0.0))


class _TwistStamped:
    def __init__(self):
        self.header = NS(stamp=None, frame_id='')
        self.twist = NS(linear=_xyz(), angular=_xyz())


class _String:
    def __init__(self, data=''):
        self.data = data


for _name, _attrs in {
        'rclpy': {}, 'rclpy.node': {'Node': _Node},
        'geometry_msgs': {}, 'geometry_msgs.msg': {
            'PoseStamped': _PoseStamped, 'TwistStamped': _TwistStamped},
        'nav_msgs': {}, 'nav_msgs.msg': {'Odometry': object},
        'std_msgs': {}, 'std_msgs.msg': {'String': _String}}.items():
    _module = types.ModuleType(_name)
    _module.__dict__.update(_attrs)
    sys.modules[_name] = _module

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cage_test.hop_test import HopTest  # noqa: E402


# ---------------------------------------------------------------- the flight
def fly(overrides=None, yaw=0.0, start=(0.3, -0.2, 0.0), offboard_at_start=False,
        pilot_takes_over_at=None, landing_undetected=False, odom_gap=None,
        ceiling_m=2.0, duration_s=120.0):
    NOW[0] = 1000.0
    node = HopTest(parameter_overrides=overrides or {})
    pos = list(start)
    st = dict(px4_link=True, local_enu=list(pos), armed=offboard_at_start,
              offboard_active=offboard_at_start, landed=True, clamped=[], envelope={})
    mode, setpoint, vz = None, None, 0.0
    landed_since = None
    out = NS(node=node, trace=[], max_speed=0.0, disarmed_at=None)
    q = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
    t = 0.0
    while t < duration_s:
        NOW[0] += DT
        t = NOW[0] - 1000.0
        # the pilot: arm at 2 s, switch to Position at 4 s -> Offboard, once
        if not offboard_at_start and out.disarmed_at is None:
            if t >= 2.0:
                st['armed'] = True
            if t >= 4.0 and node.state == 'WAIT_FOR_OFFBOARD' and not st['offboard_active']:
                st['offboard_active'] = True
        if pilot_takes_over_at is not None and t >= pilot_takes_over_at:
            st['offboard_active'] = False
        if not (odom_gap and odom_gap[0] <= t < odom_gap[1]):
            node.odometry_cb(NS(pose=NS(pose=NS(
                position=NS(x=pos[0], y=pos[1], z=pos[2]),
                orientation=NS(x=q[0], y=q[1], z=q[2], w=q[3])))))
        st['local_enu'] = list(pos)
        node.state_cb(_String(json.dumps(st)))

        n_pose, n_twist = len(node.pose_pub.msgs), len(node.twist_pub.msgs)
        node.tick()
        if len(node.twist_pub.msgs) > n_twist:
            mode, vz = 'twist', node.twist_pub.msgs[-1].twist.linear.z
        elif len(node.pose_pub.msgs) > n_pose:
            p = node.pose_pub.msgs[-1].pose.position
            z = min(max(p.z, start[2] + FLOOR_M), start[2] + ceiling_m)
            mode, setpoint = 'pose', (p.x, p.y, z)

        if st['armed'] and st['offboard_active']:
            if mode == 'pose':
                vel = [s - c for s, c in zip(setpoint, pos)]
            else:
                vel = [0.0, 0.0, vz]
            before = list(pos)
            pos = [c + DT * v for c, v in zip(pos, vel)]
            pos[2] = max(pos[2], start[2] + (0.05 if landing_undetected else 0.0))
            out.max_speed = max(out.max_speed, math.dist(before, pos) / DT)
            on_ground = pos[2] <= start[2] + 1e-9 and mode == 'twist'
            st['landed'] = on_ground
            landed_since = (landed_since or t) if on_ground else None
            if landed_since is not None and t - landed_since >= 2.0:
                st.update(armed=False, offboard_active=False)
                out.disarmed_at = t
        out.trace.append((round(t, 1), node.state, tuple(pos)))
        if out.disarmed_at is not None and t > out.disarmed_at + 1.0:
            break
    return out


def check(name, ok, detail=''):
    print(f'  [{"PASS" if ok else "FAIL"}] {name}{"  " + detail if detail else ""}')
    if not ok:
        FAILURES.append(name)


def logged(run, text):
    return any(text in message for _, _, message in run.node.logs)


def peak(run, axis, fn=max):
    return fn(p[axis] for _, _, p in run.trace)


def main():
    print('nominal: up 1.0, forward 1.0 along +x, back, down 0.5, touchdown')
    r = fly()
    for _, t, m in r.node.logs:
        print(f'    {t - 1000.0:6.1f}  {m.splitlines()[0]}')
    check('ends disarmed and ready again', r.node.state == 'WAIT_FOR_OFFBOARD' and r.disarmed_at)
    check('climbs to 1.0 m', abs(peak(r, 2) - 1.0) < 0.05, f'peak z {peak(r, 2):.2f}')
    check('flies 1.0 m forward', abs(peak(r, 0) - 1.3) < 0.05, f'peak x {peak(r, 0):.2f}')
    check('no sideways motion', abs(peak(r, 1) + 0.2) < 1e-6 and abs(peak(r, 1, min) + 0.2) < 1e-6)
    check('never faster than 0.5 m/s', r.max_speed <= 0.5, f'{r.max_speed:.2f} m/s')
    check('touches down', logged(r, 'PX4 reports landed'))
    check('whole hop under 45 s', r.disarmed_at < 45.0, f'{r.disarmed_at:.1f} s')

    print('nose along +y (vehicle turned after the stack came up)')
    r = fly(yaw=math.pi / 2)
    check('forward goes along +y', abs(peak(r, 1) - 0.8) < 0.05
          and abs(peak(r, 0) - 0.3) < 1e-6 and abs(peak(r, 0, min) - 0.3) < 1e-6,
          f'peak y {peak(r, 1):.2f}')

    print('Offboard already active when the node starts (restart in flight)')
    r = fly(offboard_at_start=True, duration_s=10.0)
    check('does not start a hop', r.node.state == 'WAIT_FOR_OFFBOARD'
          and logged(r, 'not starting a hop from the air'))

    print('pilot takes over at 12 s, during "forward"')
    r = fly(pilot_takes_over_at=12.0, duration_s=20.0)
    check('stops advancing', r.node.state == 'ABORTED' and logged(r, 'Offboard lost'))

    print('touchdown:=false')
    r = fly({'touchdown': False}, duration_s=40.0)
    check('holds at 0.5 m for the pilot', r.node.state == 'DONE'
          and abs(r.trace[-1][2][2] - 0.5) < 0.05, f'z {r.trace[-1][2][2]:.2f}')

    print('landing never detected')
    r = fly(landing_undetected=True, duration_s=50.0)
    check('keeps descending, tells the pilot to disarm',
          r.node.state == 'TOUCHDOWN' and logged(r, 'Pilot: if it is on the ground, disarm'))

    print('odometry drops out for 3 s mid-flight')
    r = fly(odom_gap=(10.0, 13.0))
    check('pauses, then finishes', logged(r, 'odometry stale') and r.disarmed_at is not None)

    print('a leg that can never be reached (climb 1.0 m under a 0.8 m ceiling)')
    r = fly(ceiling_m=0.8, duration_s=40.0)
    check('gives up and holds', r.node.state == 'FAILED' and logged(r, 'not reached in time'))

    print('bad parameters are refused at startup')
    for bad in ({'low_height_m': 1.5}, {'descend_speed_mps': 0.2}, {'speed_mps': 0.0}):
        try:
            HopTest(parameter_overrides=bad)
            check(f'refuses {bad}', False)
        except ValueError:
            check(f'refuses {bad}', True)

    print(f'\n{"ALL PASS" if not FAILURES else "FAILED: " + ", ".join(FAILURES)}')
    return 1 if FAILURES else 0


if __name__ == '__main__':
    sys.exit(main())
